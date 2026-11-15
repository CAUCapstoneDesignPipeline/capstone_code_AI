"""Decisions API 질문 방식 비교: 같은 후보·같은 노트에서 질문 문구와 유형만 바꿔 정답 노트로 잰다.

  uv run --env-file .env python experiments/decisions/variants.py [--only criteria choice ...]

변형 (학생이 후보를 내고, 후보마다 질문):
  strict    한 문장 predicate (지난 실험의 기준)
  criteria  표시 기준(무엇이 개념이고 무엇이 아닌가)과 예를 넣은 predicate
  choice    core / mentioned / not_concept 세 갈래 choice (개념 확률 = core + mentioned)
  score     not_concept / mentioned_term / core_concept 세 단계 score (개념 확률 = 앞 단계가 아닌 쪽의 합)
  boundary  criteria + "이 문자열이 노트에 쓰인 온전한 용어인가" 질문 (두 확률의 곱). 질문이 둘이라 후보를 200개까지 묻는다
임계값은 dev F1 로 고르고 test · 논문 리뷰는 그 값으로 한 번만 잰다. 노트 원문이 OpenAI 로 나간다.
"""

import argparse
import hashlib
import pickle
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.cache import scratch
from app.services.concepts.judge import CHOICE, CRITERIA, LEVELS      # 출고 profile 과 같은 문구 (한 곳에서만 고친다)
CACHE = scratch("decisions_variants_cache.pkl")

STRICT = ('In this study note, is "{s}" a technical concept that deserves its own node in a knowledge graph (a field-specific term, algorithm, model, metric, '
          'dataset or named technique), and not a generic word, a heading frame, an example value, or a person/venue name?')
CHOICES = [{"value": "core", "description": "a field-specific concept the note explains, defines, or relates to other concepts"},
           {"value": "mentioned", "description": "a field-specific term that appears in passing but is still a standard term of the field"},
           {"value": "not_concept", "description": "a generic word, section/heading frame, example value or variable, person/company/venue name, sentence fragment, or a string that cuts a longer term short or adds extra words to it"}]
BOUNDARY = ('Is "{s}" the complete term exactly as the note uses it (not a truncated piece of a longer term, and not a longer string that adds extra words around a shorter term)?')


def make(variant):
    """→ (후보마다 질문 목록을 만드는 함수, 후보 하나의 답들 → 개념 확률, 후보당 질문 수)."""
    if variant == "strict":
        return (lambda s: [{"type": "predicate", "instructions": STRICT.format(s=s)}], lambda a: a[0].probability, 1)
    if variant == "criteria":
        return (lambda s: [{"type": "predicate", "instructions": CRITERIA.format(s=s)}], lambda a: a[0].probability, 1)
    if variant == "choice":
        def reduce(a):
            ps = {x.value: x.probability for x in a[0].probabilities}
            return ps.get("core", 0.0) + ps.get("mentioned", 0.0)
        return (lambda s: [{"type": "choice", "instructions": CHOICE.format(s=s), "choices": CHOICES}], reduce, 1)
    if variant == "score":
        def reduce(a):
            return sum(x.probability for x in a[0].probabilities if x.label != "not_concept")
        return (lambda s: [{"type": "score", "instructions": CHOICE.format(s=s), "levels": LEVELS}], reduce, 1)
    if variant == "boundary":
        return (lambda s: [{"type": "predicate", "instructions": CRITERIA.format(s=s)}, {"type": "predicate", "instructions": BOUNDARY.format(s=s)}],
                lambda a: a[0].probability * a[1].probability, 2)
    raise ValueError(variant)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=["strict", "criteria", "choice", "score", "boundary"])
    ap.add_argument("--model", default="gpt-6-luna")
    ap.add_argument("--min-p", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ensemble", action="store_true", help="--only 로 고른 변형(후보 집합이 같은 것)의 확률을 평균해 견준다 (캐시만 쓴다)")
    args = ap.parse_args()

    import openai

    from app.cli import load_extractor
    from app.services.concepts.gold import Extracted, loose, score
    from app.services.concepts.student import Student, rows, suppress_nested
    from experiments.llm_only.baseline import sets_available

    ex = load_extractor(argparse.Namespace(idf=ROOT / "data/private/concepts/idf.json", cache=None))
    st = Student.load()
    client = openai.OpenAI(max_retries=2, timeout=180)
    data = sets_available(["dev", "test", "unseen"])
    prepared = {}
    for name, (gold, texts) in data.items():
        prepared[name] = {}
        for n in sorted(gold):
            text, rs = rows(ex, texts[n], symbols=True)
            p = np.round(st.probabilities(np.array([r["x"] for r in rs], float)), 6)
            prepared[name][n] = (text, rs, p, [i for i in np.argsort(-p, kind="stable") if p[i] >= args.min_p])
    cache = pickle.load(open(CACHE, "rb")) if CACHE.exists() else {}
    f1 = lambda r, p: 2 * r * p / (r + p + 1e-9)
    results, dec_all, evaluators = {}, {}, {}
    for variant in args.only:
        build, reduce, per = make(variant)
        wording = hashlib.sha256(repr([build("X"), "v1"]).encode()).hexdigest()[:10]
        jobs = []
        order = {}
        for name, per_note in prepared.items():
            for n, (text, rs, p, ordr) in per_note.items():
                cand = ordr[:400 // per]
                order[(name, n)] = cand
                key = (args.model, variant, wording, hashlib.sha256((text + "\x00" + "\x1f".join(rs[i]["surface"] for i in cand)).encode()).hexdigest())
                if key not in cache:
                    jobs.append((key, text, [rs[i]["surface"] for i in cand]))

        def call(job):
            key, text, cands = job
            qs = [q for c in cands for q in build(c)]
            t0 = time.perf_counter()
            d = client.decisions.create(model=args.model, input=text, questions=qs)
            out = []
            for k in range(len(cands)):
                ans = d.answers[k * per:(k + 1) * per]
                out.append(None if any(a.type == "refusal" for a in ans) else reduce(ans))
            return key, dict(probs=out, seconds=time.perf_counter() - t0, tokens=d.usage.input_tokens)

        t0 = time.time()
        with ThreadPoolExecutor(args.workers) as pool:
            for key, res in pool.map(call, jobs):
                cache[key] = res
        pickle.dump(cache, open(CACHE, "wb"))
        dec, secs, toks = {}, [], []
        for name, per_note in prepared.items():
            dec[name] = {}
            for n, (text, rs, p, ordr) in per_note.items():
                cand = order[(name, n)]
                key = (args.model, variant, wording, hashlib.sha256((text + "\x00" + "\x1f".join(rs[i]["surface"] for i in cand)).encode()).hexdigest())
                res = cache[key]
                secs.append(res["seconds"]); toks.append(res["tokens"])
                d = np.zeros(len(rs))
                for i, q in zip(cand, res["probs"]):
                    d[i] = 0.0 if q is None else q
                dec[name][n] = d

        def evaluate(name, combine, thr):
            gold, _ = data[name]
            sc = {}
            for n, (text, rs, p, _) in prepared[name].items():
                q = combine(p, dec[name][n])
                idx = suppress_nested(rs, q, thr)
                sc[n] = score(gold[n], [Extracted(rs[i]["surface"], frozenset({loose(rs[i]["surface"]), *(loose(text[a:b]) for a, b in rs[i]["spans"])}), float(q[i])) for i in idx])
            R = sum(s.core_hit for s in sc.values()) / sum(s.core for s in sc.values())
            P = sum(s.extracted_hit for s in sc.values()) / max(1, sum(s.extracted for s in sc.values()))
            return R, P

        dec_all[variant] = dec
        evaluators[variant] = evaluate
        print(f"\n== {variant}: latency per note median {statistics.median(secs):.2f}s p90 {np.percentile(secs, 90):.2f}s max {max(secs):.2f}s; input tokens median {int(statistics.median(toks)):,}", flush=True)
        for label, comb in (("decisions alone", lambda p, d: d), ("average with student", lambda p, d: (p + d) / 2)):
            best = max([round(x, 2) for x in np.arange(0.2, 0.96, 0.05)], key=lambda t: f1(*evaluate("dev", comb, t)))
            r = {k: evaluate(k, comb, best) for k in data}
            results[(variant, label)] = (best, r)
            print(f"   {label:22s} thr {best:.2f}  " + "   ".join(f"{r[k][0]:.1%}/{r[k][1]:.1%} ({f1(*r[k]):.3f})" for k in ("dev", "test", "unseen")), flush=True)

    if args.ensemble:
        ev = next(iter(evaluators.values()))
        print("\n== ensembles of the cached variants (thresholds chosen on dev)")
        for label, names in (("score + choice", ["score", "choice"]), ("score + criteria", ["score", "criteria"]), ("all four predicate/choice/score", [v for v in args.only if v != "boundary"])):
            avg = {k: {n: np.mean([dec_all[v][k][n] for v in names], axis=0) for n in dec_all[names[0]][k]} for k in data}

            def evaluate_e(name, combine, thr, avg=avg):
                gold, _ = data[name]
                sc = {}
                for n, (text, rs, p, _) in prepared[name].items():
                    q = combine(p, avg[name][n])
                    idx = suppress_nested(rs, q, thr)
                    sc[n] = score(gold[n], [Extracted(rs[i]["surface"], frozenset({loose(rs[i]["surface"]), *(loose(text[a:b]) for a, b in rs[i]["spans"])}), float(q[i])) for i in idx])
                R = sum(s.core_hit for s in sc.values()) / sum(s.core for s in sc.values())
                P = sum(s.extracted_hit for s in sc.values()) / max(1, sum(s.extracted for s in sc.values()))
                return R, P
            for comb_label, comb in (("alone", lambda p, d: d), ("average with student", lambda p, d: (p + d) / 2), ("student weight 0.3", lambda p, d: 0.3 * p + 0.7 * d)):
                best = max([round(x, 2) for x in np.arange(0.2, 0.96, 0.05)], key=lambda t: f1(*evaluate_e("dev", comb, t)))
                r = {k: evaluate_e(k, comb, best) for k in data}
                print(f"   {label:32s} {comb_label:22s} thr {best:.2f}  " + "   ".join(f"{r[k][0]:.1%}/{r[k][1]:.1%} ({f1(*r[k]):.3f})" for k in ("dev", "test", "unseen")), flush=True)


if __name__ == "__main__":
    main()

