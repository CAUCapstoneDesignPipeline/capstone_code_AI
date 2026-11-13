"""OpenAI Decisions API 를 개념 판정기로 쓴다: 학생이 후보를 내면(약 30ms) 후보마다 "이 노트에서 개념인가" 확률을 한 번의 요청으로 받는다.

  uv run --env-file .env python experiments/decisions/eval.py [--model gpt-6-luna] [--min-p 0.05]

노트 전체를 input 으로, 후보마다 predicate 질문 하나(최대 400)를 보낸다. 학생 확률이 min-p 이상인 후보만 묻는다 (확률 높은 순, 400개까지).
결과(후보별 확률·지연)는 캐시해 두고 임계값을 바꿔 가며 정답 노트(dev · test · 다른 분야 논문 리뷰)에서 잰다. 임계값은 dev F1 로 고른다.
노트 원문이 OpenAI 로 나간다. 호출마다 입력 토큰이 후보 수에 비례한다.
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
CACHE = Path("/private/tmp/claude-501/-Users-klee9-capstone/6ecbcd7c-aceb-429f-97e6-1e37e2c2da24/scratchpad/decisions_cache.pkl")
WORDINGS = {
    "loose": ('Is "{s}" a field-specific concept (not a generic word, heading frame, example value, or person/venue name) '
              'worth a node in a knowledge graph of this note?'),
    "strict": ('In this study note, is "{s}" a technical concept that deserves its own node in a knowledge graph (a field-specific term, algorithm, model, metric, '
               'dataset or named technique), and not a generic word, a heading frame, an example value, or a person/venue name?'),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-6-luna")
    ap.add_argument("--min-p", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--wording", default="strict", choices=sorted(WORDINGS))
    args = ap.parse_args()
    INSTR = WORDINGS[args.wording]

    import openai

    from app.cli import load_extractor
    from app.services.concepts.gold import Extracted, loose, score
    from app.services.concepts.student import Student, rows, suppress_nested
    from experiments.llm_only.baseline import sets_available

    ex = load_extractor(argparse.Namespace(idf=ROOT / "data/private/concepts/idf.json", cache=None))
    st = Student.load()
    client = openai.OpenAI(max_retries=2, timeout=120)
    data = sets_available(["dev", "test", "unseen"])
    cache = pickle.load(open(CACHE, "rb")) if CACHE.exists() else {}
    prepared, jobs = {}, []
    for name, (gold, texts) in data.items():
        prepared[name] = {}
        for n in sorted(gold):
            text, rs = rows(ex, texts[n], symbols=True)
            p = np.round(st.probabilities(np.array([r["x"] for r in rs], float)), 6)
            order = [i for i in np.argsort(-p, kind="stable") if p[i] >= args.min_p][:400]
            prepared[name][n] = (text, rs, p, order)
            key = (args.model, hashlib.sha256((INSTR + "\x00" + text + "\x00" + "\x1f".join(rs[i]["surface"] for i in order)).encode()).hexdigest())
            if key not in cache:
                jobs.append((key, text, [rs[i]["surface"] for i in order]))

    def call(job):
        key, text, cands = job
        t0 = time.perf_counter()
        d = client.decisions.create(model=args.model, input=text, questions=[{"type": "predicate", "instructions": INSTR.format(s=c)} for c in cands])
        probs = [a.probability if a.type == "predicate" else None for a in d.answers]
        return key, dict(probs=probs, seconds=time.perf_counter() - t0, tokens=d.usage.input_tokens)

    print(f"{len(jobs)} requests to make ({sum(len(c) for _, _, c in jobs)} questions); {len(cache)} cached", flush=True)
    with ThreadPoolExecutor(args.workers) as pool:
        for key, res in pool.map(call, jobs):
            cache[key] = res
    pickle.dump(cache, open(CACHE, "wb"))

    secs, toks = [], []
    dec = {}
    for name, per in prepared.items():
        dec[name] = {}
        for n, (text, rs, p, order) in per.items():
            key = (args.model, hashlib.sha256((INSTR + "\x00" + text + "\x00" + "\x1f".join(rs[i]["surface"] for i in order)).encode()).hexdigest())
            res = cache[key]
            secs.append(res["seconds"]); toks.append(res["tokens"])
            d = np.zeros(len(rs))
            for i, q in zip(order, res["probs"]):
                d[i] = 0.0 if q is None else q
            dec[name][n] = d
    print(f"decisions latency per note ({len(secs)} notes, {args.workers} in parallel): median {statistics.median(secs):.2f}s, p90 {np.percentile(secs, 90):.2f}s, max {max(secs):.2f}s; "
          f"input tokens median {int(statistics.median(toks))}, max {max(toks)}\n")

    def evaluate(name, combine, thr):
        gold, _ = data[name]
        sc = {}
        for n, (text, rs, p, order) in prepared[name].items():
            q = combine(p, dec[name][n])
            idx = suppress_nested(rs, q, thr)
            sc[n] = score(gold[n], [Extracted(rs[i]["surface"], frozenset({loose(rs[i]["surface"]), *(loose(text[a:b]) for a, b in rs[i]["spans"])}), float(q[i])) for i in idx])
        R = sum(s.core_hit for s in sc.values()) / sum(s.core for s in sc.values())
        P = sum(s.extracted_hit for s in sc.values()) / max(1, sum(s.extracted for s in sc.values()))
        return R, P

    f1 = lambda r, p: 2 * r * p / (r + p + 1e-9)
    methods = {"student alone (0.45)": (lambda p, d: p, [0.45]),
               "decisions alone": (lambda p, d: d, [round(x, 2) for x in np.arange(0.3, 0.96, 0.05)]),
               "average of student and decisions": (lambda p, d: (p + d) / 2, [round(x, 2) for x in np.arange(0.3, 0.9, 0.05)]),
               "decisions, only where student p >= 0.15": (lambda p, d: np.where(p >= 0.15, d, 0.0), [round(x, 2) for x in np.arange(0.3, 0.96, 0.05)])}
    print(f"{'method (threshold chosen on dev F1)':44s} thr   dev R/P (F1)         test R/P (F1)        new-domain R/P (F1)")
    for label, (comb, grid) in methods.items():
        best = max(grid, key=lambda t: f1(*evaluate("dev", comb, t)))
        r = {k: evaluate(k, comb, best) for k in data}
        print(f"{label:44s} {best:.2f}  " + "   ".join(f"{r[k][0]:.1%}/{r[k][1]:.1%} ({f1(*r[k]):.3f})" for k in ("dev", "test", "unseen")), flush=True)


if __name__ == "__main__":
    main()
