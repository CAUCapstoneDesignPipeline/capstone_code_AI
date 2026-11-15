"""학습한 결합기: 학생의 특징 62개 + Decisions API 답(score · criteria 두 질문의 확률)을 넣어 교사(LLM 이 뽑은 개념) 라벨로 부스팅 트리를 학습한다.

  uv run --env-file .env python experiments/decisions/combiner.py [--min-p 0.01]

학습 노트: 정답 없는 과목 노트 100 + 공개 글 98 + 어휘집용 글 468 (정답 노트·논문 리뷰는 학습에 쓰지 않는다). 후보 = 학생 확률이 min-p 이상인 것(400개까지)만 Decisions 에 묻고,
묻지 않은 후보는 개념이 아니라고 본다. 결과는 같은 두 질문으로 만든 고정 결합(학생 30% + Decisions 70%), 학생만, Decisions 답 없이 같은 자료로 다시 학습한 것과 견준다.
임계값은 dev 정답 노트 F1 로 고른다. 노트 원문이 OpenAI 로 나간다 (호출당 입력 토큰은 후보 수에 비례).
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
CACHE = scratch("decisions_variants_cache.pkl")
VARIANTS = ("score", "criteria")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-p", type=float, default=0.01)
    ap.add_argument("--model", default="gpt-6-luna")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--max-lex", type=int, default=0)
    args = ap.parse_args()

    import openai
    from sklearn.ensemble import HistGradientBoostingClassifier

    from app.cli import load_extractor
    from app.services.concepts.gold import Extracted, loose, score
    from app.services.concepts.markdown import nfc
    from app.services.concepts.student import Student, rows, suppress_nested
    from experiments.decisions.variants import make
    from experiments.distill.features import teacher_concepts
    from experiments.distill.teacher import teacher_notes
    from experiments.llm_only.baseline import sets_available

    ex = load_extractor(argparse.Namespace(idf=ROOT / "data/private/concepts/idf.json", cache=None))
    st = Student.load()
    client = openai.OpenAI(max_retries=12, timeout=180)
    cache = pickle.load(open(CACHE, "rb")) if CACHE.exists() else {}

    def load(folder):
        return {p.stem: nfc(p.read_text(encoding="utf-8")) for p in sorted((ROOT / "data/private" / folder).glob("*.md"))}

    pools = {"course": teacher_notes(), "web": load("notes-web"), "lex": load("notes-lex")}
    if args.max_lex:
        pools["lex"] = dict(list(pools["lex"].items())[:args.max_lex])
    data = sets_available(["dev", "test", "unseen"])
    items = []                                              # (집합, 이름, 원문, 정답 개념 or None)
    for pool, notes in pools.items():
        for n, t in notes.items():
            items.append((pool, n, t, None))
    for name, (gold, texts) in data.items():
        for n in sorted(gold):
            items.append((name, n, texts[n], gold[n]))

    wording = {v: hashlib.sha256(repr([make(v)[0]("X"), "v1"]).encode()).hexdigest()[:10] for v in VARIANTS}
    prepared, jobs = [], []
    t0 = time.time()
    for pool, n, text, gold in items:
        text, rs = rows(ex, text, symbols=True)
        if not rs:
            continue
        p = np.round(st.probabilities(np.array([r["x"] for r in rs], float)), 6)
        cand = [int(i) for i in np.argsort(-p, kind="stable") if p[i] >= args.min_p][:400]
        prepared.append(dict(pool=pool, name=n, text=text, rs=rs, p=p, cand=cand, gold=gold))
        surfaces = [rs[i]["surface"] for i in cand]
        for v in VARIANTS:
            key = (args.model, v, wording[v], hashlib.sha256((text + "\x00" + "\x1f".join(surfaces)).encode()).hexdigest())
            prepared[-1][v] = key
            if key not in cache:
                jobs.append((key, v, text, surfaces))
    print(f"{len(prepared)} notes prepared ({time.time() - t0:.0f}s); {len(jobs)} Decisions requests to make ({sum(len(j[3]) for j in jobs)} questions)", flush=True)

    def call(job):
        key, v, text, surfaces = job
        build, reduce, per = make(v)
        for attempt in range(60):                           # 분당 토큰 한도(TPM)에 걸리면 창이 풀릴 때까지 기다린다
            try:
                d = client.decisions.create(model=args.model, input=text, questions=[q for c in surfaces for q in build(c)])
                break
            except (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError):
                time.sleep(min(30, 3 + attempt * 2))
        else:
            raise RuntimeError("rate limit did not clear")
        out = []
        for k in range(len(surfaces)):
            ans = d.answers[k * per:(k + 1) * per]
            out.append(None if any(a.type == "refusal" for a in ans) else reduce(ans))
        return key, dict(probs=out, tokens=d.usage.input_tokens)

    done = 0
    with ThreadPoolExecutor(args.workers) as pool:
        for key, res in pool.map(call, jobs):
            cache[key] = res
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(jobs)} requests ({time.time() - t0:.0f}s)", flush=True)
                pickle.dump(cache, open(CACHE, "wb"))
    pickle.dump(cache, open(CACHE, "wb"))

    def matrix(nt, with_d):
        """후보 행만: 학생 특징 (+ Decisions 확률 둘과 평균)."""
        rs, cand = nt["rs"], nt["cand"]
        d = {v: np.array([0.0 if q is None else q for q in cache[nt[v]]["probs"]]) for v in VARIANTS}
        X = np.array([rs[i]["x"] for i in cand], float).reshape(len(cand), -1)
        if with_d:
            X = np.column_stack([X, d["score"], d["criteria"], (d["score"] + d["criteria"]) / 2])
        return X, d

    train, evals = [], {k: [] for k in data}
    for nt in prepared:
        if nt["pool"] in evals:
            evals[nt["pool"]].append(nt)
        else:
            g = teacher_concepts(nt["text"])
            if g is None:
                continue
            forms = frozenset().union(*[k for _, k in g]) if g else frozenset()
            y = [int(bool({loose(nt["rs"][i]["surface"]), *(loose(nt["text"][s:e]) for s, e in nt["rs"][i]["spans"][:3])} & forms)) for i in nt["cand"]]
            train.append((nt, np.array(y)))
    print(f"training notes {len(train)} ({sum(len(y) for _, y in train)} asked candidates, {np.mean(np.concatenate([y for _, y in train])):.1%} positive); eval notes {sum(len(v) for v in evals.values())}", flush=True)

    def fit(with_d):
        X = np.vstack([matrix(nt, with_d)[0] for nt, _ in train])
        y = np.concatenate([y for _, y in train])
        return HistGradientBoostingClassifier(max_depth=5, learning_rate=0.08, max_iter=150, min_samples_leaf=30, l2_regularization=1.0, random_state=0).fit(X, y)

    models = {"retrained, no Decisions answers": (fit(False), False), "learned combiner (features + Decisions)": (fit(True), True)}

    def prob_vector(nt, how):
        n_rows = len(nt["rs"])
        q = np.zeros(n_rows)
        X, d = matrix(nt, how in models and models[how][1])
        if how in models:
            q[nt["cand"]] = models[how][0].predict_proba(X)[:, 1]
        elif how == "fixed blend (30% student + 70% Decisions)":
            q[nt["cand"]] = 0.3 * nt["p"][nt["cand"]] + 0.7 * (d["score"] + d["criteria"]) / 2
        elif how == "student alone":
            q = nt["p"].copy()
        return q

    def evaluate(split, how, thr):
        sc = {}
        for nt in evals[split]:
            q = prob_vector(nt, how)
            idx = suppress_nested(nt["rs"], q, thr)
            sc[nt["name"]] = score(nt["gold"], [Extracted(nt["rs"][i]["surface"], frozenset({loose(nt["rs"][i]["surface"]), *(loose(nt["text"][a:b]) for a, b in nt["rs"][i]["spans"])}), float(q[i])) for i in idx])
        R = sum(s.core_hit for s in sc.values()) / sum(s.core for s in sc.values())
        P = sum(s.extracted_hit for s in sc.values()) / max(1, sum(s.extracted for s in sc.values()))
        return R, P

    f1 = lambda r, p: 2 * r * p / (r + p + 1e-9)
    print(f"\n{'method (threshold chosen on dev F1)':44s} thr   dev R/P (F1)         test R/P (F1)        new-domain R/P (F1)")
    for how in ("student alone", "fixed blend (30% student + 70% Decisions)", "retrained, no Decisions answers", "learned combiner (features + Decisions)"):
        grid = [0.45] if how == "student alone" else [round(x, 2) for x in np.arange(0.2, 0.9, 0.05)]
        best = max(grid, key=lambda t: f1(*evaluate("dev", how, t)))
        r = {k: evaluate(k, how, best) for k in data}
        print(f"{how:44s} {best:.2f}  " + "   ".join(f"{r[k][0]:.1%}/{r[k][1]:.1%} ({f1(*r[k]):.3f})" for k in ("dev", "test", "unseen")), flush=True)
    pickle.dump({k: v[0] for k, v in models.items()}, open(scratch("combiner_models.pkl"), "wb"))


if __name__ == "__main__":
    main()
