"""학생 모델: 교사(LLM 만으로 뽑은 개념)를 흉내 내는 빠른 분류기. 학습은 정답 없는 과목 노트의 교사 라벨로만 한다.

  uv run --env-file .env python experiments/distill/student.py [--threshold 0.5] [--no-nms]

1. 후보: 토큰 n-gram (candidates.py, 정답 core 의 96~98% 를 후보로 낸다).
2. 특징: 모양·위치·구조·말뭉치 통계·교사 라벨 사전 (features.py).
3. 모델: 그래디언트 부스팅. 임계값은 dev 정답 노트에서 F1 이 가장 높은 값으로 고르고, test · unseen(다른 분야) 은 그 값으로 한 번만 잰다.
4. 겹침 정리: 더 긴 후보에 언급이 모두 덮인 짧은 후보는 뺀다.
평가는 baseline.py 와 같은 정답·채점이다. 정답은 AI 초안이라 사람 검토 전의 수치다.
"""

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def fit_model(X, y, seed=0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_depth=6, learning_rate=0.06, max_iter=300, min_samples_leaf=30, l2_regularization=1.0, random_state=seed).fit(X, y)


EMB_CACHE = Path("/private/tmp/claude-501/-Users-klee9-capstone/6ecbcd7c-aceb-429f-97e6-1e37e2c2da24/scratchpad/emb-cache.pkl")


def embed_all(strings):
    """표기 → bge-m3 벡터 (정규화됨). 디스크에 캐시한다."""
    import pickle
    cache = pickle.load(open(EMB_CACHE, "rb")) if EMB_CACHE.exists() else {}
    todo = sorted(set(strings) - set(cache))
    if todo:
        from app.main import load_embedder
        emb = load_embedder("embeddings-bge-m3-1")
        t0 = time.time()
        for i in range(0, len(todo), 64):
            for t, v in zip(todo[i:i + 64], emb.embed(todo[i:i + 64])):
                cache[t] = np.asarray(v, dtype=np.float32)
            if (i // 64) % 100 == 0:
                print(f"  embedded {i}/{len(todo)} ({time.time()-t0:.0f}s)", flush=True)
        pickle.dump(cache, open(EMB_CACHE, "wb"))
    return cache


def note_sims(rows, E):
    """후보 표기와 그 노트 전체(언급 수로 가중한 평균 벡터)의 코사인 — 주제와 얼마나 가까운가."""
    V = np.stack([E[r["surface"]] for r in rows])
    w = np.array([len(r["spans"]) for r in rows], float)
    c = (V * w[:, None]).sum(0)
    c /= np.linalg.norm(c) + 1e-9
    return V @ c


def nms(rows, probs, thr):
    """임계값 이상인 후보 중, 모든 언급이 더 긴 채택 후보 안에 든 짧은 후보를 뺀다."""
    chosen = [i for i, p in enumerate(probs) if p >= thr]
    chosen.sort(key=lambda i: (-len(rows[i]["surface"]), -probs[i]))
    kept, kept_spans = [], []
    for i in chosen:
        mine = rows[i]["spans"]
        if kept_spans and all(any(ks <= s and e <= ke and (ke - ks) > (e - s) for ks, ke in kept_spans) for s, e in mine):
            continue
        kept.append(i)
        kept_spans += mine
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-nms", action="store_true")
    ap.add_argument("--threshold", type=float, default=None, help="없으면 dev 에서 F1 이 가장 높은 값")
    ap.add_argument("--max-teacher", type=int, default=0)
    ap.add_argument("--drop", default="", help="뺄 특징 묶음: corpus(df·idf·old_score) lex(교사 사전) emb(탐침·주제 거리)")
    args = ap.parse_args()

    from app.cli import load_extractor
    from app.services.concepts.gold import Extracted, loose, score
    from experiments.distill.features import FEATURES, Lexicon, note_rows, old_scores, teacher_forms
    from experiments.distill.teacher import teacher_notes
    from experiments.llm_only.baseline import sets_available

    ex = load_extractor(argparse.Namespace(idf=ROOT / "data/private/concepts/idf.json", cache=None))
    from app.services.concepts.markdown import nfc
    all_course = {nfc(p.stem): nfc(p.read_text(encoding="utf-8")) for p in sorted((ROOT / "data/private/notes").glob("*.md"))}
    from experiments.distill.candidates import generate
    df = defaultdict(int)
    t0 = time.time()
    for n, t in all_course.items():
        for k in generate(ex, t)[1]:
            df[k] += 1
    print(f"df over {len(all_course)} course notes: {len(df)} keys ({time.time()-t0:.0f}s)", flush=True)

    # --- 학습 데이터: 정답 없는 노트 + 교사 라벨
    train, skipped = [], 0
    for n, t in teacher_notes().items():
        forms = teacher_forms(t)
        if forms is None:
            skipped += 1
            continue
        text, rows = note_rows(ex, t, df, len(all_course), old_scores(ex, n, t))
        ys = [bool({loose(r["surface"]), *(loose(text[s:e]) for s, e in r["spans"][:3])} & forms) for r in rows]
        train.append((n, rows, ys))
        if args.max_teacher and len(train) >= args.max_teacher:
            break
    print(f"train notes {len(train)} (teacher labels missing: {skipped}), rows {sum(len(r) for _, r, _ in train)}, positives {sum(sum(y) for _, _, y in train)}", flush=True)
    lex = Lexicon()
    for _, rows, ys in train:
        lex.add(rows, ys)
    lex.prior = sum(sum(y) for _, _, y in train) / max(1, sum(len(y) for _, _, y in train))
    # 평가 노트의 후보 (특징은 아래에서 채운다)
    data = sets_available(["dev", "test", "unseen"])
    prep = {}
    for name, (gold, texts) in data.items():
        prep[name] = {n: note_rows(ex, texts[n], df, len(all_course), old_scores(ex, n, texts[n])) for n in sorted(gold)}
    # 임베딩: 후보 표기 → 벡터, 표기만 보고 개념인지 고르는 선형 탐침(노트 단위 교차로 학습해 새는 것을 막는다) + 노트 주제와의 가까움
    strings = [r["surface"] for _, rows, _ in train for r in rows] + [r["surface"] for per in prep.values() for _, rows in per.values() for r in rows]
    E = embed_all(strings)
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    Etr = np.vstack([np.stack([E[r["surface"]] for r in rows]) for _, rows, _ in train])
    y = np.array([int(v) for _, _, ys in train for v in ys])
    groups = np.concatenate([[i] * len(rows) for i, (_, rows, _) in enumerate(train)])
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(Etr, y, groups):
        oof[te] = LogisticRegression(C=0.5, max_iter=400).fit(Etr[tr], y[tr]).predict_proba(Etr[te])[:, 1]
    probe = LogisticRegression(C=0.5, max_iter=400).fit(Etr, y)
    sims_tr = np.concatenate([note_sims(rows, E) for _, rows, _ in train])
    X = np.array([r["x"] + lex.feats(r, yy) for _, rows, ys in train for r, yy in zip(rows, ys)], float)
    X = np.column_stack([X, oof, sims_tr])
    names = FEATURES[:-6] + [f"lex{i}" for i in range(6)] + ["emb_probe", "emb_sim"]
    drop = set()
    for g in filter(None, args.drop.split(",")):
        drop |= {"corpus": {"log_df", "idf", "old_score"}, "lex": {f"lex{i}" for i in range(6)}, "emb": {"emb_probe", "emb_sim"}}[g]
    keep_cols = [i for i, nme in enumerate(names) if nme not in drop]
    assert len(names) == X.shape[1], (len(names), X.shape)
    X = X[:, keep_cols]
    model = fit_model(X, y)
    print(f"trained on {len(y)} rows ({y.mean():.1%} positive); probe OOF AUC-ish mean prob pos {oof[y == 1].mean():.2f} vs neg {oof[y == 0].mean():.2f}", flush=True)
    evals = {}
    for name, per in prep.items():
        evals[name] = {}
        for n, (text, rows) in per.items():
            Ev = np.stack([E[r["surface"]] for r in rows])
            Xe = np.column_stack([np.array([r["x"] + lex.feats(r) for r in rows], float), probe.predict_proba(Ev)[:, 1], note_sims(rows, E)])[:, keep_cols]
            evals[name][n] = (text, rows, model.predict_proba(Xe)[:, 1])

    def run(name, thr):
        gold, _ = data[name]
        sc = {}
        for n, (text, rows, p) in evals[name].items():
            idx = nms(rows, p, thr) if not args.no_nms else [i for i, q in enumerate(p) if q >= thr]
            ext = [Extracted(rows[i]["surface"], frozenset({loose(rows[i]["surface"]), *(loose(text[s:e]) for s, e in rows[i]["spans"])}), float(p[i])) for i in idx]
            sc[n] = score(gold[n], ext)
        R = sum(s.core_hit for s in sc.values()) / sum(s.core for s in sc.values())
        P = sum(s.extracted_hit for s in sc.values()) / max(1, sum(s.extracted for s in sc.values()))
        return R, P, sum(s.extracted for s in sc.values())

    grid = [round(x, 2) for x in np.arange(0.05, 0.96, 0.05)]
    best = max(grid, key=lambda t: (lambda r, p, _: 2 * r * p / (r + p + 1e-9))(*run("dev", t))) if args.threshold is None else args.threshold
    print("\ndev curve (threshold: recall / precision / kept):")
    for t in grid[::2]:
        r, p, k = run("dev", t)
        print(f"  {t:.2f}: {r:.1%} / {p:.1%} / {k}{'   <- chosen' if t == best else ''}")
    print(f"\nthreshold {best} (chosen on dev){' · no NMS' if args.no_nms else ''}")
    for name in ("dev", "test", "unseen"):
        r, p, k = run(name, best)
        print(f"  {name:7s} core recall {r:.1%}  precision {p:.1%}  kept {k}")


if __name__ == "__main__":
    main()
