"""관계 학생(빠른 typer) 학습·평가: LLM typer 의 판정을 따라 하게 학습하고, 본 적 없는 노트에서 얼마나 같은지 잰다.

  uv run --env-file .env python experiments/relations/train_eval.py

학습: 정답 개념이 없는 과목 노트 100개의 쌍 (캐시된 LLM 판정이 라벨). 평가: dev · test 노트의 쌍 (학습에 쓰지 않았다).
유형은 8개 + none 의 9분류, 방향은 관계가 있는 쌍에서 따로 이진 분류한다. 임계값(관계로 볼 확률)은 dev 에서 F1 이 가장 높은 값이다.
"""

import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
DATA = Path("/private/tmp/claude-501/-Users-klee9-capstone/6ecbcd7c-aceb-429f-97e6-1e37e2c2da24/scratchpad/rel_dataset.pkl")


def main():
    from sklearn.ensemble import HistGradientBoostingClassifier

    from experiments.relations.pairs import SYMMETRIC, TYPES, Vocab

    rows = [r for r in pickle.load(open(DATA, "rb")) if r["label"] is not None]
    if "--clean" in sys.argv:                       # 두 개념이 모두 LLM 이 그 노트에서 뽑은 개념인 쌍만 (잡음 개념 쌍을 뺀다)
        from app.services.concepts.gold import loose
        from app.services.concepts.markdown import nfc
        from experiments.distill.features import teacher_concepts
        forms = {}
        for p in (ROOT / "data/private/notes").glob("*.md"):
            g = teacher_concepts(nfc(p.read_text(encoding="utf-8")))
            forms[nfc(p.stem)] = frozenset().union(*[k for _, k in g]) if g else frozenset()
        rows = [r for r in rows if loose(r["a"]) in forms.get(r["note"], ()) and loose(r["b"]) in forms.get(r["note"], ())]
        print(f"clean pairs (both concepts are LLM concepts): {len(rows)}")
    train = [r for r in rows if r["split"] == "none"]
    evals = {s: [r for r in rows if r["split"] == s] for s in ("dev", "test")}
    classes = ["none"] + TYPES
    vocab = Vocab([r["raw"] for r in train])
    print(f"train {len(train)} pairs from {len({r['note'] for r in train})} notes ({sum(r['label'][0] != 'none' for r in train) / len(train):.0%} relations); "
          f"dev {len(evals['dev'])}, test {len(evals['test'])}; {len(vocab.names)} features")
    X = np.array([vocab.vector(r["raw"]) for r in train], float)
    y = np.array([classes.index(r["label"][0]) for r in train])
    clf = HistGradientBoostingClassifier(max_depth=5, learning_rate=0.08, max_iter=200, min_samples_leaf=20, l2_regularization=1.0, random_state=0).fit(X, y)
    pos = y > 0
    asym = np.array([classes[c] not in SYMMETRIC for c in y]) & pos
    dir_clf = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.1, max_iter=100, min_samples_leaf=15, random_state=0).fit(
        X[asym], np.array([int(r["label"][1]) for r, m in zip(train, asym) if m]))

    def predict(items, thr):
        Xe = np.array([vocab.vector(r["raw"]) for r in items], float)
        P = clf.predict_proba(Xe)
        best = P[:, 1:].argmax(1) + 1
        is_rel = (1 - P[:, 0]) >= thr
        d = dir_clf.predict_proba(Xe)[:, 1] >= 0.5
        return [(classes[b] if m else "none", bool(dd)) for b, m, dd in zip(best, is_rel, d)], 1 - P[:, 0]

    def metrics(items, thr):
        pred, _ = predict(items, thr)
        tp = fp = fn = type_ok = both_pos = dir_ok = dir_n = 0
        per = {t: [0, 0, 0] for t in TYPES}                   # tp, fp, fn
        for r, (pt, pd_) in zip(items, pred):
            lt, ld, _ = r["label"]
            if lt != "none" and pt != "none":
                tp += 1
                type_ok += lt == pt
                if lt == pt and lt not in SYMMETRIC:
                    dir_n += 1
                    dir_ok += ld == pd_
            fp += lt == "none" and pt != "none"
            fn += lt != "none" and pt == "none"
            if pt != "none":
                per[pt][0 if lt == pt else 1] += 1
            if lt != "none" and lt != pt:
                per[lt][2] += 1
        p, rc = tp / max(1, tp + fp), tp / max(1, tp + fn)
        full_tp = sum(1 for r, (pt, pd_) in zip(items, pred) if r["label"][0] == pt != "none" and (pt in SYMMETRIC or r["label"][1] == pd_))
        full_p = full_tp / max(1, sum(pt != "none" for pt, _ in pred))
        full_r = full_tp / max(1, sum(r["label"][0] != "none" for r in items))
        return dict(detect_p=p, detect_r=rc, detect_f1=2 * p * rc / max(1e-9, p + rc), type_acc=type_ok / max(1, tp), dir_acc=dir_ok / max(1, dir_n),
                    full_p=full_p, full_r=full_r, full_f1=2 * full_p * full_r / max(1e-9, full_p + full_r), per=per, n_pos=sum(r["label"][0] != "none" for r in items))

    grid = [round(x, 2) for x in np.arange(0.2, 0.81, 0.05)]
    best_thr = max(grid, key=lambda t: metrics(evals["dev"], t)["full_f1"])
    if "--curve" in sys.argv:
        print("\nthreshold curve (full = type and direction match the LLM's label):")
        for s_, items in evals.items():
            n_notes = len({r["note"] for r in items})
            for t in (0.3, 0.5, 0.6, 0.7, 0.8, 0.9):
                m = metrics(items, t)
                pred, _ = predict(items, t)
                n_rel = sum(pt != "none" for pt, _ in pred)
                print(f"  {s_:5s} thr {t:.1f}: full precision {m['full_p']:.0%}  recall {m['full_r']:.0%}  -> {n_rel / n_notes:5.1f} relations per note (LLM typer: {m['n_pos'] / n_notes:.1f}, rules+structure today: ~1)")
    print(f"\nthreshold {best_thr} (chosen on dev by full relation F1)\n")
    print("set     pairs  teacher-relations | detect P/R   | type acc | direction acc | full (type+direction) P / R / F1")
    for s, items in evals.items():
        m = metrics(items, best_thr)
        print(f"{s:6s} {len(items):6d} {m['n_pos']:12d}      | {m['detect_p']:.0%} / {m['detect_r']:.0%}  | {m['type_acc']:.0%}      | {m['dir_acc']:.0%}           | {m['full_p']:.1%} / {m['full_r']:.1%} / {m['full_f1']:.3f}")
    m = metrics(evals["dev"], best_thr)
    print("\ndev per type (predicted right / predicted wrong / teacher's missed):")
    for t, (a, b, c) in m["per"].items():
        print(f"  {t:15s} {a:4d} / {b:4d} / {c:4d}")
    base = Counter(r["label"][0] for r in evals["dev"])
    print("\nteacher label counts on dev:", dict(base.most_common()))


if __name__ == "__main__":
    main()
