"""`extract-student-1` 학습과 내보내기: 교사(LLM 이 뽑은 개념) 라벨로 부스팅 트리를 학습해 app/services/concepts/student-1.json 에 쓴다.

  uv run --env-file .env python experiments/distill/train_student.py [--write] [--max-depth 5] [--iters 150]

학습 자료: 정답 없는 과목 노트 + data/private/notes-web 의 공개 글 (정답 노트 · unseen 은 학습에 쓰지 않는다: 그래서 보고하는 수치가 이 모델의 수치다).
임계값은 노트 단위 5겹 교차(out-of-fold) 확률로 교사에 대한 F1 이 가장 높은 값을 고른다. 내보낸 트리는 sklearn 의 확률과 같아야 한다(코드가 확인한다).
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def export_trees(model) -> dict:
    """sklearn HistGradientBoostingClassifier → JSON 으로 쓸 수 있는 트리 목록 (잎 값에 학습률이 이미 곱해져 있다)."""
    trees = []
    for stage in model._predictors:
        nodes = stage[0].nodes
        left = [int(v) if not leaf else -1 for v, leaf in zip(nodes["left"], nodes["is_leaf"])]
        right = [int(v) if not leaf else -1 for v, leaf in zip(nodes["right"], nodes["is_leaf"])]
        trees.append({"feature": [int(v) for v in nodes["feature_idx"]], "threshold": [float(v) for v in nodes["num_threshold"]],
                      "left": left, "right": right, "value": [float(v) if leaf else 0.0 for v, leaf in zip(nodes["value"], nodes["is_leaf"])]})
    return {"baseline": float(np.ravel(model._baseline_prediction)[0]), "trees": trees,
            "max_depth": int(max(int(s[0].nodes["depth"].max()) for s in model._predictors)) + 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--max-depth", type=int, default=5)
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--lr", type=float, default=0.08)
    ap.add_argument("--threshold", type=float, default=None, help="없으면 교사 F1 이 가장 높은 값. 출고 값(0.45)은 dev 정답 노트의 F1 로 골라 student-1.json 에 직접 적었다")
    args = ap.parse_args()

    import json
    import sklearn
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold

    from app.cli import load_extractor
    from app.services.concepts.gold import loose
    from app.services.concepts.markdown import nfc
    from app.services.concepts.student import FEATURES, Student, rows, suppress_nested
    from experiments.distill.features import teacher_concepts
    from experiments.distill.teacher import teacher_notes

    ex = load_extractor(argparse.Namespace(idf=ROOT / "data/private/concepts/idf.json", cache=None))
    web = {p.stem: nfc(p.read_text(encoding="utf-8")) for p in sorted((ROOT / "data/private/notes-web").glob("*.md"))}
    notes = []
    for name, text in [*teacher_notes().items(), *web.items()]:
        g = teacher_concepts(text)
        if g is None:
            continue
        forms = frozenset().union(*[k for _, k in g]) if g else frozenset()
        t2, rs = rows(ex, text)
        ys = [bool({loose(r["surface"]), *(loose(t2[s:e]) for s, e in r["spans"][:3])} & forms) for r in rs]
        notes.append(dict(name=name, text=t2, rows=rs, y=ys, forms=forms))
    X = np.array([r["x"] for d in notes for r in d["rows"]], float)
    y = np.array([int(v) for d in notes for v in d["y"]])
    group = np.concatenate([[i] * len(d["rows"]) for i, d in enumerate(d_ for d_ in notes)])
    print(f"{len(notes)} notes, {len(y)} candidates, {y.mean():.1%} positive, {len(FEATURES)} features", flush=True)

    def fit(Xt, yt):
        return HistGradientBoostingClassifier(max_depth=args.max_depth, learning_rate=args.lr, max_iter=args.iters, min_samples_leaf=30,
                                              l2_regularization=1.0, random_state=0).fit(Xt, yt)

    oof = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, group):
        oof[te] = fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    offsets = np.cumsum([0] + [len(d["rows"]) for d in notes])

    def vs_teacher(thr):
        hit = tot = tp = kept = 0
        for i, d in enumerate(notes):
            p = oof[offsets[i]:offsets[i + 1]]
            idx = suppress_nested(d["rows"], p, thr)
            got = set()
            for j in idx:
                r = d["rows"][j]
                got |= {loose(r["surface"]), *(loose(d["text"][s:e]) for s, e in r["spans"])}
            hit += sum(f in got for f in d["forms"])
            tot += len(d["forms"])
            tp += sum(d["y"][j] for j in idx)
            kept += len(idx)
        return hit / max(1, tot), tp / max(1, kept)

    grid = [round(x, 2) for x in np.arange(0.1, 0.81, 0.05)]
    curve = {t: vs_teacher(t) for t in grid}
    thr = args.threshold if args.threshold is not None else max(grid, key=lambda t: 2 * curve[t][0] * curve[t][1] / (curve[t][0] + curve[t][1] + 1e-9))
    curve.setdefault(thr, vs_teacher(thr))
    print("out-of-fold agreement with the teacher (recall of teacher forms / precision), by threshold:")
    for t in grid[::2]:
        print(f"  {t:.2f}: {curve[t][0]:.1%} / {curve[t][1]:.1%}{'   <- chosen' if t == thr else ''}")

    model = fit(X, y)
    exported = export_trees(model)
    data = {"id": "extract-student-1", "features": FEATURES, "threshold": thr, **exported,
            "trained": {"notes": len(notes), "candidates": int(len(y)), "positives": int(y.sum()), "max_depth": args.max_depth, "iterations": args.iters,
                        "learning_rate": args.lr, "teacher": "llm-only-concepts-1 (opus, effort medium)", "sklearn": sklearn.__version__,
                        "oof_teacher_recall": round(curve[thr][0], 4), "oof_teacher_precision": round(curve[thr][1], 4)}}
    student = Student(data)
    diff = float(np.abs(student.probabilities(X[:5000]) - model.predict_proba(X[:5000])[:, 1]).max())
    print(f"exported {len(data['trees'])} trees, depth {data['max_depth']}; max |numpy - sklearn| on 5000 rows = {diff:.2e}")
    assert diff < 1e-9, "내보낸 트리가 sklearn 과 다르다"
    if args.write:
        out = ROOT / "app/services/concepts/student-1.json"
        out.write_text(json.dumps(data, separators=(",", ":"), ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"→ {out.relative_to(ROOT)} ({out.stat().st_size / 1024:.0f} KB), threshold {thr}")


if __name__ == "__main__":
    main()
