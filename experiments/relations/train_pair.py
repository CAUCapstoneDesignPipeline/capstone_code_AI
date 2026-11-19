"""빠른 관계 모델 (`pairs-1`) 학습과 내보내기: 정답 노트 59개 전체로 트리 둘을 학습해 app/services/relations/pairs-1.json 에 쓴다.

  uv run --env-file .env python experiments/relations/train_pair.py --threshold 0.2 [--write]

교사 = 같은 노트를 Claude Sonnet 이 읽고 낸 관계 (llm_only 캐시, 새로 부르지 않는다), 특징 = 구조·위치 + Decisions 객관식 확률 (relations_decisions.pkl 캐시).
임계값은 pair_model.py 의 노트 단위 교차검증 예측을 풀링 판정(pool.py)으로 채점해 dev 에서 strict F1 이 가장 높은 값을 고른 것이다 (출고 값 0.2 는 탐색한 격자의
가장자리라 더 낮은 값은 재 보지 못했다: 판정이 그 아래 예측까지 다 있지는 않다). 학습 하이퍼파라미터는 교차검증과 같다.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def export_forest(model, k: int = 0) -> dict:
    """sklearn HistGradientBoostingClassifier 의 k 번째 클래스 트리 → JSON (잎 값에 학습률이 곱해져 있다)."""
    trees = []
    for stage in model._predictors:
        nodes = stage[k].nodes
        leaf = nodes["is_leaf"]
        trees.append({"feature": [int(v) for v in nodes["feature_idx"]], "threshold": [float(v) for v in nodes["num_threshold"]],
                      "left": [int(v) if not l else -1 for v, l in zip(nodes["left"], leaf)], "right": [int(v) if not l else -1 for v, l in zip(nodes["right"], leaf)],
                      "value": [float(v) if l else 0.0 for v, l in zip(nodes["value"], leaf)]})
    return {"baseline": float(np.ravel(model._baseline_prediction)[k]), "trees": trees,
            "max_depth": int(max(int(s[k].nodes["depth"].max()) for s in model._predictors)) + 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--teacher", default="llm:anthropic:claude-sonnet-5-5")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    import openai
    import sklearn
    from sklearn.ensemble import HistGradientBoostingClassifier

    from app.services.relations.pairs import CLASSES, FEATURES, PairModel
    from experiments.relations import decisions_pairs as D
    from experiments.relations import harness, llm_only
    from experiments.relations.candidates import candidate_pairs
    from experiments.relations.pair_model import features, teacher_labels

    data = harness.load()
    cache = pickle.load(open(D.CACHE, "rb"))
    client = openai.OpenAI(max_retries=2, timeout=120)
    teacher = llm_only.make(args.teacher[4:])
    Xs, yps, yts, n_notes = [], [], [], 0
    for notes in data.values():
        for note in notes.values():
            ms = harness.mentions_for(note)
            pairs = candidate_pairs(note.text, ms)
            probs, _ = D.ask_all(client, "gpt-6-luna", note, pairs, cache, named=True)
            dec = {(a, b): (p or {}) for (a, b, _), p in zip(pairs, probs)}
            yp, yt = teacher_labels(teacher, note, pairs)
            Xs.append(features(note, ms, pairs, dec)); yps.append(yp); yts.append(yt)
            n_notes += 1
    X, yp, yt = np.vstack(Xs), np.concatenate(yps), np.concatenate(yts)
    assert X.shape[1] == len(FEATURES)
    mk = lambda: HistGradientBoostingClassifier(max_depth=5, learning_rate=0.08, max_iter=150, min_samples_leaf=20, l2_regularization=1.0, random_state=0)
    pair_model = mk().fit(X, yp)
    pos = np.where(yt >= 0)[0]
    type_model = mk().fit(X[pos], yt[pos])
    classes = [int(c) for c in type_model.classes_]
    out = {"id": "pairs-1", "features": FEATURES, "classes": CLASSES, "threshold": args.threshold,
           "pair": export_forest(pair_model), "type": {"classes": classes, "per_class": [export_forest(type_model, k) for k in range(len(classes))]},
           "trained": {"notes": n_notes, "pairs": int(len(yp)), "positives": int(yp.sum()), "teacher": "relations-llm-only-1 (claude-sonnet-5-5, effort medium)",
                       "decisions": "gpt-6-luna, named choices", "max_depth": 5, "iterations": 150, "learning_rate": 0.08, "sklearn": sklearn.__version__,
                       "cv_strict_f1": {"dev": 0.425, "test": 0.407, "unseen": 0.404}}}
    m = PairModel(out)
    p, t = m.predict(X)
    d_pair = float(np.abs(p - np.round(pair_model.predict_proba(X)[:, 1], 6)).max())
    ref = type_model.classes_[type_model.predict_proba(X).argmax(1)]
    agree = float((t == ref).mean())
    print(f"쌍 {len(yp)}개 · 양성 {int(yp.sum())}개 · 트리 {len(out['pair']['trees'])} + {len(classes)}×{len(out['type']['per_class'][0]['trees'])}; "
          f"max |numpy - sklearn| (관계 확률) = {d_pair:.2e}, 유형 argmax 일치 {agree:.4%}")
    assert d_pair < 1e-5 and agree == 1.0, "내보낸 트리가 sklearn 과 다르다"
    if args.write:
        path = ROOT / "app/services/relations/pairs-1.json"
        path.write_text(json.dumps(out, separators=(",", ":"), ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"쓰기: {path} ({path.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
