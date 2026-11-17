"""관계 특징이 후보 순위를 개선하는가: 라벨 쌍에서 노트 단위 교차검증 (반복) 으로 AUC·상위 k 적중을 잰다.

  uv run --env-file .env python experiments/proposals/relation_features.py     # features.pkl 을 만든 뒤
  uv run python experiments/proposals/rank_cv.py

평가 대상은 운영 후보와 같은 조건의 쌍 (무작위 표본 제외, 코사인 0.5 이상: 후보 찾기가 코사인 0.5 미만을 빼므로). 라벨은 AI 라벨러 둘 중 하나라도 useful 이라 한 쌍.
쓸모 있는 쌍이 30개뿐이라 구간이 넓다. A 노트가 같은 쌍이 학습과 시험에 나뉘지 않게 나눈다 (B 노트는 겹칠 수 있다).
"""

import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.proposals.relation_features import BASE_FEATURES, CACHE, RELATION_FEATURES, auc  # noqa: E402

COUNTS = {"rel_in_a", "rel_in_b", "n_concepts_a", "n_concepts_b", "shared_nb", "shared_nb_specific", "same_parent", "chain2", "nb_links", "contrast_nb"}
SETS = {
    "cos 만 (학습)": ["cos"],
    "기본 (cos·낱말 겹침·길이)": BASE_FEATURES,
    "관계만": RELATION_FEATURES,
    "기본 + 관계": BASE_FEATURES + RELATION_FEATURES,
}


def components(meta):
    """묶음 = 앵커 노트(쌍의 A 쪽). 두 노트 중 하나라도 겹치는 쌍을 한 묶음으로 하면 노트들이 허브로 이어져 묶음이 2개뿐이라,
    A 노트만 같은 쌍을 같은 쪽에 둔다. B 노트는 겹칠 수 있다 (시험 쌍의 B 노트가 학습 쌍에도 나올 수 있다는 약한 누수)."""
    return [m["note_a"] for m in meta]


def design(X, names, mask):
    cols = [np.log1p(X[k][mask]) if k in COUNTS else X[k][mask] for k in names]
    return np.column_stack(cols)


def cv(Xm, y, groups, repeats=40, folds=5, C=0.3, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(seed)
    gs = sorted(set(groups))
    aucs, topk = [], []
    pooled = np.zeros(len(y))
    for _ in range(repeats):
        fold_of = dict(zip(gs, rng.permutation(len(gs)) % folds))
        f = np.array([fold_of[g] for g in groups])
        pred = np.zeros(len(y))
        for k in range(folds):
            tr, te = f != k, f == k
            if y[tr].sum() == 0 or y[te].sum() == 0 and False:
                continue
            sc = StandardScaler().fit(Xm[tr])
            m = LogisticRegression(C=C, class_weight="balanced", max_iter=1000).fit(sc.transform(Xm[tr]), y[tr])
            pred[te] = m.decision_function(sc.transform(Xm[te]))
        aucs.append(auc(y, pred))
        top = np.argsort(-pred, kind="stable")[:30]
        topk.append(int(y[top].sum()))
        pooled += pred
    return np.array(aucs), np.array(topk), pooled / repeats


def main():
    d = pickle.load(open(CACHE / "features.pkl", "rb"))
    X, y, meta = d["X"], d["y"], d["meta"]
    mask = np.array([m["arm"] != "random" for m in meta]) & (X["cos"] >= 0.5)
    y_m = y[mask]
    groups = [g for g, k in zip(components(meta), mask) if k]
    print(f"평가 쌍 {mask.sum()}개 · 쓸모 있음 {int(y_m.sum())}개 ({y_m.mean():.0%}) · 노트 묶음 {len(set(groups))}개")
    print(f"무작위로 상위 30개를 고르면 쓸모 있는 것 평균 {30 * y_m.mean():.1f}개\n")
    print("학습 없이 한 특징으로 그대로 줄 세우기 (동점은 입력 순서):")
    for k in ("cos", "token_jaccard", "direct", "shared_nb", "nb_links"):
        sc = X[k][mask]
        top = np.argsort(-sc, kind="stable")[:30]
        print(f"  {k:20s} AUC {auc(y_m, sc):.3f} · 상위 30 중 쓸모 있음 {int(y_m[top].sum())}")
    print()
    print(f"{'특징 묶음':28s} {'AUC 평균':>9s} {'(2.5~97.5%)':>14s}  상위 30 중 쓸모 있음 (평균)")
    for name, cols in SETS.items():
        a, t, _ = cv(design(X, cols, mask), y_m, groups)
        print(f"{name:28s} {a.mean():9.3f} [{np.percentile(a, 2.5):.2f}, {np.percentile(a, 97.5):.2f}]  {t.mean():.1f}")
    # 계수: 관계만·기본+관계를 전부로 맞춘 모델
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    cols = BASE_FEATURES + RELATION_FEATURES
    Xm = design(X, cols, mask)
    sc = StandardScaler().fit(Xm)
    m = LogisticRegression(C=0.3, class_weight="balanced", max_iter=1000).fit(sc.transform(Xm), y_m)
    print("\n계수 (표준화, 전체로 맞춤):")
    for k, c in sorted(zip(cols, m.coef_[0]), key=lambda kc: -abs(kc[1])):
        print(f"  {k:20s} {c:+.2f}")


if __name__ == "__main__":
    main()
