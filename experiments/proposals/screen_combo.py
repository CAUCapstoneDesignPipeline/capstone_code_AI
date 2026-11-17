"""Haiku 선별(운영 그대로)에 관계 점수를 보태면 쓸모 있는 쌍을 더 잘 찾는가 (모델 호출 없음, 학습 없음).

  uv run python experiments/proposals/screen_combo.py

선별(pairscreen)은 쓸모 있는 쌍의 40~50% 만 찾는다. 놓친 쌍을 관계 점수가 높은 것으로 되살리면(합집합) 재현율이 오르는지,
둘 다 높은 것만 남기면(교집합) 정밀도가 오르는지 본다. 관계 점수는 direct · nb_links · shared_nb 의 순위 평균이고 정한 기준은 임계값 하나뿐이라 (학습한 가중치 없음)
라벨에 맞춰 고른 값이 아니다 — 대신 임계값마다 모두 보인다. 평가 쌍은 운영 후보와 같은 조건 (무작위 표본 제외, cos 0.5 이상), 라벨은 AI 라벨러 둘 중 하나라도 useful.
"""

import csv
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.proposals.relation_features import CACHE  # noqa: E402

GOLD = ROOT / "data/private/gold"
SIGNALS = ("direct", "nb_links", "shared_nb")


def main():
    f = pickle.load(open(CACHE / "features.pkl", "rb"))
    scr = {r["id"]: r["label"] for r in csv.DictReader(open(GOLD / "pairing-screen-claude-haiku-5-5.csv", encoding="utf-8"))}
    meta, X, y = f["meta"], f["X"], f["y"]
    keep = np.array([m["arm"] != "random" for m in meta]) & (X["cos"] >= 0.5) & np.array([m["id"] in scr for m in meta])
    n = int(keep.sum())
    yk = y[keep].astype(bool)
    s = np.array([scr[m["id"]] == "useful" for m, k in zip(meta, keep) if k])
    rel = np.mean([rankdata(X[k][keep]) for k in SIGNALS], axis=0) / n          # 0~1
    print(f"평가 쌍 {n}개 · 쓸모 있음(AI 라벨러) {yk.sum()}개 ({yk.mean():.0%})\n")
    pr = lambda sel: (f"{sel.sum():4d}개 · 정밀도 {(sel & yk).sum() / max(1, sel.sum()):4.0%} · 재현율 {(sel & yk).sum() / yk.sum():4.0%}")
    print(f"{'선별만 (운영)':26s} {pr(s)}")
    print(f"{'관계 점수만 (위 33%)':26s} {pr(rel >= 1 - 0.33)}")
    print("\n합집합: 선별이 놓친 쌍 중 관계 점수가 임계값 이상이면 되살린다")
    for t in (0.9, 0.8, 0.7, 0.6):
        print(f"  임계 {t:.1f}: {pr(s | (rel >= t))}")
    print("\n교집합: 선별이 useful 이라 한 쌍 중 관계 점수가 임계값 이상인 것만")
    for t in (0.2, 0.33, 0.5):
        print(f"  임계 {t:.2f}: {pr(s & (rel >= t))}")
    # 선별이 내놓은 쌍의 순서만 바꾸는 쓰임: 선별 useful 쌍을 관계 점수로 줄 세워 위 k 개
    print("\n선별이 useful 이라 한 쌍을 관계 점수로 줄 세울 때 위 k 개의 정밀도 (카드를 만들 수 있는 수가 적을 때: 노트당 3개)")
    idx = np.where(s)[0]
    order = idx[np.argsort(-rel[idx], kind="stable")]
    for k in (5, 10, 15, 20):
        print(f"  위 {k:2d}: 관계 순 {yk[order[:k]].mean():4.0%} · 입력 순(아무 순서) {yk[idx[:k]].mean():4.0%} · cos 순 {yk[idx[np.argsort(-X['cos'][keep][idx], kind='stable')[:k]]].mean():4.0%}")


def by_labeler():
    """선별이 useful 이라 한 쌍을 관계 점수·cos 로 줄 세울 때 위 k 개의 적중, 라벨러마다 (사람 표 150쌍 안에서)."""
    from experiments.proposals.cut_eval import SIGNALS as _S, labelers
    f, idx, lab = labelers()
    scr = {r["id"]: r["label"] for r in csv.DictReader(open(GOLD / "pairing-screen-claude-haiku-5-5.csv", encoding="utf-8"))}
    X = {k: v[idx] for k, v in f["X"].items()}
    meta = [f["meta"][i] for i in idx]
    keep = np.array([m["arm"] != "random" for m in meta]) & (X["cos"] >= 0.5)
    s = np.array([scr[m["id"]] == "useful" for m in meta]) & keep
    n = int(keep.sum())
    rel = np.zeros(len(idx))
    rel[keep] = np.mean([rankdata(X[k][keep]) for k in _S], axis=0) / n
    print(f"\n라벨러마다 — 선별 useful {int(s.sum())}개를 줄 세워 위 k 개 중 쓸모 있는 수 (관계 / cos)")
    sel = np.where(s)[0]
    by_rel = sel[np.argsort(-rel[sel], kind="stable")]
    by_cos = sel[np.argsort(-X["cos"][sel], kind="stable")]
    for name, y in lab.items():
        print(f"  {name:10s} 선별 전체 {y[sel].sum():2d}/{len(sel)} · " + " · ".join(f"위 {k}: {y[by_rel[:k]].sum()}/{y[by_cos[:k]].sum()}" for k in (5, 10, 15)))


if __name__ == "__main__":
    main()
    by_labeler()
