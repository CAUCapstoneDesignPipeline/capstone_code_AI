"""관계 신호로 후보를 LLM 선별 앞에서 잘라 내면 쓸모 있는 쌍을 얼마나 지키는가 (모델 호출 없음, 학습 없음).

  uv run python experiments/proposals/cut_eval.py

두 가지를 잰다. (1) 관계 신호가 하나라도 있는 쌍만 남기는 문 (direct · nb_links · shared_nb 중 하나가 0 보다 크다),
(2) 신호를 합친 점수(세 특징의 순위 평균, 동점은 cos)로 줄 세워 위 q% 만 남기기. 비교 기준은 cos 로 줄 세워 같은 비율을 남기기.
쓸모 있음의 기준은 라벨러마다 따로 본다 (AI 라벨러 · Sonnet · Gemini · 셋 중 둘 이상). 평가 쌍은 운영 후보와 같은 조건 (무작위 표본 제외, cos 0.5 이상) 이고
사람 표 150쌍 안에 든 쌍만 쓴다 (Sonnet·Gemini 라벨이 거기만 있다). 표본 크기가 작아 숫자마다 구간이 넓다.
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


def labelers():
    rd = lambda f: {r["id"]: r for r in csv.DictReader(open(GOLD / f, encoding="utf-8"))}
    g, s, k = rd("pairing-human-gemini.csv"), rd("pairing-human-sonnet.csv"), rd("pairing-human-key.csv")
    f = pickle.load(open(CACHE / "features.pkl", "rb"))
    where = {m["id"]: i for i, m in enumerate(f["meta"])}
    ids = sorted(g, key=int)
    idx = np.array([where[k[i]["source_id"]] for i in ids])
    ai = np.array(["useful" in (k[i]["label_a"], k[i]["label_b"]) for i in ids])
    so = np.array([s[i]["label"] == "useful" for i in ids])
    ge = np.array([g[i]["label"] == "useful" for i in ids])
    vote = (ai.astype(int) + so + ge) >= 2
    return f, idx, {"AI 라벨러": ai, "Sonnet": so, "Gemini": ge, "셋 중 둘": vote}


def main():
    f, idx, lab = labelers()
    X = {k: v[idx] for k, v in f["X"].items()}
    meta = [f["meta"][i] for i in idx]
    keep = np.array([m["arm"] != "random" for m in meta]) & (X["cos"] >= 0.5)
    X = {k: v[keep] for k, v in X.items()}
    lab = {k: v[keep] for k, v in lab.items()}
    n = int(keep.sum())
    gate = np.any([X[k] > 0 for k in SIGNALS], axis=0)
    score = np.mean([rankdata(X[k]) for k in SIGNALS], axis=0) + 1e-3 * rankdata(X["cos"]) / n
    print(f"평가 쌍 {n}개 (사람 표 150쌍 중 무작위 표본·cos 0.5 미만 제외)\n")
    print(f"(1) 문: 관계 신호가 있는 쌍만 남긴다 → {gate.sum()}개 ({gate.mean():.0%}) 남음")
    print(f"{'라벨러':10s} {'쓸모':>4s} {'남은 쓸모':>9s} {'지킨 비율':>8s} {'남긴 쌍의 적중률':>14s} {'(전체 적중률)':>12s}")
    for name, y in lab.items():
        print(f"{name:10s} {int(y.sum()):4d} {int((y & gate).sum()):9d} {(y & gate).sum() / max(1, y.sum()):8.0%} {(y & gate).sum() / max(1, gate.sum()):14.0%} {y.mean():12.0%}")
    print("\n(2) 줄 세워 위 q% 만 남긴다 — 지킨 쓸모 비율 (관계 점수 / cos)")
    qs = (0.25, 0.33, 0.5, 0.67)
    print(f"{'라벨러':10s} " + " ".join(f"{int(q * 100):>3d}%:{'관계':>5s}/{'cos':<4s}" for q in qs))
    for name, y in lab.items():
        cells = []
        for q in qs:
            m = max(1, round(q * n))
            r = y[np.argsort(-score, kind="stable")[:m]].sum() / max(1, y.sum())
            c = y[np.argsort(-X["cos"], kind="stable")[:m]].sum() / max(1, y.sum())
            cells.append(f"{r:9.0%}/{c:<4.0%}")
        print(f"{name:10s} " + " ".join(cells))


if __name__ == "__main__":
    main()
