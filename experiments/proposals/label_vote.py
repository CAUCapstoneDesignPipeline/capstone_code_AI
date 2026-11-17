"""세 라벨러(AI 라벨러 둘의 합집합 · Sonnet · Gemini) 중 둘 이상이 useful 이라 한 쌍을 정답으로 순위 특징의 AUC 를 잰다 (모델 호출 없음).

  uv run python experiments/proposals/label_vote.py
"""

import csv
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.proposals.relation_features import CACHE, auc, boot_ci  # noqa: E402

GOLD = ROOT / "data/private/gold"


def main():
    rd = lambda f: {r["id"]: r for r in csv.DictReader(open(GOLD / f, encoding="utf-8"))}
    g, s, k = rd("pairing-human-gemini.csv"), rd("pairing-human-sonnet.csv"), rd("pairing-human-key.csv")
    f = pickle.load(open(CACHE / "features.pkl", "rb"))
    where = {m["id"]: i for i, m in enumerate(f["meta"])}
    ids = sorted(g, key=int)
    idx = np.array([where[k[i]["source_id"]] for i in ids])
    votes = np.array([["useful" in (k[i]["label_a"], k[i]["label_b"]), s[i]["label"] == "useful", g[i]["label"] == "useful"] for i in ids]).astype(int)
    y = (votes.sum(1) >= 2).astype(int)
    print(f"세 라벨러 중 둘 이상이 useful: {y.sum()}개 / {len(y)} · 셋 모두 {int((votes.sum(1) == 3).sum())}개")
    for name in ("cos", "token_jaccard", "direct", "shared_nb", "nb_links", "rel_in_b"):
        sc = f["X"][name][idx]
        lo, hi = boot_ci(y, sc, n=1000)
        print(f"  {name:14s} AUC {auc(y, sc):.3f} [{lo:.2f}, {hi:.2f}]")


if __name__ == "__main__":
    main()
