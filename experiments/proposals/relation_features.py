"""후보 쌍의 순위를 관계로 매길 수 있는가: 관계 그래프 특징과 라벨 쌍 275개에서의 시험 (프로토타입).

  uv run --env-file .env python experiments/proposals/labeled_pairs.py      # 라벨 쌍 ↔ 요소
  uv run --env-file .env python experiments/proposals/note_relations.py     # 노트마다 개념·관계 (OpenAI 를 부른다, 캐시)
  uv run --env-file .env python experiments/proposals/relation_features.py  # 이 파일: 특징을 만들고 AUC 를 잰다 (모델 호출 없음)

그래프: 노트마다 출고한 경로(학생 개념 + pairs-1)가 낸 관계를 개념 키로 합친다. 같은 키의 개념은 노트가 달라도 한 노드다.
쌍 (A, B) 의 두 요소는 각자 자기 문장 안에 언급이 든 개념 집합을 갖는다. 특징은 두 집합이 관계 그래프에서 어떻게 이어지는가다:
직접 선, 공통 이웃, 방향 있는 두 걸음 사슬, 두 이웃 사이의 선, 같은 상위 개념(is_a·part_of), 문장 자체가 관계를 담았는가.
라벨은 AI 라벨러 둘이 눈가림으로 붙인 것이고 쓸모 있음은 둘 중 하나라도 useful 이라 한 쌍이다 (30개). 사람 라벨이 생기면 같은 코드로 다시 잰다.
"""

import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
CACHE = ROOT / "data/private/cache/experiments/proposals"
from app.services.links.relation_rank import CHAIN, PARENT, SPECIFIC, Graph, pair_features  # noqa: E402,F401  (운영 코드로 옮겼다)

RELATION_FEATURES = ["rel_in_a", "rel_in_b", "n_concepts_a", "n_concepts_b", "direct", "shared_nb", "shared_nb_specific", "same_parent",
                     "chain2", "nb_links", "contrast_nb", "log_deg_a", "log_deg_b"]
BASE_FEATURES = ["cos", "log_len_a", "log_len_b", "token_jaccard"]


TOKEN = None


def base_features(p, ea, eb) -> dict:
    import re
    tok = lambda t: set(re.findall(r"[a-z]{3,}|[가-힣]{2,}", t.lower()))
    ta, tb = tok(ea.text), tok(eb.text)
    return {"cos": p["cos"] if p["cos"] is not None else 0.0, "log_len_a": float(np.log1p(len(ea.text))), "log_len_b": float(np.log1p(len(eb.text))),
            "token_jaccard": len(ta & tb) / max(1, len(ta | tb))}


def auc(y, s):
    """Mann-Whitney AUC (동점은 절반)."""
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum())) / (len(pos) * len(neg)))


def boot_ci(y, s, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    y, s = np.asarray(y), np.asarray(s)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() in (0, len(y)):
            continue
        vals.append(auc(y[i], s[i]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def load():
    import hashlib

    from experiments.proposals.note_relations import key
    d = pickle.load(open(CACHE / "labeled.pkl", "rb"))
    return d["pairs"]


def main():
    from experiments.proposals.labeled_pairs import build_index
    pairs = load()
    corpus, index = build_index()
    notes = {}
    from experiments.proposals.note_relations import key
    for name, note in corpus.notes.items():
        p = key(name, note.text)
        if p.exists():
            notes[name] = json.loads(p.read_text(encoding="utf-8"))
    g = Graph(notes)
    print(f"그래프: 노트 {len(notes)}개 · 노드 {len(g.adj)} · 관계 {sum(len(v) for v in g.rel_by_note.values())}")
    rows, y = [], []
    for p in pairs:
        ea, eb = index.el[p["a"]], index.el[p["b"]]
        rows.append({**base_features(p, ea, eb), **pair_features(g, ea, eb)})
        y.append(int("useful" in (p["label_a"], p["label_b"])))
    y = np.array(y)
    X = {k: np.array([r[k] for r in rows], dtype=float) for k in rows[0]}
    print(f"라벨 쌍 {len(y)}개 · 쓸모 있음 {int(y.sum())}개 ({y.mean():.1%})\n")
    print(f"{'특징':22s} {'0 이 아닌 쌍':>10s} {'쓸모 중':>7s} {'AUC':>6s} {'95% 구간':>14s}")
    for k in BASE_FEATURES + RELATION_FEATURES + ["_inter"]:
        s = X[k]
        lo, hi = boot_ci(y, s)
        print(f"{k:22s} {int((s != 0).sum()):10d} {int(((s != 0) & (y == 1)).sum()):7d} {auc(y, s):6.3f} [{lo:.2f}, {hi:.2f}]")
    meta = [{"id": p["id"], "arm": p["arm"], "note_a": index.el[p["a"]].note, "note_b": index.el[p["b"]].note} for p in pairs]
    pickle.dump({"X": X, "y": y, "meta": meta}, open(CACHE / "features.pkl", "wb"))


if __name__ == "__main__":
    main()
