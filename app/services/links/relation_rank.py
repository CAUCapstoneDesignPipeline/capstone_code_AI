"""후보 쌍의 순위를 관계 그래프로 매긴다 (LLM 호출 없음).

노트마다 출고한 경로(학생 개념 + pairs-1)가 낸 개념·관계를 개념 키로 합쳐 한 그래프로 만들고, 쌍 (A, B) 의 두 요소가 그 안에서 어떻게 이어지는지를 본다:
직접 선, 공통 이웃, 방향 있는 두 걸음 사슬, 두 이웃 사이의 선, 같은 상위 개념, 문장 자체가 관계를 담았는가.
후보 순위(코사인·채널 수)는 쓸모를 거의 예측하지 못했고 이 신호는 더 잘 줄 세웠다 (docs/eval/proposals.md). 학습한 가중치가 없다:
`relation_scores` 는 direct · nb_links · shared_nb 세 신호를 후보 목록 안의 순위로 바꿔 평균한 0~1 이다.
"""

from collections import defaultdict

import numpy as np

SPECIFIC = {"is_a", "part_of", "causes", "requires", "used_for", "example_of"}
PARENT = {"is_a", "part_of"}                      # source 가 target 의 아래에 든다
CHAIN = {("causes", "causes"), ("requires", "requires"), ("part_of", "part_of"), ("is_a", "is_a"), ("is_a", "part_of"), ("part_of", "is_a"),
         ("used_for", "used_for"), ("causes", "requires"), ("requires", "causes"), ("used_for", "requires"), ("requires", "used_for")}


class Graph:
    """모든 노트의 관계를 개념 키로 합친 그래프."""

    def __init__(self, notes: dict[str, dict]):
        self.notes = notes
        self.out = defaultdict(list)          # x → [(y, code, 확신도)]
        self.inn = defaultdict(list)
        self.adj = defaultdict(dict)          # x → {y: 가장 센 확신도}
        self.rel_by_note = defaultdict(list)
        for name, d in notes.items():
            for r in d["relations"]:
                s, t, c, w = r["source"], r["target"], r["code"], r["confidence"]
                if s == t:
                    continue
                self.out[s].append((t, c, w))
                self.inn[t].append((s, c, w))
                for x, y in ((s, t), (t, s)):
                    self.adj[x][y] = max(self.adj[x].get(y, 0.0), w)
                self.rel_by_note[name].append(r)

    def concepts_in(self, note: str, start: int, end: int) -> set[str]:
        d = self.notes.get(note)
        if not d:
            return set()
        return {k for k, c in d["concepts"].items() if any(start <= ms < end for ms, _ in c["mentions"])}

    def relations_in(self, note: str, start: int, end: int) -> int:
        """근거가 이 문장 구간 안에 있는 관계의 수."""
        return sum(1 for r in self.rel_by_note.get(note, ()) if any(start <= s < end for s, _ in r["evidence"]))


def pair_features(g: Graph, ea, eb) -> dict:
    """ea, eb: 요소 (note, start, end, text). 관계 특징."""
    ca, cb = g.concepts_in(ea.note, ea.start, ea.end), g.concepts_in(eb.note, eb.start, eb.end)
    inter = ca & cb
    na = {z: max(g.adj[x].get(z, 0.0) for x in ca) for x in ca for z in g.adj[x]} if ca else {}
    nb = {z: max(g.adj[y].get(z, 0.0) for y in cb) for y in cb for z in g.adj[y]} if cb else {}
    for z in list(na):
        if z in ca or z in cb:
            na.pop(z)
    for z in list(nb):
        if z in ca or z in cb:
            nb.pop(z)
    shared = set(na) & set(nb)
    direct = max([g.adj[x].get(y, 0.0) for x in ca for y in cb if x != y] or [0.0])
    # 두 요소 모두 이 노드를 이웃으로 둔 유형: 같은 상위 개념 (둘 다 z 의 아래)
    def parents(cs):
        return {z for x in cs for z, c, _ in g.out[x] if c in PARENT}
    same_parent = len(parents(ca) & parents(cb) - ca - cb)
    spec = set()
    for x in ca | cb:
        for z, c, w in g.out[x] + g.inn[x]:
            if c in SPECIFIC and z in shared:
                spec.add(z)
    chain = 0                                  # x →t1→ z →t2→ y 또는 반대
    for xs, ys in ((ca, cb), (cb, ca)):
        for x in xs:
            for z, c1, _ in g.out[x]:
                for y, c2, _ in g.out[z]:
                    if y in ys and (c1, c2) in CHAIN:
                        chain += 1
    links = sum(min(na[u], nb[v]) for u in na for v in nb if u != v and v in g.adj[u])        # 두 이웃 집합 사이의 선
    contrast = 0
    for x in ca | cb:
        for z, c, _ in g.out[x] + g.inn[x]:
            if c in ("contrasts_with", "analogous_to") and (z in (ca | cb) or z in na and z in nb):
                contrast += 1
    deg = lambda cs: float(np.log1p(sum(len(g.adj[x]) for x in cs)))
    return {"rel_in_a": g.relations_in(ea.note, ea.start, ea.end), "rel_in_b": g.relations_in(eb.note, eb.start, eb.end),
            "n_concepts_a": len(ca), "n_concepts_b": len(cb), "direct": direct,
            "shared_nb": float(sum(min(na[z], nb[z]) for z in shared)), "shared_nb_specific": float(len(spec)),
            "same_parent": same_parent, "chain2": chain, "nb_links": float(links), "contrast_nb": contrast,
            "log_deg_a": deg(ca), "log_deg_b": deg(cb), "_inter": len(inter)}


SCORE_SIGNALS = ("direct", "nb_links", "shared_nb")


def _avg_rank(v: np.ndarray) -> np.ndarray:
    """1..n 순위, 동점은 평균 순위."""
    order = np.argsort(v, kind="stable")
    ranks = np.empty(len(v))
    ranks[order] = np.arange(1, len(v) + 1)
    for val in np.unique(v):
        same = v == val
        if same.sum() > 1:
            ranks[same] = ranks[same].mean()
    return ranks


def relation_scores(g: Graph, pairs: list[tuple]) -> list[dict | None]:
    """(A 요소, B 요소) 목록 → 쌍마다 {"score": 0~1, "direct", "nb_links", "shared_nb"}. 순위는 이 목록 안에서만 뜻이 있다 (목록이 다르면 점수도 다르다).
    받아들인 새 지식(generated) 요소가 끼거나 관계를 아직 뽑지 않은 노트의 요소면 None."""
    def scorable(a, b) -> bool:
        return "generated" not in (a.origin, b.origin) and a.note in g.notes and b.note in g.notes      # 관계를 뽑아 둔 노트의 요소끼리만
    rows = [pair_features(g, a, b) if scorable(a, b) else None for a, b in pairs]
    live = [i for i, r in enumerate(rows) if r is not None]
    out: list[dict | None] = [None] * len(pairs)
    if not live:
        return out
    ranks = np.mean([_avg_rank(np.array([rows[i][k] for i in live], dtype=float)) for k in SCORE_SIGNALS], axis=0) / len(live)
    for n, i in enumerate(live):
        out[i] = {"score": round(float(ranks[n]), 3), **{k: round(float(rows[i][k]), 3) for k in SCORE_SIGNALS}}
    return out
