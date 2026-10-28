"""④ 노트 쌍 점수의 오프라인 시험 (설계 concept-pipeline.md 4장 ④, 7장 평가).

정답 노트 쌍 (a, b) 마다 a 로 물었을 때 b 의 순위, b 로 물었을 때 a 의 순위를 잰다 (쌍 하나에 질의 두 개).
  S        Adamic-Adar 연결 강도 (related_notes 그대로)
  S·N^α    새로움 N(a, b) = 1 − cos(μ_topic(a), μ_topic(b)) 를 곱한다. μ_topic 은 주제의 노트 개념 가중치 벡터 평균
  TF-IDF   기준선: 두 노트 개념 가중치 벡터의 코사인
설계대로 연결 후보는 다른 주제의 노트만이다 (cross_topic). 같은 주제 노트는 N = 0 이라 S·N^α 에서도 빠진다.
동점은 소수 넷째 자리로 반올림한 뒤 노트 id 순 (설계 4장 ④ 7).
"""

import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass

from .corpus import Corpus
from .linker import related_notes


@dataclass(frozen=True)
class Ranking:
    method: str
    ranks: dict[tuple[str, str], int | None]     # (질의 노트, 정답 노트) → 1부터의 순위, 못 찾으면 None

    @property
    def short(self) -> str:
        return {"S (전체)": "S", "S (다른 주제)": "S′", "TF-IDF 코사인 (다른 주제)": "TF-IDF"}.get(self.method, self.method)

    def recall_at(self, k: int) -> float:
        return sum(1 for r in self.ranks.values() if r is not None and r <= k) / len(self.ranks)

    @property
    def mrr(self) -> float:
        return sum(1 / r for r in self.ranks.values() if r is not None) / len(self.ranks)


def _cos(u: dict[str, float], v: dict[str, float]) -> float:
    dot = sum(w * v[k] for k, w in u.items() if k in v)
    nu, nv = math.sqrt(sum(w * w for w in u.values())), math.sqrt(sum(w * w for w in v.values()))
    return dot / (nu * nv) if nu and nv else 0.0


def topic_centroids(corpus: Corpus, topics: dict[str, str]) -> dict[str, dict[str, float]]:
    members = defaultdict(list)
    for n in sorted(corpus.notes):
        members[topics[n]].append(corpus.notes[n].weights)
    out = {}
    for t, vecs in members.items():
        acc = defaultdict(float)
        for v in vecs:
            for k, w in v.items():
                acc[k] += w / len(vecs)
        out[t] = dict(acc)
    return out


def novelty(corpus: Corpus, topics: dict[str, str]) -> dict[tuple[str, str], float]:
    mu = topic_centroids(corpus, topics)
    names = sorted(mu)
    return {(a, b): round(1 - _cos(mu[a], mu[b]), 4) if a != b else 0.0 for a in names for b in names}


def _rank(scores: dict[str, float], target: str) -> int | None:
    order = sorted(((round(s, 4), n) for n, s in scores.items() if s > 0), key=lambda x: (-x[0], x[1]))
    for i, (_, n) in enumerate(order, 1):
        if n == target:
            return i
    return None


def evaluate(corpus: Corpus, topics: dict[str, str], pairs: list[tuple[str, str]],
             alphas: tuple[float, ...] = (0.5, 1.0, 2.0)) -> list[Ranking]:
    """정답 쌍마다 양방향 질의의 순위. 방법: S(전체), S(다른 주제만), S·N^α, TF-IDF 코사인(다른 주제만)."""
    nov = novelty(corpus, topics)
    queries = sorted({(a, b) for a, b in pairs} | {(b, a) for a, b in pairs})
    s_all = {q: {l.note_b: l.score for l in related_notes(corpus, q, k=len(corpus.notes))} for q, _ in queries}

    def cross(q, scores):
        return {n: s for n, s in scores.items() if topics[n] != topics[q]}

    methods: dict[str, dict] = {"S (전체)": {}, "S (다른 주제)": {}, "TF-IDF 코사인 (다른 주제)": {}}
    methods.update({f"S·N^{a:g}": {} for a in alphas})
    for q, t in queries:
        s = s_all[q]
        methods["S (전체)"][(q, t)] = _rank(s, t)
        methods["S (다른 주제)"][(q, t)] = _rank(cross(q, s), t)
        tfidf = {n: _cos(corpus.notes[q].weights, corpus.notes[n].weights) for n in corpus.notes if n != q}
        methods["TF-IDF 코사인 (다른 주제)"][(q, t)] = _rank(cross(q, tfidf), t)
        for a in alphas:
            methods[f"S·N^{a:g}"][(q, t)] = _rank(
                {n: v * nov[(topics[q], topics[n])] ** a for n, v in cross(q, s).items()}, t)
    return [Ranking(m, r) for m, r in methods.items()]


# ---- 숨겼다 다시 찾기 (설계 7장): 정답 표시 없이 하는 결정적 평가 ----

@dataclass(frozen=True)
class Refind:
    method: str
    ranks: dict[tuple[str, str, str], int | None]    # (숨긴 노트, A, C) → C 의 순위, 못 찾으면 None

    def recall_at(self, k: int) -> float:
        return sum(1 for r in self.ranks.values() if r is not None and r <= k) / len(self.ranks) if self.ranks else 0.0

    @property
    def mrr(self) -> float:
        return sum(1 / r for r in self.ranks.values() if r is not None) / len(self.ranks) if self.ranks else 0.0


def _sentence_pairs(corpus: Corpus) -> dict[str, set[tuple[str, str]]]:
    """노트 → 한 문장 안에 함께 나온 개념 쌍 (정렬된 id 쌍)."""
    from app.services.relations.sentences import sentence_spans
    out = {}
    for n in sorted(corpus.notes):
        note = corpus.notes[n]
        pairs = set()
        for s, e in sentence_spans(note.text):
            inside = sorted(c for c, spans in note.spans.items() if any(s <= a and b <= e for a, b in spans))
            pairs |= {(x, y) for i, x in enumerate(inside) for y in inside[i + 1:]}
        out[n] = pairs
    return out


def hide_and_refind(corpus: Corpus, n: int = 200, seed: int = 0, hub_ratio: float = 0.15) -> list[Refind]:
    """한 노트의 한 문장에 함께 쓰인 개념 쌍 (A, C) 를 고르고, 그 노트를 통째로 숨긴 뒤 다른 노트들의 문장 동시 출현
    그래프에서 A 로 물었을 때 C 의 순위를 잰다. 고르는 쌍: A·C 가 각각 다른 노트에도 나오고, 다른 노트 어디에서도
    한 문장에 함께 쓰이지 않은 쌍 (다른 노트에 이미 적혀 있으면 새 연결이 아니다). 쌍은 seed 로 고정한 hash 순서로 n 개.

      A-B-C (Adamic-Adar)  Σ_B 1/log(1+df(B)), B 는 A·C 와 각각 한 문장에 함께 나온 개념 (허브 B 는 뺀다)
      인기도 기준선          C 가 나오는 노트 수 (df)
    A 와 이미 한 문장에 함께 나온 개념은 후보에서 뺀다 (이미 적혀 있는 연결)."""
    co = _sentence_pairs(corpus)
    hub = max(2, int(hub_ratio * len(corpus.notes)))
    where = defaultdict(set)                      # 개념 → 나오는 노트
    for nid, note in corpus.notes.items():
        for c in note.spans:
            where[c].add(nid)

    def neighbours(exclude: str) -> dict[str, set[str]]:
        g = defaultdict(set)
        for nid, pairs in co.items():
            if nid != exclude:
                for x, y in pairs:
                    g[x].add(y)
                    g[y].add(x)
        return g

    picks = []
    for nid in sorted(co):
        for a, c in sorted(co[nid]):
            if len(where[a] - {nid}) and len(where[c] - {nid}):
                picks.append((nid, a, c))
    picks.sort(key=lambda p: hashlib.sha256(f"{seed}:{p}".encode()).hexdigest())

    graphs, chosen = {}, []
    for nid, a, c in picks:
        g = graphs.setdefault(nid, neighbours(nid))
        if c in g[a]:
            continue                               # 다른 노트에 이미 함께 쓰여 있다
        chosen.append((nid, a, c))
        if len(chosen) >= n:
            break

    aa, pop = {}, {}
    for nid, a, c in chosen:
        g = graphs[nid]
        df = {x: len(where[x] - {nid}) for x in where}
        cands = {x for x in where if x != a and x not in g[a] and df[x] > 0}
        score = {}
        for b in g[a]:
            if df[b] > hub:
                continue
            for x in g[b]:
                if x in cands:
                    score[x] = score.get(x, 0.0) + 1 / math.log(1 + df[b])
        aa[(nid, a, c)] = _rank(score, c)
        pop[(nid, a, c)] = _rank({x: float(df[x]) for x in cands}, c)
    return [Refind("A-B-C (Adamic-Adar)", aa), Refind("인기도 기준선", pop)]
