"""④ 노트 쌍 점수의 오프라인 시험 (설계 concept-pipeline.md 4장 ④, 7장 평가).

정답 노트 쌍 (a, b) 마다 a 로 물었을 때 b 의 순위, b 로 물었을 때 a 의 순위를 잰다 (쌍 하나에 질의 두 개).
  S        Adamic-Adar 연결 강도 (related_notes 그대로)
  S·N^α    새로움 N(a, b) = 1 − cos(μ_topic(a), μ_topic(b)) 를 곱한다. μ_topic 은 주제의 노트 개념 가중치 벡터 평균
  TF-IDF   기준선: 두 노트 개념 가중치 벡터의 코사인
설계대로 연결 후보는 다른 주제의 노트만이다 (cross_topic). 같은 주제 노트는 N = 0 이라 S·N^α 에서도 빠진다.
동점은 소수 넷째 자리로 반올림한 뒤 노트 id 순 (설계 4장 ④ 7).
"""

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
