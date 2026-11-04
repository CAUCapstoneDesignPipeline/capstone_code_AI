"""추정 연결: 문장·제목·문단 어디에서도 같이 나온 적 없는 키워드 쌍을 그래프 구조와 의미로 잇는다 (링크 예측, 결정적, LLM 없음).

점수 = z(log(1 + 1000 · Katz)) + z(의미 유사도)
  Katz  = β·A² + β²·A³ (β = 0.05): 두 키워드 사이의 길이 2·3 걸음 수. 이미 이어진 키워드가 이어 주는 정도 (증거 연결 전체가 그래프 A)
  의미   = 키워드 벡터(그 요소 임베딩의 평균) 코사인. 임베딩이 없으면 구조 점수만 쓴다 (점수는 두 배로 놓아 같은 문턱을 쓴다)
  z     = 키워드 쌍 전체에서의 평균·표준편차로 바꾼 값 (그래프마다 표본으로 한 번 구한다)
실제 키워드 그래프(키워드 5,198 · 연결 18,050)에서 이 점수가 가린 연결을 되찾는 정도는 AUC 0.992 (문장·제목만이면 0.980) 였다.
Katz 만 쓰면 0.978 (0.939), 의미만 쓰면 0.886 (0.907). 라벨 275쌍에서 쓸모 있는 쌍을 올리는 정도는 AUC 0.729 로 참고 수준이다 (쓸모 있는 쌍 30개).
이미 이어진 쌍은 쓸모가 오히려 낮았다 (6% 대 11%) — 안 이어진 쌍을 잇는 이유다. 쓸모 있는 쌍을 가리는 데는 이 점수도 약하므로 증거가 아니라 **추정** 으로만 보인다.

비용: 모든 쌍(N²)을 계산하지 않는다. 키워드 하나에 대해 이웃 목록만 따라가는 두 걸음(희소 벡터 × 이웃 목록)과 의미 벡터 한 번 곱(N·d)으로 그 키워드의 후보를 구하고,
화면에 올린 키워드에 대해서만 부른다. z 표본(150개 키워드)은 그래프마다 한 번.
"""

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

BETA = 0.05
LOG_SCALE = 1000.0
SEMANTIC_NEIGHBORS = 10          # 구조로는 닿지 않아도 의미가 가까운 키워드를 이만큼 후보에 더한다
STAT_SAMPLES = 150
MIN_SCORE = 3.0                  # 이 점수 미만은 잇지 않는다 (구조·의미 둘 다 상위 1~2% 쯤)
TOP = 5                          # 키워드마다 잇는 추정 연결 수


@dataclass(frozen=True)
class Inferred:
    a: str
    b: str
    score: float                 # z 합
    cos: float | None            # 의미 유사도 (임베딩이 없으면 None)
    katz: float
    via: tuple[str, ...] = field(default=())     # 둘 다 이어진 키워드 (이웃이 적은 것부터 최대 3개)

    @property
    def weight(self) -> float:
        """화면에서 선 굵기에 쓰는 0.1~0.5 (증거 점수 0.3~1 보다 약하게)."""
        return round(0.1 + 0.4 * min(1.0, max(0.0, (self.score - MIN_SCORE) / 4.0)), 4)


class LinkGraph:
    def __init__(self, ids: list[str], pairs, vectors: np.ndarray | None = None):
        """ids: 키워드 id. pairs: 증거로 이어진 (id, id) 쌍 (모든 증거 수준). vectors: ids 순서의 정규화된 키워드 벡터 또는 None."""
        self.ids, self.idx = ids, {k: i for i, k in enumerate(ids)}
        n = len(ids)
        nbrs: list[set[int]] = [set() for _ in range(n)]
        for a, b in pairs:
            if a in self.idx and b in self.idx and a != b:
                nbrs[self.idx[a]].add(self.idx[b])
                nbrs[self.idx[b]].add(self.idx[a])
        self.deg = np.array([len(x) for x in nbrs])
        self.indptr = np.concatenate([[0], np.cumsum(self.deg)])
        self.indices = np.array([j for x in nbrs for j in sorted(x)], dtype=np.int64)
        self.vec = vectors
        self._stats: tuple[float, float, float, float] | None = None

    def neighbors(self, i: int) -> np.ndarray:
        return self.indices[self.indptr[i]:self.indptr[i + 1]]

    def step(self, x: np.ndarray) -> np.ndarray:
        """y = A · x (A 는 0/1 이웃 행렬). x 에서 0 이 아닌 곳의 이웃만 따라간다."""
        nz = np.flatnonzero(x)
        y = np.zeros(len(self.ids))
        if not len(nz):
            return y
        lens = self.deg[nz]
        total = int(lens.sum())
        if total:
            offsets = np.repeat(self.indptr[nz] - (np.cumsum(lens) - lens), lens)
            cols = self.indices[np.arange(total) + offsets]
            y = np.bincount(cols, weights=np.repeat(x[nz], lens), minlength=len(self.ids)).astype(float)
        return y

    def katz(self, i: int) -> np.ndarray:
        a1 = np.zeros(len(self.ids))
        a1[self.neighbors(i)] = 1.0
        a2 = self.step(a1)
        return BETA * a2 + BETA * BETA * self.step(a2)

    def _lk(self, katz: np.ndarray) -> np.ndarray:
        return np.log1p(LOG_SCALE * katz)

    def stats(self) -> tuple[float, float, float, float]:
        """(log Katz 평균, 표준편차, 의미 평균, 표준편차): 키워드 150개를 표본으로 모든 쌍에서 구한다 (그래프마다 한 번)."""
        if self._stats is None:
            n = len(self.ids)
            rng = np.random.default_rng(0)
            sample = rng.choice(n, min(STAT_SAMPLES, n), replace=False)
            lk = np.concatenate([self._lk(self.katz(int(i))) for i in sample])
            sem = np.concatenate([self.vec @ self.vec[int(i)] for i in sample]) if self.vec is not None else np.zeros(1)
            self._stats = (float(lk.mean()), float(lk.std()) or 1.0, float(sem.mean()), float(sem.std()) or 1.0)
        return self._stats

    def _scores(self, i: int):
        """키워드 i 와 모든 키워드의 (점수, katz, cos). 이미 이어진 쌍과 자기 자신은 -inf."""
        mk, sk, ms, ss = self.stats()
        katz = self.katz(i)
        z = (self._lk(katz) - mk) / sk
        cos = None
        if self.vec is not None:
            cos = self.vec @ self.vec[i]
            z = z + (cos - ms) / ss
        else:
            z = 2 * z
        z[i] = -np.inf
        z[self.neighbors(i)] = -np.inf
        return z, katz, cos

    def _via(self, i: int, j: int) -> tuple[str, ...]:
        common = np.intersect1d(self.neighbors(i), self.neighbors(j))
        return tuple(self.ids[int(c)] for c in common[np.argsort(self.deg[common], kind="stable")][:3])

    def infer(self, kid: str, top: int = TOP, min_score: float = MIN_SCORE) -> list[Inferred]:
        """kid 와 증거로는 안 이어졌지만 이어질 만한 키워드 (점수 순, 최대 top 개). 후보 = 걸어서 닿는 키워드 + 의미가 가까운 키워드."""
        i = self.idx.get(kid)
        if i is None:
            return []
        z, katz, cos = self._scores(i)
        cand = set(np.flatnonzero(katz > 0).tolist())
        if cos is not None:
            cand |= set(np.argsort(-np.where(np.isfinite(z), cos, -np.inf))[:SEMANTIC_NEIGHBORS].tolist())
        order = sorted((j for j in cand if np.isfinite(z[j]) and z[j] >= min_score), key=lambda j: (-z[j], j))[:top]
        return [Inferred(kid, self.ids[j], round(float(z[j]), 3), None if cos is None else round(float(cos[j]), 3),
                         round(float(katz[j]), 6), self._via(i, j)) for j in order]

    def score(self, a: str, b: str) -> Inferred | None:
        """아무 두 키워드의 추정 점수 (문턱과 상관없이). 이미 증거로 이어졌거나 같은 키워드면 None."""
        i, j = self.idx.get(a), self.idx.get(b)
        if i is None or j is None or i == j or j in set(self.neighbors(i).tolist()):
            return None
        z, katz, cos = self._scores(i)
        return Inferred(a, b, round(float(z[j]), 3), None if cos is None else round(float(cos[j]), 3), round(float(katz[j]), 6), self._via(i, j))


def keyword_vectors(element_keyword_ids: list[str], element_vectors: np.ndarray, ids: list[str]) -> np.ndarray:
    """키워드 벡터 = 그 요소 임베딩의 평균 (정규화). ids 순서. 요소가 없는 키워드는 0 벡터."""
    row = {k: i for i, k in enumerate(ids)}
    out = np.zeros((len(ids), element_vectors.shape[1]), dtype=np.float32)
    cnt = Counter()
    for kid, v in zip(element_keyword_ids, element_vectors):
        if kid in row:
            out[row[kid]] += v
            cnt[kid] += 1
    out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)
    return out
