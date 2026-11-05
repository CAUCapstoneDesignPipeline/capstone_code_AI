"""두 키워드 사이의 가중 경로 (결정적, LLM 없음).

이전: 2~3걸음만, 이웃이 많은 허브 키워드는 건너뜀, 이웃은 점수 상위 25개만 → 허브를 거치거나 4걸음 이상인 관련은 놓쳤다.
지금: 걸음 수 대신 **비용** 으로 길을 고른다.
  선 비용   = 1 / (ε + 증거 점수)             증거가 강한 선(같은 문장 등)은 싸고, 약한 선(문단)은 비싸다
  노드 비용 = λ · log2(1 + 이웃 수)            중간에 지나는 키워드마다. 허브는 건너뛰지 않고 비싸게 친다 (모든 것과 이어져서 말해 주는 게 적다)
  길 비용   = 선 비용 + 중간 노드 비용의 합   (두 끝 키워드는 노드 비용이 없다)
가장 싼 길에서 시작해, 쓴 중간 키워드의 비용을 올려 가며 다시 찾는다 (패널티 방식): 서로 다른 길을 k 개 얻는다. 같은 길을 조금씩 바꾼 변형을 늘어놓지 않으려는 것이다.
두 키워드의 직접 선은 쓰지 않는다 (직접 증거는 따로 보이므로 여기서는 돌아가는 길만 찾는다). 비용이 max_cost 를 넘는 길은 버린다.
기본값(λ=0.5, max_cost=10)은 라벨 275쌍 + 무작위 400쌍에서 골랐다: 쓸모 있다고 한 쌍의 90% 에서 길이 나오고 무작위 쌍은 5% 만 나온다
(이전 방식 — 3걸음·허브 제외 — 은 77% 와 3%). 가장 싼 길의 비용으로 쓸모 있는 쌍을 올리는 정도는 AUC 0.73 으로 Katz(0.725)·랜덤 워크(0.738) 와 비슷하다.
λ 를 0 → 0.3 → 0.6 으로 올리면 AUC 가 0.709 → 0.725 → 0.728 이다 (쓸모 있는 쌍이 30개뿐이라 차이는 구별되지 않는다).
"""

import heapq
import math
from dataclasses import dataclass

EPS = 0.2
LAMBDA = 0.5          # 노드 비용 계수: 이웃 10개 → +1.7, 100개 → +3.3, 500개 → +4.5
MAX_COST = 10.0
MAX_HOPS = 6
REUSE = 1.6           # 이미 쓴 중간 키워드의 비용에 곱하는 값


@dataclass(frozen=True)
class WeightedPath:
    nodes: tuple[str, ...]           # 키워드 id (두 끝 포함)
    cost: float
    bottleneck: float                # 가장 약한 선의 증거 점수
    hubs: tuple[str, ...]            # 지나온 중간 키워드 중 허브 (이웃이 hub 개 이상)

    @property
    def hops(self) -> int:
        return len(self.nodes) - 1


def edge_cost(w: float) -> float:
    return 1.0 / (EPS + w)


def node_cost(deg: int) -> float:
    return LAMBDA * math.log2(1 + deg)


def path_cost(adj: dict[str, dict[str, float]], nodes) -> float:
    c = sum(edge_cost(adj[a][b]) for a, b in zip(nodes, nodes[1:]))
    return c + sum(node_cost(len(adj[v])) for v in nodes[1:-1])


def _dijkstra(adj, deg, src, dst, mult, bound):
    """mult 로 비용을 올린 그래프에서 src → dst 최단 경로 (직접 선 src–dst 는 쓰지 않는다). bound 를 넘으면 포기."""
    dist, prev, heap = {src: 0.0}, {}, [(0.0, src)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist.get(u, math.inf):
            continue
        if u == dst:
            out = [dst]
            while out[-1] != src:
                out.append(prev[out[-1]])
            return out[::-1]
        for v, w in adj[u].items():
            if u == src and v == dst:
                continue
            step = edge_cost(w) + (0.0 if v == dst else node_cost(deg[v]))
            nd = d + step * mult.get(v, 1.0)
            if nd <= bound and nd < dist.get(v, math.inf):
                dist[v], prev[v] = nd, u
                heapq.heappush(heap, (nd, v))
    return None


def weighted_paths(adj: dict[str, dict[str, float]], a: str, b: str, k: int = 6, max_cost: float = MAX_COST,
                   max_hops: int = MAX_HOPS, hub: int = 40) -> list[WeightedPath]:
    """a, b 사이의 서로 다른 길 최대 k 개 (싼 순). adj: 키워드 id → {이웃 id: 증거 점수} (대칭)."""
    if a == b or a not in adj or b not in adj:
        return []
    deg = {v: len(n) for v, n in adj.items()}
    mult: dict[str, float] = {}
    found: dict[tuple[str, ...], WeightedPath] = {}
    misses = 0
    for _ in range(k * 3):
        nodes = _dijkstra(adj, deg, a, b, mult, max_cost * 3)
        if nodes is None:
            break
        key = tuple(nodes)
        cost = path_cost(adj, nodes)
        if key in found or len(nodes) - 1 > max_hops or cost > max_cost:
            misses += 1
            if misses >= 3:
                break
        else:
            found[key] = WeightedPath(key, round(cost, 4), min(adj[x][y] for x, y in zip(nodes, nodes[1:])),
                                      tuple(v for v in nodes[1:-1] if deg[v] >= hub))
            if len(found) >= k:
                break
        for v in nodes[1:-1]:                                    # 쓴 중간 키워드는 비싸게 해서 다른 길을 찾게 한다
            mult[v] = mult.get(v, 1.0) * REUSE
        if len(nodes) == 2:                                      # 중간 키워드가 없는 길은 직접 선뿐인데 위에서 막았으므로 오지 않는다
            break
    return sorted(found.values(), key=lambda p: (p.cost, p.nodes))
