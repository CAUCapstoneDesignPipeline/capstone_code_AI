import itertools
import math
import random

import pytest

from app.services.links import paths as P
from app.services.links.paths import WeightedPath, edge_cost, node_cost, path_cost, weighted_paths


def graph(edges):
    adj = {}
    for a, b, w in edges:
        adj.setdefault(a, {})[b] = w
        adj.setdefault(b, {})[a] = w
    return adj


def all_simple_paths(adj, a, b, max_hops):
    out = []

    def go(path):
        u = path[-1]
        if len(path) - 1 > max_hops:
            return
        for v in adj[u]:
            if v in path or (u == a and v == b):                 # 직접 선은 쓰지 않는다
                continue
            if v == b:
                out.append(tuple(path + [v]))
            else:
                go(path + [v])
    go([a])
    return out


def random_graph(n, m, seed):
    rng = random.Random(seed)
    return graph([(f"k{a}", f"k{b}", round(rng.choice([0.15, 0.3, 0.5, 1.0, 2.0]), 2)) for a, b in (rng.sample(range(n), 2) for _ in range(m))])


@pytest.mark.parametrize("seed", range(12))
def test_cheapest_path_equals_brute_force_over_all_simple_paths(seed):
    adj = random_graph(14, 26, seed)
    nodes = sorted(adj)
    for a, b in itertools.islice(itertools.combinations(nodes, 2), 25):
        got = weighted_paths(adj, a, b, k=1, max_cost=1e9, max_hops=99)
        every = all_simple_paths(adj, a, b, 99)
        if not every:
            assert got == []
            continue
        best = min(path_cost(adj, p) for p in every)
        assert got and math.isclose(got[0].cost, best, abs_tol=1e-3), (a, b)
        assert math.isclose(path_cost(adj, got[0].nodes), got[0].cost, abs_tol=1e-3)


def test_prefers_a_path_through_a_small_keyword_over_one_through_a_hub():
    edges = [("a", "x", 1.0), ("x", "b", 1.0), ("a", "hub", 1.0), ("hub", "b", 1.0)]
    edges += [("hub", f"leaf{i}", 0.3) for i in range(60)]                          # hub 는 이웃이 62개
    best = weighted_paths(graph(edges), "a", "b", k=2, hub=40)
    assert [p.nodes for p in best] == [("a", "x", "b"), ("a", "hub", "b")]            # 허브를 건너뛰지는 않고 뒤에 둔다
    assert best[0].hubs == () and best[1].hubs == ("hub",) and best[0].cost < best[1].cost
    assert math.isclose(best[1].cost - best[0].cost, node_cost(62) - node_cost(2), abs_tol=1e-3)


def test_a_strong_edge_is_cheaper_than_a_weak_one_and_a_long_strong_path_can_beat_a_short_weak_one():
    assert edge_cost(2.0) < edge_cost(1.0) < edge_cost(0.3)
    adj = graph([("a", "m", 0.15), ("m", "b", 0.15), ("a", "p", 2.0), ("p", "q", 2.0), ("q", "b", 2.0)])
    p = weighted_paths(adj, "a", "b", k=2)
    assert p[0].nodes == ("a", "p", "q", "b") and p[1].nodes == ("a", "m", "b")                 # 걸음 수가 아니라 비용 순
    assert p[0].bottleneck == 2.0 and p[1].bottleneck == 0.15


def test_the_direct_edge_is_never_a_path_and_paths_are_distinct_and_sorted():
    adj = graph([("a", "b", 5.0), ("a", "x", 1.0), ("x", "b", 1.0), ("a", "y", 1.0), ("y", "b", 1.0), ("a", "z", 0.5), ("z", "b", 0.5)])
    out = weighted_paths(adj, "a", "b", k=6)
    assert ("a", "b") not in [p.nodes for p in out] and len({p.nodes for p in out}) == len(out) == 3
    assert [p.cost for p in out] == sorted(p.cost for p in out)
    assert all(p.nodes[0] == "a" and p.nodes[-1] == "b" and len(set(p.nodes)) == len(p.nodes) for p in out)     # 단순 경로


def test_limits_unreachable_unknown_and_same_node():
    adj = graph([("a", "b", 1.0), ("c", "d", 1.0), ("b", "c", 0.15), ("d", "e", 0.15), ("e", "f", 0.15), ("f", "g", 0.15)])
    assert weighted_paths(adj, "a", "zzz") == [] and weighted_paths(adj, "a", "a") == []
    assert weighted_paths(adj, "a", "d")[0].nodes == ("a", "b", "c", "d")
    assert weighted_paths(adj, "a", "g", max_hops=3) == []                            # 걸음 수 한도
    assert weighted_paths(adj, "a", "g", max_cost=5.0) == [] and weighted_paths(adj, "a", "g", max_cost=1e9, max_hops=9)
    assert weighted_paths(graph([("a", "b", 1.0), ("c", "d", 1.0)]), "a", "d") == []   # 이어지지 않은 쌍


def test_deterministic_and_reaches_beyond_three_hops():
    adj = random_graph(40, 70, 5)
    a, b = "k1", "k7"
    assert weighted_paths(adj, a, b, max_cost=1e9, max_hops=12) == weighted_paths(adj, a, b, max_cost=1e9, max_hops=12)
    chain = graph([(f"n{i}", f"n{i + 1}", 2.0) for i in range(5)])                     # 5칸: 이전 방식(3칸)으로는 못 찾던 길
    assert weighted_paths(chain, "n0", "n5", max_cost=1e9)[0].hops == 5 and isinstance(weighted_paths(chain, "n0", "n5", max_cost=1e9)[0], WeightedPath)


def test_defaults_come_from_the_evaluation():
    assert (P.LAMBDA, P.MAX_COST, P.MAX_HOPS) == (0.5, 10.0, 6)
