import numpy as np
import pytest

from app.services.links.inference import BETA, Inferred, LinkGraph, keyword_vectors


def dense(g):
    A = np.zeros((len(g.ids), len(g.ids)))
    for i in range(len(g.ids)):
        A[i, g.neighbors(i)] = 1
    return A


def random_graph(n=60, m=150, seed=0, vectors=False):
    rng = np.random.default_rng(seed)
    ids = [f"k{i}" for i in range(n)]
    pairs = {(ids[a], ids[b]) for a, b in rng.integers(0, n, (m, 2)) if a != b}
    v = None
    if vectors:
        v = rng.standard_normal((n, 8)).astype(np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
    return LinkGraph(ids, pairs, v)


def test_step_and_katz_equal_the_dense_matrix_algebra():
    g = random_graph()
    A = dense(g)
    x = np.random.default_rng(1).random(len(g.ids)) * (np.random.default_rng(2).random(len(g.ids)) < 0.3)
    assert np.allclose(g.step(x), A @ x)
    for i in (0, 7, 33):
        assert np.allclose(g.katz(i), BETA * (A @ A)[i] + BETA ** 2 * (A @ A @ A)[i])


def test_chain_infers_the_nearer_node_first_and_never_an_existing_link():
    g = LinkGraph(list("abcde"), [("a", "b"), ("b", "c"), ("c", "d")])        # e 는 외톨이
    got = g.infer("a", top=5, min_score=-np.inf)
    assert [x.b for x in got][:2] == ["c", "d"] and "b" not in [x.b for x in got] and "a" not in [x.b for x in got]
    assert got[0].score > got[1].score and got[0].via == ("b",) and got[0].cos is None
    assert g.infer("zzz") == []


def test_via_lists_common_neighbors_with_the_fewest_links_first():
    g = LinkGraph(list("axyzq"), [("a", "x"), ("a", "y"), ("x", "z"), ("y", "z"), ("y", "q")])
    top = g.infer("a", min_score=-np.inf)[0]
    assert top.b == "z" and top.via == ("x", "y")                              # x(이웃 2) 가 y(이웃 3) 보다 먼저


def test_semantically_close_keyword_is_inferred_even_without_any_path():
    ids = ["a", "b", "c", "far", "twin"]
    v = np.array([[1, 0], [0, 1], [0.1, 1], [-1, 0.2], [0.99, 0.1]], dtype=np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    g = LinkGraph(ids, [("a", "b"), ("b", "c")], v)
    got = g.infer("a", top=5, min_score=-np.inf)
    twin = next(x for x in got if x.b == "twin")
    assert twin.katz == 0 and twin.via == () and twin.cos > 0.9 and got[0].b in ("twin", "c")
    assert all(x.cos is not None for x in got)


def test_score_of_any_pair_but_none_for_linked_or_same():
    g = random_graph(vectors=True)
    i = 5
    linked = g.ids[int(g.neighbors(i)[0])]
    assert g.score("k5", linked) is None and g.score("k5", "k5") is None and g.score("k5", "nope") is None
    free = next(k for k in g.ids if k != "k5" and g.idx[k] not in set(g.neighbors(i).tolist()))
    s = g.score("k5", free)
    assert isinstance(s, Inferred) and np.isfinite(s.score)


def test_deterministic_and_threshold_filters():
    g1, g2 = random_graph(vectors=True), random_graph(vectors=True)
    assert g1.infer("k3", min_score=-np.inf) == g2.infer("k3", min_score=-np.inf)
    everything = g1.infer("k3", top=100, min_score=-np.inf)
    assert all(x.score >= 2.0 for x in g1.infer("k3", top=100, min_score=2.0)) and len(g1.infer("k3", top=100, min_score=99.0)) == 0
    assert [x.score for x in everything] == sorted((x.score for x in everything), reverse=True)


def test_weight_grows_with_score_inside_the_display_range():
    mk = lambda s: Inferred("a", "b", s, None, 0.0).weight
    assert mk(3.0) == 0.1 and mk(5.0) == 0.3 and mk(7.0) == 0.5 and mk(99.0) == 0.5 and mk(-5.0) == 0.1


def test_keyword_vectors_average_the_elements_and_normalize():
    el = np.array([[1, 0], [0, 1], [3, 4]], dtype=np.float32)
    out = keyword_vectors(["a", "a", "b"], el, ["a", "b", "none"])
    assert np.allclose(out[0], [2 ** -0.5, 2 ** -0.5]) and np.allclose(out[1], [0.6, 0.8]) and not out[2].any()
