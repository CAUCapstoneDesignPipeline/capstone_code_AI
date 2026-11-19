"""후보 순위를 관계로 매기는 재료: 관계 점수 · 노트별 캐시 · 배경 채우기."""

import threading
from types import SimpleNamespace

from app.services.links.note_relations import NoteRelations, Warmup
from app.services.links.relation_rank import Graph, relation_scores


def note(concepts, relations):
    return {"concepts": {k: {"surface": k, "mentions": [[s, s + 1]]} for k, s in concepts.items()},
            "relations": [{"source": s, "target": t, "code": c, "confidence": w, "evidence": [[e, e + 1]]} for s, t, c, w, e in relations]}


def el(n, start, end, origin="extracted"):
    return SimpleNamespace(note=n, start=start, end=end, text="x", origin=origin)


def graph():
    # N1 의 heap 과 N2 의 avl 은 tree 로 이어지고, N3 의 dns 는 아무와도 이어지지 않는다
    return Graph({"N1": note({"heap": 0, "tree": 5}, [("heap", "tree", "is_a", 0.9, 0)]),
                  "N2": note({"avl": 0, "tree": 5}, [("avl", "tree", "is_a", 0.8, 0)]),
                  "N3": note({"dns": 0}, [])})


def test_relation_score_puts_the_connected_pair_ahead_of_the_unconnected_one():
    g = graph()
    r = relation_scores(g, [(el("N1", 0, 3), el("N3", 0, 3)), (el("N1", 0, 3), el("N2", 0, 3))])
    assert r[1]["score"] > r[0]["score"] and r[1]["shared_nb"] == 0.8 and r[0]["shared_nb"] == 0.0
    assert 0 < r[0]["score"] <= 1 and 0 < r[1]["score"] <= 1


def test_relation_score_is_none_without_relations_for_that_note_or_for_accepted_knowledge():
    g = graph()
    r = relation_scores(g, [(el("N1", 0, 3), el("NOPE", 0, 3)), (el("N1", 0, 3, "generated"), el("N2", 0, 3)), (el("N1", 0, 3), el("N2", 0, 3))])
    assert r[0] is None and r[1] is None and r[2] is not None
    assert relation_scores(g, [(el("NOPE", 0, 1), el("N1", 0, 1))]) == [None]
    assert relation_scores(g, []) == []


def test_ties_share_an_average_rank_so_equal_pairs_get_equal_scores():
    g = graph()
    r = relation_scores(g, [(el("N1", 0, 3), el("N3", 0, 3)), (el("N2", 0, 3), el("N3", 0, 3))])
    assert r[0]["score"] == r[1]["score"]


def test_note_relation_cache_is_keyed_by_note_text_and_never_calls_a_model_when_reading(tmp_path):
    store = NoteRelations(tmp_path / "c")
    assert store.read("N1", "글") is None and store.missing({"N1": "글"}) == ["N1"]
    store.dir.mkdir(parents=True)
    store.path("N1", "글").write_text('{"note": "N1", "concepts": {}, "relations": []}', encoding="utf-8")
    assert store.read("N1", "글")["note"] == "N1"
    assert store.read("N1", "고친 글") is None                      # 글이 바뀌면 다시 뽑아야 한다
    assert store.missing({"N1": "글", "N2": "다른 글"}) == ["N2"]
    g, n = store.graph({"N1": "글", "N2": "다른 글"})
    assert n == 1 and set(g.notes) == {"N1"}
    assert NoteRelations(tmp_path / "c", "other-version").read("N1", "글") is None


def test_compute_writes_the_cache_and_reads_it_back_without_calling_again(tmp_path):
    calls = []
    m = SimpleNamespace(surface="heap", span=SimpleNamespace(start=0, end=4))
    student = SimpleNamespace(select=lambda extractor, note_id, v, text: calls.append("select") or {"heap": [m]})
    rel = SimpleNamespace(source="heap", target="tree", code="is_a", confidence=0.9, evidence=[0])
    pair = SimpleNamespace(extract=lambda text, selected, policy, limits: ([SimpleNamespace(start16=0, end16=4)], [rel]))
    store = NoteRelations(tmp_path)
    out = store.compute(student, None, pair, "N1", "heap is a tree", [], None)
    assert out["concepts"]["heap"]["mentions"] == [[0, 4]] and out["relations"][0]["evidence"] == [[0, 4]]
    assert store.compute(student, None, pair, "N1", "heap is a tree", [], None) == out and calls == ["select"]


def test_warmup_fills_one_by_one_retries_rate_limits_and_skips_a_failing_note():
    done, tries = [], {"b": 0}

    def work(n):
        if n == "b":
            tries["b"] += 1
            if tries["b"] < 3:
                raise RuntimeError("rate_limit_exceeded")
        if n == "c":
            raise RuntimeError("boom")
        done.append(n)

    w = Warmup()
    assert w.start(["a", "b", "c", "d"], work, attempts=4, wait=0.0)
    w.thread.join(5)
    assert done == ["a", "b", "d"] and tries["b"] == 3                  # b 는 한도에 두 번 걸렸다가 통과, c 는 건너뛴다
    s = w.status()
    assert s["running"] is False and s["done"] == 4 and s["todo"] == 4 and "c: boom" in s["error"]


def test_warmup_does_not_start_a_second_run_while_one_is_running():
    gate = threading.Event()
    w = Warmup()
    assert w.start(["a"], lambda n: gate.wait(5))
    assert w.start(["b"], lambda n: None) is False and w.status()["running"] is True
    gate.set()
    w.thread.join(5)
    assert w.start(["b"], lambda n: None) is True
