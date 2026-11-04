from app.services.links.pair_screen import PairScreen
from app.services.llm import LLMResult


class Fake:
    provider, model = "anthropic", "fake"

    def __init__(self, drop=()):
        self.drop, self.calls = set(drop), []

    def complete_json(self, system, user, schema, name):
        ids = [l[4:] for l in user.splitlines() if l.startswith("id: ")]
        self.calls.append(ids)
        return LLMResult({"results": [{"id": i, "label": "useful" if i == "p1" else "vague", "conclusion": "결론" if i == "p1" else ""}
                                      for i in ids if i not in self.drop]}, self.provider, self.model)


def pair(i):
    return {"id": f"p{i}", "a": {"keyword": f"A{i}", "text": f"A 문장 {i}"}, "b": {"keyword": f"B{i}", "text": f"B 문장 {i}"}}


def test_labels_and_cache(tmp_path):
    fake = Fake()
    s = PairScreen(fake, tmp_path)
    r = s.screen([pair(1), pair(2)])
    assert (r["p1"]["label"], r["p1"]["conclusion"], r["p2"]["label"]) == ("useful", "결론", "vague")
    assert s.screen([pair(1), pair(2)])["p1"]["cached"] is True and len(fake.calls) == 1


def test_a_pair_the_model_skips_is_asked_once_more_in_a_small_batch_then_unsure():
    fake = Fake(drop=["p2"])
    r = PairScreen(fake, None).screen([pair(1), pair(2)])
    assert r["p2"]["label"] == "unsure" and fake.calls == [["p1", "p2"], ["p2"]]


class StreamFake(Fake):
    """stream_json 이 있는 가짜 LLM: 결과를 JSON 으로 쓰듯 필드 이벤트를 낸다."""

    def stream_json(self, system, user, schema, name, on_event=None):
        ids = [l[4:] for l in user.splitlines() if l.startswith("id: ")]
        rows = [i for i in ids if i not in self.drop]
        for n, pid in enumerate(rows):
            label = "useful" if pid == "p1" else "vague"
            note = "결론" if pid == "p1" else ""
            for e in ({"path": f"results.{n}.id", "delta": pid}, {"path": f"results.{n}.id", "value": pid},
                      {"path": f"results.{n}.label", "delta": label}, {"path": f"results.{n}.label", "value": label},
                      *([{"path": f"results.{n}.conclusion", "delta": "결"}, {"path": f"results.{n}.conclusion", "delta": "론"},
                         {"path": f"results.{n}.conclusion", "value": note}] if note else [])):
                on_event({"type": "field", **e})
        return self.complete_json(system, user, schema, name)


def test_screen_reports_each_pair_as_it_is_written_and_when_it_is_settled(tmp_path):
    seen = []
    s = PairScreen(StreamFake(), tmp_path)
    r = s.screen([pair(1), pair(2)], on_event=seen.append)
    assert r["p1"]["label"] == "useful"
    by_id = lambda i, f: [e for e in seen if e["type"] == "pair" and e["id"] == i and e["field"] == f]
    assert "".join(e["delta"] for e in by_id("p1", "conclusion") if "delta" in e) == "결론" and by_id("p1", "label")[-1] == {"type": "pair", "id": "p1", "field": "label", "value": "useful"}
    assert [e["id"] for e in seen if e["type"] == "pair_done"] == ["p1", "p2"] or {e["id"] for e in seen if e["type"] == "pair_done"} == {"p1", "p2"}
    assert next(e for e in seen if e["type"] == "pair_done" and e["id"] == "p1") == {"type": "pair_done", "id": "p1", "label": "useful", "conclusion": "결론", "cached": False}
    seen.clear()
    s.screen([pair(1), pair(2)], on_event=seen.append)                    # 저장된 결과는 모델 없이 바로 확정으로 온다
    assert [e["type"] for e in seen] == ["pair_done", "pair_done"] and all(e["cached"] for e in seen)


def test_a_pair_the_model_skips_still_gets_a_settled_event():
    seen = []
    PairScreen(StreamFake(drop=["p2"]), None).screen([pair(1), pair(2)], on_event=seen.append)
    assert {"type": "pair_done", "id": "p2", "label": "unsure", "conclusion": "", "cached": False} in seen
    assert not [e for e in seen if e["type"] == "pair" and e["id"] == "p2"]          # 쓰이지 않은 쌍의 글은 없다


def test_without_streaming_support_only_settled_events_come():
    seen = []
    PairScreen(Fake(), None).screen([pair(1)], on_event=seen.append)
    assert [e["type"] for e in seen] == ["pair_done"]
