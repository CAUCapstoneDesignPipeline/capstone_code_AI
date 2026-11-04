import json
from types import SimpleNamespace as NS

import pytest

from app.services.llm import LLMConfig
from app.services.llm.anthropic_llm import AnthropicLLM


class FakeStream:
    """messages.stream() 이 돌려주는 컨텍스트: 이벤트를 차례로 내고 마지막에 전체 메시지를 준다."""

    def __init__(self, events, final):
        self.events, self.final = events, final

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __iter__(self):
        return iter(self.events)

    def get_final_message(self):
        return self.final


def start(i, **b):
    return NS(type="content_block_start", index=i, content_block=NS(**b))


def delta(i, **d):
    return NS(type="content_block_delta", index=i, delta=NS(**d))


def stop(i):
    return NS(type="content_block_stop", index=i)


ANSWER = json.dumps({"status": "ok", "conclusion": "가" * 450})
EVENTS = [
    start(0, type="thinking"), stop(0),
    start(1, type="server_tool_use", name="web_search"),
    delta(1, type="input_json_delta", partial_json='{"query": "page '), delta(1, type="input_json_delta", partial_json='cache"}'), stop(1),
    start(2, type="web_search_tool_result", content=[NS(url="https://a.example/x", title="A"), NS(url="https://b.example/y", title="B")]), stop(2),
    start(3, type="text"), delta(3, type="text_delta", text=ANSWER[:250]), delta(3, type="text_delta", text=ANSWER[250:]), stop(3),
]
FINAL = NS(stop_reason="end_turn", model="m", usage=NS(input_tokens=10, output_tokens=20, cache_read_input_tokens=5),
           content=[NS(type="server_tool_use", name="web_search"),
                    NS(type="web_search_tool_result", content=[NS(url="https://a.example/x", title="A"), NS(url="https://b.example/y", title="B")]),
                    NS(type="text", text=ANSWER)])


def llm(events=EVENTS, final=FINAL):
    client = NS(messages=NS(stream=lambda **kw: FakeStream(events, final)))
    return AnthropicLLM(LLMConfig("anthropic", "m"), client)


def test_progress_events_follow_the_stream_and_the_result_is_unchanged():
    seen = []
    r = llm().complete_web("sys", "user", {}, "n", 2, "web_search_20250305", on_event=seen.append)
    assert [e["type"] for e in seen if e["type"] != "field"] == ["thinking", "search_start", "search", "search_results", "writing", "writing", "writing"]
    fields = [e for e in seen if e["type"] == "field"]
    assert {"type": "field", "path": "status", "value": "ok"} in fields
    assert "".join(e["delta"] for e in fields if e["path"] == "conclusion" and "delta" in e) == "가" * 450       # 결론이 쓰이는 대로 온다
    assert fields[-1] == {"type": "field", "path": "conclusion", "value": "가" * 450}
    by = {e["type"]: e for e in seen if e["type"] in ("search", "search_results")}
    assert by["search"]["query"] == "page cache" and [x["url"] for x in by["search_results"]["results"]] == ["https://a.example/x", "https://b.example/y"]
    assert [e["chars"] for e in seen if e["type"] == "writing"] == [0, 250, len(ANSWER)]   # 글자마다가 아니라 200자 넘게 쓸 때마다 알린다
    assert r.data["status"] == "ok" and [s["url"] for s in r.sources] == ["https://a.example/x", "https://b.example/y"]
    assert r.searches == 1 and r.usage == {"input_tokens": 10, "output_tokens": 20, "cache_read_input_tokens": 5}


def test_works_without_a_listener():
    assert llm().complete_web("sys", "user", {}, "n").searches == 1


def test_a_listener_that_raises_stops_the_call():
    class Stop(Exception):
        pass

    def listener(e):
        if e["type"] == "search":
            raise Stop()
    with pytest.raises(Stop):
        llm().complete_web("sys", "user", {}, "n", on_event=listener)


def test_stream_json_reports_fields_as_they_are_written_and_returns_the_parsed_result():
    answer = json.dumps({"verdict": "weak", "reason": "중간 키워드를 거친 간접 관계예요."}, ensure_ascii=False)
    events = [start(0, type="text"), delta(0, type="text_delta", text=answer[:20]), delta(0, type="text_delta", text=answer[20:]), stop(0)]
    final = NS(stop_reason="end_turn", model="m", usage=NS(input_tokens=1, output_tokens=2, cache_read_input_tokens=0), content=[NS(type="text", text=answer)])
    seen = []
    r = llm(events, final).stream_json("sys", "user", {}, "n", on_event=seen.append)
    assert r.data == {"verdict": "weak", "reason": "중간 키워드를 거친 간접 관계예요."}
    assert all(e["type"] == "field" for e in seen) and {"type": "field", "path": "verdict", "value": "weak"} in seen
    assert "".join(e["delta"] for e in seen if e.get("path") == "reason" and "delta" in e) == "중간 키워드를 거친 간접 관계예요."


def test_stream_json_refusal_and_stop_reasons_raise_like_complete_json():
    from app.services.llm import LLMError
    for stop, kind in (("refusal", "refusal"), ("max_tokens", "invalid_output")):
        final = NS(stop_reason=stop, model="m", usage=NS(input_tokens=1, output_tokens=2, cache_read_input_tokens=0), content=[])
        with pytest.raises(LLMError) as e:
            llm([], final).stream_json("sys", "user", {}, "n")
        assert e.value.kind == kind


def sent_llm(events, final):
    sent = []
    client = NS(messages=NS(stream=lambda **kw: (sent.append(kw), FakeStream(events, final))[1]))
    return AnthropicLLM(LLMConfig("anthropic", "m"), client), sent


RESULT = {"status": "ok", "conclusion": "DB 도 OS 도 캐시해서 같은 페이지가 두 번 올라가요.", "reason": ""}
RAW = json.dumps(RESULT, ensure_ascii=False)


def submit_events(raw=RAW, split=7):
    pieces = [raw[i:i + split] for i in range(0, len(raw), split)]
    return [start(0, type="thinking"), stop(0),
            start(1, type="server_tool_use", name="web_search"), delta(1, type="input_json_delta", partial_json='{"query": "x"}'), stop(1),
            start(2, type="web_search_tool_result", content=[NS(url="https://a.example/x", title="A")]), stop(2),
            start(3, type="tool_use", name="submit_knowledge"), *[delta(3, type="input_json_delta", partial_json=p) for p in pieces], stop(3)]


def submit_final(stop_reason="tool_use", name="submit_knowledge"):
    return NS(stop_reason=stop_reason, model="m", usage=NS(input_tokens=1, output_tokens=2, cache_read_input_tokens=0),
              content=[NS(type="web_search_tool_result", content=[NS(url="https://a.example/x", title="A")]), NS(type="tool_use", name=name, input=RESULT)])


def test_submit_tool_streams_the_answer_as_it_is_written_and_returns_the_tool_input():
    llm_, sent = sent_llm(submit_events(), submit_final())
    seen = []
    r = llm_.complete_web("sys", "user", {"type": "object"}, "n", 1, "web_search_20250305", on_event=seen.append, submit="submit_knowledge")
    assert r.data == RESULT and [s["url"] for s in r.sources] == ["https://a.example/x"]
    conclusion = [e for e in seen if e["type"] == "field" and e["path"] == "conclusion"]
    assert len([e for e in conclusion if "delta" in e]) >= 3                                   # 한 번에 오지 않고 조각조각 온다
    assert "".join(e["delta"] for e in conclusion if "delta" in e) == RESULT["conclusion"]
    kw = sent[0]
    assert kw["tools"][0]["type"] == "web_search_20250305" and kw["tools"][1]["name"] == "submit_knowledge" and kw["tools"][1]["eager_input_streaming"] is True
    assert kw["tools"][1]["input_schema"] == {"type": "object"} and "format" not in kw["output_config"]           # 구조화 출력은 쓰지 않는다 (검색 뒤에 한꺼번에 와서)


def test_submit_tool_input_is_reparsed_strictly_so_a_cut_off_call_is_an_error():
    from app.services.llm import LLMError
    llm_, _ = sent_llm(submit_events(RAW[:-8]), submit_final())                                 # 도구 입력이 중간에 끊겼다
    with pytest.raises(LLMError) as e:
        llm_.complete_web("sys", "user", {}, "n", submit="submit_knowledge")
    assert e.value.kind == "invalid_output"


def test_submit_mode_rejects_a_tool_call_to_something_else_and_accepts_plain_json_text():
    from app.services.llm import LLMError
    llm_, _ = sent_llm(submit_events(), submit_final(name="other_tool"))
    with pytest.raises(LLMError):
        llm_.complete_web("sys", "user", {}, "n", submit="submit_knowledge")
    text_final = NS(stop_reason="end_turn", model="m", usage=NS(input_tokens=1, output_tokens=2, cache_read_input_tokens=0), content=[NS(type="text", text=RAW)])
    llm_, _ = sent_llm([], text_final)
    assert llm_.complete_web("sys", "user", {}, "n", submit="submit_knowledge").data == RESULT     # 도구를 안 쓰고 JSON 글로 답해도 받는다
