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
    assert [e["type"] for e in seen] == ["thinking", "search_start", "search", "search_results", "writing", "writing", "writing"]
    assert seen[2]["query"] == "page cache" and [x["url"] for x in seen[3]["results"]] == ["https://a.example/x", "https://b.example/y"]
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
