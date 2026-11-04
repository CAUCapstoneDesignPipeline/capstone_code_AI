import asyncio
import hashlib
import json
import uuid
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.services.chat import ChatService, build_system

TOKEN = "chat-token"
CONV = "81738c89-df53-57b7-9a06-320a4deb50c8"
KNOWLEDGE = {
    "newKeyword": "이중 버퍼링",
    "conclusion": "DB 와 OS 가 둘 다 페이지를 캐시해서 같은 데이터가 메모리에 두 번 올라갈 수 있어요.",
    "evidence": [{"label": "A", "keyword": "디스크", "note": "DB-WAL", "text": "DB는 buffer pool에 페이지를 올려놓고 거기서 수정한다."},
                 {"label": "B", "keyword": "페이지 캐싱", "note": "OS", "text": "디스크에서 읽은 데이터를 RAM에 저장해 재사용하는 것을 페이지 캐싱이라고 한다."}],
    "external": [{"url": "https://example.com/innodb", "title": "InnoDB", "quote": "O_DIRECT 는 OS 페이지 캐시를 거치지 않는다."}],
}


def body(messages, stream=True, **over):
    return {"conversationId": CONV, "knowledge": KNOWLEDGE, "messages": messages, "stream": stream, **over}


def user(t):
    return {"role": "user", "content": t}


def bot(t):
    return {"role": "assistant", "content": t}


class FakeStream:
    """AsyncAnthropic.messages.stream() 이 돌려주는 비동기 컨텍스트. 글자 조각을 하나씩 내고 마지막에 전체 메시지를 준다."""

    def __init__(self, parts, stop="end_turn", fail_after=None, thinking=True, log=None):
        self.parts, self.stop, self.fail_after, self.thinking, self.log = parts, stop, fail_after, thinking, log
        self.exited = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        self.exited = True
        if self.log is not None:
            self.log["exited"] = True
        return False

    async def __aiter__(self):
        if self.thinking:
            yield NS(type="content_block_start", content_block=NS(type="thinking"))
        yield NS(type="content_block_start", content_block=NS(type="text"))
        for i, p in enumerate(self.parts):
            if self.fail_after is not None and i == self.fail_after:
                import anthropic
                raise anthropic.APIConnectionError(request=NS())
            await asyncio.sleep(0)
            yield NS(type="content_block_delta", delta=NS(type="text_delta", text=p))

    async def get_final_message(self):
        return NS(stop_reason=self.stop, model="fake-chat", usage=NS(input_tokens=120, output_tokens=len(self.parts), cache_read_input_tokens=0))


class FakeClient:
    def __init__(self, parts=("쉽게 ", "말해 ", "볼게요."), **kw):
        self.calls, self.parts, self.kw, self.log = [], parts, kw, {}
        self.messages = NS(stream=self._stream)

    def _stream(self, **kwargs):
        self.calls.append(kwargs)
        err = self.kw.get("raise_on_open")
        if err:
            raise err
        return FakeStream(self.parts, log=self.log, **{k: v for k, v in self.kw.items() if k != "raise_on_open"})


def make(client, token=TOKEN, **over):
    s = Settings(service_token=token, concept_idf=Path("does-not-exist.json"), max_inflight=2, **over)
    return create_app(s, chat=ChatService(client, "fake-chat", "low"))


def call(c, payload=None, raw=None, **hdr):
    raw = raw if raw is not None else json.dumps(payload, ensure_ascii=False).encode()
    h = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json", "X-Request-Id": str(uuid.uuid4()),
         "X-Input-SHA256": hashlib.sha256(raw).hexdigest(), **hdr}
    h = {k: v for k, v in h.items() if v is not None}
    return c.post("/internal/v1/chat", content=raw, headers=h)


def sse(text):
    out = []
    for block in text.strip().split("\n\n"):
        ev = [l[7:] for l in block.split("\n") if l.startswith("event: ")]
        data = [l[6:] for l in block.split("\n") if l.startswith("data: ")]
        if ev and data:
            out.append((ev[0], json.loads(data[0])))
    return out


def test_streams_tokens_as_server_sent_events():
    fake = FakeClient()
    with TestClient(make(fake)) as c:
        r = call(c, body([user("이해가 안 돼요")]))
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream") and r.headers["cache-control"] == "no-cache"
        evs = sse(r.text)
    assert [e for e, _ in evs] == ["start", "thinking", "delta", "delta", "delta", "done"]
    assert "".join(d["text"] for e, d in evs if e == "delta") == "쉽게 말해 볼게요."
    start, done = evs[0][1], evs[-1][1]
    assert start["conversationId"] == CONV and start["model"] == "fake-chat" and start["requestId"]
    assert done["stopReason"] == "end_turn" and done["usage"] == {"inputTokens": 120, "outputTokens": 3, "cacheReadTokens": 0}
    assert all("t" in d for _, d in evs)


def test_multi_turn_sends_the_card_in_system_and_every_earlier_message():
    fake = FakeClient()
    history = [user("이중 버퍼링이 뭐예요?"), bot("DB 와 OS 가 같은 걸 두 번 캐시하는 거예요."), user("이해가 안 돼요")]
    with TestClient(make(fake)) as c:
        assert call(c, body(history)).status_code == 200
    sent = fake.calls[0]
    assert sent["messages"] == history and sent["model"] == "fake-chat" and sent["output_config"] == {"effort": "low"} and sent["max_tokens"] > 0
    system = sent["system"][0]["text"]
    for needle in ("이중 버퍼링", "buffer pool에 페이지를", "https://example.com/innodb", "이해가 안 돼요", "해요체"):
        assert needle in system or needle in json.dumps(sent["messages"], ensure_ascii=False)
    assert "<card>" in system and "지시" in system                                    # 카드 안의 글은 자료일 뿐이라고 못 박는다
    assert sent["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_system_prompt_tells_the_bot_to_change_approach_when_the_user_does_not_understand():
    from app.schemas.chat import Knowledge
    s = build_system(Knowledge(**KNOWLEDGE))
    assert "되풀이하지 말고 방식을 바꿔" in s and "카드에는 없지만" in s and KNOWLEDGE["conclusion"] in s


def test_non_streaming_returns_one_json_answer():
    with TestClient(make(FakeClient())) as c:
        r = call(c, body([user("쉽게 설명해 줘")], stream=False))
    j = r.json()
    assert r.status_code == 200 and j["message"] == {"role": "assistant", "content": "쉽게 말해 볼게요."}
    assert j["stopReason"] == "end_turn" and j["conversationId"] == CONV and j["usage"]["outputTokens"] == 3 and j["model"] == "fake-chat"


@pytest.mark.parametrize("messages", [
    [bot("먼저 말해요")],                                   # 사용자 발화로 시작해야 한다
    [user("하나"), bot("둘")],                              # 사용자 발화로 끝나야 한다
    [user("하나"), user("둘")],                             # 번갈아 나와야 한다
    [],
])
def test_invalid_conversation_shapes_are_rejected(messages):
    with TestClient(make(FakeClient())) as c:
        r = call(c, body(messages))
    assert r.status_code == 400 and r.json()["error"]["code"] == "VALIDATION_FAILED"


def test_contract_checks_auth_hash_unknown_fields_and_size():
    fake = FakeClient()
    with TestClient(make(fake)) as c:
        ok = body([user("안녕")])
        assert call(c, ok, Authorization="Bearer wrong").status_code == 401
        assert call(c, ok, **{"X-Input-SHA256": "0" * 64}).json()["error"]["code"] == "INPUT_IDENTITY_MISMATCH"
        assert call(c, ok, **{"X-Request-Id": "nope"}).json()["error"]["code"] == "VALIDATION_FAILED"
        assert call(c, {**ok, "extra": 1}).status_code == 400
        assert call(c, {**ok, "knowledge": {**KNOWLEDGE, "external": [{"url": "javascript:alert(1)"}]}}).status_code == 400
        too_long = body([user("가" * 4001)])
        assert call(c, too_long).status_code == 413                                     # 자르지 않고 거절한다
        many = [m for i in range(20) for m in (user("가" * 3900), bot("나" * 3900))] + [user("끝")]
        assert call(c, body(many[:38] + [user("끝")])).json()["error"]["code"] == "INPUT_TOO_LARGE"    # 전체 글자 수 한도
    assert fake.calls == []                                                             # 거절한 요청은 모델을 부르지 않는다


def test_unavailable_when_no_chat_llm_is_configured():
    s = Settings(service_token=TOKEN, concept_idf=Path("does-not-exist.json"), max_inflight=2)
    with TestClient(create_app(s)) as c:
        r = call(c, body([user("안녕")]))
    assert r.status_code == 503 and r.json()["error"]["code"] == "SERVICE_UNAVAILABLE"


def test_provider_error_before_the_first_token_is_a_normal_http_error():
    import anthropic
    err = anthropic.RateLimitError("slow down", response=NS(status_code=429, headers={}, request=NS()), body=None)
    with TestClient(make(FakeClient(raise_on_open=err))) as c:
        r = call(c, body([user("안녕")]))
        assert r.status_code == 429 and r.json()["error"]["code"] == "AI_BUSY" and r.headers["content-type"].startswith("application/json")
        r2 = call(c, body([user("안녕")], stream=False))
        assert r2.status_code == 429
        # 자리가 새지 않았다: 오류가 나도 다음 요청이 받아진다
    with TestClient(make(FakeClient())) as c:
        assert call(c, body([user("안녕")])).status_code == 200


def test_error_after_tokens_ends_the_stream_with_an_error_event():
    with TestClient(make(FakeClient(parts=("가", "나", "다"), fail_after=2))) as c:
        evs = sse(call(c, body([user("안녕")])).text)
    kinds = [e for e, _ in evs]
    assert kinds[:4] == ["start", "thinking", "delta", "delta"] and kinds[-1] == "error" and "done" not in kinds
    assert evs[-1][1]["code"] == "MODEL_PROVIDER_ERROR" and evs[-1][1]["retryable"] is True


def test_busy_when_every_slot_is_taken():
    app = make(FakeClient())
    with TestClient(app) as c:
        async def hold():
            await app.state.inflight.acquire()
            await app.state.inflight.acquire()
        asyncio.run(hold())
        r = call(c, body([user("안녕")]))
    assert r.status_code == 429 and r.json()["error"]["code"] == "AI_BUSY"


def test_closing_the_stream_early_closes_the_model_call():
    """연결이 끊기면 제너레이터가 닫히고 그 안의 모델 스트림 컨텍스트도 닫힌다 (토큰을 더 쓰지 않는다)."""
    async def run():
        fake = FakeClient(parts=tuple("가" * 50))
        from app.schemas.chat import ChatMessage, Knowledge
        svc = ChatService(fake, "fake-chat", "low")
        agen = svc.stream(Knowledge(**KNOWLEDGE), [ChatMessage(role="user", content="안녕")])
        assert (await agen.__anext__())["type"] == "thinking" and (await agen.__anext__())["type"] == "delta"
        assert "exited" not in fake.log
        await agen.aclose()
        return fake.log
    assert asyncio.run(run()) == {"exited": True}


def test_refusal_is_reported_not_retried_with_another_model():
    with TestClient(make(FakeClient(stop="refusal"))) as c:
        evs = sse(call(c, body([user("안녕")])).text)
        assert evs[-1][0] == "error" and evs[-1][1]["code"] == "INVALID_MODEL_OUTPUT" and evs[-1][1]["retryable"] is False
        r = call(c, body([user("안녕")], stream=False))
        assert r.status_code == 502 and r.json()["error"]["code"] == "INVALID_MODEL_OUTPUT"
