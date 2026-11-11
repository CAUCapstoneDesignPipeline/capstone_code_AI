"""챗봇 operation.

  POST /chat    새 지식 카드를 쉽게 풀어 주는 여러 턴 대화. 기본은 SSE(text/event-stream) 로 토큰마다, "stream": false 면 JSON 하나

호출마다 카드(knowledge)와 지금까지의 대화(messages)를 모두 보낸다 (AI 서버는 대화를 저장하지 않는다). 공통 헤더·본문 검사는 다른 operation 과 같고
(Bearer, X-Request-Id, 본문 X-Input-SHA256), 한 번 보내면 되돌릴 수 없는 스트림이라 X-Attempt-Id·Idempotency-Key 는 받지 않는다.

SSE 이벤트 (event: 이름, data: JSON, 모두 t = 시작 뒤 초):
  start(requestId, conversationId, model) · thinking · delta(text) · done(stopReason, usage) · error(code, message, retryable)
첫 글자가 나오기 전의 오류(제공자 인증·한도 등)는 SSE 가 아니라 보통의 계약 오류 응답(HTTP 상태)으로 낸다. 글이 나온 뒤의 오류는 error 이벤트로 끝낸다.
연결을 끊으면 모델 호출도 멈춘다. 입력이 너무 크면 자르지 않고 413 으로 거절한다.
"""

import asyncio
import json
import time

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

from app.core.config import REQUEST_TIMEOUT_S
from app.core.errors import ERRORS, AiError
from app.schemas.chat import ChatRequest
from app.services.llm import LLMError

from .deps import common_checks, parse_model, read_json_body
from .reasoning import LLM_ERRORS

router = APIRouter()
MAX_CHAT_CHARS = 60_000         # 카드 + 대화 전체 글자 수 (모델 입력 한도 안쪽, 넘으면 413)


def _chars(req: ChatRequest) -> int:
    k = req.knowledge
    return (len(k.newKeyword) + len(k.conclusion) + sum(len(e.text) + len(e.keyword) + len(e.note) for e in k.evidence)
            + sum(len(e.url) + len(e.title or "") + len(e.quote or "") for e in k.external) + sum(len(m.content) for m in req.messages))


def _ai_error(e: LLMError) -> AiError:
    code, retry = LLM_ERRORS.get(e.kind, ("MODEL_PROVIDER_ERROR", None))
    return AiError(code, retry_after=retry)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/chat", operation_id="chatAboutKnowledge")        # 계약(docs api/ai-openapi.yaml)의 operationId
async def chat(request: Request):
    t0 = time.monotonic()
    common_checks(request)
    req = parse_model(ChatRequest, await read_json_body(request))
    if _chars(req) > MAX_CHAT_CHARS:
        raise AiError("INPUT_TOO_LARGE", [("messages", "too_large")])
    state = request.app.state
    svc = getattr(state, "chat", None)
    if svc is None:
        raise AiError("SERVICE_UNAVAILABLE", retry_after=30)
    if state.inflight.locked():
        raise AiError("AI_BUSY", retry_after=1)

    if not req.stream:
        async with state.inflight:
            try:
                async with asyncio.timeout(REQUEST_TIMEOUT_S):
                    out = await svc.complete(req.knowledge, req.messages)
            except TimeoutError:
                raise AiError("MODEL_TIMEOUT")
            except LLMError as e:
                raise _ai_error(e)
        return JSONResponse({**out, "conversationId": req.conversationId, "elapsedMs": int((time.monotonic() - t0) * 1000)})

    await state.inflight.acquire()                              # 스트림이 끝날 때까지 한 자리를 쓴다
    deadline = asyncio.get_running_loop().time() + REQUEST_TIMEOUT_S      # 첫 글자를 기다리는 시간과 스트림 전체가 한 마감을 나눠 쓴다 (BE 호출 제한 130초 안쪽)
    events = svc.stream(req.knowledge, req.messages)
    try:                                                         # 첫 이벤트까지 받아 봐서, 시작도 못 한 오류는 보통의 HTTP 오류로 낸다
        async with asyncio.timeout_at(deadline):
            first = await events.__anext__()
    except TimeoutError:
        state.inflight.release()
        raise AiError("MODEL_TIMEOUT")
    except LLMError as e:
        state.inflight.release()
        raise _ai_error(e)
    except BaseException:
        state.inflight.release()
        raise
    rid = request.state.request_id
    held = {"slot": True}

    async def cleanup():                                         # 한 번만: 모델 호출을 닫고 자리를 돌려준다 (본문이 시작도 못 하고 끊겨도 background 로 여기에 온다)
        if held["slot"]:
            held["slot"] = False
            try:
                await events.aclose()
            finally:
                state.inflight.release()

    async def body():
        stamp = lambda ev: {**ev, "t": round(time.monotonic() - t0, 2)}
        try:
            yield _sse("start", stamp({"requestId": rid, "conversationId": req.conversationId, "model": svc.model}))
            yield _sse(first["type"], stamp({k: v for k, v in first.items() if k != "type"}))
            try:
                async with asyncio.timeout_at(deadline):
                    async for ev in events:
                        yield _sse(ev["type"], stamp({k: v for k, v in ev.items() if k != "type"}))
            except TimeoutError:
                yield _sse("error", stamp({"code": "MODEL_TIMEOUT", "message": ERRORS["MODEL_TIMEOUT"][2], "retryable": True}))
            except LLMError as e:
                err = _ai_error(e)
                yield _sse("error", stamp({"code": err.code, "message": ERRORS[err.code][2], "retryable": ERRORS[err.code][1]}))
        finally:                                                 # 연결이 끊겨도 여기로 온다
            await cleanup()

    return StreamingResponse(body(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                             background=BackgroundTask(cleanup))
