"""챗봇의 실제 응답이 내부 계약(docs 저장소 api/ai-openapi.yaml 의 /chat)과 같은가. 계약 파일이 없으면 건너뛴다.

PyYAML 이 없거나 계약 파일이 없으면 건너뛴다. 계약 파일 위치: 환경 변수 AI_CONTRACT_YAML, 없으면 이 저장소 옆의 docs/ · capstone_docs/ 의 api/ai-openapi.yaml.
jsonschema 에 기대지 않고 필요한 만큼(type·required·properties·additionalProperties·enum·길이·개수·pattern·$ref)만 직접 검사한다.
"""

import os
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

yaml = pytest.importorskip("yaml")        # PyYAML 이 없는 환경(CI 의 잠긴 의존성)에서는 이 모듈을 건너뛴다

from test_chat import TOKEN, FakeClient, call, make, sse

ROOT = Path(__file__).resolve().parents[2]


def _find() -> Path | None:
    for c in (os.environ.get("AI_CONTRACT_YAML"), ROOT.parent / "docs/api/ai-openapi.yaml", ROOT.parent / "capstone_docs/api/ai-openapi.yaml"):
        if c and Path(c).exists():
            return Path(c)
    return None


PATH = _find()
SPEC = yaml.safe_load(PATH.read_text(encoding="utf-8")) if PATH else None
pytestmark = pytest.mark.skipif(not SPEC or "/chat" not in SPEC["paths"], reason="계약 파일(api/ai-openapi.yaml 의 /chat)이 없다")


def ref(r):
    node = SPEC
    for k in r[2:].split("/"):
        node = node[k]
    return node


def conforms(schema, value, where="$"):
    """계약 스키마 하나에 값이 맞는지. 어긋나면 AssertionError (어디가 어떻게)."""
    if "$ref" in schema:
        return conforms(ref(schema["$ref"]), value, where)
    t = schema.get("type")
    types = t if isinstance(t, list) else [t] if t else []
    py = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "array": list, "object": dict, "null": type(None)}
    if types:
        assert any(isinstance(value, py[x]) and not (x in ("integer", "number") and isinstance(value, bool)) for x in types), f"{where}: type {types} <- {value!r:.60}"
    if "enum" in schema:
        assert value in schema["enum"], f"{where}: {value!r} not in {schema['enum']}"
    if isinstance(value, str):
        assert len(value) >= schema.get("minLength", 0) and len(value) <= schema.get("maxLength", 10**9), f"{where}: length {len(value)}"
        if "pattern" in schema:
            assert re.search(schema["pattern"], value), f"{where}: pattern"
        if schema.get("format") == "uuid":
            assert re.fullmatch(r"[0-9a-fA-F-]{36}", value), f"{where}: uuid"
    if isinstance(value, (int, float)) and not isinstance(value, bool) and "minimum" in schema:
        assert value >= schema["minimum"], f"{where}: minimum"
    if isinstance(value, list):
        assert schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", 10**9), f"{where}: items {len(value)}"
        for i, v in enumerate(value):
            conforms(schema.get("items", {}), v, f"{where}[{i}]")
    if isinstance(value, dict):
        props = schema.get("properties", {})
        for k in schema.get("required", []):
            assert k in value, f"{where}: missing {k}"
        if schema.get("additionalProperties") is False:
            assert set(value) <= set(props), f"{where}: unexpected {sorted(set(value) - set(props))}"
        for k, v in value.items():
            if k in props:
                conforms(props[k], v, f"{where}.{k}")
    return True


OP = SPEC["paths"]["/chat"]["post"] if SPEC and "/chat" in SPEC["paths"] else None
REQ_EX = OP["requestBody"]["content"]["application/json"]["examples"]["normal"]["value"] if OP else None


def post(c, payload):
    return call(c, payload)


def test_operation_id_and_required_headers_match_the_contract():
    from app.core.config import Settings
    from app.main import create_app
    from pathlib import Path as P
    ai = create_app(Settings(service_token="t", concept_idf=P("none.json"))).openapi()["paths"]["/internal/v1/chat"]["post"]
    assert ai["operationId"] == OP["operationId"]
    names = [ref(p["$ref"])["name"] for p in OP["parameters"]]
    assert "X-Attempt-Id" not in names and "Idempotency-Key" not in names and {"X-Request-Id", "X-Input-SHA256"} <= set(names)


def test_the_contracts_own_request_example_is_accepted_and_the_examples_match_the_schema():
    conforms(OP["requestBody"]["content"]["application/json"]["schema"], REQ_EX)
    with TestClient(make(FakeClient())) as c:
        r = post(c, REQ_EX)
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")


@pytest.mark.parametrize("change", [lambda r: r.update(extra=1), lambda r: r.update(messages=[]),
                                    lambda r: r["knowledge"]["evidence"][0].update(label="C"),
                                    lambda r: r["messages"][-1].update(role="assistant")])                 # 순서 규칙 (스키마 밖): AI 가 400 으로 거절
def test_requests_the_contract_forbids_are_rejected_with_400(change):
    import copy
    bad = copy.deepcopy(REQ_EX)
    change(bad)
    with TestClient(make(FakeClient())) as c:
        r = post(c, bad)
        assert r.status_code == 400 and r.json()["error"]["code"] == "VALIDATION_FAILED"


def test_a_message_over_4000_characters_is_413_not_400():
    """스키마로는 maxLength 위반이지만 AI 는 '입력이 너무 크다' (413 INPUT_TOO_LARGE) 로 답한다. 계약 설명도 같다."""
    import copy
    bad = copy.deepcopy(REQ_EX)
    bad["messages"][0]["content"] = "가" * 4001
    with TestClient(make(FakeClient())) as c:
        r = post(c, bad)
    assert r.status_code == 413 and r.json()["error"]["code"] == "INPUT_TOO_LARGE"
    assert "4,000자" in OP["description"] and "413" in OP["description"]


def test_every_sse_event_the_server_sends_has_exactly_the_contract_fields():
    names = OP["responses"]["200"]["content"]["text/event-stream"]["x-sse-events"]
    with TestClient(make(FakeClient(parts=("쉽게 ", "말해 ", "볼게요.")))) as c:
        evs = sse(post(c, {**REQ_EX, "stream": True}).text)
    assert [e for e, _ in evs][0] == "start" and [e for e, _ in evs][-1] == "done"
    for name, data in evs:
        schema = ref(names[name])
        conforms(schema, data, name)
        assert set(data) == set(schema["properties"]), f"{name}: 계약과 다른 필드 {set(data) ^ set(schema['properties'])}"
    t = [d["t"] for _, d in evs]
    assert t == sorted(t)
    assert "".join(d["text"] for e, d in evs if e == "delta") == "쉽게 말해 볼게요."


def test_a_stream_that_fails_after_the_first_text_ends_with_a_contract_error_event():
    names = OP["responses"]["200"]["content"]["text/event-stream"]["x-sse-events"]
    with TestClient(make(FakeClient(parts=("가", "나", "다"), fail_after=2))) as c:
        evs = sse(post(c, REQ_EX).text)
    name, data = evs[-1]
    assert name == "error" and conforms(ref(names["error"]), data) and "done" not in [e for e, _ in evs]


def test_the_non_streaming_response_matches_the_contract():
    schema = OP["responses"]["200"]["content"]["application/json"]["schema"]
    with TestClient(make(FakeClient())) as c:
        r = post(c, {**REQ_EX, "stream": False})
    assert r.status_code == 200
    conforms(schema, r.json())
    assert set(r.json()) == set(ref(schema["$ref"])["properties"])


def test_errors_before_the_first_text_are_the_contract_error_responses():
    """계약이 선언한 오류 응답(HTTP 상태)과 실제가 같다: 400 · 401 · 413 · 429 · 503."""
    import asyncio
    from app.core.config import Settings
    from app.main import create_app
    declared = {int(k) for k in OP["responses"] if k != "200"}
    seen = {}
    big = {**REQ_EX, "messages": [{"role": "user" if i % 2 == 0 else "assistant", "content": "가" * 4000} for i in range(15)]}
    with TestClient(make(FakeClient())) as c:
        seen[413] = post(c, big).status_code                                                                 # 합쳐서 60,000자 초과
        seen[400] = post(c, {**REQ_EX, "stream": "yes"}).status_code
        seen[401] = c.post("/internal/v1/chat", content=b"{}", headers={"Content-Type": "application/json"}).status_code
    app = make(FakeClient())
    with TestClient(app) as c:
        async def hold():
            await app.state.inflight.acquire()
            await app.state.inflight.acquire()
        asyncio.run(hold())
        seen[429] = post(c, REQ_EX).status_code                                                              # 처리 자리가 가득 찼다
    off = create_app(Settings(service_token=TOKEN, concept_idf=Path("none.json"), max_inflight=2))          # 챗봇이 꺼져 있다 (기본)
    with TestClient(off) as c:
        r = post(c, REQ_EX)
        seen[503] = r.status_code
        assert r.json()["error"]["code"] == "SERVICE_UNAVAILABLE"
    assert all(seen[k] == k for k in seen), seen
    assert set(seen) <= declared, set(seen) - declared
