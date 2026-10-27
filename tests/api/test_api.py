import copy
import hashlib
import json
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import EXTRACT_PROFILE, Settings
from app.main import create_app
from app.services.concepts import ConceptExtractor, ExtractConfig

TOKEN = "test-service-token"
# 계약 문서(api/ai-api.md §4)의 인공 원문. 재시도는 UTF-16 [10,13)
BODY = "# 지식 🧠\n**재시도**는 중복을 만든다.\n"
STAGE = "81738c89-df53-57b7-9a06-320a4deb50c8"
NOTE_ID = "10f77e94-6b1f-5e82-b8cd-f6847b0e5479"

REQUEST = {
    "context": {
        "analysisJobId": "d551b3d2-8825-5ae9-8e0e-96c4f63b5021",
        "stageExecutionId": STAGE,
        "userId": "c49bdcd3-9f13-5186-93c2-28eccf386140",
        "noteId": NOTE_ID,
        "noteVersion": 3,
        "inputRevision": 4,
        "graphRevision": 7,
        "pipelineVersion": "pipeline-fixture-1",
    },
    "profileVersion": EXTRACT_PROFILE,
    "note": {
        "noteId": NOTE_ID,
        "noteVersion": 3,
        "inputRevision": 4,
        "title": "검증 예시 note-1",
        "body": BODY,
        "bodySha256": hashlib.sha256(BODY.encode()).hexdigest(),
        "topic": {"id": None, "name": "미분류"},
    },
    "policy": {
        "version": "relations-fixture-1",
        "relationTypes": [{"code": "cause", "label": "원인", "description": "예시"}],
        "compositionRules": [],
    },
}


def settings(token=TOKEN, idf="does-not-exist.json"):
    return Settings(service_token=token, concept_idf=Path(idf), max_inflight=4)


@pytest.fixture(scope="module")
def client():
    app = create_app(settings(), extractor=ConceptExtractor(ExtractConfig()))
    with TestClient(app) as c:
        yield c


def headers(raw: bytes, **over):
    h = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json",
         "X-Request-Id": str(uuid.uuid4()), "X-Attempt-Id": str(uuid.uuid4()),
         "Idempotency-Key": STAGE, "X-Input-SHA256": hashlib.sha256(raw).hexdigest()}
    h.update(over)
    return {k: v for k, v in h.items() if v is not None}


def post(client, body=None, raw: bytes | None = None, **hdr):
    raw = raw if raw is not None else json.dumps(body or REQUEST, ensure_ascii=False).encode()
    return client.post("/internal/v1/extract", content=raw, headers=headers(raw, **hdr))


def code(resp):
    return resp.json()["error"]["code"]


def test_contract_fixture_hash():
    # 계약 문서에 적힌 bodySha256 과 같아야 한다
    assert REQUEST["note"]["bodySha256"] == "f74ba80f796eecfc7f65b9f5ac89f7d1976da171cbe6cdcf2311f0a1bf7371d3"


def test_extract_ok(client):
    r = post(client)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["context"] == REQUEST["context"]
    # 계약 문서의 예시 응답과 같은 근거·관계 (api/ai-api.md extractKnowledge)
    assert data["evidence"] == [{"ref": "e-1", "startOffset": 8, "endOffset": 25, "quote": "**재시도**는 중복을 만든다."}]
    refs = {c["label"]: c["ref"] for c in data["concepts"]}
    assert [(r["sourceRef"], r["typeCode"], r["targetRef"], r["evidenceRefs"]) for r in data["relations"]] == [
        (refs["재시도"], "cause", refs["중복"], ["e-1"])]
    assert data["meta"]["schemaVersion"] == "1.0" and data["meta"]["profileVersion"] == EXTRACT_PROFILE
    by_label = {c["label"]: c for c in data["concepts"]}
    assert by_label["재시도"]["mentions"] == [{"startOffset": 10, "endOffset": 13, "quote": "재시도"}]
    body16 = BODY.encode("utf-16-le")
    for c in data["concepts"]:
        assert 0 <= c["confidence"] <= 1
        for m in c["mentions"]:
            assert body16[2 * m["startOffset"]:2 * m["endOffset"]].decode("utf-16-le") == m["quote"]
    assert r.headers["X-Request-Id"]


def test_extract_is_deterministic(client):
    a, b = post(client).json(), post(client).json()
    assert a["concepts"] == b["concepts"]
    assert a["meta"]["modelRevision"] == b["meta"]["modelRevision"]


def test_extract_spans_skip_code_blocks_and_count_utf16(client):
    """§4: 위치는 UTF-16 (이모지 = 2), 코드 블록(들여쓴 것 포함) 안은 추출하지 않지만 위치는 당기지 않는다."""
    body = "# 재시도 🧠🧠\n\n- 예시\n  ```\n  재시도 중복 오류\n  ```\n\n**재시도**는 중복을 만든다. 🧠 중복은 오류를 만든다.\n"
    req = copy.deepcopy(REQUEST)
    req["note"].update(body=body, bodySha256=hashlib.sha256(body.encode()).hexdigest())
    r = post(client, req)
    assert r.status_code == 200, r.text
    data = r.json()
    body16 = body.encode("utf-16-le")
    code_start, code_end = body.index("  ```"), body.rindex("```") + 3
    to16 = lambda i: len(body[:i].encode("utf-16-le")) // 2
    spans = data["evidence"] + [m for c in data["concepts"] for m in c["mentions"]]
    assert spans
    for s in spans:
        assert body16[2 * s["startOffset"]:2 * s["endOffset"]].decode("utf-16-le") == s["quote"]
        assert s["endOffset"] <= to16(code_start) or s["startOffset"] >= to16(code_end)
    again = post(client, req).json()
    assert {k: again[k] for k in ("concepts", "evidence", "relations")} == \
        {k: data[k] for k in ("concepts", "evidence", "relations")}


def test_request_id_echoed(client):
    rid = str(uuid.uuid4())
    assert post(client, **{"X-Request-Id": rid}).headers["X-Request-Id"] == rid


@pytest.mark.parametrize("hdr,status,err", [
    ({"Authorization": None}, 401, "SERVICE_UNAUTHENTICATED"),
    ({"Authorization": "Bearer wrong"}, 401, "SERVICE_UNAUTHENTICATED"),
    ({"X-Request-Id": None}, 400, "VALIDATION_FAILED"),
    ({"X-Attempt-Id": "not-a-uuid"}, 400, "VALIDATION_FAILED"),
    ({"X-Input-SHA256": "0" * 64}, 409, "INPUT_IDENTITY_MISMATCH"),
    ({"Idempotency-Key": str(uuid.uuid4())}, 409, "INPUT_IDENTITY_MISMATCH"),
    ({"Content-Type": "text/plain"}, 415, "UNSUPPORTED_MEDIA_TYPE"),
    ({"Content-Encoding": "gzip"}, 415, "UNSUPPORTED_MEDIA_TYPE"),
    ({"traceparent": "00-" + "0" * 32 + "-" + "1" * 16 + "-01"}, 400, "VALIDATION_FAILED"),
])
def test_header_errors(client, hdr, status, err):
    r = post(client, **hdr)
    assert r.status_code == status and code(r) == err, r.text
    e = r.json()["error"]
    assert uuid.UUID(e["requestId"]) and isinstance(e["retryable"], bool)


def mutate(fn):
    body = copy.deepcopy(REQUEST)
    fn(body)
    return body


@pytest.mark.parametrize("body,status,err", [
    (mutate(lambda b: b.update(extra=1)), 400, "VALIDATION_FAILED"),                      # 추가 key
    (mutate(lambda b: b["context"].pop("userId")), 400, "VALIDATION_FAILED"),              # 필수 누락
    (mutate(lambda b: b["note"].update(noteVersion="3")), 400, "VALIDATION_FAILED"),       # 타입 변환 금지
    (mutate(lambda b: b["note"].update(body="가" * 50001)), 413, "INPUT_TOO_LARGE"),       # 선언된 크기 초과
    (mutate(lambda b: b["context"].update(noteVersion=2)), 422, "INVALID_REFERENCE"),      # context ↔ note
    (mutate(lambda b: b["note"].update(bodySha256="a" * 64)), 422, "INVALID_REFERENCE"),   # 원문 해시
    (mutate(lambda b: b.update(profileVersion="unknown-1")), 422, "UNSUPPORTED_PROFILE"),
    (mutate(lambda b: b["policy"].update(relationTypes=[])), 422, "INVALID_POLICY"),
    (mutate(lambda b: b["policy"].update(compositionRules=[
        {"firstType": "cause", "secondType": "cause", "resultType": "missing_type"}])), 422, "INVALID_POLICY"),
])
def test_body_errors(client, body, status, err):
    r = post(client, body)
    assert r.status_code == status and code(r) == err, r.text


def test_not_nfc_rejected(client):
    import unicodedata
    nfd = unicodedata.normalize("NFD", "# 한글 노트")
    body = mutate(lambda b: b["note"].update(body=nfd, bodySha256=hashlib.sha256(nfd.encode()).hexdigest()))
    r = post(client, body)
    assert r.status_code == 422 and r.json()["error"]["details"] == [{"field": "note.body", "reason": "not_nfc"}]


def test_duplicate_json_key_rejected(client):
    raw = json.dumps(REQUEST, ensure_ascii=False).encode()
    raw = raw.replace(b'"profileVersion"', b'"profileVersion": "x", "profileVersion"', 1)
    r = post(client, raw=raw)
    assert r.status_code == 400 and code(r) == "VALIDATION_FAILED"


def test_details_do_not_leak_values(client):
    r = post(client, mutate(lambda b: b["note"].update(title="")))
    assert r.status_code == 400
    assert "검증 예시" not in r.text and BODY not in r.text


def test_health(client):
    h = {"Authorization": f"Bearer {TOKEN}", "X-Request-Id": str(uuid.uuid4())}
    assert client.get("/internal/v1/health/live", headers=h).json() == {"status": "alive", "schemaVersion": "1.0"}
    ready = client.get("/internal/v1/health/ready", headers=h).json()
    assert ready["status"] == "ready" and EXTRACT_PROFILE in ready["profileVersions"]
    assert client.get("/internal/v1/health/live", headers={"X-Request-Id": str(uuid.uuid4())}).status_code == 401


def test_not_ready_without_extractor(tmp_path):
    # IDF 파일이 없으면 추출기를 만들지 못해 준비되지 않은 상태다
    with TestClient(create_app(settings(idf=tmp_path / "missing.json"), extractor=None)) as c:
        h = {"Authorization": f"Bearer {TOKEN}", "X-Request-Id": str(uuid.uuid4())}
        r = c.get("/internal/v1/health/ready", headers=h)
    assert r.status_code == 503 and code(r) == "SERVICE_UNAVAILABLE" and r.headers["Retry-After"] == "30"


def test_missing_token_config_is_503():
    with TestClient(create_app(settings(token=""), extractor=ConceptExtractor(ExtractConfig()))) as c:
        r = c.get("/internal/v1/health/live", headers={"Authorization": "Bearer x", "X-Request-Id": str(uuid.uuid4())})
    assert r.status_code == 503 and code(r) == "SERVICE_UNAVAILABLE"


def test_concepts_only_profile(client):
    r = post(client, mutate(lambda b: b.update(profileVersion="extract-concepts-det-1")))
    assert r.status_code == 200 and r.json()["evidence"] == [] and r.json()["relations"] == []


def test_relation_needs_matching_policy_code(client):
    # policy 에 cause 계열 code 가 없으면 규칙이 찾은 관계도 내보내지 않는다
    body = mutate(lambda b: b["policy"].update(relationTypes=[{"code": "is_a", "label": "종류", "description": "x"}]))
    r = post(client, body)
    assert r.status_code == 200 and r.json()["relations"] == [] and r.json()["evidence"] == []


def test_hybrid_profile_unavailable_without_llm(client):
    r = post(client, mutate(lambda b: b.update(profileVersion="extract-hybrid-1")))
    assert r.status_code == 503 and code(r) == "SERVICE_UNAVAILABLE"


class FakeTyper:
    provider, model = "fake", "fake-model"
    calls = 0

    def type_pairs(self, items, policy_types):
        from app.services.relations import Typed
        FakeTyper.calls += 1
        return {it.item_id: Typed("cause", True, 0.55) for it in items}


def test_hybrid_profile_uses_typer_for_unresolved_pairs():
    body_text = "# 메모\n지수 백오프 그리고 재시도 정책.\n"
    body = mutate(lambda b: b["note"].update(body=body_text, bodySha256=hashlib.sha256(body_text.encode()).hexdigest()))
    body["profileVersion"] = "extract-hybrid-1"
    app = create_app(settings(), extractor=ConceptExtractor(ExtractConfig()), typer=FakeTyper())
    with TestClient(app) as c:
        r = post(c, body)
    data = r.json()
    assert r.status_code == 200, r.text
    assert data["meta"]["provider"] == "local+fake" and "fake-model" in data["meta"]["model"]
    assert data["relations"] and all(rel["typeCode"] == "cause" for rel in data["relations"])
    assert FakeTyper.calls == 1
