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
    assert data["evidence"] == [] and data["relations"] == []
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
