import copy
import hashlib
import json
import math
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.services.embeddings import HashEmbedder

TOKEN = "test-service-token"
STAGE = "332de24b-02f8-5e53-8395-f19b1b89b035"
PROFILE = "embeddings-hash-dev-1"
REQUEST = {
    "context": {
        "analysisJobId": "d551b3d2-8825-5ae9-8e0e-96c4f63b5021", "stageExecutionId": STAGE,
        "userId": "c49bdcd3-9f13-5186-93c2-28eccf386140", "noteId": "10f77e94-6b1f-5e82-b8cd-f6847b0e5479",
        "noteVersion": 3, "inputRevision": 4, "graphRevision": 7, "pipelineVersion": "pipeline-fixture-1",
    },
    "profileVersion": PROFILE,
    "expectedDimensions": 256,
    "items": [{"itemId": "concept-b", "text": "중복: 같은 결과가 두 번 생김"},
              {"itemId": "concept-a", "text": "재시도: 실패한 요청을 다시 보냄"}],
}


@pytest.fixture(scope="module")
def client():
    s = Settings(service_token=TOKEN, concept_idf=Path("does-not-exist.json"))
    with TestClient(create_app(s, embedders={PROFILE: HashEmbedder(dims=256)})) as c:
        yield c


def post(client, body=None):
    raw = json.dumps(body or REQUEST, ensure_ascii=False).encode()
    h = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json", "X-Request-Id": str(uuid.uuid4()),
         "X-Attempt-Id": str(uuid.uuid4()), "Idempotency-Key": STAGE, "X-Input-SHA256": hashlib.sha256(raw).hexdigest()}
    return client.post("/internal/v1/embeddings", content=raw, headers=h)


def mutate(fn):
    body = copy.deepcopy(REQUEST)
    fn(body)
    return body


def test_embeddings_ok(client):
    r = post(client)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["context"] == REQUEST["context"] and data["dimensions"] == 256
    assert [it["itemId"] for it in data["items"]] == ["concept-b", "concept-a"]   # 요청 순서·id 그대로
    for it in data["items"]:
        v = it["vector"]
        assert len(v) == 256 and all(math.isfinite(x) for x in v)
        assert abs(math.sqrt(sum(x * x for x in v)) - 1) < 1e-3
    assert data["meta"]["promptVersion"] is None and data["meta"]["schemaVersion"] == "1.0"


def test_embeddings_deterministic(client):
    assert post(client).json()["items"] == post(client).json()["items"]


@pytest.mark.parametrize("body,status,err", [
    (mutate(lambda b: b.update(expectedDimensions=384)), 422, "UNSUPPORTED_PROFILE"),
    (mutate(lambda b: b.update(profileVersion="nope-1")), 422, "UNSUPPORTED_PROFILE"),
    (mutate(lambda b: b["items"][1].update(itemId="concept-b")), 422, "INVALID_REFERENCE"),
    (mutate(lambda b: b["items"][0].update(text="한")), 422, "INVALID_REFERENCE"),   # NFD "한"
    (mutate(lambda b: b.update(items=[])), 400, "VALIDATION_FAILED"),
    (mutate(lambda b: b["items"][0].update(itemId="bad id!")), 400, "VALIDATION_FAILED"),
    (mutate(lambda b: b["items"][0].update(text="가" * 2001)), 413, "INPUT_TOO_LARGE"),
    (mutate(lambda b: b.update(profileVersion="embeddings-e5-small-1", expectedDimensions=384)), 503,
     "SERVICE_UNAVAILABLE"),                                                     # 등록됐지만 이 배포에서 꺼짐
])
def test_embeddings_errors(client, body, status, err):
    r = post(client, body)
    assert r.status_code == status and r.json()["error"]["code"] == err, r.text


def test_hash_embedder_similarity():
    e = HashEmbedder(dims=256)
    a, b, c = e.embed(["페이지 캐시", "페이지 캐시와 버퍼", "Bellman equation"])
    cos = lambda x, y: sum(p * q for p, q in zip(x, y))
    assert cos(a, b) > cos(a, c)
