"""POST /external-knowledge: 성공(found · none), 공통 헤더·본문 검사, 꺼져 있으면 503, 모르는 profile, 한 자리, 시간 제한, 응답 자체 검증."""

import hashlib
import json
import threading
import time
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.services.links.external import Found
from app.services.llm import LLMError

TOKEN = "t"
PROFILE = "external-knowledge-llm-1"
CONTEXT = {"analysisJobId": "d551b3d2-8825-5ae9-8e0e-96c4f63b5021", "stageExecutionId": "6a1e1a8e-9d3f-4d8b-a5ad-1d1e2d5a7c11",
           "userId": "c49bdcd3-9f13-5186-93c2-28eccf386140", "noteId": "10f77e94-6b1f-5e82-b8cd-f6847b0e5479", "noteVersion": 3,
           "inputRevision": 4, "graphRevision": 7, "pipelineVersion": "pipeline-fixture-1"}
CLAIM = {"conclusion": "WAL 덕분에 장애가 나도 되돌릴 정보가 로그에 남아서 트랜잭션을 통째로 취소할 수 있어요.",
         "evidence": [{"label": "A", "keyword": "WAL", "text": "WAL은 로그를 먼저 디스크에 쓴다."},
                      {"label": "B", "keyword": "Atomicity", "text": "Atomicity는 전부 반영되거나 전혀 반영되지 않는 성질이다."}]}
ITEM = {"summary": "로그에 되돌릴 정보를 먼저 남겨 롤백하는 방법을 설명해요.", "sourceUrl": "https://example.com/aries", "title": "ARIES recovery",
        "retrievedAt": "2026-11-11T09:00:00Z", "support": "yes"}
ITEM2 = {**ITEM, "sourceUrl": "https://example.org/wal", "title": None, "support": "partly"}


class FakeFinder:
    model = "fake-gen"

    def __init__(self, found=None, error=None, delay=0.0):
        self.found = found if found is not None else Found("found", [dict(ITEM)], 1, 4190, 1080, "anthropic", "fake-gen")
        self.error, self.delay, self.calls = error, delay, []

    def find(self, claim, max_sources=1):
        self.calls.append((claim, max_sources))
        time.sleep(self.delay)
        if self.error:
            raise self.error
        return self.found


def make(finder=None, **settings):
    return create_app(Settings(service_token=TOKEN, concept_idf=Path("none.json"), **settings), external_finder=finder)


def body(**over):
    return {"context": dict(CONTEXT), "profileVersion": PROFILE, "claim": json.loads(json.dumps(CLAIM)), **over}


def post(c, payload, **hdr):
    raw = json.dumps(payload, ensure_ascii=False).encode()
    h = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json", "X-Request-Id": str(uuid.uuid4()), "X-Attempt-Id": str(uuid.uuid4()),
         "Idempotency-Key": payload["context"]["stageExecutionId"], "X-Input-SHA256": hashlib.sha256(raw).hexdigest(), **hdr}
    return c.post("/internal/v1/external-knowledge", content=raw, headers={k: v for k, v in h.items() if v is not None})


def err(r):
    return r.json()["error"]["code"]


def test_found_returns_the_card_fields_and_usage():
    f = FakeFinder()
    with TestClient(make(f)) as c:
        r = post(c, body())
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["status"] == "found" and d["externalKnowledge"] == [ITEM] and d["usage"] == {"searches": 1, "inputTokens": 4190, "outputTokens": 1080}
    assert d["context"] == CONTEXT and d["meta"]["profileVersion"] == PROFILE and d["meta"]["model"] == "fake-gen" and d["meta"]["provider"] == "anthropic"
    assert d["meta"]["promptVersion"] == PROFILE and d["meta"]["modelRevision"].startswith("fake-gen@")
    assert f.calls[0][1] == 1 and f.calls[0][0]["conclusion"] == CLAIM["conclusion"]         # maxSources 기본 1


def test_max_sources_is_passed_through_and_a_title_can_be_null():
    f = FakeFinder(Found("found", [dict(ITEM), dict(ITEM2)], 1, 10, 20, "anthropic", "fake-gen"))
    with TestClient(make(f)) as c:
        r = post(c, body(maxSources=3))
    assert r.status_code == 200 and f.calls[0][1] == 3
    assert [i["support"] for i in r.json()["externalKnowledge"]] == ["yes", "partly"] and r.json()["externalKnowledge"][1]["title"] is None


def test_none_is_a_200_with_an_empty_list():
    with TestClient(make(FakeFinder(Found("none", [], 1, 100, 50, "anthropic", "fake-gen")))) as c:
        r = post(c, body())
    assert r.status_code == 200 and r.json()["status"] == "none" and r.json()["externalKnowledge"] == []


def test_it_is_off_by_default_and_readiness_lists_it_only_when_the_finder_exists():
    from app.api.v1.health import ready_profiles
    with TestClient(make()) as c:
        r = post(c, body())
        assert r.status_code == 503 and err(r) == "SERVICE_UNAVAILABLE"
        assert PROFILE not in ready_profiles(c.app.state)
    with TestClient(make(FakeFinder())) as c:
        assert PROFILE in ready_profiles(c.app.state)
    assert Settings.from_env().external_llm is False


def test_the_flag_builds_the_finder_only_when_on_and_a_bad_config_leaves_it_unready(monkeypatch):
    monkeypatch.setenv("AI_EXTERNAL_LLM", "1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("AI_EXTERNAL_LLM_MODEL", "claude-sonnet-5-5")
    app = create_app(Settings.from_env())
    with TestClient(app) as c:
        assert c.app.state.external_finder is not None and PROFILE in __import__("app.api.v1.health", fromlist=["x"]).ready_profiles(c.app.state)
    monkeypatch.setenv("AI_EXTERNAL_LLM_PROVIDER", "openai")             # 웹 검색은 anthropic 만
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    with TestClient(create_app(Settings.from_env())) as c:
        assert c.app.state.external_finder is None


@pytest.mark.parametrize("edit, status, code", [
    (lambda b: b.update(extra=1), 400, "VALIDATION_FAILED"),
    (lambda b: b.pop("claim"), 400, "VALIDATION_FAILED"),
    (lambda b: b["claim"].update(conclusion=""), 400, "VALIDATION_FAILED"),
    (lambda b: b["claim"].update(evidence=[]), 400, "VALIDATION_FAILED"),
    (lambda b: b["claim"]["evidence"][0].update(label="C"), 400, "VALIDATION_FAILED"),
    (lambda b: b.update(maxSources=0), 400, "VALIDATION_FAILED"),
    (lambda b: b.update(maxSources=4), 400, "VALIDATION_FAILED"),
    (lambda b: b.update(profileVersion="nope"), 422, "UNSUPPORTED_PROFILE"),
    (lambda b: b.update(profileVersion="explain-llm-8"), 422, "UNSUPPORTED_PROFILE")])      # 다른 operation 의 profile
def test_invalid_requests(edit, status, code):
    b = body()
    edit(b)
    with TestClient(make(FakeFinder())) as c:
        r = post(c, b)
    assert (r.status_code, err(r)) == (status, code)


def test_the_common_headers_are_enforced():
    f = FakeFinder()
    with TestClient(make(f)) as c:
        assert post(c, body(), Authorization=None).status_code == 401
        assert post(c, body(), Authorization="Bearer wrong").status_code in (401, 403)
        assert post(c, body(), **{"X-Attempt-Id": None}).status_code == 400
        assert post(c, body(), **{"Idempotency-Key": str(uuid.uuid4())}).status_code == 409        # stageExecutionId 와 다르다
        assert post(c, body(), **{"X-Input-SHA256": "0" * 64}).status_code == 409
    assert f.calls == []                                                                           # 검사에서 막히면 모델을 부르지 않는다


@pytest.mark.parametrize("found", [
    Found("found", [], 1, 0, 0),                                                    # found 인데 항목이 없다
    Found("none", [dict(ITEM)], 1, 0, 0),                                          # none 인데 항목이 있다
    Found("found", [dict(ITEM), dict(ITEM2)], 1, 0, 0),                            # maxSources(기본 1) 를 넘는다
    Found("found", [dict(ITEM, sourceUrl="http://localhost/x")], 1, 0, 0),         # 안전하지 않은 URL
    Found("found", [dict(ITEM, summary="가" * 301)], 1, 0, 0)])                    # 계약 길이 초과 → 잘라 내지 않고 오류
def test_a_result_that_breaks_the_contract_is_a_502_not_a_success(found):
    found.provider, found.model = "anthropic", "fake-gen"
    with TestClient(make(FakeFinder(found))) as c:
        r = post(c, body())
    assert r.status_code in (502, 500) and err(r) in ("INVALID_MODEL_OUTPUT", "OUTPUT_LIMIT_EXCEEDED")


def test_duplicate_urls_and_a_weaker_source_first_are_502():
    with TestClient(make(FakeFinder(Found("found", [dict(ITEM), dict(ITEM)], 1, 0, 0, "anthropic", "m")))) as c:
        assert post(c, body(maxSources=3)).status_code == 502
    with TestClient(make(FakeFinder(Found("found", [dict(ITEM2), dict(ITEM)], 1, 0, 0, "anthropic", "m")))) as c:
        assert post(c, body(maxSources=3)).status_code == 502


@pytest.mark.parametrize("kind, status, code", [("timeout", 504, "MODEL_TIMEOUT"), ("rate_limit", 429, "AI_BUSY"), ("auth", 503, "SERVICE_UNAVAILABLE"),
                                                ("provider", 502, "MODEL_PROVIDER_ERROR"), ("invalid_output", 502, "INVALID_MODEL_OUTPUT")])
def test_model_errors_map_to_contract_errors(kind, status, code):
    with TestClient(make(FakeFinder(error=LLMError(kind, "x")))) as c:
        r = post(c, body())
    assert (r.status_code, err(r)) == (status, code)


def test_a_second_call_while_one_runs_is_429_busy():
    f = FakeFinder(delay=0.6)
    with TestClient(make(f, max_inflight=1)) as c:
        out = {}
        t = threading.Thread(target=lambda: out.update(first=post(c, body())))
        t.start()
        time.sleep(0.25)
        second = post(c, body())
        t.join()
    assert second.status_code == 429 and err(second) == "AI_BUSY" and out["first"].status_code == 200
