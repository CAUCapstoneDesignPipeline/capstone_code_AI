"""/extract 의 기본 파이프라인 (profileVersion "extract-default") 과 그것을 바꾸는 GET·PUT·DELETE /pipeline."""

import uuid
from dataclasses import replace

from fake_decisions import FakeDecisions, FakePairs, tiny_model
from fastapi.testclient import TestClient
from test_api import TOKEN, post, settings
from test_extract_pairs import app as both_app
from test_extract_pairs import req
from test_extract_two_phase import app as plain_app
from test_extract_two_phase import request

from app.core.config import DEFAULT_ALIAS, Settings

BOTH, STUDENT = "extract-student-decisions-pairs-1", "extract-student-1"
JUDGE = {"WAL": 1.0, "체크포인트": 1.0}


def hdr():
    return {"Authorization": f"Bearer {TOKEN}", "X-Request-Id": str(uuid.uuid4())}


def full():
    return both_app(FakePairs(default={"none": 0.1, "is_a_ab": 0.9}), FakeDecisions(JUDGE))


def test_the_default_is_the_new_pipeline_when_decisions_are_ready_and_the_student_when_not():
    with TestClient(full()) as c:
        p = c.get("/internal/v1/pipeline", headers=hdr()).json()
        assert p["alias"] == DEFAULT_ALIAS and p["default"] == BOTH and p["source"] == "auto"
        assert {x["profileVersion"]: x["default"] for x in p["profiles"] if x["default"]} == {BOTH: True}
        by = {x["profileVersion"]: x for x in p["profiles"]}
        assert by[BOTH]["ready"] and by[BOTH]["externalProvider"] == "openai" and by[STUDENT]["externalProvider"] == "none"
    with TestClient(plain_app()) as c:                       # 키·Decisions 가 없으면 OpenAI 없이 도는 학생으로 물러난다
        p = c.get("/internal/v1/pipeline", headers=hdr()).json()
        assert p["default"] == STUDENT and p["source"] == "auto"
        assert not {x["profileVersion"]: x for x in p["profiles"]}[BOTH]["ready"]


def test_the_alias_runs_the_default_and_reports_the_profile_that_ran():
    with TestClient(full()) as c:
        r = post(c, req(DEFAULT_ALIAS))
        assert r.status_code == 200, r.text
        d = r.json()
        assert d["meta"]["profileVersion"] == BOTH and d["meta"]["provider"] == "local+openai" and d["relations"]
        direct = post(c, req(BOTH)).json()
        assert (d["concepts"], d["relations"]) == (direct["concepts"], direct["relations"])      # 별명은 같은 profile 이다
    with TestClient(plain_app()) as c:
        r = post(c, request(DEFAULT_ALIAS))
        assert r.status_code == 200 and r.json()["meta"]["profileVersion"] == STUDENT and r.json()["meta"]["provider"] == "local"


def test_switching_changes_only_the_alias():
    with TestClient(full()) as c:
        r = c.put(f"/internal/v1/pipeline/{STUDENT}", headers=hdr())
        assert r.status_code == 200 and r.json()["default"] == STUDENT and r.json()["source"] == "override"
        assert post(c, req(DEFAULT_ALIAS)).json()["meta"]["profileVersion"] == STUDENT
        assert post(c, req(BOTH)).json()["meta"]["profileVersion"] == BOTH                       # 직접 부른 profile 은 그대로
        r = c.delete("/internal/v1/pipeline", headers=hdr())
        assert r.json()["default"] == BOTH and r.json()["source"] == "auto"
        assert post(c, req(DEFAULT_ALIAS)).json()["meta"]["profileVersion"] == BOTH


def test_switching_to_something_that_is_not_ready_or_unknown_is_refused_and_keeps_the_default():
    with TestClient(plain_app()) as c:
        r = c.put(f"/internal/v1/pipeline/{BOTH}", headers=hdr())
        assert r.status_code == 503 and r.json()["error"]["code"] == "SERVICE_UNAVAILABLE"
        r = c.put("/internal/v1/pipeline/extract-nope-1", headers=hdr())
        assert r.status_code == 422 and r.json()["error"]["code"] == "UNSUPPORTED_PROFILE"
        r = c.put(f"/internal/v1/pipeline/{DEFAULT_ALIAS}", headers=hdr())                       # 별명을 별명으로 둘 수 없다
        assert r.status_code == 422
        assert c.get("/internal/v1/pipeline", headers=hdr()).json()["default"] == STUDENT


def test_the_pipeline_calls_need_the_service_token():
    with TestClient(full()) as c:
        for call in (lambda h: c.get("/internal/v1/pipeline", headers=h), lambda h: c.put(f"/internal/v1/pipeline/{STUDENT}", headers=h),
                     lambda h: c.delete("/internal/v1/pipeline", headers=h)):
            assert call({"X-Request-Id": str(uuid.uuid4())}).status_code == 401
            assert call({"Authorization": "Bearer wrong", "X-Request-Id": str(uuid.uuid4())}).status_code == 401
        assert c.get("/internal/v1/pipeline", headers=hdr()).json()["source"] == "auto"


def test_a_pinned_default_that_is_not_ready_fails_loudly_instead_of_running_something_else():
    s = replace(settings(), pipeline=BOTH)                       # AI_PIPELINE 로 못박았는데 Decisions 가 없다
    from app.main import create_app
    from app.services.concepts import ConceptExtractor, ExtractConfig
    from app.services.concepts.extract import Idf
    from app.services.concepts.student import Student
    with TestClient(create_app(s, extractor=ConceptExtractor(ExtractConfig(), Idf({}, 10, "t")), student=Student.load())) as c:
        assert c.get("/internal/v1/pipeline", headers=hdr()).json()["source"] == "env"
        r = post(c, request(DEFAULT_ALIAS))
        assert r.status_code == 503 and r.json()["error"]["code"] == "SERVICE_UNAVAILABLE"
    bad = replace(settings(), pipeline="extract-nope-1")
    with TestClient(create_app(bad, extractor=ConceptExtractor(ExtractConfig(), Idf({}, 10, "t")), student=Student.load())) as c:
        assert post(c, request(DEFAULT_ALIAS)).status_code == 422


def test_decisions_are_on_by_default_only_when_there_is_an_openai_key(monkeypatch):
    monkeypatch.delenv("AI_DECISIONS_JUDGE", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert Settings.from_env().decisions_judge is False
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert Settings.from_env().decisions_judge is True
    monkeypatch.setenv("AI_DECISIONS_JUDGE", "0")                # 키가 있어도 끌 수 있다
    assert Settings.from_env().decisions_judge is False
    monkeypatch.delenv("OPENAI_API_KEY")
    monkeypatch.setenv("AI_DECISIONS_JUDGE", "1")
    assert Settings.from_env().decisions_judge is True
    monkeypatch.setenv("AI_PIPELINE", BOTH)
    assert Settings.from_env().pipeline == BOTH
