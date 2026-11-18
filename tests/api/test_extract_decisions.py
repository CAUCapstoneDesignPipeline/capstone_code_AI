"""/extract 의 extract-student-decisions-1: 학생 후보를 Decisions 가 판정해 섞는다 (기본 꺼짐, 가짜 클라이언트)."""

import httpx
import openai
from fastapi.testclient import TestClient

from app.api.v1.health import ready_profiles
from app.main import create_app
from app.services.concepts import ConceptExtractor, ExtractConfig
from app.services.concepts.extract import Idf
from app.services.concepts.judge import DecisionsJudge, StudentJudge
from app.services.concepts.student import Student
from test_api import post, settings
from test_extract_two_phase import BODY, request
from fake_decisions import FakeDecisions

PROFILE = "extract-student-decisions-1"


def app(client=None):
    student = Student.load()
    judge = StudentJudge(student, DecisionsJudge(client, "fake-decisions")) if client else None
    return create_app(settings(), extractor=ConceptExtractor(ExtractConfig(), Idf({}, 10, "t")), student=student, student_judge=judge)


def test_the_profile_returns_the_judged_concepts_with_a_valid_response():
    with TestClient(app(FakeDecisions({"WAL": 1.0, "체크포인트": 1.0}))) as c:
        r = post(c, request(PROFILE))
        assert r.status_code == 200, r.text
        d = r.json()
        labels = {x["label"] for x in d["concepts"]}
        assert {"WAL", "체크포인트"} <= labels
        assert d["meta"]["provider"] == "local+openai" and d["meta"]["model"].startswith("fake-decisions")
        assert d["meta"]["promptVersion"].startswith("decisions-1+rules-") and d["meta"]["modelRevision"].startswith("extract-student-1+") and d["meta"]["modelRevision"].endswith("+decisions-1+fake-decisions")
        assert all(0.7 <= x["confidence"] <= 1 for x in d["concepts"]) and all(m["quote"] in BODY for x in d["concepts"] for m in x["mentions"])
        assert post(c, request(PROFILE)).json()["concepts"] == d["concepts"]


def test_the_profile_is_off_unless_configured_and_the_student_stays_available():
    with TestClient(app()) as c:
        off = post(c, request(PROFILE))
        assert off.status_code == 503 and off.json()["error"]["code"] == "SERVICE_UNAVAILABLE"
        assert PROFILE not in ready_profiles(c.app.state) and "extract-student-1" in ready_profiles(c.app.state)
        assert post(c, request("extract-student-1")).status_code == 200
    with TestClient(app(FakeDecisions())) as c:
        assert PROFILE in ready_profiles(c.app.state)


def test_provider_failures_are_contract_errors_so_the_caller_keeps_the_student_result():
    req = httpx.Request("POST", "http://x")
    for error, status, code in ((openai.RateLimitError("x", response=httpx.Response(429, request=req), body=None), 429, "AI_BUSY"),
                                (openai.APITimeoutError(request=req), 504, "MODEL_TIMEOUT"),
                                (openai.InternalServerError("x", response=httpx.Response(500, request=req), body=None), 502, "MODEL_PROVIDER_ERROR")):
        class Failing(FakeDecisions):
            def create(self, model, input, questions, error=error):
                raise error
        with TestClient(app(Failing())) as c:
            r = post(c, request(PROFILE))
            assert (r.status_code, r.json()["error"]["code"]) == (status, code)


def test_relations_come_from_the_same_rules_as_the_other_two_phase_profiles():
    with TestClient(app(FakeDecisions({"WAL": 1.0, "체크포인트": 1.0}))) as c:
        d = post(c, request(PROFILE)).json()
        refs = {x["ref"] for x in d["concepts"]}
        assert all(r["sourceRef"] in refs and r["targetRef"] in refs for r in d["relations"])


def test_the_judge_is_built_from_the_environment(monkeypatch):
    from app.core.config import Settings
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("AI_DECISIONS_JUDGE", raising=False)
    with TestClient(create_app(Settings.from_env())) as c:
        assert c.app.state.student_judge is not None and PROFILE in ready_profiles(c.app.state)       # 키가 있으면 기본 켜짐
    monkeypatch.setenv("AI_DECISIONS_JUDGE", "0")
    with TestClient(create_app(Settings.from_env())) as c:
        assert c.app.state.student_judge is None and PROFILE not in ready_profiles(c.app.state)       # 0 이면 키가 있어도 꺼짐
    monkeypatch.setenv("AI_DECISIONS_JUDGE", "1")
    with TestClient(create_app(Settings.from_env())) as c:
        assert c.app.state.student_judge.model == "gpt-6-luna" and PROFILE in ready_profiles(c.app.state)
    monkeypatch.setenv("AI_DECISIONS_MODEL", "pinned-model")
    with TestClient(create_app(Settings.from_env())) as c:
        assert c.app.state.student_judge.model == "pinned-model"
    monkeypatch.delenv("OPENAI_API_KEY")
    with TestClient(create_app(Settings.from_env())) as c:                                              # 키가 없으면 이 profile 만 준비되지 않는다 (1 이어도)
        assert c.app.state.student_judge is None and "extract-student-1" in ready_profiles(c.app.state)
