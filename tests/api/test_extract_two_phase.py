"""/extract 의 두 단계 profile: extract-student-1 (빠른 첫 결과, LLM 없음) 과 extract-llm-1 (LLM 이 개념을 직접 뽑음, 기본 꺼짐)."""

import copy
import hashlib
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.api.v1.health import ready_profiles
from app.main import create_app
from app.services.concepts import ConceptExtractor, ExtractConfig
from app.services.concepts.extract import Idf
from app.services.concepts.llm_concepts import ConceptLLM
from app.services.llm import LLMError
from test_api import REQUEST, post, settings

BODY = ("# 데이터베이스 로그 🧠\n\n**WAL**(Write Ahead Logging)은 로그를 먼저 쓰는 규칙이다. WAL 덕분에 복구할 수 있다.\n"
        "체크포인트는 복구 시간을 줄인다. **체크포인트**는 로그를 줄여서 복구한다. 체크포인트 이후의 로그만 적용한다.\n")


class FakeLLM:
    provider, model = "anthropic", "fake-concepts"

    def __init__(self, data=None, error=None):
        self.data, self.error = data, error

    def complete_json(self, system, user, schema, name):
        if self.error:
            raise self.error
        return SimpleNamespace(data=self.data)


def request(profile):
    body = copy.deepcopy(REQUEST)
    body["profileVersion"] = profile
    body["note"]["body"] = BODY
    body["note"]["bodySha256"] = hashlib.sha256(BODY.encode()).hexdigest()
    return body


def app(llm=None):
    return create_app(settings(), extractor=ConceptExtractor(ExtractConfig(), Idf({}, 10, "t")), concept_llm=ConceptLLM(llm) if llm else None)


def test_student_profile_answers_without_any_llm_and_with_a_valid_response():
    with TestClient(app()) as c:
        r = post(c, request("extract-student-1"))
        assert r.status_code == 200, r.text
        d = r.json()
        assert d["meta"]["profileVersion"] == "extract-student-1" and d["meta"]["provider"] == "local"
        assert d["meta"]["modelRevision"].startswith("extract-student-1+") and d["meta"]["durationMs"] < 5000
        assert all(0 <= x["confidence"] <= 1 for x in d["concepts"])
        assert d["concepts"] and all(m["quote"] in BODY for x in d["concepts"] for m in x["mentions"])      # span 은 서버가 원문과 맞는지 이미 확인한다 (어긋나면 502)
        assert post(c, request("extract-student-1")).json()["concepts"] == d["concepts"]      # 같은 입력은 같은 결과


def test_llm_profile_returns_the_concepts_the_model_names_that_are_in_the_note():
    llm = FakeLLM({"concepts": [{"surface": "WAL", "aliases": ["Write Ahead Logging"]}, {"surface": "체크포인트", "aliases": []}, {"surface": "Raft", "aliases": []}]})
    with TestClient(app(llm)) as c:
        r = post(c, request("extract-llm-1"))
        assert r.status_code == 200, r.text
        d = r.json()
        labels = {x["label"] for x in d["concepts"]}
        assert {"WAL", "체크포인트"} <= labels and not any("Raft" in x for x in labels)
        assert d["meta"]["provider"] == "local+anthropic" and d["meta"]["model"].startswith("fake-concepts")
        assert d["meta"]["promptVersion"].startswith("llm-only-concepts-1")
        assert all(x["confidence"] == 1.0 for x in d["concepts"])


def test_llm_profile_is_off_unless_configured_and_the_student_stays_available():
    with TestClient(app()) as c:
        off = post(c, request("extract-llm-1"))
        assert off.status_code == 503 and off.json()["error"]["code"] == "SERVICE_UNAVAILABLE"
        assert "extract-llm-1" not in ready_profiles(c.app.state) and "extract-student-1" in ready_profiles(c.app.state)
    with TestClient(app(FakeLLM({"concepts": []}))) as c:
        assert "extract-llm-1" in ready_profiles(c.app.state)


def test_llm_failures_are_contract_errors_so_the_caller_keeps_the_student_result():
    for kind, status, code in (("refusal", 502, "INVALID_MODEL_OUTPUT"), ("timeout", 504, "MODEL_TIMEOUT"), ("rate_limit", 429, "AI_BUSY"), ("provider", 502, "MODEL_PROVIDER_ERROR")):
        with TestClient(app(FakeLLM(error=LLMError(kind, "x")))) as c:
            r = post(c, request("extract-llm-1"))
            assert (r.status_code, r.json()["error"]["code"]) == (status, code)


def test_both_profiles_yield_relations_from_the_same_rules():
    llm = FakeLLM({"concepts": [{"surface": "WAL", "aliases": []}, {"surface": "체크포인트", "aliases": []}]})
    with TestClient(app(llm)) as c:
        for profile in ("extract-student-1", "extract-llm-1"):
            d = post(c, request(profile)).json()
            refs = {x["ref"] for x in d["concepts"]}
            assert all(r["sourceRef"] in refs and r["targetRef"] in refs for r in d["relations"])


def test_the_concepts_llm_defaults_to_sonnet_unless_a_model_is_pinned(monkeypatch):
    from app.core.config import Settings
    for var in ("AI_CONCEPTS_LLM_MODEL", "AI_LLM_MODEL", "AI_CONCEPTS_LLM_PROVIDER", "AI_LLM_PROVIDER"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("AI_CONCEPTS_LLM", "1")
    with TestClient(create_app(Settings.from_env())) as c:
        assert c.app.state.concept_llm.model == "claude-sonnet-5-5" and "extract-llm-1" in ready_profiles(c.app.state)
    monkeypatch.setenv("AI_CONCEPTS_LLM_MODEL", "claude-opus-5-5")
    with TestClient(create_app(Settings.from_env())) as c:
        assert c.app.state.concept_llm.model == "claude-opus-5-5"
    monkeypatch.delenv("AI_CONCEPTS_LLM", raising=False)
    with TestClient(create_app(Settings.from_env())) as c:
        assert c.app.state.concept_llm is None                         # 기본 꺼짐

