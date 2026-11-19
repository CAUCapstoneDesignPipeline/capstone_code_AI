"""학생 + Decisions 판정: 후보 판정 확률을 학생 확률과 섞어 개념을 고르고, 오류는 LLMError 로 바꾼다 (Decisions 는 가짜 클라이언트)."""

import threading

import httpx
import openai
import pytest

from app.services.concepts import ConceptExtractor, ExtractConfig
from app.services.concepts.extract import Idf
from app.services.concepts.judge import MAX_CANDIDATES, THRESHOLD, WEIGHT, DecisionsJudge, StudentJudge
from app.services.concepts.student import Student
from app.services.llm import LLMError
from fake_decisions import FakeDecisions

BODY = ("# 데이터베이스 로그\n\n**WAL**(Write Ahead Logging)은 로그를 먼저 쓰는 규칙이다. WAL 덕분에 복구할 수 있다.\n"
        "체크포인트는 복구 시간을 줄인다. **체크포인트**는 로그를 줄여서 복구한다. 체크포인트 이후의 로그만 적용한다.\n")


def extractor():
    return ConceptExtractor(ExtractConfig(), Idf({}, 10, "t"))


def judge(client):
    return StudentJudge(Student.load(), DecisionsJudge(client, "fake-decisions"))


def test_the_judged_concepts_are_kept_and_the_rest_dropped():
    fake = FakeDecisions({"WAL": 1.0, "체크포인트": 1.0})
    out = judge(fake).select(extractor(), "n", 1, BODY)
    assert {"wal", "체크포인트"} <= set(out) and all(m.surface in BODY for ms in out.values() for m in ms)
    assert all(m.score >= 2 * THRESHOLD for ms in out.values() for m in ms) and all(m.score <= 2.0 for ms in out.values() for m in ms)


def test_a_candidate_the_judge_rejects_is_dropped_even_if_the_student_likes_it():
    fake = FakeDecisions({})
    assert judge(fake).select(extractor(), "n", 1, BODY) == {}                  # 0.3 × 학생 확률은 임계값에 못 닿는다 (학생 혼자서는 통과할 후보도)
    assert WEIGHT * 0.0 + (1 - WEIGHT) * 1.0 < THRESHOLD


def test_two_requests_go_out_once_per_note_with_one_question_per_candidate():
    fake = FakeDecisions({"WAL": 1.0})
    judge(fake).select(extractor(), "n", 1, BODY)
    assert len(fake.calls) == 2 and {c[0] for c in fake.calls} == {"fake-decisions"} and all(c[1] == BODY for c in fake.calls)
    assert {q["type"] for c in fake.calls for q in c[2]} == {"score", "predicate"}
    assert len(fake.calls[0][2]) == len(fake.calls[1][2]) and 0 < len(fake.calls[0][2]) <= MAX_CANDIDATES


def test_the_formula_is_the_measured_blend():
    fake = FakeDecisions({"WAL": 0.9})
    j = judge(fake)
    out = j.select(extractor(), "n", 1, BODY)
    from app.services.concepts.student import rows
    text, rs = rows(extractor(), BODY, symbols=True)
    p = {r["key"]: float(q) for r, q in zip(rs, j.student.probabilities(__import__("numpy").array([r["x"] for r in rs], float)))}
    expected = round((1 - WEIGHT) * round(p["wal"], 6) + WEIGHT * 0.9, 4)
    assert out["wal"][0].score == round(2 * expected, 4)


def test_a_refused_candidate_counts_as_not_a_concept():
    fake = FakeDecisions({"WAL": 1.0, "체크포인트": 1.0}, refuse={"WAL"})
    out = judge(fake).select(extractor(), "n", 1, BODY)
    assert "wal" not in out and "체크포인트" in out


def test_the_morphology_lock_is_free_while_the_network_call_runs():
    lock = threading.Lock()
    fake = FakeDecisions({"WAL": 1.0}, lock=lock)
    judge(fake).select(extractor(), "n", 1, BODY, lock=lock)
    assert fake.lock_free == [True, True] and not lock.locked()


def test_an_empty_note_makes_no_request():
    fake = FakeDecisions()
    assert judge(fake).select(extractor(), "n", 1, "") == {} and fake.calls == []


def test_the_result_is_deterministic():
    a = judge(FakeDecisions({"WAL": 1.0, "체크포인트": 0.95})).select(extractor(), "n", 1, BODY)
    b = judge(FakeDecisions({"WAL": 1.0, "체크포인트": 0.95})).select(extractor(), "n", 1, BODY)
    assert a == b and list(a) == list(b)


def _status_error(cls, code):
    return cls("x", response=httpx.Response(code, request=httpx.Request("POST", "http://x")), body=None)


@pytest.mark.parametrize("error,kind", [
    (_status_error(openai.RateLimitError, 429), "rate_limit"),
    (openai.APITimeoutError(request=httpx.Request("POST", "http://x")), "timeout"),
    (_status_error(openai.AuthenticationError, 401), "auth"),
    (_status_error(openai.InternalServerError, 500), "provider"),
    (openai.APIConnectionError(request=httpx.Request("POST", "http://x")), "provider"),
])
def test_provider_errors_become_llm_errors(error, kind):
    class Failing(FakeDecisions):
        def create(self, model, input, questions):
            raise error
    with pytest.raises(LLMError) as e:
        judge(Failing()).select(extractor(), "n", 1, BODY)
    assert e.value.kind == kind


def test_a_wrong_number_of_answers_is_an_invalid_output():
    class Short(FakeDecisions):
        def create(self, model, input, questions):
            r = super().create(model, input, questions)
            r.answers = r.answers[:-1]
            return r
    with pytest.raises(LLMError) as e:
        judge(Short({"WAL": 1.0})).select(extractor(), "n", 1, BODY)
    assert e.value.kind == "invalid_output"
