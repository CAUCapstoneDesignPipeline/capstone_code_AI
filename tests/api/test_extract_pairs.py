"""/extract 의 extract-student-pairs-1 · extract-student-decisions-pairs-1: 관계를 후보 쌍마다 Decisions 로 판정해 학습한 분류기로 고른다 (기본 꺼짐, 가짜 클라이언트)."""

from fake_decisions import FakeDecisions, FakePairs, tiny_model
from fastapi.testclient import TestClient
from test_api import post, settings
from test_extract_two_phase import BODY, request

from app.api.v1.health import ready_profiles
from app.main import create_app
from app.services.concepts import ConceptExtractor, ExtractConfig
from app.services.concepts.extract import Idf
from app.services.concepts.judge import DecisionsJudge, StudentJudge
from app.services.concepts.student import Student
from app.services.relations.pairs import DecisionsPairs, PairRelations

STUDENT, BOTH = "extract-student-pairs-1", "extract-student-decisions-pairs-1"


def req(profile):
    body = request(profile)
    body["policy"]["relationTypes"] = [{"code": "is_a", "label": "종류", "description": "x"}]       # 시험용 분류기가 내는 유형
    return body


def app(pairs=None, judge=None):
    student = Student.load()
    return create_app(settings(), extractor=ConceptExtractor(ExtractConfig(), Idf({}, 10, "t")), student=student,
                      student_judge=StudentJudge(student, DecisionsJudge(judge, "fake-decisions")) if judge else None,
                      pair_relations=PairRelations(DecisionsPairs(pairs, "fake-pairs"), tiny_model()) if pairs else None)


def test_the_pairs_profile_returns_judged_relations_with_valid_evidence():
    pairs = FakePairs(default={"none": 0.1, "is_a_ab": 0.9})
    with TestClient(app(pairs)) as c:
        r = post(c, req(STUDENT))
        assert r.status_code == 200, r.text
        d = r.json()
        assert d["relations"] and all(x["typeCode"] for x in d["relations"])
        assert all(e["quote"] == BODY.encode("utf-16-le")[2 * e["startOffset"]:2 * e["endOffset"]].decode("utf-16-le") for e in d["evidence"])     # 근거는 원문 그대로
        m = d["meta"]
        assert m["provider"] == "local+openai" and m["model"] == "fake-pairs+pair-relations" and "pairs-1" in m["promptVersion"] and m["promptVersion"].startswith("student-1+")
        assert m["modelRevision"].startswith("extract-student-1+") and "+pairs-test+" in m["modelRevision"] and m["modelRevision"].endswith("+fake-pairs")
        assert len(pairs.calls) == 1 and pairs.calls[0][1] == BODY                      # 노트는 한 번, 질문은 쌍마다
        assert post(c, req(STUDENT)).json()["relations"] == d["relations"]        # 같은 답이면 같은 결과


def test_without_the_judge_the_relations_are_empty_but_the_concepts_stay():
    with TestClient(app(FakePairs())) as c:
        d = post(c, req(STUDENT)).json()
        assert d["concepts"] and d["relations"] == [] and d["evidence"] == []


def test_the_pairs_profiles_are_off_unless_configured():
    with TestClient(app()) as c:
        for profile in (STUDENT, BOTH):
            r = post(c, req(profile))
            assert r.status_code == 503 and r.json()["error"]["code"] == "SERVICE_UNAVAILABLE"
        assert STUDENT not in ready_profiles(c.app.state) and "extract-student-1" in ready_profiles(c.app.state)
    with TestClient(app(FakePairs())) as c:
        assert STUDENT in ready_profiles(c.app.state) and BOTH not in ready_profiles(c.app.state)       # 개념 판정이 없으면 둘 다 쓰는 profile 은 꺼져 있다
    with TestClient(app(FakePairs(), FakeDecisions({"WAL": 1.0, "체크포인트": 1.0}))) as c:
        assert {STUDENT, BOTH} <= set(ready_profiles(c.app.state))


def test_the_combined_profile_judges_concepts_first_and_then_their_relations():
    judge, pairs = FakeDecisions({"WAL": 1.0, "체크포인트": 1.0}), FakePairs(default={"none": 0.1, "is_a_ab": 0.9})
    with TestClient(app(pairs, judge)) as c:
        r = post(c, req(BOTH))
        assert r.status_code == 200, r.text
        d = r.json()
        assert {"WAL", "체크포인트"} <= {x["label"] for x in d["concepts"]} and d["relations"]
        assert d["meta"]["promptVersion"].startswith("decisions-1+pairs-1") and d["meta"]["model"] == "fake-decisions+pair-relations"
