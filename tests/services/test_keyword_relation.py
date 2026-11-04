import pytest

from app.services.links.relation_llm import KeywordRelationJudge
from app.services.llm import LLMError, LLMResult

A = {"name": "TCP", "concepts": ["TCP는 전송 프로토콜의 일종이다."]}
B = {"name": "UDP", "concepts": []}
EV = [{"level": "sentence", "note": "n1", "text": "TCP와 UDP는 모두 포트 번호를 담는다."},
      {"level": "path", "label": "TCP — 전송 프로토콜", "note": "n2", "text": "TCP는 전송 프로토콜이다."}]


class Fake:
    provider, model = "anthropic", "fake"

    def __init__(self, data):
        self.data, self.seen = data, []

    def complete_json(self, system, user, schema, name):
        self.seen.append((system, user, schema))
        return LLMResult(self.data, self.provider, self.model)


def reply(**kw):
    return {"verdict": "related", "type": "contrasts_with", "direction": "symmetric", "reason": "둘을 대비한다.",
            "evidence": ["E1"], **kw}


def judge(data, cache=None):
    return KeywordRelationJudge(Fake(data), cache).judge(A, B, EV)


def test_prompt_carries_evidence_ids_and_types():
    j = KeywordRelationJudge(Fake(reply()), None)
    j.judge(A, B, EV)
    system, user, schema = j.llm.seen[0]
    assert '"id": "E1"' in user and '"id": "E2"' in user and "TCP" in user
    assert "contrasts_with" in system and set(schema["properties"]["verdict"]["enum"]) == {"related", "weak", "unrelated"}


def test_valid_judgment_passes_through():
    r = judge(reply(evidence=["E1", "E1"]))
    assert (r["verdict"], r["type"], r["evidence"]) == ("related", "contrasts_with", ["E1"])


def test_invented_evidence_ids_are_dropped_and_uncited_judgments_are_refused():
    assert judge(reply(evidence=["E1", "E9"]))["evidence"] == ["E1"]
    for ev in ([], ["E9"]):
        r = judge(reply(evidence=ev))
        assert r["verdict"] == "unsure" and r["evidence"] == [] and r["type"] == "none"


def test_related_on_path_evidence_alone_is_downgraded_to_weak():
    assert judge(reply(evidence=["E2"]))["verdict"] == "weak"
    assert judge(reply(evidence=["E1", "E2"]))["verdict"] == "related"


def test_unrelated_needs_no_citation_and_has_no_type():
    r = judge(reply(verdict="unrelated", type="is_a", evidence=[]))
    assert (r["verdict"], r["type"]) == ("unrelated", "none")


@pytest.mark.parametrize("bad", [{"verdict": "maybe"}, {"type": "likes"}, {"direction": "up"}])
def test_invalid_enum_is_an_error(bad):
    with pytest.raises(LLMError):
        judge(reply(**bad))


def test_cache_is_keyed_by_input_and_model(tmp_path):
    fake = Fake(reply())
    j = KeywordRelationJudge(fake, tmp_path)
    assert j.judge(A, B, EV)["cached"] is False and j.judge(A, B, EV)["cached"] is True and len(fake.seen) == 1
    j.judge(A, B, EV[:1])                                           # 증거가 다르면 다시 묻는다
    assert len(fake.seen) == 2


def test_judge_streams_fields_and_reports_a_cached_result_without_calling_the_model(tmp_path):
    class S(Fake):
        def stream_json(self, system, user, schema, name, on_event=None):
            for e in ({"path": "verdict", "value": "related"}, {"path": "reason", "delta": "둘을 "}, {"path": "reason", "delta": "대비한다."}):
                on_event({"type": "field", **e})
            return self.complete_json(system, user, schema, name)

    fake = S(reply())
    j = KeywordRelationJudge(fake, tmp_path)
    seen = []
    assert j.judge(A, B, EV, on_event=seen.append)["verdict"] == "related"
    assert [e["type"] for e in seen] == ["field"] * 3 and "".join(e["delta"] for e in seen if "delta" in e) == "둘을 대비한다."
    seen.clear()
    assert j.judge(A, B, EV, on_event=seen.append)["cached"] is True and seen == [{"type": "cached"}] and len(fake.seen) == 1
