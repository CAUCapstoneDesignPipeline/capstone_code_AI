"""LLM 개념 뽑기(extract-llm-1): 원문에 없는 개념은 버리고, 나온 곳을 모두 UTF-16 위치로 돌려준다."""

from types import SimpleNamespace

import pytest

from app.api.v1.extract import check_output, to_concepts
from app.services.concepts.llm_concepts import ConceptLLM
from app.services.llm import LLMError

NOTE = "# 로그 🧠\n\n**WAL**(Write Ahead Logging)은 로그를 먼저 쓴다. wal 덕분에 복구한다. Flow-Matching 은 방법이고 flow matching 은 또 나온다. 모델 하나.\n"


class FakeLLM:
    provider, model = "anthropic", "fake-concepts"

    def __init__(self, data=None, error=None):
        self.data, self.error, self.calls = data, error, 0

    def complete_json(self, system, user, schema, name):
        self.calls += 1
        if self.error:
            raise self.error
        return SimpleNamespace(data=self.data)


def sel(data, text=NOTE):
    return ConceptLLM(FakeLLM(data)).select("n", 1, text)


def test_concepts_not_in_the_note_are_dropped_and_every_occurrence_is_found():
    out = sel({"concepts": [{"surface": "WAL", "aliases": ["Write Ahead Logging"]}, {"surface": "Raft", "aliases": []}, {"surface": "Flow Matching", "aliases": []}]})
    assert set(out) == {"wal", "flow-matching"}                      # Raft 는 원문에 없다 (key 는 첫 언급의 표기: 원문에서 처음 나온 Flow-Matching)
    assert len(out["wal"]) == 3                                      # WAL · Write Ahead Logging · wal
    assert len(out["flow-matching"]) == 2                           # 하이픈·대소문자 차이를 같은 표기로 본다
    assert all(m.method.value == "llm" for ms in out.values() for m in ms)


def test_positions_are_utf16_and_the_response_check_accepts_them():
    out = sel({"concepts": [{"surface": "WAL", "aliases": []}, {"surface": "복구", "aliases": []}]})
    concepts, _ = to_concepts(out, NOTE)
    check_output(NOTE, concepts, [], [], set())                      # 이모지(2단위) 뒤의 위치도 맞다
    assert all(c.confidence == 1.0 for c in concepts)


def test_latin_forms_need_word_boundaries_and_korean_forms_take_particles():
    out = sel({"concepts": [{"surface": "ode", "aliases": []}, {"surface": "모델", "aliases": []}, {"surface": "match", "aliases": []}]})
    assert set(out) == {"모델"}


def test_duplicates_merge_and_order_follows_the_first_mention():
    out = sel({"concepts": [{"surface": "Flow Matching", "aliases": []}, {"surface": "WAL", "aliases": []}, {"surface": "flow-matching", "aliases": []}]})
    assert list(out) == ["wal", "flow-matching"]


def test_malformed_output_gives_no_concepts_and_llm_errors_propagate():
    assert sel({"concepts": "no"}) == {} and sel(None) == {} and sel({"concepts": [{"aliases": ["x"]}, 3]}) == {}
    with pytest.raises(LLMError):
        ConceptLLM(FakeLLM(error=LLMError("refusal", "x"))).select("n", 1, NOTE)


def test_one_llm_call_per_note():
    llm = FakeLLM({"concepts": []})
    ConceptLLM(llm).select("n", 1, NOTE)
    assert llm.calls == 1
