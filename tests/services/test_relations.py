import json
from types import SimpleNamespace

import pytest

from app.services.concepts import ConceptExtractor, ExtractConfig
from app.services.relations import PairItem, RelationTyper, TyperError, extract_relations
from app.services.relations.rules import classify, code_for
from app.services.relations.sentences import sentence_spans

POLICY = [(c, c, "x") for c in ("is_a", "part_of", "causes", "used_for", "requires", "contrasts_with",
                                "example_of", "analogous_to")]


@pytest.mark.parametrize("between,after,semantic,direction", [
    ("는 ", "을 만든다.", "causes", "ab"),
    ("**는 ", "을 만든다.", "causes", "ab"),               # 마크다운 기호 무시
    ("는 ", "의 일종이다.", "is_a", "ab"),
    ("는 ", "의 일부다.", "part_of", "ab"),
    (" 때문에 ", "가 생긴다", "causes", "ab"),
    ("를 위해 ", "를 쓴다", "used_for", "ba"),
    ("는 ", "가 필요하다", "requires", "ab"),
    ("는 ", "와 달리 실제", "contrasts_with", "ab"),      # K-medoids는 K-means와 달리
    ("는 ", "와 반대로 Combine", "contrasts_with", "ab"),   # Quick Sort는 Merge Sort와 반대로
    ("는 ", "처럼 동작한다", "analogous_to", "ab"),
    ("처럼, OS는 **", "이라는", "analogous_to", "ab"),
    (" 같은 ", "는", "example_of", "ab"),
])
def test_rules(between, after, semantic, direction):
    rule = classify(between, after)
    assert rule and (rule.semantic, rule.direction) == (semantic, direction)


def test_no_rule_for_plain_list():
    assert classify(", ", ", 그리고") is None


@pytest.mark.parametrize("between,after", [
    ("과 달리 ", "다항 시간"),       # Knapsack과 달리 의사다항… : 주제가 없으면 대조 상대를 모른다
    ("처럼 ", "가 있는 경우"),       # if-else처럼 분기가… : 예시일 뿐 유사 관계가 아니다
    ("처럼 ", " >> 아이템 수"),      # Netflix처럼 사용자 수…
    ("처럼 같은 ", "를 반복적으로"),   # 회귀 테스트처럼 같은 테스트를… : "같은" 은 주제어가 아니다
    ("처럼 사고하는 ", ""),          # 인간처럼 사고하는 기계
])
def test_cue_without_topic_is_not_a_relation(between, after):
    assert classify(between, after) is None


def test_code_for_uses_policy_synonyms():
    assert code_for("causes", {"cause"}) == "cause"
    assert code_for("causes", {"is_a"}) is None


def test_sentences_skip_headings_and_code():
    text = "# 제목\n- 첫 문장이다. 둘째 문장.\n```\n코드 줄\n```\n| a | b |\n|--|--|\n"
    spans = [text[s:e] for s, e in sentence_spans(text)]
    assert spans == ["첫 문장이다.", "둘째 문장.", "a | b"]


NOTE = "# WAL\n\nDB의 **WAL**처럼, OS는 **저널링**이라는 것으로 이런 문제를 해결한다. 😀\n**재시도**는 중복을 만든다.\n"


def selected(text):
    ms = ConceptExtractor(ExtractConfig()).extract("n", text)
    out = {}
    for m in ms:
        out.setdefault(m.key, []).append(m)
    return out


def test_extract_relations_offsets_and_types():
    ev, rels = extract_relations(NOTE, selected(NOTE), POLICY)
    body16 = NOTE.encode("utf-16-le")
    for e in ev:
        assert body16[2 * e.start16:2 * e.end16].decode("utf-16-le") == e.quote
    got = {(r.source, r.code, r.target) for r in rels}
    assert ("wal", "analogous_to", "저널링") in got
    assert ("재시도", "causes", "중복") in got
    assert all(r.method == "rule" for r in rels)


def test_extract_relations_deterministic():
    a = extract_relations(NOTE, selected(NOTE), POLICY)
    b = extract_relations(NOTE, selected(NOTE), POLICY)
    assert a == b


# ---- LLM typer (가짜 LLM) ----

class FakeLLM:
    provider, model = "fake", "fake-model"

    def __init__(self, reply=None, error=None):
        self.calls, self.reply, self.error = [], reply, error

    def complete_json(self, system, user, schema, name):
        from app.services.llm import LLMResult
        self.calls.append({"system": system, "user": user, "schema": schema, "name": name})
        if self.error:
            raise self.error
        items = json.loads(user)
        reply = self.reply if self.reply is not None else {"items": [
            {"id": it["id"], "typeCode": "causes", "direction": "a_to_b", "confidence": 1.7} for it in items]}
        return LLMResult(reply, self.provider, self.model)


ITEMS = [PairItem("p1", "지수 백오프 그리고 재시도 정책.", "지수 백오프", "재시도 정책")]


def test_typer_calls_once_then_uses_cache(tmp_path):
    llm = FakeLLM()
    first = RelationTyper(llm, tmp_path).type_pairs(ITEMS, POLICY)
    assert first["p1"].code == "causes" and first["p1"].confidence == 1.0       # 0~1 로 자른다
    enum = llm.calls[0]["schema"]["properties"]["items"]["items"]["properties"]["typeCode"]["enum"]
    assert enum == [c for c, _, _ in POLICY] + ["none"]
    llm2 = FakeLLM()
    assert RelationTyper(llm2, tmp_path).type_pairs(ITEMS, POLICY) == first and llm2.calls == []


def test_typer_cache_is_per_provider_and_model(tmp_path):
    RelationTyper(FakeLLM(), tmp_path).type_pairs(ITEMS, POLICY)
    other = FakeLLM()
    other.provider = "other"
    RelationTyper(other, tmp_path).type_pairs(ITEMS, POLICY)
    assert len(other.calls) == 1                       # 다른 제공자는 캐시를 공유하지 않는다


def test_typer_none_and_missing_items(tmp_path):
    assert RelationTyper(FakeLLM(reply={"items": []}), tmp_path).type_pairs(ITEMS, POLICY) == {}


def test_typer_propagates_llm_error(tmp_path):
    with pytest.raises(TyperError) as e:
        RelationTyper(FakeLLM(error=TyperError("invalid_output", "x")), tmp_path).type_pairs(ITEMS, POLICY)
    assert e.value.kind == "invalid_output"
