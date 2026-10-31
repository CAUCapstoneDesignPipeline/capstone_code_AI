import pytest
from pydantic import ValidationError

from app.schemas.knowledge import Concept, Keyword

N1 = "11111111-1111-4111-8111-111111111111"
N2 = "22222222-2222-4222-8222-222222222222"
C1 = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
C2 = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
K1 = "cccccccc-cccc-4ccc-8ccc-cccccccccccc"


def source(note=N1):
    return {"noteId": note, "noteVersion": 1, "span": {"startOffset": 0, "endOffset": 5, "quote": "힙은 트리다"}}


def concept(cid=C1, **over):
    return {"conceptId": cid, "origin": "extracted", "explanation": "힙은 완전 이진 트리예요.",
            "sources": [source()], **over}


def keyword(*concepts, **over):
    return {"keywordId": K1, "name": "힙", "concepts": list(concepts) or [concept()], **over}


def test_keyword_groups_concepts_from_several_pages():
    k = Keyword(**keyword(concept(C1), concept(C2, sources=[source(N2)])))
    assert k.note_ids() == {N1, N2}


def test_external_defaults_to_empty():
    assert Concept(**concept()).external == []


@pytest.mark.parametrize("over", [
    {"sources": [source(), source(N2)]},                        # extracted 는 페이지 하나
    {"external": [{"url": "https://a.dev", "retrievedAt": "2026-10-31T09:00:00Z"}]},
    {"explanation": "   "},
    {"unknown": 1},
])
def test_extracted_concept_rules(over):
    with pytest.raises(ValidationError):
        Concept(**concept(**over))


def test_generated_concept_needs_two_different_notes_and_may_cite_the_web():
    ext = [{"url": "https://example.com/heap", "title": "Heap", "retrievedAt": "2026-10-31T09:00:00+09:00"}]
    ok = Concept(**concept(origin="generated", sources=[source(N1), source(N2)], external=ext))
    assert ok.external[0].url == "https://example.com/heap"
    with pytest.raises(ValidationError):
        Concept(**concept(origin="generated", sources=[source(N1), source(N1)]))   # 같은 노트 둘은 기여 하나
    with pytest.raises(ValidationError):
        Concept(**concept(origin="generated", sources=[source(N1)]))


def test_external_url_must_be_http():
    bad = [{"url": "javascript:alert(1)", "retrievedAt": "2026-10-31T09:00:00Z"}]
    with pytest.raises(ValidationError):
        Concept(**concept(origin="generated", sources=[source(N1), source(N2)], external=bad))


@pytest.mark.parametrize("data", [
    keyword(concepts=[]),                                       # 개념 없는 키워드는 쓸모없다
    keyword(concept(C1), concept(C1, sources=[source(N2)])),    # conceptId 중복
    keyword(aliases=["힙"]),
])
def test_keyword_rules(data):
    with pytest.raises(ValidationError):
        Keyword(**data)
