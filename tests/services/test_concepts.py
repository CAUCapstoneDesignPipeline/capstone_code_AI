import pytest

from app.services.concepts import ConceptExtractor, ExtractConfig, Idf, Method, build_idf, concept_key
from app.services.concepts.markdown import Utf16Index, mask, region_parts

NOTE = """# WAL (Write Ahead Logging)

DB의 **WAL**처럼, OS는 **저널링**이라는 것으로 이런 문제를 해결한다.
WAL은 buffer pool 의 페이지를 디스크에 쓰기 전에 로그를 먼저 기록한다. 😀

## 참고1: Deadlock: 교착 상태

```python
# 코드 안의 Kafka 는 개념이 아니다
print("Kafka")
```

| 용어 | 설명 |
|--|--|
| Banker's Algorithm | 교착 상태 회피 |
| 0-1 Knapsack | DP 문제 |
"""


@pytest.fixture(scope="module")
def ex():
    return ConceptExtractor(ExtractConfig())


def utf16_slice(text, start, end):
    return text.encode("utf-16-le")[start * 2:end * 2].decode("utf-16-le")


def test_mask_keeps_length_and_hides_code():
    masked = mask(NOTE)
    assert len(masked) == len(NOTE)
    assert "Kafka" not in masked
    assert masked.count("\n") == NOTE.count("\n")


def test_utf16_index_counts_emoji_as_two():
    text = "a😀b"
    u16 = Utf16Index(text)
    assert [u16(i) for i in range(4)] == [0, 1, 3, 4]


def test_region_parts_split_and_strip_prefix():
    text = "참고1: Deadlock: 교착 상태"
    parts = [text[s:e] for s, e in region_parts(text, 0, len(text))]
    assert parts == ["Deadlock", "교착 상태"]


def test_offsets_point_back_to_surface(ex):
    for m in ex.extract("n1", NOTE):
        assert utf16_slice(NOTE, m.span.start, m.span.end) == m.surface


def test_candidates_cover_phrases(ex):
    keys = {c.key for c in ex.candidates(NOTE)}
    for k in ("buffer pool", "write ahead logging", "0-1 knapsack", "banker's algorithm"):
        assert k in keys, k
    assert "kafka" not in keys          # 코드 블록
    assert "문제" not in keys            # 불용어


def test_selected_concepts(ex):
    keys = {m.key for m in ex.extract("n1", NOTE)}
    # 제목·굵은 글씨·표 첫 칸에 나온 용어는 구조 가중치로 선택된다
    for k in ("wal", "저널링", "교착 상태", "deadlock", "banker's algorithm", "0-1 knapsack"):
        assert k in keys, k
    assert "참고1" not in keys           # 제목 틀 문구


def test_markdown_method_for_structural_terms(ex):
    by_key = {}
    for m in ex.extract("n1", NOTE):
        by_key.setdefault(m.key, set()).add(m.method)
    assert Method.MARKDOWN in by_key["저널링"]


def test_deterministic_across_instances():
    a = ConceptExtractor(ExtractConfig()).extract("n1", NOTE)
    b = ConceptExtractor(ExtractConfig()).extract("n1", NOTE)
    assert a == b


def test_idf_version_stable_and_order_independent(ex):
    texts = {"a": NOTE, "b": "# TCP\n\nTCP는 전송 프로토콜이다.", "c": "WAL 과 저널링"}
    v1 = build_idf(ex, texts).version
    v2 = build_idf(ex, dict(reversed(list(texts.items())))).version
    assert v1 == v2


def test_idf_changes_scores_not_offsets(ex, tmp_path):
    texts = {"a": NOTE, "b": "WAL WAL WAL 저널링", "c": "WAL 로그"}
    idf = build_idf(ex, texts)
    path = tmp_path / "idf.json"
    idf.save(path, ex.config_version)
    with_idf = ConceptExtractor(ExtractConfig(), Idf.load(path))
    ms = with_idf.extract("n1", NOTE)
    assert ms and all(m.method_version.endswith(idf.version) for m in ms)


def test_cache_roundtrip(tmp_path):
    e1 = ConceptExtractor(ExtractConfig(), cache_dir=tmp_path)
    first = e1.extract("n1", NOTE)
    assert list(tmp_path.glob("*.json"))
    second = ConceptExtractor(ExtractConfig(), cache_dir=tmp_path).extract("n1", NOTE)
    assert first == second


def test_concept_key_normalizes():
    assert concept_key("  WAL ") == concept_key("wal")
    assert concept_key("(Write  Ahead Logging)") == "write ahead logging"
