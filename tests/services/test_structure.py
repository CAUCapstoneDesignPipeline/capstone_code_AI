"""마크다운 구조 관계. 각 경우는 1주차 노트에서 실제로 맞거나 틀렸던 모양이다."""

import pytest

from app.services.concepts import ConceptExtractor, ExtractConfig
from app.services.relations import extract_relations
from app.services.relations.extract import _py_offsets
from app.services.relations.structure import structural_candidates

POLICY = [(c, c, "x") for c in ("is_a", "part_of", "requires", "contrasts_with", "related_to")]


@pytest.fixture(scope="module")
def ex():
    return ConceptExtractor(ExtractConfig())


def triples(ex, text, policy=POLICY):
    ms = ex.extract("n", text)
    sel = {}
    for m in ms:
        sel.setdefault(m.key, []).append(m)
    _, rels = extract_relations(text, sel, policy, structure=True)
    return {(r.source, r.code, r.target) for r in rels}


def test_heading_kinds_is_a(ex):
    text = "# IP의 종류\n\n## 공인 IP\n\n설명.\n\n## 사설 IP\n\n설명.\n"
    got = triples(ex, text)
    assert ("공인 ip", "is_a", "ip") in got and ("사설 ip", "is_a", "ip") in got


def test_list_conditions_requires(ex):
    text = "# Deadlock의 조건\n\n- **상호 배제**: 한 번에 하나만 쓴다\n- **순환 대기**: 서로 기다린다\n"
    got = triples(ex, text)
    assert ("deadlock", "requires", "상호 배제") in got and ("deadlock", "requires", "순환 대기") in got


def test_genus_definition(ex):
    got = triples(ex, "# 스레드\n\n스레드는 프로세스 안의 실행 단위이다.\n")
    assert ("스레드", "is_a", "실행 단위") in got


def test_definition_line(ex):
    got = triples(ex, "# 용어\n\n- **Think rationally**: 논리적으로 사고하는 기계\n")
    assert ("think rationally", "is_a", "기계") in got


def test_compare_table(ex):
    text = "# 정렬\n\n| 항목 | Quick Sort | Merge Sort |\n|--|--|--|\n| 안정성 | X | O |\n"
    assert ("quick sort", "contrasts_with", "merge sort") in triples(ex, text)


def test_vs_heading_contrasts_title_sides(ex):
    text = "# Static Scope vs. Dynamic Scope\n\n- **Static scope**: 선언 위치\n- **Dynamic scope**: 호출 순서\n"
    got = triples(ex, text)
    assert ("static scope", "contrasts_with", "dynamic scope") in got


def test_related_needs_policy_code(ex):
    text = "# Memory Layout\n\n## Stack 영역\n\n설명.\n"
    assert any(c == "related_to" for _, c, _ in triples(ex, text))
    assert triples(ex, text, [p for p in POLICY if p[0] != "related_to"]) == set()


# ---- 실제 노트에서 틀렸던 모양 ----

@pytest.mark.parametrize("text", [
    "# Syntax Analyzer Part 1\n\n## Abstract Syntax Tree\n\n설명.\n",            # "Part 1" 은 part_of 단서가 아니다
    "# 조건 커버리지 (Condition Coverage)\n\n## 장학금 대상자 선정\n\n설명.\n",     # 이름 안의 '조건'
    "# NFA 구성 방법\n\n## LR Item\n\n설명.\n",                                    # '구성' = 만들기
])
def test_cue_inside_name_is_not_typed(ex, text):
    assert not {t for t in triples(ex, text) if t[1] != "related_to"}


def test_items_under_vs_are_not_contrasted(ex):
    text = "# User-Oriented vs System-Oriented\n\n- **Turnaround time**: 걸린 시간\n- **Processor utilization**: 사용률\n"
    assert ("turnaround time", "contrasts_with", "processor utilization") not in triples(ex, text)


def test_data_table_is_not_comparison(ex):
    text = "# 풀이\n\n| Candidate | TID 10 | TID 20 |\n|--|--|--|\n| A | 1 | 0 |\n"
    assert not any(c == "contrasts_with" for _, c, _ in triples(ex, text))


def test_sentence_list_items_skipped(ex):
    text = "# 저널링\n\n- 데이터 블록에 내용을 씀\n- inode의 크기, 블록 목록 등을 갱신한다.\n"
    assert not triples(ex, text)


def test_acronym_is_not_is_a(ex):
    assert not any(c == "is_a" for _, c, _ in triples(ex, "# 로그\n\nWAL은 Write Ahead Logging이다.\n"))


def test_genus_rejects_two_clause_sentence(ex):
    text = "# 메모리\n\n커널 영역은 운영체제의 메모리 영역이고, 이를 쓰는 CPU 모드가 커널 모드다.\n"
    assert ("커널 영역", "is_a", "커널 모드") not in triples(ex, text)


def test_structure_deterministic(ex):
    text = "# IP의 종류\n\n## 공인 IP\n\n- **NAT**: 주소를 바꾸는 기술\n"
    assert extract_relations.__name__ and triples(ex, text) == triples(ex, text)


def test_evidence_spans_are_exact_slices(ex):
    text = "# IP의 종류 😀\n\n## 공인 IP\n\n스레드는 프로세스 안의 실행 단위이다.\n"
    ms = ex.extract("n", text)
    sel = {}
    for m in ms:
        sel.setdefault(m.key, []).append(m)
    ev, _ = extract_relations(text, sel, POLICY, structure=True)
    body16 = text.encode("utf-16-le")
    assert ev and all(body16[2 * e.start16:2 * e.end16].decode("utf-16-le") == e.quote for e in ev)
