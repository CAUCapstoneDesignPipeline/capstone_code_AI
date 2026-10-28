"""노트 사이 연결 (결정적)."""

from app.api.v1.extract import select_concepts
from app.core.config import EXTRACT_PROFILES
from app.services.concepts import ConceptExtractor, ExtractConfig
from app.services.links import build_corpus, related_notes
from app.services.links.corpus import note_aliases
from app.services.relations.extract import _py_offsets

PROF = EXTRACT_PROFILES["extract-det-3"]
POLICY = [(c, c, "x") for c in ("is_a", "part_of", "requires", "contrasts_with", "causes")]

NOTES = {
    "db-wal": "# WAL\n\n**WAL**은 데이터를 쓰기 전에 로그를 먼저 기록한다. WAL은 Write Ahead Logging이다.\n"
              "체크포인트는 버퍼를 디스크에 내린다.\n",
    "os-journal": "# 저널링\n\n저널링은 파일 시스템의 변경을 먼저 기록한다. DB의 **WAL**처럼, 저널을 쓴다.\n"
                  "체크포인트가 있다.\n",
    "ml": "# 회귀\n\n선형 회귀는 오차를 줄인다. 경사 하강법은 손실을 줄인다.\n",
}


def corpus():
    ex = ConceptExtractor(ExtractConfig())
    return build_corpus(NOTES, ex, lambda ms: select_concepts(ms, PROF), POLICY, PROF.limits)


def test_shared_concept_links_notes():
    links = related_notes(corpus(), "db-wal", k=5)
    assert links and links[0].note_b == "os-journal"
    assert "WAL" in {b.label_a for b in links[0].bridges}
    assert all(l.note_b != "ml" for l in links)          # 공유 개념이 없으면 잇지 않는다


def test_bridge_evidence_contains_concept():
    c = corpus()
    link = related_notes(c, "db-wal", k=1)[0]
    b = next(x for x in link.bridges if x.label_a == "WAL")
    s, e = b.evidence_a
    assert "WAL" in c.notes["db-wal"].text[s:e]
    s, e = b.evidence_b
    assert "WAL" in c.notes["os-journal"].text[s:e]


def test_deterministic():
    assert related_notes(corpus(), "db-wal") == related_notes(corpus(), "db-wal")


def test_aliases_are_note_local_translation_and_acronym():
    ex = ConceptExtractor(ExtractConfig())
    text = "# 개념\n\n**교착 상태 (Deadlock)** 는 서로 기다리는 상태다.\n\n회복: Deadlock 을 끝낸다.\n" \
           "WAL은 Write Ahead Logging이다. 온라인 쇼핑몰 FP 계산과 Function Point.\n"
    ms = ex.extract("n", text)
    sel = {}
    for m in ms:
        sel.setdefault(m.key, []).append(m)
    pairs = note_aliases(text, _py_offsets(text, sel))
    assert ("deadlock", "교착 상태") in pairs
    assert ("wal", "write ahead logging") in pairs
    assert not any("회복" in p for pair in pairs for p in pair)          # 콜론 뒤는 설명이지 번역이 아니다
    assert not any("쇼핑몰" in p for pair in pairs for p in pair)        # 약어 표기 전체가 영문이어야 한다


def test_hub_concepts_ignored():
    notes = {f"n{i}": f"# 노트 {i}\n\n공통개념은 자주 나온다. 고유개념{i}도 있다.\n" for i in range(10)}
    ex = ConceptExtractor(ExtractConfig())
    c = build_corpus(notes, ex, lambda ms: select_concepts(ms, PROF), POLICY, PROF.limits)
    assert related_notes(c, "n0", hub_ratio=0.15) == []                 # 10개 노트 모두에 나온 개념뿐이면 잇지 않는다


def test_with_note_links_pasted_text_and_keeps_df_consistent():
    c = corpus()
    ex = ConceptExtractor(ExtractConfig())
    from app.services.links import build_entry
    entry = build_entry("paste", "# 로그\n\n**WAL**은 로그를 먼저 쓴다. 체크포인트도 있다.\n", ex,
                        lambda ms: select_concepts(ms, PROF), POLICY, PROF.limits)
    c2 = c.with_note(entry)
    assert "paste" not in c.notes                                        # 원래 색인은 그대로
    assert related_notes(c2, "paste", k=3)[0].note_b in ("db-wal", "os-journal")
    # 이미 있는 노트를 바꿔도 df 는 처음부터 만든 것과 같다
    replaced = c.with_note(build_entry("ml", NOTES["db-wal"], ex, lambda ms: select_concepts(ms, PROF), POLICY, PROF.limits))
    rebuilt = build_corpus({**NOTES, "ml": NOTES["db-wal"]}, ex, lambda ms: select_concepts(ms, PROF), POLICY, PROF.limits)
    assert replaced.df == rebuilt.df


def test_alias_pair_counts_once():
    ex = ConceptExtractor(ExtractConfig())
    notes = {"a": "# 교착 상태 (Deadlock)\n\n**교착 상태 (Deadlock)** 는 서로 기다린다. Deadlock 을 막는다. 교착 상태가 생긴다.\n",
             "b": "# 회복\n\n교착 상태를 끝내려면 프로세스를 멈춘다. 교착 상태가 다시 생긴다.\n",
             "c": "# 회귀\n\n선형 회귀는 오차를 줄인다.\n"}
    c = build_corpus(notes, ex, lambda ms: select_concepts(ms, PROF), POLICY, PROF.limits)
    link = related_notes(c, "a", k=1)[0]
    targets = [b.concept_b for b in link.bridges]
    assert len(targets) == len(set(targets))                    # B 의 한 개념에 다리 하나


def test_min_confidence_uses_only_valid_concepts():
    c = corpus()
    every = related_notes(c, "db-wal", k=5)
    assert every
    t = 0.9
    for link in related_notes(c, "db-wal", k=5, min_confidence=t):
        for b in link.bridges:
            assert c.notes["db-wal"].confidence[b.concept_a] >= t and c.notes[link.note_b].confidence[b.concept_b] >= t
    assert related_notes(c, "db-wal", k=5, min_confidence=1.01) == []


def test_refused_note_keeps_rule_relations():
    """관계 LLM 이 한 노트를 거절해도 색인은 만들어지고, 그 노트는 규칙 관계만 쓴다. 시간 초과는 그대로 올린다."""
    import pytest
    from app.services.llm import LLMError

    class Refuses:
        provider, model = "anthropic", "m"

        def __init__(self, kind):
            self.kind = kind

        def type_pairs(self, items, policy_types):
            raise LLMError(self.kind, "x")

    ex = ConceptExtractor(ExtractConfig())
    c = build_corpus(NOTES, ex, lambda ms: select_concepts(ms, PROF), POLICY, PROF.limits, typer=Refuses("refusal"))
    assert set(c.notes) == set(NOTES)
    with pytest.raises(LLMError):
        build_corpus(NOTES, ex, lambda ms: select_concepts(ms, PROF), POLICY, PROF.limits, typer=Refuses("timeout"))


def test_offline_evaluation_ranks_gold_pair_and_novelty():
    from app.services.links.evaluate import evaluate, novelty
    texts = {**NOTES, "db-wal2": "# WAL 메모\n\n**WAL** 은 로그를 먼저 기록한다. 체크포인트는 버퍼를 내린다.\n"}
    topics = {"db-wal": "DB", "db-wal2": "DB", "os-journal": "OS", "ml": "ML"}
    ex = ConceptExtractor(ExtractConfig())
    c = build_corpus(texts, ex, lambda ms: select_concepts(ms, PROF), POLICY, PROF.limits)
    nov = novelty(c, topics)
    assert nov[("DB", "DB")] == 0.0 and 0 < nov[("DB", "OS")] <= 1
    by = {r.method: r for r in evaluate(c, topics, [("db-wal", "os-journal")], alphas=(1.0,))}
    assert by["S (다른 주제)"].ranks[("db-wal", "os-journal")] == 1     # 같은 주제(db-wal2)는 후보에서 빠진다
    n1 = by["S·N^1"].ranks
    assert n1[("db-wal", "os-journal")] == 1 and n1[("os-journal", "db-wal")] in (1, 2)   # 반대쪽은 db-wal2 도 후보


def test_hide_and_refind_finds_pair_through_other_notes():
    from app.services.links.evaluate import hide_and_refind
    texts = {"a": "# 메모\n\n재시도와 장애는 함께 나타난다.\n",
             "b": "# 네트워크\n\n재시도는 타임아웃 뒤에 일어난다. 재시도는 중복을 만든다.\n",
             "c": "# DB\n\n중복은 장애를 만든다. 장애는 복구가 필요하다.\n"}
    ex = ConceptExtractor(ExtractConfig())
    c = build_corpus(texts, ex, lambda ms: select_concepts(ms, PROF), POLICY, PROF.limits)
    aa, pop = hide_and_refind(c, n=50, seed=0, hub_ratio=1.0)
    assert aa.ranks and set(aa.ranks) == set(pop.ranks)
    hidden = {(a, x): r for (nid, a, x), r in aa.ranks.items() if nid == "a"}
    assert hidden[("장애", "재시도")] == 1 or hidden.get(("재시도", "장애")) == 1     # 다른 두 노트의 '중복' 을 거쳐 다시 찾는다
    assert hide_and_refind(c, n=50, seed=0, hub_ratio=1.0)[0].ranks == aa.ranks      # 같은 seed 면 같은 결과
