"""결정적 판정 규칙: 개념 동일성, 간선 함의, 조사."""

from app.services.reasoning.entailment import check_edge
from app.services.reasoning.identity import Side, judge
from app.services.reasoning.text import stated_alias


def side(label, doc, ctx, topic_id="t1", surface=None):
    return Side(label, surface or label, topic_id, doc, doc, frozenset(ctx))


def test_same_label_shared_context_is_same():
    l = side("페이지 캐시", "페이지 캐시는 디스크 블록을 둔다", {"페이지캐시", "디스크", "버퍼", "메모리", "커널"})
    r = side("페이지 캐시", "페이지 캐시와 저널링", {"페이지캐시", "디스크", "버퍼", "메모리", "저널링"}, topic_id="t2")
    assert judge(l, r, 0.9, strict=True).decision == "same"


def test_same_label_needs_shared_concepts_across_topics():
    l = side("K-means", "K-means", {"kmeans", "클러스터", "중심", "거리", "dbscan"})
    r = side("K-Means", "K-Means", {"kmeans", "클러스터", "중심", "거리", "gmm"}, topic_id="t2")
    assert judge(l, r, 1.0, strict=True).decision == "same"              # 다른 개념 3개를 함께 다룬다
    lone = side("K-Means", "K-Means", {"kmeans", "이미지"}, topic_id="t2")
    assert judge(l, lone, 1.0, strict=True).decision == "uncertain"


def test_short_acronym_without_shared_context_is_uncertain():
    # DP = Dynamic Programming (알고리즘) vs DP = Decision Problem (계산 이론)
    l = side("DP", "DP는 부분 문제를 저장한다", {"dp", "메모이제이션", "점화식", "부분문제"})
    r = side("DP", "DP는 예/아니오 문제다", {"dp", "튜링기계", "환원", "np"})
    assert judge(l, r, 0.95, strict=False).decision == "uncertain"


def test_alias_written_in_note_is_same_label():
    doc = "**교착 상태 (Deadlock)** 는 서로 기다리는 상태다."
    assert stated_alias(doc, "교착 상태", "Deadlock")
    l = side("교착 상태", doc, {"교착상태", "프로세스", "자원"})
    r = side("Deadlock", "Deadlock detection", {"deadlock", "프로세스", "자원"})
    assert judge(l, r, 0.3, strict=False).decision == "same"


def test_different_labels_are_different_unless_embedding_is_close():
    l, r = side("스레드", "스레드는", {"스레드"}), side("프로세스", "프로세스는", {"프로세스"})
    assert judge(l, r, 0.4, strict=False).decision == "different"
    assert judge(l, r, 0.95, strict=False).decision == "uncertain"


def test_edge_entailment():
    q = ["**재시도**는 중복을 만든다."]
    assert check_edge("재시도", "중복", "causes", "원인", q).outcome == "entailed"
    assert check_edge("재시도", "오류", "causes", "원인", q).outcome == "unknown"
    assert check_edge("TCP", "전송 프로토콜", "is_a", "종류", ["TCP는 전송 프로토콜의 일종이 아니다."]).outcome == "contradicted"
    assert check_edge("A", "B", "team_new_type", "새 유형", ["A는 B다"]).outcome == "unknown"   # 모르는 유형
    row = ["| 관점 | K-Means | GMM | DBSCAN |"]
    assert check_edge("K-means", "DBSCAN", "contrasts_with", "대비", row).outcome == "entailed"


def test_style_samples_take_prose_not_structure():
    from app.services.reasoning.explain_llm import style_samples
    body = ("---\ntitle: x\n---\n# 제목은 빼야 한다\n\n지금까지 배운 것들은 하나의 뿌리에서 출발했다.\n"
            "| 표 | 머리 |\n```\ncode line 는 빼야 한다 아주 길게\n```\n"
            "- **WAL**은 그냥 방문 로그의 그 로그 정도로 이해하면 된다.\n![img](u)\n짧은 줄\n"
            "하나하나 읽어보면, 이런 특징이 있으면 데이터가 이상해지진 않겠구나~ 싶다.\n")
    got = style_samples(body)
    assert got == ["지금까지 배운 것들은 하나의 뿌리에서 출발했다.", "WAL은 그냥 방문 로그의 그 로그 정도로 이해하면 된다.",
                   "하나하나 읽어보면, 이런 특징이 있으면 데이터가 이상해지진 않겠구나~ 싶다."]


def test_declared_aliases_follow_parenthesized_translations():
    from app.services.reasoning.text import declared_aliases, mentions_concept
    docs = ["프로세스는 최소 하나 이상의 스레드(thread)를 포함한다.", "**가상 메모리 (Virtual Memory)** 는 크다."]
    assert "스레드" in declared_aliases("thread", docs) and "가상 메모리" in declared_aliases("Virtual Memory", docs)
    assert mentions_concept("스레드는 실행 단위다", "thread", docs)
    assert not mentions_concept("프로세스는 크다", "thread", docs)


def test_alias_from_definition_line_and_naming_sentence():
    assert stated_alias("ACID\n- **Atomicity**: 원자성\n- **Durability**: 영구성\n", "원자성", "Atomicity")
    assert stated_alias("이렇게 인증하는 것을 Chain of Trust, 신뢰의 사슬이라고 한다.", "신뢰의 사슬", "Chain of Trust")
    # 한 줄이 두 표기뿐일 때만: 정의 문장이 붙으면 별칭이 아니다
    assert not stated_alias("- Page Table: 각 프로세스의 가상 주소와 실제 주소를 잇는다\n", "Page Table", "가상 주소")
    assert not stated_alias("- Atomicity: 원자성, Consistency: 일관성\n", "Atomicity", "Consistency")


def test_merge_llm_combination_never_adds_low_confidence_merges():
    from app.services.reasoning.identity import Side, Verdict
    from app.services.reasoning.identity_llm import PairOut, combine, needs_llm
    a = Side("Mutex", "Mutex", "t1", "Mutex 로 임계 구역을 보호한다.", "", frozenset())
    b = Side("뮤텍스", "뮤텍스", "t1", "뮤텍스는 락이다.", "", frozenset())
    c = Side("뮤텍스", "뮤텍스", "t2", "뮤텍스는 락이다.", "", frozenset())
    rule_diff, rule_same = Verdict("different", 0.7, "표기가 다릅니다."), Verdict("same", 0.85, "표기가 같습니다.")
    assert not needs_llm(rule_same, a, b) and needs_llm(rule_same, a, c) and needs_llm(rule_diff, a, b)
    assert combine(rule_diff, PairOut("p", "same", 0.9, "음역입니다."), a, b).decision == "same"
    assert combine(rule_diff, PairOut("p", "same", 0.6, "아마 같습니다."), a, b).decision == "uncertain"
    assert combine(rule_diff, PairOut("p", "different", 0.9, "다릅니다."), a, b).decision == "different"
    assert combine(rule_same, PairOut("p", "different", 0.9, "다릅니다."), a, c).decision == "uncertain"
    assert combine(rule_same, None, a, b) == rule_same
