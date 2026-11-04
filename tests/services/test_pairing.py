import numpy as np

from app.services.links.pairing import COSINE_FLOOR, Element, PairingIndex, gate


def el(i, kw, note, text, aliases=()):
    return Element(i, f"k-{kw}", kw, tuple(aliases), note, f"n-{note}", f"c{i}", text, 0.9, 0, len(text), text)


def test_gate_keeps_only_self_contained_sentences():
    assert gate("트랜잭션은 모든 연산이 함께 성공하거나 함께 실패하는 단위예요.")
    assert not gate("짧은 말")                                                       # 너무 짧다
    assert not gate("그래서 이 값은 로그에 먼저 기록한다고 볼 수 있다는 이야기다.")        # 앞 문장에 기대는 말로 시작
    assert not gate("Client | 사용자로부터 입력을 받아 범위를 확인하고 DB 를 구동 | Server | 처리")   # 표 칸
    assert not gate("자세한 내용은 https://example.com/aaa/bbb/ccc 에서 볼 수 있다고 한다.")


def test_terms_channel_needs_two_shared_midfrequency_words_outside_keyword_names():
    els = [el(0, "WAL", "A", "로그를 디스크에 먼저 기록해서 장애 복구를 가능하게 한다고 한다"),
           el(1, "복구", "B", "장애가 나면 디스크에 남은 로그를 읽어서 복구를 진행한다고 한다"),     # 로그·디스크·장애 공유
           el(2, "정렬", "C", "퀵 정렬은 기준 원소를 골라 나누는 방식으로 동작한다고 한다 정말로"),
           el(3, "탐색", "D", "이진 탐색은 정렬된 배열의 가운데를 비교하며 범위를 줄여 간다고 한다")]
    idx = PairingIndex(els)
    cands, counts = idx.candidates(idx.anchors("A"))
    assert [(c.a.keyword, c.b.keyword) for c in cands] == [("WAL", "복구")]
    assert cands[0].channels == ("terms",) and {"로그를", "디스크에"} <= set(cands[0].shared)
    assert counts["terms"] == 1


def test_pivot_channel_links_elements_that_mention_the_same_other_keyword():
    els = [el(0, "WAL", "A", "WAL 은 버퍼풀의 더러운 페이지를 내보내기 전에 로그를 먼저 기록한다"),
           el(1, "버퍼풀", "A", "버퍼풀은 자주 쓰는 페이지를 메모리에 올려 두는 공간이라고 한다"),
           el(2, "캐시", "B", "운영체제의 페이지 캐시와 비슷하게 버퍼풀 역시 읽기 비용을 줄여 준다"),
           el(3, "프로세스", "C", "프로세스 스케줄러는 버퍼풀 크기와 상관없이 CPU 시간을 나누어 준다고 한다")]
    idx = PairingIndex(els)
    cands, _ = idx.candidates([0], per_anchor=5, per_partner=5)
    assert {(c.b.keyword, c.via) for c in cands} == {("캐시", ("버퍼풀",)), ("프로세스", ("버퍼풀",))}


def test_partners_must_be_other_note_and_other_keyword():
    els = [el(0, "WAL", "A", "로그를 디스크에 먼저 기록해서 장애 복구를 가능하게 한다고 한다"),
           el(1, "체크포인트", "A", "디스크에 남은 로그를 읽어서 장애 복구 시간을 줄여 준다고 한다"),      # 같은 노트
           el(2, "WAL", "B", "로그를 디스크에 먼저 기록해서 장애 복구를 가능하게 한다고 한다")]         # 같은 키워드
    idx = PairingIndex(els)
    assert idx.candidates([0])[0] == []


def test_cosine_floor_drops_unrelated_neighbors_and_keeps_close_ones():
    els = [el(0, "WAL", "A", "로그를 디스크에 먼저 기록해서 장애 복구를 가능하게 한다고 한다"),
           el(1, "복구", "B", "장애가 나면 디스크에 남은 로그를 읽어서 복구를 진행한다고 한다"),
           el(2, "정렬", "C", "퀵 정렬은 기준 원소를 골라 나누는 방식으로 동작한다고 한다 정말로")]
    v = np.array([[1, 0], [0.8, 0.6], [0.1, 0.995]], dtype=np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    cands, counts = PairingIndex(els, v).candidates([0])
    assert [(c.b.keyword, c.cos) for c in cands] == [("복구", 0.8)] and counts["embed"] == 2 and counts["afterFloor"] == 1
    assert COSINE_FLOOR == 0.5


def test_same_sentence_under_two_keywords_is_one_candidate_and_agreeing_channels_rank_first():
    twin = "버퍼풀에 올린 페이지를 고쳐서 디스크에는 나중에 한꺼번에 쓴다고 한다"
    els = [el(0, "디스크", "A", twin), el(1, "버퍼풀", "A", twin),
           el(2, "캐시", "B", "버퍼풀에 올린 페이지는 디스크에 쓰기 전에 메모리에서 고쳐진다고 한다"),
           el(3, "쓰기", "C", "디스크에 쓰는 일은 한꺼번에 모아서 처리하면 빨라진다고 한다 정말로")]
    v = np.array([[1, 0], [1, 0], [0.9, 0.436], [0.7, 0.714]], dtype=np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    cands, _ = PairingIndex(els, v).candidates([0, 1], per_anchor=5, per_partner=5)
    assert len({(c.a.text, c.b.text) for c in cands}) == len(cands)                # 같은 문장 쌍은 한 번
    assert cands == sorted(cands, key=lambda c: -c.score)


def _random_elements(n, rng):
    syl = ["로그", "버퍼", "캐시", "트랜잭션", "디스크", "페이지", "복구", "정렬", "탐색", "해시", "큐", "스택", "WAL", "ACID", "락"]
    words = [a + b for a in syl for b in syl if a != b] + syl
    out = []
    for i in range(n):
        kw = f"{rng.choice(words)}{i % 17}"
        text = " ".join(rng.choice(words) for _ in range(rng.randint(5, 12))) + " 라고 한다 정말로 그렇다"
        out.append(Element(i, f"k{i % 23}", kw, (), f"n{i % 11}", f"n{i % 11}", f"c{i}", text, 0.5, 0, len(text), text))
    return out


def test_mention_index_equals_scanning_every_name_in_every_text():
    import random
    rng = random.Random(3)
    els = _random_elements(300, rng)
    idx = PairingIndex(els)
    for i, e in enumerate(els):
        naive = {kid for kid, names in idx.kw_names.items() if kid != e.keyword_id and any(n in idx.low[i] for n in names)} if idx.ok[i] else set()
        assert idx.elem_kws[i] == naive
    for kid, ms in idx.mentioners.items():
        assert ms == [i for i in range(len(els)) if kid in idx.elem_kws[i]]


def test_batched_nearest_equals_scoring_every_pair():
    import random
    rng = random.Random(4)
    els = _random_elements(200, rng)
    v = np.random.default_rng(0).standard_normal((200, 16)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    idx = PairingIndex(els, v)
    anchors = [i for i in range(0, 200, 7) if idx.ok[i]]
    got = idx._nearest_all(anchors, 10)
    for i in anchors:
        valid = [j for j in range(200) if j != i and idx.ok[j] and els[j].keyword_id != els[i].keyword_id and els[j].note != els[i].note]
        want = sorted(valid, key=lambda j: -float(v[i] @ v[j]))[:10]
        assert set(got[i]) == set(want)


def test_work_per_anchor_does_not_grow_with_the_number_of_other_notes():
    """요소 쌍을 모두 비교하지 않는다: 후보 수는 요소당 신호가 가리키는 수로 묶인다 (빈도 상한 덕분)."""
    import random
    for n in (200, 800):
        els = _random_elements(n, random.Random(5))
        v = np.random.default_rng(1).standard_normal((n, 16)).astype(np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        idx = PairingIndex(els, v)
        _, counts = idx.candidates(idx.anchors("n0"), per_anchor=3, limit=12)
        anchors = len(idx.anchors("n0"))
        assert counts["union"] <= anchors * (10 + 40 * 8 + 25 * 40)          # 요소당 임베딩 10 + 공통 키워드 · 공유 낱말의 상한
