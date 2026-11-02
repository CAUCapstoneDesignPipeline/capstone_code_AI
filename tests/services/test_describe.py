import pytest

from app.services.concepts.describe import Describer

TEXT = ("# 알고리즘\n\n알고리즘은 문제를 푸는 절차다. 정렬이 대표적이다.\n\n## 일관성\n\n- **Consistency**: 일관성\n"
        "트랜잭션은 Consistency 를 지킨다.\n\n## 빈 제목\n\n### 다음 제목\n\n내용이다.\n")


def spans(word):
    out, i = [], TEXT.find(word)
    while i != -1:
        out.append((i, i + len(word)))
        i = TEXT.find(word, i + 1)
    return out


def pick(word, nth=None):
    s = spans(word) if nth is None else [spans(word)[nth]]
    r = Describer(TEXT).of(s)
    return TEXT[r[0]:r[1]] if r else None


@pytest.mark.parametrize("word, nth, want", [
    ("알고리즘", None, "알고리즘은 문제를 푸는 절차다."),          # 제목과 정의문 둘 다 나오면 정의문
    ("알고리즘", 0, "알고리즘은 문제를 푸는 절차다."),             # 제목에만 나온 언급은 제목 아래 첫 문장
    ("Consistency", None, "**Consistency**: 일관성"),              # 용어: 풀이 꼴이 일반 문장보다 먼저
    ("Consistency", 1, "트랜잭션은 Consistency 를 지킨다."),       # 그 문장만 주면 그 문장
    ("빈 제목", None, None),                                       # 아래에 다른 제목이 먼저 오면 설명이 없다
])
def test_description_choice(word, nth, want):
    assert pick(word, nth) == want


def test_describer_is_deterministic_and_reusable():
    d = Describer(TEXT)
    assert d.of(spans("정렬")) == d.of(list(reversed(spans("정렬"))))
    assert d.of([]) is None
