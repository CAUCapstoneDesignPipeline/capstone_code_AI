"""키워드의 설명 고르기 (결정적). 설명은 모델이 쓴 글이 아니라, 노트에서 그 키워드를 풀어 쓴 문장 하나다 (원문 구간).

고르는 순서 (낮을수록 먼저, 같으면 앞쪽 문장):
  0. 키워드로 시작해 바로 뜻을 말하는 문장: "**Consistency**: …", "Dynamic Programming(DP)는 …", "X란 …"
  1. 정의문 꼴의 문장 ("… 은/는 … 이다")
  2. 키워드가 나오는 그 밖의 문장
  너무 짧은 문장(낱말 하나)은 뒤로, 너무 긴 문장도 조금 뒤로 민다.
제목에만 나오는 키워드는 그 제목 아래 첫 문장(다음 제목 전까지)을 설명으로 쓴다. 없으면 설명이 없다.
"""

import re
from bisect import bisect_left, bisect_right

from app.services.relations.sentences import sentence_spans

from .markdown import mask

_AFTER = re.compile(r"^[*_`]*\s*(?:\([^)\n]{0,40}\)[*_`]*\s*)?(?:은|는|이란|란|:|：|—|=)")
_DEFINITIONAL = re.compile(r"(?:은|는|이란|란)\s.*(?:이다|다|입니다)[.!]?\s*$")
MIN_CHARS = 8
LONG_CHARS = 300


class Describer:
    """노트 하나의 문장·제목 위치를 한 번만 계산해 두고, 키워드마다 설명 구간을 고른다."""

    def __init__(self, text: str):
        self.text = text
        self.sents = sentence_spans(text)
        self._starts = [s for s, _ in self.sents]
        pos, heads = 0, []
        for line in mask(text).split("\n"):
            if line.lstrip().startswith("#"):
                heads.append(pos)
            pos += len(line) + 1
        self._heads = heads

    def _cost(self, sent: tuple[int, int], mention: tuple[int, int] | None) -> int:
        s, e = sent
        body = self.text[s:e]
        plain = body.replace("*", "").replace("_", "").strip()
        if mention and mention[0] - s <= 4 and _AFTER.match(self.text[mention[1]:e]):
            cost = 0
        elif _DEFINITIONAL.search(body):
            cost = 1
        else:
            cost = 2
        if len(plain) < MIN_CHARS:
            cost += 3
        elif len(plain) > LONG_CHARS:
            cost += 1
        return cost

    def of(self, spans: list[tuple[int, int]]) -> tuple[int, int] | None:
        """spans: 키워드의 언급 위치 (Python 문자). 설명 구간 (start, end) 또는 None."""
        best = None
        for a, b in sorted(spans):
            i = bisect_right(self._starts, a) - 1
            if i >= 0 and self.sents[i][0] <= a and b <= self.sents[i][1]:
                cand = (self._cost(self.sents[i], (a, b)), self.sents[i][0], self.sents[i])
            else:                                            # 제목(또는 문장 밖) 에서 나온 언급: 아래 첫 문장
                j = bisect_left(self._starts, b)
                if j >= len(self.sents):
                    continue
                s = self.sents[j]
                k = bisect_right(self._heads, b)
                if k < len(self._heads) and self._heads[k] < s[0]:
                    continue                                 # 사이에 다른 제목이 있으면 그 제목의 내용이다
                cand = (self._cost(s, None) + 1, s[0], s)
            if best is None or cand[:2] < best[:2]:
                best = cand
        return best[2] if best else None
