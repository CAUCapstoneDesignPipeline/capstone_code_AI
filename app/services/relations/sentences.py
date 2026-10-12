"""근거가 될 문장 구간 찾기. 위치는 원문(Python 문자) 기준이고, 근거 quote 는 이 구간을 원문에서 그대로 자른 것이다.

계약 예시처럼 마크다운 기호는 그대로 둔다 ("**재시도**는 중복을 만든다."). 제목 줄과 코드 블록은 근거로 쓰지 않는다.
"""

import re

from app.services.concepts.markdown import mask

_ENDER = re.compile(r"(?<=[.!?。])\s+")
_LEAD = re.compile(r"^(?:\s|[-*+>|]\s|\d+[.)]\s)+")   # 목록 기호·인용·표 칸 앞부분
_TABLE_SEP = re.compile(r"^[\s:|-]+$")


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """문장 [start, end) 목록. 줄을 먼저 나누고, 줄 안에서 마침표·물음표·느낌표 뒤 공백으로 나눈다."""
    masked = mask(text)
    spans = []
    pos = 0
    for line in masked.split("\n"):
        line_start = pos
        pos += len(line) + 1
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ("|" in stripped and _TABLE_SEP.match(stripped)):
            continue
        start = 0
        for m in _ENDER.finditer(line):
            _add(text, spans, line_start + start, line_start + m.start())
            start = m.end()
        _add(text, spans, line_start + start, line_start + len(line))
    return spans


def _add(text: str, spans: list, start: int, end: int):
    seg = text[start:end]
    lead = _LEAD.match(seg)
    if lead:
        start += lead.end()
    seg = text[start:end].rstrip(" \t|")
    end = start + len(seg)
    if end - start >= 2:
        spans.append((start, end))
