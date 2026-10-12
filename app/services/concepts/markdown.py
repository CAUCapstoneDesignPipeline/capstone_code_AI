"""마크다운 처리: 개념이 아닌 구간 가리기, 구조 구간 찾기, UTF-16 위치 변환.

가리기는 지우지 않고 같은 길이의 공백으로 덮는다. 그래서 가린 뒤의 위치가 원문 위치와 같다.
"""

import re
import unicodedata
from bisect import bisect_right

HEADING, BOLD, TABLE = "heading", "bold", "table"

_MASK_PATTERNS = [
    re.compile(r"\A---[ \t]*\n.*?\n(?:---|\.\.\.)[ \t]*(?=\n|\Z)", re.S),   # 맨 앞 YAML frontmatter (title·tags·date)
    re.compile(r"^(```|~~~).*?^\1[^\n]*$", re.S | re.M),   # 코드 블록
    re.compile(r"\$\$.*?\$\$", re.S),                       # 블록 수식
    re.compile(r"(?<!\\)\$[^$\n]+?\$"),                     # 인라인 수식
    re.compile(r"`[^`\n]+`"),                               # 인라인 코드
    re.compile(r"!\[[^\]\n]*\]\([^)\n]*\)"),                # 이미지
    re.compile(r"(?<=\])\([^)\n]*\)"),                      # 링크 주소 (링크 글자는 남긴다)
    re.compile(r"https?://\S+"),                            # 맨 주소
    re.compile(r"<[^>\n]+>"),                               # HTML 태그
]


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def mask(text: str) -> str:
    """개념 후보에서 뺄 구간을 같은 길이의 공백으로 덮는다. 줄바꿈은 남겨 줄 구조를 유지한다."""
    chars = list(text)
    for pat in _MASK_PATTERNS:
        for m in pat.finditer(text):
            for i in range(m.start(), m.end()):
                if chars[i] != "\n":
                    chars[i] = " "
    return "".join(chars)


_HEADING = re.compile(r"^#{1,6}[ \t]+(.+?)[ \t#]*$", re.M)
_BOLD = re.compile(r"\*\*([^*\n]+?)\*\*|__([^_\n]+?)__")
_TABLE_ROW = re.compile(r"^[ \t]*\|([^|\n]*)\|", re.M)
_TABLE_SEP = re.compile(r"^[\s:|-]+$")


def regions(masked: str) -> list[tuple[int, int, str]]:
    """구조 구간 (start, end, 종류). 가린 원문에서 찾으므로 코드 블록 안의 '#' 은 제목이 아니다."""
    out = []
    for m in _HEADING.finditer(masked):
        out.append((m.start(1), m.end(1), HEADING))
    for m in _BOLD.finditer(masked):
        g = 1 if m.group(1) is not None else 2
        out.append((m.start(g), m.end(g), BOLD))
    for m in _TABLE_ROW.finditer(masked):
        cell = m.group(1)
        if cell.strip() and not _TABLE_SEP.match(cell):
            out.append((m.start(1), m.end(1), TABLE))
    return sorted(out)


# 구조 구간 안의 틀 문구: "참고1: 트랜잭션", "1. WAL", "Q. ..." → 앞부분 제거
_PREFIX = re.compile(r"^\s*(?:(?:참고|예시|예|Q|A|Step|단계)\s*\d*\s*[:.)]\s*|\d+(?:\.\d+)*[.)]?\s+|[-*•]\s+)")
# 한 구간 안에서 여러 용어를 나누는 기호: "Deadlock: 교착 상태", "A — B", "A / B", "WAL (Write Ahead Logging)"
_SPLIT = re.compile(r"\s*(?::|—|–|\s-\s|\s/\s|,|\(|\))\s*")


def region_parts(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """구조 구간을 용어 단위 조각으로 나눈다. 조각의 원문 위치를 돌려준다."""
    lead = _PREFIX.match(text, start, end)          # 앞의 틀 문구를 먼저 떼고 나눈다
    if lead and lead.end() <= end:
        start = lead.end()
    seg = text[start:end]
    parts, pos = [], 0
    for m in list(_SPLIT.finditer(seg)) + [None]:
        stop = m.start() if m else len(seg)
        core = seg[pos:stop]
        stripped = core.strip().strip("()[]{}\"'“”‘’")
        if stripped:
            s = start + pos + core.index(stripped)
            parts.append((s, s + len(stripped)))
        pos = m.end() if m else len(seg)
    return parts


class Utf16Index:
    """Python 문자 위치 → UTF-16 코드 단위 위치. BMP 밖 문자(이모지 등)는 2단위다."""

    def __init__(self, text: str):
        self._wide = [i for i, ch in enumerate(text) if ord(ch) > 0xFFFF]

    def __call__(self, i: int) -> int:
        return i + bisect_right(self._wide, i - 1)
