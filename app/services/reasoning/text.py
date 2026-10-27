"""판정에 쓰는 결정적 문자열 도구: 표기 비교, 노트 안 별칭, 약어, 관계 유형의 의미 이름."""

import re

from app.services.links.corpus import loose_key
from app.services.relations.rules import SEMANTIC_CODES
from app.services.relations.structure import _is_acronym

_MD = re.compile(r"[*_`~]")


def plain(text: str) -> str:
    """마크다운 강조 기호를 뺀 글 (비교용. 응답의 인용·위치에는 쓰지 않는다)."""
    return _MD.sub("", text)


def mentions(text: str, label: str) -> bool:
    """text 안에 label 이 (공백·하이픈·대소문자 차이를 무시하고) 나온다."""
    k = loose_key(label)
    return bool(k) and k in loose_key(plain(text))


def same_label(a: str, b: str) -> bool:
    return loose_key(a) == loose_key(b)


def stated_alias(text: str, a: str, b: str) -> bool:
    """노트가 두 표기를 같은 것으로 적었다:
    - 'X (Y)' / 'Y (X)' 처럼 바로 괄호로 병기
    - 한 줄이 두 표기뿐인 정의 목록: '- **Atomicity**: 원자성'
    - 이름 붙이기: 'Chain of Trust, 신뢰의 사슬이라고 한다'
    - 한쪽이 다른 쪽의 약어이고 둘 다 그 노트에 나온다 (WAL ↔ Write Ahead Logging)."""
    t = plain(text)
    for x, y in ((a, b), (b, a)):
        ex, ey = re.escape(x), re.escape(y)
        if re.search(ex + r"\s*\(\s*" + ey + r"\s*\)", t, re.I):
            return True
        if re.search(r"^[ \t]*(?:[-*+][ \t]+)?" + ex + r"[ \t]*[:：][ \t]*" + ey + r"[ \t]*$", t, re.I | re.M):
            return True
        if re.search(ex + r"[ \t]*,[ \t]*" + ey + r"(?:이)?라고[ \t]*(?:한다|부른다|하며|부르며|해|불러)", t, re.I):
            return True
    if (_is_acronym(a, b) or _is_acronym(b, a)) and mentions(t, a) and mentions(t, b):
        return True
    return False


def short_acronym(label: str) -> bool:
    """DP, MDP 처럼 짧은 영문 약어. 주제에 따라 다른 뜻일 수 있다 (Dynamic Programming / Decision Problem)."""
    s = label.strip()
    return 1 < len(s) <= 4 and s.isascii() and s.isalpha() and s.upper() == s


def semantic_of(code: str) -> str | None:
    """policy code → 규칙이 아는 의미 이름 (is_a 등). 모르는 code 면 None."""
    return next((sem for sem, codes in SEMANTIC_CODES.items() if code in codes), None)


_WORD = r"[0-9A-Za-z가-힣][0-9A-Za-z가-힣\-_.]*"


def declared_aliases(label: str, documents: list[str]) -> set[str]:
    """문서가 label 과 같은 것이라고 적은 다른 표기: 'label (X)', 'X (label)' 의 X (앞쪽은 낱말 1~3개 후보 모두).
    번역어·약어 병기를 따라가기 위한 것이다 (스레드(thread), Virtual Memory(가상 메모리))."""
    out = set()
    lab = re.escape(label)
    for doc in documents:
        t = plain(doc)
        for m in re.finditer(lab + r"\s*\(\s*([^()\n]{1,40}?)\s*\)", t, re.I):
            out.add(m.group(1).strip())
        for m in re.finditer(r"((?:" + _WORD + r"\s+){0,2}" + _WORD + r")\s*\(\s*" + lab + r"\s*\)", t, re.I):
            words = m.group(1).split()
            out.update(" ".join(words[-n:]) for n in range(1, len(words) + 1))
    return {x for x in out if len(loose_key(x)) >= 2 and loose_key(x) != loose_key(label)}


def mentions_concept(text: str, label: str, documents: list[str] = (), _memo: dict | None = None) -> bool:
    """text 가 label 또는 문서가 같은 것이라고 적은 다른 표기를 말한다."""
    if mentions(text, label):
        return True
    if not documents:
        return False
    if _memo is not None:
        if label not in _memo:
            _memo[label] = declared_aliases(label, list(documents))
        aliases = _memo[label]
    else:
        aliases = declared_aliases(label, list(documents))
    return any(mentions(text, a) for a in aliases)
