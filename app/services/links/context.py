"""경로 단계의 문맥 근거: 단계 근거가 있는 markdown 절에서, 그 단계의 두 개념을 말하는 다른 문장 몇 개.

설명(새 지식)은 두 관계만으로는 "A 는 C 다" 에서 끝난다. C 의 성질·조건처럼 결론을 풀어 쓸 재료가 필요하다.
그 재료를 원문 구간으로 주면 설명의 모든 주장이 인용할 근거를 갖고, 설명 검증이 그 주장을 확인할 수 있다.
"""

import re

from app.services.relations.sentences import sentence_spans

from .corpus import NoteEntry

_HEADING = re.compile(r"^(#{1,6})\s")
MIN_CHARS, MAX_CHARS = 8, 300


def py_index(text: str, u16: int) -> int:
    """UTF-16 위치 → Python 문자 위치."""
    n = 0
    for i, ch in enumerate(text):
        if n >= u16:
            return i
        n += 2 if ord(ch) > 0xFFFF else 1
    return len(text)


def heading_path(text: str, pos: int) -> str:
    """pos 위의 제목 경로 ("DB > WAL > 장점"). 코드 블록 안의 # 줄은 제목이 아니다. 제목이 없으면 빈 문자열."""
    from app.services.concepts.markdown import mask
    masked = mask(text[:pos])
    path: list[tuple[int, str]] = []
    for line in masked.split("\n"):
        m = _HEADING.match(line)
        if m:
            level = len(m.group(1))
            path = [(lv, t) for lv, t in path if lv < level] + [(level, line[m.end():].strip(" #*"))]
    return " > ".join(t for _, t in path if t)


def section(text: str, pos: int) -> tuple[int, int]:
    """pos 가 들어 있는 절 [start, end): 바로 앞 제목부터 같은 수준 이하의 다음 제목 전까지. 제목이 없으면 문서 전체."""
    starts, p = [], 0
    lines = text.split("\n")
    for ln in lines:
        starts.append(p)
        p += len(ln) + 1
    li = max(k for k, s in enumerate(starts) if s <= pos)
    lo, level = 0, 7
    for k in range(li, -1, -1):
        m = _HEADING.match(lines[k])
        if m:
            lo, level = k, len(m.group(1))
            break
    for k in range(li + 1, len(lines)):
        m = _HEADING.match(lines[k])
        if m and len(m.group(1)) <= level:
            return starts[lo], starts[k]
    return starts[lo], len(text)


def context_spans(entry: NoteEntry, rel: int, limit: int = 4) -> list[tuple[int, int]]:
    """relations[rel] 의 근거가 있는 절에서 그 관계의 두 개념 중 하나 이상을 말하는 문장 (Python 위치, 문서 순서).
    두 개념을 모두 말하는 문장, 근거에 가까운 문장 먼저. 근거 문장 자체는 뺀다."""
    x, _, y = entry.relations[rel]
    text = entry.text
    ev = [(py_index(text, s), py_index(text, t)) for s, t, _ in entry.relation_evidence[rel]]
    if not ev:
        return []

    def mentions(cid):
        return [sp for k in entry.ids.get(cid, frozenset({cid})) for sp in entry.spans.get(k, [])]

    mx, my = mentions(x), mentions(y)
    areas = [section(text, a) for a, _ in ev]
    picked = []
    for s, e in sentence_spans(text):
        if not MIN_CHARS <= e - s <= MAX_CHARS or not any(lo <= s and e <= hi for lo, hi in areas):
            continue
        if any(s < b and a < e for a, b in ev):
            continue
        score = any(s <= a and b <= e for a, b in mx) + any(s <= a and b <= e for a, b in my)
        if score:
            dist = min(abs(s - a) for a, _ in ev)
            picked.append((-score, dist, s, e))
    picked.sort()
    return sorted((s, e) for _, _, s, e in picked[:limit])
