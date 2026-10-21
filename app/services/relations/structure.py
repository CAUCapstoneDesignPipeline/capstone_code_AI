"""마크다운 구조에서 관계 찾기 (결정적).

공부 노트는 구조가 많다: 제목 위계, 제목 아래 목록, **용어**: 정의, "X는 … Y이다", 비교 표.
사람이 노트를 훑을 때처럼 이 구조를 읽어 관계 후보를 만든다. 근거는 항상 원문 구간(제목 줄, 목록 줄, 문장, 표 머리)이다.

  1. 제목 위계     하위 제목의 개념 ↔ 상위 제목의 개념. 유형은 상위 제목의 단서어로 정한다
                   (종류 → is_a, 구성 → part_of, 예시 → example_of, 조건 → requires, 비교 → 하위끼리 contrasts_with)
                   단서가 없으면 related (policy 에 related 계열 code 가 있을 때만 남는다)
  2. 제목 아래 목록 개념으로 시작하는 항목 ↔ 그 제목의 개념 (규칙은 1과 같다)
  3. 정의 → is_a  "**X**: …한 Y", "X는 … Y이다" — 한국어는 상위 개념(Y)이 끝에 온다
                   X 가 Y 의 약어(WAL ↔ Write Ahead Logging)면 같은 개념이므로 관계로 만들지 않는다
  4. 비교 표       머리 행의 칸들이 개념이면 그 개념끼리 contrasts_with
"""

import re
from dataclasses import dataclass

from app.services.concepts.markdown import mask

from .sentences import sentence_spans


@dataclass(frozen=True)
class StructCandidate:
    source: str                         # 개념 key
    target: str
    semantic: str                       # rules.SEMANTIC_CODES 의 의미 이름
    evidence: tuple[tuple[int, int], ...]   # 원문(Python 문자) 구간
    confidence: float
    origin: str                         # heading | list | definition | genus | table


# 상위 제목의 단서어 → 하위 개념과의 관계 (의미, 하위가 주어인가)
# 단서어는 뜻이 하나로 읽히는 것만 둔다. 실제 노트에서 틀린 단서:
#   "Part 1" (강의 순서), "NFA 구성 방법" (만들기), "FP-Tree 구조", "예시: …" 아래 항목 (예시의 일부),
#   영어 "Requirements" (소프트웨어 공학의 주제 이름)
# 단서어는 제목의 마지막 단어(머리 명사)여야 한다: "Deadlock의 조건", "DNS 서버 종류" 는 되고
# "조건 커버리지", "조건부 독립", "노드 타입별 … 규칙" 은 안 된다. 끝의 괄호 설명은 무시한다.
_END = r"\s*(?:\([^)]*\))?\s*$"
_CUES = [
    ("is_a", True, re.compile(r"(?:종류|유형|분류|타입|\btypes|\bkinds|categories)" + _END, re.I)),
    ("part_of", True, re.compile(r"(?:구성\s*요소|요소|\bcomponents)" + _END, re.I)),
    ("requires", False, re.compile(r"(?:조건|요건|전제)" + _END, re.I)),
]
_VS = re.compile(r"\s(?:vs\.?|versus|대)\s", re.I)
_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t#]*$")
_LIST = re.compile(r"^([ \t]*)(?:[-*+]|\d+[.)])[ \t]+(.*)$")
_DEFINITION = re.compile(r"^[ \t]*(?:[-*+][ \t]+)?\*\*([^*\n]{1,60})\*\*[ \t]*[:：—–][ \t]*(\S.*)$")
_TABLE_SEP = re.compile(r"^[\s:|-]+$")
_TOPIC_AFTER = re.compile(r"^\**\s*(?:은|는|이란|란)\s")
# 정의문의 끝: Y 뒤에 서술격 조사·마침표만 온다 ("…한 Y이다.", "…를 말한다" 는 Y 가 '것' 이라 개념이 아님)
_GENUS_END = re.compile(r"^\**\s*(?:이다|다|입니다)\s*[.!]?\s*$")         # 서술격 조사가 있어야 정의문이다
_CLAUSE = re.compile(r"이고|이며|으며|지만|면서|는데")                         # 두 절짜리 문장은 끝의 Y 가 X 의 상위가 아니다
_MODIFIER = re.compile(r"(?:는|은|한|된|인|적인|의|던|할|될|되는|하는)\s*\**\s*$")   # "…하는 Y", "…의 Y" 꾸밈말
_DEF_END = re.compile(r"^[\s.)\]」』*]*(?:이다|다|입니다|임)?[\s.]*$")
_MD = re.compile(r"[*_`]")
# 개념이 아니라 번호·단계 표시인 것 ("Phase 1", "2. 소프트웨어", "1️⃣ …")
_ENUM = re.compile(r"^(?:\d+[.)]|(?:phase|step|part|stage|case|round|lecture|chapter|level|week|단계)\s*\d)", re.I)
_SKIP_HEADING = re.compile(r"목차|차례|참고\s*문헌|참고\s*자료|출처|references|table of contents|\bcontents\b", re.I)
# 비교 표: 첫 칸 머리가 비교 기준을 뜻해야 한다 (TID·ID 같은 데이터 표는 아니다)
_COMPARE_HEAD = re.compile(r"항목|비교|관점|구분|특징|기준|분류|연산|측면|aspect|criteri|feature|property|metric", re.I)
# 목록 항목이 '용어' 일 때: 개념 바로 뒤가 : — ( 이거나 항목이 거의 그 개념뿐
_TERM_AFTER = re.compile(r"^\**\s*(?:[:：—–(]|$)")


def _usable(o) -> bool:
    return o is not None and not _ENUM.match(o.surface) and not any(
        ord(ch) > 0xFFFF or 0x2600 <= ord(ch) <= 0x27BF or ch in "\ufe0f\u20e3" for ch in o.surface)


def _cue(title: str):
    if _VS.search(title):
        return "vs", None               # 비교 제목: 제목 안의 양쪽끼리만 대조하고, 하위 항목은 잇지 않는다
    for semantic, child_is_source, pat in _CUES:
        if pat.search(title):
            return semantic, child_is_source
    return "related", False             # 단서 없음: 상위 → 하위


def _is_acronym(short: str, long: str) -> bool:
    """WAL ↔ Write Ahead Logging. 약어와 풀어 쓴 말은 같은 개념이다.
    약어 쪽은 표기 전체가 영문 약어여야 한다 ("온라인 쇼핑몰 FP 계산" 은 FP 의 약어 표기가 아니다)."""
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9&]*", short.strip()):
        return False
    s = re.sub(r"[^A-Za-z]", "", short)
    words = [w for w in re.split(r"[\s\-_/]+", long) if w and w[0].isalpha()]
    return 2 <= len(s) == len(words) and s.lower() == "".join(w[0] for w in words).lower()


def structural_candidates(text: str, occs) -> list[StructCandidate]:
    """occs: 응답에 실을 개념의 언급들 (start, end, key, surface — Python 문자 위치, 정렬됨)."""
    masked = mask(text)
    out: list[StructCandidate] = []

    def first_occ(start: int, end: int, at_start: bool = False):
        """구간 안의 첫 개념 언급. at_start 면 구간 맨 앞(강조 기호 허용)에서 시작해야 한다."""
        for o in occs:
            if start <= o.start and o.end <= end:
                if at_start and _MD.sub("", text[start:o.start]).strip():
                    return None
                return o
            if o.start >= end:
                break
        return None

    def cue_target(parent, title_span, semantic):
        """단서어(종류·구성·예시·조건) 바로 앞의 개념. 없으면 제목의 대표 개념."""
        if semantic in ("related", "vs"):
            return parent
        ts, te = title_span
        pat = next(p for sem, _, p in _CUES if sem == semantic)
        m = pat.search(text, ts, te)
        before = [o for o in occs if ts <= o.start and o.end <= m.start() and _usable(o)] if m else []
        # 단서어 바로 앞에 붙은 개념만 ("X의 종류", "X 종류"). "Recap: Recommender Systems 유형" 의 Recap 은 아니다
        if before and re.fullmatch(r"[\s*_]*(?:의)?[\s*_]*", text[before[-1].end:m.start()]):
            return before[-1]
        return parent

    def link(parent, child, title_span, evidence, conf, origin, siblings=None):
        if parent is None or child is None:
            return
        title = text[title_span[0]:title_span[1]]
        semantic, child_is_source = _cue(title)
        if semantic == "vs":            # "A vs B" 아래 항목은 각 쪽의 속성이지 서로 대조되는 것이 아니다
            return
        parent = cue_target(parent, title_span, semantic)
        if parent.key == child.key:
            return
        src, tgt = (child, parent) if child_is_source else (parent, child)
        out.append(StructCandidate(src.key, tgt.key, semantic, evidence, conf if semantic != "related" else 0.6,
                                   origin))

    headings = []          # (level, title_span, main occ, children occs)
    list_stack = []        # (indent, occ, line_span)
    lines, pos = [], 0
    for line in masked.split("\n"):
        lines.append((pos, pos + len(line), line))
        pos += len(line) + 1

    table: list[tuple[int, int, str]] = []

    def flush_table():
        if len(table) >= 2 and _TABLE_SEP.match(table[1][2].strip()):
            hs, he, hline = table[0]
            cells, cpos = [], hs
            for part in hline.split("|"):
                cs = cpos
                cpos += len(part) + 1
                if part.strip():
                    cells.append((cs, cs + len(part)))
            if not cells or not _COMPARE_HEAD.search(text[cells[0][0]:cells[0][1]]):
                table.clear()
                return
            cols = [first_occ(cs, ce) for cs, ce in cells[1:]]       # 첫 칸은 행 이름
            cols = [c for c in cols if c is not None]
            keys = {c.key for c in cols}
            if 2 <= len(keys) <= 4 and len(keys) == len(cols):
                for i, a in enumerate(cols):
                    for b in cols[i + 1:]:
                        out.append(StructCandidate(a.key, b.key, "contrasts_with", ((hs, he),), 0.7, "table"))
        table.clear()

    for ls, le, line in lines:
        if line.lstrip().startswith("|"):
            table.append((ls, le, line))
            continue
        if table:
            flush_table()

        m = _HEADING.match(line)
        if m:
            level = len(m.group(1))
            ts, te = ls + m.start(2), ls + m.end(2)
            main = first_occ(ts, te)
            if not _usable(main) or _SKIP_HEADING.search(text[ts:te]):
                main = None
            vs = _VS.search(text, ts, te)
            if vs:
                left = [o for o in occs if ts <= o.start and o.end <= vs.start() and _usable(o)]
                right = [o for o in occs if vs.end() <= o.start and o.end <= te and _usable(o)]
                if left and right and left[-1].key != right[0].key:
                    out.append(StructCandidate(left[-1].key, right[0].key, "contrasts_with", ((ts, te),), 0.75, "vs"))
            while headings and headings[-1][0] >= level:
                headings.pop()
            parent = next((h for h in reversed(headings) if h[2] is not None), None)
            # 하위 제목이 그 자체로 분류 제목("… 에 따른 분류")이면 그 개념은 잇지 않는다
            if parent and main and _cue(text[ts:te])[0] not in ("is_a", "part_of", "requires"):
                link(parent[2], main, parent[1], (parent[1], (ts, te)), 0.75, "heading", parent[3])
                parent[3].append(main)
            headings.append((level, (ts, te), main, []))
            list_stack.clear()
            continue

        m = _LIST.match(line)
        if m:
            indent = len(m.group(1).expandtabs(4))
            cs = ls + m.start(2)
            item = first_occ(cs, le, at_start=True)
            if item is not None and not (_usable(item) and (
                    _TERM_AFTER.match(text[item.end:le]) or len(_MD.sub("", text[cs:le]).strip()) <= len(item.surface) + 4)):
                item = None                           # "서버의 인증서가 … 확인" 같은 문장 항목은 관계 단서가 아니다
            while list_stack and list_stack[-1][0] >= indent:
                list_stack.pop()
            span = (cs, le)
            if item:
                if list_stack and list_stack[-1][1] is not None:          # 중첩 목록: 바로 위 항목이 부모
                    p_indent, p_occ, p_span = list_stack[-1]
                    link(p_occ, item, p_span, (p_span, span), 0.65, "list")
                else:
                    head = next((h for h in reversed(headings) if h[2] is not None), None)
                    if head:
                        link(head[2], item, head[1], (head[1], span), 0.65, "list", head[3])
                        head[3].append(item)
            list_stack.append((indent, item, span))
        elif line.strip():
            list_stack.clear()

        m = _DEFINITION.match(line)
        if m:                                 # **X**: …한 Y  →  X is_a Y
            x = first_occ(ls + m.start(1), ls + m.end(1))
            ds, de = ls + m.start(2), le
            y = None
            for o in occs:
                if ds <= o.start and o.end <= de and _DEF_END.match(text[o.end:de]) \
                        and _MODIFIER.search(text[ds:o.start]) and not o.surface.endswith("적"):
                    y = o
            if x and y and x.key != y.key and not _is_acronym(x.surface, y.surface) \
                    and not _is_acronym(y.surface, x.surface):
                out.append(StructCandidate(x.key, y.key, "is_a", ((ls + len(line) - len(line.lstrip()), le),),
                                           0.75, "definition"))
    if table:
        flush_table()

    # "X는 … Y이다"  →  X is_a Y
    for ss, se in sentence_spans(text):
        inside = [o for o in occs if ss <= o.start and o.end <= se]
        if len(inside) < 2:
            continue
        x, y = inside[0], inside[-1]
        if _MD.sub("", text[ss:x.start]).strip() or not _TOPIC_AFTER.match(text[x.end:se]):
            continue                          # 문장이 주제어 X 로 시작하지 않는다
        if not _GENUS_END.match(text[y.end:se]) or x.key == y.key:
            continue                          # 문장이 Y(이다) 로 끝나지 않는다
        if _CLAUSE.search(text[x.end:y.start]) or y.surface.endswith("적") or not (_usable(x) and _usable(y)):
            continue
        if _is_acronym(x.surface, y.surface) or _is_acronym(y.surface, x.surface):
            continue
        out.append(StructCandidate(x.key, y.key, "is_a", ((ss, se),), 0.75, "genus"))
    return out
