"""한국어 표현 규칙으로 관계 유형을 정한다 (결정적).

한 문장 안에서 앞 개념 A, 뒤 개념 B 가 있을 때 A 와 B 사이 글자(between)와 B 바로 뒤 글자(after)를 본다.
규칙은 의미 이름(causes 등)을 낸다. 요청 policy 에 그 의미의 code 가 있을 때만 관계가 된다.
"""

import re
from dataclasses import dataclass

_MD = re.compile(r"[*_`~]")
TOPIC = r"^\s*(?:은|는|이|가|이란|란)\s*$"   # "A는 B..." 의 A 뒤 조사


@dataclass(frozen=True)
class Rule:
    semantic: str
    between: re.Pattern
    after: re.Pattern | None
    direction: str        # "ab": A→B, "ba": B→A
    confidence: float


def _r(semantic, between, after, direction, confidence):
    return Rule(semantic, re.compile(between), re.compile(after) if after else None, direction, confidence)


RULES = [
    _r("is_a", TOPIC, r"^\s*(?:의\s*)?(?:일종|한\s*종류|하위\s*개념|종류\s*중\s*하나)", "ab", 0.85),
    _r("part_of", TOPIC, r"^\s*(?:의\s*)?(?:일부|구성\s*요소|부분)", "ab", 0.85),
    _r("causes", TOPIC, r"^\s*(?:을|를|이|가)?\s*(?:유발|야기|초래|일으키|일으킨|만든|만들|발생시)", "ab", 0.85),
    _r("causes", r"^\s*(?:(?:이|가|은|는)\s*)?(?:때문에|로\s*인해|으로\s*인해|로\s*인하여|으로\s*인하여)\s", None, "ab", 0.8),
    _r("used_for", TOPIC, r"^\s*(?:을|를)\s*(?:위해|위한)", "ab", 0.8),
    _r("used_for", r"^\s*(?:을|를)\s*위해(?:서)?\s*$", None, "ba", 0.8),       # "B를 위해 A"
    # "A는 B에 사용된다" — 서술어여야 한다. "최근에 사용된 PTE" 같은 꾸밈말은 아니다
    _r("used_for", TOPIC, r"^\s*(?:에|에서)\s*(?:사용(?:된다|한다|돼|해|됨)|쓰인다|쓴다|활용(?:된다|한다|됨))", "ab", 0.8),
    _r("requires", TOPIC, r"^\s*(?:이|가)\s*필요", "ab", 0.8),
    # "A는 B와 달리" — 대조의 상대는 문장 주제 A 다 (cue 뒤에 오는 개념이 아니다)
    _r("contrasts_with", TOPIC, r"^\s*(?:와|과)\s*(?:달리|다르게|대조적으로|반대로)", "ab", 0.8),
    _r("example_of", TOPIC, r"^\s*(?:의\s*)?(?:예|예시|사례)(?:이|다|로|\s|$)", "ab", 0.75),
    _r("example_of", r"^\s*(?:등의|같은|와\s*같은|과\s*같은)\s*$", None, "ab", 0.7),
    # "A는 B처럼" (주제 A ~ B)
    _r("analogous_to", TOPIC, r"^\s*(?:처럼|와\s*같이|과\s*같이|와\s*비슷하게|과\s*비슷하게|와\s*유사하게|과\s*유사하게)",
       "ab", 0.7),
    # "B처럼, X는 C" (DB의 WAL처럼, OS는 저널링…) — cue 뒤에 새 주제절이 있어야 한다. "if-else처럼 분기가" 는 아니다
    # 주제어는 명사여야 한다: "같은", "사고하는" 처럼 용언 꾸밈말(하/되/있/없/같… + 은/는)이면 아니다
    _r("analogous_to", r"^\s*(?:처럼|와\s*같이|과\s*같이),?\s*[^\s,]+(?<![하되있없같많작높낮좋않])(?:은|는|도)\s*$",
       None, "ab", 0.7),
]

# 의미 이름 → policy 에서 찾을 code 후보 (앞에 있는 것 우선)
SEMANTIC_CODES = {
    "is_a": ("is_a",),
    "part_of": ("part_of",),
    "causes": ("causes", "cause"),
    "used_for": ("used_for",),
    "requires": ("requires",),
    "contrasts_with": ("contrasts_with",),
    "example_of": ("example_of",),
    "analogous_to": ("analogous_to", "similar_to"),
}


def classify(between: str, after: str) -> Rule | None:
    """처음 맞는 규칙. 마크다운 강조 기호는 지우고 본다."""
    b, a = _MD.sub("", between), _MD.sub("", after)
    for rule in RULES:
        if rule.between.search(b) and (rule.after is None or rule.after.search(a)):
            return rule
    return None


def code_for(semantic: str, policy_codes: set[str]) -> str | None:
    return next((c for c in SEMANTIC_CODES.get(semantic, ()) if c in policy_codes), None)
