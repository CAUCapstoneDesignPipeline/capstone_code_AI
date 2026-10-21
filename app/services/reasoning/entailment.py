"""간선의 근거가 그 관계를 말하는가 (verifyEdgeEntailment). 결정적.

생성(추출) 규칙을 다시 돌리면 같은 판단을 되풀이할 뿐이라 독립 검증이 아니다 (계약 §5). 그래서 다른 기준을 쓴다:
  1. 두 개념이 근거 안에 실제로 나오는가 (표기 차이 무시)
  2. 그 유형을 나타내는 단서 말이 근거에 있는가 (유형별 단서 사전, 추출 규칙의 문형과 다르다).
     대비(contrasts_with)는 비교 표의 같은 행에 두 개념이 나란히 있어도 된다
  3. 단서 바로 뒤에 부정(아니다·않는다·없다…)이 붙는가 → contradicted

entailed 는 1·2 가 모두 맞을 때, contradicted 는 1 이 맞고 단서가 부정될 때, 나머지는 unknown 이다.
규칙이 모르는 유형(팀이 새로 만든 code)은 단서를 모르므로 두 개념이 나와도 unknown 이다.
"""

import re
from dataclasses import dataclass

from .text import mentions, plain, semantic_of

CUES = {
    "is_a": r"종류|유형|분류|일종|하위\s*개념|형태|방식|기법|알고리즘|이다|입니다|\bis an?\b|type of|kind of",
    "part_of": r"구성|요소|포함|이루|일부|부분|속한|들어\s*있|part of|consists? of|component",
    "causes": r"유발|야기|초래|일으|발생|때문|인해|원인|결과|만든|만들|이어진|낳|cause|leads? to|results? in",
    "requires": r"필요|요구|전제|조건|요건|있어야|해야|의존|require|needs?|depends? on",
    "used_for": r"위해|위한|사용|쓰인|쓴다|쓰는|활용|용도|목적|used (?:for|to)",
    "contrasts_with": r"\bvs\.?|versus|반면|달리|다르|차이|비교|대비|대조|반대|장단점|compare",
    "example_of": r"예시|예를|예로|예\b|사례|대표적|등의|같은|e\.g\.|for example|such as",
    "analogous_to": r"처럼|같이|비슷|유사|마찬가지|닮|similar|analog",
    "related": r".",
}
_NEGATION = re.compile(r"^.{0,12}?(?:아니다|아닌|아님|않는다|않다|않음|않아|없다|없음|없이|불필요|필요\s*없|"
                       r"\bnot\b|n't\b|\bnever\b)")


@dataclass(frozen=True)
class EdgeResult:
    outcome: str                # entailed | contradicted | unknown
    confidence: float
    reason: str


def check_edge(source: str, target: str, code: str, label: str, quotes: list[str]) -> EdgeResult:
    """quotes: 그 간선의 근거 원문들 (요청 evidence 의 quote)."""
    text = "\n".join(plain(q) for q in quotes)
    together = any(mentions(q, source) and mentions(q, target) for q in quotes)
    if not (mentions(text, source) and mentions(text, target)):
        missing = [x for x in (source, target) if not mentions(text, x)]
        return EdgeResult("unknown", 0.6, f"근거에 '{missing[0]}' 이(가) 나오지 않습니다.")

    sem = semantic_of(code)
    if sem is None:
        return EdgeResult("unknown", 0.4, f"'{label}' 유형을 판정할 단서 규칙이 없습니다. 두 개념은 근거에 나옵니다.")
    if sem == "contrasts_with":                 # 비교 표: 같은 머리 행·같은 열 묶음에 두 개념이 나란히 있다
        for q in quotes:
            if q.lstrip().startswith("|") and mentions(q, source) and mentions(q, target):
                return EdgeResult("entailed", 0.75, "비교 표의 같은 행에 두 개념이 나란히 있습니다.")
    cue = None
    for q in quotes:
        m = re.search(CUES[sem], plain(q), re.I)
        if m and (mentions(q, source) or mentions(q, target)):
            cue = (q, m)
            break
    if cue is None:
        return EdgeResult("unknown", 0.55, f"두 개념은 근거에 나오지만 '{label}' 관계를 나타내는 말이 없습니다.")
    q, m = cue
    after = plain(q)[m.end():]
    if sem != "related" and _NEGATION.search(after):
        return EdgeResult("contradicted", 0.7, f"근거가 '{m.group(0)}' 뒤에서 부정합니다.")
    where = "한 문장 안에" if together else "여러 근거에 나뉘어"
    return EdgeResult("entailed", 0.8 if together else 0.7,
                      f"두 개념이 {where} 나오고 '{label}' 관계를 나타내는 말('{m.group(0).strip()}')이 있습니다.")
