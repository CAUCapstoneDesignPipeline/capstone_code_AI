"""두 occurrence 가 같은 개념인가 (decideConceptMerges · verifyBridgeIdentity). 결정적.

근거로 쓰는 것
  1. 표기: 느슨한 표기 키가 같거나, 어느 한 노트가 둘을 같은 것으로 적었다 (괄호 병기, 약어 ↔ 풀어 쓴 말)
  2. 주제: 두 문서의 topic id 가 같다
  3. 문맥: 두 노트가 함께 쓰는 명사구의 비율. 표기가 같아도 문맥이 겹치지 않으면 다른 뜻일 수 있다
  4. 임베딩 cosine (요청의 similarity): 표기가 다를 때 '애매함' 을 가르는 데만 쓴다. 그 자체로 같다고 하지 않는다

병합(merge)은 같은 주제 안의 같은 표기를 바로 같다고 본다. 매개 검증(bridge)은 노트 사이를 잇는 근거라서 더 엄격하다:
같은 표기여도 함께 다루는 개념이 충분해야 같다고 하고, 짧은 약어는 문맥 없이는 애매하다고 둔다.
"""

from dataclasses import dataclass

from app.services.links.corpus import loose_key

from .text import same_label, short_acronym, stated_alias


@dataclass(frozen=True)
class Side:
    label: str
    surface: str
    topic_id: str | None
    quote: str                  # 근거 원문
    document: str               # 그 근거가 있는 노트 원문
    context: frozenset          # 그 노트의 명사구 후보 전체 (loose_key)


@dataclass(frozen=True)
class Verdict:
    decision: str               # same | different | uncertain
    confidence: float
    reason: str


# '문맥이 겹친다' 의 기준: 두 노트가 함께 쓰는 명사구 수 ÷ 작은 쪽 노트의 명사구 수 (overlap coefficient).
# 명사구는 TF-IDF 로 거르기 전 후보 전체다. 거른 개념을 쓰면 노트가 적을 때 두 노트가 함께 쓰는 말일수록
# IDF 가 낮아 빠지므로, 노트 묶음과 무관하게 같은 기준이 되도록 후보 전체를 쓴다. 비율이라 노트 길이에도 덜 민감하다.
# 참고 측정 (개발 노트 141개): 관련 쌍 중앙값 0.16, 무작위 쌍 p90 0.087 · p95 0.116.
STRONG_RATIO, STRONG_MIN = 0.12, 3
WEAK_RATIO, WEAK_MIN = 0.06, 2
CLOSE_SIMILARITY = 0.92


def _shared(l: Side, r: Side) -> tuple[int, float]:
    names = {loose_key(n) for n in (l.label, l.surface, r.label, r.surface)}   # 판정 대상 자체는 문맥에서 뺀다
    a, b = l.context - names, r.context - names
    n = len(a & b)
    return n, (n / min(len(a), len(b)) if a and b else 0.0)


def judge(left: Side, right: Side, similarity: float, strict: bool) -> Verdict:
    """strict=False: 병합 판단, True: 노트 사이 매개 개념 검증."""
    names_equal = any(same_label(x, y) for x in (left.label, left.surface) for y in (right.label, right.surface))
    alias = not names_equal and any(stated_alias(doc, x, y)
                                    for doc in (left.document, right.document)
                                    for x in (left.label, left.surface) for y in (right.label, right.surface))
    if not (names_equal or alias):
        if similarity >= CLOSE_SIMILARITY:
            return Verdict("uncertain", 0.5, f"표기는 다르지만 임베딩이 매우 가깝습니다 (cosine {similarity:.2f}). "
                                             "원문에 같은 개념이라는 표시는 없습니다.")
        return Verdict("different", 0.7 if similarity < 0.7 else 0.55,
                       "표기가 다르고, 어느 원문에도 둘을 같은 것으로 적은 곳(괄호 병기·약어)이 없습니다.")

    how = "표기가 같습니다" if names_equal else "원문이 둘을 같은 것으로 적었습니다 (괄호 병기 또는 약어)"
    same_topic = left.topic_id is not None and left.topic_id.lower() == (right.topic_id or "").lower()
    n, ratio = _shared(left, right)
    strong = n >= STRONG_MIN and ratio >= STRONG_RATIO
    weak = n >= WEAK_MIN and ratio >= WEAK_RATIO
    ambiguous = short_acronym(left.surface) or short_acronym(right.surface)
    ctx = f"두 노트가 함께 쓰는 다른 명사구 {n}개 (작은 쪽의 {ratio:.0%})"

    if strong:
        return Verdict("same", 0.85 if same_topic else 0.8, f"{how}. {ctx}로 문맥이 겹칩니다.")
    if ambiguous:
        return Verdict("uncertain", 0.5, f"{how}. 하지만 짧은 약어이고 {ctx}뿐이라 다른 뜻일 수 있습니다.")
    if weak and (same_topic or not strict):
        return Verdict("same", 0.75 if same_topic else 0.65, f"{how}. {ctx}입니다.")
    if same_topic and not strict:
        return Verdict("same", 0.7, f"{how}. 같은 주제의 노트입니다.")
    return Verdict("uncertain", 0.5, f"{how}. 하지만 {ctx}뿐입니다{'' if same_topic else ' (주제도 다릅니다)'}.")
