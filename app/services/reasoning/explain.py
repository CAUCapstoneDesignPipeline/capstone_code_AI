"""A-B-C 경로 설명의 검증 (verifyExplanation, profile explanation-det-2). 결정적.

설명(새 지식 제안)은 항상 LLM 이 쓴다 (explain_llm.py, generateExplanation). 여기서는 생성기와 다른 기준으로
주장마다 본다: 인용이 경로의 어느 단계 근거인가, 주장이 그 단계의 두 개념을 말하는가, 인용 원문에 두 개념이
실제로 나오는가, 결론이 두 전제의 근거를 모두 잇는가, 합성 규칙이 있는가.
개념은 경로의 대표 표기 하나뿐이지만 두 노트가 다른 표기(번역어·약어)를 쓸 수 있으므로, 문서가 괄호로 같은 것이라
적은 표기도 받는다 (explanation-det-2).
하나라도 fail 이면 overall fail, uncertain·빈 단계(gap)가 있으면 uncertain, 아니면 pass.
합성 규칙의 최종 적용과 후보/가설 결정은 BE 가 다시 확인한다 (계약 §5).
"""

from dataclasses import dataclass, field

from .text import mentions_concept


@dataclass(frozen=True)
class Step:
    order: int
    from_label: str
    to_label: str
    code: str
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class ClaimOut:
    ref: str
    kind: str
    text: str
    evidence_ids: tuple[str, ...]


@dataclass
class CheckResult:
    overall: str
    verdicts: list[tuple[str, str, str, tuple[str, ...]]] = field(default_factory=list)   # (ref, outcome, reason, ids)
    gaps: list[tuple[int | None, str]] = field(default_factory=list)


def verify(a: str, b: str, c: str, s1: Step, s2: Step, proposed: str | None, rule_result: str | None,
           claims: list[ClaimOut], evidence_quote: dict[str, str], documents: list[str] = ()) -> CheckResult:
    """claims 의 evidenceId 는 요청 evidence 안에 있다고 이미 확인했다 (없으면 422).
    documents: 요청 문서 원문. 문서가 괄호로 같은 것이라 적은 표기(스레드(thread))도 그 개념을 말한 것으로 본다."""
    memo: dict = {}

    def mentions(text, label):
        return mentions_concept(text, label, documents, memo)

    res = CheckResult("pass")
    steps = {1: s1, 2: s2}
    covered = set()
    has_conclusion = False
    for cl in claims:
        ids = set(cl.evidence_ids)
        step_hits = [s for s in (s1, s2) if ids & set(s.evidence_ids)]
        outside = ids - set(s1.evidence_ids) - set(s2.evidence_ids)

        if cl.kind == "assumption":
            res.verdicts.append((cl.ref, "uncertain", "원문 근거가 없는 가정입니다.", cl.evidence_ids))
            continue

        if cl.kind == "premise":
            if not step_hits:
                res.verdicts.append((cl.ref, "fail", "경로의 어느 단계 근거도 인용하지 않습니다.", cl.evidence_ids))
                continue
            s = max(step_hits, key=lambda s: (mentions(cl.text, s.from_label) + mentions(cl.text, s.to_label), -s.order))
            if not (mentions(cl.text, s.from_label) and mentions(cl.text, s.to_label)):
                res.verdicts.append((cl.ref, "fail", f"주장이 단계 {s.order}의 두 개념 '{s.from_label}', '{s.to_label}'을(를) "
                                                     "함께 말하지 않습니다.", cl.evidence_ids))
                continue
            quotes = [evidence_quote[x] for x in cl.evidence_ids if x in s.evidence_ids]
            if not any(mentions(q, s.from_label) and mentions(q, s.to_label) for q in quotes) and not (
                    any(mentions(q, s.from_label) for q in quotes) and any(mentions(q, s.to_label) for q in quotes)):
                res.verdicts.append((cl.ref, "uncertain", "인용한 원문에 두 개념이 모두 나오지는 않습니다.", cl.evidence_ids))
                res.gaps.append((s.order, f"단계 {s.order}의 근거 원문에 '{s.from_label}'·'{s.to_label}'이(가) "
                                          "함께 나오지 않습니다."))
                covered.add(s.order)
                continue
            covered.add(s.order)
            if outside:
                res.verdicts.append((cl.ref, "uncertain", "경로 밖의 근거도 인용합니다.", cl.evidence_ids))
            else:
                res.verdicts.append((cl.ref, "pass", f"단계 {s.order}의 근거 원문에 두 개념이 나오고 주장이 그 관계를 말합니다.",
                                     cl.evidence_ids))
            continue

        if cl.kind == "condition":
            if not step_hits:
                res.verdicts.append((cl.ref, "fail", "경로의 근거를 인용하지 않는 조건입니다.", cl.evidence_ids))
            elif any(mentions(evidence_quote[x], b) for x in cl.evidence_ids) and mentions(cl.text, b):
                res.verdicts.append((cl.ref, "pass", f"조건이 매개 개념 '{b}'과(와) 그 근거를 가리킵니다.", cl.evidence_ids))
            else:
                res.verdicts.append((cl.ref, "uncertain", "조건이 경로의 개념을 분명히 가리키지 않습니다.", cl.evidence_ids))
            continue

        # conclusion
        has_conclusion = True
        if len(step_hits) < 2:
            res.verdicts.append((cl.ref, "fail", "결론이 두 전제의 근거를 모두 잇지 않습니다.", cl.evidence_ids))
        elif not (mentions(cl.text, a) and mentions(cl.text, c)):
            res.verdicts.append((cl.ref, "fail", f"결론이 시작 개념 '{a}'과(와) 끝 개념 '{c}'을(를) 함께 말하지 않습니다.",
                                 cl.evidence_ids))
        elif proposed is None or rule_result is None:
            res.verdicts.append((cl.ref, "uncertain", "합성 규칙이 없어 결론은 가설입니다.", cl.evidence_ids))
            res.gaps.append((None, f"정책에 {s1.code} + {s2.code} 합성 규칙이 없습니다."))
        elif proposed != rule_result:
            res.verdicts.append((cl.ref, "fail", "결론의 유형이 합성 규칙 결과와 다릅니다.", cl.evidence_ids))
        else:
            res.verdicts.append((cl.ref, "pass", f"두 전제의 근거를 잇고 합성 규칙({s1.code} + {s2.code} → {proposed})에 맞습니다.",
                                 cl.evidence_ids))

    for order in (1, 2):
        if order not in covered:
            st = steps[order]
            res.gaps.append((order, f"단계 {order}('{st.from_label}' → '{st.to_label}')을(를) 말하는 전제가 없습니다."))
    if not has_conclusion:
        res.gaps.append((None, "결론이 없습니다."))

    outcomes = {o for _, o, _, _ in res.verdicts}
    res.overall = "fail" if "fail" in outcomes or not has_conclusion else (
        "uncertain" if "uncertain" in outcomes or res.gaps else "pass")
    res.gaps = res.gaps[:16]
    return res
