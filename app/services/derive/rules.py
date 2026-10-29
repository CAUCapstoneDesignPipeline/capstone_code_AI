"""조건부 결론 도출: R1 구현 적용 · R2 비용 합성 (결정적).

  R1  알고리즘 a 가 역할 r 에서 쓰는 연산을 자료 구조 s 가 모두 제공하면 → s 로 a 의 r 을 구현할 수 있다.
      최적이거나 가장 빠르다는 뜻은 아니다. s 가 r 의 구현이라고 노트에 없으면 필요한 연산이 둘 이상이고 모두 제공할 때만 후보다.
  R2  R1 이 성립하고, a 의 연산 횟수·s 의 연산 비용·r 의 크기 상한·r 밖 비용이 모두 노트에 있으면
      → Σ 횟수 × 비용[n := 크기 상한] + 밖 비용 을 정리한 시간 상한.

상태
  conclusion  필요한 전제가 모두 원문에 있고 규칙 검사를 통과 (soundness 판정 전)
  held        필수 전제가 없다 (missing 에 무엇이 없는지). 다른 독립 결론은 그대로 둔다 (R2 가 보류여도 R1 은 남는다)
  rejected    전제가 서로 모순되거나 (한 노트는 제공한다, 다른 노트는 제공하지 않는다) 자료 구조가 연산을 제공하지 않는다
  known       이미 노트에 있는 결론이다: 한 노트 안에 필수 전제가 모두 있거나, 노트가 같은 결론을 이미 썼다 (stated_result)

두 문서의 기여: 필수 전제가 한 노트에 모두 있으면 새 결론이 아니다 (known). 결론은 둘 이상의 노트가 서로 다른 필수 전제를 낼 때만 나온다.
모델 상식이나 웹으로 빠진 전제를 채우지 않는다. 생성된 결론은 사실로 되돌려 넣지 않는다 (사실은 노트 근거에서만 온다).
"""

import hashlib
from dataclasses import dataclass, field

from . import bigo
from .facts import OPS, ROLES, SIZE_VAR, Fact
from .korean import josa

_CASE_WORD = {"amortized": "상각 ", "average": "평균 "}
RULE_VERSIONS = {"R1": "r1-implementation-1", "R2": "r2-cost-1"}


@dataclass
class Step:
    text: str
    fact_ids: list[str]
    expr: str | None = None             # 이 단계에서 계산한 식 (R2)


@dataclass
class Derivation:
    derivation_id: str
    rule: str
    status: str
    reasons: list[str]
    algorithm: tuple[str, str]          # (key, 노트 표기)
    structure: tuple[str, str]
    role: str
    conclusion: str
    premises: list[str]                 # 사실 id
    notes: list[str]                    # 전제를 낸 노트 id
    steps: list[Step] = field(default_factory=list)
    conditions: list[tuple[str, str | None]] = field(default_factory=list)   # (조건 문장, 사실 id)
    missing: list[str] = field(default_factory=list)
    result: str | None = None           # R2: O(...)
    case: str | None = None             # R2: worst | amortized | average
    depends_on: str | None = None
    duplicate_of: list[str] = field(default_factory=list)
    automation: str = "extracted"       # extracted | manual | mixed
    explanation: str = ""
    soundness: object = None            # sound_llm.SoundOut (derive-sound-1)

    @property
    def rule_version(self) -> str:
        return RULE_VERSIONS[self.rule]


class _Index:
    def __init__(self, facts: list[Fact]):
        self.by_id = {f.fact_id: f for f in facts}
        self.label: dict[str, str] = {}
        self.needs, self.uses, self.size, self.extra, self.requires = {}, {}, {}, {}, {}
        self.implements, self.op_cost, self.stated = {}, {}, {}
        for f in facts:
            self.label.setdefault(f.subject_key, f.subject)
            if f.kind == "needs_op" and not f.negated:
                self.needs.setdefault((f.subject_key, f.role), []).append(f)
            elif f.kind == "uses" and not f.negated:
                self.uses.setdefault((f.subject_key, f.role), []).append(f)
            elif f.kind == "size_bound":
                self.size.setdefault((f.subject_key, f.role), []).append(f)
            elif f.kind == "extra_cost":
                self.extra.setdefault(f.subject_key, []).append(f)
            elif f.kind == "requires":
                self.requires.setdefault(f.subject_key, []).append(f)
            elif f.kind == "implements":
                self.implements.setdefault((f.subject_key, f.role), []).append(f)
            elif f.kind == "op_cost":
                self.op_cost.setdefault((f.subject_key, f.op), []).append(f)
            elif f.kind == "stated_result":
                self.stated.setdefault((f.subject_key, f.other_key), []).append(f)
                if f.other:
                    self.label.setdefault(f.other_key, f.other)

    def structures(self) -> set[str]:
        return {k for k, _ in self.implements} | {k for k, _ in self.op_cost}


def _ops(facts) -> list[str]:
    """연산을 어휘 순서(삽입 → 꺼내기 → 키 바꾸기 …)로. 설명이 동작 순서대로 읽힌다."""
    order = list(OPS)
    return sorted({f.op for f in facts}, key=order.index)


def _did(*parts) -> str:
    return "d-" + hashlib.sha256("\x1f".join(map(str, parts)).encode()).hexdigest()[:16]


def _automation(facts: list[Fact]) -> str:
    kinds = {f.origin for f in facts}
    return kinds.pop() if len(kinds) == 1 else "mixed"


def _single_note(slots: list[list[Fact]]) -> str | None:
    """모든 필수 자리를 채우는 노트가 하나 있으면 그 노트 id (한 노트 안에서 이미 이어진다)."""
    common = None
    for s in slots:
        notes = {f.note for f in s}
        common = notes if common is None else common & notes
    return sorted(common)[0] if common else None


def _pick(slots: list[list[Fact]]) -> list[Fact]:
    """자리마다 사실 하나. 이미 고른 노트와 다른 노트의 사실을 먼저 고르지 않고, 원문 순서대로 첫 사실을 고른다 (같은 입력이면 같은 결과)."""
    return [sorted(s, key=lambda f: (f.note, f.evidence.start, f.fact_id))[0] for s in slots]


def _conditions(ix: _Index, keys: list[str]) -> list[tuple[str, str | None]]:
    out = []
    for k in keys:
        for f in sorted(ix.requires.get(k, []), key=lambda f: f.fact_id):
            out.append((f"{f.subject}: {f.condition}", f.fact_id))
    return out


def _cite(f: Fact, titles: dict[str, str]) -> str:
    return f"(노트 「{titles.get(f.note, f.note)}」)"


def r1(ix: _Index, a: str, role: str, s: str, titles: dict[str, str]) -> Derivation | None:
    a_label, s_label, role_name = ix.label.get(a, a), ix.label.get(s, s), ROLES.get(role, role)
    needs = ix.needs.get((a, role), [])
    impl = [f for f in ix.implements.get((s, role), []) if not f.negated]
    impl_no = [f for f in ix.implements.get((s, role), []) if f.negated]
    ops = _ops(needs)
    provided = {op: [f for f in ix.op_cost.get((s, op), []) if not f.negated] for op in ops}
    refused = {op: [f for f in ix.op_cost.get((s, op), []) if f.negated and not f.expr] for op in ops}
    # 후보: s 가 그 역할의 구현이라고 노트에 있거나, 필요한 연산 둘 이상을 모두 말한다.
    # 연산 하나("찾기")만 겹치는 것으로는 역할의 계약을 지킨다고 볼 수 없다 (실제 노트: 배열 ↔ 조인의 순서 있는 집합)
    if not impl and not impl_no and not (len(ops) >= 2 and all(provided[op] or refused[op] for op in ops)):
        return None
    reasons, missing, status = [], [], "conclusion"
    if impl_no and impl:
        status = "rejected"
        reasons.append(f"노트끼리 {josa(s_label, '이')} {josa(role_name, '을')} 구현하는지에 대해 서로 다르게 말해요")
    elif impl_no:
        status = "rejected"
        reasons.append(f"노트에 {josa(s_label, '은')} {role_name}의 구현이 아니라고 적혀 있어요")
    for op in ops:
        if provided[op] and refused[op]:
            status = "rejected"
            reasons.append(f"노트끼리 {josa(s_label, '이')} {OPS[op]} 연산을 제공하는지 서로 다르게 말해요")
        elif refused[op]:
            status = "rejected"
            reasons.append(f"노트에 {josa(s_label, '은')} {OPS[op]} 연산을 제공하지 않는다고 적혀 있어요")
        elif not provided[op] and status != "rejected":
            missing.append(f"{josa(s_label, '이')} {OPS[op]} 연산을 제공하는지")
    if not ops:
        missing.append(f"{josa(a_label, '이')} {role_name}에서 어떤 연산을 쓰는지")
    if missing and status != "rejected":
        status = "held"
        reasons.append("필요한 전제가 노트에 없어요: " + ", ".join(missing))

    slots = [[f for f in needs if f.op == op] for op in ops] + [provided[op] for op in ops if provided[op]]
    if not ops:
        slots = [ix.uses.get((a, role), [])] + ([impl] if impl else [])
    slots = [s_ for s_ in slots if s_]
    chosen = _pick(slots)
    if impl and status == "conclusion":
        chosen.append(_pick([impl])[0])
    if status == "rejected":
        chosen += [f for op in ops for f in refused[op]] + impl_no
    dup = [f.fact_id for f in ix.stated.get((a, s), [])]
    if status == "conclusion":
        one = _single_note(slots)
        if one:
            status = "known"
            reasons.append("한 노트 안에 필요한 전제가 모두 있어요 (두 노트를 함께 써서 얻은 결론이 아니에요)")
        elif dup:
            status = "known"
            reasons.append("다른 노트에 이미 같은 결론이 적혀 있어요")
    if status == "conclusion":
        reasons.append("두 노트의 전제로 규칙 R1 을 통과했어요")

    conclusion = f"{josa(s_label, '으로')} {a_label}의 {josa(role_name, '을')} 구현할 수 있어요"
    by_op = {f.op: f for f in chosen if f.kind == "needs_op"}
    cost = {f.op: f for f in chosen if f.kind == "op_cost"}
    steps = []
    if ops:
        steps.append(Step(f"{josa(a_label, '은')} {role_name}에서 " + ", ".join(OPS[o] for o in ops) + " 연산을 써요 "
                          + " ".join(sorted({_cite(f, titles) for f in by_op.values()})), [f.fact_id for f in by_op.values()]))
    if cost:
        steps.append(Step(f"{josa(s_label, '은')} " + ", ".join(OPS[o] for o in ops if o in cost) + " 연산을 모두 제공해요 "
                          + " ".join(sorted({_cite(f, titles) for f in cost.values()})), [f.fact_id for f in cost.values()]))
    if status in ("conclusion", "known"):
        steps.append(Step(f"그래서 {conclusion}. 가장 빠르거나 가장 좋은 구현이라는 뜻은 아니에요.", []))
    d = Derivation(_did("R1", a, role, s, *sorted(f.fact_id for f in chosen)), "R1", status, reasons, (a, a_label),
                   (s, s_label), role, conclusion, sorted({f.fact_id for f in chosen}),
                   sorted({f.note for f in chosen}), steps=steps, conditions=_conditions(ix, [a, s]), missing=missing,
                   duplicate_of=dup if status == "known" else [], automation=_automation(chosen) if chosen else "extracted")
    d.explanation = explain(d)
    return d


def _best_cost(facts: list[Fact]) -> Fact | None:
    """비용 식이 있는 사실 중 최악 비용을 먼저, 같은 경우는 원문 순서로."""
    with_expr = [f for f in facts if f.expr]
    order = {"worst": 0, None: 1, "average": 2, "amortized": 3}
    return sorted(with_expr, key=lambda f: (order.get(f.case, 4), f.note, f.evidence.start, f.fact_id))[0] if with_expr else None


def r2(ix: _Index, a: str, role: str, s: str, base: Derivation, titles: dict[str, str]) -> Derivation:
    a_label, s_label, role_name = base.algorithm[1], base.structure[1], ROLES.get(role, role)
    ops = _ops(ix.needs.get((a, role), []))
    missing, unreadable = [], []
    counts, costs = {}, {}
    for op in ops:
        with_count = [f for f in ix.needs.get((a, role), []) if f.op == op and f.expr]
        if with_count:
            counts[op] = with_count
        else:
            missing.append(f"{josa(a_label, '이')} {OPS[op]} 연산을 몇 번 하는지")
        c = _best_cost([f for f in ix.op_cost.get((s, op), []) if not f.negated])
        if c:
            costs[op] = c
        else:
            missing.append(f"{s_label}의 {OPS[op]} 연산 비용")
    size = [f for f in ix.size.get((a, role), [])]
    extra = [f for f in ix.extra.get(a, []) if f.role in (None, role)]

    parsed: dict[str, bigo.Poly] = {}

    def read(f: Fact) -> bigo.Poly | None:
        if f.fact_id not in parsed:
            try:
                parsed[f.fact_id] = bigo.parse(f.expr)
            except bigo.ParseError:
                unreadable.append(f"「{f.expr}」")
                parsed[f.fact_id] = None
        return parsed[f.fact_id]

    count_f = {op: _pick([fs])[0] for op, fs in counts.items()}
    for f in (*count_f.values(), *costs.values()):
        read(f)
    needs_size = any(SIZE_VAR in bigo.variables(parsed[c.fact_id]) for c in costs.values() if parsed.get(c.fact_id))
    size_f = _pick([size])[0] if size else None
    if needs_size and size_f is None:
        missing.append(f"{a_label}에서 {role_name} 크기의 상한")
    if size_f and needs_size:
        read(size_f)
    extra_f = _pick([extra])[0] if extra else None
    if extra_f is None:
        missing.append(f"{a_label}에서 {role_name} 연산 밖의 비용")
    else:
        read(extra_f)

    reasons, status = [], "conclusion"
    if missing or unreadable:
        status = "held"
        if missing:
            reasons.append("필요한 전제가 노트에 없어요: " + ", ".join(missing))
        if unreadable:
            reasons.append("식을 읽지 못했어요: " + ", ".join(unreadable))

    chosen = [*count_f.values(), *costs.values(), *([size_f] if size_f and needs_size else []),
              *([extra_f] if extra_f else [])]
    steps, result, case = [], None, None
    if status == "conclusion":
        bound = parsed[size_f.fact_id] if size_f and needs_size else None
        terms, shown = [], []
        for op in ops:
            per = parsed[costs[op].fact_id]
            if bound is not None:
                per = bigo.substitute(per, SIZE_VAR, bound)
            term = bigo.mul(parsed[count_f[op].fact_id], per)
            terms.append(term)
            shown.append((op, count_f[op].expr, bigo.show(per), bigo.show(term)))
        ops_total = bigo.add(*terms)
        total = bigo.simplify(bigo.add(ops_total, parsed[extra_f.fact_id]))
        result = bigo.big_o(total)
        cases = {costs[op].case for op in ops}
        case = "amortized" if "amortized" in cases else "average" if "average" in cases else "worst"
        steps.append(Step(f"{josa(a_label, '은')} {role_name}에서 " + ", ".join(
            f"{josa(OPS[op], '을')} 최대 {c}번" for op, c, _, _ in shown) + " 해요 "
            + " ".join(sorted({_cite(f, titles) for f in count_f.values()})), [f.fact_id for f in count_f.values()]))
        cost_of = lambda op: f"{josa(OPS[op], '은')} {_CASE_WORD.get(costs[op].case, '')}O({costs[op].expr})"
        steps.append(Step(f"{s_label}에서 " + ", ".join(cost_of(op) for op in ops[:-1])
                          + (", " if len(ops) > 1 else "") + josa(cost_of(ops[-1]), "예요")
                          + f" (n 은 {role_name} 크기) " + " ".join(sorted({_cite(f, titles) for f in costs.values()})),
                          [costs[op].fact_id for op in ops]))
        if bound is not None:
            steps.append(Step(f"{role_name} 크기 n 은 최대 {size_f.expr}라서 (n ≤ {size_f.expr}) 각 연산의 비용에 넣으면 "
                              + josa(", ".join(f"{OPS[op]} O({per})" for op, _, per, _ in shown), "예요") + " " + _cite(size_f, titles),
                              [size_f.fact_id], ", ".join(f"O({per})" for _, _, per, _ in shown)))
        steps.append(Step("횟수와 비용을 곱해 더하면 " + " + ".join(f"O({t})" for *_, t in shown)
                          + f" = {josa(bigo.big_o(ops_total), '예요')}", [], bigo.big_o(ops_total)))
        steps.append(Step(f"{role_name} 밖의 일에는 {josa(f'O({extra_f.expr})', '이')} 들어요 " + _cite(extra_f, titles)
                          + f". 모두 더하면 {josa(result, '예요')}", [extra_f.fact_id], result))
        stated = ix.stated.get((a, s), [])
        same = []
        for f in stated:
            if f.expr:
                try:
                    if bigo.equivalent(bigo.parse(f.expr), total):
                        same.append(f.fact_id)
                except bigo.ParseError:
                    pass
        one = _single_note([[f] for f in chosen]) if chosen else None
        if one:
            status = "known"
            reasons.append("한 노트 안에 필요한 전제가 모두 있어요 (두 노트를 함께 써서 얻은 결론이 아니에요)")
        elif same:
            status = "known"
            reasons.append("다른 노트에 이미 같은 시간 상한이 적혀 있어요")
        else:
            reasons.append("두 노트의 전제로 규칙 R2 를 통과했어요")
        dup = same if status == "known" and same else []
    else:
        dup = []
    word = {"amortized": "상각 ", "average": "평균 "}.get(case or "", "")
    conclusion = (f"{josa(s_label, '으로')} 구현한 {a_label}의 {word}시간은 {result} 이하예요" if result else
                  f"{josa(s_label, '으로')} 구현한 {a_label}의 시간 상한")
    conditions = list(base.conditions)
    if result:
        conditions.append(("상한이에요. 실제 실행이 이보다 빠를 수 있고, 가장 빠른 구현이라는 뜻은 아니에요", None))
        if case == "amortized":
            conditions.append(("연산 비용 중 상각 비용이 있어서 연산 하나하나가 아니라 전체 합에 대한 상한이에요", None))
    d = Derivation(_did("R2", a, role, s, *sorted(f.fact_id for f in chosen)), "R2", status, reasons, base.algorithm,
                   base.structure, role, conclusion, sorted({f.fact_id for f in chosen} | set(base.premises)),
                   sorted({f.note for f in chosen} | set(base.notes)), steps=steps, conditions=conditions,
                   missing=missing, result=result, case=case, depends_on=base.derivation_id, duplicate_of=dup,
                   automation=_automation([*chosen, *(ix.by_id[x] for x in base.premises)]))
    d.explanation = explain(d)
    return d


def explain(d: Derivation) -> str:
    """사용자에게 보이는 친절한 설명 (해요체). 단계마다 어느 노트에서 온 사실인지 적는다."""
    lines = [s.text for s in d.steps]
    if d.status == "held":
        lines.append("아직 결론을 내지 않았어요. " + "; ".join(d.reasons))
    elif d.status == "rejected":
        lines.append("이 결론은 버렸어요. " + "; ".join(d.reasons))
    conds = [c for c, _ in d.conditions]
    if conds and d.status in ("conclusion", "known"):
        lines.append("이 결론이 성립하는 조건: " + " / ".join(conds))
    return "\n".join(lines)


def derive(facts: list[Fact], titles: dict[str, str] | None = None) -> list[Derivation]:
    """사실들 → 도출 결과 (R1 마다, R1 이 성립하거나 이미 알려졌으면 R2 도). 같은 입력이면 같은 순서·같은 결과다."""
    titles = titles or {}
    ix = _Index(facts)
    out = []
    algos = sorted({k for k in (*ix.needs, *ix.uses)})
    for a, role in algos:
        for s in sorted(ix.structures()):
            if s == a:
                continue
            d1 = r1(ix, a, role, s, titles)
            if d1 is None:
                continue
            out.append(d1)
            if d1.status in ("conclusion", "known") and ix.needs.get((a, role)):
                out.append(r2(ix, a, role, s, d1, titles))
    return out
