"""한 노트 안의 개념 사이 관계와 근거 문장 (설계 docs/concept-pipeline.md, 계약 extractKnowledge).

1. 후보 쌍 (결정적): 같은 문장 안에서 가까이 나온 서로 다른 두 개념
2. 근거 (결정적): 그 문장의 원문 위치. LLM 이 인용을 만들지 않으므로 quote 는 항상 원문 slice 다
3. 유형: 표현 규칙 → 요청 policy 의 code. 규칙이 못 정한 쌍만 (선택) LLM typer 에 묻는다
4. 선택 (결정적): 확신도 → 위치 → key 순으로 정렬해 관계·근거 한도 안에서 고른다
"""

from dataclasses import dataclass

from app.services.concepts import Mention
from app.services.concepts.markdown import Utf16Index

from .rules import classify, code_for
from .sentences import sentence_spans
from .structure import structural_candidates
from .typer import PairItem


@dataclass(frozen=True)
class Evidence:
    start16: int
    end16: int
    quote: str


@dataclass(frozen=True)
class Relation:
    source: str                     # 개념 key
    target: str
    code: str
    evidence: tuple[int, ...]       # Evidence 목록의 번호
    confidence: float
    method: str                     # "rule" | "llm"


@dataclass(frozen=True)
class Limits:
    max_between_chars: int = 40     # 두 개념 사이 글자 수
    max_pair_distance: int = 3      # 문장 안 언급 순서로 몇 개 떨어진 쌍까지
    max_evidence_per_relation: int = 8   # ExtractedRelation.evidenceRefs maxItems
    max_relations: int = 300        # ExtractResponse.relations maxItems
    max_evidence: int = 400         # ExtractResponse.evidence maxItems


@dataclass(frozen=True)
class _Occ:
    start: int
    end: int
    key: str
    surface: str


def _py_offsets(text: str, mentions: dict[str, list[Mention]]) -> list[_Occ]:
    """UTF-16 위치 → Python 문자 위치."""
    u16 = Utf16Index(text)
    back = {u16(i): i for i in range(len(text) + 1)}
    occ = [_Occ(back[m.span.start], back[m.span.end], key, m.surface) for key, ms in mentions.items() for m in ms]
    return sorted(set(occ), key=lambda o: (o.start, o.end, o.key))


def extract_relations(text: str, mentions: dict[str, list[Mention]], policy_types: list[tuple[str, str, str]],
                      typer=None, limits: Limits = Limits(), structure: bool = False) -> tuple[list[Evidence], list[Relation]]:
    """mentions: 응답에 실을 개념 key → 그 개념의 언급들. 관계는 이 개념들 사이에서만 만든다.
    structure=True 면 마크다운 구조(제목·목록·정의·비교 표)에서 찾은 관계도 더한다."""
    codes = {c for c, _, _ in policy_types}
    occs = _py_offsets(text, mentions)
    sentences = sentence_spans(text)

    # 근거는 원문 구간 (start, end) 으로 모은다. 문장일 수도, 제목 줄·표 머리일 수도 있다
    found: dict[tuple[str, str, str], dict] = {}
    pending: list[PairItem] = []
    pending_meta: dict[str, tuple[tuple[int, int], _Occ, _Occ]] = {}

    def add(src_key: str, tgt_key: str, code: str, spans, conf: float, method: str):
        if src_key == tgt_key:
            return
        rec = found.setdefault((src_key, code, tgt_key), {"evidence": set(), "confidence": 0.0, "method": method})
        rec["evidence"].update(spans)
        if conf > rec["confidence"]:
            rec["confidence"], rec["method"] = conf, method

    for s_idx, (ss, se) in enumerate(sentences):
        inside, last_end = [], -1
        for o in occs:
            if ss <= o.start and o.end <= se and o.start >= last_end:   # 겹치는 언급은 앞의 것만
                inside.append(o)
                last_end = o.end
        resolved = set()
        for i, a in enumerate(inside):
            for j in range(i + 1, min(len(inside), i + 1 + limits.max_pair_distance)):
                b = inside[j]
                if a.key == b.key:
                    continue
                between = text[a.end:b.start]
                if len(between) > limits.max_between_chars:
                    continue
                rule = classify(between, text[b.end:min(se, b.end + 20)])
                code = code_for(rule.semantic, codes) if rule else None
                if code:
                    src, tgt = (a, b) if rule.direction == "ab" else (b, a)
                    add(src.key, tgt.key, code, [(ss, se)], rule.confidence, "rule")
                    resolved.add(frozenset((a.key, b.key)))
                elif typer is not None and frozenset((a.key, b.key)) not in resolved:
                    item_id = f"s{s_idx}-{i}-{j}"
                    pending.append(PairItem(item_id, text[ss:se], a.surface, b.surface))
                    pending_meta[item_id] = ((ss, se), a, b)

    if structure:
        for c in structural_candidates(text, occs):
            code = code_for(c.semantic, codes)
            if code:
                add(c.source, c.target, code, c.evidence, c.confidence, "structure")

    if typer is not None and pending:
        decided = {frozenset((k[0], k[2])) for k in found}
        for item_id, typed in typer.type_pairs(pending, policy_types).items():
            span, a, b = pending_meta[item_id]
            if frozenset((a.key, b.key)) in decided:
                continue   # 규칙·구조가 이미 정한 쌍이면 그쪽을 따른다
            src, tgt = (a, b) if typed.a_is_source else (b, a)
            add(src.key, tgt.key, typed.code, [span], typed.confidence, "llm")

    # 같은 두 개념 사이에 구체적인 유형이 있으면 막연한 related 계열은 뺀다
    generic = {code_for("related", codes)} - {None}
    typed_pairs = {frozenset((k[0], k[2])) for k in found if k[1] not in generic}
    for k in [k for k in found if k[1] in generic and frozenset((k[0], k[2])) in typed_pairs]:
        del found[k]

    # 선택: 확신도 높은 순 → 첫 근거 위치 → key. 근거는 위치 순으로 최대 8개.
    ranked = sorted(found.items(), key=lambda kv: (-kv[1]["confidence"], min(kv[1]["evidence"]), kv[0]))
    chosen, used = [], set()
    for (src, code, tgt), rec in ranked:
        ev = sorted(rec["evidence"])[: limits.max_evidence_per_relation]
        if len(chosen) >= limits.max_relations or len(used | set(ev)) > limits.max_evidence:
            continue
        used |= set(ev)
        chosen.append((src, code, tgt, ev, rec))

    u16 = Utf16Index(text)
    order = sorted(used)
    index = {span: n for n, span in enumerate(order)}
    evidence = [Evidence(u16(s), u16(e), text[s:e]) for s, e in order]
    relations = [Relation(src, tgt, code, tuple(index[sp] for sp in ev), round(rec["confidence"], 4), rec["method"])
                 for src, code, tgt, ev, rec in chosen]
    relations.sort(key=lambda r: (r.evidence[0], r.source, r.code, r.target))
    return evidence, relations
