"""여러 노트·근거를 받는 operation 의 교차 필드 검증 (ai-api.md §3 1·2·4·5·7·8·9, §5).

스키마(pydantic)로 못 잡는 것만 여기서 본다. 입력 문제는 422 (INVALID_REFERENCE · INVALID_SPAN · INVALID_POLICY ·
UNSUPPORTED_PROFILE), 식별자 불일치는 409 다. 원문 값은 오류에 넣지 않고 필드 경로와 이유만 넣는다.
"""

import hashlib
import unicodedata

from app.core.errors import AiError


def check_idempotency(idempotency_key: str, ctx):
    if idempotency_key.lower() != ctx.stageExecutionId.lower():
        raise AiError("INPUT_IDENTITY_MISMATCH", [("header.Idempotency-Key", "invalid")])


def check_policy(pol) -> tuple[set[str], dict[tuple[str, str], str]]:
    """(유형 code 집합, (firstType, secondType) → resultType). 빈 relationTypes 는 422 INVALID_POLICY (§3.5)."""
    codes = [t.code for t in pol.relationTypes]
    problems = []
    if not codes:
        problems.append(("policy.relationTypes", "missing"))
    if len(codes) != len(set(codes)):
        problems.append(("policy.relationTypes", "duplicate"))
    pairs = [(r.firstType, r.secondType) for r in pol.compositionRules]
    if len(pairs) != len(set(pairs)):
        problems.append(("policy.compositionRules", "duplicate"))
    for i, r in enumerate(pol.compositionRules):
        for f in ("firstType", "secondType", "resultType"):
            if getattr(r, f) not in codes:
                problems.append((f"policy.compositionRules.{i}.{f}", "unknown_reference"))
    if problems:
        raise AiError("INVALID_POLICY", problems)
    return set(codes), {(r.firstType, r.secondType): r.resultType for r in pol.compositionRules}


def check_snapshot(note, field: str):
    """원문은 저장된 NFC 그대로, bodySha256 은 원문 UTF-8 의 hash (§3.2). 몰래 NFC 로 바꾸지 않는다 (§4)."""
    if unicodedata.normalize("NFC", note.body) != note.body:
        raise AiError("INVALID_REFERENCE", [(f"{field}.body", "not_nfc")])
    if hashlib.sha256(note.body.encode("utf-8")).hexdigest() != note.bodySha256:
        raise AiError("INVALID_REFERENCE", [(f"{field}.bodySha256", "invalid")])


def check_documents(ctx, documents, field: str = "documents") -> dict:
    """noteId → NoteSnapshot. 트리거 노트가 들어 있으면 context 와 version 이 같아야 한다 (§3.1)."""
    out = {}
    for i, d in enumerate(documents):
        key = d.noteId.lower()
        if key in out:
            raise AiError("INVALID_REFERENCE", [(f"{field}.{i}.noteId", "duplicate")])
        check_snapshot(d, f"{field}.{i}")
        out[key] = d
    trigger = out.get(ctx.noteId.lower())
    if trigger is not None:
        bad = [f"context.{f}" for f in ("noteVersion", "inputRevision") if getattr(ctx, f) != getattr(trigger, f)]
        if bad:
            raise AiError("INVALID_REFERENCE", [(f, "invalid") for f in bad])
    return out


def utf16_slice(body: str, start: int, end: int) -> str | None:
    """UTF-16 [start, end) 의 원문. 범위 밖이면 None, surrogate 중간이면 UnicodeDecodeError."""
    b = body.encode("utf-16-le")
    if not (0 <= start < end <= len(b) // 2):
        return None
    return b[2 * start:2 * end].decode("utf-16-le")


def check_evidence(evidence, docs: dict, field: str = "evidence") -> dict:
    """evidenceId → GroundedSpan. 문서와 version·hash 가 맞고, offset 의 원문 slice 가 quote 와 같아야 한다 (§3.4)."""
    out = {}
    for i, e in enumerate(evidence):
        f = f"{field}.{i}"
        key = e.evidenceId.lower()
        if key in out:
            raise AiError("INVALID_REFERENCE", [(f"{f}.evidenceId", "duplicate")])
        doc = docs.get(e.noteId.lower())
        if doc is None:
            raise AiError("INVALID_REFERENCE", [(f"{f}.noteId", "unknown_reference")])
        if e.noteVersion != doc.noteVersion or e.bodySha256 != doc.bodySha256:
            raise AiError("INVALID_REFERENCE", [(f"{f}.{'noteVersion' if e.noteVersion != doc.noteVersion else 'bodySha256'}",
                                                 "invalid")])
        try:
            got = utf16_slice(doc.body, e.startOffset, e.endOffset)
        except UnicodeDecodeError:
            raise AiError("INVALID_SPAN", [(f, "surrogate_boundary")])
        if got is None:
            raise AiError("INVALID_SPAN", [(f, "out_of_range")])
        if got != e.quote:
            raise AiError("INVALID_SPAN", [(f, "quote_mismatch")])
        out[key] = e
    return out


def check_ids(ids: list[str], field: str, known: dict | None = None):
    """목록 안 중복 금지, known 이 있으면 그 안에 있어야 한다."""
    seen = set()
    for i, x in enumerate(ids):
        k = x.lower()
        if k in seen:
            raise AiError("INVALID_REFERENCE", [(f"{field}.{i}", "duplicate")])
        if known is not None and k not in known:
            raise AiError("INVALID_REFERENCE", [(f"{field}.{i}", "unknown_reference")])
        seen.add(k)


def check_pairs(pairs, ev: dict, docs: dict, field: str = "pairs"):
    """pairId 유일, occurrence 의 근거가 요청에 있고 surface 가 quote 안에 그대로 있으며 topicId 가 문서 주제와 같다 (§3.7)."""
    check_ids([p.pairId for p in pairs], f"{field}.pairId")
    for i, p in enumerate(pairs):
        for side in ("left", "right"):
            o = getattr(p, side)
            f = f"{field}.{i}.{side}"
            e = ev.get(o.evidenceId.lower())
            if e is None:
                raise AiError("INVALID_REFERENCE", [(f"{f}.evidenceId", "unknown_reference")])
            if o.surface not in e.quote:
                raise AiError("INVALID_REFERENCE", [(f"{f}.surface", "invalid")])
            topic = docs[e.noteId.lower()].topic.id
            if (o.topicId or "").lower() != (topic or "").lower():
                raise AiError("INVALID_REFERENCE", [(f"{f}.topicId", "invalid")])


def check_path(path, ev: dict, codes: set[str], rules: dict, field: str = "path"):
    """서로 다른 A·B·C, 단계 1(A→B)·2(B→C), 같은 B, 두 단계의 원문 노트가 다르고,
    proposedTypeCode 는 합성 규칙 결과이거나 규칙이 없으면 null (§3.8)."""
    ids = [c.conceptId.lower() for c in path.concepts]
    a, b, c = path.sourceConceptId.lower(), path.bridgeConceptId.lower(), path.targetConceptId.lower()
    if len(set(ids)) != 3 or ids != [a, b, c]:
        raise AiError("INVALID_REFERENCE", [(f"{field}.concepts", "invalid")])
    steps = sorted(path.steps, key=lambda s: s.stepOrder)
    if [s.stepOrder for s in steps] != [1, 2]:
        raise AiError("INVALID_REFERENCE", [(f"{field}.steps", "invalid")])
    s1, s2 = steps
    if (s1.fromConceptId.lower(), s1.toConceptId.lower(), s2.fromConceptId.lower(), s2.toConceptId.lower()) != (a, b, b, c):
        raise AiError("INVALID_REFERENCE", [(f"{field}.steps", "invalid")])
    if s1.relationId.lower() == s2.relationId.lower():
        raise AiError("INVALID_REFERENCE", [(f"{field}.steps.1.relationId", "duplicate")])
    notes = []
    for i, s in enumerate(steps):
        if s.typeCode not in codes:
            raise AiError("INVALID_POLICY", [(f"{field}.steps.{i}.typeCode", "unknown_reference")])
        check_ids(s.evidenceIds, f"{field}.steps.{i}.evidenceIds", ev)
        notes.append({ev[x.lower()].noteId.lower() for x in s.evidenceIds})
    if notes[0] == notes[1] and len(notes[0]) == 1:
        raise AiError("INVALID_REFERENCE", [(f"{field}.steps", "invalid")])     # 한 노트 안 경로는 연결 후보가 아니다
    expected = rules.get((s1.typeCode, s2.typeCode))
    if path.proposedTypeCode != expected:
        raise AiError("INVALID_REFERENCE", [(f"{field}.proposedTypeCode", "invalid")])
    return s1, s2


def check_independence(model: str, prompt: str, provenance, field: str = "generationProvenance"):
    """검증 profile 은 관련 생성 provenance 각각과 모델 또는 prompt 가 달라야 한다 (§5). 같으면 422."""
    for i, p in enumerate(provenance):
        if p.model == model and (p.promptVersion or "") == prompt:
            raise AiError("UNSUPPORTED_PROFILE", [(f"{field}.{i}", "invalid")])
