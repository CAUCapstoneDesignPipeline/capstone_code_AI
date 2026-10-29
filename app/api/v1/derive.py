"""계산용 사실 추출과 조건부 결론 도출 (R1 구현 적용 · R2 비용 합성). 계약 v0.1.0 에 없는 AI 쪽 제안 operation.

  POST /facts         extractFacts         노트 하나 → 계산용 사실 (facts-llm-1). 노트를 저장할 때 한 번, 리비전마다
  POST /derivations   deriveConclusions    여러 노트의 사실 → 조건부 결론 (derive-rules-1: 규칙만, derive-sound-1: + soundness 판정)

BE 는 /facts 결과를 노트 리비전과 함께 저장하고, 분석 버튼·저장 때 관련 노트의 사실을 모아 /derivations 를 부른다.
사실의 근거는 요청 문서의 원문과 version·hash·위치·quote 가 모두 맞아야 한다 (옛 리비전의 사실은 422).
사람이 적은 사실(origin=manual)도 같은 검사를 받는다. 응답의 premiseFactIds·noteIds 로 BE 는 원문이 바뀐 결론을 재검토 상태로 돌린다.
"""

from fastapi import APIRouter, Request

from app.core.errors import AiError
from app.schemas.derive import (Condition, DeriveRequest, DeriveResponse, DeriveStep, Derivation, Dropped, Fact,
                                FactsRequest, FactsResponse, NoveltyScope, Soundness, Subject, UnusedFact)
from app.services.derive import facts as F
from app.services.derive.rules import derive
from app.services.derive.sound_llm import Premise, SoundIn, apply

from .grounding import check_documents, check_independence, check_snapshot, utf16_slice
from .reasoning import build, meta, prepare, run

router = APIRouter()


def fact_out(f: F.Fact) -> Fact:
    e = f.evidence
    return build(Fact, factId=f.fact_id, kind=f.kind, subject=f.subject[:200], subjectKey=f.subject_key, role=f.role,
                 op=f.op, expr=f.expr, case=f.case, negated=f.negated, condition=f.condition, other=f.other,
                 otherKey=f.other_key, origin=f.origin,
                 evidence={"noteId": e.note_id, "noteVersion": e.note_version, "bodySha256": e.body_sha256,
                           "startOffset": e.start, "endOffset": e.end, "quote": e.quote})


@router.post("/facts", response_model=FactsResponse, operation_id="extractFacts")
async def facts(request: Request):
    t0, data, req, profile = await prepare(request, FactsRequest, "facts")
    check_snapshot(req.note, "note")
    if req.note.noteId.lower() != req.context.noteId.lower() or req.note.noteVersion != req.context.noteVersion \
            or req.note.inputRevision != req.context.inputRevision:
        raise AiError("INVALID_REFERENCE", [("note.noteId", "invalid")])
    ex = request.app.state.fact_extractor
    n = req.note
    got = await run(request, profile, lambda: ex.extract(n.noteId, n.noteVersion, n.body, n.bodySha256,
                                                         list(req.knownSubjects)))
    if len(got.facts) > 300:                    # 잘라서 성공으로 답하지 않는다
        raise AiError("OUTPUT_LIMIT_EXCEEDED")
    out = [fact_out(f) for f in got.facts]
    return build(FactsResponse, context=data["context"], meta=meta(profile, t0, got.provider, got.model), facts=out,
                 dropped=[Dropped(reason=k, count=v) for k, v in sorted(got.dropped.items())])


def _facts_in(req, docs) -> list[F.Fact]:
    """요청 사실 → 엔진 사실. 근거가 문서 원문과 맞는지 본다 (version·hash·위치·quote)."""
    out, seen = [], set()
    for i, f in enumerate(req.facts):
        p = f"facts.{i}"
        if f.factId in seen:
            raise AiError("INVALID_REFERENCE", [(f"{p}.factId", "duplicate")])
        seen.add(f.factId)
        e = f.evidence
        doc = docs.get(e.noteId.lower())
        if doc is None:
            raise AiError("INVALID_REFERENCE", [(f"{p}.evidence.noteId", "unknown_reference")])
        if e.noteVersion != doc.noteVersion or e.bodySha256 != doc.bodySha256:
            raise AiError("INVALID_REFERENCE", [(f"{p}.evidence.{'noteVersion' if e.noteVersion != doc.noteVersion else 'bodySha256'}",
                                                 "invalid")])
        try:
            got = utf16_slice(doc.body, e.startOffset, e.endOffset)
        except UnicodeDecodeError:
            raise AiError("INVALID_SPAN", [(f"{p}.evidence", "surrogate_boundary")])
        if got is None:
            raise AiError("INVALID_SPAN", [(f"{p}.evidence", "out_of_range")])
        if got != e.quote:
            raise AiError("INVALID_SPAN", [(f"{p}.evidence.quote", "invalid")])
        out.append(F.Fact(f.factId, f.kind, f.subject, f.subjectKey,
                          F.FactEvidence(doc.noteId, e.noteVersion, e.bodySha256, e.startOffset, e.endOffset, e.quote),
                          role=f.role, op=f.op, expr=f.expr, case=f.case, negated=f.negated, condition=f.condition,
                          other=f.other, other_key=f.otherKey, origin=f.origin))
    return out


def derivation_out(d) -> Derivation:
    s = getattr(d, "soundness", None)
    return build(Derivation, derivationId=d.derivation_id, rule=d.rule, ruleVersion=d.rule_version, status=d.status,
                 reasons=d.reasons, algorithm=Subject(key=d.algorithm[0], label=d.algorithm[1][:200]),
                 structure=Subject(key=d.structure[0], label=d.structure[1][:200]), role=d.role,
                 conclusion=d.conclusion, result=d.result, case=d.case,
                 conditions=[Condition(text=t, factId=fid) for t, fid in d.conditions],
                 steps=[DeriveStep(text=st.text, factIds=st.fact_ids, expr=st.expr) for st in d.steps],
                 premiseFactIds=d.premises, noteIds=d.notes, missing=d.missing, dependsOn=d.depends_on,
                 duplicateOfFactIds=d.duplicate_of, automation=d.automation, explanation=d.explanation,
                 soundness=None if s is None else Soundness(verdict=s.verdict, reason=s.reason,
                                                            unfaithfulFactIds=list(s.unfaithful)))


@router.post("/derivations", response_model=DeriveResponse, operation_id="deriveConclusions")
async def derivations(request: Request):
    t0, data, req, profile = await prepare(request, DeriveRequest, "derive")
    docs = check_documents(req.context, req.documents)
    facts_in = _facts_in(req, docs)
    judge = request.app.state.soundness_judge if "soundness_judge" in profile.requires else None
    if judge is None and profile.requires:
        raise AiError("SERVICE_UNAVAILABLE", retry_after=30)
    if judge is not None and req.generationProvenance:      # 사실을 뽑은 모델·프롬프트와 판정 모델·프롬프트가 같으면 안 된다
        check_independence(judge.model, profile.prompt, req.generationProvenance)
    titles = {d.noteId: d.title for d in req.documents}

    def work():
        fs = F.FactSet.of(facts_in)
        out = derive(fs.facts, titles)
        provider, model = "local", None
        todo = [d for d in out if d.status == "conclusion"]
        if judge is not None and todo:
            by_id = {f.fact_id: f for f in fs.facts}
            items = [SoundIn(d.derivation_id, d.conclusion, tuple(c for c, _ in d.conditions),
                             tuple(s.text for s in d.steps),
                             tuple(Premise(x, by_id[x].text(), by_id[x].evidence.quote) for x in d.premises))
                     for d in todo]
            verdicts, provider, model = judge.judge(items)
            apply(out, {v.ref: v for v in verdicts}, by_id)
        return out, fs.problems, provider, model

    out, problems, provider, model = await run(request, profile, work)
    m = meta(profile, t0, "local+" + provider, model) if model else meta(profile, t0)
    return build(DeriveResponse, context=data["context"], meta=m, derivations=[derivation_out(d) for d in out],
                 unusedFacts=[UnusedFact(factId=p.fact_id, reason=p.reason) for p in problems],
                 noveltyScope=NoveltyScope(noteIds=sorted(d.noteId for d in req.documents)))
