"""노트 사이 추론 operation (계약 ai-api.md).

  POST /merge-decisions            decideConceptMerges     같은 개념인가 (BE 가 찾은 후보 쌍만)
  POST /verifications/bridges      verifyBridgeIdentity    두 노트를 잇는 매개 개념이 정말 같은가
  POST /verifications/edges        verifyEdgeEntailment    간선 근거가 그 관계를 말하는가 (규칙 또는 LLM)
  POST /explanations               generateExplanation     A-B-C 경로 → 근거가 달린 새 지식 제안 (A–C). 항상 LLM
  POST /verifications/explanations verifyExplanation       설명의 주장별 판정·빈 단계

설명 생성(explain-llm-8)과 LLM 간선 검증(edges-llm-1)은 LLM, 나머지는 결정적 profile 이다 (app.services.reasoning). 순서: 공통 헤더·본문 검사 → 스키마(400/413) → 교차 필드(409/422)
→ 실행 → 응답 자체 검증(502) → 응답. 저장·병합·후보 확정은 하지 않는다 (BE 책임).
"""

import asyncio
import time

from fastapi import APIRouter, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import ValidationError

from app.core.config import REQUEST_TIMEOUT_S, SCHEMA_VERSION
from app.core.errors import AiError
from app.schemas.extract import ResponseMeta
from app.schemas.reasoning import (Claim, ClaimVerdict, EdgeVerdict, ExplainRequest, ExplainResponse, Explanation, Gap,
                                   MergeDecision, MergeRequest, MergeResponse, VerifyBridgesRequest,
                                   VerifyBridgesResponse, VerifyEdgesRequest, VerifyEdgesResponse,
                                   VerifyExplanationRequest, VerifyExplanationResponse)
from app.services.links.corpus import loose_key
from app.services.llm import LLMError
from app.services.reasoning import REASONING_PROFILES, ReasoningProfile
from app.services.reasoning.entailment import check_edge
from app.services.reasoning.entailment_llm import EdgeIn
from app.services.reasoning.explain import ClaimOut, Step, verify
from app.services.reasoning.explanation_llm import combine
from app.services.reasoning.identity import Side, judge

from .deps import common_checks, parse_model, read_json_body, require_uuid_header
from .grounding import (check_documents, check_evidence, check_idempotency, check_ids, check_independence,
                        check_pairs, check_path, check_policy)

router = APIRouter()
# LLM 오류 → 계약 오류 코드 (§6). 제공자 rate limit 은 429 로 통일
LLM_ERRORS = {"timeout": ("MODEL_TIMEOUT", None), "rate_limit": ("AI_BUSY", 30), "auth": ("SERVICE_UNAVAILABLE", 60),
              "provider": ("MODEL_PROVIDER_ERROR", None), "invalid_output": ("INVALID_MODEL_OUTPUT", None),
              "refusal": ("INVALID_MODEL_OUTPUT", None)}     # 모델이 거절: 재시도해도 같으므로 retryable=false


async def prepare(request: Request, model_cls, operation: str):
    t0 = time.monotonic()
    common_checks(request)
    require_uuid_header(request, "X-Attempt-Id")
    idem = require_uuid_header(request, "Idempotency-Key")
    data = await read_json_body(request)
    req = parse_model(model_cls, data)
    check_idempotency(idem, req.context)
    profile = REASONING_PROFILES.get(req.profileVersion)
    if profile is None or profile.operation != operation:
        raise AiError("UNSUPPORTED_PROFILE", [("profileVersion", "unknown_reference")])
    return t0, data, req, profile


async def run(request: Request, profile: ReasoningProfile, work):
    state = request.app.state
    if not profile.ready(state):
        raise AiError("SERVICE_UNAVAILABLE", retry_after=30)
    if state.inflight.locked():
        raise AiError("AI_BUSY", retry_after=1)
    async with state.inflight:
        try:
            return await asyncio.wait_for(run_in_threadpool(work), REQUEST_TIMEOUT_S)
        except asyncio.TimeoutError:
            raise AiError("MODEL_TIMEOUT")
        except LLMError as e:
            code, retry = LLM_ERRORS[e.kind]
            raise AiError(code, retry_after=retry)


def meta(profile: ReasoningProfile, t0: float, provider: str = "local", model: str | None = None) -> ResponseMeta:
    """LLM profile 은 실제로 호출한 제공자·모델 id 를 적는다. modelRevision 은 모델 id + 코드(프롬프트) hash."""
    return ResponseMeta(profileVersion=profile.id, provider=provider, model=model or profile.model,
                        modelRevision=(f"{model}@{profile.revision}" if model else profile.revision)[:120],
                        promptVersion=profile.prompt, schemaVersion=SCHEMA_VERSION,
                        preprocessingVersion="md-plain-1", durationMs=min(120000, int((time.monotonic() - t0) * 1000)))


def build(model_cls, **fields):
    """응답 자체 검증 (§3.10): 계약 스키마를 어기면 502 INVALID_MODEL_OUTPUT. 잘라서 성공으로 답하지 않는다."""
    try:
        return model_cls(**fields)
    except ValidationError as e:
        too_long = any(x["type"] in ("string_too_long", "too_long") for x in e.errors())
        raise AiError("OUTPUT_LIMIT_EXCEEDED" if too_long else "INVALID_MODEL_OUTPUT")


def note_context(state, docs: dict) -> dict[str, frozenset]:
    """noteId(소문자) → 그 노트의 명사구 후보 전체 (loose_key). IDF 와 무관해 어떤 노트 묶음에서도 같은 기준이다."""
    out = {}
    for key, d in docs.items():
        with state.kiwi_lock:
            cands = state.extractor.candidates(d.body)
        out[key] = frozenset(loose_key(c.key) for c in cands)
    return out


def _identity(state, req, docs, ev, strict: bool) -> list[MergeDecision]:
    ctx = note_context(state, {e.noteId.lower(): docs[e.noteId.lower()] for e in ev.values()})

    def side(o):
        e = ev[o.evidenceId.lower()]
        return Side(o.label, o.surface, o.topicId, e.quote, docs[e.noteId.lower()].body, ctx[e.noteId.lower()])

    out = []
    for p in req.pairs:
        v = judge(side(p.left), side(p.right), p.similarity, strict)
        out.append(MergeDecision(pairId=p.pairId, decision=v.decision, confidence=v.confidence, reason=v.reason))
    return out


def _check_one_each(ids_in: list[str], ids_out: list[str]):
    """요청 id 마다 정확히 하나 (§3.7). 빠짐·중복·추가는 출력 오류."""
    if sorted(ids_in) != sorted(ids_out):
        raise AiError("INVALID_MODEL_OUTPUT")


@router.post("/merge-decisions", response_model=MergeResponse, operation_id="decideConceptMerges")
async def merge_decisions(request: Request):
    t0, data, req, profile = await prepare(request, MergeRequest, "merge")
    docs = check_documents(req.context, req.documents)
    ev = check_evidence(req.evidence, docs)
    check_pairs(req.pairs, ev, docs)
    decisions = await run(request, profile, lambda: _identity(request.app.state, req, docs, ev, strict=False))
    _check_one_each([p.pairId for p in req.pairs], [d.pairId for d in decisions])
    return build(MergeResponse, context=data["context"], meta=meta(profile, t0), decisions=decisions)


@router.post("/verifications/bridges", response_model=VerifyBridgesResponse, operation_id="verifyBridgeIdentity")
async def verify_bridges(request: Request):
    t0, data, req, profile = await prepare(request, VerifyBridgesRequest, "bridges")
    docs = check_documents(req.context, req.documents)
    ev = check_evidence(req.evidence, docs)
    check_pairs(req.pairs, ev, docs)
    check_independence(profile.model, profile.prompt, req.generationProvenance)
    verdicts = await run(request, profile, lambda: _identity(request.app.state, req, docs, ev, strict=True))
    _check_one_each([p.pairId for p in req.pairs], [v.pairId for v in verdicts])
    return build(VerifyBridgesResponse, context=data["context"], meta=meta(profile, t0), verdicts=verdicts)


@router.post("/verifications/edges", response_model=VerifyEdgesResponse, operation_id="verifyEdgeEntailment")
async def verify_edges(request: Request):
    t0, data, req, profile = await prepare(request, VerifyEdgesRequest, "edges")
    docs = check_documents(req.context, req.documents)
    ev = check_evidence(req.evidence, docs)
    codes, _ = check_policy(req.policy)
    check_ids([e.edgeId for e in req.edges], "edges.edgeId")
    for i, e in enumerate(req.edges):
        if e.typeCode not in codes:
            raise AiError("INVALID_POLICY", [(f"edges.{i}.typeCode", "unknown_reference")])
        if e.source.conceptId.lower() == e.target.conceptId.lower():
            raise AiError("INVALID_REFERENCE", [(f"edges.{i}.target.conceptId", "invalid")])
        check_ids(e.evidenceIds, f"edges.{i}.evidenceIds", ev)
    judge = request.app.state.edge_judge if "edge_judge" in profile.requires else None
    if judge is None and profile.requires:
        raise AiError("SERVICE_UNAVAILABLE", retry_after=30)
    # 독립성은 실제로 부를 모델 id 로 본다 (LLM profile 의 model 은 'llm' 이라는 자리표시)
    check_independence(judge.model if judge else profile.model, profile.prompt, req.generationProvenance)
    types = {t.code: (t.label, t.description) for t in req.policy.relationTypes}
    provider, model = "local", None

    def rules():
        out = []
        for e in req.edges:
            r = check_edge(e.source.label, e.target.label, e.typeCode, types[e.typeCode][0],
                           [ev[x.lower()].quote for x in e.evidenceIds])
            out.append(EdgeVerdict(edgeId=e.edgeId, outcome=r.outcome, confidence=r.confidence, reason=r.reason))
        return out

    def llm():
        items = [EdgeIn(e.edgeId, e.source.label, e.target.label, e.typeCode, *types[e.typeCode],
                        tuple(ev[x.lower()].quote for x in e.evidenceIds)) for e in req.edges]
        outs, prov, mod = judge.judge(items)
        return [build(EdgeVerdict, edgeId=o.edge_id, outcome=o.outcome, confidence=o.confidence, reason=o.reason)
                for o in outs], prov, mod

    if judge is None:
        verdicts = await run(request, profile, rules)
    else:
        verdicts, provider, model = await run(request, profile, llm)
    _check_one_each([e.edgeId for e in req.edges], [v.edgeId for v in verdicts])
    return build(VerifyEdgesResponse, context=data["context"], meta=meta(profile, t0, provider, model),
                 verdicts=verdicts)


def _path_parts(req, ev):
    codes, rules = check_policy(req.policy)
    s1, s2 = check_path(req.path, ev, codes, rules)
    label = {c.conceptId.lower(): c.label for c in req.path.concepts}
    orig = {k: e.evidenceId for k, e in ev.items()}             # 응답에는 요청에 적힌 id 그대로

    def step(s):
        return Step(s.stepOrder, label[s.fromConceptId.lower()], label[s.toConceptId.lower()], s.typeCode,
                    tuple(orig[x.lower()] for x in s.evidenceIds))

    a, b, c = (label[x.lower()] for x in (req.path.sourceConceptId, req.path.bridgeConceptId,
                                          req.path.targetConceptId))
    quote = {e.evidenceId: e.quote for e in ev.values()}
    return a, b, c, step(s1), step(s2), rules, quote


def _context_ids(req, ev, s1, s2) -> list[str]:
    """문맥 근거: 단계 근거가 아니고 단계 노트에 있는 요청 근거 (요청에 적힌 id 그대로, 요청 순서)."""
    step_ids = {x.lower() for x in (*s1.evidence_ids, *s2.evidence_ids)}
    notes = {ev[x].noteId.lower() for x in step_ids}
    return [e.evidenceId for e in req.evidence if e.evidenceId.lower() not in step_ids and e.noteId.lower() in notes]


@router.post("/explanations", response_model=ExplainResponse, operation_id="generateExplanation")
async def explanations(request: Request):
    t0, data, req, profile = await prepare(request, ExplainRequest, "explain")
    docs = check_documents(req.context, req.documents)
    ev = check_evidence(req.evidence, docs)
    a, b, c, s1, s2, rules, quote = _path_parts(req, ev)
    note_of = {e.evidenceId: e.noteId.lower() for e in ev.values()}
    types = {t.code: (t.label, t.description) for t in req.policy.relationTypes}
    titles = {d.noteId.lower(): d.title for d in req.documents}
    context_ids = _context_ids(req, ev, s1, s2)
    g = await run(request, profile, lambda: request.app.state.explainer.generate(       # 설명은 항상 LLM 이 쓴다
        a, b, c, s1, s2, req.path.proposedTypeCode, types, quote, note_of, titles,
        {d.noteId.lower(): d.body for d in req.documents}, req.context.noteId.lower(),  # 트리거 노트의 용어
        tuple(context_ids)))
    summary, claims, provider, model = g.summary, g.claims, g.provider, g.model
    explanation = build(Explanation, summary=summary, claims=[
        build(Claim, claimRef=cl.ref, kind=cl.kind, text=cl.text, evidenceIds=list(cl.evidence_ids)) for cl in claims])
    # 자체 검증 (§3.9): ref 유일, 인용은 요청 근거, 결론은 두 전제의 근거를 모두 잇는다
    refs = [cl.claimRef for cl in explanation.claims]
    conclusions = [cl for cl in explanation.claims if cl.kind == "conclusion"]
    if len(refs) != len(set(refs)) or not all(x in quote for cl in explanation.claims for x in cl.evidenceIds) \
            or not conclusions or not all(set(cl.evidenceIds) & set(s.evidence_ids)
                                          for cl in conclusions for s in (s1, s2)):
        raise AiError("INVALID_MODEL_OUTPUT")
    return build(ExplainResponse, context=data["context"], meta=meta(profile, t0, provider, model),
                 explanation=explanation)


@router.post("/verifications/explanations", response_model=VerifyExplanationResponse,
             operation_id="verifyExplanation")
async def verify_explanations(request: Request):
    t0, data, req, profile = await prepare(request, VerifyExplanationRequest, "explanation")
    docs = check_documents(req.context, req.documents)
    ev = check_evidence(req.evidence, docs)
    a, b, c, s1, s2, rules, quote = _path_parts(req, ev)
    check_ids([cl.claimRef for cl in req.explanation.claims], "explanation.claims.claimRef")
    for i, cl in enumerate(req.explanation.claims):
        check_ids(cl.evidenceIds, f"explanation.claims.{i}.evidenceIds", ev)
    orig = {k: e.evidenceId for k, e in ev.items()}
    claims = [ClaimOut(cl.claimRef, cl.kind, cl.text, tuple(orig[x.lower()] for x in cl.evidenceIds))
              for cl in req.explanation.claims]

    bodies = [d.body for d in req.documents]
    # 문맥 근거를 받지 않는 profile(det-2)은 단계 근거가 아닌 인용을 모두 경로 밖으로 본다
    context_ids = frozenset(_context_ids(req, ev, s1, s2)) if profile.context_evidence else None
    judge = request.app.state.explanation_judge if "explanation_judge" in profile.requires else None
    if judge is None and profile.requires:
        raise AiError("SERVICE_UNAVAILABLE", retry_after=30)
    # 독립성은 실제로 부를 모델 id 로 본다 (LLM profile 의 model 은 'llm' 이라는 자리표시)
    check_independence(judge.model if judge else profile.model, profile.prompt, req.generationProvenance)
    provider, model = "local", None

    def check():
        det = verify(a, b, c, s1, s2, req.path.proposedTypeCode, rules.get((s1.code, s2.code)), claims, quote, bodies,
                     context_ids)
        if judge is None:
            return det, "local", None
        note_of = {e.evidenceId: e.noteId.lower() for e in ev.values()}
        j = judge.judge(a, b, c, claims, quote, note_of)
        return combine(det, j, claims, a, c), j.provider, j.model

    res, provider, model = await run(request, profile, check)
    verdicts = [build(ClaimVerdict, claimRef=ref, outcome=o, reason=r, evidenceIds=list(ids))
                for ref, o, r, ids in res.verdicts]
    _check_one_each([cl.ref for cl in claims], [v.claimRef for v in verdicts])
    gaps = [build(Gap, stepOrder=s, reason=r) for s, r in res.gaps]
    return build(VerifyExplanationResponse, context=data["context"], meta=meta(profile, t0, provider, model),
                 overall=res.overall,
                 claimVerdicts=verdicts, gaps=gaps)
