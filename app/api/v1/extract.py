"""POST /internal/v1/extract (extractKnowledge). 개념 + (profile 에 따라) 관계·근거."""

import asyncio
import hashlib
import time
import unicodedata
from collections import defaultdict

from fastapi import APIRouter, Request
from fastapi.concurrency import run_in_threadpool

from app.core.config import EXTRACT_PROFILES, REQUEST_TIMEOUT_S, SCHEMA_VERSION, ExtractProfile
from app.core.errors import AiError
from app.schemas.common import Span
from app.schemas.extract import (ExtractedConcept, ExtractedEvidence, ExtractedRelation, ExtractRequest,
                                 ExtractResponse, ResponseMeta)
from app.services.concepts import Mention
from app.services.concepts.filter_llm import PROMPT_VERSION as FILTER_PROMPT
from app.services.concepts.judge import PROMPT_VERSION as DECISIONS_PROMPT
from app.services.llm import LLMError
from app.services.relations import PROMPT_VERSION, extract_relations
from app.services.relations.pairs import PROMPT_VERSION as PAIRS_PROMPT

from .deps import common_checks, parse_model, read_json_body, require_uuid_header

router = APIRouter()

# LLM 오류 → 계약 오류 코드 (§6). 제공자 rate limit 은 429 로 통일
_TYPER_ERRORS = {"timeout": ("MODEL_TIMEOUT", None), "rate_limit": ("AI_BUSY", 30),
                 "auth": ("SERVICE_UNAVAILABLE", 60), "provider": ("MODEL_PROVIDER_ERROR", None),
                 "invalid_output": ("INVALID_MODEL_OUTPUT", None), "refusal": ("INVALID_MODEL_OUTPUT", None)}


@router.post("/extract", response_model=ExtractResponse, operation_id="extractKnowledge")
async def extract(request: Request):
    t0 = time.monotonic()
    common_checks(request)
    require_uuid_header(request, "X-Attempt-Id")
    idem = require_uuid_header(request, "Idempotency-Key")
    data = await read_json_body(request)
    req: ExtractRequest = parse_model(ExtractRequest, data)
    profile = check_semantics(req, idem)

    state = request.app.state
    if state.extractor is None or (profile.uses_llm and state.typer is None) or \
            (profile.concept_filter and state.concept_filter is None) or (profile.student and state.student is None) or \
            (profile.llm_concepts and state.concept_llm is None) or (profile.decisions and state.student_judge is None) or \
            (profile.uses_pairs and state.pair_relations is None):
        raise AiError("SERVICE_UNAVAILABLE", retry_after=30)
    if state.inflight.locked():
        raise AiError("AI_BUSY", retry_after=1)
    policy_types = [(t.code, t.label, t.description) for t in req.policy.relationTypes]

    def work():
        if profile.llm_concepts:                  # LLM 이 개념을 직접 뽑는다 (형태소 분석기를 쓰지 않는다)
            selected = state.concept_llm.select(req.note.noteId, req.note.noteVersion, req.note.body)
        elif profile.decisions:                   # 학생 후보를 Decisions 가 판정한다 (형태소 분석기 잠금은 후보를 만드는 동안만 — 네트워크 호출 중에는 풀어 둔다)
            selected = state.student_judge.select(state.extractor, req.note.noteId, req.note.noteVersion, req.note.body, lock=state.kiwi_lock)
        elif profile.student:
            with state.kiwi_lock:
                selected = state.student.select(state.extractor, req.note.noteId, req.note.noteVersion, req.note.body)
        else:
            with state.kiwi_lock:   # 형태소 분석기 한 개를 여러 스레드가 동시에 쓰지 않게
                mentions = state.extractor.extract(req.note.noteId, req.note.body, req.note.noteVersion)
            selected = select_concepts(mentions, profile)
            if profile.concept_filter:
                selected = filter_concepts(selected, req.note.body, state.concept_filter)
        evidence, relations = [], []
        if profile.uses_pairs:                    # 후보 쌍을 Decisions 로 판정 → 학습한 분류기 (네트워크 호출이 한 번 더 든다)
            evidence, relations = state.pair_relations.extract(req.note.body, selected, policy_types, profile.limits)
        elif profile.relations != "none":
            typer = state.typer if profile.uses_llm else None
            evidence, relations = extract_relations(req.note.body, selected, policy_types, typer, profile.limits,
                                                    structure=profile.uses_structure)
        return selected, evidence, relations

    async with state.inflight:
        try:
            selected, evidence, relations = await asyncio.wait_for(run_in_threadpool(work), REQUEST_TIMEOUT_S)
        except asyncio.TimeoutError:
            raise AiError("MODEL_TIMEOUT")
        except LLMError as e:
            code, retry = _TYPER_ERRORS[e.kind]
            raise AiError(code, retry_after=retry)

    concepts, refs = to_concepts(selected, req.note.body)
    ev_out = [ExtractedEvidence(ref=f"e-{i}", startOffset=e.start16, endOffset=e.end16, quote=e.quote)
              for i, e in enumerate(evidence, 1)]
    rel_out = [ExtractedRelation(ref=f"r-{i}", sourceRef=refs[r.source], targetRef=refs[r.target], typeCode=r.code,
                                 evidenceRefs=[f"e-{n + 1}" for n in r.evidence], confidence=r.confidence)
               for i, r in enumerate(relations, 1)]
    check_output(req.note.body, concepts, ev_out, rel_out, {c for c, _, _ in policy_types})

    cfg = state.extractor.config_version
    llm = profile.uses_llm
    if profile.student or profile.llm_concepts:
        src = state.concept_llm if profile.llm_concepts else state.student_judge if profile.decisions else None
        pairs = state.pair_relations if profile.uses_pairs else None
        ext = src or pairs                                  # 바깥 서비스를 부르는 쪽 (제공자·모델을 알린다)
        revision = (f"{src.prompt_version}+{src.model}" if profile.llm_concepts else state.student_judge.version + f"+{src.model}" if profile.decisions
                    else state.student.version) + (f"+{pairs.version}+{pairs.decisions.model}" if pairs else "")
        relations_tag = "pair-relations" if pairs else "rules-relations"
        concept_prompt = DECISIONS_PROMPT if profile.decisions else src.prompt_version if src else "student-1"
        return ExtractResponse(context=data["context"], concepts=concepts, evidence=ev_out, relations=rel_out, meta=ResponseMeta(
            profileVersion=req.profileVersion, provider=f"local+{ext.provider}" if ext else "local",
            model=f"{src.model if src else pairs.decisions.model if pairs else 'ngram-student'}+{relations_tag}",
            modelRevision=revision,
            promptVersion=concept_prompt + (f"+{PAIRS_PROMPT}" if pairs else f"+rules-{cfg}"),
            schemaVersion=SCHEMA_VERSION, preprocessingVersion=f"md-mask-{cfg}",
            durationMs=min(120000, int((time.monotonic() - t0) * 1000))))
    meta = ResponseMeta(
        profileVersion=req.profileVersion,
        provider=f"local+{state.typer.provider}" if llm else "local",
        model=(f"kiwi-tfidf-markdown+{state.typer.model}" + ("+concept-filter" if profile.concept_filter else "")
               if llm else "kiwi-tfidf-markdown"),
        modelRevision=state.extractor.method_version,
        promptVersion=(f"{PROMPT_VERSION}+{FILTER_PROMPT}+rules-{cfg}" if profile.concept_filter
                       else f"{PROMPT_VERSION}+rules-{cfg}" if llm else f"rules-{cfg}"),
        schemaVersion=SCHEMA_VERSION, preprocessingVersion=f"md-mask-{cfg}",
        durationMs=min(120000, int((time.monotonic() - t0) * 1000)),
    )
    return ExtractResponse(context=data["context"], meta=meta, concepts=concepts, evidence=ev_out, relations=rel_out)


def check_semantics(req: ExtractRequest, idempotency_key: str) -> ExtractProfile:
    """스키마로 못 잡는 교차 필드 검증 (§3). 실패는 409 또는 422."""
    ctx, note = req.context, req.note
    if idempotency_key.lower() != ctx.stageExecutionId.lower():
        raise AiError("INPUT_IDENTITY_MISMATCH", [("header.Idempotency-Key", "invalid")])

    mismatched = [f"context.{f}" for f in ("noteId", "noteVersion", "inputRevision")
                  if str(getattr(ctx, f)).lower() != str(getattr(note, f)).lower()]
    if mismatched:
        raise AiError("INVALID_REFERENCE", [(f, "invalid") for f in mismatched])

    # 원문은 저장된 NFC 그대로여야 한다. 몰래 NFC 로 바꾸지 않고 거절한다 (§4).
    if unicodedata.normalize("NFC", note.body) != note.body:
        raise AiError("INVALID_REFERENCE", [("note.body", "not_nfc")])
    if hashlib.sha256(note.body.encode("utf-8")).hexdigest() != note.bodySha256:
        raise AiError("INVALID_REFERENCE", [("note.bodySha256", "invalid")])

    profile = EXTRACT_PROFILES.get(req.profileVersion)
    if profile is None:
        raise AiError("UNSUPPORTED_PROFILE", [("profileVersion", "unknown_reference")])

    pol = req.policy
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
    return profile


def select_concepts(mentions: list[Mention], profile: ExtractProfile) -> dict[str, list[Mention]]:
    """profile 의 선택 규칙: 점수 순(같으면 key 순)으로 max_concepts 개, 개념마다 위치 순으로 max_mentions 개."""
    by_key: dict[str, list[Mention]] = defaultdict(list)
    for m in mentions:
        by_key[m.key].append(m)
    ranked = sorted(by_key.items(), key=lambda kv: (-kv[1][0].score, kv[0]))[: profile.max_concepts]
    return {k: sorted(ms, key=lambda m: (m.span.start, m.span.end))[: profile.max_mentions] for k, ms in ranked}


def concept_label(ms: list[Mention]) -> str:
    """가장 많이 쓰인 표기, 같으면 사전순."""
    counts: dict[str, int] = defaultdict(int)
    for m in ms:
        counts[m.surface] += 1
    return sorted(counts, key=lambda s: (-counts[s], s))[0]


def filter_concepts(selected: dict[str, list[Mention]], body: str, concept_filter) -> dict[str, list[Mention]]:
    """LLM 이 개념이 아니라고 한 후보를 뺀다. 순서·언급은 그대로 (extract-hybrid-3)."""
    labels = {k: concept_label(ms) for k, ms in selected.items()}
    keep = concept_filter.keep(body, list(labels.values()))
    return {k: ms for k, ms in selected.items() if labels[k] in keep}


def to_concepts(selected: dict[str, list[Mention]], body: str) -> tuple[list[ExtractedConcept], dict[str, str]]:
    out, refs = [], {}
    for i, (key, ms) in enumerate(selected.items(), 1):
        label = concept_label(ms)
        refs[key] = f"c-{i}"
        out.append(ExtractedConcept(
            ref=refs[key], label=label[:200],
            mentions=[Span(startOffset=m.span.start, endOffset=m.span.end, quote=m.surface) for m in ms],
            confidence=round(min(ms[0].score, 2.0) / 2.0, 4)))
    return out, refs


def check_output(body: str, concepts, evidence, relations, codes: set[str]):
    """보내기 전 자체 검증 (§3.3, §4): 모든 span 이 원문 UTF-16 slice 와 같고, 참조·유형이 맞아야 한다.
    어긋나면 모델 출력 오류(502)다. 잘라서 성공으로 답하지 않는다."""
    body16 = body.encode("utf-16-le")
    n16 = len(body16) // 2

    def span_ok(s, e, quote):
        try:
            return 0 <= s < e <= n16 and body16[2 * s:2 * e].decode("utf-16-le") == quote
        except UnicodeDecodeError:          # surrogate 중간 절단
            return False

    spans = [(m.startOffset, m.endOffset, m.quote) for c in concepts for m in c.mentions]
    spans += [(e.startOffset, e.endOffset, e.quote) for e in evidence]
    if not all(span_ok(*s) for s in spans):
        raise AiError("INVALID_MODEL_OUTPUT")
    concept_refs, evidence_refs = {c.ref for c in concepts}, {e.ref for e in evidence}
    for r in relations:
        if r.sourceRef not in concept_refs or r.targetRef not in concept_refs or r.sourceRef == r.targetRef \
                or r.typeCode not in codes or not set(r.evidenceRefs) <= evidence_refs \
                or len(set(r.evidenceRefs)) != len(r.evidenceRefs):
            raise AiError("INVALID_MODEL_OUTPUT")
    if len(concepts) > 200 or len(evidence) > 400 or len(relations) > 300:
        raise AiError("OUTPUT_LIMIT_EXCEEDED")
