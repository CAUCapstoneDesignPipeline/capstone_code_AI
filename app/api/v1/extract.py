"""POST /internal/v1/extract (extractKnowledge). 지금은 결정적 개념 추출만 하고 evidence·relations 는 빈 배열이다."""

import hashlib
import time
import unicodedata
from collections import defaultdict

from fastapi import APIRouter, Request
from fastapi.concurrency import run_in_threadpool

from app.core.config import PROFILES, SCHEMA_VERSION
from app.core.errors import AiError
from app.schemas.common import Span
from app.schemas.extract import ExtractedConcept, ExtractRequest, ExtractResponse, ResponseMeta
from app.services.concepts import Mention

from .deps import common_checks, parse_model, read_json_body, require_uuid_header

router = APIRouter()

MAX_CONCEPTS = 200                      # ExtractResponse.concepts maxItems
MAX_MENTIONS = 20                       # ExtractedConcept.mentions maxItems


@router.post("/extract", response_model=ExtractResponse, operation_id="extractKnowledge")
async def extract(request: Request):
    t0 = time.monotonic()
    common_checks(request)
    require_uuid_header(request, "X-Attempt-Id")
    idem = require_uuid_header(request, "Idempotency-Key")
    data = await read_json_body(request)
    req: ExtractRequest = parse_model(ExtractRequest, data)
    check_semantics(req, idem)

    state = request.app.state
    if state.extractor is None:
        raise AiError("SERVICE_UNAVAILABLE", retry_after=30)
    if state.inflight.locked():
        raise AiError("AI_BUSY", retry_after=1)
    async with state.inflight:
        def work():
            with state.kiwi_lock:   # 형태소 분석기 한 개를 여러 스레드가 동시에 쓰지 않게
                return state.extractor.extract(req.note.noteId, req.note.body, req.note.noteVersion)
        mentions = await run_in_threadpool(work)

    concepts = to_concepts(mentions, req.note.body)
    cfg = state.extractor.config_version
    meta = ResponseMeta(
        profileVersion=req.profileVersion, provider="local", model="kiwi-tfidf-markdown",
        modelRevision=state.extractor.method_version, promptVersion=f"rules-{cfg}", schemaVersion=SCHEMA_VERSION,
        preprocessingVersion=f"md-mask-{cfg}",
        durationMs=min(120000, int((time.monotonic() - t0) * 1000)),
    )
    return ExtractResponse(context=data["context"], meta=meta, concepts=concepts, evidence=[], relations=[])


def check_semantics(req: ExtractRequest, idempotency_key: str):
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

    if req.profileVersion not in PROFILES:
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


def to_concepts(mentions: list[Mention], body: str) -> list[ExtractedConcept]:
    """추출 결과 → 계약 형식. 점수 순(같으면 key 순)으로 최대 200개, 개념마다 위치 순으로 언급 최대 20개."""
    by_key: dict[str, list[Mention]] = defaultdict(list)
    for m in mentions:
        by_key[m.key].append(m)
    ranked = sorted(by_key.items(), key=lambda kv: (-kv[1][0].score, kv[0]))[:MAX_CONCEPTS]

    body16 = body.encode("utf-16-le")
    n16 = len(body16) // 2
    out = []
    for i, (_, ms) in enumerate(ranked, 1):
        counts: dict[str, int] = defaultdict(int)
        for m in ms:
            counts[m.surface] += 1
        label = sorted(counts, key=lambda s: (-counts[s], s))[0]   # 가장 많이 쓰인 표기, 같으면 사전순
        spans = []
        for m in sorted(ms, key=lambda m: (m.span.start, m.span.end))[:MAX_MENTIONS]:
            s, e = m.span.start, m.span.end
            # 출력 자체 검증: 범위·surrogate 경계·원문 일치. 어긋나면 모델 출력 오류다 (§3, §4).
            try:
                ok = 0 <= s < e <= n16 and body16[2 * s:2 * e].decode("utf-16-le") == m.surface
            except UnicodeDecodeError:
                ok = False
            if not ok:
                raise AiError("INVALID_MODEL_OUTPUT")
            spans.append(Span(startOffset=s, endOffset=e, quote=m.surface))
        out.append(ExtractedConcept(ref=f"c-{i}", label=label[:200], mentions=spans,
                                    confidence=round(min(ms[0].score, 2.0) / 2.0, 4)))
    return out
