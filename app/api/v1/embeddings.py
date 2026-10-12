"""POST /internal/v1/embeddings (createEmbeddings). 벡터만 만든다. 저장·cosine·threshold 는 BE 책임."""

import asyncio
import math
import time
import unicodedata

from fastapi import APIRouter, Request
from fastapi.concurrency import run_in_threadpool

from app.core.config import EMBEDDING_PROFILES, EMBEDDING_TIMEOUT_S, SCHEMA_VERSION
from app.core.errors import AiError
from app.schemas.embeddings import EmbeddedItem, EmbeddingsRequest, EmbeddingsResponse
from app.schemas.extract import ResponseMeta
from app.services.embeddings import EmbeddingError

from .deps import common_checks, parse_model, read_json_body, require_uuid_header

router = APIRouter()


@router.post("/embeddings", response_model=EmbeddingsResponse, operation_id="createEmbeddings")
async def embeddings(request: Request):
    t0 = time.monotonic()
    common_checks(request)
    require_uuid_header(request, "X-Attempt-Id")
    idem = require_uuid_header(request, "Idempotency-Key")
    data = await read_json_body(request)
    req: EmbeddingsRequest = parse_model(EmbeddingsRequest, data)

    if idem.lower() != req.context.stageExecutionId.lower():
        raise AiError("INPUT_IDENTITY_MISMATCH", [("header.Idempotency-Key", "invalid")])
    if req.profileVersion not in EMBEDDING_PROFILES:
        raise AiError("UNSUPPORTED_PROFILE", [("profileVersion", "unknown_reference")])
    embedder = request.app.state.embedders.get(req.profileVersion)
    if embedder is None:                    # 등록은 됐지만 이 배포에서 켜지 않았거나 아직 적재 중
        raise AiError("SERVICE_UNAVAILABLE", retry_after=30)
    if req.expectedDimensions != embedder.dims:
        raise AiError("UNSUPPORTED_PROFILE", [("expectedDimensions", "invalid")])

    ids = [it.itemId for it in req.items]
    dups = sorted({i for i in ids if ids.count(i) > 1})
    if dups:
        raise AiError("INVALID_REFERENCE", [(f"items.{ids.index(d)}.itemId", "duplicate") for d in dups])
    not_nfc = [n for n, it in enumerate(req.items) if unicodedata.normalize("NFC", it.text) != it.text]
    if not_nfc:
        raise AiError("INVALID_REFERENCE", [(f"items.{n}.text", "not_nfc") for n in not_nfc])

    state = request.app.state
    if state.inflight.locked():
        raise AiError("AI_BUSY", retry_after=1)
    async with state.inflight:
        def work():
            with state.embed_lock:
                return embedder.embed([it.text for it in req.items])
        try:
            vectors = await asyncio.wait_for(run_in_threadpool(work), EMBEDDING_TIMEOUT_S)
        except asyncio.TimeoutError:
            raise AiError("MODEL_TIMEOUT")
        except EmbeddingError as e:
            if e.kind == "too_large":       # 토큰 한도 초과: 잘라서 성공하지 않는다 (§6)
                raise AiError("INPUT_TOO_LARGE", [(f"items.{e.index}.text", "too_large")])
            raise AiError("INVALID_MODEL_OUTPUT")

    # 출력 검증 (§3.6): id 집합·개수 일치, 차원, 유한 실수, norm > 0
    if len(vectors) != len(req.items) or any(
            len(v) != embedder.dims or not all(math.isfinite(x) for x in v) or not any(v) for v in vectors):
        raise AiError("INVALID_MODEL_OUTPUT")

    meta = ResponseMeta(
        profileVersion=req.profileVersion, provider=embedder.provider, model=embedder.model,
        modelRevision=embedder.revision, promptVersion=None, schemaVersion=SCHEMA_VERSION,
        preprocessingVersion=embedder.preprocessing_version,
        durationMs=min(120000, int((time.monotonic() - t0) * 1000)),
    )
    items = [EmbeddedItem(itemId=it.itemId, vector=v) for it, v in zip(req.items, vectors)]
    return EmbeddingsResponse(context=data["context"], meta=meta, dimensions=embedder.dims, items=items)
