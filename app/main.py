"""내부 BE–AI API (계약: docs api/ai-api.md, api/ai-openapi.yaml).

  POST /internal/v1/extract        extractKnowledge — 결정적 개념 추출 (evidence·relations 는 빈 배열)
  GET  /internal/v1/health/live    getAiLiveness
  GET  /internal/v1/health/ready   getAiReadiness

실행: uv run ai serve   (또는 uvicorn app.main:app). 설정은 app/core/config.py.
"""

import asyncio
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool

from app.api.v1 import extract, health
from app.api.v1.deps import UUID_RE
from app.core.config import API_PREFIX, EXTRACT_PROFILE, PROFILES, Settings
from app.core.errors import AiError, error_response, install_handlers, new_request_id
from app.services.concepts import ConceptExtractor, Idf


def create_app(settings: Settings | None = None, extractor: ConceptExtractor | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.extractor is None and settings.concept_idf.exists():
            app.state.extractor = await run_in_threadpool(
                ConceptExtractor, PROFILES[EXTRACT_PROFILE], Idf.load(settings.concept_idf))
        yield

    app = FastAPI(title="CAPSTONE 내부 BE–AI API", version="0.1.0", lifespan=lifespan,
                  docs_url=f"{API_PREFIX}/docs", openapi_url=f"{API_PREFIX}/openapi.json", redoc_url=None)
    app.state.settings = settings
    app.state.extractor = extractor
    app.state.inflight = asyncio.Semaphore(settings.max_inflight)
    app.state.kiwi_lock = threading.Lock()

    @app.middleware("http")
    async def request_id(request: Request, call_next):
        """모든 응답에 X-Request-Id. 누락·형식 오류면 새 UUID 로 추적하고 라우트에서 400 을 낸다."""
        rid = request.headers.get("x-request-id")
        valid = bool(rid and UUID_RE.match(rid))
        request.state.request_id = rid if valid else new_request_id()
        request.state.request_id_valid = valid
        try:
            response = await call_next(request)
        except AiError as e:
            return error_response(e, request.state.request_id)
        response.headers["X-Request-Id"] = request.state.request_id
        return response

    install_handlers(app)
    app.include_router(extract.router, prefix=API_PREFIX)
    app.include_router(health.router, prefix=API_PREFIX)
    return app


app = create_app()
