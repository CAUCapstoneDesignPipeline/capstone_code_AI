"""내부 BE–AI API (계약: docs api/ai-api.md, api/ai-openapi.yaml).

  POST /internal/v1/extract        extractKnowledge — 개념 + 관계·근거 (profile 에 따라)
  POST /internal/v1/embeddings     createEmbeddings — 개념 검색용 벡터
  GET  /internal/v1/health/live    getAiLiveness
  GET  /internal/v1/health/ready   getAiReadiness

실행: uv run ai serve   (또는 uvicorn app.main:app). 설정은 app/core/config.py.
"""

import asyncio
import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool

from app.api.v1 import embeddings, extract, health
from app.api.v1.deps import UUID_RE
from app.core.config import API_PREFIX, EMBEDDING_PROFILES, EXTRACT_PROFILE, EXTRACT_PROFILES, Settings
from app.core.errors import AiError, error_response, install_handlers, new_request_id
from app.services.concepts import ConceptExtractor, Idf, StaleIdfError
from app.services.embeddings import HashEmbedder, SentenceTransformerEmbedder

log = logging.getLogger("app")


def load_embedder(profile_id: str):
    p = EMBEDDING_PROFILES[profile_id]
    if p.backend == "hash":
        return HashEmbedder(dims=p.dims)
    emb = SentenceTransformerEmbedder(p.model, p.revision, prefix=p.prefix)
    if emb.dims != p.dims:
        raise RuntimeError(f"{profile_id}: 모델 차원 {emb.dims} ≠ profile {p.dims}")
    return emb


def create_app(settings: Settings | None = None, extractor: ConceptExtractor | None = None,
               typer=None, embedders: dict | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        st = app.state
        if st.extractor is None and settings.concept_idf.exists():
            try:
                st.extractor = await run_in_threadpool(
                    ConceptExtractor, EXTRACT_PROFILES[EXTRACT_PROFILE].concepts, Idf.load(settings.concept_idf))
            except StaleIdfError as e:   # 다른 설정으로 만든 IDF 로 조용히 돌지 않고, 준비되지 않은 상태로 둔다
                log.error("추출기를 만들지 못했다: %s", e)
        if st.typer is None and settings.relation_llm:
            from app.services.llm import LLMConfig, make_llm
            from app.services.relations import RelationTyper
            try:
                cfg = LLMConfig.from_env("relation")
                if cfg is None:
                    raise ValueError("AI_LLM_PROVIDER 가 없다 (anthropic | openai | google)")
                st.typer = RelationTyper(make_llm(cfg), settings.relation_cache)
                log.info("관계 판정 LLM: %s · %s", cfg.provider, cfg.model)
            except Exception as e:      # 설정이 틀리면 hybrid profile 만 준비되지 않은 상태로 둔다
                log.error("관계 판정 LLM 을 만들지 못했다: %s", e)
        for pid in settings.embedding_profiles:
            if pid in st.embedders:
                continue
            if pid not in EMBEDDING_PROFILES:
                log.error("알 수 없는 임베딩 profile: %s", pid)
                continue
            try:
                st.embedders[pid] = await run_in_threadpool(load_embedder, pid)
            except Exception as e:      # 모델을 못 올리면 그 profile 만 준비되지 않은 상태로 둔다
                log.error("임베딩 profile %s 적재 실패: %s", pid, e)
        yield

    app = FastAPI(title="CAPSTONE 내부 BE–AI API", version="0.1.0", lifespan=lifespan,
                  docs_url=f"{API_PREFIX}/docs", openapi_url=f"{API_PREFIX}/openapi.json", redoc_url=None)
    app.state.settings = settings
    app.state.extractor = extractor
    app.state.typer = typer
    app.state.embedders = dict(embedders or {})
    app.state.inflight = asyncio.Semaphore(settings.max_inflight)
    app.state.kiwi_lock = threading.Lock()
    app.state.embed_lock = threading.Lock()

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
    for router in (extract.router, embeddings.router, health.router):
        app.include_router(router, prefix=API_PREFIX)
    if settings.dev_ui:                 # 개발용 playground. 로컬 노트를 읽으므로 배포에서는 켜지 않는다
        from pathlib import Path

        from fastapi.staticfiles import StaticFiles

        from app.dev.router import router as dev_router
        app.mount("/dev/static", StaticFiles(directory=Path(__file__).parent / "dev" / "static"), name="dev-static")
        app.include_router(dev_router)
    return app


app = create_app()
