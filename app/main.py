"""내부 BE–AI API (계약: docs api/ai-api.md, api/ai-openapi.yaml).

  POST /internal/v1/extract        extractKnowledge — 개념 + 관계·근거 (profile 에 따라)
  POST /internal/v1/embeddings     createEmbeddings — 개념 검색용 벡터
  POST /internal/v1/merge-decisions              decideConceptMerges — 같은 개념인가
  POST /internal/v1/verifications/bridges        verifyBridgeIdentity — 두 노트의 매개 개념이 같은가
  POST /internal/v1/verifications/edges          verifyEdgeEntailment — 간선 근거가 관계를 말하는가
  POST /internal/v1/explanations                 generateExplanation — A-B-C 경로의 근거 기반 설명 (새 지식 제안)
  POST /internal/v1/verifications/explanations   verifyExplanation — 설명의 주장별 판정
  POST /internal/v1/note-links     findNoteLinks — 노트 사이 연결·경로 후보 (계약에 없는 AI 쪽 제안)
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

from app.api.v1 import embeddings, extract, health, links, reasoning
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


def role_llm_config(role: str, default_effort: str):
    """역할별 LLM 설정. AI_{ROLE}_LLM_* → AI_LLM_* → (ANTHROPIC_API_KEY 가 있으면) anthropic 기본 모델."""
    import os
    from dataclasses import replace

    from app.services.llm import DEFAULT_MODELS, LLMConfig
    cfg = LLMConfig.from_env(role)
    if cfg is None and os.environ.get("ANTHROPIC_API_KEY"):
        cfg = LLMConfig("anthropic", DEFAULT_MODELS["anthropic"], float(os.environ.get("AI_LLM_TIMEOUT_SECONDS", "90")))
    if cfg is not None:
        cfg = replace(cfg, effort=os.environ.get(f"AI_{role.upper()}_LLM_EFFORT", default_effort))
    return cfg


def explain_llm_config():
    return role_llm_config("explain", "medium")


def create_app(settings: Settings | None = None, extractor: ConceptExtractor | None = None,
               typer=None, embedders: dict | None = None, explainer=None, edge_judge=None,
               explanation_judge=None, merge_judge=None, concept_filter=None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        st = app.state
        if st.extractor is None and settings.concept_idf.exists():
            try:
                st.extractor = await run_in_threadpool(
                    ConceptExtractor, EXTRACT_PROFILES[EXTRACT_PROFILE].concepts, Idf.load(settings.concept_idf),
                    settings.concept_cache)
            except StaleIdfError as e:   # 다른 설정으로 만든 IDF 로 조용히 돌지 않고, 준비되지 않은 상태로 둔다
                log.error("추출기를 만들지 못했다: %s", e)
        if st.typer is None and settings.relation_llm:
            from app.services.llm import make_llm
            from app.services.relations import RelationTyper
            try:
                cfg = role_llm_config("relation", "low")     # 문장 속 관계 유형 고르기는 짧은 판단이라 effort low
                if cfg is None:
                    raise ValueError("관계 LLM 제공자가 없다 (AI_RELATION_LLM_PROVIDER, AI_LLM_PROVIDER 또는 ANTHROPIC_API_KEY)")
                llm = make_llm(cfg)
                st.typer = RelationTyper(llm, settings.relation_cache)
                if st.concept_filter is None:          # 개념 거르기도 같은 관계 LLM (짧은 판단, effort low)
                    from app.services.concepts.filter_llm import ConceptFilter
                    st.concept_filter = ConceptFilter(
                        llm, settings.relation_cache / "concepts" if settings.relation_cache else None)
                log.info("관계 판정 LLM: %s · %s", cfg.provider, cfg.model)
            except Exception as e:      # 설정이 틀리면 hybrid profile 만 준비되지 않은 상태로 둔다
                log.error("관계 판정 LLM 을 만들지 못했다: %s", e)
        if st.explainer is None and settings.explain_llm:
            from app.services.llm import make_llm
            from app.services.reasoning.explain_llm import ExplainLLM
            try:
                cfg = explain_llm_config()
                if cfg is None:
                    raise ValueError("설명 LLM 제공자가 없다 (AI_EXPLAIN_LLM_PROVIDER, AI_LLM_PROVIDER 또는 ANTHROPIC_API_KEY)")
                st.explainer = ExplainLLM(make_llm(cfg), settings.explain_cache)
                log.info("설명 생성 LLM: %s · %s · effort %s", cfg.provider, cfg.model, cfg.effort)
            except Exception as e:      # explain-llm-8 만 준비되지 않은 상태로 둔다
                log.error("설명 생성 LLM 을 만들지 못했다: %s", e)
        if (st.edge_judge is None or st.explanation_judge is None or st.merge_judge is None) and settings.verify_llm:
            from app.services.llm import make_llm
            from app.services.reasoning.entailment_llm import EdgeJudge
            from app.services.reasoning.explanation_llm import ExplanationJudge
            from app.services.reasoning.identity_llm import MergeJudge
            try:
                cfg = role_llm_config("verify", "low")
                if cfg is None:
                    raise ValueError("검증 LLM 제공자가 없다 (AI_VERIFY_LLM_PROVIDER, AI_LLM_PROVIDER 또는 ANTHROPIC_API_KEY)")
                llm = make_llm(cfg)
                st.edge_judge = st.edge_judge or EdgeJudge(llm, settings.verify_cache)
                st.explanation_judge = st.explanation_judge or ExplanationJudge(
                    llm, settings.verify_cache / "explanations" if settings.verify_cache else None)
                st.merge_judge = st.merge_judge or MergeJudge(
                    llm, settings.verify_cache / "merges" if settings.verify_cache else None)
                log.info("검증 LLM (간선·설명·병합): %s · %s · effort %s", cfg.provider, cfg.model, cfg.effort)
            except Exception as e:      # edges-llm-1, explanation-hybrid-2, merge-hybrid-1 만 준비되지 않은 상태로 둔다
                log.error("검증 LLM 을 만들지 못했다: %s", e)
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
    app.state.explainer = explainer
    app.state.edge_judge = edge_judge
    app.state.explanation_judge = explanation_judge
    app.state.merge_judge = merge_judge
    app.state.concept_filter = concept_filter
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
    for router in (extract.router, embeddings.router, reasoning.router, links.router, health.router):
        app.include_router(router, prefix=API_PREFIX)
    if settings.dev_ui:                 # 개발용 playground. 로컬 노트를 읽으므로 배포에서는 켜지 않는다
        from pathlib import Path

        from fastapi.staticfiles import StaticFiles

        from app.dev.router import router as dev_router
        app.mount("/dev/static", StaticFiles(directory=Path(__file__).parent / "dev" / "static"), name="dev-static")
        app.include_router(dev_router)
    return app


app = create_app()
