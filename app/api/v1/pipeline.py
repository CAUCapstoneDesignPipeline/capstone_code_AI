"""/extract 의 기본 파이프라인 (AI 쪽 제안 operation: 계약 v0.1.0 에 없다).

BE 는 profileVersion 으로 방법을 고른다. 그 자리에 `extract-default` 를 보내면 서버가 지금의 기본 파이프라인(학생 + Decisions 개념, pairs-1 관계)으로 돌고,
응답의 meta.profileVersion 은 실제로 돈 profile id 다. 개발자는 아래 호출로 BE 를 다시 배포하지 않고 기본을 바꾼다.

  GET    /internal/v1/pipeline                    지금 기본, 고를 수 있는 profile, 준비 여부
  PUT    /internal/v1/pipeline/{profileVersion}   기본을 바꾼다 (준비된 profile 만. 아니면 503, 모르는 id 면 422)
  DELETE /internal/v1/pipeline                    AI_PIPELINE (없으면 자동) 으로 되돌린다

바꾼 값은 이 프로세스의 메모리에만 있다 (다시 시작하면 AI_PIPELINE 으로 돌아온다. 서버를 여러 개 띄웠다면 서버마다 따로 바꾼다).
profileVersion 을 직접 쓰는 요청에는 영향이 없다.
"""

import logging

from fastapi import APIRouter, Request

from app.core.config import DEFAULT_ALIAS, DEFAULT_PIPELINE, EXTRACT_PROFILES, ExtractProfile
from app.core.errors import AiError
from app.schemas.pipeline import Pipeline, PipelineProfile

from .deps import common_checks

router = APIRouter()
log = logging.getLogger("app.pipeline")


def profile_ready(state, p: ExtractProfile) -> bool:
    """이 profile 로 지금 /extract 를 받을 수 있나."""
    return (state.extractor is not None and (not p.uses_llm or state.typer is not None)
            and (not p.concept_filter or state.concept_filter is not None)
            and (not p.student or state.student is not None) and (not p.llm_concepts or state.concept_llm is not None)
            and (not p.decisions or state.student_judge is not None) and (not p.uses_pairs or state.pair_relations is not None))


def default_source(state) -> tuple[str | None, str]:
    """(기본 profile id, 출처). 정해 둔 값(PUT·AI_PIPELINE)은 준비가 안 돼도 그대로 둔다 — 조용히 다른 방법으로 돌지 않고 요청이 503 이 된다."""
    if getattr(state, "pipeline_override", None):
        return state.pipeline_override, "override"
    if state.settings.pipeline:
        return state.settings.pipeline, "env"
    for pid in DEFAULT_PIPELINE:
        if profile_ready(state, EXTRACT_PROFILES[pid]):
            return pid, "auto"
    return None, "auto"


def resolve_profile(state, profile_version: str) -> str:
    """요청의 profileVersion → 실제 profile id (alias 만 바꾼다)."""
    if profile_version != DEFAULT_ALIAS:
        return profile_version
    pid, _ = default_source(state)
    if pid is None:
        raise AiError("SERVICE_UNAVAILABLE", retry_after=30)
    if pid not in EXTRACT_PROFILES:                          # AI_PIPELINE 이 틀린 값
        raise AiError("UNSUPPORTED_PROFILE", [("profileVersion", "unknown_reference")])
    return pid


def _provider(p: ExtractProfile) -> str:
    if p.decisions or p.uses_pairs:
        return "openai"
    return "llm" if p.llm_concepts or p.uses_llm else "none"


def _concepts(p: ExtractProfile) -> str:
    if p.llm_concepts:
        return "LLM 이 노트를 읽고 직접 뽑는다"
    if p.decisions:
        return "학생 + Decisions 판정 (약 1초)"
    if p.student:
        return "학생 (학습한 모델, 수십 ms)"
    return "결정적 규칙" + (" + LLM 거르기" if p.concept_filter else "")


def _relations(p: ExtractProfile) -> str:
    if p.uses_pairs:
        return "후보 쌍마다 Decisions 판정 + 학습한 분류기 (pairs-1, 약 0.8초)"
    if p.relations == "none":
        return "없음"
    return "규칙·구조" + (" + LLM" if p.uses_llm else "") if p.relations.startswith("rules") else p.relations


def describe(state) -> Pipeline:
    pid, source = default_source(state)
    return Pipeline(
        alias=DEFAULT_ALIAS, default=pid, source=source, autoOrder=list(DEFAULT_PIPELINE),
        profiles=[PipelineProfile(profileVersion=p.id, ready=profile_ready(state, p), concepts=_concepts(p), relations=_relations(p),
                                  externalProvider=_provider(p), default=p.id == pid) for p in EXTRACT_PROFILES.values()])


@router.get("/pipeline", response_model=Pipeline, operation_id="getExtractPipeline")
async def get_pipeline(request: Request):
    common_checks(request)
    return describe(request.app.state)


@router.put("/pipeline/{profileVersion}", response_model=Pipeline, operation_id="setExtractPipeline")
async def set_pipeline(profileVersion: str, request: Request):
    common_checks(request)
    state = request.app.state
    profile = EXTRACT_PROFILES.get(profileVersion)
    if profile is None:
        raise AiError("UNSUPPORTED_PROFILE", [("profileVersion", "unknown_reference")])
    if not profile_ready(state, profile):                    # 켜져 있지 않은 profile 을 기본으로 두면 alias 요청이 모두 503 이 된다
        raise AiError("SERVICE_UNAVAILABLE", retry_after=30)
    previous = default_source(state)[0]
    state.pipeline_override = profile.id
    log.info("기본 파이프라인: %s → %s", previous, profile.id)
    return describe(state)


@router.delete("/pipeline", response_model=Pipeline, operation_id="resetExtractPipeline")
async def reset_pipeline(request: Request):
    common_checks(request)
    state = request.app.state
    previous = default_source(state)[0]
    state.pipeline_override = None
    log.info("기본 파이프라인을 되돌림: %s → %s", previous, default_source(state)[0])
    return describe(state)
