"""GET /internal/v1/health/live, /health/ready. 서비스 Bearer 와 X-Request-Id 가 필요하다."""

from fastapi import APIRouter, Request

from app.core.config import EXTRACT_PROFILES, SCHEMA_VERSION
from app.core.errors import AiError
from app.schemas.health import Liveness, Readiness

from .deps import common_checks

router = APIRouter(prefix="/health")


def ready_profiles(state) -> list[str]:
    """지금 요청을 받을 수 있는 profile."""
    out = []
    if state.extractor is not None:
        out += [p.id for p in EXTRACT_PROFILES.values() if p.relations != "rules+llm" or state.typer is not None]
    out += list(state.embedders)
    return sorted(out)


@router.get("/live", response_model=Liveness, operation_id="getAiLiveness")
async def live(request: Request):
    common_checks(request)
    return Liveness(status="alive", schemaVersion=SCHEMA_VERSION)


@router.get("/ready", response_model=Readiness, operation_id="getAiReadiness")
async def ready(request: Request):
    common_checks(request)
    profiles = ready_profiles(request.app.state)
    if request.app.state.extractor is None or not profiles:
        raise AiError("SERVICE_UNAVAILABLE", retry_after=30)
    # 계약은 7개 operation 의 profile (minItems 7) 을 기대하지만 지금은 extract·embeddings 만 구현했다.
    return Readiness(status="ready", schemaVersion=SCHEMA_VERSION, profileVersions=profiles)
