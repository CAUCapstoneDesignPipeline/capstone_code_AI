"""GET /internal/v1/health/live, /health/ready. 서비스 Bearer 와 X-Request-Id 가 필요하다."""

from fastapi import APIRouter, Request

from app.core.config import PROFILES, SCHEMA_VERSION
from app.core.errors import AiError
from app.schemas.health import Liveness, Readiness

from .deps import common_checks

router = APIRouter(prefix="/health")


@router.get("/live", response_model=Liveness, operation_id="getAiLiveness")
async def live(request: Request):
    common_checks(request)
    return Liveness(status="alive", schemaVersion=SCHEMA_VERSION)


@router.get("/ready", response_model=Readiness, operation_id="getAiReadiness")
async def ready(request: Request):
    common_checks(request)
    if request.app.state.extractor is None:
        raise AiError("SERVICE_UNAVAILABLE", retry_after=30)
    # 계약은 7개 operation 의 profile (minItems 7) 을 기대하지만 지금은 extract 만 구현했다.
    return Readiness(status="ready", schemaVersion=SCHEMA_VERSION, profileVersions=sorted(PROFILES))
