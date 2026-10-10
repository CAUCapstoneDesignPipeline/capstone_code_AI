"""내부 API 오류 계약 (docs api/ai-api.md §6). 코드마다 HTTP 상태·재시도 여부·문구가 고정이다."""

import uuid

from fastapi.responses import JSONResponse

# code: (HTTP 상태, retryable, message)
ERRORS: dict[str, tuple[int, bool, str]] = {
    "VALIDATION_FAILED": (400, False, "요청 형식이 올바르지 않습니다."),
    "SERVICE_UNAUTHENTICATED": (401, False, "서비스 인증이 필요합니다."),
    "SERVICE_FORBIDDEN": (403, False, "이 operation을 호출할 수 없습니다."),
    "INPUT_IDENTITY_MISMATCH": (409, False, "요청 식별자 또는 본문 해시가 일치하지 않습니다."),
    "INPUT_TOO_LARGE": (413, False, "분석 입력이 허용 크기를 넘었습니다."),
    "UNSUPPORTED_MEDIA_TYPE": (415, False, "비압축 UTF-8 JSON만 지원합니다."),
    "INVALID_REFERENCE": (422, False, "입력의 참조가 올바르지 않습니다."),
    "INVALID_SPAN": (422, False, "근거 구간이 원문과 일치하지 않습니다."),
    "UNSUPPORTED_PROFILE": (422, False, "지원하지 않는 추론 profile입니다."),
    "INVALID_POLICY": (422, False, "관계 유형 또는 규칙이 올바르지 않습니다."),
    "AI_BUSY": (429, True, "AI 처리 용량이 가득 찼습니다."),
    "INTERNAL": (500, True, "AI 처리 중 오류가 발생했습니다."),
    "MODEL_PROVIDER_ERROR": (502, True, "모델 제공자 호출에 실패했습니다."),
    "INVALID_MODEL_OUTPUT": (502, False, "모델 출력이 계약을 충족하지 않습니다."),
    "OUTPUT_LIMIT_EXCEEDED": (502, False, "모델 출력이 허용 크기를 넘었습니다."),
    "SERVICE_UNAVAILABLE": (503, True, "AI 서비스를 일시적으로 사용할 수 없습니다."),
    "MODEL_TIMEOUT": (504, True, "모델 처리 시간이 초과되었습니다."),
}

DETAIL_REASONS = {"missing", "invalid", "too_large", "out_of_range", "unknown_reference",
                  "quote_mismatch", "surrogate_boundary", "not_nfc", "duplicate"}


class AiError(Exception):
    def __init__(self, code: str, details: list[tuple[str, str]] | None = None, retry_after: int | None = None):
        assert code in ERRORS, code
        assert all(r in DETAIL_REASONS for _, r in details or []), details
        self.code, self.details, self.retry_after = code, details or [], retry_after
        super().__init__(code)


def error_response(err: AiError, request_id: str) -> JSONResponse:
    status, retryable, message = ERRORS[err.code]
    body = {"code": err.code, "message": message, "retryable": retryable, "requestId": request_id}
    if err.details:
        # 필드 경로와 이유만 담는다. 원문 값·본문·secret 은 넣지 않는다.
        body["details"] = [{"field": f[:160], "reason": r} for f, r in err.details[:20]]
    headers = {"X-Request-Id": request_id}
    if err.retry_after is not None and status in (429, 503):
        headers["Retry-After"] = str(max(1, min(60, err.retry_after)))
    return JSONResponse({"error": body}, status_code=status, headers=headers)


def new_request_id() -> str:
    return str(uuid.uuid4())


def install_handlers(app):
    """오류를 항상 계약 형식으로 낸다. FastAPI 기본 422 {"detail": ...} 를 쓰지 않는다 (§6)."""
    from fastapi import Request
    from fastapi.exceptions import RequestValidationError

    def rid(request: Request) -> str:
        return getattr(request.state, "request_id", None) or new_request_id()

    @app.exception_handler(AiError)
    async def ai_error_handler(request: Request, exc: AiError):
        return error_response(exc, rid(request))

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        return error_response(AiError("VALIDATION_FAILED"), rid(request))

    @app.exception_handler(Exception)
    async def internal_handler(request: Request, exc: Exception):
        return error_response(AiError("INTERNAL"), rid(request))
