"""모든 내부 operation 에 공통인 검사 (ai-api.md §2·§3·§6): 서비스 인증, 필수 헤더, 본문 바이트·JSON 검사."""

import hashlib
import hmac
import json
import re

from fastapi import Request
from pydantic import ValidationError

from app.core.config import MAX_BODY_BYTES
from app.core.errors import AiError

UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
_SHA = re.compile(r"^[0-9a-f]{64}$")
_TRACEPARENT = re.compile(r"^00-([0-9a-f]{32})-([0-9a-f]{16})-[0-9a-f]{2}$")


def common_checks(request: Request):
    """health·POST 모두: Bearer, X-Request-Id, traceparent."""
    check_auth(request)
    if not request.state.request_id_valid:
        raise AiError("VALIDATION_FAILED", [("header.X-Request-Id", "invalid")])
    check_traceparent(request)


def check_auth(request: Request):
    token = request.app.state.settings.service_token
    if not token:
        raise AiError("SERVICE_UNAVAILABLE", retry_after=30)   # 설정 오류는 503 (§6)
    scheme, _, given = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() != "bearer" or not hmac.compare_digest(given.strip().encode(), token.encode()):
        raise AiError("SERVICE_UNAUTHENTICATED")


def require_uuid_header(request: Request, name: str) -> str:
    value = request.headers.get(name)
    if not value or not UUID_RE.match(value):
        raise AiError("VALIDATION_FAILED", [(f"header.{name}", "missing" if not value else "invalid")])
    return value


def check_traceparent(request: Request):
    tp = request.headers.get("traceparent")
    if tp is None:
        return
    m = _TRACEPARENT.match(tp)
    if not m or set(m.group(1)) == {"0"} or set(m.group(2)) == {"0"}:
        raise AiError("VALIDATION_FAILED", [("header.traceparent", "invalid")])


def has_lone_surrogate(value) -> bool:
    if isinstance(value, str):
        return any(0xD800 <= ord(ch) <= 0xDFFF for ch in value)
    if isinstance(value, dict):
        return any(has_lone_surrogate(k) or has_lone_surrogate(v) for k, v in value.items())
    if isinstance(value, list):
        return any(has_lone_surrogate(v) for v in value)
    return False


def _reject_duplicate_keys(pairs):
    keys = [k for k, _ in pairs]
    if len(keys) != len(set(keys)):
        dup = next(k for k in keys if keys.count(k) > 1)
        raise AiError("VALIDATION_FAILED", [(f"body.{dup}", "duplicate")])
    return dict(pairs)


async def read_json_body(request: Request) -> dict:
    """미디어 타입 → 크기 → X-Input-SHA256 → UTF-8 → JSON(중복 key·lone surrogate 거부) 순서로 검사한다."""
    media, _, params = request.headers.get("content-type", "").partition(";")
    charset = re.search(r"charset=([\w-]+)", params, re.I)
    encoding = request.headers.get("content-encoding", "identity").lower()
    if media.strip().lower() != "application/json" or (charset and charset.group(1).lower() != "utf-8") \
            or encoding != "identity":
        raise AiError("UNSUPPORTED_MEDIA_TYPE")
    length = request.headers.get("content-length")
    if length and length.isdigit() and int(length) > MAX_BODY_BYTES:
        raise AiError("INPUT_TOO_LARGE", [("body", "too_large")])
    raw = await request.body()
    if len(raw) > MAX_BODY_BYTES:
        raise AiError("INPUT_TOO_LARGE", [("body", "too_large")])

    given = request.headers.get("x-input-sha256")
    if not given or not _SHA.match(given):
        raise AiError("VALIDATION_FAILED", [("header.X-Input-SHA256", "missing" if not given else "invalid")])
    if not hmac.compare_digest(hashlib.sha256(raw).hexdigest(), given):
        raise AiError("INPUT_IDENTITY_MISMATCH", [("header.X-Input-SHA256", "invalid")])

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise AiError("UNSUPPORTED_MEDIA_TYPE")
    try:
        data = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except json.JSONDecodeError:
        raise AiError("VALIDATION_FAILED", [("body", "invalid")])
    if has_lone_surrogate(data):
        raise AiError("VALIDATION_FAILED", [("body", "surrogate_boundary")])
    return data


_REASONS = {"missing": "missing", "extra_forbidden": "invalid", "greater_than": "out_of_range",
            "greater_than_equal": "out_of_range", "less_than": "out_of_range", "less_than_equal": "out_of_range",
            "string_too_long": "too_large", "too_long": "too_large"}


def parse_model(model_cls, data: dict):
    """스키마 검증. 타입·필수·추가 key 는 400, 선언한 크기 제한 초과는 413 (§6). 입력값은 오류에 담지 않는다."""
    try:
        return model_cls.model_validate_json(json.dumps(data, ensure_ascii=False))
    except ValidationError as exc:
        errs = exc.errors(include_input=False, include_url=False)
        details = [(".".join(str(p) for p in e["loc"]) or "body", _REASONS.get(e["type"], "invalid")) for e in errs]
        too_large = any(e["type"] in ("string_too_long", "too_long") for e in errs)
        raise AiError("INPUT_TOO_LARGE" if too_large else "VALIDATION_FAILED", details)
