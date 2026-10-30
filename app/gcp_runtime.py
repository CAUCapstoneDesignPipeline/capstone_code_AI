"""Deployment-only entrypoint: private drain gate and disabled response caches.

No product API/profile/algorithm changes. Public model weights alone are cached.
"""
import logging
import os
from dataclasses import replace
from pathlib import Path

from fastapi import Request

from app.api.v1.deps import UUID_RE
from app.core.config import Settings
from app.core.errors import AiError, error_response, new_request_id
from app.main import create_app


class DiagnosticFilter(logging.Filter):
    def filter(self, record):
        record.msg = "AI runtime diagnostic; inspect authenticated readiness and operation journal"
        record.args = ()
        record.exc_info = None
        record.exc_text = None
        record.stack_info = None
        return True


def production_app():
    # SDK/config errors can embed response bodies. Runtime logs contain no raw diagnostics.
    handler = logging.StreamHandler()
    handler.addFilter(DiagnosticFilter())
    logging.basicConfig(handlers=[handler], level=logging.WARNING, force=True)
    settings = replace(Settings.from_env(), concept_cache=None, relation_cache=None,
                       explain_cache=None, verify_cache=None, dev_ui=False)
    app = create_app(settings)
    drain = Path(os.environ.get("AI_DRAIN_FILE", "/run/ai/control/drain"))

    @app.middleware("http")
    async def drain_gate(request: Request, call_next):
        if drain.exists() and (request.method == "POST" or request.url.path.endswith("/health/ready")):
            # This outer gate runs before create_app's request-ID middleware.
            request_id = request.headers.get("x-request-id")
            if not request_id or not UUID_RE.fullmatch(request_id):
                request_id = new_request_id()
            return error_response(AiError("SERVICE_UNAVAILABLE", retry_after=30), request_id)
        return await call_next(request)

    return app


app = production_app()
