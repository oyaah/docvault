"""API key authentication middleware."""

import os
import secrets
import logging
from pathlib import Path

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from docvault.config import settings

logger = logging.getLogger(__name__)

# Public endpoints that don't require auth
PUBLIC_PATHS = {"/api/health", "/api/health/live", "/api/health/ready", "/api/metrics", "/docs", "/openapi.json", "/redoc"}


def _load_api_keys() -> set[str]:
    """Load API keys from environment or file."""
    keys = set()

    # From environment variable (comma-separated)
    env_keys = os.environ.get("DOCVAULT_API_KEYS", "")
    if env_keys:
        keys.update(k.strip() for k in env_keys.split(",") if k.strip())

    # From file
    keys_file = settings.data_dir / "api_keys.txt"
    if keys_file.exists():
        for line in keys_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                keys.add(line)

    return keys


def generate_api_key() -> str:
    """Generate a new API key."""
    return f"dv-{secrets.token_urlsafe(32)}"


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Validates API key in Authorization header or X-API-Key header.

    If no API keys are configured (env or file), auth is disabled (open access).
    This allows easy local development while enforcing auth in production.
    """

    def __init__(self, app):
        super().__init__(app)
        self._keys = _load_api_keys()
        if self._keys:
            logger.info(f"API auth enabled: {len(self._keys)} key(s) loaded")
        elif settings.require_auth:
            # Fail closed: no keys + auth required -> reject all protected traffic.
            logger.error(
                "API auth REQUIRED but no keys configured — all protected endpoints "
                "will return 503. Set DOCVAULT_API_KEYS, create data/api_keys.txt, "
                "or set DOCVAULT_REQUIRE_AUTH=false for local dev."
            )
        else:
            logger.warning("API auth disabled (DOCVAULT_REQUIRE_AUTH=false) — open access.")

    async def dispatch(self, request: Request, call_next):
        # Public endpoints never require auth.
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        if not self._keys:
            if settings.require_auth:
                return JSONResponse(
                    status_code=503,
                    content={"detail": "Server misconfigured: no API keys configured and auth is required."},
                )
            return await call_next(request)  # explicit dev-mode open access

        auth_header = request.headers.get("Authorization", "")
        api_key_header = request.headers.get("X-API-Key", "")
        key = auth_header[7:] if auth_header.startswith("Bearer ") else api_key_header

        if not key or key not in self._keys:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key. Use Authorization: Bearer <key> or X-API-Key."},
            )

        return await call_next(request)
