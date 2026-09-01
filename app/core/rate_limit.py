from __future__ import annotations

import time
from collections import defaultdict, deque
from threading import Lock

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response

from app.core.config import settings
from app.observability.request_utils import get_client_ip


class SlidingWindowLimiter:
    """In-process sliding window. Adequate for a single uvicorn worker."""

    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def allow(self, key: str, limit: int, window_seconds: float = 60.0) -> bool:
        now = time.monotonic()
        cutoff = now - window_seconds
        with self._lock:
            bucket = self._hits[key]
            while bucket and bucket[0] < cutoff:
                bucket.popleft()
            if len(bucket) >= limit:
                return False
            bucket.append(now)
            return True


_limiter = SlidingWindowLimiter()


def _bucket_for_path(path: str) -> tuple[str, int]:
    if path.startswith("/api/v1/auth"):
        return "auth", settings.RATE_LIMIT_AUTH_PER_MINUTE
    if path.startswith("/api/v1/chat") or path.startswith("/api/v1/contracts"):
        return "chat", settings.RATE_LIMIT_CHAT_PER_MINUTE
    if "/upload" in path or path.startswith("/api/v1/documents"):
        return "upload", settings.RATE_LIMIT_UPLOAD_PER_MINUTE
    return "default", settings.RATE_LIMIT_DEFAULT_PER_MINUTE


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        if not settings.RATE_LIMIT_ENABLED:
            return await call_next(request)
        if request.method == "OPTIONS":
            return await call_next(request)
        if request.url.path in {"/health", "/ready", "/docs", "/openapi.json", "/redoc"}:
            return await call_next(request)

        ip = get_client_ip(
            x_forwarded_for=(
                request.headers.get("x-forwarded-for")
                if settings.TRUST_FORWARDED_FOR
                else None
            ),
            x_real_ip=(
                request.headers.get("x-real-ip")
                if settings.TRUST_FORWARDED_FOR
                else None
            ),
            client_host=request.client.host if request.client else None,
        ) or "unknown"
        bucket, limit = _bucket_for_path(request.url.path)
        key = f"{ip}:{bucket}"
        if not _limiter.allow(key, limit):
            return JSONResponse(
                {
                    "detail": "Rate limit exceeded. Please wait and try again.",
                },
                status_code=429,
                headers={"Retry-After": "60"},
            )
        return await call_next(request)
