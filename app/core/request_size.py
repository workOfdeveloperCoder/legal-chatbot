from __future__ import annotations

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response

from app.core.config import settings


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject oversized request bodies using Content-Length (Nginx also caps)."""

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.method in {"GET", "HEAD", "OPTIONS"}:
            return await call_next(request)
        if request.url.path in {"/health", "/ready"}:
            return await call_next(request)

        raw = request.headers.get("content-length")
        if raw is None:
            return await call_next(request)
        try:
            length = int(raw)
        except ValueError:
            return JSONResponse(
                {"detail": "Invalid Content-Length header."},
                status_code=400,
            )
        if length > settings.MAX_REQUEST_BYTES:
            return JSONResponse(
                {
                    "detail": (
                        "Request body is too large. "
                        f"Maximum size is {settings.MAX_REQUEST_BYTES} bytes."
                    ),
                },
                status_code=413,
            )
        return await call_next(request)
