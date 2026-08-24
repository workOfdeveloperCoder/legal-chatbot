from __future__ import annotations

from typing import Any

from qdrant_client import AsyncQdrantClient

from app.core.config import settings


def qdrant_connection_kwargs() -> dict[str, Any]:
    """
    Single source of Qdrant connection settings.

    Precedence:
      1. QDRANT_URL (full URL, may include https://)
      2. QDRANT_HOST + QDRANT_PORT + QDRANT_HTTPS

    QDRANT_API_KEY is always attached when set. Never construct a client
    that ignores a configured key.
    """
    kwargs: dict[str, Any] = {
        "timeout": settings.QDRANT_TIMEOUT,
        "prefer_grpc": False,
        "check_compatibility": False,
    }
    api_key = (settings.QDRANT_API_KEY or "").strip() or None
    if api_key:
        kwargs["api_key"] = api_key

    url = (settings.QDRANT_URL or "").strip()
    if url:
        kwargs["url"] = url.rstrip("/")
        return kwargs

    kwargs["host"] = settings.QDRANT_HOST
    kwargs["port"] = settings.QDRANT_PORT
    kwargs["https"] = bool(settings.QDRANT_HTTPS)
    return kwargs


def build_async_qdrant_client() -> AsyncQdrantClient:
    return AsyncQdrantClient(**qdrant_connection_kwargs())


def qdrant_endpoint_label() -> str:
    url = (settings.QDRANT_URL or "").strip()
    if url:
        return url
    scheme = "https" if settings.QDRANT_HTTPS else "http"
    return f"{scheme}://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}"
