from __future__ import annotations

import pytest
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.core.request_size import RequestSizeLimitMiddleware


def _scope(method: str, path: str, headers: list[tuple[bytes, bytes]]) -> dict:
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": [(b"host", b"testserver"), *headers],
        "client": ("127.0.0.1", 50000),
        "server": ("testserver", 80),
    }


async def _dispatch(
    path: str,
    *,
    method: str = "POST",
    headers: list[tuple[bytes, bytes]] | None = None,
) -> Response:
    middleware = RequestSizeLimitMiddleware(app=None)

    async def receive():
        return {"type": "http.request", "body": b"tiny", "more_body": False}

    async def call_next(_request):
        return JSONResponse({"ok": True})

    request = Request(_scope(method, path, headers or []), receive)
    return await middleware.dispatch(request, call_next)


@pytest.mark.asyncio
async def test_oversized_content_length_returns_413(monkeypatch):
    from app.core import config

    monkeypatch.setattr(config.settings, "MAX_REQUEST_BYTES", 100)
    response = await _dispatch(
        "/echo",
        headers=[(b"content-length", b"500")],
    )
    assert response.status_code == 413
    body = response.body.decode()
    assert "too large" in body.lower()


@pytest.mark.asyncio
async def test_health_is_not_size_limited(monkeypatch):
    from app.core import config

    monkeypatch.setattr(config.settings, "MAX_REQUEST_BYTES", 1)
    response = await _dispatch("/health", method="GET")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_chat_stream_within_limit_is_allowed(monkeypatch):
    from app.core import config

    monkeypatch.setattr(config.settings, "MAX_REQUEST_BYTES", 1024)
    response = await _dispatch(
        "/api/v1/chat/stream",
        headers=[(b"content-length", b"64")],
    )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_invalid_content_length_returns_400():
    response = await _dispatch(
        "/echo",
        headers=[(b"content-length", b"nope")],
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_upload_under_max_request_bytes_is_allowed(monkeypatch):
    from app.core import config

    monkeypatch.setattr(config.settings, "MAX_REQUEST_BYTES", 32 * 1024 * 1024)
    response = await _dispatch(
        "/api/v1/documents/upload",
        headers=[(b"content-length", str(25 * 1024 * 1024).encode())],
    )
    assert response.status_code == 200
