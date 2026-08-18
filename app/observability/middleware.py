from __future__ import annotations

import json
import time
import uuid
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.core.config import settings
from app.models.request_log import RequestLog
from app.observability.context import (
    bind_request_context,
    get_request_context,
    reset_request_context,
)
from app.observability.device import parse_device_info
from app.observability.request_utils import (
    extract_user_id_from_authorization,
    get_client_ip,
)
from app.observability.sanitize import (
    parse_json_body,
    sanitize_headers,
    truncate_payload,
)
from app.services.request_log_service import RequestLogService


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app) -> None:
        super().__init__(app)
        self._request_log_service = RequestLogService()
        self._skip_paths = {
            path.strip()
            for path in settings.LOG_SKIP_PATHS.split(",")
            if path.strip()
        }

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        if request.url.path in self._skip_paths:
            return await call_next(request)

        request_id = uuid.uuid4()
        started_at = time.perf_counter()

        body_bytes = await request.body()
        request.state.body_bytes = body_bytes

        ip_address = get_client_ip(
            x_forwarded_for=request.headers.get("x-forwarded-for"),
            x_real_ip=request.headers.get("x-real-ip"),
            client_host=request.client.host if request.client else None,
        )
        user_agent = request.headers.get("user-agent")
        user_id = extract_user_id_from_authorization(
            request.headers.get("authorization"),
        )

        context_token = bind_request_context(
            request_id=request_id,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            method=request.method,
            path=request.url.path,
            query_params=dict(request.query_params),
            request_body=truncate_payload(
                parse_json_body(body_bytes),
                max_bytes=settings.LOG_BODY_MAX_BYTES,
            ),
        )

        response: Response | None = None
        response_body: dict | list | str | None = None
        status_code = 500

        try:
            response = await call_next(request)
            status_code = response.status_code

            if settings.LOG_ENABLE_RESPONSE_BODY:
                response_body, response = await self._capture_response_body(
                    response,
                )

            response.headers["X-Request-ID"] = str(request_id)
        finally:
            duration_ms = int((time.perf_counter() - started_at) * 1000)
            device = parse_device_info(user_agent)
            context = get_request_context()

            if settings.LOG_ENABLE_REQUEST_LOGGING:
                entry = RequestLog(
                    request_id=request_id,
                    user_id=context.user_id,
                    method=request.method,
                    path=request.url.path,
                    query_params=dict(request.query_params) or None,
                    path_params=dict(request.path_params) or None,
                    request_headers=sanitize_headers(
                        {key: value for key, value in request.headers.items()}
                    ),
                    request_body=truncate_payload(
                        parse_json_body(body_bytes),
                        max_bytes=settings.LOG_BODY_MAX_BYTES,
                    ),
                    response_status=status_code,
                    response_body=truncate_payload(
                        response_body,
                        max_bytes=settings.LOG_BODY_MAX_BYTES,
                    ),
                    ip_address=ip_address,
                    user_agent=user_agent,
                    device_type=device.device_type,
                    device_os=device.device_os,
                    device_browser=device.device_browser,
                    duration_ms=duration_ms,
                )
                await self._request_log_service.record(entry)

            reset_request_context(context_token)

        return response

    async def _capture_response_body(
        self,
        response: Response,
    ) -> tuple[dict | list | str | None, Response]:
        content_type = response.headers.get("content-type", "")

        if "application/json" not in content_type.lower():
            return None, response

        body_chunks: list[bytes] = []

        async for chunk in response.body_iterator:
            body_chunks.append(chunk)

        raw = b"".join(body_chunks)

        parsed = truncate_payload(
            parse_json_body(raw),
            max_bytes=settings.LOG_BODY_MAX_BYTES,
        )

        new_response = Response(
            content=raw,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type,
            background=response.background,
        )

        return parsed, new_response
