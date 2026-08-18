from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.services.error_log_service import ErrorLogService

logger = logging.getLogger(__name__)
_error_log_service = ErrorLogService()


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(HTTPException)
    async def http_exception_handler(
        request: Request,
        exc: HTTPException,
    ) -> JSONResponse:
        await _error_log_service.record_exception(
            request,
            exc,
            status_code=exc.status_code,
        )

        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
            headers=getattr(exc, "headers", None) or None,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        await _error_log_service.record_exception(
            request,
            exc,
            status_code=422,
        )

        return JSONResponse(
            status_code=422,
            content={"detail": exc.errors()},
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        logger.exception(
            "Unhandled exception on %s %s",
            request.method,
            request.url.path,
        )

        await _error_log_service.record_exception(
            request,
            exc,
            status_code=500,
        )

        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error."},
        )
