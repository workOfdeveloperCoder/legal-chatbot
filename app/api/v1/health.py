from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.core.config import settings
from app.database.session import engine
from app.vector.client import qdrant_endpoint_label
from app.vector.qdrant import qdrant_service

router = APIRouter(tags=["Health"])


@router.get("/health", include_in_schema=False)
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready", include_in_schema=False)
async def ready() -> JSONResponse:
    checks: dict[str, object] = {
        "postgres": False,
        "qdrant": False,
        "embeddings": False,
    }
    errors: list[str] = []

    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        checks["postgres"] = True
    except Exception as exc:  # noqa: BLE001
        errors.append(f"postgres: {type(exc).__name__}")

    try:
        checks["qdrant"] = await qdrant_service.health()
        if not checks["qdrant"]:
            errors.append(f"qdrant: unreachable at {qdrant_endpoint_label()}")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"qdrant: {type(exc).__name__}")

    try:
        import httpx

        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(
                f"{settings.embedding_base_url}/api/tags",
            )
        checks["embeddings"] = response.status_code < 500
        if not checks["embeddings"]:
            errors.append(f"embeddings: HTTP {response.status_code}")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"embeddings: {type(exc).__name__}")

    # Chat LLM is not probed here — readiness must stay cheap.

    ok = all(bool(value) for value in checks.values())
    payload = {
        "status": "ready" if ok else "degraded",
        "checks": checks,
        "qdrant_endpoint": qdrant_endpoint_label(),
        "qdrant_api_key": bool((settings.QDRANT_API_KEY or "").strip()),
        "llm_provider": settings.LLM_PROVIDER,
        "llm_api_key_configured": (
            bool(settings.resolved_llm_api_key)
            if settings.is_online_llm
            else True
        ),
        "errors": errors,
    }
    return JSONResponse(payload, status_code=200 if ok else 503)
