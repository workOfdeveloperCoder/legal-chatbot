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
        provider = (settings.EMBEDDING_PROVIDER or "ollama").lower().strip()
        if provider == "ollama":
            import httpx

            async with httpx.AsyncClient(timeout=3.0) as client:
                response = await client.get(
                    f"{settings.embedding_base_url}/api/tags",
                )
            checks["embeddings"] = response.status_code < 500
            if not checks["embeddings"]:
                errors.append(f"embeddings: HTTP {response.status_code}")
        elif provider in {"huggingface", "hf"}:
            checks["embeddings"] = bool(
                settings.embedding_api_key_for_provider(provider)
            )
            if not checks["embeddings"]:
                errors.append("embeddings: missing HUGGINGFACE_API_KEY or HF_TOKEN")
        elif provider in {"nomic", "openrouter", "fireworks", "openai", "openai_compatible"}:
            checks["embeddings"] = bool(
                settings.embedding_api_key_for_provider(provider)
            )
            if not checks["embeddings"]:
                errors.append(f"embeddings: missing API key for {provider}")
        else:
            checks["embeddings"] = True
    except Exception as exc:  # noqa: BLE001
        errors.append(f"embeddings: {type(exc).__name__}")

    # Chat LLM is not probed here — readiness must stay cheap.

    embedding_provider = (settings.EMBEDDING_PROVIDER or "ollama").lower().strip()
    online_embed_providers = {
        "openrouter",
        "fireworks",
        "nomic",
        "openai",
        "openai_compatible",
    }
    ok = all(bool(value) for value in checks.values())
    llm_online = settings.is_online_llm
    embed_online = embedding_provider in online_embed_providers
    embed_key_ok = bool(settings.embedding_api_key_for_provider(embedding_provider))
    if embedding_provider in {"huggingface", "hf"}:
        embed_online = bool(settings.embedding_api_key_for_provider(embedding_provider))

    payload = {
        "status": "ready" if ok else "degraded",
        "checks": checks,
        "stack": {
            "chat": {
                "provider": settings.LLM_PROVIDER,
                "model": settings.chat_model_for_provider(),
                "online": llm_online,
                "api_key_configured": (
                    bool(settings.resolved_llm_api_key)
                    if llm_online
                    else True
                ),
            },
            "search": {
                "provider": embedding_provider,
                "model": settings.EMBEDDING_MODEL,
                "online": embed_online,
                "api_key_configured": embed_key_ok,
                "vector_size": settings.VECTOR_SIZE,
            },
            "qdrant": {
                "endpoint": qdrant_endpoint_label(),
                "collection": settings.LEGAL_QDRANT_COLLECTION,
                "ok": bool(checks["qdrant"]),
            },
        },
        "qdrant_endpoint": qdrant_endpoint_label(),
        "qdrant_api_key": bool((settings.QDRANT_API_KEY or "").strip()),
        "llm_provider": settings.LLM_PROVIDER,
        "embedding_provider": embedding_provider,
        "llm_api_key_configured": (
            bool(settings.resolved_llm_api_key)
            if settings.is_online_llm
            else True
        ),
        "embedding_api_key_configured": embed_key_ok,
        "errors": errors,
    }
    return JSONResponse(payload, status_code=200 if ok else 503)
