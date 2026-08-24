from __future__ import annotations

import logging

from app.core.config import Settings

logger = logging.getLogger(__name__)


class ProductionConfigError(RuntimeError):
    """Raised when ENVIRONMENT=production is missing required settings."""


def validate_production_settings(settings: Settings) -> None:
    """
    Fail fast on an insecure production configuration.

    Local development (ENVIRONMENT != production) is unchanged.
    """
    if not settings.is_production:
        return

    errors: list[str] = []

    if not (settings.QDRANT_API_KEY or "").strip():
        errors.append(
            "QDRANT_API_KEY is required when ENVIRONMENT=production. "
            "Do not run an unsecured Qdrant against production traffic."
        )

    if settings.trusted_hosts == ["*"]:
        errors.append(
            "TRUSTED_HOSTS must be set to the public API hostname(s) "
            "when ENVIRONMENT=production (do not use *)."
        )

    if settings.is_online_llm and not settings.resolved_llm_api_key:
        errors.append(
            "LLM_API_KEY (or the provider alias) is required for hosted "
            f"chat provider {settings.LLM_PROVIDER!r} in production."
        )

    if errors:
        raise ProductionConfigError(" ".join(errors))

    if settings.LOG_STORE_CHAT_BODIES or settings.LOG_ENABLE_RESPONSE_BODY:
        logger.warning(
            "Production is storing chat or response bodies. "
            "Client legal data will be persisted in request logs."
        )


def qdrant_api_key_configured(settings: Settings) -> bool:
    return bool((settings.QDRANT_API_KEY or "").strip())
