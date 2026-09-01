from __future__ import annotations

from urllib.parse import urlparse

from app.llm.provider_config import _LOCAL_HOST_MARKERS


def _remote_ollama_url() -> bool:
    from app.core.config import settings

    provider = (settings.EMBEDDING_PROVIDER or "ollama").lower().strip()
    if provider != "ollama":
        return False
    host = (urlparse(settings.embedding_base_url).hostname or "").lower()
    return host not in _LOCAL_HOST_MARKERS


def online_only_embedding_error(provider: str) -> str:
    return (
        f"EMBEDDING_PROVIDER={provider!r} runs or downloads models on this host, "
        "but EMBEDDING_ONLINE_ONLY=true. Options that keep 768-d Qdrant working:\n"
        "  1) Remote Ollama sidecar (free): EMBEDDING_PROVIDER=ollama, "
        "EMBEDDING_URL=http://<embed-host>:11434 — models live on another VM, not here.\n"
        "  2) Fireworks API: EMBEDDING_PROVIDER=fireworks + FIREWORKS_API_KEY "
        "(nomic-ai/nomic-embed-text-v1.5).\n"
        "  3) Nomic Atlas API: EMBEDDING_PROVIDER=nomic + NOMIC_API_KEY (work email).\n"
        "  4) OpenRouter embed + re-index Qdrant to 1536-d (breaks current index)."
    )
