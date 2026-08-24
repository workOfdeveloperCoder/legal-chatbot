from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Fallback before adapter modules are imported (tests / early config).
ONLINE_PROVIDERS = frozenset(
    {
        "openai",
        "openai_compatible",
        "openai-compatible",
        "deepseek",
        "groq",
        "openrouter",
        "anthropic",
        "claude",
        "gemini",
        "mistral",
        "together",
        "fireworks",
    }
)

# Canonical env names for ANY hosted chat LLM. Vendor-specific key names
# (OPENAI_API_KEY, GOOGLE_API_KEY, …) are optional aliases only.
CANONICAL_PROVIDER_ENV = "LLM_PROVIDER"
CANONICAL_MODEL_ENV = "CHAT_MODEL"
CANONICAL_API_KEY_ENV = "LLM_API_KEY"
CANONICAL_URL_ENV = "LLM_URL"

DEFAULT_CHAT_MODELS: dict[str, str] = {
    "ollama": "deepseek-r1:32b",
    "openai": "gpt-4o-mini",
    "openai_compatible": "gpt-4o-mini",
    "gemini": "gemini-3.6-flash",
    "anthropic": "claude-sonnet-4-5",
    "deepseek": "deepseek-chat",
    "groq": "llama-3.3-70b-versatile",
    "mistral": "mistral-small-latest",
    "openrouter": "openai/gpt-4o-mini",
    "together": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "fireworks": "accounts/fireworks/models/llama-v3p3-70b-instruct",
}

# Optional aliases if LLM_API_KEY is empty. Never required.
PROVIDER_API_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "openai": ("OPENAI_API_KEY",),
    "openai_compatible": ("OPENAI_API_KEY",),
    "gemini": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "deepseek": ("DEEPSEEK_API_KEY",),
    "groq": ("GROQ_API_KEY",),
    "mistral": ("MISTRAL_API_KEY",),
    "openrouter": ("OPENROUTER_API_KEY",),
    "together": ("TOGETHER_API_KEY",),
    "fireworks": ("FIREWORKS_API_KEY",),
}

_PROVIDER_DEFAULT_URLS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "groq": "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "anthropic": "https://api.anthropic.com",
    "claude": "https://api.anthropic.com",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
    "mistral": "https://api.mistral.ai/v1",
    "together": "https://api.together.xyz/v1",
    "fireworks": "https://api.fireworks.ai/inference/v1",
}

_LOCAL_TIMEOUT_SENTINEL = 1800
_ONLINE_DEFAULT_TIMEOUT = 120

_LOCAL_HOST_MARKERS = ("localhost", "127.0.0.1", "0.0.0.0")


def normalize_provider(provider: str | None) -> str:
    name = (provider or "ollama").lower().strip()
    if name == "openai-compatible":
        return "openai_compatible"
    if name == "claude":
        return "anthropic"
    if name == "google":
        return "gemini"
    return name


def is_online_provider(provider: str | None) -> bool:
    name = normalize_provider(provider)
    try:
        from app.llm.registry import LLMAdapterRegistry

        registered = LLMAdapterRegistry.is_online(name)
        if registered is not None:
            return registered
    except Exception:  # noqa: BLE001 — registry may not be loaded yet
        pass
    return name in ONLINE_PROVIDERS


def is_local_url(url: str | None) -> bool:
    if not url:
        return True
    lowered = url.lower()
    return any(marker in lowered for marker in _LOCAL_HOST_MARKERS)


def _default_url(name: str) -> str | None:
    try:
        from app.llm.registry import LLMAdapterRegistry

        registered = LLMAdapterRegistry.default_url(name)
        if registered:
            return registered
    except Exception:  # noqa: BLE001
        pass
    return _PROVIDER_DEFAULT_URLS.get(name)


def resolve_llm_base_url(provider: str | None, configured_url: str | None) -> str:
    """
    Use the hosted API URL when the provider is online but LLM_URL is still
    the local Ollama default.
    """
    name = normalize_provider(provider)
    default_online = _default_url(name)
    url = (configured_url or "").rstrip("/")
    if default_online and is_local_url(url):
        return default_online
    if url:
        return url
    return default_online or "http://localhost:11434"


def default_chat_model(provider: str | None) -> str:
    name = normalize_provider(provider)
    if name in DEFAULT_CHAT_MODELS:
        return DEFAULT_CHAT_MODELS[name]
    if is_online_provider(name):
        return DEFAULT_CHAT_MODELS["openai"]
    return DEFAULT_CHAT_MODELS["ollama"]


def provider_api_key_aliases(provider: str | None) -> tuple[str, ...]:
    name = normalize_provider(provider)
    return PROVIDER_API_KEY_ALIASES.get(name, ())


def nonempty_secret(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def resolve_chat_model(provider: str | None, model: str | None) -> str:
    """
    Map leftover local Ollama tags onto the hosted model for that provider.
    """
    name = normalize_provider(provider)
    chosen = (model or "").strip()
    if not chosen:
        return default_chat_model(name)

    looks_local = ":" in chosen
    if not looks_local:
        return chosen

    if is_online_provider(name):
        mapped = default_chat_model(name)
        if name == "deepseek" and "r1" in chosen.lower():
            mapped = "deepseek-reasoner"
        logger.warning(
            "CHAT_MODEL %s looks like a local Ollama tag; using %s for %s",
            chosen,
            mapped,
            name,
        )
        return mapped
    return chosen


def effective_llm_timeout(
    provider: str | None,
    configured_timeout: int,
) -> int:
    if (
        is_online_provider(provider)
        and configured_timeout == _LOCAL_TIMEOUT_SENTINEL
    ):
        return _ONLINE_DEFAULT_TIMEOUT
    return configured_timeout


def adapter_provider_name(provider: str | None) -> str:
    return normalize_provider(provider)
