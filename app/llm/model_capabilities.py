from __future__ import annotations

from dataclasses import dataclass

from app.core.config import settings
from app.llm.provider_config import is_online_provider, normalize_provider, resolve_chat_model


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    """Provider/model capability profile — not hard-coded in PromptBuilder/RAG."""

    model_name: str
    provider: str
    context_window: int
    max_output_tokens: int
    tokenizer_id: str
    input_price_per_1m: float | None = None
    output_price_per_1m: float | None = None
    returns_usage: bool = False
    supports_streaming: bool = True
    supports_temperature: bool = True
    uses_max_completion_tokens: bool = False
    timeout_seconds: int | None = None
    # False when counting uses a stand-in encoding (e.g. Gemini via cl100k).
    tokenizer_native: bool = True

    @property
    def has_pricing(self) -> bool:
        return (
            self.input_price_per_1m is not None
            and self.output_price_per_1m is not None
        )


# Known profiles. Values are defaults; Settings overrides apply on resolve().
_MODEL_PROFILES: dict[str, ModelCapabilities] = {
    "deepseek-r1:32b": ModelCapabilities(
        model_name="deepseek-r1:32b",
        provider="ollama",
        context_window=32768,
        max_output_tokens=4096,
        tokenizer_id="cl100k_base",
        input_price_per_1m=0.0,
        output_price_per_1m=0.0,
        returns_usage=True,
    ),
    "deepseek-r1": ModelCapabilities(
        model_name="deepseek-r1",
        provider="ollama",
        context_window=65536,
        max_output_tokens=8192,
        tokenizer_id="cl100k_base",
        input_price_per_1m=0.0,
        output_price_per_1m=0.0,
        returns_usage=True,
    ),
    "deepseek-chat": ModelCapabilities(
        model_name="deepseek-chat",
        provider="deepseek",
        context_window=65536,
        max_output_tokens=8192,
        tokenizer_id="cl100k_base",
        input_price_per_1m=0.27,
        output_price_per_1m=1.10,
        returns_usage=True,
    ),
    "deepseek-reasoner": ModelCapabilities(
        model_name="deepseek-reasoner",
        provider="deepseek",
        context_window=65536,
        max_output_tokens=8192,
        tokenizer_id="cl100k_base",
        input_price_per_1m=0.55,
        output_price_per_1m=2.19,
        returns_usage=True,
        supports_temperature=False,
    ),
    "gpt-4o": ModelCapabilities(
        model_name="gpt-4o",
        provider="openai",
        context_window=128000,
        max_output_tokens=16384,
        tokenizer_id="o200k_base",
        input_price_per_1m=2.50,
        output_price_per_1m=10.00,
        returns_usage=True,
    ),
    "gpt-4o-mini": ModelCapabilities(
        model_name="gpt-4o-mini",
        provider="openai",
        context_window=128000,
        max_output_tokens=16384,
        tokenizer_id="o200k_base",
        input_price_per_1m=0.15,
        output_price_per_1m=0.60,
        returns_usage=True,
    ),
    "gpt-4.1": ModelCapabilities(
        model_name="gpt-4.1",
        provider="openai",
        context_window=1047576,
        max_output_tokens=32768,
        tokenizer_id="o200k_base",
        input_price_per_1m=2.00,
        output_price_per_1m=8.00,
        returns_usage=True,
    ),
    "gpt-4.1-mini": ModelCapabilities(
        model_name="gpt-4.1-mini",
        provider="openai",
        context_window=1047576,
        max_output_tokens=32768,
        tokenizer_id="o200k_base",
        input_price_per_1m=0.40,
        output_price_per_1m=1.60,
        returns_usage=True,
    ),
    "gpt-4.1-nano": ModelCapabilities(
        model_name="gpt-4.1-nano",
        provider="openai",
        context_window=1047576,
        max_output_tokens=32768,
        tokenizer_id="o200k_base",
        input_price_per_1m=0.10,
        output_price_per_1m=0.40,
        returns_usage=True,
    ),
    "claude-sonnet-4-5": ModelCapabilities(
        model_name="claude-sonnet-4-5",
        provider="anthropic",
        context_window=200000,
        max_output_tokens=16384,
        tokenizer_id="cl100k_base",
        input_price_per_1m=3.00,
        output_price_per_1m=15.00,
        returns_usage=True,
    ),
    "claude-3-5-sonnet": ModelCapabilities(
        model_name="claude-3-5-sonnet",
        provider="anthropic",
        context_window=200000,
        max_output_tokens=8192,
        tokenizer_id="cl100k_base",
        input_price_per_1m=3.00,
        output_price_per_1m=15.00,
        returns_usage=True,
    ),
    "claude-haiku-4-5": ModelCapabilities(
        model_name="claude-haiku-4-5",
        provider="anthropic",
        context_window=200000,
        max_output_tokens=8192,
        tokenizer_id="cl100k_base",
        input_price_per_1m=1.00,
        output_price_per_1m=5.00,
        returns_usage=True,
    ),
    "gemini-2.0-flash": ModelCapabilities(
        model_name="gemini-2.0-flash",
        provider="gemini",
        context_window=1048576,
        max_output_tokens=8192,
        tokenizer_id="cl100k_base",
        input_price_per_1m=0.10,
        output_price_per_1m=0.40,
        returns_usage=True,
        tokenizer_native=False,
    ),
    "gemini-3.6-flash": ModelCapabilities(
        model_name="gemini-3.6-flash",
        provider="gemini",
        context_window=1048576,
        max_output_tokens=8192,
        tokenizer_id="cl100k_base",
        input_price_per_1m=0.10,
        output_price_per_1m=0.40,
        returns_usage=True,
        tokenizer_native=False,
    ),
    "o4-mini": ModelCapabilities(
        model_name="o4-mini",
        provider="openai",
        context_window=200000,
        max_output_tokens=100000,
        tokenizer_id="o200k_base",
        input_price_per_1m=1.10,
        output_price_per_1m=4.40,
        returns_usage=True,
        supports_temperature=False,
        uses_max_completion_tokens=True,
    ),
    "o3-mini": ModelCapabilities(
        model_name="o3-mini",
        provider="openai",
        context_window=200000,
        max_output_tokens=100000,
        tokenizer_id="o200k_base",
        input_price_per_1m=1.10,
        output_price_per_1m=4.40,
        returns_usage=True,
        supports_temperature=False,
        uses_max_completion_tokens=True,
    ),
}


def _unknown_online_defaults(provider: str) -> tuple[int, str]:
    """Sensible window/tokenizer when the hosted model is not in the catalog."""
    if provider in {"openai", "openrouter"}:
        return 128000, "o200k_base"
    if provider in {"deepseek", "groq", "openai_compatible"}:
        return 65536, "cl100k_base"
    return settings.TOKEN_DEFAULT_CONTEXT_WINDOW, (
        settings.TOKEN_TOKENIZER_ID or "cl100k_base"
    )


class ModelCapabilityRegistry:
    """Resolve capability profiles for local and online providers."""

    @classmethod
    def resolve(
        cls,
        *,
        provider: str | None = None,
        model_name: str | None = None,
        context_window: int | None = None,
        max_output_tokens: int | None = None,
        tokenizer_id: str | None = None,
    ) -> ModelCapabilities:
        provider_name = normalize_provider(provider or settings.LLM_PROVIDER)
        model = resolve_chat_model(
            provider_name,
            model_name or settings.CHAT_MODEL,
        )

        looked_up = cls._lookup(model)
        if looked_up is None:
            default_window, default_tokenizer = (
                _unknown_online_defaults(provider_name)
                if is_online_provider(provider_name)
                else (
                    settings.TOKEN_DEFAULT_CONTEXT_WINDOW,
                    settings.TOKEN_TOKENIZER_ID or "cl100k_base",
                )
            )
            profile = ModelCapabilities(
                model_name=model,
                provider=provider_name,
                context_window=default_window,
                max_output_tokens=max(
                    settings.TOKEN_RESERVED_OUTPUT_TOKENS,
                    2048,
                ),
                tokenizer_id=default_tokenizer,
                input_price_per_1m=None,
                output_price_per_1m=None,
                returns_usage=provider_name == "ollama"
                or is_online_provider(provider_name),
                uses_max_completion_tokens=model.lower().startswith("o"),
                supports_temperature=not model.lower().startswith("o"),
                tokenizer_native=provider_name != "gemini",
            )
            known = False
        else:
            profile = looked_up
            known = True

        # Known models keep their native windows; settings override unknowns
        # and always control the tokenizer id / operational reserved output
        # separately via TokenBudgetLimits.
        resolved_window = (
            context_window
            if context_window is not None
            else profile.context_window
        )
        resolved_max_out = (
            max_output_tokens
            if max_output_tokens is not None
            else profile.max_output_tokens
        )

        resolved_tokenizer = (
            tokenizer_id
            or (profile.tokenizer_id if known else None)
            or profile.tokenizer_id
            or settings.TOKEN_TOKENIZER_ID
            or "cl100k_base"
        )

        tokenizer_native = (
            False if provider_name == "gemini" else profile.tokenizer_native
        )

        return ModelCapabilities(
            model_name=model,
            provider=provider_name,
            context_window=resolved_window,
            max_output_tokens=resolved_max_out,
            tokenizer_id=resolved_tokenizer,
            input_price_per_1m=profile.input_price_per_1m,
            output_price_per_1m=profile.output_price_per_1m,
            returns_usage=profile.returns_usage,
            supports_streaming=profile.supports_streaming,
            supports_temperature=profile.supports_temperature,
            uses_max_completion_tokens=profile.uses_max_completion_tokens,
            timeout_seconds=profile.timeout_seconds,
            tokenizer_native=tokenizer_native,
        )

    @classmethod
    def _lookup(cls, model_name: str) -> ModelCapabilities | None:
        key = model_name.strip().lower()
        if key in _MODEL_PROFILES:
            return _MODEL_PROFILES[key]
        # prefix match: deepseek-r1:32b-q4_K_M → deepseek-r1:32b
        # gpt-4o-2024-08-06 → gpt-4o
        matches = [
            (known, profile)
            for known, profile in _MODEL_PROFILES.items()
            if key.startswith(known.lower())
        ]
        if not matches:
            return None
        matches.sort(key=lambda item: len(item[0]), reverse=True)
        return matches[0][1]


def estimate_cost_usd(
    *,
    capabilities: ModelCapabilities,
    input_tokens: int,
    output_tokens: int,
) -> float | None:
    """Return estimated USD cost, or None when pricing is unknown."""
    if (
        capabilities.input_price_per_1m is None
        or capabilities.output_price_per_1m is None
    ):
        return None
    return (
        (input_tokens / 1_000_000.0) * capabilities.input_price_per_1m
        + (output_tokens / 1_000_000.0) * capabilities.output_price_per_1m
    )
