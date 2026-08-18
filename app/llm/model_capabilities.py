from __future__ import annotations

from dataclasses import dataclass

from app.core.config import settings


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
    timeout_seconds: int | None = None

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
}


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
        provider_name = (provider or settings.LLM_PROVIDER or "ollama").lower()
        model = (model_name or settings.CHAT_MODEL or "deepseek-r1:32b").strip()

        looked_up = cls._lookup(model)
        if looked_up is None:
            profile = ModelCapabilities(
                model_name=model,
                provider=provider_name,
                context_window=settings.TOKEN_DEFAULT_CONTEXT_WINDOW,
                max_output_tokens=max(
                    settings.TOKEN_RESERVED_OUTPUT_TOKENS,
                    2048,
                ),
                tokenizer_id=settings.TOKEN_TOKENIZER_ID or "cl100k_base",
                input_price_per_1m=None,
                output_price_per_1m=None,
                returns_usage=provider_name
                in {"ollama", "openai", "openai_compatible"},
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
            else (
                profile.context_window
                if known
                else settings.TOKEN_DEFAULT_CONTEXT_WINDOW
            )
        )
        resolved_max_out = (
            max_output_tokens
            if max_output_tokens is not None
            else profile.max_output_tokens
        )

        resolved_tokenizer = (
            tokenizer_id
            or (profile.tokenizer_id if known else None)
            or settings.TOKEN_TOKENIZER_ID
            or "cl100k_base"
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
        )

    @classmethod
    def _lookup(cls, model_name: str) -> ModelCapabilities | None:
        key = model_name.strip().lower()
        if key in _MODEL_PROFILES:
            return _MODEL_PROFILES[key]
        # prefix match: deepseek-r1:32b-q4_K_M → deepseek-r1:32b
        for known, profile in _MODEL_PROFILES.items():
            if key.startswith(known.lower()):
                return profile
        return None


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
