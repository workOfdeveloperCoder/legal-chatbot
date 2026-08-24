from __future__ import annotations

from typing import Mapping

from app.core.config import settings
from app.llm.base import BaseLLM
from app.llm.execution_profile import ExecutionProfile
from app.llm.model_capabilities import (
    ModelCapabilities,
    ModelCapabilityRegistry,
    estimate_cost_usd,
)
from app.llm.provider_config import is_online_provider
from app.llm.token_counter import build_token_counter
from app.schemas.llm import (
    LLMStatusResponse,
    TokenCountResponse,
    TokenUsageBreakdown,
    TokenUsageResponse,
)


def _as_int(value: object, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def apply_provider_usage(
    budget: Mapping[str, object] | None,
    *,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    total_tokens: int | None,
    capabilities: ModelCapabilities | None = None,
) -> dict[str, object] | None:
    """Merge provider-reported usage into the budget estimate."""
    if budget is None:
        return None
    meta = dict(budget)
    if prompt_tokens is not None:
        meta["input_tokens"] = prompt_tokens
        meta["usage_source"] = "provider"
        if completion_tokens is not None:
            meta["output_tokens"] = completion_tokens
        if total_tokens is not None:
            meta["total_tokens"] = total_tokens
        elif completion_tokens is not None:
            meta["total_tokens"] = prompt_tokens + completion_tokens

    input_tokens = _as_int(meta.get("input_tokens"))
    available = _as_int(meta.get("available_input_tokens"))
    window = _as_int(meta.get("context_window"))
    total = _as_int(meta.get("total_tokens"), input_tokens)
    meta["remaining_input_tokens"] = max(0, available - input_tokens)
    meta["usage_percent"] = (
        round(100.0 * total / window, 2) if window > 0 else 0.0
    )

    caps = capabilities
    if caps is None:
        caps = ModelCapabilityRegistry.resolve(
            provider=str(meta.get("provider") or "") or None,
            model_name=str(meta.get("model") or "") or None,
        )
    if settings.TOKEN_COST_TRACKING_ENABLED:
        cost = estimate_cost_usd(
            capabilities=caps,
            input_tokens=input_tokens,
            output_tokens=_as_int(meta.get("output_tokens")),
        )
        if cost is not None:
            meta["estimated_cost"] = cost
    return meta


def token_usage_from_budget(
    budget: Mapping[str, object] | None,
    *,
    capabilities: ModelCapabilities | None = None,
) -> TokenUsageResponse | None:
    if not budget:
        return None
    caps = capabilities or ModelCapabilityRegistry.resolve(
        provider=str(budget.get("provider") or "") or None,
        model_name=str(budget.get("model") or "") or None,
    )
    input_tokens = _as_int(budget.get("input_tokens"))
    available = _as_int(
        budget.get("available_input_tokens"),
        max(
            0,
            caps.context_window
            - caps.max_output_tokens
            - settings.TOKEN_SAFETY_MARGIN,
        ),
    )
    remaining = _as_int(
        budget.get("remaining_input_tokens"),
        max(0, available - input_tokens),
    )
    window = _as_int(budget.get("context_window"), caps.context_window)
    total = _as_int(budget.get("total_tokens"))
    percent = _as_float(budget.get("usage_percent"))
    if percent is None:
        percent = round(100.0 * total / window, 2) if window else 0.0
    warning_at = settings.TOKEN_USAGE_WARNING_PERCENT
    return TokenUsageResponse(
        model=str(budget.get("model") or caps.model_name),
        provider=str(budget.get("provider") or caps.provider),
        tokenizer_id=(
            str(budget["tokenizer_id"])
            if budget.get("tokenizer_id") is not None
            else caps.tokenizer_id
        ),
        tokenizer_exact=bool(budget.get("tokenizer_exact")),
        context_window=window,
        max_output_tokens=caps.max_output_tokens,
        input_tokens=input_tokens,
        output_tokens=_as_int(budget.get("output_tokens")),
        total_tokens=total,
        available_input_tokens=available,
        remaining_input_tokens=remaining,
        usage_percent=percent,
        warning=percent >= warning_at,
        over_limit=input_tokens > available if available else False,
        reserved_output_tokens=_as_int(budget.get("reserved_output_tokens")),
        safety_margin_tokens=_as_int(budget.get("safety_margin_tokens")),
        estimated_cost_usd=_as_float(budget.get("estimated_cost")),
        budget_trimmed=bool(budget.get("budget_trimmed")),
        trimming_reason=(
            str(budget["trimming_reason"])
            if budget.get("trimming_reason") is not None
            else None
        ),
        usage_source=str(budget.get("usage_source") or "estimate"),
        breakdown=TokenUsageBreakdown(
            system_tokens=_as_int(budget.get("system_tokens")),
            query_tokens=_as_int(budget.get("query_tokens")),
            history_tokens=_as_int(budget.get("history_tokens")),
            legal_evidence_tokens=_as_int(budget.get("legal_evidence_tokens")),
            conversation_evidence_tokens=_as_int(
                budget.get("conversation_evidence_tokens")
            ),
            matter_evidence_tokens=_as_int(budget.get("matter_evidence_tokens")),
            scaffolding_tokens=_as_int(budget.get("scaffolding_tokens")),
        ),
    )


class TokenUsageService:
    """
    Live token limits and counting for the current chat model.

    Embeddings stay local; this service only reflects the chat LLM.
    """

    def __init__(self, llm: BaseLLM | None = None) -> None:
        self._llm = llm
        caps = None
        if llm is not None:
            getter = getattr(llm, "capabilities", None)
            try:
                raw = getter() if callable(getter) else getter
            except Exception:  # noqa: BLE001
                raw = None
            if isinstance(raw, ModelCapabilities):
                caps = raw
        self.capabilities = caps or ModelCapabilityRegistry.resolve()
        self._profile = ExecutionProfile.for_capabilities(self.capabilities)
        self.counter = build_token_counter(
            self.capabilities.tokenizer_id,
            enabled=settings.TOKEN_COUNTING_ENABLED,
        )
        limits = self._profile.token_limits(self.capabilities)
        self.reserved_output_tokens = max(256, limits.reserved_output_tokens)
        self.safety_margin_tokens = limits.safety_margin
        self.available_input_tokens = max(
            512,
            self.capabilities.context_window
            - self.reserved_output_tokens
            - self.safety_margin_tokens,
        )

    def status(self) -> LLMStatusResponse:
        caps = self.capabilities
        return LLMStatusResponse(
            provider=caps.provider,
            model=caps.model_name,
            online=is_online_provider(caps.provider),
            context_window=caps.context_window,
            max_output_tokens=caps.max_output_tokens,
            tokenizer_id=self.counter.tokenizer_id,
            tokenizer_exact=bool(
                self.counter.is_exact and caps.tokenizer_native
            ),
            available_input_tokens=self.available_input_tokens,
            reserved_output_tokens=self.reserved_output_tokens,
            safety_margin_tokens=self.safety_margin_tokens,
            warning_percent=settings.TOKEN_USAGE_WARNING_PERCENT,
            input_price_per_1m=caps.input_price_per_1m,
            output_price_per_1m=caps.output_price_per_1m,
        )

    def count(
        self,
        text: str,
        *,
        extra_texts: list[str] | None = None,
        include_system_prompt: bool = True,
    ) -> TokenCountResponse:
        tokens = self.counter.count(text)
        for extra in extra_texts or []:
            tokens += self.counter.count(extra)
        if include_system_prompt:
            tokens += self.counter.count(self._profile.system_prompt)

        remaining = max(0, self.available_input_tokens - tokens)
        window = self.capabilities.context_window
        percent = round(100.0 * tokens / window, 2) if window else 0.0
        warning_at = settings.TOKEN_USAGE_WARNING_PERCENT
        return TokenCountResponse(
            tokens=tokens,
            context_window=window,
            available_input_tokens=self.available_input_tokens,
            remaining_tokens=remaining,
            usage_percent=percent,
            warning=percent >= warning_at,
            over_limit=tokens > self.available_input_tokens,
            model=self.capabilities.model_name,
            provider=self.capabilities.provider,
            tokenizer_id=self.counter.tokenizer_id,
            tokenizer_exact=bool(
                self.counter.is_exact and self.capabilities.tokenizer_native
            ),
        )
