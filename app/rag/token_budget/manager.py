from __future__ import annotations

import logging

from app.core.config import settings
from app.llm.model_capabilities import (
    ModelCapabilities,
    ModelCapabilityRegistry,
    estimate_cost_usd,
)
from app.llm.token_counter import TokenCounter, build_token_counter
from app.rag.models import Memory, Message, RetrievedChunk, SourceType
from app.rag.prompts import LEGAL_SYSTEM_PROMPT
from app.rag.token_budget.models import (
    PackedContext,
    TokenBudgetLimits,
    TokenUsageMetadata,
)
from app.rag.token_budget.packing import (
    compress_conversation_history,
    group_chunks_by_document,
    pack_evidence_by_budget,
)

logger = logging.getLogger(__name__)

_MIN_QUESTION_TOKENS = 32


class TokenBudgetExceeded(ValueError):
    """Raised when the packed prompt cannot fit the model context window."""


class TokenBudgetManager:
    """
    Fits system/query/history/evidence into a model context window.

    Infrastructure only — does not perform legal reasoning.
    PromptBuilder formats the packed result; LLM generates.
    """

    def __init__(
        self,
        *,
        capabilities: ModelCapabilities | None = None,
        counter: TokenCounter | None = None,
        limits: TokenBudgetLimits | None = None,
    ) -> None:
        self.capabilities = capabilities or ModelCapabilityRegistry.resolve()
        self.limits = limits or TokenBudgetLimits(
            context_window=self.capabilities.context_window,
            reserved_output_tokens=min(
                self.capabilities.max_output_tokens,
                settings.TOKEN_RESERVED_OUTPUT_TOKENS,
            ),
            safety_margin=settings.TOKEN_SAFETY_MARGIN,
            max_conversation_tokens=settings.TOKEN_MAX_CONVERSATION_TOKENS,
            max_legal_evidence_tokens=settings.TOKEN_MAX_LEGAL_EVIDENCE_TOKENS,
            max_matter_evidence_tokens=settings.TOKEN_MAX_MATTER_EVIDENCE_TOKENS,
            max_conversation_document_tokens=settings.TOKEN_MAX_CONVERSATION_DOCUMENT_TOKENS,
            scaffolding_overhead_tokens=settings.TOKEN_SCAFFOLDING_OVERHEAD,
            token_counting_enabled=settings.TOKEN_COUNTING_ENABLED,
            cost_tracking_enabled=settings.TOKEN_COST_TRACKING_ENABLED,
        )
        if counter is not None:
            self.counter = counter
        else:
            raw = build_token_counter(
                self.capabilities.tokenizer_id,
                enabled=self.limits.token_counting_enabled,
            )
            if raw.is_exact:
                self.counter = raw
            else:
                from app.llm.token_counter import ScaledTokenCounter

                self.counter = ScaledTokenCounter(
                    raw,
                    settings.TOKEN_FALLBACK_SAFETY_FACTOR,
                )
                logger.warning(
                    "Token packing using conservative estimator "
                    "tokenizer=%s factor=%s",
                    raw.tokenizer_id,
                    settings.TOKEN_FALLBACK_SAFETY_FACTOR,
                )

    def prepare(
        self,
        *,
        question: str,
        history: list[Message],
        chunks: list[RetrievedChunk],
        memories: list[Memory] | None = None,
        system_prompt: str | None = None,
        conversation_summary: str | None = None,
        reserved_output_tokens: int | None = None,
        document_qa_mode: bool = False,
        duplicate_system_in_user: bool = True,
    ) -> PackedContext:
        """
        Allocate available input tokens and pack context for the LLM call.

        available_input_tokens =
            context_window - reserved_output_tokens - safety_margin
        """
        caps = self.capabilities
        limits = self.limits
        reserved = reserved_output_tokens or limits.reserved_output_tokens
        reserved = max(256, min(reserved, caps.max_output_tokens, caps.context_window // 2))

        available_input = (
            caps.context_window - reserved - limits.safety_margin
        )
        if available_input < 512:
            # Protect against pathological config: shrink reservation.
            reserved = max(256, caps.context_window // 4)
            available_input = (
                caps.context_window - reserved - limits.safety_margin
            )

        system_text = system_prompt if system_prompt is not None else LEGAL_SYSTEM_PROMPT
        system_tokens = self.counter.count(system_text)
        query_tokens = self.counter.count(question)
        scaffolding = limits.scaffolding_overhead_tokens

        # PromptBuilder may also embed the system prompt in the user message.
        # Skip that reservation when the execution profile already sends it
        # only as the system role (hosted models).
        prompt_system_dup = 0
        if duplicate_system_in_user:
            if system_text.strip() == LEGAL_SYSTEM_PROMPT.strip():
                prompt_system_dup = self.counter.count(LEGAL_SYSTEM_PROMPT)
            elif system_prompt is None:
                prompt_system_dup = self.counter.count(LEGAL_SYSTEM_PROMPT)

        fixed = system_tokens + prompt_system_dup + query_tokens + scaffolding
        remaining = available_input - fixed
        trim_reasons: list[str] = []

        if remaining < 0:
            trim_reasons.append("fixed_prompt_exceeds_available_input")
            # Shed non-essential fixed costs first (scaffolding, then dup system).
            overflow = -remaining
            if scaffolding and overflow > 0:
                cut = min(scaffolding, overflow)
                scaffolding -= cut
                overflow -= cut
            if prompt_system_dup and overflow > 0:
                cut = min(prompt_system_dup, overflow)
                prompt_system_dup -= cut
                overflow -= cut
            fixed = system_tokens + prompt_system_dup + query_tokens + scaffolding
            remaining = max(0, available_input - fixed)
            if overflow > 0:
                fitted, query_tokens = self._truncate_to_token_budget(
                    question,
                    max_tokens=max(
                        _MIN_QUESTION_TOKENS,
                        available_input - system_tokens - prompt_system_dup - scaffolding,
                    ),
                )
                if fitted != question:
                    question = fitted
                    trim_reasons.append("question_truncated_to_fit_context")
                fixed = system_tokens + prompt_system_dup + query_tokens + scaffolding
                remaining = available_input - fixed
                overflow = max(0, -remaining)
                remaining = max(0, remaining)
                if overflow > 0:
                    trim_reasons.append("available_input_below_minimum_system_query")
                    raise TokenBudgetExceeded(
                        "This question is too large for the model's context "
                        "window after reserving space for the answer. "
                        "Please shorten the message and try again."
                    )

        # Mode-aware evidence preference: document Q&A prioritizes matter/conversation.
        legal_cap = min(limits.max_legal_evidence_tokens, remaining)
        matter_cap = min(limits.max_matter_evidence_tokens, remaining)
        conversation_cap = min(limits.max_conversation_document_tokens, remaining)
        history_cap = min(limits.max_conversation_tokens, remaining)

        if document_qa_mode:
            # Give more room to uploaded docs; keep a thin legal band.
            matter_share = int(remaining * 0.45)
            conversation_share = int(remaining * 0.25)
            legal_share = int(remaining * 0.15)
            history_share = remaining - (
                matter_share + conversation_share + legal_share
            )
        else:
            legal_share = int(remaining * 0.50)
            matter_share = int(remaining * 0.18)
            conversation_share = int(remaining * 0.12)
            history_share = remaining - (
                legal_share + matter_share + conversation_share
            )

        legal_budget = max(0, min(legal_cap, legal_share))
        matter_budget = max(0, min(matter_cap, matter_share))
        conversation_budget = max(0, min(conversation_cap, conversation_share))
        history_budget = max(0, min(history_cap, history_share))

        # Reclaim unused evidence budget into history if evidence is light.
        evidence_tokens_raw = sum(self.counter.count(c.text) for c in chunks)
        if evidence_tokens_raw < (legal_budget + matter_budget + conversation_budget) * 0.4:
            unused = (
                legal_budget + matter_budget + conversation_budget
            ) - evidence_tokens_raw
            history_budget = min(
                limits.max_conversation_tokens,
                history_budget + max(0, unused // 2),
            )

        summary = conversation_summary or self._summary_from_memories(memories)
        packed_history, kept_summary, active_ctx, history_reasons = (
            compress_conversation_history(
                history,
                counter=self.counter,
                budget=history_budget,
                conversation_summary=summary,
            )
        )
        trim_reasons.extend(history_reasons)

        packed_chunks, _used_by_source, evidence_reasons = pack_evidence_by_budget(
            chunks,
            counter=self.counter,
            legal_budget=legal_budget,
            conversation_budget=conversation_budget,
            matter_budget=matter_budget,
        )
        trim_reasons.extend(evidence_reasons)

        # If still over remaining evidence+history envelope, drop lowest-score chunks.
        history_tokens = sum(
            self.counter.count(m.content) + 2 for m in packed_history
        )
        if kept_summary:
            history_tokens += self.counter.count(kept_summary)
        if active_ctx:
            history_tokens += self.counter.count(active_ctx)

        evidence_tokens = sum(self.counter.count(c.text) for c in packed_chunks)
        variable_budget = remaining
        if history_tokens + evidence_tokens > variable_budget:
            overflow = history_tokens + evidence_tokens - variable_budget
            packed_chunks, dropped = self._drop_lowest_until(
                packed_chunks,
                overflow=overflow,
            )
            if dropped:
                trim_reasons.append(
                    f"secondary_drop_{dropped}_chunks_for_overflow"
                )
            evidence_tokens = sum(
                self.counter.count(c.text) for c in packed_chunks
            )

        legal_tokens = sum(
            self.counter.count(c.text)
            for c in packed_chunks
            if (c.source_type or SourceType.LEGAL.value) == SourceType.LEGAL.value
        )
        conversation_tokens = sum(
            self.counter.count(c.text)
            for c in packed_chunks
            if c.source_type == SourceType.CONVERSATION.value
        )
        matter_tokens = sum(
            self.counter.count(c.text)
            for c in packed_chunks
            if c.source_type == SourceType.MATTER.value
        )

        input_tokens = (
            system_tokens
            + prompt_system_dup
            + query_tokens
            + scaffolding
            + history_tokens
            + evidence_tokens
        )
        # Hard guarantee: never exceed available_input.
        if input_tokens > available_input:
            overflow = input_tokens - available_input
            packed_chunks, dropped = self._drop_lowest_until(
                packed_chunks,
                overflow=overflow,
            )
            evidence_tokens = sum(
                self.counter.count(c.text) for c in packed_chunks
            )
            legal_tokens = sum(
                self.counter.count(c.text)
                for c in packed_chunks
                if (c.source_type or SourceType.LEGAL.value)
                == SourceType.LEGAL.value
            )
            conversation_tokens = sum(
                self.counter.count(c.text)
                for c in packed_chunks
                if c.source_type == SourceType.CONVERSATION.value
            )
            matter_tokens = sum(
                self.counter.count(c.text)
                for c in packed_chunks
                if c.source_type == SourceType.MATTER.value
            )
            input_tokens = (
                system_tokens
                + prompt_system_dup
                + query_tokens
                + scaffolding
                + history_tokens
                + evidence_tokens
            )
            trim_reasons.append("hard_cap_enforced_to_context_window")
            if dropped:
                trim_reasons.append(f"hard_cap_dropped_{dropped}_chunks")

        # Final safety: if history alone still overflows, clear older history.
        if input_tokens > available_input and packed_history:
            packed_history = packed_history[-2:] if len(packed_history) > 2 else []
            history_tokens = sum(
                self.counter.count(m.content) + 2 for m in packed_history
            )
            input_tokens = (
                system_tokens
                + prompt_system_dup
                + query_tokens
                + scaffolding
                + history_tokens
                + evidence_tokens
            )
            trim_reasons.append("emergency_history_collapse")

        if input_tokens > available_input:
            # Last resort: drop all variable context and clamp the report.
            packed_chunks = []
            packed_history = []
            history_tokens = 0
            evidence_tokens = 0
            legal_tokens = conversation_tokens = matter_tokens = 0
            input_tokens = min(
                available_input,
                system_tokens + query_tokens + scaffolding,
            )
            trim_reasons.append("emergency_clear_variable_context")
            logger.error(
                "Token budget could not fit system+query into available_input=%s "
                "context_window=%s reserved=%s",
                available_input,
                caps.context_window,
                reserved,
            )

        docs = group_chunks_by_document(packed_chunks)
        duplicates_removed = max(0, len(chunks) - len(packed_chunks))
        # more accurate: from reasons if present
        for reason in trim_reasons:
            if reason.startswith("removed_") and reason.endswith("_duplicate_chunks"):
                try:
                    duplicates_removed = int(reason.split("_")[1])
                except (IndexError, ValueError):
                    pass

        cost = None
        if limits.cost_tracking_enabled:
            cost = estimate_cost_usd(
                capabilities=caps,
                input_tokens=input_tokens,
                output_tokens=reserved,
            )

        budget_trimmed = bool(trim_reasons) or len(packed_chunks) < len(chunks) or (
            len(packed_history) < len(history)
        )
        trimming_reason = "; ".join(trim_reasons) if trim_reasons else None

        remaining_input = max(0, available_input - input_tokens)
        usage_percent = 0.0
        if caps.context_window > 0:
            usage_percent = round(
                100.0 * (input_tokens + reserved) / caps.context_window,
                2,
            )

        metadata = TokenUsageMetadata(
            model=caps.model_name,
            provider=caps.provider,
            context_window=caps.context_window,
            input_tokens=input_tokens,
            output_tokens=reserved,
            total_tokens=input_tokens + reserved,
            system_tokens=system_tokens,
            query_tokens=query_tokens,
            history_tokens=history_tokens,
            legal_evidence_tokens=legal_tokens,
            conversation_evidence_tokens=conversation_tokens,
            matter_evidence_tokens=matter_tokens,
            reserved_output_tokens=reserved,
            safety_margin_tokens=limits.safety_margin,
            scaffolding_tokens=scaffolding,
            available_input_tokens=available_input,
            remaining_input_tokens=remaining_input,
            usage_percent=usage_percent,
            usage_source="estimate",
            estimated_cost=cost,
            budget_trimmed=budget_trimmed,
            trimming_reason=trimming_reason,
            tokenizer_id=self.counter.tokenizer_id,
            tokenizer_exact=bool(
                self.counter.is_exact and caps.tokenizer_native
            ),
            documents_retained=len(docs),
            chunks_retained=len(packed_chunks),
            chunks_dropped=max(0, len(chunks) - len(packed_chunks)),
            duplicates_removed=duplicates_removed,
        )

        logger.info(
            "Token budget model=%s window=%s available_input=%s "
            "input=%s reserved_out=%s trimmed=%s reason=%s "
            "chunks=%s/%s docs=%s",
            caps.model_name,
            caps.context_window,
            available_input,
            input_tokens,
            reserved,
            budget_trimmed,
            trimming_reason,
            len(packed_chunks),
            len(chunks),
            len(docs),
        )

        return PackedContext(
            history=packed_history,
            chunks=packed_chunks,
            conversation_summary=kept_summary,
            active_legal_context=active_ctx,
            reserved_output_tokens=reserved,
            metadata=metadata,
            question=question,
        )

    def ensure_prompt_within_budget(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        packed: PackedContext,
    ) -> PackedContext:
        """
        Post-PromptBuilder safety check. If the formatted prompt exceeds the
        available input budget, drop lowest-ranked evidence and signal trim.
        """
        reserved = packed.reserved_output_tokens
        available = (
            self.capabilities.context_window
            - reserved
            - self.limits.safety_margin
        )
        system_text = system_prompt if system_prompt is not None else LEGAL_SYSTEM_PROMPT
        total = self.counter.count(system_text) + self.counter.count(prompt)
        if total <= available:
            if packed.metadata:
                packed.metadata.input_tokens = total
                packed.metadata.total_tokens = total + reserved
                packed.metadata.remaining_input_tokens = max(0, available - total)
                if self.capabilities.context_window > 0:
                    packed.metadata.usage_percent = round(
                        100.0
                        * packed.metadata.total_tokens
                        / self.capabilities.context_window,
                        2,
                    )
            return packed

        if packed.chunks:
            overflow = total - available
            chunks, dropped = self._drop_lowest_until(packed.chunks, overflow=overflow)
            packed.chunks = chunks
            if packed.metadata:
                packed.metadata.budget_trimmed = True
                reason = f"post_build_trim_dropped_{dropped}"
                packed.metadata.trimming_reason = (
                    f"{packed.metadata.trimming_reason}; {reason}"
                    if packed.metadata.trimming_reason
                    else reason
                )
                packed.metadata.chunks_retained = len(chunks)
                packed.metadata.chunks_dropped += dropped
                packed.metadata.input_tokens = min(available, total - overflow)
                packed.metadata.total_tokens = (
                    packed.metadata.input_tokens + reserved
                )
                packed.metadata.remaining_input_tokens = max(
                    0,
                    available - packed.metadata.input_tokens,
                )
                if self.capabilities.context_window > 0:
                    packed.metadata.usage_percent = round(
                        100.0
                        * packed.metadata.total_tokens
                        / self.capabilities.context_window,
                        2,
                    )
            return packed

        raise TokenBudgetExceeded(
            "The assembled prompt exceeds the model context window. "
            "Please shorten the question and try again."
        )

    def _truncate_to_token_budget(
        self,
        text: str,
        *,
        max_tokens: int,
    ) -> tuple[str, int]:
        """Keep the start of the user question; never drop the system prompt."""
        current = self.counter.count(text)
        if max_tokens < 1:
            return text, current
        if current <= max_tokens:
            return text, current
        lo, hi = 0, len(text)
        fitted = ""
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = text[:mid].rstrip()
            tokens = self.counter.count(candidate)
            if tokens <= max_tokens:
                fitted = candidate
                lo = mid + 1
            else:
                hi = mid - 1
        return fitted, self.counter.count(fitted)

    def _drop_lowest_until(
        self,
        chunks: list[RetrievedChunk],
        *,
        overflow: int,
    ) -> tuple[list[RetrievedChunk], int]:
        if overflow <= 0 or not chunks:
            return chunks, 0
        from app.rag.token_budget.packing import score_evidence_chunk

        ordered = sorted(
            chunks,
            key=score_evidence_chunk,
        )
        remaining = list(chunks)
        freed = 0
        dropped = 0
        for chunk in ordered:
            if freed >= overflow:
                break
            # Keep at least one chunk if possible.
            if len(remaining) <= 1:
                break
            remaining = [c for c in remaining if c is not chunk]
            freed += self.counter.count(chunk.text)
            dropped += 1
        return remaining, dropped

    @staticmethod
    def _summary_from_memories(memories: list[Memory] | None) -> str | None:
        if not memories:
            return None
        # Prefer legal_context / chat memories as lightweight summaries.
        candidates = [
            m for m in memories
            if (m.memory_type.value if hasattr(m.memory_type, "value") else str(m.memory_type))
            in {"legal_context", "chat"}
        ]
        if not candidates:
            candidates = list(memories)
        top = sorted(candidates, key=lambda m: m.score, reverse=True)[:3]
        texts = [m.text.strip() for m in top if m.text and m.text.strip()]
        if not texts:
            return None
        return "Conversation memory summary:\n- " + "\n- ".join(texts)


# Alias requested in the product brief.
ContextBudgetManager = TokenBudgetManager
