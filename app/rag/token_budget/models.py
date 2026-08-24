from __future__ import annotations

from dataclasses import dataclass, field

from app.rag.models import Message, RetrievedChunk


@dataclass(slots=True)
class TokenBudgetLimits:
    """Configurable budget caps (from Settings / env)."""

    context_window: int
    reserved_output_tokens: int
    safety_margin: int
    max_conversation_tokens: int
    max_legal_evidence_tokens: int
    max_matter_evidence_tokens: int
    max_conversation_document_tokens: int
    scaffolding_overhead_tokens: int = 600
    token_counting_enabled: bool = True
    cost_tracking_enabled: bool = True


@dataclass(slots=True)
class TokenUsageMetadata:
    """Observability — never include document contents."""

    model: str
    context_window: int
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    system_tokens: int = 0
    query_tokens: int = 0
    history_tokens: int = 0
    legal_evidence_tokens: int = 0
    conversation_evidence_tokens: int = 0
    matter_evidence_tokens: int = 0
    reserved_output_tokens: int = 0
    safety_margin_tokens: int = 0
    scaffolding_tokens: int = 0
    available_input_tokens: int = 0
    estimated_cost: float | None = None
    budget_trimmed: bool = False
    trimming_reason: str | None = None
    tokenizer_id: str | None = None
    tokenizer_exact: bool = False
    documents_retained: int = 0
    chunks_retained: int = 0
    chunks_dropped: int = 0
    duplicates_removed: int = 0
    remaining_input_tokens: int = 0
    usage_percent: float = 0.0
    usage_source: str = "estimate"
    provider: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "provider": self.provider,
            "context_window": self.context_window,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "system_tokens": self.system_tokens,
            "query_tokens": self.query_tokens,
            "history_tokens": self.history_tokens,
            "legal_evidence_tokens": self.legal_evidence_tokens,
            "conversation_evidence_tokens": self.conversation_evidence_tokens,
            "matter_evidence_tokens": self.matter_evidence_tokens,
            "reserved_output_tokens": self.reserved_output_tokens,
            "safety_margin_tokens": self.safety_margin_tokens,
            "scaffolding_tokens": self.scaffolding_tokens,
            "available_input_tokens": self.available_input_tokens,
            "remaining_input_tokens": self.remaining_input_tokens,
            "usage_percent": self.usage_percent,
            "usage_source": self.usage_source,
            "estimated_cost": self.estimated_cost,
            "budget_trimmed": self.budget_trimmed,
            "trimming_reason": self.trimming_reason,
            "tokenizer_id": self.tokenizer_id,
            "tokenizer_exact": self.tokenizer_exact,
            "documents_retained": self.documents_retained,
            "chunks_retained": self.chunks_retained,
            "chunks_dropped": self.chunks_dropped,
            "duplicates_removed": self.duplicates_removed,
        }


@dataclass(slots=True)
class PackedContext:
    """Budget-fitted components for PromptBuilder (formatting only)."""

    history: list[Message] = field(default_factory=list)
    chunks: list[RetrievedChunk] = field(default_factory=list)
    conversation_summary: str | None = None
    active_legal_context: str | None = None
    reserved_output_tokens: int = 0
    metadata: TokenUsageMetadata | None = None
    question: str | None = None
