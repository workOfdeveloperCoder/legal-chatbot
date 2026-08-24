from __future__ import annotations

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    messages: list[ChatMessage]

    temperature: float = 0.1

    max_tokens: int = 2048


class ChatCompletionResponse(BaseModel):
    content: str

    prompt_tokens: int | None = None

    completion_tokens: int | None = None

    total_tokens: int | None = None


class TokenUsageBreakdown(BaseModel):
    system_tokens: int = 0
    query_tokens: int = 0
    history_tokens: int = 0
    legal_evidence_tokens: int = 0
    conversation_evidence_tokens: int = 0
    matter_evidence_tokens: int = 0
    scaffolding_tokens: int = 0


class TokenUsageResponse(BaseModel):
    """Cursor-style context-window usage for the current model."""

    model: str
    provider: str
    tokenizer_id: str | None = None
    tokenizer_exact: bool = False
    context_window: int
    max_output_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    available_input_tokens: int = 0
    remaining_input_tokens: int = 0
    usage_percent: float = 0.0
    warning: bool = False
    over_limit: bool = False
    reserved_output_tokens: int = 0
    safety_margin_tokens: int = 0
    estimated_cost_usd: float | None = None
    budget_trimmed: bool = False
    trimming_reason: str | None = None
    usage_source: str = "estimate"
    breakdown: TokenUsageBreakdown = Field(default_factory=TokenUsageBreakdown)


class LLMStatusResponse(BaseModel):
    """Live model limits for the composer (shown before a message is sent)."""

    provider: str
    model: str
    online: bool
    context_window: int
    max_output_tokens: int
    tokenizer_id: str
    tokenizer_exact: bool
    available_input_tokens: int
    reserved_output_tokens: int
    safety_margin_tokens: int
    warning_percent: float
    input_price_per_1m: float | None = None
    output_price_per_1m: float | None = None


class TokenCountRequest(BaseModel):
    text: str = Field(default="", max_length=20000)
    extra_texts: list[str] = Field(default_factory=list, max_length=40)
    include_system_prompt: bool = True


class TokenCountResponse(BaseModel):
    tokens: int
    context_window: int
    available_input_tokens: int
    remaining_tokens: int
    usage_percent: float
    warning: bool
    over_limit: bool
    model: str
    provider: str
    tokenizer_id: str
    tokenizer_exact: bool
