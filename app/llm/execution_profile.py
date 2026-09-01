from __future__ import annotations

from dataclasses import dataclass

from app.core.config import settings
from app.llm.model_capabilities import ModelCapabilities, ModelCapabilityRegistry
from app.llm.provider_config import is_online_provider
from app.rag.prompts import LEGAL_SYSTEM_PROMPT, LEGAL_SYSTEM_PROMPT_COMPACT
from app.rag.token_budget.models import TokenBudgetLimits


@dataclass(frozen=True, slots=True)
class ExecutionProfile:
    """
    How RAG should talk to the current chat model.

    Local 32B models need a long system prompt, duplicated user-prompt
    instructions, and optional two-stage reasoning.
    Hosted models (GPT, Claude, DeepSeek API, …) already follow a system
    message — sending less context yields a tighter, cheaper answer.
    """

    compact_prompts: bool
    embed_system_in_user_prompt: bool
    two_stage_reasoning: bool
    reserved_output_tokens: int
    safety_margin: int
    scaffolding_overhead_tokens: int
    max_conversation_tokens: int
    max_legal_evidence_tokens: int
    max_matter_evidence_tokens: int
    max_conversation_document_tokens: int

    @property
    def system_prompt(self) -> str:
        if self.compact_prompts:
            return LEGAL_SYSTEM_PROMPT_COMPACT
        return LEGAL_SYSTEM_PROMPT

    def token_limits(self, capabilities: ModelCapabilities) -> TokenBudgetLimits:
        reserved = min(
            self.reserved_output_tokens,
            capabilities.max_output_tokens,
        )
        return TokenBudgetLimits(
            context_window=capabilities.context_window,
            reserved_output_tokens=reserved,
            safety_margin=self.safety_margin,
            max_conversation_tokens=self.max_conversation_tokens,
            max_legal_evidence_tokens=self.max_legal_evidence_tokens,
            max_matter_evidence_tokens=self.max_matter_evidence_tokens,
            max_conversation_document_tokens=self.max_conversation_document_tokens,
            max_web_evidence_tokens=(
                settings.TOKEN_ONLINE_MAX_WEB_EVIDENCE_TOKENS
                if self.compact_prompts
                else settings.TOKEN_MAX_WEB_EVIDENCE_TOKENS
            ),
            scaffolding_overhead_tokens=self.scaffolding_overhead_tokens,
            token_counting_enabled=settings.TOKEN_COUNTING_ENABLED,
            cost_tracking_enabled=settings.TOKEN_COST_TRACKING_ENABLED,
        )

    @classmethod
    def for_capabilities(
        cls,
        capabilities: ModelCapabilities | None,
    ) -> ExecutionProfile:
        caps = capabilities or ModelCapabilityRegistry.resolve()
        if is_online_provider(caps.provider):
            return cls.online(caps)
        return cls.local(caps)

    @classmethod
    def local(cls, capabilities: ModelCapabilities | None = None) -> ExecutionProfile:
        caps = capabilities or ModelCapabilityRegistry.resolve()
        reserved = min(
            caps.max_output_tokens,
            settings.TOKEN_RESERVED_OUTPUT_TOKENS,
        )
        return cls(
            compact_prompts=False,
            embed_system_in_user_prompt=True,
            two_stage_reasoning=settings.ENABLE_TWO_STAGE_REASONING,
            reserved_output_tokens=reserved,
            safety_margin=settings.TOKEN_SAFETY_MARGIN,
            scaffolding_overhead_tokens=settings.TOKEN_SCAFFOLDING_OVERHEAD,
            max_conversation_tokens=settings.TOKEN_MAX_CONVERSATION_TOKENS,
            max_legal_evidence_tokens=settings.TOKEN_MAX_LEGAL_EVIDENCE_TOKENS,
            max_matter_evidence_tokens=settings.TOKEN_MAX_MATTER_EVIDENCE_TOKENS,
            max_conversation_document_tokens=settings.TOKEN_MAX_CONVERSATION_DOCUMENT_TOKENS,
        )

    @classmethod
    def online(cls, capabilities: ModelCapabilities | None = None) -> ExecutionProfile:
        caps = capabilities or ModelCapabilityRegistry.resolve()
        reserved = min(
            caps.max_output_tokens,
            settings.TOKEN_ONLINE_RESERVED_OUTPUT,
        )
        return cls(
            compact_prompts=True,
            embed_system_in_user_prompt=False,
            two_stage_reasoning=settings.LLM_TWO_STAGE_ONLINE,
            reserved_output_tokens=reserved,
            safety_margin=settings.TOKEN_ONLINE_SAFETY_MARGIN,
            scaffolding_overhead_tokens=settings.TOKEN_ONLINE_SCAFFOLDING,
            max_conversation_tokens=settings.TOKEN_ONLINE_MAX_CONVERSATION_TOKENS,
            max_legal_evidence_tokens=settings.TOKEN_ONLINE_MAX_LEGAL_EVIDENCE_TOKENS,
            max_matter_evidence_tokens=settings.TOKEN_ONLINE_MAX_MATTER_EVIDENCE_TOKENS,
            max_conversation_document_tokens=(
                settings.TOKEN_ONLINE_MAX_CONVERSATION_DOCUMENT_TOKENS
            ),
        )
