from __future__ import annotations

import re

from app.rag.conversation_context import (
    ActiveLegalContext,
    ConversationContextBuilder,
)
from app.rag.models import Message


_FOLLOWUP_PATTERN = re.compile(
    r"^(?:what about|how about|and what|does this apply|what if|"
    r"which section|show me where|compare this|and for|"
    r"does that apply|what happens if|can it be|can this be|"
    r"can that be|is it|is this)\b",
    re.IGNORECASE,
)

_EXPLICIT_INTENT_PATTERN = re.compile(
    r"\b(?:under the|pursuant to|according to|in punjab|in sindh|"
    r"in khyber|in balochistan|limitation act|penal code|"
    r"constitution of pakistan|code of criminal procedure|"
    r"code of civil procedure|contract act|family law|"
    r"specific performance|qanun-e-shahadat)\b",
    re.IGNORECASE,
)

_TOPIC_PATTERN = re.compile(
    r"\b(?:section|article|u/s)\s+([0-9A-Za-z\-]+)|"
    r"\b(limitation|appeal|bail|murder|contract|negligence|"
    r"inheritance|injunction|recovery|arbitration|"
    r"constitutional petition|specific performance|"
    r"limitation period|double jeopardy|discretion|"
    r"unreasonableness|clog|54-c|54 c)\b",
    re.IGNORECASE,
)

_SHORT_FOLLOWUP = re.compile(
    r"^(?:what about|how about|and |also |but )?.{1,80}$",
    re.IGNORECASE,
)


class ContextResolver:
    """
    Resolve follow-up questions into standalone retrieval queries.

    Uses structured ActiveLegalContext when available; falls back to
    topic extraction from recent history. Explicit current intent wins.
    """

    def __init__(
        self,
        context_builder: ConversationContextBuilder | None = None,
    ) -> None:
        self._context_builder = context_builder or ConversationContextBuilder()

    def build_active_context(
        self,
        question: str,
        history: list[Message] | None = None,
        *,
        matter_id: str | None = None,
        document_names: list[str] | None = None,
    ) -> ActiveLegalContext:
        return self._context_builder.build(
            history,
            current_question=question,
            matter_id=matter_id,
            document_names=document_names,
        )

    def resolve(
        self,
        question: str,
        history: list[Message] | None = None,
        *,
        active_context: ActiveLegalContext | None = None,
    ) -> str:
        normalized = " ".join(question.split()).strip()
        if not normalized:
            return question

        if self._has_explicit_intent(normalized):
            return normalized

        if not self._is_followup(normalized):
            return normalized

        topic = None
        if active_context is not None:
            topic = active_context.to_topic_string()
        if not topic and history:
            # Build structured context from history when caller did not pass one.
            built = self._context_builder.build(history, current_question=normalized)
            topic = built.to_topic_string() or self._extract_conversation_topic(history)
        if not topic:
            return normalized

        return self._compose_contextual_query(normalized, topic)

    @staticmethod
    def _has_explicit_intent(question: str) -> bool:
        return bool(_EXPLICIT_INTENT_PATTERN.search(question))

    @staticmethod
    def _is_followup(question: str) -> bool:
        if _FOLLOWUP_PATTERN.search(question):
            return True
        lower = question.lower().strip("?. ")
        if lower.startswith(("and ", "but ", "also ")):
            return True
        words = lower.split()
        if len(words) <= 8 and _TOPIC_PATTERN.search(lower):
            return True
        return False

    @staticmethod
    def _extract_conversation_topic(history: list[Message]) -> str | None:
        recent = history[-6:]
        topic_parts: list[str] = []

        for message in reversed(recent):
            for match in _TOPIC_PATTERN.finditer(message.content):
                fragment = match.group(0).strip()
                if fragment.lower() not in {
                    part.lower() for part in topic_parts
                }:
                    topic_parts.append(fragment)

        if topic_parts:
            return ", ".join(topic_parts[:3])

        for message in reversed(recent):
            if message.role == "user":
                cleaned = message.content.strip()
                if len(cleaned) > 15:
                    return cleaned[:120].rstrip()
                break

        return None

    @staticmethod
    def _compose_contextual_query(question: str, topic: str) -> str:
        lower = question.lower().strip("?. ")

        if lower.startswith(("can it ", "can this ", "can that ")):
            action = re.sub(
                r"^can (?:it|this|that)\s+",
                "",
                lower,
                flags=re.IGNORECASE,
            ).strip("?. ")
            return (
                f"Can {action} relating to {topic} under Pakistan law"
            )

        if lower.startswith("what about"):
            subject = re.sub(
                r"^what about\s+",
                "",
                lower,
                flags=re.IGNORECASE,
            ).strip("?. ")
            return (
                f"{subject} relating to {topic} under Pakistan law"
            )

        return f"{question.strip('?. ')} in the context of {topic} under Pakistan law"
