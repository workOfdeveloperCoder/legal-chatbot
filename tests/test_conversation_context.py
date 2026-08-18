from __future__ import annotations

from app.rag.conversation_context import ConversationContextBuilder
from app.rag.context_resolver import ContextResolver
from app.rag.models import Message


def test_active_context_extracts_statutes_and_issue():
    history = [
        Message(role="user", content="Explain Section 54-C."),
        Message(
            role="assistant",
            content="Section 54-C can restrict interim injunctions.",
        ),
    ]
    builder = ConversationContextBuilder()
    ctx = builder.build(history, current_question="What about double jeopardy?")
    assert any("54-C" in s or "54-c" in s.lower() for s in ctx.active_statutes)
    assert ctx.active_issue is not None
    topic = ctx.to_topic_string()
    assert topic
    assert "54" in topic.lower() or "clog" in topic.lower() or "discretion" in topic.lower()


def test_followup_retains_section_54c_context():
    history = [
        Message(role="user", content="Explain Section 54-C."),
        Message(
            role="assistant",
            content="It concerns interim injunctions and deposit requirements.",
        ),
    ]
    resolver = ContextResolver()
    active = resolver.build_active_context(
        "What about double jeopardy?",
        history=history,
    )
    resolved = resolver.resolve(
        "What about double jeopardy?",
        history=history,
        active_context=active,
    )
    lower = resolved.lower()
    assert "double jeopardy" in lower
    assert "54" in lower or "section" in lower
