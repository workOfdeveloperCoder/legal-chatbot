from __future__ import annotations

import pytest

from app.rag.legal_query_planner import AnswerMode, LegalQueryPlanner
from app.services.quick_actions import list_quick_actions, resolve_quick_action
from app.services.query_router import QueryRouter, Task


def test_catalog_has_six_frontend_actions():
    actions = list_quick_actions()
    assert len(actions) == 6
    assert [action.id for action in actions] == [1, 2, 3, 4, 5, 6]
    assert [action.icon for action in actions] == [
        "FilePlus",
        "Scale",
        "FilePenLine",
        "FileCog",
        "Gavel",
        "Landmark",
    ]
    by_slug = {action.slug: action for action in actions}
    assert by_slug["find_authorities"].enables_web_search is True
    assert by_slug["compare_provisions"].enables_web_search is True
    assert by_slug["summarize_document"].requires_document is True
    assert by_slug["analyze_contract"].requires_document is True


@pytest.mark.parametrize(
    ("value", "slug"),
    [
        (1, "draft_document"),
        ("2", "find_authorities"),
        ("summarize_document", "summarize_document"),
        ("prepare-hearing", "prepare_hearing"),
    ],
)
def test_resolve_quick_action(value, slug):
    action = resolve_quick_action(value)
    assert action is not None
    assert action.slug == slug


@pytest.mark.parametrize(
    ("title", "slug", "task"),
    [
        ("Draft the Document", "draft_document", Task.DOCUMENT_DRAFT),
        ("Find Legal Authorties", "find_authorities", Task.CASE_SEARCH),
        ("Find Legal Authorities", "find_authorities", Task.CASE_SEARCH),
        ("Sumarized Document", "summarize_document", Task.SUMMARIZATION),
        ("Analyze a Contract", "analyze_contract", Task.CONTRACT_REVIEW),
        ("Prepare for Hearing", "prepare_hearing", Task.HEARING_PREP),
        ("Compare Legal Provisions", "compare_provisions", Task.COMPARE_PROVISIONS),
        ("Compare Laws side by side", "compare_provisions", Task.COMPARE_PROVISIONS),
    ],
)
@pytest.mark.asyncio
async def test_card_title_routes_to_feature(title, slug, task):
    action = resolve_quick_action(title)
    assert action is not None
    assert action.slug == slug
    router = QueryRouter()
    result = await router.route(title)
    assert result.task == task


def test_quick_actions_json_matches_frontend_cards():
    from app.schemas.chat import QuickActionResponse, QuickActionsListResponse

    payload = QuickActionsListResponse(
        quick_actions=[
            QuickActionResponse(
                id=action.id,
                slug=action.slug,
                title=action.title,
                description=action.description,
                icon=action.icon,
                prompt=action.prompt,
                requires_document=action.requires_document,
                enables_web_search=action.enables_web_search,
            )
            for action in list_quick_actions()
        ]
    )
    body = payload.model_dump(by_alias=True)
    cards = body["quickActions"]
    assert [card["id"] for card in cards] == [1, 2, 3, 4, 5, 6]
    assert [card["icon"] for card in cards] == [
        "FilePlus",
        "Scale",
        "FilePenLine",
        "FileCog",
        "Gavel",
        "Landmark",
    ]
    assert cards[0]["title"] == "Draft the Document"
    assert cards[3]["requiresDocument"] is True
    assert cards[1]["enablesWebSearch"] is True


def test_is_card_click_matches_prompt_and_title():
    from app.services.quick_actions import is_card_click, resolve_quick_action

    action = resolve_quick_action("draft_document")
    assert action is not None
    assert is_card_click("Draft the Document", action) is True
    assert is_card_click(action.prompt, action) is True
    assert is_card_click("Please draft a plaint for ejectment", action) is False


def test_find_authorities_prompt_resolves_and_opens():
    from app.services.quick_actions import (
        CARD_OPENING_REPLIES,
        resolve_from_message,
        should_return_card_opening,
    )

    prompt = (
        "Find the governing Pakistani legal authorities for my issue. "
        "Search statutes, rules, and case law, and cite what you find. "
        "Ask me the issue, jurisdiction, and any section or citation "
        "I already have if that is missing."
    )
    action = resolve_from_message(prompt)
    assert action is not None
    assert action.slug == "find_authorities"
    assert should_return_card_opening(prompt, action) is True
    assert "issue" in CARD_OPENING_REPLIES["find_authorities"].lower()

    # FE offline fallback may omit the trailing Ask-me sentence.
    short = (
        "Find the governing Pakistani legal authorities for my issue. "
        "Search statutes, rules, and case law, and cite what you find."
    )
    assert should_return_card_opening(
        short,
        action,
        quick_action="find_authorities",
    )


def test_prepare_hearing_short_prompt_opens_without_slug():
    from app.services.quick_actions import (
        CARD_OPENING_REPLIES,
        resolve_from_message,
        should_return_card_opening,
    )

    short = (
        "Help me prepare for a hearing. Build a structured note with the "
        "issues, facts to prove, supporting authorities, likely objections, "
        "and oral submissions."
    )
    action = resolve_from_message(short)
    assert action is not None
    assert action.slug == "prepare_hearing"
    assert should_return_card_opening(short, action) is True
    assert "forum" in CARD_OPENING_REPLIES["prepare_hearing"].lower()


def test_prepare_hearing_slug_always_opens():
    from app.services.quick_actions import (
        resolve_quick_action,
        should_return_card_opening,
    )

    action = resolve_quick_action("prepare_hearing")
    assert should_return_card_opening(
        "literally anything on first card tap",
        action,
        quick_action="prepare_hearing",
    )




def test_real_issue_without_slug_is_not_opening():
    from app.services.quick_actions import (
        resolve_quick_action,
        should_return_card_opening,
    )

    action = resolve_quick_action("find_authorities")
    assert (
        should_return_card_opening(
            "Is electricity theft under section 54-C of the Electricity Act "
            "1910 cognizable in Lahore?",
            action,
        )
        is False
    )

def test_card_opening_replies_cover_clarifying_actions():
    from app.services.quick_actions import CARD_OPENING_REPLIES

    assert "draft_document" in CARD_OPENING_REPLIES
    assert "find_authorities" in CARD_OPENING_REPLIES
    assert "type" in CARD_OPENING_REPLIES["draft_document"].lower()


@pytest.mark.asyncio
async def test_section_54c_rewrite_adds_electricity_act():
    from app.rag.query_rewriter import QueryRewriter

    result = await QueryRewriter().rewrite(question="what is section 54-c")
    assert "Electricity Act" in result.rewritten_query
    assert "54-C" in result.rewritten_query or "54-c" in result.rewritten_query.lower()



@pytest.mark.asyncio
async def test_router_honors_quick_action_over_message():
    router = QueryRouter()
    result = await router.route(
        "hello",
        quick_action="draft_document",
    )
    assert result.task == Task.DOCUMENT_DRAFT
    assert result.web_search is False

    result = await router.route(
        "hello",
        quick_action="find_authorities",
    )
    assert result.task == Task.CASE_SEARCH
    assert result.web_search is True


@pytest.mark.asyncio
async def test_router_detects_hearing_and_compare():
    router = QueryRouter()
    hearing = await router.detect("Prepare for hearing on the bail application")
    assert hearing == Task.HEARING_PREP
    compare = await router.detect("Compare section 497 CrPC with section 498")
    assert compare == Task.COMPARE_PROVISIONS


@pytest.mark.asyncio
async def test_router_detects_internet_search_intent():
    router = QueryRouter()
    result = await router.route("Search the internet for the latest bail judgments")
    assert result.web_search is True
    assert result.task in {Task.LEGAL_QA, Task.CASE_SEARCH, Task.CASE_ANALYSIS}


@pytest.mark.asyncio
async def test_planner_sets_web_search_only_when_toggle_on():
    planner = LegalQueryPlanner()
    off = await planner.plan(
        question="Section 497 CrPC",
        quick_action="find_authorities",
        web_search=False,
    )
    assert off.task == Task.CASE_SEARCH
    assert off.web_search is False
    on = await planner.plan(
        question="Section 497 CrPC",
        quick_action="find_authorities",
        web_search=True,
    )
    assert on.web_search is True


@pytest.mark.asyncio
async def test_summarize_without_upload_asks_for_document():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock
    from uuid import uuid4

    from app.schemas.chat import ChatRequest
    from app.services.chat_service import ChatService

    user = SimpleNamespace(id=uuid4())
    conversation = SimpleNamespace(
        id=uuid4(),
        matter_id=None,
        title="New chat",
    )
    conversation_service = MagicMock()
    conversation_service.get_or_create = AsyncMock(return_value=conversation)
    conversation_service.save_exchange = AsyncMock()
    documents = MagicMock()
    documents.has_for_scope = AsyncMock(return_value=False)
    activity = MagicMock()
    activity.record = AsyncMock()

    service = ChatService(
        rag_service=MagicMock(),
        conversation_service=conversation_service,
        memory_service=MagicMock(),
        prompt_context_service=MagicMock(),
        query_router=MagicMock(),
        document_repository=documents,
        activity_log_service=activity,
    )
    payload = ChatRequest(message="Sumarized Document")
    response = await service.chat(user=user, payload=payload)
    assert "upload" in response.response.lower()
    service._rag.execute.assert_not_called()
    conversation_service.save_exchange.assert_awaited()


@pytest.mark.asyncio
async def test_planner_hearing_and_compare_modes():
    planner = LegalQueryPlanner()
    hearing = await planner.plan(
        question="Build hearing notes for tomorrow's injunction",
        quick_action="prepare_hearing",
    )
    assert hearing.answer_mode == AnswerMode.HEARING_PREP
    assert hearing.requires_legal_authority is True

    compare = await planner.plan(
        question="Compare Article 9 and Article 10A",
        quick_action="compare_provisions",
    )
    assert compare.answer_mode == AnswerMode.COMPARE_PROVISIONS
    assert compare.web_search is False
    compare_on = await planner.plan(
        question="Compare Article 9 and Article 10A",
        quick_action="compare_provisions",
        web_search=True,
    )
    assert compare_on.web_search is True
