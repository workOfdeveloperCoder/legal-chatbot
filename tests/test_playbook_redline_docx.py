from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.contracts.docx_export import (
    export_clauses,
    export_draft,
    export_playbook,
    export_redline,
)
from app.contracts.playbook import (
    PAKISTAN_COMMERCIAL,
    PlaybookVerdict,
    evaluate_playbook,
    get_playbook,
)
from app.contracts.redline import (
    RedlineOp,
    document_hunks,
    playbook_hunks,
    word_diff,
)
from app.rag.legal_query_planner import AnswerMode, LegalQueryPlanner, RetrievalStrategy
from app.schemas.chat import ChatRequest
from app.schemas.contracts import (
    ClauseCard,
    ClauseExtractionResponse,
    ClauseStatus,
    PlaybookFindingResponse,
    PlaybookReviewResponse,
    PlaybookSeverity,
    PlaybookSummary,
    PlaybookVerdict as SchemaVerdict,
    RedlineHunkResponse,
    RedlineResponse,
    RedlineSpanResponse,
)
from app.services.chat_service import ChatService
from app.services.contract_analysis_service import ContractAnalysisService
from app.services.query_router import QueryRouter, Task
from app.services.quick_actions import list_quick_actions, resolve_quick_action


def _card(key: str, value: str | None, status: ClauseStatus = ClauseStatus.FOUND) -> ClauseCard:
    return ClauseCard(
        key=key,
        label=key.replace("_", " ").title(),
        status=status if value else ClauseStatus.MISSING,
        value=value,
        quote=value,
        quote_grounded=bool(value),
        confidence=0.9 if value else 0.0,
    )


def test_default_playbook_covers_all_twelve_keys():
    keys = [rule.key for rule in PAKISTAN_COMMERCIAL.rules]
    assert keys == [
        "parties",
        "effective_date",
        "term",
        "governing_law",
        "dispute_resolution",
        "payment",
        "liability_cap",
        "indemnity",
        "termination",
        "renewal",
        "confidentiality",
        "assignment",
    ]


def test_playbook_pass_and_off_playbook():
    cards = [
        _card("parties", "Acme and Beta"),
        _card("governing_law", "Laws of Pakistan"),
        _card("liability_cap", "Liability is unlimited"),
        _card("termination", "30 days written notice"),
        _card("confidentiality", "Confidential for 3 years"),
        _card("effective_date", None),
        _card("term", "one year"),
        _card("dispute_resolution", "Lahore High Court"),
        _card("payment", "PKR 100,000"),
        _card("indemnity", None, ClauseStatus.MISSING),
        _card("renewal", None, ClauseStatus.MISSING),
        _card("assignment", "may assign without consent"),
    ]
    findings = {item.key: item for item in evaluate_playbook(cards, get_playbook(None))}
    assert findings["governing_law"].verdict == PlaybookVerdict.PASS
    assert findings["liability_cap"].verdict == PlaybookVerdict.OFF_PLAYBOOK
    assert findings["effective_date"].verdict == PlaybookVerdict.MISSING
    assert findings["indemnity"].verdict == PlaybookVerdict.NOT_APPLICABLE
    assert findings["assignment"].verdict == PlaybookVerdict.OFF_PLAYBOOK
    assert findings["liability_cap"].suggested_language


def test_word_diff_marks_insert_and_delete():
    spans = word_diff("pay within 30 days", "pay within 45 days")
    ops = [span.op for span in spans if span.op != RedlineOp.EQUAL]
    assert RedlineOp.DELETE in ops
    assert RedlineOp.INSERT in ops


def test_document_hunks_detect_changed_paragraph():
    left = "Clause 1 stays.\n\nOld liability is unlimited.\n\nClause 3 stays."
    right = "Clause 1 stays.\n\nLiability is capped at PKR 1,000,000.\n\nClause 3 stays."
    hunks = document_hunks(left, right)
    assert len(hunks) == 1
    assert "unlimited" in hunks[0].left
    assert "capped" in hunks[0].right


def test_playbook_hunks_only_for_actionable_findings():
    cards = [
        _card("governing_law", "Laws of Delaware"),
        _card("parties", "Acme and Beta"),
    ]
    # Fill remaining required keys so evaluate still runs on full playbook.
    for key in (
        "effective_date",
        "term",
        "dispute_resolution",
        "payment",
        "liability_cap",
        "indemnity",
        "termination",
        "renewal",
        "confidentiality",
        "assignment",
    ):
        if key not in {card.key for card in cards}:
            cards.append(_card(key, "present value with notice terminate confidential consent"))
    findings = evaluate_playbook(cards, get_playbook(None))
    hunks = playbook_hunks(findings)
    keys = {hunk.key for hunk in hunks}
    assert "governing_law" in keys
    assert "parties" not in keys or any(
        finding.key == "parties" and finding.verdict != PlaybookVerdict.PASS
        for finding in findings
    )


def test_docx_exports_are_real_zip_packages():
    extraction = ClauseExtractionResponse(
        document_id=uuid4(),
        filename="nda.txt",
        cards=[_card("parties", "Acme and Beta")],
    )
    clauses = export_clauses(extraction)
    assert clauses[:2] == b"PK"

    review = PlaybookReviewResponse(
        playbook_id="pakistan_commercial_v1",
        playbook_name="Pakistan commercial (standard)",
        document_id=extraction.document_id,
        filename="nda.txt",
        findings=[
            PlaybookFindingResponse(
                key="parties",
                label="Parties",
                verdict=SchemaVerdict.PASS,
                severity=PlaybookSeverity.BLOCKER,
                extracted_value="Acme and Beta",
                reason="ok",
            )
        ],
        summary=PlaybookSummary(pass_count=1),
        extraction=extraction,
    )
    assert export_playbook(review)[:2] == b"PK"

    redline = RedlineResponse(
        mode="documents",
        left_filename="a.txt",
        right_filename="b.txt",
        hunks=[
            RedlineHunkResponse(
                label="Change 1",
                left="old",
                right="new",
                spans=[
                    RedlineSpanResponse(op="delete", text="old"),
                    RedlineSpanResponse(op="insert", text="new"),
                ],
                markdown="~~old~~**new**",
            )
        ],
    )
    assert export_redline(redline)[:2] == b"PK"
    assert export_draft(title="Notice", body="## Facts\n- One")[:2] == b"PK"


@pytest.mark.asyncio
async def test_router_and_planner_for_playbook_and_redline():
    router = QueryRouter()
    playbook = await router.route("Run playbook review on this NDA")
    assert playbook.task == Task.PLAYBOOK_REVIEW
    redline = await router.route("Show redline between the two drafts")
    assert redline.task == Task.REDLINE

    planner = LegalQueryPlanner()
    plan = await planner.plan(
        question="Playbook Review",
        quick_action="playbook_review",
        has_uploaded_documents=True,
    )
    assert plan.answer_mode == AnswerMode.PLAYBOOK_REVIEW
    assert plan.retrieval_strategy == RetrievalStrategy.NONE


def test_hidden_actions_do_not_inflate_catalog():
    assert len(list_quick_actions()) == 6
    assert resolve_quick_action("playbook_review") is not None
    assert resolve_quick_action("Redline") is not None


@pytest.mark.asyncio
async def test_playbook_chat_skips_rag():
    user = SimpleNamespace(id=uuid4())
    conversation = SimpleNamespace(id=uuid4(), matter_id=None, title="Chat")
    conversation_service = MagicMock()
    conversation_service.get_or_create = AsyncMock(return_value=conversation)
    conversation_service.save_exchange = AsyncMock()
    documents = MagicMock()
    documents.has_for_scope = AsyncMock(return_value=True)
    activity = MagicMock()
    activity.record = AsyncMock()
    review = PlaybookReviewResponse(
        playbook_id="pakistan_commercial_v1",
        playbook_name="Pakistan commercial (standard)",
        document_id=uuid4(),
        filename="nda.txt",
        findings=[],
        summary=PlaybookSummary(),
        extraction=ClauseExtractionResponse(
            document_id=uuid4(),
            filename="nda.txt",
            cards=[],
        ),
    )
    contracts = MagicMock()
    contracts.playbook_review_for_chat = AsyncMock(return_value=review)
    contracts.playbook_markdown = lambda item: "## Playbook review"
    context = MagicMock()
    context.build = AsyncMock(
        return_value=SimpleNamespace(history=[], memories=[])
    )
    rag = MagicMock()
    rag.execute = AsyncMock()

    service = ChatService(
        rag_service=rag,
        conversation_service=conversation_service,
        memory_service=MagicMock(),
        prompt_context_service=context,
        query_router=QueryRouter(),
        document_repository=documents,
        activity_log_service=activity,
        contract_analysis=contracts,
    )
    response = await service.chat(
        user=user,
        payload=ChatRequest(message="Playbook Review"),
    )
    rag.execute.assert_not_called()
    assert response.playbook_review is not None
    assert "Playbook review" in response.response


@pytest.mark.asyncio
async def test_redline_chat_skips_rag():
    user = SimpleNamespace(id=uuid4())
    conversation = SimpleNamespace(id=uuid4(), matter_id=uuid4(), title="Chat")
    conversation_service = MagicMock()
    conversation_service.get_or_create = AsyncMock(return_value=conversation)
    conversation_service.save_exchange = AsyncMock()
    documents = MagicMock()
    documents.has_for_scope = AsyncMock(return_value=True)
    activity = MagicMock()
    activity.record = AsyncMock()
    redline = RedlineResponse(
        mode="playbook",
        left_filename="nda.txt",
        right_filename="Playbook suggested language",
        hunks=[
            RedlineHunkResponse(
                key="governing_law",
                label="Governing law",
                left="Delaware",
                right="Laws of Pakistan",
                markdown="~~Delaware~~**Laws of Pakistan**",
            )
        ],
    )
    contracts = MagicMock()
    contracts.redline_for_chat = AsyncMock(return_value=redline)
    contracts.redline_markdown = lambda item: "## Playbook redline"
    context = MagicMock()
    context.build = AsyncMock(
        return_value=SimpleNamespace(history=[], memories=[])
    )
    rag = MagicMock()
    rag.execute = AsyncMock()

    service = ChatService(
        rag_service=rag,
        conversation_service=conversation_service,
        memory_service=MagicMock(),
        prompt_context_service=context,
        query_router=QueryRouter(),
        document_repository=documents,
        activity_log_service=activity,
        contract_analysis=contracts,
    )
    response = await service.chat(
        user=user,
        payload=ChatRequest(message="Redline"),
    )
    rag.execute.assert_not_called()
    assert response.redline is not None


@pytest.mark.asyncio
async def test_service_export_draft_bytes():
    service = ContractAnalysisService(
        llm=MagicMock(),
        documents=MagicMock(),
        matters=MagicMock(),
    )
    from app.schemas.contracts import ExportKind

    content, filename = await service.export_bytes(
        user_id=uuid4(),
        kind=ExportKind.DRAFT,
        title="Legal Notice",
        body="Please cure the breach within 15 days.",
    )
    assert content[:2] == b"PK"
    assert filename.endswith(".docx")
