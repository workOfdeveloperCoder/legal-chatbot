from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.contracts.extractor import ClauseExtractor, quote_in_document
from app.contracts.fields import DEFAULT_CLAUSE_FIELDS, resolve_fields
from app.contracts.format import format_clause_cards, format_review_table
from app.contracts.json_parse import parse_json_object
from app.rag.legal_query_planner import AnswerMode, LegalQueryPlanner, RetrievalStrategy
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.contracts import (
    ClauseCard,
    ClauseExtractionResponse,
    ClauseFieldSpec,
    ClauseStatus,
    ReviewCell,
    ReviewRow,
    ReviewTableResponse,
)
from app.schemas.llm import ChatCompletionResponse
from app.services.chat_service import ChatService
from app.services.contract_analysis_service import ContractAnalysisService
from app.services.query_router import QueryRouter, Task
from app.services.quick_actions import list_quick_actions, resolve_quick_action


SAMPLE_CONTRACT = """
This Services Agreement is made on 1 March 2025 between
Acme Private Limited ("Client") and Beta Counsel ("Counsel").
This Agreement shall be governed by the laws of Pakistan.
Liability of Counsel is capped at PKR 1,000,000.
Either party may terminate on 30 days written notice.
Confidential information shall not be disclosed for 3 years.
"""


class ScriptedLLM:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls = 0
        self.requests = []

    async def generate(self, request):
        self.calls += 1
        self.requests.append(request)
        return ChatCompletionResponse(content=self.content)


def _extraction_json(**overrides) -> str:
    fields = {
        "parties": {
            "value": "Acme Private Limited and Beta Counsel",
            "quote": 'Acme Private Limited ("Client") and Beta Counsel ("Counsel")',
            "status": "found",
            "confidence": 0.95,
        },
        "governing_law": {
            "value": "Laws of Pakistan",
            "quote": "governed by the laws of Pakistan",
            "status": "found",
            "confidence": 0.9,
        },
        "liability_cap": {
            "value": "PKR 1,000,000",
            "quote": "capped at PKR 1,000,000",
            "status": "found",
            "confidence": 0.9,
        },
        "termination": {
            "value": "30 days written notice",
            "quote": "terminate on 30 days written notice",
            "status": "found",
            "confidence": 0.85,
        },
        "renewal": {
            "value": None,
            "quote": None,
            "status": "missing",
            "confidence": 0.8,
        },
    }
    fields.update(overrides)
    payload = {
        "fields": [
            {"key": key, **value} for key, value in fields.items()
        ]
    }
    return json.dumps(payload)


def test_default_field_keys_are_stable_for_playbooks():
    keys = [field.key for field in DEFAULT_CLAUSE_FIELDS]
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


def test_resolve_fields_keeps_unknown_keys_out():
    fields = resolve_fields(["parties", "nope", "liability_cap"])
    assert [field.key for field in fields] == ["parties", "liability_cap"]


def test_parse_json_from_think_tags_and_fences():
    blob = (
        "<think>planning</think>\n"
        "```json\n"
        '{"fields":[{"key":"parties","value":"A","status":"found"}]}\n'
        "```\n"
    )
    parsed = parse_json_object(blob)
    assert parsed["fields"][0]["key"] == "parties"


def test_parse_json_embedded_in_prose():
    parsed = parse_json_object(
        'Here you go: {"fields":[{"key":"term","value":"1 year","status":"found"}]} thanks'
    )
    assert parsed["fields"][0]["key"] == "term"


def test_quote_grounding_requires_document_span():
    assert quote_in_document("laws of Pakistan", SAMPLE_CONTRACT) is True
    assert quote_in_document("governed by Delaware law", SAMPLE_CONTRACT) is False


@pytest.mark.asyncio
async def test_extractor_builds_cards_and_drops_ungrounded_quote():
    raw = _extraction_json(
        payment={
            "value": "not in the paper",
            "quote": "this quote is invented",
            "status": "found",
            "confidence": 0.7,
        }
    )
    extractor = ClauseExtractor(ScriptedLLM(raw))
    result = await extractor.extract(
        document_id=uuid4(),
        filename="nda.txt",
        text=SAMPLE_CONTRACT,
        fields=resolve_fields(
            ["parties", "governing_law", "liability_cap", "termination", "renewal", "payment"]
        ),
    )
    by_key = {card.key: card for card in result.cards}
    assert by_key["parties"].status == ClauseStatus.FOUND
    assert by_key["parties"].quote_grounded is True
    assert by_key["renewal"].status == ClauseStatus.MISSING
    assert by_key["payment"].quote is None
    assert by_key["payment"].quote_grounded is False


@pytest.mark.asyncio
async def test_extractor_empty_text_marks_missing():
    extractor = ClauseExtractor(ScriptedLLM("{}"))
    result = await extractor.extract(
        document_id=uuid4(),
        filename="empty.txt",
        text="   ",
        fields=resolve_fields(["parties"]),
    )
    assert result.cards[0].status == ClauseStatus.MISSING
    assert extractor._llm.calls == 0  # type: ignore[attr-defined]


def test_clause_cards_markdown_and_table():
    doc_id = uuid4()
    extraction = ClauseExtractionResponse(
        document_id=doc_id,
        filename="nda.txt",
        cards=[
            ClauseCard(
                key="parties",
                label="Parties",
                status=ClauseStatus.FOUND,
                value="Acme and Beta",
                quote="Acme Private Limited",
                quote_grounded=True,
                confidence=0.9,
            ),
            ClauseCard(
                key="renewal",
                label="Renewal",
                status=ClauseStatus.MISSING,
            ),
        ],
    )
    markdown = format_clause_cards(extraction)
    assert "**Parties** — Acme and Beta" in markdown
    assert "**Renewal**" in markdown

    table = ReviewTableResponse(
        columns=[
            ClauseFieldSpec(key="parties", label="Parties", question="Who?"),
            ClauseFieldSpec(key="renewal", label="Renewal", question="Renew?"),
        ],
        rows=[
            ReviewRow(
                document_id=doc_id,
                filename="nda.txt",
                cells=[
                    ReviewCell(
                        key="parties",
                        status=ClauseStatus.FOUND,
                        value="Acme and Beta",
                        quote="Acme Private Limited",
                        quote_grounded=True,
                        confidence=0.9,
                    ),
                    ReviewCell(
                        key="renewal",
                        status=ClauseStatus.MISSING,
                    ),
                ],
            )
        ],
        document_count=1,
    )
    grid = format_review_table(table)
    assert "| Document" in grid
    assert "nda.txt" in grid
    assert "Acme and Beta" in grid


@pytest.mark.asyncio
async def test_review_table_one_row_per_document():
    user_id = uuid4()
    matter_id = uuid4()
    doc_a = SimpleNamespace(
        id=uuid4(),
        filename="a.txt",
        owner_id=user_id,
        processed=True,
        extracted_text=SAMPLE_CONTRACT,
    )
    doc_b = SimpleNamespace(
        id=uuid4(),
        filename="b.txt",
        owner_id=user_id,
        processed=True,
        extracted_text=SAMPLE_CONTRACT,
    )
    documents = MagicMock()
    documents.list_for_scope = AsyncMock(return_value=[doc_a, doc_b])
    matters = MagicMock()
    matters.get = AsyncMock(return_value=SimpleNamespace(id=matter_id, owner_id=user_id))
    llm = ScriptedLLM(_extraction_json())
    service = ContractAnalysisService(
        llm=llm,
        documents=documents,
        matters=matters,
    )
    table = await service.review_table(
        user_id=user_id,
        matter_id=matter_id,
        field_keys=["parties", "governing_law"],
    )
    assert table.document_count == 2
    assert len(table.rows) == 2
    assert {row.filename for row in table.rows} == {"a.txt", "b.txt"}
    assert llm.calls == 2
    assert [column.key for column in table.columns] == ["parties", "governing_law"]


@pytest.mark.asyncio
async def test_extract_document_404_for_other_user():
    documents = MagicMock()
    documents.get_by_id = AsyncMock(
        return_value=SimpleNamespace(id=uuid4(), owner_id=uuid4())
    )
    service = ContractAnalysisService(
        llm=ScriptedLLM("{}"),
        documents=documents,
        matters=MagicMock(),
    )
    with pytest.raises(HTTPException) as exc:
        await service.extract_document(user_id=uuid4(), document_id=uuid4())
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_router_detects_review_table_and_clause_extraction():
    router = QueryRouter()
    table = await router.route("Build a review table across all contracts")
    assert table.task == Task.REVIEW_TABLE
    clauses = await router.route("Extract clauses from this NDA")
    assert clauses.task == Task.CONTRACT_REVIEW


@pytest.mark.asyncio
async def test_planner_review_table_skips_retrieval():
    planner = LegalQueryPlanner()
    plan = await planner.plan(
        question="Build a review table",
        quick_action="review_table",
        has_uploaded_documents=True,
    )
    assert plan.task == Task.REVIEW_TABLE
    assert plan.answer_mode == AnswerMode.REVIEW_TABLE
    assert plan.retrieval_strategy == RetrievalStrategy.NONE
    assert plan.uploaded_document_primary is True


def test_hidden_review_table_does_not_change_six_card_catalog():
    assert len(list_quick_actions()) == 6
    action = resolve_quick_action("review_table")
    assert action is not None
    assert action.slug == "review_table"
    assert resolve_quick_action("Diligence table") is not None


def test_chat_response_serializes_camel_case_cards():
    payload = ChatResponse(
        conversation_id=uuid4(),
        response="ok",
        clause_cards=ClauseExtractionResponse(
            document_id=uuid4(),
            filename="nda.txt",
            cards=[
                ClauseCard(
                    key="parties",
                    label="Parties",
                    status=ClauseStatus.FOUND,
                    value="A and B",
                    quote_grounded=True,
                )
            ],
        ),
    )
    body = payload.model_dump(by_alias=True)
    assert "clauseCards" in body
    assert body["clauseCards"]["documentId"]
    assert body["clauseCards"]["cards"][0]["quoteGrounded"] is True


@pytest.mark.asyncio
async def test_analyze_contract_attaches_clause_cards():
    user = SimpleNamespace(id=uuid4())
    conversation = SimpleNamespace(id=uuid4(), matter_id=None, title="Chat")
    conversation_service = MagicMock()
    conversation_service.get_or_create = AsyncMock(return_value=conversation)
    conversation_service.save_exchange = AsyncMock()
    documents = MagicMock()
    documents.has_for_scope = AsyncMock(return_value=True)
    activity = MagicMock()
    activity.record = AsyncMock()
    rag = MagicMock()
    rag.execute = AsyncMock(
        return_value={
            "answer": "Narrative review of the NDA.",
            "citations": [],
            "sources": [],
            "resources": [],
            "retrieval_metadata": None,
            "grounding_status": "grounded",
        }
    )
    extraction = ClauseExtractionResponse(
        document_id=uuid4(),
        filename="nda.txt",
        cards=[
            ClauseCard(
                key="parties",
                label="Parties",
                status=ClauseStatus.FOUND,
                value="Acme and Beta",
            )
        ],
    )
    contracts = MagicMock()
    contracts.extract_for_chat = AsyncMock(return_value=extraction)
    contracts.cards_markdown = lambda item: format_clause_cards(item)
    context = MagicMock()
    context.build = AsyncMock(
        return_value=SimpleNamespace(history=[], memories=[])
    )
    memory = MagicMock()
    memory.process_chat = AsyncMock()

    service = ChatService(
        rag_service=rag,
        conversation_service=conversation_service,
        memory_service=memory,
        prompt_context_service=context,
        query_router=QueryRouter(),
        document_repository=documents,
        activity_log_service=activity,
        contract_analysis=contracts,
    )
    payload = ChatRequest(message="Analyze a Contract")
    response = await service.chat(user=user, payload=payload)
    assert response.clause_cards is not None
    assert "Clause cards" in response.response
    assert "Acme and Beta" in response.response
    rag.execute.assert_awaited()
    contracts.extract_for_chat.assert_awaited()


@pytest.mark.asyncio
async def test_review_table_chat_skips_rag():
    user = SimpleNamespace(id=uuid4())
    conversation = SimpleNamespace(id=uuid4(), matter_id=uuid4(), title="Chat")
    conversation_service = MagicMock()
    conversation_service.get_or_create = AsyncMock(return_value=conversation)
    conversation_service.save_exchange = AsyncMock()
    documents = MagicMock()
    documents.has_for_scope = AsyncMock(return_value=True)
    activity = MagicMock()
    activity.record = AsyncMock()
    rag = MagicMock()
    rag.execute = AsyncMock()
    table = ReviewTableResponse(
        matter_id=conversation.matter_id,
        columns=[ClauseFieldSpec(key="parties", label="Parties", question="Who?")],
        rows=[
            ReviewRow(
                document_id=uuid4(),
                filename="nda.txt",
                cells=[
                    ReviewCell(
                        key="parties",
                        status=ClauseStatus.FOUND,
                        value="Acme",
                    )
                ],
            )
        ],
        document_count=1,
    )
    contracts = MagicMock()
    contracts.review_table = AsyncMock(return_value=table)
    contracts.table_markdown = lambda item: format_review_table(item)
    context = MagicMock()
    context.build = AsyncMock(
        return_value=SimpleNamespace(history=[], memories=[])
    )

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
    payload = ChatRequest(message="Review Table")
    response = await service.chat(user=user, payload=payload)
    rag.execute.assert_not_called()
    assert response.review_table is not None
    assert "Review table" in response.response
    assert "nda.txt" in response.response


@pytest.mark.asyncio
async def test_review_table_without_upload_asks_for_document():
    user = SimpleNamespace(id=uuid4())
    conversation = SimpleNamespace(id=uuid4(), matter_id=None, title="Chat")
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
        query_router=QueryRouter(),
        document_repository=documents,
        activity_log_service=activity,
        contract_analysis=MagicMock(),
    )
    payload = ChatRequest(message="Review Table")
    response = await service.chat(user=user, payload=payload)
    assert "upload" in response.response.lower()
    service._rag.execute.assert_not_called()
