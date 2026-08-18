"""Language detection for Pakistani legal chat (en-PK / ur-PK / pa-PK / mixed)."""

from __future__ import annotations

import pytest

from app.rag.language import (
    ResponseStyle,
    UserLanguage,
    detect_language,
)


class TestRequiredExamples:
    def test_english_legal_question(self):
        result = detect_language("Explain bail under section 497 PPC.")
        assert result.language == UserLanguage.EN_PK
        assert result.response_language == UserLanguage.EN_PK
        assert result.response_style == ResponseStyle.ENGLISH
        assert result.explicit_override is False
        assert result.retrieval_language == "original"
        assert result.confidence >= 0.6

    def test_urdu_script(self):
        result = detect_language("دفعہ 497 کے تحت ضمانت کا طریقہ کیا ہے؟")
        assert result.language == UserLanguage.UR_PK
        assert result.response_language == UserLanguage.UR_PK
        assert result.response_style == ResponseStyle.URDU_SCRIPT
        assert result.script.value in {"arabic", "mixed"}

    def test_roman_urdu_not_english(self):
        result = detect_language(
            "mujhe section 497 ke under bail ka procedure batao"
        )
        assert result.language == UserLanguage.UR_PK
        assert result.response_style == ResponseStyle.ROMAN_URDU
        assert result.language != UserLanguage.EN_PK

    def test_urdu_with_english_legal_terms(self):
        result = detect_language("اس case میں bail کیسے مل سکتی ہے؟")
        assert result.language in {UserLanguage.UR_PK, UserLanguage.MIXED}
        assert result.response_language == UserLanguage.UR_PK
        assert result.response_style == ResponseStyle.URDU_SCRIPT
        assert result.language != UserLanguage.EN_PK

    def test_pakistani_punjabi_not_urdu(self):
        result = detect_language(
            "eh case di bail application kithay file karni ae?"
        )
        assert result.language == UserLanguage.PA_PK
        assert result.response_language == UserLanguage.PA_PK
        assert result.response_style == ResponseStyle.PUNJABI_ROMAN
        assert result.language != UserLanguage.UR_PK

    def test_mixed_with_urdu_override(self):
        result = detect_language(
            "mujhe is case mein bail ka procedure Urdu mein explain karo"
        )
        assert result.explicit_override is True
        assert result.response_language == UserLanguage.UR_PK
        assert result.response_style in {
            ResponseStyle.ROMAN_URDU,
            ResponseStyle.URDU_SCRIPT,
        }
        assert result.language in {UserLanguage.MIXED, UserLanguage.UR_PK}

    def test_explicit_english_override(self):
        result = detect_language("Explain this in English: دفعہ 497 کے تحت ضمانت")
        assert result.explicit_override is True
        assert result.response_language == UserLanguage.EN_PK
        assert result.response_style == ResponseStyle.ENGLISH
        assert result.language in {UserLanguage.MIXED, UserLanguage.UR_PK}

    def test_citation_preserved_as_roman_urdu(self):
        result = detect_language(
            "PLD 2020 SC 123 mein section 497 ka kya principle hai?"
        )
        assert result.language == UserLanguage.UR_PK
        assert result.response_style == ResponseStyle.ROMAN_URDU
        assert result.language != UserLanguage.EN_PK


class TestRomanUrduAndPunjabi:
    @pytest.mark.parametrize(
        "text",
        [
            "mujhe bail ka procedure batao",
            "mujhe is case ka jawab Urdu mein chahiye",
            "yeh section kis offence ke liye hai",
            "bail kab mil sakti hai",
            "Is FIR mein accused ki bail ka kya procedure hai?",
            "Is section ke under bail mil sakti hai?",
        ],
    )
    def test_roman_urdu_examples(self, text: str):
        result = detect_language(text)
        assert result.language in {UserLanguage.UR_PK, UserLanguage.MIXED}
        assert result.response_language == UserLanguage.UR_PK
        assert result.language != UserLanguage.EN_PK

    @pytest.mark.parametrize(
        "text",
        [
            "eh case kithay file hona ae?",
            "menu is qanoon bare daso",
            "eh banda bail te bahar aa sakda ae?",
            "Eh case di bail application kithay file honi ae?",
        ],
    )
    def test_pakistani_punjabi_examples(self, text: str):
        result = detect_language(text)
        assert result.language == UserLanguage.PA_PK
        assert result.response_style == ResponseStyle.PUNJABI_ROMAN

    def test_code_switch_simple_urdu_request(self):
        result = detect_language("Can you explain دفعہ 497 in simple Urdu?")
        assert result.explicit_override is True
        assert result.response_language == UserLanguage.UR_PK
        assert result.language in {UserLanguage.MIXED, UserLanguage.UR_PK}


class TestMetadata:
    def test_to_metadata_keys(self):
        meta = detect_language("Explain bail under section 497 PPC.").to_metadata()
        assert meta["language"] == "en-PK"
        assert meta["retrieval_language"] == "original"
        assert 0.0 <= float(meta["confidence"]) <= 1.0
        assert meta["explicit_override"] is False


class TestNoManualLanguageSelection:
    def test_chat_request_has_no_language_field(self):
        from app.schemas.chat import ChatRequest

        assert "language" not in ChatRequest.model_fields
        assert "locale" not in ChatRequest.model_fields


class TestRoutingAndRewriting:
    @pytest.mark.asyncio
    async def test_urdu_statute_query_is_not_general_chat(self):
        from app.services.query_router import QueryRouter, Task

        router = QueryRouter()
        result = await router.route("دفعہ 497 کے تحت ضمانت کا طریقہ کیا ہے؟")
        assert result.task in {Task.STATUTE_SEARCH, Task.LEGAL_QA}
        assert result.task != Task.GENERAL_CHAT

    @pytest.mark.asyncio
    async def test_rewriter_preserves_original_and_adds_section_anchor(self):
        from app.rag.query_rewriter import QueryRewriter

        rewriter = QueryRewriter()
        question = "دفعہ 497 کے تحت ضمانت کا طریقہ کیا ہے؟"
        result = await rewriter.rewrite(question=question, task="statute_search")
        assert result.original_query == question
        assert "دفعہ 497" in result.rewritten_query
        assert "section 497" in result.rewritten_query.lower()
        assert result.filters.get("section") == "497"
        assert "bail" in result.legal_terms

    @pytest.mark.asyncio
    async def test_rewriter_preserves_citation(self):
        from app.rag.query_rewriter import QueryRewriter

        rewriter = QueryRewriter()
        question = "PLD 2020 SC 123 mein section 497 ka kya principle hai?"
        result = await rewriter.rewrite(question=question, task="legal_qa")
        assert "PLD 2020 SC 123" in result.original_query
        assert "PLD 2020 SC 123" in result.rewritten_query
        assert "497" in result.rewritten_query

    @pytest.mark.asyncio
    async def test_punjabi_query_is_not_translated(self):
        from app.rag.query_rewriter import QueryRewriter

        rewriter = QueryRewriter()
        question = "eh case di bail application kithay file karni ae?"
        result = await rewriter.rewrite(question=question, task="legal_qa")
        assert result.original_query == question
        assert "kithay" in result.rewritten_query
        assert "bail" in result.rewritten_query.lower()

    @pytest.mark.asyncio
    async def test_planner_extracts_arabic_section(self):
        from app.rag.legal_query_planner import LegalQueryPlanner

        planner = LegalQueryPlanner()
        plan = await planner.plan(
            question="دفعہ 497 کے تحت ضمانت کا طریقہ کیا ہے؟",
        )
        assert "497" in plan.sections
        assert "bail" in plan.entities


class TestPromptLanguageInstruction:
    def test_prompt_includes_detected_response_language(self):
        from app.rag.models import RetrievedChunk, SourceType
        from app.rag.prompt_builder import PromptBuilder

        detection = detect_language(
            "mujhe section 497 ke under bail ka procedure batao"
        )
        prompt = PromptBuilder().build(
            question="mujhe section 497 ke under bail ka procedure batao",
            history=[],
            chunks=[
                RetrievedChunk(
                    id="1",
                    chunk_id="1",
                    score=0.9,
                    text="Section 497 PPC addresses bail in certain cases.",
                    source_type=SourceType.LEGAL.value,
                )
            ],
            memories=[],
            language=detection,
        )
        assert "RESPONSE LANGUAGE:" in prompt
        assert "response_language: ur-PK" in prompt
        assert "roman_urdu" in prompt
        assert "LANGUAGE AND CODE-SWITCHING" in prompt

    def test_explicit_override_is_flagged_in_prompt(self):
        from app.rag.models import RetrievedChunk, SourceType
        from app.rag.prompt_builder import PromptBuilder

        question = "Explain this in English: دفعہ 497 کے تحت ضمانت"
        detection = detect_language(question)
        prompt = PromptBuilder().build(
            question=question,
            history=[],
            chunks=[
                RetrievedChunk(
                    id="1",
                    chunk_id="1",
                    score=0.9,
                    text="Section 497.",
                    source_type=SourceType.LEGAL.value,
                )
            ],
            memories=[],
            language=detection,
        )
        assert "explicit_override: true" in prompt
        assert "response_language: en-PK" in prompt
        assert "دفعہ 497" in prompt


class TestLanguageAwareRAG:
    @pytest.mark.asyncio
    async def test_urdu_query_keeps_original_and_exposes_language_metadata(self):
        from types import SimpleNamespace
        from unittest.mock import AsyncMock

        from app.rag.models import RetrievalMetadata, RetrievalOutcome, RetrievedChunk, SourceType
        from app.rag.prompt_builder import PromptBuilder
        from app.rag.query_rewriter import QueryRewriter
        from app.rag.rag_service import RAGService
        from app.services.query_router import Task
        from app.services.response_formatter import ResponseFormatter

        question = "دفعہ 497 کے تحت ضمانت کا طریقہ کیا ہے؟"
        chunk = RetrievedChunk(
            id="legal-1",
            chunk_id="legal-1",
            score=0.92,
            relevance_score=0.92,
            text=(
                "Section 497 of the Pakistan Penal Code concerns bail. "
                "The court may grant bail subject to the statutory conditions."
            ),
            law_name="Pakistan Penal Code",
            sections=["497"],
            source_type=SourceType.LEGAL.value,
        )
        retriever = AsyncMock()
        retriever.search = AsyncMock(
            return_value=RetrievalOutcome(
                chunks=[chunk],
                metadata=RetrievalMetadata(legal_chunks=1, total_selected=1),
            )
        )
        llm = AsyncMock()
        llm.generate = AsyncMock(
            return_value=SimpleNamespace(
                content="ضمانت کا طریقہ Section 497 PPC کے تحت ہے۔ [Source 1]",
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
            )
        )

        service = RAGService(
            llm=llm,
            retriever=retriever,
            query_rewriter=QueryRewriter(),
            prompt_builder=PromptBuilder(),
            response_formatter=ResponseFormatter(),
        )
        result = await service.execute(
            question=question,
            history=[],
            memories=[],
            task=Task.STATUTE_SEARCH,
            user_id="user-1",
        )

        search_query = retriever.search.await_args.kwargs["query"]
        assert "دفعہ 497" in search_query
        assert "section 497" in search_query.lower()
        assert result["retrieval_metadata"] is not None
        language = result["retrieval_metadata"].language
        assert language is not None
        assert language["user_language"] == "ur-PK"
        assert language["response_language"] == "ur-PK"
        assert language["retrieval_language"] == "original"
        assert "497" in (result["answer"] or "")

    @pytest.mark.asyncio
    async def test_mixed_query_does_not_break_retrieval(self):
        from types import SimpleNamespace
        from unittest.mock import AsyncMock

        from app.rag.models import RetrievalMetadata, RetrievalOutcome, RetrievedChunk, SourceType
        from app.rag.prompt_builder import PromptBuilder
        from app.rag.query_rewriter import QueryRewriter
        from app.rag.rag_service import RAGService
        from app.services.query_router import Task
        from app.services.response_formatter import ResponseFormatter

        question = "اس case میں bail کیسے مل سکتی ہے؟"
        chunk = RetrievedChunk(
            id="legal-2",
            chunk_id="legal-2",
            score=0.9,
            relevance_score=0.9,
            text="Bail may be granted in a pending case subject to law.",
            source_type=SourceType.LEGAL.value,
        )
        retriever = AsyncMock()
        retriever.search = AsyncMock(
            return_value=RetrievalOutcome(
                chunks=[chunk],
                metadata=RetrievalMetadata(legal_chunks=1, total_selected=1),
            )
        )
        llm = AsyncMock()
        llm.generate = AsyncMock(
            return_value=SimpleNamespace(
                content="اس case میں bail مل سکتی ہے اگر قانونی شرائط پوری ہوں۔ [Source 1]",
                prompt_tokens=8,
                completion_tokens=12,
                total_tokens=20,
            )
        )
        service = RAGService(
            llm=llm,
            retriever=retriever,
            query_rewriter=QueryRewriter(),
            prompt_builder=PromptBuilder(),
            response_formatter=ResponseFormatter(),
        )
        result = await service.execute(
            question=question,
            history=[],
            memories=[],
            task=Task.LEGAL_QA,
            user_id="user-1",
        )
        search_query = retriever.search.await_args.kwargs["query"]
        assert "bail" in search_query.lower()
        assert "اس" in search_query
        language = result["retrieval_metadata"].language
        assert language["language"] != "en-PK"
        assert language["retrieval_language"] == "original"


