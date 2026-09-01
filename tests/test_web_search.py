from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.rag.authority import AuthorityType, classify_authority
from app.rag.models import RetrievedChunk, SourceType
from app.rag.prompt_builder import PromptBuilder
from app.search.web_search import WebSearchHit, WebSearchService
from app.services.response_formatter import ResponseFormatter


def test_official_url_detection():
    searcher = WebSearchService()
    assert searcher.is_official_url("https://pakistancode.gov.pk/english/LHtm.php")
    assert searcher.is_official_url("https://www.supremecourt.gov.pk/judgments/")
    assert searcher.is_official_url("https://lhc.gov.pk/casedetail") is True
    assert searcher.is_official_url("https://example.com/bail-guide") is False


def test_web_hit_becomes_web_chunk_not_legal():
    searcher = WebSearchService()
    hit = WebSearchHit(
        title="Pakistan Code — Section 497",
        url="https://pakistancode.gov.pk/english/497",
        snippet="When any person accused of a non-bailable offence is arrested...",
        score=0.9,
        official=True,
    )
    chunk = searcher.to_chunk(hit, 1)
    assert chunk.source_type == SourceType.WEB.value
    assert chunk.url == hit.url
    assert chunk.document_type == "official_web"
    authority = classify_authority(
        source_type=chunk.source_type,
        document_type=chunk.document_type,
        filename=chunk.filename,
        display_name=chunk.display_name,
    )
    assert authority == AuthorityType.WEB_OFFICIAL


def test_unofficial_web_is_low_authority():
    authority = classify_authority(
        source_type=SourceType.WEB.value,
        document_type="web",
        filename="Blog post about bail",
        display_name="Blog post about bail",
    )
    assert authority == AuthorityType.WEB_SEARCH


def test_prompt_groups_web_separately_from_legal():
    builder = PromptBuilder()
    legal = RetrievedChunk(
        id="legal-1",
        score=0.9,
        text="Section 497 CrPC governs bail in non-bailable offences.",
        filename="CrPC",
        source_type=SourceType.LEGAL.value,
    )
    web = RetrievedChunk(
        id="web-1",
        score=0.8,
        text="A news site summarizes a recent bail order.",
        filename="News",
        source_type=SourceType.WEB.value,
        url="https://example.com/bail",
        source_reference="https://example.com/bail",
    )
    prompt = builder.build(
        question="What is the law on bail?",
        history=[],
        chunks=[legal, web],
        memories=[],
        task="case_search",
        include_system_in_user=False,
    )
    assert "INTERNET SEARCH" in prompt
    assert "URL: https://example.com/bail" in prompt
    assert "not binding" in prompt.lower()


def test_formatter_exposes_web_url_and_counts():
    from app.rag.models import RetrievalMetadata

    chunk = RetrievedChunk(
        id="web-1",
        score=0.8,
        text="Snippet",
        filename="Pakistan Code",
        source_type=SourceType.WEB.value,
        url="https://pakistancode.gov.pk/x",
        source_reference="https://pakistancode.gov.pk/x",
        document_id="https://pakistancode.gov.pk/x",
    )
    result = ResponseFormatter().format(
        answer="Answer [Source 1]",
        chunks=[chunk],
        retrieval_metadata=RetrievalMetadata(web_chunks=1, total_selected=1),
        sources_used=[1],
        build_resources=True,
        include_web=True,
    )
    assert result["citations"][0].url == "https://pakistancode.gov.pk/x"
    assert result["sources"][0].url == "https://pakistancode.gov.pk/x"
    assert result["retrieval_metadata"].web_chunks == 1
    assert result["retrieval_metadata"].web_sources_used == 1


@pytest.mark.asyncio
async def test_tavily_search_parses_results(monkeypatch):
    payload = {
        "results": [
            {
                "title": "CrPC 497",
                "url": "https://pakistancode.gov.pk/497",
                "content": "Bail in non-bailable offences.",
                "score": 0.88,
            }
        ]
    }
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json = MagicMock(return_value=payload)

    class FakeClient:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, *args, **kwargs):
            del args, kwargs
            return response

    monkeypatch.setattr("app.search.web_search.httpx.AsyncClient", FakeClient)
    from app.core import config

    monkeypatch.setattr(config.settings, "WEB_SEARCH_ENABLED", True)
    monkeypatch.setattr(config.settings, "WEB_SEARCH_PROVIDER", "tavily")
    monkeypatch.setattr(config.settings, "WEB_SEARCH_API_KEY", "tvly-test")
    monkeypatch.setattr(config.settings, "TAVILY_API_KEY", "tvly-test")

    hits = await WebSearchService().search("section 497 CrPC")
    assert len(hits) == 1
    assert hits[0].official is True
    assert "Bail" in hits[0].snippet


@pytest.mark.asyncio
async def test_disabled_search_returns_empty(monkeypatch):
    from app.core import config

    monkeypatch.setattr(config.settings, "WEB_SEARCH_ENABLED", False)
    hits = await WebSearchService().search("anything")
    assert hits == []


def test_duckduckgo_url_unwrap():
    wrapped = (
        "https://duckduckgo.com/l/?uddg="
        "https%3A%2F%2Fpakistancode.gov.pk%2Fenglish%2F497"
    )
    url = WebSearchService._unwrap_duckduckgo_url(wrapped)
    assert url == "https://pakistancode.gov.pk/english/497"


@pytest.mark.asyncio
async def test_duckduckgo_parses_html_results(monkeypatch):
    html = """
    <html><body>
    <a class="result__a" href="https://duckduckgo.com/l/?uddg=https%3A%2F%2Fpakistancode.gov.pk%2F497">
      CrPC 497
    </a>
    <a class="result__snippet">Bail in non-bailable offences.</a>
    </body></html>
    """
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.status_code = 200
    response.text = html

    class FakeClient:
        def __init__(self, *args, **kwargs):
            # Browser-like UA required; custom LegalGPT UA is bot-challenged.
            headers = kwargs.get("headers") or {}
            assert "Chrome" in headers.get("User-Agent", "")
            assert kwargs.get("trust_env") is False
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, *args, **kwargs):
            del args, kwargs
            return response

    monkeypatch.setattr("app.search.web_search.httpx.AsyncClient", FakeClient)
    from app.core import config

    monkeypatch.setattr(config.settings, "WEB_SEARCH_ENABLED", True)
    monkeypatch.setattr(config.settings, "WEB_SEARCH_PROVIDER", "duckduckgo")
    monkeypatch.setattr(config.settings, "WEB_SEARCH_API_KEY", None)
    monkeypatch.setattr(config.settings, "TAVILY_API_KEY", None)
    monkeypatch.setattr(config.settings, "BRAVE_API_KEY", None)

    hits = await WebSearchService().search("section 497 CrPC")
    assert len(hits) == 1
    assert hits[0].url == "https://pakistancode.gov.pk/497"
    assert hits[0].official is True


def test_prefer_web_treats_web_hits_as_partial_evidence():
    from app.rag.answer_guard import EvidenceStrength
    from app.rag.evidence_assessment import assess_evidence_for_question

    web = RetrievedChunk(
        id="web-1",
        score=0.55,
        relevance_score=0.55,
        text="Section 497 CrPC: bail in non-bailable offences under Pakistan Code.",
        filename="Pakistan Code",
        source_type=SourceType.WEB.value,
        url="https://pakistancode.gov.pk/497",
        source_reference="https://pakistancode.gov.pk/497",
    )
    assessment = assess_evidence_for_question(
        [web],
        "What is section 497 CrPC?",
        requires_legal_authority=True,
        prefer_web=True,
    )
    assert assessment.overall_strength == EvidenceStrength.PARTIAL


def test_prompt_internet_search_mode_when_prefer_web():
    builder = PromptBuilder()
    web = RetrievedChunk(
        id="web-1",
        score=0.8,
        text="Official page summarizes section 497 CrPC bail rules.",
        filename="Pakistan Code",
        source_type=SourceType.WEB.value,
        url="https://pakistancode.gov.pk/497",
        source_reference="https://pakistancode.gov.pk/497",
    )
    prompt = builder.build(
        question="What is section 497 CrPC?",
        history=[],
        chunks=[web],
        memories=[],
        task="case_search",
        include_system_in_user=False,
        prefer_web=True,
    )
    assert "INTERNET SEARCH MODE" in prompt
    assert "INTERNET SEARCH" in prompt
    assert "primary research material" in prompt


def test_web_toggle_off_does_not_search_when_corpus_empty(monkeypatch):
    from app.core import config
    from app.rag.rag_service import RAGService

    monkeypatch.setattr(config.settings, "WEB_SEARCH_ENABLED", True)
    rag = RAGService.__new__(RAGService)
    assert (
        rag._should_run_web_search(
            web_search=False,
            document_task=False,
        )
        is False
    )
    assert (
        rag._should_run_web_search(
            web_search=True,
            document_task=False,
        )
        is True
    )


@pytest.mark.asyncio
async def test_case_query_does_not_imply_web_search():
    from app.rag.legal_query_planner import LegalQueryPlanner

    plan = await LegalQueryPlanner().plan(
        question="give me case related to section 417",
        web_search=False,
    )
    assert plan.web_search is False


def test_resources_exclude_web_when_toggle_off():
    from app.rag.models import RetrievalMetadata

    legal = RetrievedChunk(
        id="legal-1",
        score=0.9,
        text="Section 417 PPC cheating provision.",
        filename="Pakistan Penal Code",
        source_type=SourceType.LEGAL.value,
        law_name="Pakistan Penal Code",
    )
    web = RetrievedChunk(
        id="web-1",
        score=0.8,
        text="Unrelated web snippet.",
        filename="Some site",
        source_type=SourceType.WEB.value,
        url="https://example.com/417",
    )
    off = ResponseFormatter().format(
        answer="Answer",
        chunks=[legal, web],
        retrieval_metadata=RetrievalMetadata(
            legal_chunks=1,
            web_chunks=1,
            total_selected=2,
        ),
        build_resources=True,
        include_web=False,
    )
    assert len(off["resources"]) == 1
    assert off["resources"][0].source_type == SourceType.LEGAL.value
    assert off["retrieval_metadata"].web_chunks == 0

    many_legal = [
        RetrievedChunk(
            id=f"legal-{i}",
            score=0.9 - i * 0.01,
            text=f"Legal chunk {i} about section 417.",
            filename=f"case-{i}.txt",
            source_type=SourceType.LEGAL.value,
        )
        for i in range(6)
    ]
    qdrant_only = ResponseFormatter().format(
        answer="Answer",
        chunks=many_legal,
        build_resources=True,
        include_web=False,
    )
    assert len(qdrant_only["resources"]) == 6

    on = ResponseFormatter().format(
        answer="Answer",
        chunks=[legal, web],
        retrieval_metadata=RetrievalMetadata(
            legal_chunks=1,
            web_chunks=1,
            total_selected=2,
        ),
        build_resources=True,
        include_web=True,
    )
    assert len(on["resources"]) == 2
    types = {item.source_type for item in on["resources"]}
    assert SourceType.WEB.value in types
    assert SourceType.LEGAL.value in types


def test_resources_show_retrieved_corpus_even_when_model_cites_wrong_sources():
    from app.rag.models import RetrievalMetadata

    chunks = [
        RetrievedChunk(
            id="legal-cornelius",
            score=0.9,
            text="PRIMARY TASK FOR NEWLY-ESTABLISHED INDEPENDENT GOVERNMENT Cornelius",
            filename="1963J1.txt",
            source_type=SourceType.LEGAL.value,
            title="Restoration of Judicial Responsibility",
        ),
        RetrievedChunk(
            id="conv-void",
            score=0.8,
            text="VOID ORDERS unrelated article",
            filename="2002J15.txt",
            source_type=SourceType.CONVERSATION.value,
            title="Void Orders",
        ),
        RetrievedChunk(
            id="conv-clog",
            score=0.79,
            text="CLOG ON DISCRETION unrelated article",
            filename="2000J8.txt",
            source_type=SourceType.CONVERSATION.value,
            title="CLOG ON DISCRETION",
        ),
    ]
    result = ResponseFormatter().format(
        answer="Answer",
        chunks=chunks,
        retrieval_metadata=RetrievalMetadata(legal_chunks=1, total_selected=3),
        build_resources=True,
        include_web=False,
        sources_used=[2, 3],
    )
    resource_names = {item.filename for item in result["resources"]}
    assert resource_names == {"1963J1.txt"}
    cited_names = {item.filename for item in result["sources"]}
    assert "1963J1.txt" not in cited_names
    assert cited_names == {"2002J15.txt", "2000J8.txt"}
