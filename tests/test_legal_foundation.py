"""Foundation tests: LegalQueryPlanner, EvidenceEngine, authority."""

from __future__ import annotations

import pytest

from app.rag.authority import AuthorityType, classify_authority
from app.rag.evidence_engine import EvidenceEngine, EvidenceRole
from app.rag.legal_query_planner import (
    AnswerMode,
    LegalQueryPlanner,
    RetrievalStrategy,
)
from app.rag.models import RetrievedChunk, SourceType
from app.services.query_router import Task


class TestLegalQueryPlanner:
    @pytest.mark.asyncio
    async def test_document_question_plan(self):
        planner = LegalQueryPlanner()
        plan = await planner.plan(
            question="What is the problem identified with section 54-C?",
            has_uploaded_documents=True,
        )
        assert plan.task == Task.DOCUMENT_QA
        assert plan.answer_mode == AnswerMode.DOCUMENT_QA
        assert plan.retrieval_strategy == RetrievalStrategy.PRIVATE_DOCS_FIRST
        assert plan.uploaded_document_primary is True
        assert plan.requires_legal_authority is False
        assert "54-c" in [s.lower() for s in plan.sections] or plan.sections

    @pytest.mark.asyncio
    async def test_mixed_legal_correctness_plan(self):
        planner = LegalQueryPlanner()
        plan = await planner.plan(
            question="Is the author's argument about section 54-C legally correct?",
            has_uploaded_documents=True,
        )
        assert plan.task == Task.MIXED_QA
        assert plan.answer_mode == AnswerMode.MIXED_LEGAL_ANALYSIS
        assert plan.retrieval_strategy == RetrievalStrategy.MIXED_DOC_AND_LAW
        assert plan.requires_legal_authority is True
        assert plan.uploaded_document_primary is False

    @pytest.mark.asyncio
    async def test_document_scoped_plan(self):
        planner = LegalQueryPlanner()
        plan = await planner.plan(
            question="Summarize this document.",
            has_uploaded_documents=True,
            document_id="11111111-1111-1111-1111-111111111111",
        )
        assert plan.document_scoped is True
        assert plan.retrieval_strategy == RetrievalStrategy.DOCUMENT_SCOPED

    @pytest.mark.asyncio
    async def test_legal_research_plan(self):
        planner = LegalQueryPlanner()
        plan = await planner.plan(
            question="What is the limitation period under the Limitation Act?",
            has_uploaded_documents=False,
        )
        assert plan.answer_mode == AnswerMode.LEGAL_RESEARCH
        assert plan.retrieval_strategy == RetrievalStrategy.LEGAL_FIRST
        assert plan.requires_legal_authority is True


class TestAuthorityClassification:
    def test_matter_article_classified_as_commentary(self):
        authority = classify_authority(
            source_type="matter",
            filename="2000J8.txt",
            display_name="CLOG ON DISCRETION",
        )
        assert authority in {
            AuthorityType.MATTER_DOCUMENT,
            AuthorityType.ARTICLE_COMMENTARY,
        }

    def test_statute_classified(self):
        authority = classify_authority(
            source_type="legal",
            document_type="statute",
            law_name="Electricity Act, 1910",
        )
        assert authority == AuthorityType.STATUTE

    def test_supreme_court_classified(self):
        authority = classify_authority(
            source_type="legal",
            document_type="judgment",
            court="Supreme Court",
        )
        assert authority == AuthorityType.SUPREME_COURT


class TestEvidenceEngine:
    def test_builds_roles_and_metadata(self):
        chunks = [
            RetrievedChunk(
                id="1",
                chunk_id="1",
                score=0.9,
                relevance_score=0.9,
                text=(
                    "Section 54-C of the Electricity Act can operate as a clog "
                    "on judicial discretion and may affect injunction relief."
                ),
                source_type=SourceType.MATTER.value,
                filename="2000J8.txt",
                display_name="CLOG ON DISCRETION",
                document_id="11111111-1111-1111-1111-111111111111",
            )
        ]
        bundle = EvidenceEngine().build(
            chunks,
            question="What is the problem identified with section 54-C?",
            document_qa_mode=True,
            requires_legal_authority=False,
        )
        assert len(bundle.items) == 1
        assert bundle.items[0].authority_type in {
            AuthorityType.MATTER_DOCUMENT,
            AuthorityType.ARTICLE_COMMENTARY,
        }
        assert bundle.items[0].role in {
            EvidenceRole.SUPPORTING,
            EvidenceRole.RELEVANT,
            EvidenceRole.WEAK,
        }
        meta = bundle.to_metadata()
        assert meta["evidence_count"] == 1
        assert "overall_strength" in meta
