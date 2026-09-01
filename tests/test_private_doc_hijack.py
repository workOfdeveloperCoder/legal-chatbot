from __future__ import annotations

from app.rag.models import RetrievedChunk, SourceType
from app.rag.rag_service import RAGService


def test_question_targets_uploaded_docs():
    assert RAGService._question_targets_uploaded_docs(
        "Summarize the uploaded document"
    )
    assert RAGService._question_targets_uploaded_docs(
        "What does the author argue about void orders?"
    )
    assert not RAGService._question_targets_uploaded_docs(
        "what is sections and articles in law"
    )


def test_chunk_matches_question_rejects_unrelated_upload():
    void_chunk = RetrievedChunk(
        id="c1",
        score=0.9,
        relevance_score=0.9,
        text="VOID ORDERS by Ch. Manzoor Hussain discusses nullity of orders.",
        filename="2000J8.txt",
        title="CLOG ON DISCRETION",
        source_type=SourceType.CONVERSATION.value,
    )
    assert not RAGService._chunk_matches_question(
        void_chunk,
        "what is sections and articles in law",
    )

    void_justice_only = RetrievedChunk(
        id="c3",
        score=0.58,
        relevance_score=0.58,
        text=(
            "The Supreme Court held that orders passed without hearing "
            "must be set aside for proper administration of justice."
        ),
        filename="2002J15.txt",
        title="Void Orders",
        source_type=SourceType.CONVERSATION.value,
    )
    assert not RAGService._chunk_matches_question(
        void_justice_only,
        "What does Justice A. R. Cornelius identify as the primary task "
        "of a newly established independent government?",
    )

    cornelius_corpus = RetrievedChunk(
        id="c4",
        score=0.69,
        relevance_score=0.69,
        text=(
            "Address by Mr. Justice A. R. Cornelius, Chief Justice of Pakistan. "
            "A primary task for a newly-established independent national "
            "government is repairing damage done to national character."
        ),
        filename="1963J1.txt",
        source_type=SourceType.LEGAL.value,
    )
    assert RAGService._chunk_matches_question(
        cornelius_corpus,
        "What does Justice A. R. Cornelius identify as the primary task "
        "of a newly established independent government?",
    )

    related = RetrievedChunk(
        id="c2",
        score=0.4,
        relevance_score=0.4,
        text="Statutes are divided into sections and articles for citation.",
        filename="statute-structure.txt",
        source_type=SourceType.CONVERSATION.value,
    )
    assert RAGService._chunk_matches_question(
        related,
        "what is sections and articles in law",
    )
