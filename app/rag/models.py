from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class SourceType(str, Enum):
    """Retrieval tier / authority classification."""

    LEGAL = "legal"
    CONVERSATION = "conversation"
    MATTER = "matter"
    WEB = "web"


class MemoryScope(str, Enum):
    """
    Controls who can access memory.

    Security boundary.
    """

    USER = "user"

    MATTER = "matter"

    CONVERSATION = "conversation"

    DOCUMENT = "document"



# =====================================================
# Memory Source Type
# =====================================================

class MemoryType(str, Enum):
    """
    Where memory came from.
    """

    CHAT = "chat"

    DOCUMENT = "document"

    LEGAL_CONTEXT = "legal_context"

    PREFERENCE = "preference"



# =====================================================
# Conversation Message DTO
# =====================================================

class Message(BaseModel):
    """
    Conversation message DTO.

    Used by:

        PromptBuilder
        RAG context

    Stored permanently in PostgreSQL.
    """

    role: str

    content: str

    created_at: datetime | None = None



# =====================================================
# Long Term Memory DTO
# =====================================================

class Memory(BaseModel):
    """
    Long term vector memory.


    Storage:

        Qdrant user_memory


    Visibility scopes:

        user
            |
            +-- available in all user conversations


        matter
            |
            +-- available inside same matter


        conversation
            |
            +-- available only inside same conversation


        document
            |
            +-- extracted from uploaded document


    PostgreSQL:

        conversations
        messages
        documents


    Qdrant:

        embeddings
        searchable memory
    """

    id: str | None = None


    # Actual stored memory text

    text: str



    # Similarity score from Qdrant

    score: float = Field(
        default=0.0,
        ge=0.0,
    )



    # -----------------------------
    # Classification
    # -----------------------------

    scope: MemoryScope


    memory_type: MemoryType



    # -----------------------------
    # Security ownership
    # -----------------------------

    user_id: str



    # -----------------------------
    # Optional relations
    # -----------------------------

    matter_id: str | None = None


    conversation_id: str | None = None


    document_id: str | None = None



    created_at: datetime | None = None



# =====================================================
# Retrieved Document Chunk DTO
# =====================================================

class RetrievedChunk(BaseModel):
    """
    RAG retrieved document chunk.


    Sources:

        legal_documents

        chatbot_documents


    Used by:

        QdrantRetriever
        Reranker
        PromptBuilder
        ResponseFormatter
    """


    id: str


    score: float


    text: str



    # -----------------------------
    # Document metadata
    # -----------------------------

    filename: str | None = None


    title: str | None = None


    heading: str | None = None


    document_id: str | None = None



    # -----------------------------
    # Legal metadata
    # -----------------------------

    law_name: str | None = None


    document_type: str | None = None


    category: str | None = None


    sub_category: str | None = None


    practice_area: str | None = None



    court: str | None = None


    year: int | None = None


    jurisdiction: str | None = None


    sections: list[str] = Field(
        default_factory=list
    )


    keywords: list[str] = Field(
        default_factory=list
    )


    summary: str | None = None

    # -----------------------------
    # Source tracking
    # -----------------------------

    source: str | None = None

    source_type: str | None = None

    scope: str | None = None

    chunk_id: str | None = None

    chunk_index: int | None = None

    page_number: int | None = None

    start_offset: int | None = None

    end_offset: int | None = None

    paragraph_index: int | None = None

    section: str | None = None

    matter_id: str | None = None

    conversation_id: str | None = None

    source_reference: str | None = None

    relevance_score: float | None = None

    display_name: str | None = None

    author: str | None = None

    parent_chunk_id: str | None = None
    parent_text: str | None = None
    expanded_text: str | None = None
    validation_status: str | None = None
    publication_year: int | None = None
    chunk_role: str | None = None
    embedded: bool | None = None
    url: str | None = None


class RetrievalMetadata(BaseModel):
    """Structured retrieval diagnostics for observability and API responses."""

    legal_chunks: int = 0
    conversation_chunks: int = 0
    matter_chunks: int = 0
    web_chunks: int = 0
    total_selected: int = 0
    collections_queried: list[str] = Field(default_factory=list)
    degraded_sources: list[str] = Field(default_factory=list)
    filters_applied: dict[str, str] = Field(default_factory=dict)


class RetrievalOutcome(BaseModel):
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    metadata: RetrievalMetadata = Field(default_factory=RetrievalMetadata)