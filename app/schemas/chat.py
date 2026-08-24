from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field

from app.schemas.llm import TokenUsageResponse


class ChatRequest(BaseModel):
    matter_id: UUID | None = None
    conversation_id: UUID | None = None
    document_id: UUID | None = None

    message: str = Field(
        min_length=1,
        max_length=20000,
    )

    regenerate: bool = False


class Citation(BaseModel):
    """
    Readable legal resource for the frontend.

    Built from Qdrant / legal-gpt chunks retrieved by RAG.
    """

    id: str | None = None
    document_id: str | None = None

    # Human-readable excerpt from the retrieved chunk
    excerpt: str | None = None

    filename: str | None = None
    title: str | None = None
    heading: str | None = None

    law_name: str | None = None
    document_type: str | None = None

    category: str | None = None
    sub_category: str | None = None
    practice_area: str | None = None

    court: str | None = None
    year: int | None = None
    jurisdiction: str | None = None

    sections: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)

    summary: str | None = None

    score: float = 0.0

    source_type: str | None = None
    chunk_id: str | None = None
    page: int | None = None
    source_reference: str | None = None
    matter_id: str | None = None
    conversation_id: str | None = None
    display_name: str | None = None


class SourceReference(BaseModel):
    """Chunk-level evidence used for citation validation and highlighting."""

    id: str
    source_number: int
    source_type: str
    document_id: str | None = None
    document_name: str | None = None
    display_name: str | None = None
    filename: str | None = None
    author: str | None = None
    chunk_id: str | None = None
    chunk_index: int | None = None
    page: int | None = None
    start_offset: int | None = None
    end_offset: int | None = None
    paragraph_index: int | None = None
    section: str | None = None
    excerpt: str | None = None
    text: str | None = None
    source_reference: str | None = None
    relevance: float = 0.0
    relevance_percent: int | None = None
    matter_id: str | None = None
    conversation_id: str | None = None
    law_name: str | None = None
    court: str | None = None
    year: int | None = None
    sections: list[str] = Field(default_factory=list)


class EvidenceHighlight(BaseModel):
    """One retrieved chunk location attached to a document resource."""

    source_id: str
    source_number: int
    chunk_id: str | None = None
    chunk_index: int | None = None
    excerpt: str | None = None
    text: str | None = None
    start_offset: int | None = None
    end_offset: int | None = None
    page: int | None = None
    relevance: float = 0.0
    relevance_percent: int | None = None


class ResourceReference(BaseModel):
    """
    Unique document (or legal authority entry) referenced in a reply.

    Multiple chunk-level sources from the same document_id collapse into
    one resource with multiple evidence highlights.
    """

    id: str
    document_id: str | None = None
    display_name: str | None = None
    filename: str | None = None
    author: str | None = None
    source_type: str
    matter_id: str | None = None
    conversation_id: str | None = None
    law_name: str | None = None
    court: str | None = None
    year: int | None = None
    source_ids: list[str] = Field(default_factory=list)
    source_numbers: list[int] = Field(default_factory=list)
    evidence: list[EvidenceHighlight] = Field(default_factory=list)
    primary_excerpt: str | None = None
    relevance: float = 0.0
    relevance_percent: int | None = None


class RetrievalMetadataResponse(BaseModel):
    legal_chunks: int = 0
    conversation_chunks: int = 0
    matter_chunks: int = 0
    total_selected: int = 0
    collections_queried: list[str] = Field(default_factory=list)
    degraded_sources: list[str] = Field(default_factory=list)
    filters_applied: dict[str, str] = Field(default_factory=dict)
    evidence_strength: str | None = None
    legal_sources_used: int = 0
    conversation_sources_used: int = 0
    matter_sources_used: int = 0
    citation_validation_status: str | None = None
    used_conversation_context: bool = False
    query_plan: dict[str, object] | None = None
    evidence_metadata: dict[str, object] | None = None
    token_budget: dict[str, object] | None = None
    pipeline_versions: dict[str, str] | None = None
    performance: dict[str, object] | None = None
    llm_execution: dict[str, object] | None = None
    conversation_context: dict[str, object] | None = None
    language: dict[str, object] | None = None


class ChatResponse(BaseModel):
    conversation_id: UUID

    response: str

    citations: list[Citation] = []

    sources: list[SourceReference] = Field(default_factory=list)

    resources: list[ResourceReference] = Field(default_factory=list)

    retrieval_metadata: RetrievalMetadataResponse | None = None

    grounding_status: str | None = None

    token_usage: TokenUsageResponse | None = None
