from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, computed_field

from app.rag.document_metadata import derive_author_line, derive_display_name


class LibraryDocumentDetail(BaseModel):
    """Full legal-corpus document rebuilt from Qdrant chunks."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    type: str = "document"
    scope: str = "library"
    source_type: str = "legal"
    title: str | None = None
    law_name: str | None = None
    document_type: str | None = None
    year: int | None = None
    mime_type: str | None = "text/plain"
    text: str | None = None
    text_preview: str | None = None
    character_count: int = 0
    chunk_count: int | None = None
    processed: bool = True
    vectorized: bool = True
    created_at: datetime
    updated_at: datetime

    @computed_field
    @property
    def ready_for_qa(self) -> bool:
        return bool(self.text)

    @computed_field
    @property
    def display_name(self) -> str:
        return derive_display_name(self.text, self.title or self.filename)

    @computed_field
    @property
    def author(self) -> str | None:
        return derive_author_line(self.text)
