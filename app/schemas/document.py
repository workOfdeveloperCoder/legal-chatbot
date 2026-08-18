from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, computed_field

from app.rag.document_metadata import derive_display_name, derive_author_line


class DocumentCreate(BaseModel):
    """
    Used when uploading a document.

    Either:
    - matter_id -> shared with all conversations in matter
    - conversation_id -> private to one conversation
    - neither -> user memory document
    """

    matter_id: UUID | None = None

    conversation_id: UUID | None = None

    description: str | None = None



class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID

    filename: str

    mime_type: str | None

    storage_path: str

    scope: str

    matter_id: UUID | None

    conversation_id: UUID | None

    processed: bool

    vectorized: bool = False

    created_at: datetime

    @computed_field
    @property
    def ready_for_qa(self) -> bool:
        """True only when extraction and vector indexing both succeeded."""
        return self.processed and self.vectorized


TEXT_PREVIEW_LIMIT = 500


class MatterDocumentItem(BaseModel):
    """
    Summary row for the matter document library (list view).

    Full merged text is available from the document detail endpoint.
    """

    id: UUID

    filename: str

    type: Literal["document"] = "document"

    scope: str

    matter_id: UUID

    mime_type: str | None = None

    text_preview: str | None = None

    character_count: int = 0

    processed: bool

    vectorized: bool = False

    created_at: datetime

    updated_at: datetime

    @computed_field
    @property
    def ready_for_qa(self) -> bool:
        return self.processed and self.vectorized


class MatterDocumentDetail(BaseModel):
    """Full matter document payload with merged text content."""

    id: UUID

    filename: str

    type: Literal["document"] = "document"

    scope: str

    matter_id: UUID | None = None

    conversation_id: UUID | None = None

    mime_type: str | None = None

    text: str | None = None

    text_preview: str | None = None

    character_count: int = 0

    chunk_count: int | None = None

    processed: bool

    vectorized: bool = False

    created_at: datetime

    updated_at: datetime

    @computed_field
    @property
    def ready_for_qa(self) -> bool:
        return self.processed and self.vectorized

    @computed_field
    @property
    def display_name(self) -> str:
        return derive_display_name(self.text, self.filename)

    @computed_field
    @property
    def author(self) -> str | None:
        return derive_author_line(self.text)


class MatterDocumentListResponse(BaseModel):
    matter_id: UUID

    items: list[MatterDocumentItem]

    total: int