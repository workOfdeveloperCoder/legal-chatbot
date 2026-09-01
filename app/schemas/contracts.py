from __future__ import annotations

from enum import Enum
from uuid import UUID

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class ClauseStatus(str, Enum):
    FOUND = "found"
    MISSING = "missing"
    UNCLEAR = "unclear"
    ERROR = "error"


class ClauseCard(BaseModel):
    """One extracted term. Playbook rules later match on `key`."""

    model_config = ConfigDict(populate_by_name=True)

    key: str
    label: str
    status: ClauseStatus
    value: str | None = None
    quote: str | None = None
    quote_grounded: bool = Field(
        default=False,
        serialization_alias="quoteGrounded",
        validation_alias=AliasChoices("quote_grounded", "quoteGrounded"),
    )
    confidence: float = 0.0
    locked: bool = False


class ClauseFieldSpec(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    key: str
    label: str
    question: str


class ClauseExtractionResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    document_id: UUID = Field(
        serialization_alias="documentId",
        validation_alias=AliasChoices("document_id", "documentId"),
    )
    filename: str
    truncated: bool = False
    cards: list[ClauseCard] = Field(default_factory=list)


class ReviewCell(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    key: str
    status: ClauseStatus
    value: str | None = None
    quote: str | None = None
    quote_grounded: bool = Field(
        default=False,
        serialization_alias="quoteGrounded",
        validation_alias=AliasChoices("quote_grounded", "quoteGrounded"),
    )
    confidence: float = 0.0
    locked: bool = False


class ReviewRow(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    document_id: UUID = Field(
        serialization_alias="documentId",
        validation_alias=AliasChoices("document_id", "documentId"),
    )
    filename: str
    truncated: bool = False
    error: str | None = None
    cells: list[ReviewCell] = Field(default_factory=list)


class ReviewTableResponse(BaseModel):
    """Matter (or conversation) grid: one row per file, one column per field."""

    model_config = ConfigDict(populate_by_name=True)

    matter_id: UUID | None = Field(
        default=None,
        serialization_alias="matterId",
        validation_alias=AliasChoices("matter_id", "matterId"),
    )
    conversation_id: UUID | None = Field(
        default=None,
        serialization_alias="conversationId",
        validation_alias=AliasChoices("conversation_id", "conversationId"),
    )
    columns: list[ClauseFieldSpec] = Field(default_factory=list)
    rows: list[ReviewRow] = Field(default_factory=list)
    document_count: int = Field(
        default=0,
        serialization_alias="documentCount",
        validation_alias=AliasChoices("document_count", "documentCount"),
    )
    truncated_documents: bool = Field(
        default=False,
        serialization_alias="truncatedDocuments",
        validation_alias=AliasChoices("truncated_documents", "truncatedDocuments"),
    )


class ReviewTableRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    field_keys: list[str] | None = Field(
        default=None,
        validation_alias=AliasChoices("field_keys", "fieldKeys"),
        description="Subset of default keys. Omit to use the full catalog.",
    )
    extra_questions: list[ClauseFieldSpec] | None = Field(
        default=None,
        validation_alias=AliasChoices("extra_questions", "extraQuestions"),
        description="Custom columns, e.g. a playbook question.",
    )
    document_ids: list[UUID] | None = Field(
        default=None,
        validation_alias=AliasChoices("document_ids", "documentIds"),
        description="Limit to these files. Omit for every matter document.",
    )


class ClauseFieldsResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    fields: list[ClauseFieldSpec]


class PlaybookSeverity(str, Enum):
    BLOCKER = "blocker"
    WARNING = "warning"
    INFO = "info"


class PlaybookVerdict(str, Enum):
    PASS = "pass"
    FALLBACK = "fallback"
    MISSING = "missing"
    OFF_PLAYBOOK = "off_playbook"
    UNCLEAR = "unclear"
    NOT_APPLICABLE = "not_applicable"


class PlaybookRuleSpec(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    key: str
    required: bool = True
    accept_if_any: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("accept_if_any", "acceptIfAny"),
        serialization_alias="acceptIfAny",
    )
    accept_if_fallback: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("accept_if_fallback", "acceptIfFallback"),
        serialization_alias="acceptIfFallback",
    )
    reject_if_any: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("reject_if_any", "rejectIfAny"),
        serialization_alias="rejectIfAny",
    )
    preferred_position: str = Field(
        default="",
        validation_alias=AliasChoices("preferred_position", "preferredPosition"),
        serialization_alias="preferredPosition",
    )
    fallback_position: str = Field(
        default="",
        validation_alias=AliasChoices("fallback_position", "fallbackPosition"),
        serialization_alias="fallbackPosition",
    )
    suggested_language: str = Field(
        default="",
        validation_alias=AliasChoices("suggested_language", "suggestedLanguage"),
        serialization_alias="suggestedLanguage",
    )
    severity: PlaybookSeverity = PlaybookSeverity.WARNING
    notes: str = ""


class PlaybookSummary(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    pass_count: int = Field(
        default=0,
        serialization_alias="passCount",
        validation_alias=AliasChoices("pass_count", "passCount"),
    )
    fallback_count: int = Field(
        default=0,
        serialization_alias="fallbackCount",
        validation_alias=AliasChoices("fallback_count", "fallbackCount"),
    )
    missing_count: int = Field(
        default=0,
        serialization_alias="missingCount",
        validation_alias=AliasChoices("missing_count", "missingCount"),
    )
    off_playbook_count: int = Field(
        default=0,
        serialization_alias="offPlaybookCount",
        validation_alias=AliasChoices("off_playbook_count", "offPlaybookCount"),
    )
    unclear_count: int = Field(
        default=0,
        serialization_alias="unclearCount",
        validation_alias=AliasChoices("unclear_count", "unclearCount"),
    )
    blocker_count: int = Field(
        default=0,
        serialization_alias="blockerCount",
        validation_alias=AliasChoices("blocker_count", "blockerCount"),
    )


class PlaybookFindingResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    key: str
    label: str
    verdict: PlaybookVerdict
    severity: PlaybookSeverity
    extracted_value: str | None = Field(
        default=None,
        serialization_alias="extractedValue",
        validation_alias=AliasChoices("extracted_value", "extractedValue"),
    )
    quote: str | None = None
    preferred_position: str = Field(
        default="",
        serialization_alias="preferredPosition",
        validation_alias=AliasChoices("preferred_position", "preferredPosition"),
    )
    fallback_position: str = Field(
        default="",
        serialization_alias="fallbackPosition",
        validation_alias=AliasChoices("fallback_position", "fallbackPosition"),
    )
    suggested_language: str = Field(
        default="",
        serialization_alias="suggestedLanguage",
        validation_alias=AliasChoices("suggested_language", "suggestedLanguage"),
    )
    reason: str
    required: bool = True


class PlaybookCatalogItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    name: str
    description: str
    rules: list[PlaybookRuleSpec] = Field(default_factory=list)


class PlaybookReviewRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    playbook_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("playbook_id", "playbookId"),
    )
    rules: list[PlaybookRuleSpec] | None = None


class PlaybookReviewResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    playbook_id: str = Field(
        serialization_alias="playbookId",
        validation_alias=AliasChoices("playbook_id", "playbookId"),
    )
    playbook_name: str = Field(
        serialization_alias="playbookName",
        validation_alias=AliasChoices("playbook_name", "playbookName"),
    )
    document_id: UUID = Field(
        serialization_alias="documentId",
        validation_alias=AliasChoices("document_id", "documentId"),
    )
    filename: str
    findings: list[PlaybookFindingResponse] = Field(default_factory=list)
    summary: PlaybookSummary = Field(default_factory=PlaybookSummary)
    extraction: ClauseExtractionResponse | None = None


class RedlineSpanResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    op: str
    text: str


class RedlineHunkResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    key: str | None = None
    label: str
    left: str = ""
    right: str = ""
    spans: list[RedlineSpanResponse] = Field(default_factory=list)
    markdown: str = ""
    html: str = ""


class RedlineRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    against_document_id: UUID | None = Field(
        default=None,
        validation_alias=AliasChoices("against_document_id", "againstDocumentId"),
    )
    mode: str = "documents"
    playbook_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("playbook_id", "playbookId"),
    )
    rules: list[PlaybookRuleSpec] | None = None


class RedlineResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    mode: str
    left_document_id: UUID | None = Field(
        default=None,
        serialization_alias="leftDocumentId",
        validation_alias=AliasChoices("left_document_id", "leftDocumentId"),
    )
    right_document_id: UUID | None = Field(
        default=None,
        serialization_alias="rightDocumentId",
        validation_alias=AliasChoices("right_document_id", "rightDocumentId"),
    )
    left_filename: str = Field(
        default="",
        serialization_alias="leftFilename",
        validation_alias=AliasChoices("left_filename", "leftFilename"),
    )
    right_filename: str = Field(
        default="",
        serialization_alias="rightFilename",
        validation_alias=AliasChoices("right_filename", "rightFilename"),
    )
    hunks: list[RedlineHunkResponse] = Field(default_factory=list)
    unchanged: bool = False


class ExportKind(str, Enum):
    CLAUSES = "clauses"
    PLAYBOOK = "playbook"
    REDLINE = "redline"
    REVIEW_TABLE = "review_table"
    DRAFT = "draft"


class ExportRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    kind: ExportKind
    document_id: UUID | None = Field(
        default=None,
        validation_alias=AliasChoices("document_id", "documentId"),
    )
    against_document_id: UUID | None = Field(
        default=None,
        validation_alias=AliasChoices("against_document_id", "againstDocumentId"),
    )
    matter_id: UUID | None = Field(
        default=None,
        validation_alias=AliasChoices("matter_id", "matterId"),
    )
    conversation_id: UUID | None = Field(
        default=None,
        validation_alias=AliasChoices("conversation_id", "conversationId"),
    )
    playbook_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("playbook_id", "playbookId"),
    )
    title: str | None = None
    body: str | None = None

