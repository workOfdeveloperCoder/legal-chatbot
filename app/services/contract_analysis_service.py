from __future__ import annotations

import asyncio
import logging
from uuid import UUID

from fastapi import HTTPException, status

from app.contracts.extractor import ClauseExtractor
from app.contracts.fields import (
    EXTRACT_CONCURRENCY,
    MAX_REVIEW_DOCUMENTS,
    ClauseField,
    resolve_fields,
)
from app.contracts.format import (
    format_clause_cards,
    format_playbook_review,
    format_redline,
    format_review_table,
)
from app.contracts.playbook import (
    Playbook,
    PlaybookFinding,
    PlaybookRule,
    PlaybookSeverity as EngineSeverity,
    PlaybookVerdict as EngineVerdict,
    evaluate_playbook,
    get_playbook,
    playbook_from_rules,
)
from app.contracts.redline import (
    document_hunks,
    playbook_hunks,
    spans_to_html,
    spans_to_markdown,
)
from app.contracts import docx_export
from app.llm.base import BaseLLM
from app.models.document import Document
from app.repositories.document_repository import DocumentRepository
from app.repositories.matter_repository import MatterRepository
from app.schemas.contracts import (
    ClauseExtractionResponse,
    ClauseFieldSpec,
    ClauseStatus,
    ExportKind,
    PlaybookCatalogItem,
    PlaybookFindingResponse,
    PlaybookReviewResponse,
    PlaybookRuleSpec,
    PlaybookSeverity,
    PlaybookSummary,
    PlaybookVerdict,
    RedlineHunkResponse,
    RedlineResponse,
    RedlineSpanResponse,
    ReviewCell,
    ReviewRow,
    ReviewTableResponse,
)

logger = logging.getLogger(__name__)


class ContractAnalysisService:
    """
    Clause extraction and review tables.

    Field keys are stable so playbook review and redline can overlay later.
    """

    def __init__(
        self,
        *,
        llm: BaseLLM,
        documents: DocumentRepository,
        matters: MatterRepository,
    ) -> None:
        self._extractor = ClauseExtractor(llm)
        self._documents = documents
        self._matters = matters

    async def extract_document(
        self,
        *,
        user_id: UUID,
        document_id: UUID,
        field_keys: list[str] | None = None,
        extra_questions: list[ClauseFieldSpec] | None = None,
    ) -> ClauseExtractionResponse:
        document = await self._require_owned_document(user_id, document_id)
        text = (document.extracted_text or "").strip()
        if not text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "This document has no extracted text. "
                    "Re-upload it and try again."
                ),
            )
        fields = resolve_fields(field_keys, extra_as_fields(extra_questions))
        return await self._extractor.extract(
            document_id=document.id,
            filename=document.filename,
            text=text,
            fields=fields,
        )

    async def review_table(
        self,
        *,
        user_id: UUID,
        matter_id: UUID | None = None,
        conversation_id: UUID | None = None,
        field_keys: list[str] | None = None,
        extra_questions: list[ClauseFieldSpec] | None = None,
        document_ids: list[UUID] | None = None,
    ) -> ReviewTableResponse:
        if matter_id is not None:
            await self._require_matter(user_id, matter_id)

        documents = await self._documents.list_for_scope(
            owner_id=user_id,
            conversation_id=conversation_id,
            matter_id=matter_id,
        )
        if document_ids:
            wanted = set(document_ids)
            documents = [doc for doc in documents if doc.id in wanted]

        documents = [
            doc
            for doc in documents
            if doc.processed and (doc.extracted_text or "").strip()
        ]
        total = len(documents)
        truncated_docs = total > MAX_REVIEW_DOCUMENTS
        documents = documents[:MAX_REVIEW_DOCUMENTS]

        fields = resolve_fields(field_keys, extra_as_fields(extra_questions))
        columns = [
            ClauseFieldSpec(key=field.key, label=field.label, question=field.question)
            for field in fields
        ]
        if not documents:
            return ReviewTableResponse(
                matter_id=matter_id,
                conversation_id=conversation_id,
                columns=columns,
                rows=[],
                document_count=0,
                truncated_documents=False,
            )

        semaphore = asyncio.Semaphore(EXTRACT_CONCURRENCY)

        async def one(document: Document) -> ReviewRow:
            async with semaphore:
                return await self._row_for_document(document, fields)

        rows = await asyncio.gather(*[one(document) for document in documents])
        return ReviewTableResponse(
            matter_id=matter_id,
            conversation_id=conversation_id,
            columns=columns,
            rows=list(rows),
            document_count=total,
            truncated_documents=truncated_docs,
        )

    async def extract_for_chat(
        self,
        *,
        user_id: UUID,
        document_id: UUID | None,
        conversation_id: UUID | None,
        matter_id: UUID | None,
    ) -> ClauseExtractionResponse | None:
        document = await self._pick_document(
            user_id=user_id,
            document_id=document_id,
            conversation_id=conversation_id,
            matter_id=matter_id,
        )
        if document is None:
            return None
        text = (document.extracted_text or "").strip()
        if not text:
            return None
        try:
            return await self._extractor.extract(
                document_id=document.id,
                filename=document.filename,
                text=text,
            )
        except Exception:
            logger.exception(
                "Clause extraction failed document=%s",
                document.id,
            )
            return None

    def cards_markdown(self, extraction: ClauseExtractionResponse) -> str:
        return format_clause_cards(extraction)

    def table_markdown(self, table: ReviewTableResponse) -> str:
        return format_review_table(table)

    def playbook_markdown(self, review: PlaybookReviewResponse) -> str:
        return format_playbook_review(review)

    def redline_markdown(self, redline: RedlineResponse) -> str:
        return format_redline(redline)

    async def playbook_review_document(
        self,
        *,
        user_id: UUID,
        document_id: UUID,
        playbook_id: str | None = None,
        rules: list[PlaybookRuleSpec] | None = None,
    ) -> PlaybookReviewResponse:
        extraction = await self.extract_document(
            user_id=user_id,
            document_id=document_id,
        )
        playbook = self._resolve_playbook(playbook_id, rules)
        findings = evaluate_playbook(extraction.cards, playbook)
        return self._playbook_response(playbook, extraction, findings)

    async def playbook_review_for_chat(
        self,
        *,
        user_id: UUID,
        document_id: UUID | None,
        conversation_id: UUID | None,
        matter_id: UUID | None,
        playbook_id: str | None = None,
    ) -> PlaybookReviewResponse | None:
        extraction = await self.extract_for_chat(
            user_id=user_id,
            document_id=document_id,
            conversation_id=conversation_id,
            matter_id=matter_id,
        )
        if extraction is None:
            return None
        playbook = get_playbook(playbook_id)
        findings = evaluate_playbook(extraction.cards, playbook)
        return self._playbook_response(playbook, extraction, findings)

    async def redline_documents(
        self,
        *,
        user_id: UUID,
        document_id: UUID,
        against_document_id: UUID,
    ) -> RedlineResponse:
        left = await self._require_owned_document(user_id, document_id)
        right = await self._require_owned_document(user_id, against_document_id)
        left_text = (left.extracted_text or "").strip()
        right_text = (right.extracted_text or "").strip()
        if not left_text or not right_text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Both documents need extracted text before a redline.",
            )
        hunks = document_hunks(left_text, right_text)
        return RedlineResponse(
            mode="documents",
            left_document_id=left.id,
            right_document_id=right.id,
            left_filename=left.filename,
            right_filename=right.filename,
            hunks=[_hunk_response(hunk) for hunk in hunks],
            unchanged=all(
                all(span.op.value == "equal" for span in hunk.spans)
                for hunk in hunks
            ),
        )

    async def redline_playbook(
        self,
        *,
        user_id: UUID,
        document_id: UUID,
        playbook_id: str | None = None,
        rules: list[PlaybookRuleSpec] | None = None,
    ) -> RedlineResponse:
        review = await self.playbook_review_document(
            user_id=user_id,
            document_id=document_id,
            playbook_id=playbook_id,
            rules=rules,
        )
        engine_findings = [
            PlaybookFinding(
                key=item.key,
                label=item.label,
                verdict=EngineVerdict(item.verdict.value),
                severity=EngineSeverity(item.severity.value),
                extracted_value=item.extracted_value,
                quote=item.quote,
                preferred_position=item.preferred_position,
                fallback_position=item.fallback_position,
                suggested_language=item.suggested_language,
                reason=item.reason,
                required=item.required,
            )
            for item in review.findings
        ]
        hunks = playbook_hunks(engine_findings)
        return RedlineResponse(
            mode="playbook",
            left_document_id=review.document_id,
            right_document_id=None,
            left_filename=review.filename,
            right_filename="Playbook suggested language",
            hunks=[_hunk_response(hunk) for hunk in hunks],
            unchanged=len(hunks) == 0,
        )

    async def redline_for_chat(
        self,
        *,
        user_id: UUID,
        document_id: UUID | None,
        conversation_id: UUID | None,
        matter_id: UUID | None,
    ) -> RedlineResponse | None:
        documents = await self._documents.list_for_scope(
            owner_id=user_id,
            conversation_id=conversation_id,
            matter_id=matter_id,
        )
        ready = [
            doc
            for doc in documents
            if (doc.extracted_text or "").strip()
        ]
        if document_id is not None:
            pinned = [
                doc for doc in ready if doc.id == document_id
            ]
            if pinned:
                ready = pinned + [doc for doc in ready if doc.id != document_id]
        if len(ready) >= 2:
            return await self.redline_documents(
                user_id=user_id,
                document_id=ready[0].id,
                against_document_id=ready[1].id,
            )
        if not ready:
            return None
        return await self.redline_playbook(
            user_id=user_id,
            document_id=ready[0].id,
        )

    async def export_bytes(
        self,
        *,
        user_id: UUID,
        kind: ExportKind,
        document_id: UUID | None = None,
        against_document_id: UUID | None = None,
        matter_id: UUID | None = None,
        conversation_id: UUID | None = None,
        playbook_id: str | None = None,
        title: str | None = None,
        body: str | None = None,
    ) -> tuple[bytes, str]:
        if kind == ExportKind.CLAUSES:
            if document_id is None:
                raise HTTPException(status_code=400, detail="document_id is required.")
            extraction = await self.extract_document(
                user_id=user_id,
                document_id=document_id,
            )
            return docx_export.export_clauses(extraction), _docx_name(
                extraction.filename, "clauses"
            )
        if kind == ExportKind.PLAYBOOK:
            if document_id is None:
                raise HTTPException(status_code=400, detail="document_id is required.")
            review = await self.playbook_review_document(
                user_id=user_id,
                document_id=document_id,
                playbook_id=playbook_id,
            )
            return docx_export.export_playbook(review), _docx_name(
                review.filename, "playbook"
            )
        if kind == ExportKind.REDLINE:
            if document_id is None:
                raise HTTPException(status_code=400, detail="document_id is required.")
            if against_document_id is not None:
                redline = await self.redline_documents(
                    user_id=user_id,
                    document_id=document_id,
                    against_document_id=against_document_id,
                )
            else:
                redline = await self.redline_playbook(
                    user_id=user_id,
                    document_id=document_id,
                    playbook_id=playbook_id,
                )
            return docx_export.export_redline(redline), _docx_name(
                redline.left_filename or "redline", "redline"
            )
        if kind == ExportKind.REVIEW_TABLE:
            table = await self.review_table(
                user_id=user_id,
                matter_id=matter_id,
                conversation_id=conversation_id,
            )
            stem = "review-table"
            return docx_export.export_review_table(table), f"{stem}.docx"
        if kind == ExportKind.DRAFT:
            return (
                docx_export.export_draft(
                    title=title or "Legal draft",
                    body=body or "",
                ),
                _docx_name(title or "draft", "draft"),
            )
        raise HTTPException(status_code=400, detail="Unknown export kind.")

    def catalog_playbooks(self) -> list[PlaybookCatalogItem]:
        from app.contracts.playbook import PLAYBOOKS

        items: list[PlaybookCatalogItem] = []
        for playbook in PLAYBOOKS.values():
            items.append(
                PlaybookCatalogItem(
                    id=playbook.id,
                    name=playbook.name,
                    description=playbook.description,
                    rules=[
                        PlaybookRuleSpec(
                            key=rule.key,
                            required=rule.required,
                            accept_if_any=list(rule.accept_if_any),
                            accept_if_fallback=list(rule.accept_if_fallback),
                            reject_if_any=list(rule.reject_if_any),
                            preferred_position=rule.preferred_position,
                            fallback_position=rule.fallback_position,
                            suggested_language=rule.suggested_language,
                            severity=PlaybookSeverity(rule.severity.value),
                            notes=rule.notes,
                        )
                        for rule in playbook.rules
                    ],
                )
            )
        return items

    def _resolve_playbook(
        self,
        playbook_id: str | None,
        rules: list[PlaybookRuleSpec] | None,
    ) -> Playbook:
        if rules:
            return playbook_from_rules(
                name="Custom playbook",
                playbook_id=playbook_id or "custom",
                rules=[
                    PlaybookRule(
                        key=item.key,
                        required=item.required,
                        accept_if_any=tuple(item.accept_if_any),
                        accept_if_fallback=tuple(item.accept_if_fallback),
                        reject_if_any=tuple(item.reject_if_any),
                        preferred_position=item.preferred_position,
                        fallback_position=item.fallback_position,
                        suggested_language=item.suggested_language,
                        severity=EngineSeverity(item.severity.value),
                        notes=item.notes,
                    )
                    for item in rules
                ],
            )
        return get_playbook(playbook_id)

    def _playbook_response(
        self,
        playbook: Playbook,
        extraction: ClauseExtractionResponse,
        findings: list[PlaybookFinding],
    ) -> PlaybookReviewResponse:
        mapped = [_finding_response(item) for item in findings]
        return PlaybookReviewResponse(
            playbook_id=playbook.id,
            playbook_name=playbook.name,
            document_id=extraction.document_id,
            filename=extraction.filename,
            findings=mapped,
            summary=_summary(mapped),
            extraction=extraction,
        )

    async def _row_for_document(
        self,
        document: Document,
        fields: tuple[ClauseField, ...],
    ) -> ReviewRow:
        try:
            extraction = await self._extractor.extract(
                document_id=document.id,
                filename=document.filename,
                text=document.extracted_text or "",
                fields=fields,
            )
        except Exception:
            logger.exception(
                "Review table row failed document=%s",
                document.id,
            )
            return ReviewRow(
                document_id=document.id,
                filename=document.filename,
                error="Extraction failed for this file.",
                cells=[
                    ReviewCell(
                        key=field.key,
                        status=ClauseStatus.ERROR,
                        value=None,
                        quote=None,
                        quote_grounded=False,
                        confidence=0.0,
                    )
                    for field in fields
                ],
            )
        return ReviewRow(
            document_id=extraction.document_id,
            filename=extraction.filename,
            truncated=extraction.truncated,
            cells=[
                ReviewCell(
                    key=card.key,
                    status=card.status,
                    value=card.value,
                    quote=card.quote,
                    quote_grounded=card.quote_grounded,
                    confidence=card.confidence,
                    locked=card.locked,
                )
                for card in extraction.cards
            ],
        )

    async def _pick_document(
        self,
        *,
        user_id: UUID,
        document_id: UUID | None,
        conversation_id: UUID | None,
        matter_id: UUID | None,
    ) -> Document | None:
        if document_id is not None:
            document = await self._documents.get_by_id(document_id)
            if (
                document is not None
                and document.owner_id == user_id
                and (document.extracted_text or "").strip()
            ):
                return document
        documents = await self._documents.list_for_scope(
            owner_id=user_id,
            conversation_id=conversation_id,
            matter_id=matter_id,
        )
        for document in documents:
            if (document.extracted_text or "").strip():
                return document
        return None

    async def _require_owned_document(
        self,
        user_id: UUID,
        document_id: UUID,
    ) -> Document:
        document = await self._documents.get_by_id(document_id)
        if document is None or document.owner_id != user_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found.",
            )
        return document

    async def _require_matter(self, user_id: UUID, matter_id: UUID) -> None:
        matter = await self._matters.get(matter_id)
        if matter is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Matter not found.",
            )
        if matter.owner_id != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to access this matter.",
            )


def extra_as_fields(
    extra: list[ClauseFieldSpec] | None,
) -> list[ClauseField] | None:
    if not extra:
        return None
    return [
        ClauseField(key=item.key, label=item.label, question=item.question)
        for item in extra
    ]


def _finding_response(finding: PlaybookFinding) -> PlaybookFindingResponse:
    return PlaybookFindingResponse(
        key=finding.key,
        label=finding.label,
        verdict=PlaybookVerdict(finding.verdict.value),
        severity=PlaybookSeverity(finding.severity.value),
        extracted_value=finding.extracted_value,
        quote=finding.quote,
        preferred_position=finding.preferred_position,
        fallback_position=finding.fallback_position,
        suggested_language=finding.suggested_language,
        reason=finding.reason,
        required=finding.required,
    )


def _summary(findings: list[PlaybookFindingResponse]) -> PlaybookSummary:
    pass_count = sum(1 for item in findings if item.verdict == PlaybookVerdict.PASS)
    fallback_count = sum(
        1 for item in findings if item.verdict == PlaybookVerdict.FALLBACK
    )
    missing_count = sum(
        1 for item in findings if item.verdict == PlaybookVerdict.MISSING
    )
    off_count = sum(
        1 for item in findings if item.verdict == PlaybookVerdict.OFF_PLAYBOOK
    )
    unclear_count = sum(
        1 for item in findings if item.verdict == PlaybookVerdict.UNCLEAR
    )
    blocker_count = sum(
        1
        for item in findings
        if item.severity == PlaybookSeverity.BLOCKER
        and item.verdict
        in {
            PlaybookVerdict.MISSING,
            PlaybookVerdict.OFF_PLAYBOOK,
            PlaybookVerdict.UNCLEAR,
        }
    )
    return PlaybookSummary(
        pass_count=pass_count,
        fallback_count=fallback_count,
        missing_count=missing_count,
        off_playbook_count=off_count,
        unclear_count=unclear_count,
        blocker_count=blocker_count,
    )


def _hunk_response(hunk) -> RedlineHunkResponse:
    spans = [
        RedlineSpanResponse(op=span.op.value, text=span.text)
        for span in hunk.spans
    ]
    return RedlineHunkResponse(
        key=hunk.key,
        label=hunk.label,
        left=hunk.left,
        right=hunk.right,
        spans=spans,
        markdown=spans_to_markdown(hunk.spans),
        html=spans_to_html(hunk.spans),
    )


def _docx_name(stem: str, suffix: str) -> str:
    clean = "".join(
        char if char.isalnum() or char in {"-", "_"} else "-"
        for char in (stem or "export").rsplit(".", 1)[0]
    ).strip("-") or "export"
    return f"{clean}-{suffix}.docx"
