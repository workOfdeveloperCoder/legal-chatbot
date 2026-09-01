from __future__ import annotations

import re
import uuid
from pathlib import Path

from app.models.document import Document
from app.rag.document_identity import normalize_document_id, recover_document_id
from app.rag.models import RetrievedChunk


def _is_postgres_uuid(value: str | None) -> bool:
    if not value:
        return False
    try:
        uuid.UUID(str(value).strip())
        return True
    except (TypeError, ValueError, AttributeError):
        return False


_BYLINE_PATTERN = re.compile(
    r"^by\s+.+",
    re.IGNORECASE,
)


def derive_display_name(
    extracted_text: str | None,
    filename: str,
) -> str:
    """
    Derive a human-readable title from extracted document text.

    Uses the first substantive line when it looks like a title
    (e.g. "CLOG ON DISCRETION" followed by "By Author...").
    Falls back to filename stem — never invents data.
    """
    if not extracted_text or not extracted_text.strip():
        return Path(filename).stem

    lines = [
        line.strip()
        for line in extracted_text.splitlines()
        if line.strip()
    ]
    if not lines:
        return Path(filename).stem

    first = lines[0]
    if (
        len(first) <= 120
        and not first.lower().startswith(("section ", "chapter ", "page "))
    ):
        if len(lines) > 1 and _BYLINE_PATTERN.match(lines[1]):
            return first
        if len(first) <= 80 and first.isupper():
            return first
        if len(first) <= 80 and not first.endswith("."):
            return first

    return Path(filename).stem


def derive_author_line(extracted_text: str | None) -> str | None:
    if not extracted_text:
        return None

    for line in extracted_text.splitlines():
        stripped = line.strip()
        if _BYLINE_PATTERN.match(stripped):
            return stripped
    return None


def enrich_chunks_from_documents(
    chunks: list[RetrievedChunk],
    documents: dict[str, Document],
) -> list[RetrievedChunk]:
    enriched: list[RetrievedChunk] = []

    for chunk in chunks:
        recovered_id = recover_document_id(
            document_id=chunk.document_id,
            chunk_id=chunk.chunk_id or chunk.id,
        )

        if not recovered_id:
            enriched.append(chunk)
            continue

        document = documents.get(recovered_id)
        if document is None and chunk.document_id:
            document = documents.get(normalize_document_id(chunk.document_id) or "")
            if document is None:
                document = documents.get(chunk.document_id)
        if document is None:
            if recovered_id != chunk.document_id:
                enriched.append(
                    chunk.model_copy(update={"document_id": recovered_id})
                )
            else:
                enriched.append(chunk)
            continue

        display_name = derive_display_name(
            document.extracted_text,
            document.filename,
        )
        author = derive_author_line(document.extracted_text)
        updates: dict = {
            "document_id": recovered_id,
            "filename": chunk.filename or document.filename,
            "display_name": display_name,
        }
        if author:
            updates["author"] = author
        if chunk.matter_id is None and document.matter_id:
            updates["matter_id"] = str(document.matter_id)

        enriched.append(chunk.model_copy(update=updates))

    return enriched


async def load_documents_for_chunks(
    repository,
    chunks: list[RetrievedChunk],
) -> dict[str, Document]:
    document_ids: set[str] = set()
    for chunk in chunks:
        recovered = recover_document_id(
            document_id=chunk.document_id,
            chunk_id=chunk.chunk_id or chunk.id,
        )
        if recovered:
            document_ids.add(recovered)
        elif chunk.document_id:
            document_ids.add(str(chunk.document_id))

    documents: dict[str, Document] = {}

    for document_id in document_ids:
        if not _is_postgres_uuid(document_id):
            continue
        document = await repository.get_by_id(document_id)
        if document is not None:
            documents[normalize_document_id(str(document.id)) or str(document.id)] = document

    return documents
