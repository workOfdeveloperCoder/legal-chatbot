from __future__ import annotations

import re
import uuid

_CHUNK_ID_DOC_PREFIX = re.compile(
    r"^([0-9a-fA-F]{8}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{4}-?"
    r"[0-9a-fA-F]{4}-?[0-9a-fA-F]{12})(?::|/|$)",
)


def normalize_document_id(document_id: str | None) -> str | None:
    """
    Canonical document identity for deduplication and API linking.

    Normalizes UUID strings to a single lowercase hyphenated form.
    """
    if document_id is None:
        return None

    value = str(document_id).strip()
    if not value:
        return None

    try:
        return str(uuid.UUID(value)).lower()
    except ValueError:
        return value.lower()


def recover_document_id(
    *,
    document_id: str | None = None,
    chunk_id: str | None = None,
) -> str | None:
    """
    Recover a canonical document_id from explicit id or chunk_id prefixes.

    Chunk ids are often stored as ``{document_uuid}:{index}``.
    """
    normalized = normalize_document_id(document_id)
    if normalized:
        return normalized

    if not chunk_id:
        return None

    match = _CHUNK_ID_DOC_PREFIX.match(str(chunk_id).strip())
    if not match:
        return None

    return normalize_document_id(match.group(1))


def chunk_identity_key(
    *,
    document_id: str | None,
    chunk_id: str | None,
    chunk_index: int | None = None,
    source_id: str | None = None,
) -> str:
    """
    Stable identity for a retrieved chunk.

    Prefer document_id + chunk_id, then document_id + chunk_index.
    """
    doc = recover_document_id(document_id=document_id, chunk_id=chunk_id)
    cid = (chunk_id or "").strip().lower()

    if doc and cid:
        return f"doc:{doc}:chunk:{cid}"

    if doc and chunk_index is not None:
        return f"doc:{doc}:index:{chunk_index}"

    if cid:
        return f"chunk:{cid}"

    if source_id:
        return f"source:{source_id}"

    return "unknown"


def normalize_filename(filename: str | None) -> str | None:
    if not filename:
        return None
    value = str(filename).strip().lower()
    return value or None


def resource_identity_key(
    *,
    document_id: str | None,
    chunk_id: str | None = None,
    source_id: str | None = None,
    filename: str | None = None,
    source_type: str | None = None,
    display_name: str | None = None,
) -> str:
    """
    HARD INVARIANT: one document_id = one resource.

    Same filename + different document_id remain separate resources.
    Filename/display_name are display-only and never used for merge.
    Legal corpus entries without document_id remain one resource per chunk.
    """
    del filename, source_type, display_name  # display-only; unused by design

    doc = recover_document_id(document_id=document_id, chunk_id=chunk_id)
    if doc:
        return f"doc:{doc}"

    cid = (chunk_id or "").strip().lower()
    if cid:
        return f"legal-chunk:{cid}"

    if source_id:
        return f"source:{source_id}"

    return "unknown"
