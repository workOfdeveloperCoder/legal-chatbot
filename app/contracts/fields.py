"""Canonical contract fields shared by extraction, review tables, and later playbooks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ClauseField:
    """One column in a review table / one card in clause extraction."""

    key: str
    label: str
    question: str


# Keep this list stable. Playbook rules and redline later match on `key`.
DEFAULT_CLAUSE_FIELDS: tuple[ClauseField, ...] = (
    ClauseField(
        "parties",
        "Parties",
        "Who are the contracting parties (legal names and roles)?",
    ),
    ClauseField(
        "effective_date",
        "Effective date",
        "What is the effective date or commencement date?",
    ),
    ClauseField(
        "term",
        "Term",
        "What is the term or duration of the agreement?",
    ),
    ClauseField(
        "governing_law",
        "Governing law",
        "Which law governs the agreement?",
    ),
    ClauseField(
        "dispute_resolution",
        "Dispute resolution",
        "How are disputes resolved (court, arbitration, seat, forum)?",
    ),
    ClauseField(
        "payment",
        "Payment",
        "What are the payment, fee, or consideration terms?",
    ),
    ClauseField(
        "liability_cap",
        "Liability cap",
        "Is there a limitation of liability or cap on damages?",
    ),
    ClauseField(
        "indemnity",
        "Indemnity",
        "What indemnity or hold-harmless obligations exist, and who bears them?",
    ),
    ClauseField(
        "termination",
        "Termination",
        "How can the agreement be terminated, and on what notice?",
    ),
    ClauseField(
        "renewal",
        "Renewal",
        "Is there renewal or auto-renewal, and on what terms?",
    ),
    ClauseField(
        "confidentiality",
        "Confidentiality",
        "What confidentiality or non-disclosure obligations apply?",
    ),
    ClauseField(
        "assignment",
        "Assignment",
        "Can rights or obligations be assigned, and with whose consent?",
    ),
)

_BY_KEY = {field.key: field for field in DEFAULT_CLAUSE_FIELDS}

MAX_CUSTOM_FIELDS = 8
MAX_REVIEW_DOCUMENTS = 20
MAX_DOCUMENT_CHARS = 24_000
EXTRACT_CONCURRENCY = 3


def field_by_key(key: str) -> ClauseField | None:
    return _BY_KEY.get((key or "").strip().lower())


def resolve_fields(
    keys: list[str] | None = None,
    extra: list[ClauseField] | None = None,
) -> tuple[ClauseField, ...]:
    """
    Build the extraction column set.

    Unknown keys are ignored. Extra fields (custom questions) are appended
    after the defaults, capped so the LLM JSON stays small.
    """
    if keys:
        selected = tuple(
            field
            for key in keys
            if (field := field_by_key(key)) is not None
        )
        if selected:
            base = selected
        else:
            base = DEFAULT_CLAUSE_FIELDS
    else:
        base = DEFAULT_CLAUSE_FIELDS

    extras: list[ClauseField] = []
    seen = {field.key for field in base}
    for field in extra or ():
        key = (field.key or "").strip().lower()[:64]
        if not key or key in seen:
            continue
        label = (field.label or field.question or key).strip()[:80]
        question = (field.question or field.label or key).strip()[:240]
        extras.append(ClauseField(key=key, label=label, question=question))
        seen.add(key)
        if len(extras) >= MAX_CUSTOM_FIELDS:
            break
    return base + tuple(extras)
