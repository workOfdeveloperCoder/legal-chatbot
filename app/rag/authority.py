from __future__ import annotations

from enum import Enum


class AuthorityType(str, Enum):
    """Explicit legal-source authority classification."""

    CONSTITUTION = "constitution"
    STATUTE = "statute"
    RULES = "rules"
    SUPREME_COURT = "supreme_court"
    HIGH_COURT = "high_court"
    TRIBUNAL = "tribunal"
    OFFICIAL_LEGAL = "official_legal"
    LEGAL_CORPUS = "legal_corpus"
    UPLOADED_DOCUMENT = "uploaded_document"
    ARTICLE_COMMENTARY = "article_commentary"
    CONVERSATION_DOCUMENT = "conversation_document"
    MATTER_DOCUMENT = "matter_document"
    UNKNOWN = "unknown"


# Higher = more authoritative for legal propositions.
AUTHORITY_RANK: dict[AuthorityType, int] = {
    AuthorityType.CONSTITUTION: 100,
    AuthorityType.STATUTE: 95,
    AuthorityType.RULES: 90,
    AuthorityType.SUPREME_COURT: 88,
    AuthorityType.HIGH_COURT: 80,
    AuthorityType.TRIBUNAL: 70,
    AuthorityType.OFFICIAL_LEGAL: 65,
    AuthorityType.LEGAL_CORPUS: 60,
    AuthorityType.MATTER_DOCUMENT: 30,
    AuthorityType.CONVERSATION_DOCUMENT: 28,
    AuthorityType.UPLOADED_DOCUMENT: 26,
    AuthorityType.ARTICLE_COMMENTARY: 20,
    AuthorityType.UNKNOWN: 10,
}


def classify_authority(
    *,
    source_type: str | None = None,
    document_type: str | None = None,
    law_name: str | None = None,
    court: str | None = None,
    filename: str | None = None,
    display_name: str | None = None,
) -> AuthorityType:
    """Classify a retrieved chunk into an authority type."""
    source = (source_type or "").lower().strip()
    doc_type = (document_type or "").lower().strip()
    law = (law_name or "").lower().strip()
    court_name = (court or "").lower().strip()
    title = f"{display_name or ''} {filename or ''}".lower()

    if source == "conversation":
        if any(token in title for token in ("article", "commentary", "opinion")):
            return AuthorityType.ARTICLE_COMMENTARY
        return AuthorityType.CONVERSATION_DOCUMENT

    if source == "matter":
        if any(token in title for token in ("article", "commentary", "opinion")):
            return AuthorityType.ARTICLE_COMMENTARY
        return AuthorityType.MATTER_DOCUMENT

    if "constitution" in law or "constitution" in doc_type:
        return AuthorityType.CONSTITUTION

    if any(
        token in doc_type or token in law
        for token in ("statute", "act", "ordinance", "code", "ppc", "crpc", "cpc")
    ):
        return AuthorityType.STATUTE

    if any(token in doc_type or token in law for token in ("rule", "regulation", "sro")):
        return AuthorityType.RULES

    if "supreme court" in court_name or court_name in {"sc", "supreme"}:
        return AuthorityType.SUPREME_COURT

    if "high court" in court_name or "high court" in law:
        return AuthorityType.HIGH_COURT

    if any(token in court_name or token in doc_type for token in ("tribunal", "commission")):
        return AuthorityType.TRIBUNAL

    if source == "legal":
        if any(token in doc_type for token in ("judgment", "case", "precedent")):
            return AuthorityType.OFFICIAL_LEGAL
        return AuthorityType.LEGAL_CORPUS

    if source in {"matter", "conversation"}:
        return AuthorityType.UPLOADED_DOCUMENT

    return AuthorityType.UNKNOWN


def attribution_phrase(authority: AuthorityType) -> str:
    """Canonical attribution language for prompts and answers."""
    mapping = {
        AuthorityType.CONSTITUTION: "The Constitution provides...",
        AuthorityType.STATUTE: "The statute provides...",
        AuthorityType.RULES: "The rules provide...",
        AuthorityType.SUPREME_COURT: "The Supreme Court held...",
        AuthorityType.HIGH_COURT: "The High Court held...",
        AuthorityType.TRIBUNAL: "The tribunal decided...",
        AuthorityType.OFFICIAL_LEGAL: "The available authorities indicate...",
        AuthorityType.LEGAL_CORPUS: "The available authorities indicate...",
        AuthorityType.ARTICLE_COMMENTARY: "The article argues...",
        AuthorityType.UPLOADED_DOCUMENT: "The uploaded document states...",
        AuthorityType.CONVERSATION_DOCUMENT: "The uploaded document states...",
        AuthorityType.MATTER_DOCUMENT: "The matter document states...",
        AuthorityType.UNKNOWN: "The available evidence indicates...",
    }
    return mapping[authority]
