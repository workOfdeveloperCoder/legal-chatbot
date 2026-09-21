"""Normalize Pakistani statute section ids for retrieval / filters / rerank."""
from __future__ import annotations

import re

# ASCII hyphen, non-breaking hyphen, en-dash, em-dash, minus
_HYPHENS = re.compile(r"[\s\u00ad\u2010\u2011\u2012\u2013\u2014\u2212\-]+")


def normalize_hyphens(text: str) -> str:
    return _HYPHENS.sub("-", text or "")


def normalize_section_id(raw: str | None) -> str:
    """
    Collapse ``54C`` / ``54-C`` / ``54‑C`` / ``Section 54-C`` → ``54-c``.
    """
    if not raw:
        return ""
    cleaned = normalize_hyphens(str(raw)).strip().lower()
    cleaned = re.sub(r"^(?:section|sec\.?|u/s|article)\s*", "", cleaned)
    cleaned = cleaned.strip(" .:")
    # Insert hyphen between trailing digits and a single letter suffix: 54c → 54-c
    cleaned = re.sub(r"^(\d+)([a-z])$", r"\1-\2", cleaned)
    return cleaned


def section_keyword_variants(raw: str | None) -> list[str]:
    """
    Build general MatchValue candidates for payload ``keywords`` / ``sections``.
    No statute-specific special cases — works for any section id the user typed.

    Keep distinct casings: Qdrant ``MatchValue`` is case-sensitive.
    """
    nid = normalize_section_id(raw)
    if not nid:
        return []
    compact = nid.replace("-", "")
    titled_hyphen = (
        re.sub(
            r"^(\d+)-([a-z])$",
            lambda m: f"{m.group(1)}-{m.group(2).upper()}",
            nid,
        )
        if "-" in nid
        else nid.upper()
    )
    variants = [
        f"section {titled_hyphen}",
        f"Section {titled_hyphen}",
        f"section {nid}",
        f"Section {nid}",
        f"section {compact}",
        f"Section {compact}",
        titled_hyphen,
        nid,
        compact,
        compact.upper(),
        f"Article {titled_hyphen}",
        f"article {nid}",
    ]
    seen: set[str] = set()
    out: list[str] = []
    for item in variants:
        # Exact string dedupe only (preserve case variants for MatchValue).
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def section_text_needles(raw: str | None) -> list[str]:
    """Substrings to look for inside chunk_text (incl. unicode hyphen forms)."""
    nid = normalize_section_id(raw)
    if not nid:
        return []
    compact = nid.replace("-", "")
    titled_hyphen = (
        re.sub(r"^(\d+)-([a-z])$", lambda m: f"{m.group(1)}-{m.group(2).upper()}", nid)
        if "-" in nid
        else nid
    )
    base = [
        nid,
        compact,
        titled_hyphen,
        f"section {nid}",
        f"section {compact}",
        f"section {titled_hyphen}",
        nid.replace("-", "\u2011"),
        nid.replace("-", "\u2013"),
        titled_hyphen.replace("-", "\u2011"),
        f"section {nid.replace('-', '\u2011')}",
        f"section {titled_hyphen.replace('-', '\u2011')}",
    ]
    seen: set[str] = set()
    out: list[str] = []
    for item in base:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


_SECTION_TOKEN = re.compile(
    r"\b\d{1,4}[a-z]?\b|\b\d{1,4}-[a-z]\b",
    re.IGNORECASE,
)
_WORD_TOKEN = re.compile(r"[a-z0-9]{3,}", re.IGNORECASE)


def tokenize_for_rerank(text: str) -> set[str]:
    """
    Keep statute tokens like ``54-c`` / ``54c`` that plain word splits drop.
    """
    lowered = normalize_hyphens(text or "").lower()
    tokens = {m.group(0).lower() for m in _WORD_TOKEN.finditer(lowered)}
    for match in _SECTION_TOKEN.finditer(lowered):
        raw = match.group(0).lower()
        tokens.add(raw)
        nid = normalize_section_id(raw)
        if nid:
            tokens.add(nid)
            tokens.add(nid.replace("-", ""))
    return tokens
