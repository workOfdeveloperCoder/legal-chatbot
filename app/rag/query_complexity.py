from __future__ import annotations

import re
from enum import Enum


class QueryComplexity(str, Enum):
    """Legal question complexity for retrieval depth and reasoning."""

    SIMPLE = "simple"
    RESEARCH = "research"
    COMPLEX = "complex"


_SIMPLE_PATTERNS = (
    re.compile(r"^what is section\s+[0-9A-Za-z\-]+\??$", re.I),
    re.compile(r"^define section\s+[0-9A-Za-z\-]+\??$", re.I),
    re.compile(r"^what does section\s+[0-9A-Za-z\-]+\s+(?:say|provide|state)\??$", re.I),
    re.compile(r"^explain section\s+[0-9A-Za-z\-]+\.?$", re.I),
    re.compile(r"^summarize (?:this|the) (?:document|article|clause)\.?$", re.I),
)

_COMPLEX_MARKERS = (
    "double jeopardy",
    "conflict",
    "contradict",
    "compare",
    "legally correct",
    "reflect pakistani law",
    "reflect the law",
    "consistent with",
    "clog on judicial discretion",
    "judicial discretion",
    "can it be challenged",
    "can this be challenged",
    "whether the argument",
    "is the argument",
    "does the argument",
    "multiple sources",
    "conflicting",
    "exception",
    "exceptions",
    "in the context of",
    "where the consumer",
    "where the plaintiff",
    "where the defendant",
)

_RESEARCH_MARKERS = (
    "effect of",
    "impact of",
    "application of",
    "apply to",
    "apply in",
    "prevent",
    "allow",
    "bar",
    "permit",
    "scope of",
    "interpretation",
    "interpret",
    "relationship between",
    "distinction between",
    "difference between",
    "when can",
    "when does",
    "under what circumstances",
    "how does",
    "what is the effect",
    "what is the law on",
    "what is the legal position",
    "legal assessment",
    "legal position",
)


def classify_query_complexity(
    question: str,
    *,
    task: str | None = None,
    mixed_qa: bool = False,
    document_qa: bool = False,
    follow_up: bool = False,
) -> QueryComplexity:
    normalized = " ".join(question.split()).strip()
    lower = normalized.lower()

    if mixed_qa:
        return QueryComplexity.COMPLEX

    if any(marker in lower for marker in _COMPLEX_MARKERS):
        return QueryComplexity.COMPLEX

    if follow_up and len(lower.split()) <= 12:
        if any(
            phrase in lower
            for phrase in (
                "can it",
                "can this",
                "what about",
                "how about",
                "does that",
                "does this",
            )
        ):
            return QueryComplexity.RESEARCH

    for pattern in _SIMPLE_PATTERNS:
        if pattern.match(lower):
            return QueryComplexity.SIMPLE

    if document_qa and not any(marker in lower for marker in _RESEARCH_MARKERS):
        if len(lower.split()) <= 14:
            return QueryComplexity.SIMPLE

    concept_count = _count_legal_concepts(lower)
    if concept_count >= 3:
        return QueryComplexity.COMPLEX

    if any(marker in lower for marker in _RESEARCH_MARKERS):
        return QueryComplexity.RESEARCH

    if concept_count >= 2 and "?" in normalized:
        return QueryComplexity.RESEARCH

    task_name = task.value if hasattr(task, "value") else str(task or "")
    if task_name in {"case_analysis", "mixed_qa", "contract_review"}:
        return QueryComplexity.COMPLEX

    if len(lower.split()) <= 10 and concept_count <= 1:
        return QueryComplexity.SIMPLE

    return QueryComplexity.RESEARCH


def _count_legal_concepts(text: str) -> int:
    concepts: set[str] = set()

    for match in re.finditer(
        r"(?:(?:\b(?:section|article|u/s)\s+)|(?:دفعہ\s*))([0-9A-Za-z\-]+)",
        text,
        re.I,
    ):
        concepts.add(f"section:{match.group(1).lower()}")

    keywords = (
        "injunction",
        "double jeopardy",
        "bail",
        "appeal",
        "limitation",
        "discretion",
        "clog",
        "contract",
        "negligence",
        "murder",
        "precedent",
        "judgment",
        "statute",
        "constitution",
        "specific performance",
        "arbitration",
        "maintenance",
        "defamation",
        "possession",
        "recovery",
        "ضمانت",
        "zamanat",
    )
    for keyword in keywords:
        if keyword in text:
            concepts.add(keyword)

    return len(concepts)
