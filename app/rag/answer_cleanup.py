from __future__ import annotations

import re

# Individual markers: [Source 1]
_SOURCE_INLINE = re.compile(r"\s*\[Source\s+\d+\]", re.IGNORECASE)

# Combined markers: [Source 1, Source 2] / [Source 1 and Source 2] / [Sources 1, 2]
_SOURCE_COMBINED = re.compile(
    r"\s*\[Sources?\s+\d+(?:\s*(?:,|and|&)\s*(?:Source\s+)?\d+)+\]",
    re.IGNORECASE,
)

# Parenthetical variants: (Source 1) / (Sources 1, 2)
_SOURCE_PAREN = re.compile(
    r"\s*\(Sources?\s+\d+(?:\s*(?:,|and|&)\s*(?:Source\s+)?\d+)*\)",
    re.IGNORECASE,
)

# Trailing Sources / References / Citations blocks with markers or bare bullets
_SOURCES_TRAILING = re.compile(
    r"\n+\s*(?:#{1,3}\s*)?(?:Sources?|References?|Citations?)\s*:?\s*"
    r"(?:\n\s*(?:[-*•]\s*)?(?:\[Sources?\s+[^\]]+\]|\(Sources?\s+[^)]+\)|[^\n]*))*\s*$",
    re.IGNORECASE | re.MULTILINE,
)

_SOURCES_BLOCK = re.compile(
    r"\n+\s*(?:#{1,3}\s*)?(?:Sources?|References?|Citations?)\s*:?\s*"
    r"\n(?:\s*(?:[-*•]\s*)?(?:\[Sources?\s+[^\]]+\]|\(Sources?\s+[^)]+\)|[^\S\n]*)\s*\n?)+",
    re.IGNORECASE,
)

_SOURCES_TRAILING_BARE = re.compile(
    r"\n+\s*(?:#{1,3}\s*)?(?:Sources?|References?|Citations?)\s*:?\s*"
    r"(?:\n\s*[-*•]\s*)*\s*$",
    re.IGNORECASE,
)

# Inline "Sources:" mid-sentence leftovers after stripping
_SOURCES_INLINE_LABEL = re.compile(
    r"(?:^|\n)\s*(?:#{1,3}\s*)?(?:Sources?|References?|Citations?)\s*:?\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def clean_answer_for_display(answer: str) -> str:
    """
    Remove LLM source marker text from the answer body.

    Structured sources are returned separately in ChatResponse.sources[] /
    resources[]. The UI renders clickable resource cards — not static markers.
    """
    if not answer:
        return ""

    text = answer.strip()
    text = _SOURCES_TRAILING.sub("", text)
    text = _SOURCES_BLOCK.sub("", text)
    text = _SOURCES_TRAILING_BARE.sub("", text)
    text = _SOURCE_COMBINED.sub("", text)
    text = _SOURCE_PAREN.sub("", text)
    text = _SOURCE_INLINE.sub("", text)
    text = _SOURCES_INLINE_LABEL.sub("", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\s+([.,;:])", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
