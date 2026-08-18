from __future__ import annotations

import re

# DeepSeek R1 wraps internal reasoning in think tags
_THINK_OPEN = chr(60) + "think" + chr(62)
_THINK_CLOSE = chr(60) + "/" + "think" + chr(62)

_THINKING_BLOCK = re.compile(
    re.escape(_THINK_OPEN) + r".*?" + re.escape(_THINK_CLOSE),
    re.DOTALL | re.IGNORECASE,
)

_THINKING_UNCLOSED = re.compile(
    re.escape(_THINK_OPEN) + r".*",
    re.DOTALL | re.IGNORECASE,
)

_INTERNAL_ANALYSIS_HEADER = re.compile(
    r"(?:^|\n)\s*(?:EVIDENCE ANALYSIS|INTERNAL ANALYSIS|REASONING NOTES?)\s*:?\s*\n",
    re.IGNORECASE,
)


def strip_reasoning_output(text: str) -> str:
    """
    Remove DeepSeek R1 chain-of-thought and internal analysis blocks.

    Never expose model reasoning to the API consumer.
    """
    if not text:
        return ""

    cleaned = text.strip()
    cleaned = _THINKING_BLOCK.sub("", cleaned)
    cleaned = _THINKING_UNCLOSED.sub("", cleaned)

    if _INTERNAL_ANALYSIS_HEADER.search(cleaned):
        parts = _INTERNAL_ANALYSIS_HEADER.split(cleaned, maxsplit=1)
        if len(parts) == 2 and parts[1].strip():
            cleaned = parts[1].strip()

    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()
