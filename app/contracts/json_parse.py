"""Pull a JSON object out of messy LLM text (fences, DeepSeek think tags)."""

from __future__ import annotations

import json
import re
from typing import Any


_THINK_BLOCK = re.compile(
    r"<think>.*?</think>",
    re.IGNORECASE | re.DOTALL,
)
_FENCE = re.compile(
    r"```(?:json)?\s*(\{.*?\})\s*```",
    re.IGNORECASE | re.DOTALL,
)


def parse_json_object(text: str) -> dict[str, Any]:
    """
    Return the first JSON object in `text`.

    Raises ValueError if nothing parseable is found.
    """
    if not (text or "").strip():
        raise ValueError("empty model output")

    cleaned = _THINK_BLOCK.sub("", text).strip()
    candidates: list[str] = []

    fenced = _FENCE.search(cleaned)
    if fenced:
        candidates.append(fenced.group(1))

    snippet = _first_object(cleaned)
    if snippet:
        candidates.append(snippet)

    candidates.append(cleaned)

    seen: set[str] = set()
    errors: list[str] = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            errors.append(str(exc))
            continue
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
            return {"fields": parsed}
        errors.append("JSON root is not an object")

    raise ValueError(errors[0] if errors else "no JSON object in model output")


def _first_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for index, char in enumerate(text[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None
