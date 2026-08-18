from __future__ import annotations

import json
import re
from typing import Any

SENSITIVE_KEY_PATTERN = re.compile(
    r"(password|passwd|secret|token|authorization|api[_-]?key|"
    r"refresh[_-]?token|access[_-]?token|credential|private[_-]?key|"
    r"password_hash|client_secret)",
    re.IGNORECASE,
)

REDACTED = "[REDACTED]"


def _is_sensitive_key(key: str) -> bool:
    return bool(SENSITIVE_KEY_PATTERN.search(key))


def sanitize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return sanitize_mapping(value)

    if isinstance(value, list):
        return [sanitize_value(item) for item in value]

    return value


def sanitize_mapping(data: dict[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}

    for key, value in data.items():
        if _is_sensitive_key(key):
            sanitized[key] = REDACTED
            continue

        sanitized[key] = sanitize_value(value)

    return sanitized


def sanitize_headers(headers: dict[str, str]) -> dict[str, str]:
    sanitized: dict[str, str] = {}

    for key, value in headers.items():
        if _is_sensitive_key(key):
            sanitized[key] = REDACTED
        else:
            sanitized[key] = value

    return sanitized


def parse_json_body(raw: bytes) -> dict | list | str | None:
    if not raw:
        return None

    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError:
        return "[binary body omitted]"

    try:
        parsed = json.loads(decoded)
    except json.JSONDecodeError:
        return decoded

    return sanitize_value(parsed)


def truncate_payload(
    payload: dict | list | str | None,
    *,
    max_bytes: int,
) -> dict | list | str | None:
    if payload is None:
        return None

    serialized = json.dumps(payload, default=str)

    if len(serialized.encode("utf-8")) <= max_bytes:
        return payload

    return {
        "_truncated": True,
        "preview": serialized[:max_bytes],
    }
