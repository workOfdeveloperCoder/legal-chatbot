from __future__ import annotations


def nomic_task_text(text: str, *, task_type: str) -> str:
    prefix = f"{task_type}: "
    stripped = text.strip()
    if stripped.startswith(prefix):
        return stripped
    return f"{prefix}{stripped}"
