"""Word-level redline between two texts, or playbook suggested language."""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from enum import Enum

from app.contracts.playbook import PlaybookFinding, PlaybookVerdict

_TOKEN = re.compile(r"\S+|\s+")
_PARA = re.compile(r"\n\s*\n")
MAX_REDLINE_CHARS = 24_000


class RedlineOp(str, Enum):
    EQUAL = "equal"
    DELETE = "delete"
    INSERT = "insert"


@dataclass(frozen=True, slots=True)
class RedlineSpan:
    op: RedlineOp
    text: str


@dataclass(frozen=True, slots=True)
class RedlineHunk:
    key: str | None
    label: str
    left: str
    right: str
    spans: tuple[RedlineSpan, ...]


def word_diff(left: str, right: str) -> tuple[RedlineSpan, ...]:
    left_tokens = _TOKEN.findall(left or "")
    right_tokens = _TOKEN.findall(right or "")
    matcher = difflib.SequenceMatcher(a=left_tokens, b=right_tokens, autojunk=False)
    spans: list[RedlineSpan] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            spans.append(RedlineSpan(RedlineOp.EQUAL, "".join(left_tokens[i1:i2])))
        elif tag == "delete":
            spans.append(RedlineSpan(RedlineOp.DELETE, "".join(left_tokens[i1:i2])))
        elif tag == "insert":
            spans.append(RedlineSpan(RedlineOp.INSERT, "".join(right_tokens[j1:j2])))
        else:
            spans.append(RedlineSpan(RedlineOp.DELETE, "".join(left_tokens[i1:i2])))
            spans.append(RedlineSpan(RedlineOp.INSERT, "".join(right_tokens[j1:j2])))
    return tuple(span for span in spans if span.text)


def document_hunks(left: str, right: str) -> tuple[RedlineHunk, ...]:
    left_text = (left or "")[:MAX_REDLINE_CHARS]
    right_text = (right or "")[:MAX_REDLINE_CHARS]
    left_paras = _paragraphs(left_text)
    right_paras = _paragraphs(right_text)
    if len(left_paras) <= 1 and len(right_paras) <= 1:
        return (
            RedlineHunk(
                key=None,
                label="Document",
                left=left_text,
                right=right_text,
                spans=word_diff(left_text, right_text),
            ),
        )

    matcher = difflib.SequenceMatcher(a=left_paras, b=right_paras, autojunk=False)
    hunks: list[RedlineHunk] = []
    index = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        index += 1
        old = "\n\n".join(left_paras[i1:i2])
        new = "\n\n".join(right_paras[j1:j2])
        hunks.append(
            RedlineHunk(
                key=None,
                label=f"Change {index}",
                left=old,
                right=new,
                spans=word_diff(old, new),
            )
        )
    if not hunks:
        hunks.append(
            RedlineHunk(
                key=None,
                label="Document",
                left=left_text,
                right=right_text,
                spans=(RedlineSpan(RedlineOp.EQUAL, left_text),),
            )
        )
    return tuple(hunks)


def playbook_hunks(findings: list[PlaybookFinding]) -> tuple[RedlineHunk, ...]:
    actionable = {
        PlaybookVerdict.MISSING,
        PlaybookVerdict.OFF_PLAYBOOK,
        PlaybookVerdict.FALLBACK,
        PlaybookVerdict.UNCLEAR,
    }
    hunks: list[RedlineHunk] = []
    for finding in findings:
        if finding.verdict not in actionable:
            continue
        if not (finding.suggested_language or "").strip():
            continue
        current = (finding.quote or finding.extracted_value or "").strip()
        suggested = finding.suggested_language.strip()
        hunks.append(
            RedlineHunk(
                key=finding.key,
                label=finding.label,
                left=current,
                right=suggested,
                spans=word_diff(current, suggested),
            )
        )
    return tuple(hunks)


def spans_to_markdown(spans: tuple[RedlineSpan, ...]) -> str:
    parts: list[str] = []
    for span in spans:
        text = span.text.replace("\n", " ")
        if span.op == RedlineOp.DELETE:
            parts.append(f"~~{text}~~")
        elif span.op == RedlineOp.INSERT:
            parts.append(f"**{text}**")
        else:
            parts.append(text)
    return "".join(parts).strip()


def spans_to_html(spans: tuple[RedlineSpan, ...]) -> str:
    parts: list[str] = []
    for span in spans:
        escaped = (
            span.text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        if span.op == RedlineOp.DELETE:
            parts.append(f'<del class="redline-del">{escaped}</del>')
        elif span.op == RedlineOp.INSERT:
            parts.append(f'<ins class="redline-ins">{escaped}</ins>')
        else:
            parts.append(escaped)
    return "".join(parts)


def _paragraphs(text: str) -> list[str]:
    parts = [_WS_STRIP(part) for part in _PARA.split((text or "").strip())]
    return [part for part in parts if part]


def _WS_STRIP(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text).strip()
