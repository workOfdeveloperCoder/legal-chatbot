from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.rag.models import Message

_SECTION_RE = re.compile(
    r"\b(?:section|sec\.?|u/s|article)\s+([0-9A-Za-z\-]+)",
    re.IGNORECASE,
)
_CASE_RE = re.compile(
    r"\b(?:PLD|SCMR|MLD|YLR|CLC|PCr\.LJ)\s+\d{4}\s+[A-Za-z]+\s+\d+",
    re.IGNORECASE,
)
_STATUTE_RE = re.compile(
    r"\b([A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*)*\s+Act(?:,?\s*\d{4})?|"
    r"PPC|CrPC|CPC|Qanun-e-Shahadat)\b",
)
_ISSUE_RE = re.compile(
    r"\b(clog|discretion|double jeopardy|injunction|bail|limitation|"
    r"appeal|negligence|contract|inheritance|unreasonableness)\b",
    re.IGNORECASE,
)
_DOC_RE = re.compile(
    r"\b([A-Za-z0-9_\- ]+\.(?:txt|pdf|docx)|CLOG ON DISCRETION)\b",
    re.IGNORECASE,
)


@dataclass(slots=True)
class ActiveLegalContext:
    """Structured conversation memory for follow-up resolution."""

    active_issue: str | None = None
    active_statutes: list[str] = field(default_factory=list)
    active_cases: list[str] = field(default_factory=list)
    active_documents: list[str] = field(default_factory=list)
    active_matter: str | None = None
    unresolved_questions: list[str] = field(default_factory=list)

    def to_prompt_block(self) -> str | None:
        parts: list[str] = []
        if self.active_issue:
            parts.append(f"Active issue: {self.active_issue}")
        if self.active_statutes:
            parts.append("Active statutes: " + ", ".join(self.active_statutes[:5]))
        if self.active_cases:
            parts.append("Active cases: " + ", ".join(self.active_cases[:5]))
        if self.active_documents:
            parts.append(
                "Active documents: " + ", ".join(self.active_documents[:5])
            )
        if self.active_matter:
            parts.append(f"Active matter: {self.active_matter}")
        if self.unresolved_questions:
            parts.append(
                "Unresolved: " + "; ".join(self.unresolved_questions[:3])
            )
        if not parts:
            return None
        return "ACTIVE LEGAL CONTEXT:\n" + "\n".join(f"- {p}" for p in parts)

    def to_topic_string(self) -> str | None:
        bits: list[str] = []
        if self.active_issue:
            bits.append(self.active_issue)
        bits.extend(self.active_statutes[:2])
        bits.extend(self.active_cases[:1])
        bits.extend(self.active_documents[:1])
        if not bits:
            return None
        # de-dupe preserving order
        seen: set[str] = set()
        ordered: list[str] = []
        for bit in bits:
            key = bit.lower()
            if key in seen:
                continue
            seen.add(key)
            ordered.append(bit)
        return ", ".join(ordered[:4])

    def to_metadata(self) -> dict[str, object]:
        return {
            "active_issue": self.active_issue,
            "active_statutes": list(self.active_statutes),
            "active_cases": list(self.active_cases),
            "active_documents": list(self.active_documents),
            "active_matter": self.active_matter,
            "unresolved_questions": list(self.unresolved_questions),
        }


class ConversationContextBuilder:
    """Build structured legal context from conversation history."""

    def build(
        self,
        history: list[Message] | None,
        *,
        current_question: str | None = None,
        matter_id: str | None = None,
        document_names: list[str] | None = None,
    ) -> ActiveLegalContext:
        ctx = ActiveLegalContext(active_matter=matter_id)
        if document_names:
            for name in document_names:
                if name and name not in ctx.active_documents:
                    ctx.active_documents.append(name)

        texts: list[str] = []
        if history:
            for message in history[-10:]:
                texts.append(message.content or "")
        if current_question:
            texts.append(current_question)

        for text in texts:
            self._ingest(text, ctx)

        # Prefer the most recent substantive user question as unresolved if short follow-up.
        if history:
            for message in reversed(history[-6:]):
                if message.role != "user":
                    continue
                content = (message.content or "").strip()
                if len(content) > 20 and "?" in content:
                    if content not in ctx.unresolved_questions:
                        ctx.unresolved_questions.append(content[:180])
                    break

        return ctx

    def _ingest(self, text: str, ctx: ActiveLegalContext) -> None:
        if not text:
            return
        for match in _SECTION_RE.finditer(text):
            statute = f"Section {match.group(1)}"
            if statute.lower() not in {s.lower() for s in ctx.active_statutes}:
                ctx.active_statutes.append(statute)
        for match in _STATUTE_RE.finditer(text):
            name = match.group(0).strip()
            if name.lower() not in {s.lower() for s in ctx.active_statutes}:
                ctx.active_statutes.append(name)
        for match in _CASE_RE.finditer(text):
            cite = match.group(0).strip()
            if cite.lower() not in {c.lower() for c in ctx.active_cases}:
                ctx.active_cases.append(cite)
        for match in _DOC_RE.finditer(text):
            doc = match.group(0).strip()
            if doc.lower() not in {d.lower() for d in ctx.active_documents}:
                ctx.active_documents.append(doc)
        issue_match = _ISSUE_RE.search(text)
        if issue_match and not ctx.active_issue:
            # Prefer issue near a section mention.
            section = _SECTION_RE.search(text)
            if section:
                ctx.active_issue = (
                    f"{issue_match.group(0)} relating to Section {section.group(1)}"
                )
            else:
                ctx.active_issue = issue_match.group(0)
