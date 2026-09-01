from __future__ import annotations

import logging
import re
from uuid import UUID

from app.contracts.fields import (
    MAX_DOCUMENT_CHARS,
    ClauseField,
    resolve_fields,
)
from app.contracts.json_parse import parse_json_object
from app.llm.base import BaseLLM
from app.schemas.contracts import ClauseCard, ClauseExtractionResponse, ClauseStatus
from app.schemas.llm import ChatCompletionRequest, ChatMessage

logger = logging.getLogger(__name__)

_WS = re.compile(r"\s+")

_SYSTEM = (
    "You extract contract terms for a Pakistani commercial lawyer. "
    "Reply with a single JSON object only. No markdown, no commentary, "
    "no chain of thought. If a term is absent, status is missing and "
    "value is null. Quotes must be verbatim spans from the document."
)


class ClauseExtractor:
    """One document → structured clause cards. Playbooks later match on card.key."""

    def __init__(self, llm: BaseLLM) -> None:
        self._llm = llm

    async def extract(
        self,
        *,
        document_id: UUID,
        filename: str,
        text: str,
        fields: tuple[ClauseField, ...] | None = None,
    ) -> ClauseExtractionResponse:
        columns = fields or resolve_fields()
        truncated = (text or "").strip()
        truncated_flag = False
        if len(truncated) > MAX_DOCUMENT_CHARS:
            truncated = truncated[:MAX_DOCUMENT_CHARS]
            truncated_flag = True

        if not truncated:
            cards = [
                ClauseCard(
                    key=field.key,
                    label=field.label,
                    status=ClauseStatus.MISSING,
                    value=None,
                    quote=None,
                    quote_grounded=False,
                    confidence=0.0,
                )
                for field in columns
            ]
            return ClauseExtractionResponse(
                document_id=document_id,
                filename=filename,
                truncated=False,
                cards=cards,
            )

        raw = await self._llm.generate(
            ChatCompletionRequest(
                messages=[
                    ChatMessage(role="system", content=_SYSTEM),
                    ChatMessage(
                        role="user",
                        content=self._user_prompt(truncated, columns),
                    ),
                ],
                temperature=0.0,
                max_tokens=2500,
            )
        )
        try:
            payload = parse_json_object(raw.content or "")
        except ValueError:
            logger.warning(
                "Clause extraction JSON parse failed document=%s",
                document_id,
            )
            payload = {}

        by_key = self._index_fields(payload)
        cards = [
            self._card_from_model(field, by_key.get(field.key), truncated)
            for field in columns
        ]
        return ClauseExtractionResponse(
            document_id=document_id,
            filename=filename,
            truncated=truncated_flag,
            cards=cards,
        )

    def _user_prompt(self, text: str, fields: tuple[ClauseField, ...]) -> str:
        catalog = "\n".join(
            f'- "{field.key}": {field.question}' for field in fields
        )
        keys = ", ".join(f'"{field.key}"' for field in fields)
        return (
            "Extract these fields from the contract.\n"
            f"{catalog}\n\n"
            "Return JSON of the form:\n"
            '{"fields":[{"key":"...","value":"...","quote":"...","'
            'status":"found|missing|unclear","confidence":0.0}]}\n'
            f"Use only these keys: {keys}.\n"
            "value is a short professional answer, not a full clause.\n"
            "quote is a short verbatim excerpt (max 40 words) or null.\n"
            "confidence is 0 to 1.\n\n"
            "CONTRACT TEXT:\n"
            f"{text}"
        )

    @staticmethod
    def _index_fields(payload: dict) -> dict[str, dict]:
        rows = payload.get("fields")
        if not isinstance(rows, list):
            # Allow a flat {key: {value, quote, ...}} object.
            rows = [
                {"key": key, **value}
                if isinstance(value, dict)
                else {"key": key, "value": value}
                for key, value in payload.items()
                if key != "fields"
            ]
        indexed: dict[str, dict] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            key = str(row.get("key") or "").strip().lower()
            if key:
                indexed[key] = row
        return indexed

    @staticmethod
    def _card_from_model(
        field: ClauseField,
        row: dict | None,
        document_text: str,
    ) -> ClauseCard:
        if not row:
            return ClauseCard(
                key=field.key,
                label=field.label,
                status=ClauseStatus.UNCLEAR,
                value=None,
                quote=None,
                quote_grounded=False,
                confidence=0.0,
            )

        raw_status = str(row.get("status") or "").strip().lower()
        try:
            status = ClauseStatus(raw_status)
        except ValueError:
            status = ClauseStatus.UNCLEAR

        value = _clean_optional(row.get("value"))
        quote = _clean_optional(row.get("quote"))
        if quote and len(quote) > 400:
            quote = quote[:400].rstrip() + "…"

        grounded = quote_in_document(quote, document_text) if quote else False
        if quote and not grounded:
            quote = None

        if status == ClauseStatus.FOUND and not value:
            status = ClauseStatus.UNCLEAR
        if not value and not quote:
            status = ClauseStatus.MISSING

        confidence = _as_confidence(row.get("confidence"))
        if status == ClauseStatus.MISSING:
            confidence = min(confidence, 0.4)

        return ClauseCard(
            key=field.key,
            label=field.label,
            status=status,
            value=value,
            quote=quote,
            quote_grounded=grounded,
            confidence=confidence,
        )


def quote_in_document(quote: str | None, document_text: str) -> bool:
    if not quote or not document_text:
        return False
    needle = _normalize(quote)
    haystack = _normalize(document_text)
    if len(needle) < 12:
        return needle in haystack
    if needle in haystack:
        return True
    # Allow light quote shortening by the model.
    words = needle.split()
    if len(words) >= 6:
        return " ".join(words[:6]) in haystack
    return False


def _normalize(text: str) -> str:
    return _WS.sub(" ", text).strip().lower()


def _clean_optional(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"null", "none", "n/a", "na", "-"}:
        return None
    return text


def _as_confidence(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(1.0, number))
