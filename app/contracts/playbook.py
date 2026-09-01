"""Playbook rules scored against the 12 canonical clause keys."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from app.contracts.fields import DEFAULT_CLAUSE_FIELDS, field_by_key
from app.schemas.contracts import ClauseCard, ClauseStatus

_WS = re.compile(r"\s+")


class PlaybookVerdict(str, Enum):
    PASS = "pass"
    FALLBACK = "fallback"
    MISSING = "missing"
    OFF_PLAYBOOK = "off_playbook"
    UNCLEAR = "unclear"
    NOT_APPLICABLE = "not_applicable"


class PlaybookSeverity(str, Enum):
    BLOCKER = "blocker"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True, slots=True)
class PlaybookRule:
    key: str
    required: bool = True
    accept_if_any: tuple[str, ...] = ()
    accept_if_fallback: tuple[str, ...] = ()
    reject_if_any: tuple[str, ...] = ()
    preferred_position: str = ""
    fallback_position: str = ""
    suggested_language: str = ""
    severity: PlaybookSeverity = PlaybookSeverity.WARNING
    notes: str = ""


@dataclass(frozen=True, slots=True)
class Playbook:
    id: str
    name: str
    description: str
    rules: tuple[PlaybookRule, ...]


@dataclass(frozen=True, slots=True)
class PlaybookFinding:
    key: str
    label: str
    verdict: PlaybookVerdict
    severity: PlaybookSeverity
    extracted_value: str | None
    quote: str | None
    preferred_position: str
    fallback_position: str
    suggested_language: str
    reason: str
    required: bool


PAKISTAN_COMMERCIAL = Playbook(
    id="pakistan_commercial_v1",
    name="Pakistan commercial (standard)",
    description=(
        "First-pass positions for Pakistani commercial contracts: "
        "local governing law and forum, a liability cap, termination "
        "notice, confidentiality, and consent to assignment."
    ),
    rules=(
        PlaybookRule(
            key="parties",
            required=True,
            preferred_position="Both parties identified by legal name and role.",
            suggested_language=(
                "This Agreement is made between [Party A], a company "
                "incorporated in Pakistan (“Party A”), and [Party B] "
                "(“Party B”)."
            ),
            severity=PlaybookSeverity.BLOCKER,
            notes="Must identify both contracting parties.",
        ),
        PlaybookRule(
            key="effective_date",
            required=True,
            preferred_position="A stated effective or commencement date.",
            suggested_language=(
                "This Agreement takes effect on [date] (the “Effective Date”)."
            ),
            severity=PlaybookSeverity.WARNING,
        ),
        PlaybookRule(
            key="term",
            required=True,
            preferred_position="A fixed term or a clearly stated duration.",
            suggested_language=(
                "This Agreement shall continue for a term of [period] "
                "from the Effective Date, unless terminated earlier in "
                "accordance with its terms."
            ),
            severity=PlaybookSeverity.WARNING,
        ),
        PlaybookRule(
            key="governing_law",
            required=True,
            accept_if_any=("pakistan", "islamic republic of pakistan"),
            reject_if_any=(
                "delaware",
                "england and wales",
                "laws of england",
                "new york",
                "singapore law",
                "california",
            ),
            preferred_position="Laws of Pakistan.",
            fallback_position=(
                "Pakistan law with a foreign arbitration seat only if approved."
            ),
            suggested_language=(
                "This Agreement shall be governed by and construed in "
                "accordance with the laws of Pakistan."
            ),
            severity=PlaybookSeverity.BLOCKER,
            notes="Foreign exclusive governing law is off-playbook.",
        ),
        PlaybookRule(
            key="dispute_resolution",
            required=True,
            accept_if_any=(
                "pakistan",
                "islamabad",
                "lahore",
                "karachi",
                "sindh",
                "punjab",
                "high court",
                "supreme court",
                "arbitration in pakistan",
            ),
            accept_if_fallback=("arbitration", "mediation"),
            reject_if_any=(
                "exclusive jurisdiction of the courts of england",
                "courts of new york",
                "delaware courts",
            ),
            preferred_position=(
                "Pakistan courts or arbitration seated in Pakistan."
            ),
            fallback_position="Arbitration, if the seat is acceptable.",
            suggested_language=(
                "Any dispute arising out of this Agreement shall be "
                "referred to the competent courts at [Islamabad / Lahore / "
                "Karachi], or to arbitration seated in Pakistan under "
                "the Arbitration Act, 1940."
            ),
            severity=PlaybookSeverity.BLOCKER,
        ),
        PlaybookRule(
            key="payment",
            required=True,
            preferred_position="Consideration, fees, or payment mechanics stated.",
            suggested_language=(
                "In consideration of the services, [paying party] shall "
                "pay [amount] within [days] days of invoice, in Pakistani "
                "Rupees, without set-off except as required by law."
            ),
            severity=PlaybookSeverity.WARNING,
        ),
        PlaybookRule(
            key="liability_cap",
            required=True,
            accept_if_any=(
                "cap",
                "capped",
                "limited to",
                "not exceed",
                "maximum",
                "limitation of liability",
            ),
            reject_if_any=(
                "unlimited liability",
                "uncapped",
                "no limitation of liability",
                "liability shall be unlimited",
            ),
            preferred_position="A stated cap on liability (fees or a fixed sum).",
            suggested_language=(
                "Subject to liability that cannot be limited by law, each "
                "party’s aggregate liability under this Agreement shall "
                "not exceed the fees paid in the [twelve] months preceding "
                "the claim."
            ),
            severity=PlaybookSeverity.BLOCKER,
            notes="Unlimited liability is off-playbook.",
        ),
        PlaybookRule(
            key="indemnity",
            required=False,
            reject_if_any=(
                "unlimited indemnity",
                "indemnify for all losses whatsoever",
                "without limit",
            ),
            preferred_position=(
                "Mutual or narrowly scoped indemnity; not unlimited."
            ),
            fallback_position="One-way indemnity if commercially justified.",
            suggested_language=(
                "Each party shall indemnify the other against third-party "
                "claims arising from its breach or negligence, subject to "
                "the limitation of liability in this Agreement."
            ),
            severity=PlaybookSeverity.WARNING,
        ),
        PlaybookRule(
            key="termination",
            required=True,
            accept_if_any=("terminate", "termination", "notice"),
            preferred_position=(
                "Termination for convenience and/or cause, with notice."
            ),
            suggested_language=(
                "Either party may terminate this Agreement for convenience "
                "on [30] days’ written notice, and immediately for material "
                "breach that remains unremedied [15] days after notice."
            ),
            severity=PlaybookSeverity.WARNING,
        ),
        PlaybookRule(
            key="renewal",
            required=False,
            reject_if_any=(
                "automatic renewal without notice",
                "auto-renew without notice",
                "perpetual",
            ),
            preferred_position=(
                "Renewal only by written agreement, or auto-renew with "
                "opt-out notice."
            ),
            fallback_position="Auto-renewal with at least 30 days’ opt-out notice.",
            suggested_language=(
                "This Agreement shall not renew automatically. Any renewal "
                "requires a written instrument signed by both parties."
            ),
            severity=PlaybookSeverity.WARNING,
        ),
        PlaybookRule(
            key="confidentiality",
            required=True,
            accept_if_any=(
                "confidential",
                "non-disclosure",
                "non disclosure",
                "nda",
            ),
            preferred_position="Confidentiality for a stated term.",
            suggested_language=(
                "Each party shall keep the other’s confidential information "
                "in confidence for [three] years after expiry or termination, "
                "and use it only to perform this Agreement."
            ),
            severity=PlaybookSeverity.WARNING,
        ),
        PlaybookRule(
            key="assignment",
            required=False,
            accept_if_any=(
                "prior written consent",
                "written consent",
                "consent of the other",
                "shall not assign",
            ),
            reject_if_any=(
                "freely assign",
                "assign without consent",
                "may assign this agreement without",
            ),
            preferred_position="No assignment without prior written consent.",
            suggested_language=(
                "Neither party may assign this Agreement without the prior "
                "written consent of the other, except to an affiliate or "
                "in connection with a solvent reorganization."
            ),
            severity=PlaybookSeverity.WARNING,
        ),
    ),
)

PLAYBOOKS: dict[str, Playbook] = {
    PAKISTAN_COMMERCIAL.id: PAKISTAN_COMMERCIAL,
}

_LABEL = {field.key: field.label for field in DEFAULT_CLAUSE_FIELDS}


def get_playbook(playbook_id: str | None) -> Playbook:
    if not playbook_id:
        return PAKISTAN_COMMERCIAL
    found = PLAYBOOKS.get(playbook_id.strip().lower())
    if found is None:
        return PAKISTAN_COMMERCIAL
    return found


def playbook_from_rules(
    *,
    name: str,
    rules: list[PlaybookRule],
    playbook_id: str = "custom",
    description: str = "",
) -> Playbook:
    cleaned: list[PlaybookRule] = []
    seen: set[str] = set()
    for rule in rules:
        key = (rule.key or "").strip().lower()
        if field_by_key(key) is None or key in seen:
            continue
        cleaned.append(
            PlaybookRule(
                key=key,
                required=rule.required,
                accept_if_any=rule.accept_if_any,
                accept_if_fallback=rule.accept_if_fallback,
                reject_if_any=rule.reject_if_any,
                preferred_position=rule.preferred_position,
                fallback_position=rule.fallback_position,
                suggested_language=rule.suggested_language,
                severity=rule.severity,
                notes=rule.notes,
            )
        )
        seen.add(key)
    return Playbook(
        id=playbook_id or "custom",
        name=name or "Custom playbook",
        description=description,
        rules=tuple(cleaned),
    )


def evaluate_playbook(
    cards: list[ClauseCard],
    playbook: Playbook,
) -> list[PlaybookFinding]:
    by_key = {card.key: card for card in cards}
    return [_score(rule, by_key.get(rule.key)) for rule in playbook.rules]


def _score(rule: PlaybookRule, card: ClauseCard | None) -> PlaybookFinding:
    label = _LABEL.get(rule.key, rule.key.replace("_", " ").title())
    value = card.value if card else None
    quote = card.quote if card else None
    status = card.status if card else ClauseStatus.MISSING
    haystack = _normalize(f"{value or ''} {quote or ''}")

    if card is None or status == ClauseStatus.ERROR:
        verdict = PlaybookVerdict.UNCLEAR
        reason = "This term could not be extracted."
    elif status in {ClauseStatus.MISSING, ClauseStatus.UNCLEAR} and not value:
        if status == ClauseStatus.UNCLEAR:
            verdict = PlaybookVerdict.UNCLEAR
            reason = "Extraction was unclear; lawyer review required."
        elif rule.required:
            verdict = PlaybookVerdict.MISSING
            reason = "Required term is not in the document."
        else:
            verdict = PlaybookVerdict.NOT_APPLICABLE
            reason = "Optional term is absent."
    elif _any_phrase(haystack, rule.reject_if_any):
        verdict = PlaybookVerdict.OFF_PLAYBOOK
        reason = "Contains language the playbook rejects."
    elif _any_phrase(haystack, rule.accept_if_any):
        verdict = PlaybookVerdict.PASS
        reason = "Aligns with the preferred position."
    elif _any_phrase(haystack, rule.accept_if_fallback):
        verdict = PlaybookVerdict.FALLBACK
        reason = "Uses an acceptable fallback, not the preferred position."
    elif rule.accept_if_any or rule.accept_if_fallback:
        verdict = PlaybookVerdict.OFF_PLAYBOOK
        reason = "Present, but does not match preferred or fallback language."
    else:
        verdict = PlaybookVerdict.PASS
        reason = "Term is present."

    if rule.notes and verdict in {
        PlaybookVerdict.OFF_PLAYBOOK,
        PlaybookVerdict.MISSING,
    }:
        reason = f"{reason} {rule.notes}".strip()

    return PlaybookFinding(
        key=rule.key,
        label=label,
        verdict=verdict,
        severity=rule.severity,
        extracted_value=value,
        quote=quote,
        preferred_position=rule.preferred_position,
        fallback_position=rule.fallback_position,
        suggested_language=rule.suggested_language,
        reason=reason,
        required=rule.required,
    )


def _normalize(text: str) -> str:
    return _WS.sub(" ", (text or "").replace("’", "'").lower()).strip()


def _any_phrase(haystack: str, phrases: tuple[str, ...]) -> bool:
    return any(_normalize(phrase) in haystack for phrase in phrases if phrase)
