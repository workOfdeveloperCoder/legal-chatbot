from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class QuickAction:
    """Empty-state card the frontend can render and send back on chat."""

    id: int
    slug: str
    title: str
    description: str
    icon: str
    prompt: str
    task: str
    requires_document: bool = False
    enables_web_search: bool = False


QUICK_ACTIONS: tuple[QuickAction, ...] = (
    QuickAction(
        id=1,
        slug="draft_document",
        title="Draft the Document",
        description="Create any legal document",
        icon="FilePlus",
        prompt=(
            "Help me draft a legal document. Ask what type I need "
            "(notice, plaint, petition, affidavit, contract, or other), "
            "the parties, the material facts, and the relief sought, then "
            "draft it in professional Pakistani legal form."
        ),
        task="document_draft",
    ),
    QuickAction(
        id=2,
        slug="find_authorities",
        title="Find Legal Authorities",
        description="Search cases, laws and more",
        icon="Scale",
        prompt=(
            "Find the governing Pakistani legal authorities for my issue. "
            "Search statutes, rules, and case law, and cite what you find. "
            "Ask me the issue, jurisdiction, and any section or citation "
            "I already have if that is missing."
        ),
        task="case_search",
        enables_web_search=True,
    ),
    QuickAction(
        id=3,
        slug="summarize_document",
        title="Summarize Document",
        description="Get key points in seconds",
        icon="FilePenLine",
        prompt=(
            "Summarize the uploaded document. Give the key points, "
            "parties or author position, and any legal issues it raises. "
            "If no document is attached, ask me to upload one."
        ),
        task="summarization",
        requires_document=True,
    ),
    QuickAction(
        id=4,
        slug="analyze_contract",
        title="Analyze a Contract",
        description="Review clauses and risks",
        icon="FileCog",
        prompt=(
            "Analyze the uploaded contract. Identify material clauses, "
            "obligations, termination, liability, and legal risks under "
            "Pakistani law. Extract key terms as structured clause cards. "
            "If no contract is attached, ask me to upload one "
            "or paste the clauses to review."
        ),
        task="contract_review",
        requires_document=True,
    ),
    QuickAction(
        id=5,
        slug="prepare_hearing",
        title="Prepare for Hearing",
        description="Build arguments and notes",
        icon="Gavel",
        prompt=(
            "Help me prepare for a hearing. Build a structured note with "
            "the issues, facts to prove, supporting authorities, likely "
            "objections, and oral submissions. Ask for the forum, stage, "
            "and my client's position if those are missing."
        ),
        task="hearing_prep",
    ),
    QuickAction(
        id=6,
        slug="compare_provisions",
        title="Compare Legal Provisions",
        description="Compare laws side by side",
        icon="Landmark",
        prompt=(
            "Compare the legal provisions I name side by side. Explain "
            "overlap, differences, which provision prevails, and how a "
            "Pakistani court would apply them. Ask which sections, articles, "
            "or statutes to compare if I have not named them yet."
        ),
        task="compare_provisions",
        enables_web_search=True,
    ),
)

HIDDEN_ACTIONS: tuple[QuickAction, ...] = (
    QuickAction(
        id=7,
        slug="review_table",
        title="Review Table",
        description="Extract the same clauses from every uploaded file",
        icon="Table",
        prompt=(
            "Build a review table across every uploaded contract in this "
            "matter. Extract parties, dates, governing law, liability, "
            "termination, and the other standard terms into a grid, one "
            "row per document."
        ),
        task="review_table",
        requires_document=True,
    ),
    QuickAction(
        id=8,
        slug="playbook_review",
        title="Playbook Review",
        description="Score the contract against firm positions",
        icon="BookCheck",
        prompt=(
            "Run a playbook review on the uploaded contract using the "
            "standard Pakistani commercial positions. Flag missing, "
            "off-playbook, and fallback terms, and suggest language."
        ),
        task="playbook_review",
        requires_document=True,
    ),
    QuickAction(
        id=9,
        slug="redline",
        title="Redline",
        description="Diff two drafts or show playbook markup",
        icon="GitCompare",
        prompt=(
            "Build a redline. If two documents are uploaded, compare the "
            "newer against the older. If only one is uploaded, redline it "
            "against the playbook suggested language."
        ),
        task="redline",
        requires_document=True,
    ),
)

_BY_SLUG = {action.slug: action for action in QUICK_ACTIONS + HIDDEN_ACTIONS}
_BY_ID = {action.id: action for action in QUICK_ACTIONS + HIDDEN_ACTIONS}


def _normalize_label(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


# Frontend card titles (including common typos in mock JSON).
_TITLE_ALIASES = {
    "draft the document": "draft_document",
    "find legal authorties": "find_authorities",
    "find legal authorities": "find_authorities",
    "sumarized document": "summarize_document",
    "summarized document": "summarize_document",
    "summarize document": "summarize_document",
    "analyze a contract": "analyze_contract",
    "prepare for hearing": "prepare_hearing",
    "compare legal provisions": "compare_provisions",
    "compare laws side by side": "compare_provisions",
    "compare laws": "compare_provisions",
    "review table": "review_table",
    "build a review table": "review_table",
    "diligence table": "review_table",
    "extract clauses from all documents": "review_table",
    "playbook review": "playbook_review",
    "run playbook": "playbook_review",
    "apply playbook": "playbook_review",
    "redline": "redline",
    "show redline": "redline",
    "compare drafts": "redline",
}
for _action in QUICK_ACTIONS + HIDDEN_ACTIONS:
    _TITLE_ALIASES[_normalize_label(_action.title)] = _action.slug

MISSING_DOCUMENT_REPLIES = {
    "summarize_document": (
        "Please upload the document you want summarized, then tap "
        "Summarize Document again. I will return the key points, parties "
        "or author position, and any legal issues it raises."
    ),
    "analyze_contract": (
        "Please upload the contract (PDF or text), or paste the clauses "
        "to review. I will identify material terms, obligations, and "
        "legal risks under Pakistani law."
    ),
    "review_table": (
        "Please upload at least one contract to this matter or "
        "conversation, then ask for a review table. I will extract the "
        "same key terms from every file into a grid."
    ),
    "playbook_review": (
        "Please upload the contract to score against the playbook, then "
        "ask for a playbook review again."
    ),
    "redline": (
        "Please upload one contract (for a playbook redline) or two "
        "versions (for a document redline), then ask again."
    ),
}

# First tap on these cards only needs clarifying questions — skip the LLM.
CARD_OPENING_REPLIES = {
    "draft_document": (
        "I can draft that for you in professional Pakistani legal form.\n\n"
        "Please tell me:\n"
        "1. **Document type** — notice, plaint, petition, affidavit, "
        "contract, or other\n"
        "2. **Parties** — names and roles (e.g. plaintiff / defendant)\n"
        "3. **Material facts** — what happened, in brief\n"
        "4. **Relief sought** — what you want the document to achieve\n\n"
        "Once I have those, I will prepare a first draft for your review."
    ),
    "find_authorities": (
        "I can find the governing Pakistani authorities for your issue.\n\n"
        "Please share:\n"
        "1. The **legal issue** or question\n"
        "2. **Jurisdiction** (e.g. Islamabad, Lahore, Karachi) if known\n"
        "3. Any **section, statute, or citation** you already have\n\n"
        "I will search statutes, rules, and case law and cite what I find."
    ),
    "prepare_hearing": (
        "I can build a hearing preparation note for you.\n\n"
        "Please share:\n"
        "1. **Forum** and stage (e.g. trial, bail, appeal)\n"
        "2. Your **client's position**\n"
        "3. The **issues** before the court\n"
        "4. Any authorities or facts already on hand\n\n"
        "I will structure issues, facts to prove, authorities, likely "
        "objections, and oral submissions."
    ),
    "compare_provisions": (
        "I can compare legal provisions side by side.\n\n"
        "Please name the **sections, articles, or statutes** to compare "
        "(for example Section 54-C of the Electricity Act, 1910 and a "
        "related rule). I will explain overlap, differences, which "
        "provision prevails, and how a Pakistani court would apply them."
    ),
}


def list_quick_actions() -> tuple[QuickAction, ...]:
    return QUICK_ACTIONS


def resolve_quick_action(value: str | int | None) -> QuickAction | None:
    if value is None:
        return None
    if isinstance(value, int):
        return _BY_ID.get(value)
    text = str(value).strip()
    if not text:
        return None
    slug = text.lower().replace("-", "_").replace(" ", "_")
    if slug in _BY_SLUG:
        return _BY_SLUG[slug]
    if slug.isdigit():
        return _BY_ID.get(int(slug))
    aliased = _TITLE_ALIASES.get(_normalize_label(text))
    if aliased:
        return _BY_SLUG[aliased]
    return None


def _prompt_stem(prompt: str) -> str:
    """Canned prompt without the trailing clarifying Ask-for/Ask-me sentence."""
    normalized = _normalize_label(prompt)
    for marker in (
        " ask me ",
        " ask for ",
        " if no document",
        " if no contract",
        " if those are missing",
    ):
        if marker in normalized:
            normalized = normalized.split(marker)[0]
            break
    return normalized.rstrip("., ")


def resolve_from_message(message: str | None) -> QuickAction | None:
    """Match a card tap that sends the title or the canned prompt as the message."""
    action = resolve_quick_action(message)
    if action is not None:
        return action
    normalized = _normalize_label(message)
    if not normalized:
        return None
    for candidate in QUICK_ACTIONS + HIDDEN_ACTIONS:
        if _normalize_label(candidate.title) == normalized:
            return candidate
        if _normalize_label(candidate.prompt) == normalized:
            return candidate
        stem = _prompt_stem(candidate.prompt)
        if stem and normalized.rstrip("., ") == stem:
            return candidate
        if stem and len(stem) >= 40 and normalized.startswith(stem[:48]):
            return candidate
    return None


def is_card_click(message: str | None, action: QuickAction | None) -> bool:
    """True when the user tapped a catalog card (title or canned prompt)."""
    if action is None or not (message or "").strip():
        return False
    normalized = _normalize_label(message)
    if _TITLE_ALIASES.get(normalized) == action.slug:
        return True
    if _normalize_label(action.title) == normalized:
        return True
    if _normalize_label(action.prompt) == normalized:
        return True
    stem = _prompt_stem(action.prompt)
    if stem and normalized.rstrip("., ") == stem:
        return True
    if stem and len(stem) >= 40 and normalized.startswith(stem[:48]):
        return True
    return False


def should_return_card_opening(
    message: str | None,
    action: QuickAction | None,
    *,
    quick_action: str | int | None = None,
) -> bool:
    """
    First tap on clarifying-first cards must NOT run retrieval/LLM.

    Otherwise hung Ollama / empty corpus returns 503 or 'insufficient'
    instead of asking for forum / issue / parties.
    """
    if action is None or action.slug not in CARD_OPENING_REPLIES:
        return False
    if is_card_click(message, action):
        return True
    # FE card tap always sends the slug; typed follow-ups do not.
    forced = resolve_quick_action(quick_action)
    return forced is not None and forced.slug == action.slug
