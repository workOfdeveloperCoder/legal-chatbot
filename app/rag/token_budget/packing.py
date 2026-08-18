from __future__ import annotations

import re
from copy import deepcopy

from app.llm.token_counter import TokenCounter
from app.rag.authority import AUTHORITY_RANK, classify_authority
from app.rag.evidence import (
    dedupe_near_duplicate_chunks,
    passages_are_near_duplicates,
)
from app.rag.models import Message, RetrievedChunk, SourceType

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_LEGAL_ISSUE_RE = re.compile(
    r"\b(?:section|article|u/s)\s+[0-9A-Za-z\-]+|"
    r"\b(?:clog|discretion|injunction|limitation|bail|appeal|"
    r"double jeopardy|54-c)\b",
    re.IGNORECASE,
)


def score_evidence_chunk(chunk: RetrievedChunk) -> float:
    """Rank evidence for packing: relevance, authority, strength proxies."""
    relevance = float(chunk.relevance_score if chunk.relevance_score is not None else chunk.score or 0.0)
    authority = classify_authority(
        source_type=chunk.source_type,
        document_type=chunk.document_type,
        law_name=chunk.law_name,
        court=chunk.court,
        filename=chunk.filename,
        display_name=chunk.display_name,
    )
    authority_score = AUTHORITY_RANK.get(authority, 10) / 100.0

    source_priority = {
        SourceType.LEGAL.value: 1.0,
        SourceType.MATTER.value: 0.85,
        SourceType.CONVERSATION.value: 0.80,
    }.get((chunk.source_type or SourceType.LEGAL.value).lower(), 0.7)

    # Prefer chunks that still look like complete passages.
    text_len = len(chunk.text or "")
    completeness = 1.0 if text_len >= 200 else 0.7 if text_len >= 80 else 0.4

    return (
        relevance * 0.45
        + authority_score * 0.25
        + source_priority * 0.20
        + completeness * 0.10
    )


def remove_duplicate_evidence(
    chunks: list[RetrievedChunk],
) -> tuple[list[RetrievedChunk], int]:
    """Exact + near-duplicate removal while preserving metadata survivors."""
    if not chunks:
        return [], 0

    before = len(chunks)
    seen_keys: set[str] = set()
    unique: list[RetrievedChunk] = []
    for chunk in chunks:
        key = _dedupe_key(chunk)
        if key in seen_keys:
            continue
        # Also skip near-dups against already kept same-document passages.
        if any(
            kept.document_id
            and chunk.document_id
            and kept.document_id == chunk.document_id
            and passages_are_near_duplicates(kept.text, chunk.text)
            for kept in unique
        ):
            continue
        seen_keys.add(key)
        unique.append(chunk)

    near_deduped = dedupe_near_duplicate_chunks(unique)
    removed = before - len(near_deduped)
    return near_deduped, max(0, removed)


def _dedupe_key(chunk: RetrievedChunk) -> str:
    chunk_id = chunk.chunk_id or chunk.id
    if chunk.document_id and chunk_id:
        return f"{chunk.document_id}:{chunk_id}"
    if chunk.document_id is not None and chunk.start_offset is not None:
        return f"{chunk.document_id}:{chunk.start_offset}:{chunk.end_offset}"
    return f"{chunk.document_id}:{chunk.chunk_index}:{hash((chunk.text or '')[:240])}"


def pack_evidence_by_budget(
    chunks: list[RetrievedChunk],
    *,
    counter: TokenCounter,
    legal_budget: int,
    conversation_budget: int,
    matter_budget: int,
) -> tuple[list[RetrievedChunk], dict[str, int], list[str]]:
    """
    Pack ranked evidence into per-source budgets.

    Prefers complete passages. Truncates only as a last resort at sentence
    boundaries, preserving document_id / chunk_id / offsets / citation metadata.
    """
    reasons: list[str] = []
    deduped, removed = remove_duplicate_evidence(chunks)
    if removed:
        reasons.append(f"removed_{removed}_duplicate_chunks")

    ranked = sorted(
        enumerate(deduped),
        key=lambda pair: score_evidence_chunk(pair[1]),
        reverse=True,
    )

    budgets = {
        SourceType.LEGAL.value: max(0, legal_budget),
        SourceType.CONVERSATION.value: max(0, conversation_budget),
        SourceType.MATTER.value: max(0, matter_budget),
    }
    used = {
        SourceType.LEGAL.value: 0,
        SourceType.CONVERSATION.value: 0,
        SourceType.MATTER.value: 0,
    }
    selected: dict[int, RetrievedChunk] = {}
    docs_seen: set[str] = set()

    # First pass: keep highest-scoring complete chunks under budget.
    for original_index, chunk in ranked:
        source = (chunk.source_type or SourceType.LEGAL.value).lower()
        if source not in budgets:
            source = SourceType.LEGAL.value
        remaining = budgets[source] - used[source]
        if remaining <= 0:
            continue

        tokens = counter.count(chunk.text)
        if tokens <= remaining:
            selected[original_index] = chunk
            used[source] += tokens
            if chunk.document_id:
                docs_seen.add(chunk.document_id)
            continue

        # Prefer skipping low-value leftovers over shredding text, unless this
        # is the only remaining evidence for a new high-value document.
        doc_id = chunk.document_id or ""
        must_keep = bool(doc_id) and doc_id not in docs_seen and score_evidence_chunk(chunk) >= 0.55
        if not must_keep and tokens > remaining * 1.5:
            continue

        truncated = truncate_passage_to_tokens(
            chunk,
            counter=counter,
            max_tokens=remaining,
        )
        if truncated is None:
            continue
        selected[original_index] = truncated
        used[source] += counter.count(truncated.text)
        if chunk.document_id:
            docs_seen.add(chunk.document_id)
        reasons.append("truncated_passage_to_fit_budget")

    packed = [selected[i] for i in sorted(selected)]
    if len(packed) < len(deduped):
        reasons.append(
            f"dropped_{len(deduped) - len(packed)}_chunks_for_token_budget"
        )

    return packed, used, reasons


def truncate_passage_to_tokens(
    chunk: RetrievedChunk,
    *,
    counter: TokenCounter,
    max_tokens: int,
) -> RetrievedChunk | None:
    """Truncate at sentence boundaries; keep citation/source metadata."""
    if max_tokens <= 0:
        return None
    text = chunk.text or ""
    if not text:
        return None
    if counter.count(text) <= max_tokens:
        return chunk

    sentences = _SENTENCE_SPLIT.split(text.strip())
    kept: list[str] = []
    for sentence in sentences:
        candidate = " ".join(kept + [sentence]).strip()
        if counter.count(candidate) > max_tokens:
            break
        kept.append(sentence)

    if not kept:
        # Absolute last resort: keep a prefix that fits.
        words = text.split()
        lo, hi = 0, len(words)
        best = ""
        while lo < hi:
            mid = (lo + hi + 1) // 2
            candidate = " ".join(words[:mid])
            if counter.count(candidate) <= max_tokens:
                best = candidate
                lo = mid
            else:
                hi = mid - 1
        if not best:
            return None
        kept_text = best
    else:
        kept_text = " ".join(kept).strip()

    packed = deepcopy(chunk)
    packed.text = kept_text
    # Offsets stay as original span metadata for citation validity.
    return packed


def compress_conversation_history(
    history: list[Message],
    *,
    counter: TokenCounter,
    budget: int,
    conversation_summary: str | None = None,
) -> tuple[list[Message], str | None, str | None, list[str]]:
    """
    Fit history into budget using recent messages + summary + legal context.

    Does not blindly keep last-N regardless of size.
    """
    reasons: list[str] = []
    if budget <= 0 or not history:
        summary = conversation_summary
        active = extract_active_legal_context(history)
        return [], summary, active, (["history_budget_exhausted"] if history else [])

    active = extract_active_legal_context(history)
    summary = conversation_summary
    remaining = budget

    # Reserve room for summary + active legal issue when present.
    summary_block = None
    if summary:
        summary_tokens = counter.count(summary)
        if summary_tokens <= max(32, remaining // 4):
            summary_block = summary
            remaining -= summary_tokens
        else:
            reasons.append("conversation_summary_omitted_for_budget")

    active_block = None
    if active:
        active_tokens = counter.count(active)
        if active_tokens <= max(24, remaining // 5):
            active_block = active
            remaining -= active_tokens
        else:
            reasons.append("active_legal_context_omitted_for_budget")

    # Prefer most recent messages; keep pairs when possible.
    selected_rev: list[Message] = []
    for message in reversed(history):
        cost = counter.count(message.content) + 2  # role overhead
        if cost > remaining:
            # Try a compressed single-line form for older assistant turns.
            if message.role == "assistant" and remaining >= 24:
                compressed = _compress_message(message, counter, remaining)
                if compressed is not None:
                    selected_rev.append(compressed)
                    remaining -= counter.count(compressed.content) + 2
                    reasons.append("compressed_older_assistant_message")
            continue
        selected_rev.append(message)
        remaining -= cost

    selected = list(reversed(selected_rev))
    if len(selected) < len(history):
        reasons.append(
            f"trimmed_history_{len(history) - len(selected)}_of_{len(history)}"
        )

    return selected, summary_block, active_block, reasons


def extract_active_legal_context(history: list[Message]) -> str | None:
    """Preserve the current legal issue when history must shrink."""
    if not history:
        return None
    hits: list[str] = []
    for message in reversed(history[-8:]):
        text = (message.content or "").strip()
        if not text:
            continue
        match = _LEGAL_ISSUE_RE.search(text)
        if match:
            snippet = " ".join(text.split())
            if len(snippet) > 220:
                snippet = snippet[:217] + "..."
            hits.append(f"{message.role}: {snippet}")
        if len(hits) >= 2:
            break
    if not hits:
        return None
    return "Active legal context:\n" + "\n".join(reversed(hits))


def _compress_message(
    message: Message,
    counter: TokenCounter,
    max_tokens: int,
) -> Message | None:
    text = " ".join((message.content or "").split())
    if not text:
        return None
    # Keep legal issue cues if present.
    match = _LEGAL_ISSUE_RE.search(text)
    if match:
        start = max(0, match.start() - 40)
        text = text[start : start + 280]
    else:
        text = text[:280]
    prefix = "[compressed] "
    while text and counter.count(prefix + text) > max_tokens:
        text = text[: max(0, len(text) - 40)]
    if not text:
        return None
    return Message(role=message.role, content=prefix + text, created_at=message.created_at)


def group_chunks_by_document(
    chunks: list[RetrievedChunk],
) -> dict[str, list[RetrievedChunk]]:
    """One document_id → list of evidence chunks (resource identity)."""
    grouped: dict[str, list[RetrievedChunk]] = {}
    for chunk in chunks:
        key = chunk.document_id or f"__anon__:{chunk.chunk_id or chunk.id}"
        grouped.setdefault(key, []).append(chunk)
    return grouped
