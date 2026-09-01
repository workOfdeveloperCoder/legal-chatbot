from __future__ import annotations

from app.schemas.contracts import (
    ClauseCard,
    ClauseExtractionResponse,
    ClauseStatus,
    PlaybookReviewResponse,
    PlaybookVerdict,
    RedlineResponse,
    ReviewTableResponse,
)

_STATUS_MARK = {
    ClauseStatus.FOUND: "",
    ClauseStatus.MISSING: " — not found",
    ClauseStatus.UNCLEAR: " — unclear",
    ClauseStatus.ERROR: " — error",
}


def format_clause_cards(extraction: ClauseExtractionResponse) -> str:
    lines = [
        f"## Clause cards — {extraction.filename}",
        "",
    ]
    for card in extraction.cards:
        lines.append(_card_line(card))
        if card.quote:
            lines.append(f"  > {card.quote}")
    if extraction.truncated:
        lines.append("")
        lines.append(
            "_Document was truncated for extraction; later pages may not "
            "be reflected._"
        )
    return "\n".join(lines).strip()


def format_review_table(table: ReviewTableResponse) -> str:
    headers = ["Document"] + [column.label for column in table.columns]
    rows: list[list[str]] = [headers]
    for row in table.rows:
        cells = {cell.key: cell for cell in row.cells}
        values = [_pipe_safe(row.filename)]
        for column in table.columns:
            cell = cells.get(column.key)
            if cell is None:
                values.append("—")
                continue
            if cell.status == ClauseStatus.MISSING:
                values.append("—")
            elif cell.status == ClauseStatus.ERROR:
                values.append("error")
            else:
                values.append(_pipe_safe(cell.value or cell.status.value))
        rows.append(values)

    widths = [
        max(len(row[index]) for row in rows)
        for index in range(len(headers))
    ]
    rendered = [
        "| "
        + " | ".join(
            row[index].ljust(widths[index]) for index in range(len(headers))
        )
        + " |"
        for row in rows
    ]
    separator = (
        "| "
        + " | ".join("-" * widths[index] for index in range(len(headers)))
        + " |"
    )
    rendered.insert(1, separator)
    note = ""
    if table.truncated_documents:
        note = (
            f"\n\nShowing {len(table.rows)} of {table.document_count} "
            "documents (newest first)."
        )
    return (
        "## Review table\n\n"
        + "\n".join(rendered)
        + note
    )


def _card_line(card: ClauseCard) -> str:
    mark = _STATUS_MARK.get(card.status, "")
    if card.status == ClauseStatus.FOUND and card.value:
        return f"- **{card.label}** — {card.value}"
    if card.value:
        return f"- **{card.label}** — {card.value}{mark}"
    return f"- **{card.label}**{mark or ' — not found'}"


def _pipe_safe(value: str) -> str:
    return value.replace("|", "/").replace("\n", " ").strip()[:80]


_VERDICT_MARK = {
    PlaybookVerdict.PASS: "pass",
    PlaybookVerdict.FALLBACK: "fallback",
    PlaybookVerdict.MISSING: "missing",
    PlaybookVerdict.OFF_PLAYBOOK: "off-playbook",
    PlaybookVerdict.UNCLEAR: "unclear",
    PlaybookVerdict.NOT_APPLICABLE: "n/a",
}


def format_playbook_review(review: PlaybookReviewResponse) -> str:
    summary = review.summary
    lines = [
        f"## Playbook review — {review.filename}",
        f"_{review.playbook_name}_",
        "",
        (
            f"Pass {summary.pass_count} · Fallback {summary.fallback_count} · "
            f"Missing {summary.missing_count} · Off-playbook "
            f"{summary.off_playbook_count} · Blockers {summary.blocker_count}"
        ),
        "",
    ]
    for finding in review.findings:
        mark = _VERDICT_MARK.get(finding.verdict, finding.verdict.value)
        lines.append(f"- **{finding.label}** — {mark}")
        if finding.extracted_value:
            lines.append(f"  Extracted: {finding.extracted_value}")
        lines.append(f"  {finding.reason}")
        if (
            finding.verdict
            in {
                PlaybookVerdict.MISSING,
                PlaybookVerdict.OFF_PLAYBOOK,
                PlaybookVerdict.FALLBACK,
            }
            and finding.suggested_language
        ):
            lines.append(f"  Suggested: {finding.suggested_language}")
    return "\n".join(lines).strip()


def format_redline(redline: RedlineResponse) -> str:
    title = (
        "Playbook redline" if redline.mode == "playbook" else "Redline"
    )
    lines = [
        f"## {title}",
        (
            f"{redline.left_filename or 'Current'} → "
            f"{redline.right_filename or 'Proposed'}"
        ),
        "",
    ]
    if redline.unchanged:
        lines.append("No differences.")
        return "\n".join(lines).strip()
    for hunk in redline.hunks:
        lines.append(f"### {hunk.label}")
        lines.append(hunk.markdown or "(no text)")
        lines.append("")
    return "\n".join(lines).strip()
