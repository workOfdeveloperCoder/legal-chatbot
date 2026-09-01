from __future__ import annotations

import re

# Individual markers: [Source 1]
_SOURCE_INLINE = re.compile(r"\s*\[Source\s+\d+\]", re.IGNORECASE)

# Combined markers: [Source 1, Source 2] / [Source 1 and Source 2] / [Sources 1, 2]
_SOURCE_COMBINED = re.compile(
    r"\s*\[Sources?\s+\d+(?:\s*(?:,|and|&)\s*(?:Source\s+)?\d+)+\]",
    re.IGNORECASE,
)

# Parenthetical variants: (Source 1) / (Sources 1, 2)
_SOURCE_PAREN = re.compile(
    r"\s*\(Sources?\s+\d+(?:\s*(?:,|and|&)\s*(?:Source\s+)?\d+)*\)",
    re.IGNORECASE,
)

# Trailing Sources / References / Citations blocks with markers or bare bullets
_SOURCES_TRAILING = re.compile(
    r"\n+\s*(?:#{1,3}\s*)?(?:Sources?|References?|Citations?)\s*:?\s*"
    r"(?:\n\s*(?:[-*•]\s*)?(?:\[Sources?\s+[^\]]+\]|\(Sources?\s+[^)]+\)|[^\n]*))*\s*$",
    re.IGNORECASE | re.MULTILINE,
)

_SOURCES_BLOCK = re.compile(
    r"\n+\s*(?:#{1,3}\s*)?(?:Sources?|References?|Citations?)\s*:?\s*"
    r"\n(?:\s*(?:[-*•]\s*)?(?:\[Sources?\s+[^\]]+\]|\(Sources?\s+[^)]+\)|[^\S\n]*)\s*\n?)+",
    re.IGNORECASE,
)

_SOURCES_TRAILING_BARE = re.compile(
    r"\n+\s*(?:#{1,3}\s*)?(?:Sources?|References?|Citations?)\s*:?\s*"
    r"(?:\n\s*[-*•]\s*)*\s*$",
    re.IGNORECASE,
)

# Inline "Sources:" mid-sentence leftovers after stripping
_SOURCES_INLINE_LABEL = re.compile(
    r"(?:^|\n)\s*(?:#{1,3}\s*)?(?:Sources?|References?|Citations?)\s*:?\s*$",
    re.IGNORECASE | re.MULTILINE,
)

_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "to",
        "of",
        "in",
        "on",
        "or",
        "is",
        "be",
        "by",
        "as",
        "at",
        "if",
        "it",
        "for",
        "and",
        "not",
        "no",
        "so",
        "do",
        "we",
        "he",
        "she",
        "they",
        "me",
        "my",
        "our",
        "you",
        "your",
        "can",
        "may",
        "did",
        "was",
        "are",
        "has",
        "had",
        "but",
        "who",
        "how",
        "why",
        "any",
        "all",
    }
)

_KNOWN_SPLITS = frozenset(
    {
        ("opt", "ion"),
        ("app", "eal"),
        ("pe", "titi"),
        ("titi", "on"),
        ("le", "ave"),
        ("ha", "ve"),
        ("ans", "wer"),
        ("a", "nswe"),
        ("answe", "r"),
        ("nswe", "r"),
        ("s", "upp"),
        ("supp", "orting"),
        ("eviden", "ce"),
        ("evi", "denc"),
        ("denc", "e"),
        ("evidenc", "e"),
        ("den", "ce"),
        ("ev", "i"),
        ("rel", "ati"),
        ("ati", "ve"),
        ("ve", "s"),
        ("victi", "m"),
        ("sec", "tion"),
        ("arti", "cle"),
        ("t", "o"),
    }
)


def _is_stop_token(token: str) -> bool:
    if len(token) == 1 and token.isupper():
        return False
    return token.lower() in _STOP_WORDS


def repair_fragmented_words(text: str) -> str:
    """
    Collapse stream/R1 artifacts that insert spaces inside words.

    Example: "Close rel ati ve s" → "Close relatives"
    Leaves normal phrases like "to file a petition" alone.
    """
    if not text or " " not in text:
        return text

    def _pass(blob: str) -> str:
        rough = re.split(r"(\s+)", blob)
        parts: list[str] = []
        for piece in rough:
            if not piece or piece.isspace() or piece.isalpha():
                parts.append(piece)
                continue
            match = re.match(r"^([A-Za-z]+)([^A-Za-z]+)$", piece)
            if match:
                parts.append(match.group(1))
                parts.append(match.group(2))
            else:
                parts.append(piece)

        out: list[str] = []
        i = 0
        n = len(parts)

        while i < n:
            part = parts[i]
            if not part.isalpha():
                out.append(part)
                i += 1
                continue

            run = [part]
            j = i
            while j + 2 < n and parts[j + 1] == " " and parts[j + 2].isalpha():
                nxt = parts[j + 2]
                if _is_stop_token(nxt):
                    if (
                        nxt.lower() == "on"
                        and len(run) >= 2
                        and all(len(token) <= 4 for token in run)
                    ):
                        run.append(nxt)
                        j += 2
                    break
                if _is_stop_token(run[0]) and len(run) == 1:
                    break
                if len(nxt) >= 4 and len(run) >= 2:
                    break
                if len(nxt) > 4:
                    break
                if len(nxt) == 1:
                    # Trailing crumb only for long fragment runs, or "A nswe r".
                    if len(run) >= 3:
                        run.append(nxt)
                        j += 2
                    elif (
                        len(run) == 2
                        and len(run[0]) == 1
                        and run[0].isupper()
                        and len(run[1]) >= 3
                    ):
                        run.append(nxt)
                        j += 2
                    break
                run.append(nxt)
                j += 2

            if (
                len(run) >= 3
                and all(len(token) <= 4 for token in run)
                and sum(1 for token in run if len(token) <= 3) >= len(run) - 1
            ):
                out.append("".join(run))
                i = j + 1
                continue

            if (
                i + 2 < n
                and parts[i + 1] == " "
                and parts[i + 2].isalpha()
                and not _is_stop_token(part)
                and not _is_stop_token(parts[i + 2])
            ):
                nxt = parts[i + 2]
                key = (part.lower(), nxt.lower())
                merge = False
                if key in _KNOWN_SPLITS:
                    merge = True
                elif len(part) <= 2 and len(nxt) <= 3:
                    merge = True
                elif 3 <= len(part) <= 5 and len(nxt) <= 2 and nxt.islower():
                    merge = True
                elif (
                    len(part) == 1
                    and part.isupper()
                    and nxt.islower()
                    and 3 <= len(nxt) <= 4
                ):
                    merge = True
                if merge:
                    out.append(part + nxt)
                    i += 3
                    continue

            out.append(part)
            i += 1

        return "".join(out)

    repaired = text
    for _ in range(4):
        nxt = _pass(repaired)
        if nxt == repaired:
            break
        repaired = nxt
    return repaired


def clean_answer_for_display(answer: str) -> str:
    """
    Remove LLM source marker text from the answer body.

    Structured sources are returned separately in ChatResponse.sources[] /
    resources[]. The UI renders clickable resource cards — not static markers.
    """
    if not answer:
        return ""

    text = answer.strip()
    text = _SOURCES_TRAILING.sub("", text)
    text = _SOURCES_BLOCK.sub("", text)
    text = _SOURCES_TRAILING_BARE.sub("", text)
    text = _SOURCE_COMBINED.sub("", text)
    text = _SOURCE_PAREN.sub("", text)
    text = _SOURCE_INLINE.sub("", text)
    text = _SOURCES_INLINE_LABEL.sub("", text)
    # Soft repair for local-R1 answers that omit spaces between tokens.
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([A-Za-z])(\d)", r"\1 \2", text)
    # Soft repair for stream joins / R1 crumbs that split words with spaces.
    text = repair_fragmented_words(text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\s+([.,;:])", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
