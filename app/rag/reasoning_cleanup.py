from __future__ import annotations

import re

# DeepSeek R1 wraps internal reasoning in think tags
_THINK_OPEN = chr(60) + "think" + chr(62)
_THINK_CLOSE = chr(60) + "/" + "think" + chr(62)

_THINKING_BLOCK = re.compile(
    re.escape(_THINK_OPEN) + r".*?" + re.escape(_THINK_CLOSE),
    re.DOTALL | re.IGNORECASE,
)

_THINKING_UNCLOSED = re.compile(
    re.escape(_THINK_OPEN) + r".*",
    re.DOTALL | re.IGNORECASE,
)

_INTERNAL_ANALYSIS_HEADER = re.compile(
    r"(?:^|\n)\s*(?:EVIDENCE ANALYSIS|INTERNAL ANALYSIS|REASONING NOTES?)\s*:?\s*\n",
    re.IGNORECASE,
)


def _is_word_end(text: str) -> bool:
    if not text:
        return False
    char = text[-1]
    return char.isalnum() or char in ")]\"'%"


def _is_word_start(text: str) -> bool:
    if not text:
        return False
    char = text[0]
    return char.isalnum() or char in "([\"'"


def _should_insert_space(left: str, right: str) -> bool:
    """
    Insert a space only when both sides look like finished words.

    DeepSeek/Ollama often stream BPE crumbs ("rel","ati","ve"). Blindly
    inserting spaces between alphanumeric pieces produces "rel ati ve".
    Prefer concatenation; only space when the right piece starts a new
    capitalized word after a lowercase end (e.g. "heldThe" → "held The"),
    or when left ends with a closing bracket/quote before a letter.
    """
    if not left or not right:
        return False
    if left[-1].isspace() or right[0].isspace():
        return False
    if not (_is_word_end(left) and _is_word_start(right)):
        return False

    # "held""The" or "PPC""Act" style missing spaces
    if left[-1].islower() and right[0].isupper():
        return True
    if left[-1] in ")]\"'%" and right[0].isalnum():
        return True

    # Never invent spaces between lowercase/digit crumbs — those are subwords.
    return False


def append_stream_text(existing: str, chunk: str) -> str:
    """Join one streamed piece onto the answer without splitting subwords."""
    left = existing or ""
    right = chunk or ""
    if not right:
        return left
    if not left:
        return right
    if _should_insert_space(left, right):
        return f"{left} {right}"
    return left + right


def join_stream_text(chunks: list[str] | tuple[str, ...]) -> str:
    """Assemble streamed tokens without inserting spaces inside words."""
    assembled = ""
    for chunk in chunks:
        assembled = append_stream_text(assembled, chunk)
    return assembled


def strip_reasoning_output(text: str) -> str:
    """
    Remove DeepSeek R1 chain-of-thought and internal analysis blocks.

    Never expose model reasoning to the API consumer.
    Keep internal spaces, newlines, and indentation intact.
    """
    if not text:
        return ""

    def _replace_think_block(match: re.Match[str]) -> str:
        return "\n" if "\n" in match.group(0) else " "

    cleaned = _THINKING_BLOCK.sub(_replace_think_block, text)
    cleaned = _THINKING_UNCLOSED.sub("", cleaned)

    if _INTERNAL_ANALYSIS_HEADER.search(cleaned):
        parts = _INTERNAL_ANALYSIS_HEADER.split(cleaned, maxsplit=1)
        if len(parts) == 2 and parts[1].strip():
            cleaned = parts[1]

    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip("\n")


class ReasoningStreamFilter:
    """
    Strip R1 think-tags from a token stream without dropping the answer.

    Per-chunk strip_reasoning_output treats an unclosed <think> as "drop
    everything remaining in this chunk", so a split tag can erase the
    actual answer or leak reasoning from later chunks.
    """

    def __init__(self) -> None:
        self._buffer = ""
        self._in_think = False

    def feed(self, chunk: str) -> str:
        visible, _thinking = self.feed_parts(chunk)
        return visible

    def feed_parts(self, chunk: str) -> tuple[str, str]:
        if not chunk:
            return "", ""
        self._buffer += chunk
        return self._drain(flush=False)

    def finish(self) -> str:
        visible, _thinking = self._drain(flush=True)
        return visible

    def finish_parts(self) -> tuple[str, str]:
        return self._drain(flush=True)

    def _drain(self, *, flush: bool) -> tuple[str, str]:
        visible: list[str] = []
        thinking: list[str] = []
        open_tag = _THINK_OPEN
        close_tag = _THINK_CLOSE
        while self._buffer:
            lowered = self._buffer.lower()
            if self._in_think:
                close_at = lowered.find(close_tag)
                if close_at < 0:
                    hold = len(close_tag) - 1
                    if flush:
                        thinking.append(self._buffer)
                        self._buffer = ""
                    elif len(self._buffer) > hold:
                        thinking.append(self._buffer[:-hold])
                        self._buffer = self._buffer[-hold:]
                    break
                thinking.append(self._buffer[:close_at])
                self._buffer = self._buffer[close_at + len(close_tag) :]
                self._in_think = False
                continue
            open_at = lowered.find(open_tag)
            if open_at < 0:
                hold = len(open_tag) - 1
                if flush or len(self._buffer) <= hold:
                    if flush:
                        visible.append(self._buffer)
                        self._buffer = ""
                    break
                visible.append(self._buffer[:-hold])
                self._buffer = self._buffer[-hold:]
                break
            if open_at > 0:
                visible.append(self._buffer[:open_at])
            self._buffer = self._buffer[open_at + len(open_tag) :]
            self._in_think = True
        return "".join(visible), "".join(thinking)
