from __future__ import annotations

import re
from typing import Final


class ConversationTitleService:
    """
    Generates short, meaningful conversation titles.

    This service is deterministic and does not call an LLM.

    Examples
    --------
    What is Section 302 PPC?
        -> Section 302 PPC

    Draft a legal notice for cheque dishonour
        -> Draft Legal Notice For Cheque Dishonour

    Can landlord evict tenant in Punjab?
        -> Landlord Evict Tenant Punjab
    """

    DEFAULT_TITLE: Final[str] = "New Conversation"

    MAX_TITLE_LENGTH: Final[int] = 60

    QUESTION_PREFIXES: Final[tuple[str, ...]] = (
        "what is",
        "what are",
        "how to",
        "how do i",
        "how can i",
        "can i",
        "can you",
        "could you",
        "would you",
        "please",
        "kindly",
        "tell me",
        "explain",
        "give me",
        "show me",
        "help me",
    )

    STOP_WORDS: Final[set[str]] = {
        "a",
        "an",
        "the",
        "of",
        "about",
        "regarding",
        "please",
        "kindly",
        "me",
        "my",
        "is",
        "are",
        "was",
        "were",
    }

    LEGAL_TERMS: Final[dict[str, str]] = {
        "ppc": "PPC",
        "crpc": "CrPC",
        "cpc": "CPC",
        "qso": "QSO",
        "fir": "FIR",
        "firs": "FIRs",
        "constitution": "Constitution",
    }

    def generate(self, message: str | None) -> str:
        """
        Generate a conversation title.

        Parameters
        ----------
        message:
            First user message.

        Returns
        -------
        str
            Clean title suitable for sidebar display.
        """

        if not message:
            return self.DEFAULT_TITLE

        text = self._normalize(message)

        if not text:
            return self.DEFAULT_TITLE

        text = self._remove_prefix(text)

        text = self._replace_legal_terms(text)

        text = self._remove_stop_words(text)

        text = self._cleanup(text)

        if not text:
            return self.DEFAULT_TITLE

        return self._truncate(text)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _normalize(self, text: str) -> str:

        text = text.replace("\n", " ")

        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def _remove_prefix(self, text: str) -> str:

        lower = text.lower()

        for prefix in self.QUESTION_PREFIXES:

            if lower.startswith(prefix):

                return text[len(prefix):].strip()

        return text

    def _replace_legal_terms(self, text: str) -> str:

        words: list[str] = []

        for word in text.split():

            punctuation = ""

            if word and word[-1] in ",.;:!?":

                punctuation = word[-1]

                word = word[:-1]

            replacement = self.LEGAL_TERMS.get(
                word.lower(),
                word,
            )

            words.append(replacement + punctuation)

        return " ".join(words)

    def _remove_stop_words(self, text: str) -> str:

        words: list[str] = []

        for word in text.split():

            if word.lower() not in self.STOP_WORDS:

                words.append(word)

        return " ".join(words)

    def _cleanup(self, text: str) -> str:

        text = re.sub(
            r"[^\w\s/\-]",
            "",
            text,
        )

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip().title()

    def _truncate(self, text: str) -> str:

        if len(text) <= self.MAX_TITLE_LENGTH:

            return text

        truncated = text[: self.MAX_TITLE_LENGTH]

        last_space = truncated.rfind(" ")

        if last_space > 20:

            truncated = truncated[:last_space]

        return truncated.rstrip() + "..."

    # ------------------------------------------------------------------
    # Public Utility
    # ------------------------------------------------------------------

    def regenerate(
        self,
        title: str,
        fallback: str,
    ) -> str:
        """
        Regenerate a title if current one is invalid.
        """

        if not title:

            return self.generate(fallback)

        if title == self.DEFAULT_TITLE:

            return self.generate(fallback)

        return title