from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Protocol

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


class TokenCounter(Protocol):
    """Counts tokens for budgeting. Prefer real tokenizers over estimators."""

    @property
    def tokenizer_id(self) -> str: ...

    @property
    def is_exact(self) -> bool: ...

    def count(self, text: str | None) -> int: ...


class FallbackTokenEstimator:
    """
    Heuristic estimator used only when a real tokenizer is unavailable.

    Uses punctuation-aware word pieces (~0.75 tokens/word average for English
    legal prose), not raw character length.
    """

    def __init__(self, tokenizer_id: str = "fallback_word_estimator") -> None:
        self._tokenizer_id = tokenizer_id

    @property
    def tokenizer_id(self) -> str:
        return self._tokenizer_id

    @property
    def is_exact(self) -> bool:
        return False

    def count(self, text: str | None) -> int:
        if not text:
            return 0
        pieces = _WORD_RE.findall(text)
        if not pieces:
            return 0
        # Over-estimate so packing stays under the real model window.
        return max(1, int(round(len(pieces) * 1.35)))


class TiktokenCounter:
    """Exact (or near-exact) counting via tiktoken encodings."""

    def __init__(self, encoding_name: str, encoding) -> None:
        self._tokenizer_id = encoding_name
        self._encoding = encoding

    @property
    def tokenizer_id(self) -> str:
        return self._tokenizer_id

    @property
    def is_exact(self) -> bool:
        return True

    def count(self, text: str | None) -> int:
        if not text:
            return 0
        return len(self._encoding.encode(text))


class FixedTokenCounter:
    """Deterministic counter for unit tests (1 token per whitespace-split)."""

    def __init__(self, tokenizer_id: str = "test_whitespace") -> None:
        self._tokenizer_id = tokenizer_id

    @property
    def tokenizer_id(self) -> str:
        return self._tokenizer_id

    @property
    def is_exact(self) -> bool:
        return True

    def count(self, text: str | None) -> int:
        if not text or not text.strip():
            return 0
        return len(text.split())


class ScaledTokenCounter:
    """Inflates counts from an inexact estimator so packing stays conservative."""

    def __init__(self, inner: TokenCounter, factor: float) -> None:
        self._inner = inner
        self._factor = max(1.0, factor)

    @property
    def tokenizer_id(self) -> str:
        return self._inner.tokenizer_id

    @property
    def is_exact(self) -> bool:
        return False

    def count(self, text: str | None) -> int:
        raw = self._inner.count(text)
        return max(raw, int(round(raw * self._factor)))


@lru_cache(maxsize=8)
def _load_tiktoken_encoding(encoding_name: str):
    import tiktoken

    return tiktoken.get_encoding(encoding_name)


def build_token_counter(
    tokenizer_id: str | None = None,
    *,
    enabled: bool = True,
) -> TokenCounter:
    """
    Build the best available counter for a tokenizer id.

    Falls back to FallbackTokenEstimator when tiktoken is missing or
    the encoding cannot be loaded.
    """
    name = (tokenizer_id or "cl100k_base").strip() or "cl100k_base"
    if not enabled:
        return FallbackTokenEstimator(tokenizer_id=f"disabled:{name}")

    if name in {"fallback", "fallback_word_estimator", "estimate"}:
        return FallbackTokenEstimator()

    try:
        encoding = _load_tiktoken_encoding(name)
        return TiktokenCounter(name, encoding)
    except Exception as exc:  # noqa: BLE001 — soft fallback is intentional
        logger.warning(
            "Tokenizer %s unavailable (%s); using conservative fallback estimator",
            name,
            exc,
        )
        return FallbackTokenEstimator(tokenizer_id=f"fallback_after:{name}")
