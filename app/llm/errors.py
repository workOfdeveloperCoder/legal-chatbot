from __future__ import annotations


class LLMError(Exception):
    """Base LLM failure."""

    def __init__(self, message: str, *, permanent: bool = False) -> None:
        super().__init__(message)
        self.permanent = permanent


class LLMTransientError(LLMError):
    """Retryable failure (timeout, 429, 5xx, connection)."""

    def __init__(self, message: str) -> None:
        super().__init__(message, permanent=False)


class LLMPermanentError(LLMError):
    """Non-retryable failure (auth, bad request, malformed response)."""

    def __init__(self, message: str) -> None:
        super().__init__(message, permanent=True)


class LLMCancelledError(LLMError):
    """Request cancelled by client or shutdown."""

    def __init__(self, message: str = "LLM request cancelled") -> None:
        super().__init__(message, permanent=True)


class LLMAllProvidersFailed(LLMError):
    """Primary and fallback providers failed."""

    def __init__(self, message: str) -> None:
        super().__init__(message, permanent=True)
