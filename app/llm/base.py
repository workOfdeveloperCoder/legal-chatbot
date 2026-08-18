from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from app.llm.model_capabilities import ModelCapabilities
from app.schemas.llm import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)


class BaseLLM(ABC):
    """Provider adapter interface — application code depends on this only."""

    provider_name: str = "unknown"
    model_name: str = "unknown"

    @abstractmethod
    async def generate(
        self,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        raise NotImplementedError

    async def generate_stream(
        self,
        request: ChatCompletionRequest,
    ) -> AsyncIterator[str]:
        """
        Stream final user-facing answer tokens only.

        Default: fall back to non-streaming generate (no chain-of-thought).
        Providers may override with native streaming.
        """
        response = await self.generate(request)
        if response.content:
            yield response.content

    def capabilities(self) -> ModelCapabilities | None:
        return None

    async def aclose(self) -> None:
        """Release HTTP clients / connections."""
        return None

    def provider_metadata(self) -> dict[str, Any]:
        return {
            "provider": self.provider_name,
            "model": self.model_name,
        }
