from __future__ import annotations

from abc import ABC, abstractmethod


class BaseEmbedding(ABC):
    """
    Base interface for embedding providers.

    Implementations:
    - Ollama
    - OpenAI
    - VoyageAI
    - SentenceTransformers
    """

    @abstractmethod
    async def embed_query(
        self,
        text: str,
    ) -> list[float]:
        """
        Generate an embedding for a single query.
        """
        raise NotImplementedError

    @abstractmethod
    async def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple documents.
        """
        raise NotImplementedError
    