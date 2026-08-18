from __future__ import annotations

from app.embeddings.base import BaseEmbedding
from app.embeddings.factory import EmbeddingFactory


class EmbeddingService:
    """
    Application service for embedding generation.

    The rest of the application should depend only on this
    service instead of concrete embedding providers.

    Responsibilities
    ----------------
    - Hide provider implementation
    - Generate query embeddings
    - Generate document embeddings
    """

    def __init__(
        self,
        embedding: BaseEmbedding | None = None,
    ) -> None:

        self._embedding = embedding or EmbeddingFactory.create()

    async def embed_query(
        self,
        text: str,
    ) -> list[float]:
        """
        Generate an embedding for a user query.
        """

        text = text.strip()

        if not text:
            raise ValueError(
                "Query text cannot be empty."
            )

        return await self._embedding.embed_query(
            text=text,
        )

    async def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple documents.
        Empty documents are ignored.
        """

        documents = [
            text.strip()
            for text in texts
            if text and text.strip()
        ]

        if not documents:
            return []

        return await self._embedding.embed_documents(
            texts=documents,
        )

    async def close(self) -> None:
        """
        Release provider resources if supported.
        """

        close = getattr(
            self._embedding,
            "aclose",
            None,
        )

        if callable(close):
            await close()