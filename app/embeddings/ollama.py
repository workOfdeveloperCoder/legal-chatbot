from __future__ import annotations

import httpx

from app.core.config import settings
from app.embeddings.base import BaseEmbedding


class OllamaEmbedding(BaseEmbedding):
    """
    Ollama embedding provider.

    Uses:
        POST /api/embed

    Docs:
        https://ollama.com/library/nomic-embed-text
    """

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=settings.embedding_base_url,
            timeout=httpx.Timeout(
                connect=300,   # wait to connect to Ollama (5 min)
                read=3600,     # wait for response generation (60 min)
                write=300,     # upload request body (5 min)
                pool=300,      # connection pool wait (5 min)
            ),
        )

        self._model = settings.EMBEDDING_MODEL

    async def embed_query(
        self,
        text: str,
    ) -> list[float]:

        response = await self._client.post(
            "/api/embed",
            json={
                "model": self._model,
                "input": text,
            },
        )

        response.raise_for_status()

        data = response.json()

        embeddings = data.get("embeddings")

        if not embeddings:
            raise RuntimeError(
                "Ollama returned no embeddings."
            )

        return embeddings[0]

    async def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:

        if not texts:
            return []

        response = await self._client.post(
            "/api/embed",
            json={
                "model": self._model,
                "input": texts,
            },
        )

        response.raise_for_status()

        data = response.json()

        embeddings = data.get("embeddings")

        if embeddings is None:
            raise RuntimeError(
                "Ollama returned no embeddings."
            )

        return embeddings

    async def aclose(self) -> None:
        await self._client.aclose()