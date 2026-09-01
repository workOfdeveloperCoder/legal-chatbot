from __future__ import annotations

import httpx

from app.core.config import settings
from app.embeddings.base import BaseEmbedding
from app.embeddings.validation import assert_vector_size


class NomicEmbedding(BaseEmbedding):
    """
    Nomic Atlas text embeddings (768-d nomic-embed-text).

    Matches the existing legal_documents Qdrant index without local Ollama.
    """

    def __init__(self) -> None:
        api_key = settings.embedding_api_key_for_provider("nomic")
        if not api_key:
            raise ValueError(
                "Nomic embeddings require NOMIC_API_KEY or EMBEDDING_API_KEY."
            )

        timeout = httpx.Timeout(
            connect=float(settings.EMBEDDING_CONNECT_TIMEOUT),
            read=float(settings.EMBEDDING_TIMEOUT),
            write=float(settings.EMBEDDING_CONNECT_TIMEOUT),
            pool=float(settings.EMBEDDING_CONNECT_TIMEOUT),
        )
        self._client = httpx.AsyncClient(
            base_url=settings.embedding_base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )
        self._model = settings.EMBEDDING_MODEL

    async def _embed(
        self,
        texts: list[str],
        *,
        task_type: str,
    ) -> list[list[float]]:
        response = await self._client.post(
            "/embedding/text",
            json={
                "model": self._model,
                "texts": texts,
                "task_type": task_type,
            },
        )
        response.raise_for_status()
        data = response.json()
        embeddings = data.get("embeddings")
        if not isinstance(embeddings, list) or not embeddings:
            raise RuntimeError("Nomic API returned no embeddings.")
        vectors = [
            [float(value) for value in row]
            for row in embeddings
            if isinstance(row, list)
        ]
        if len(vectors) != len(texts):
            raise RuntimeError("Nomic API returned an unexpected embedding count.")
        for vector in vectors:
            assert_vector_size(vector, provider="nomic")
        return vectors

    async def embed_query(
        self,
        text: str,
    ) -> list[float]:
        return (await self._embed([text], task_type="search_query"))[0]

    async def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        if not texts:
            return []
        return await self._embed(texts, task_type="search_document")

    async def aclose(self) -> None:
        await self._client.aclose()
