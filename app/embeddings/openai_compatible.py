from __future__ import annotations

import httpx

from app.core.config import settings
from app.embeddings.base import BaseEmbedding
from app.embeddings.nomic_format import nomic_task_text
from app.embeddings.validation import assert_vector_size


class OpenAICompatibleEmbedding(BaseEmbedding):
    """
    OpenAI-compatible embeddings API (POST /embeddings).

    Used for OpenRouter, OpenAI, and custom OpenAI-compatible hosts.
    """

    def __init__(
        self,
        *,
        provider_name: str = "openai_compatible",
    ) -> None:
        self._provider = provider_name
        self._model = settings.EMBEDDING_MODEL
        headers: dict[str, str] = {}
        api_key = settings.embedding_api_key_for_provider(provider_name)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if provider_name == "openrouter":
            headers["HTTP-Referer"] = "http://localhost"
            headers["X-Title"] = settings.APP_NAME

        timeout = httpx.Timeout(
            connect=float(settings.EMBEDDING_CONNECT_TIMEOUT),
            read=float(settings.EMBEDDING_TIMEOUT),
            write=float(settings.EMBEDDING_CONNECT_TIMEOUT),
            pool=float(settings.EMBEDDING_CONNECT_TIMEOUT),
        )
        self._client = httpx.AsyncClient(
            base_url=settings.embedding_base_url,
            headers=headers,
            timeout=timeout,
        )

    def _prepare_inputs(
        self,
        texts: list[str],
        *,
        input_type: str | None,
    ) -> list[str]:
        if input_type and "nomic" in self._model.lower():
            task_type = (
                "search_query"
                if input_type == "search_query"
                else "search_document"
            )
            return [
                nomic_task_text(text, task_type=task_type)
                for text in texts
            ]
        return texts

    def _payload(
        self,
        *,
        texts: list[str],
        input_type: str | None,
    ) -> dict[str, object]:
        prepared = self._prepare_inputs(texts, input_type=input_type)
        payload: dict[str, object] = {
            "model": self._model,
            "input": prepared[0] if len(prepared) == 1 else prepared,
        }
        if input_type and self._provider == "openrouter":
            payload["input_type"] = input_type
        if "nomic" in self._model.lower():
            payload["dimensions"] = settings.VECTOR_SIZE
        return payload

    def _parse_embeddings(self, data: dict[str, object]) -> list[list[float]]:
        rows = data.get("data")
        if not isinstance(rows, list) or not rows:
            raise RuntimeError("Embeddings API returned no data.")

        ordered = sorted(
            rows,
            key=lambda row: int(row.get("index", 0))
            if isinstance(row, dict)
            else 0,
        )
        vectors: list[list[float]] = []
        for row in ordered:
            if not isinstance(row, dict):
                continue
            embedding = row.get("embedding")
            if not isinstance(embedding, list):
                raise RuntimeError("Embeddings API row missing embedding vector.")
            vectors.append([float(value) for value in embedding])
        if not vectors:
            raise RuntimeError("Embeddings API returned empty vectors.")
        return vectors

    async def embed_query(
        self,
        text: str,
    ) -> list[float]:
        response = await self._client.post(
            "/embeddings",
            json=self._payload(
                texts=[text],
                input_type="search_query",
            ),
        )
        response.raise_for_status()
        vector = self._parse_embeddings(response.json())[0]
        assert_vector_size(vector, provider=self._provider)
        return vector

    async def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        if not texts:
            return []

        response = await self._client.post(
            "/embeddings",
            json=self._payload(
                texts=texts,
                input_type="search_document",
            ),
        )
        response.raise_for_status()
        vectors = self._parse_embeddings(response.json())
        for vector in vectors:
            assert_vector_size(vector, provider=self._provider)
        return vectors

    async def aclose(self) -> None:
        await self._client.aclose()
