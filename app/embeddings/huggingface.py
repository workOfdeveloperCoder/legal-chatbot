from __future__ import annotations

import asyncio
import os
from functools import lru_cache

from fastembed import TextEmbedding

from app.core.config import settings
from app.embeddings.base import BaseEmbedding
from app.embeddings.nomic_format import nomic_task_text
from app.embeddings.validation import assert_vector_size


@lru_cache(maxsize=1)
def _text_embedding_model(model_name: str) -> TextEmbedding:
    api_key = settings.embedding_api_key_for_provider("huggingface")
    if api_key:
        os.environ.setdefault("HF_TOKEN", api_key)
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", api_key)
    return TextEmbedding(model_name)


class HuggingFaceEmbedding(BaseEmbedding):
    """
    Local nomic-embed-text via fastembed + Hugging Face Hub download.

    HF serverless inference does not host nomic-embed-text-v1.5, so we use
    your HF token only to download the ONNX model once. No Ollama daemon.
    """

    def __init__(self) -> None:
        self._model_name = settings.EMBEDDING_MODEL
        self._model: TextEmbedding | None = None
        self._lock = asyncio.Lock()

    async def _get_model(self) -> TextEmbedding:
        if self._model is not None:
            return self._model
        async with self._lock:
            if self._model is None:
                self._model = await asyncio.to_thread(
                    _text_embedding_model,
                    self._model_name,
                )
        return self._model

    async def _embed(
        self,
        texts: list[str],
        *,
        task_type: str,
    ) -> list[list[float]]:
        model = await self._get_model()
        payload = [
            nomic_task_text(text, task_type=task_type)
            for text in texts
        ]
        vectors = await asyncio.to_thread(
            lambda: [
                vector.tolist()
                for vector in model.embed(payload)
            ],
        )
        if len(vectors) != len(texts):
            raise RuntimeError(
                "fastembed returned an unexpected embedding count."
            )
        for vector in vectors:
            assert_vector_size(vector, provider="huggingface")
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
        return None
