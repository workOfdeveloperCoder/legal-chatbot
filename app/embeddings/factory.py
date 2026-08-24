from __future__ import annotations

from app.core.config import settings
from app.embeddings.base import BaseEmbedding
from app.embeddings.ollama import OllamaEmbedding


class EmbeddingFactory:
    """
    Creates embedding provider implementations.

    Chat LLM provider is independent of embeddings. The public legal corpus
    and user document collections are 768-d nomic vectors, so embeddings
    stay on local Ollama even when chat uses an online model.
    """

    @staticmethod
    def create() -> BaseEmbedding:
        provider = (settings.EMBEDDING_PROVIDER or "ollama").lower().strip()

        match provider:
            case "ollama":
                return OllamaEmbedding()

            case _:
                raise ValueError(
                    f"Unsupported embedding provider: {provider}"
                )
