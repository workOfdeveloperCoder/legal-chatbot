from __future__ import annotations

from app.core.config import settings
from app.embeddings.base import BaseEmbedding
from app.embeddings.ollama import OllamaEmbedding


class EmbeddingFactory:
    """
    Creates embedding provider implementations.

    This isolates provider selection from the rest
    of the application.

    Future providers:
    - OpenAI
    - VoyageAI
    - SentenceTransformers
    - Gemini
    """

    @staticmethod
    def create() -> BaseEmbedding:
        provider = settings.LLM_PROVIDER.lower()

        match provider:
            case "ollama":
                return OllamaEmbedding()

            case _:
                raise ValueError(
                    f"Unsupported embedding provider: {provider}"
                )