from __future__ import annotations

from app.core.config import settings
from app.embeddings.base import BaseEmbedding
from app.embeddings.huggingface import HuggingFaceEmbedding
from app.embeddings.nomic import NomicEmbedding
from app.embeddings.ollama import OllamaEmbedding
from app.embeddings.online_policy import online_only_embedding_error
from app.embeddings.openai_compatible import OpenAICompatibleEmbedding


class EmbeddingFactory:
    """
    Creates embedding provider implementations.

    Server without local models: EMBEDDING_ONLINE_ONLY=true plus either a cloud
    embed API (fireworks, nomic) or remote Ollama (EMBEDDING_URL on another host).
    """

    @staticmethod
    def create() -> BaseEmbedding:
        provider = (settings.EMBEDDING_PROVIDER or "ollama").lower().strip()

        if settings.EMBEDDING_ONLINE_ONLY and settings.embedding_runs_locally:
            raise ValueError(online_only_embedding_error(provider))

        match provider:
            case "ollama":
                return OllamaEmbedding()

            case "openrouter":
                return OpenAICompatibleEmbedding(provider_name="openrouter")

            case "fireworks":
                return OpenAICompatibleEmbedding(provider_name="fireworks")

            case "openai" | "openai_compatible":
                return OpenAICompatibleEmbedding(provider_name=provider)

            case "nomic":
                return NomicEmbedding()

            case "huggingface" | "hf":
                return HuggingFaceEmbedding()

            case _:
                raise ValueError(
                    f"Unsupported embedding provider: {provider}"
                )
