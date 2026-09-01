from __future__ import annotations

from app.core.config import settings


def assert_vector_size(
    vector: list[float],
    *,
    provider: str,
) -> None:
    expected = settings.embedding_output_dimensions
    qdrant_size = settings.VECTOR_SIZE
    actual = len(vector)
    if actual == expected:
        return

    hint = (
        f"Embedding model returned {actual}-d vectors (expected {expected}-d). "
    )
    if actual != qdrant_size:
        hint += (
            f"Qdrant legal_documents is indexed at {qdrant_size}-d nomic. "
            "Use a nomic embedding provider for library search, or re-index Qdrant "
            f"after setting VECTOR_SIZE={actual}."
        )
    elif provider == "openrouter":
        hint += (
            "For free OpenRouter embed + existing Qdrant library, use remote Ollama "
            "nomic or nomic/fireworks API — Nemotron embed is a different vector space."
        )
    else:
        hint += (
            "Use EMBEDDING_DIMENSIONS or a nomic-compatible provider for Qdrant search."
        )

    raise RuntimeError(
        f"Embedding dimension mismatch ({actual} != {expected}). {hint}"
    )
