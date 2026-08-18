from __future__ import annotations

from dataclasses import dataclass

from app.core.config import settings


@dataclass(frozen=True, slots=True)
class PipelineVersions:
    """Traceability for generated answers."""

    model: str
    provider: str
    prompt_version: str
    retrieval_version: str
    reranker_version: str
    embedding_version: str

    def to_dict(self) -> dict[str, str]:
        return {
            "model": str(self.model),
            "provider": str(self.provider),
            "prompt_version": str(self.prompt_version),
            "retrieval_version": str(self.retrieval_version),
            "reranker_version": str(self.reranker_version),
            "embedding_version": str(self.embedding_version),
        }


def current_pipeline_versions(
    *,
    model: str | None = None,
    provider: str | None = None,
) -> PipelineVersions:
    return PipelineVersions(
        model=model or settings.CHAT_MODEL,
        provider=(provider or settings.LLM_PROVIDER).lower(),
        prompt_version=settings.PROMPT_VERSION,
        retrieval_version=settings.RETRIEVAL_VERSION,
        reranker_version=settings.RERANKER_VERSION,
        embedding_version=settings.EMBEDDING_VERSION,
    )
