from __future__ import annotations

from app.embeddings.factory import EmbeddingFactory
from app.embeddings.huggingface import HuggingFaceEmbedding
from app.embeddings.nomic import NomicEmbedding
from app.embeddings.openai_compatible import OpenAICompatibleEmbedding
from app.embeddings.nomic_format import nomic_task_text


def test_online_only_blocks_local_providers(monkeypatch):
    from app.core import config

    monkeypatch.setattr(config.settings, "EMBEDDING_ONLINE_ONLY", True)
    monkeypatch.setattr(config.settings, "EMBEDDING_PROVIDER", "huggingface")
    monkeypatch.setattr(config.settings, "EMBEDDING_URL", None)

    try:
        EmbeddingFactory.create()
    except ValueError as exc:
        assert "EMBEDDING_ONLINE_ONLY" in str(exc) or "downloads" in str(exc)
    else:
        raise AssertionError("expected online-only guard")


def test_online_only_allows_remote_ollama(monkeypatch):
    from app.core import config
    from app.embeddings.ollama import OllamaEmbedding

    monkeypatch.setattr(config.settings, "EMBEDDING_ONLINE_ONLY", True)
    monkeypatch.setattr(config.settings, "EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setattr(config.settings, "EMBEDDING_URL", "http://10.0.0.5:11434")

    embedding = EmbeddingFactory.create()
    assert isinstance(embedding, OllamaEmbedding)


def test_openrouter_embedding_factory(monkeypatch):
    from app.core import config

    monkeypatch.setattr(config.settings, "EMBEDDING_PROVIDER", "openrouter")
    monkeypatch.setattr(config.settings, "EMBEDDING_URL", None)
    monkeypatch.setattr(config.settings, "LLM_API_KEY", "or-test-key")
    monkeypatch.setattr(
        config.settings,
        "EMBEDDING_MODEL",
        "openai/text-embedding-3-small",
    )

    embedding = EmbeddingFactory.create()
    assert isinstance(embedding, OpenAICompatibleEmbedding)
    assert embedding._provider == "openrouter"
    assert str(embedding._client.base_url).rstrip("/") == (
        "https://openrouter.ai/api/v1"
    )


def test_fireworks_embedding_factory(monkeypatch):
    from app.core import config

    monkeypatch.setattr(config.settings, "EMBEDDING_PROVIDER", "fireworks")
    monkeypatch.setattr(config.settings, "EMBEDDING_URL", None)
    monkeypatch.setattr(config.settings, "EMBEDDING_ONLINE_ONLY", False)
    monkeypatch.setattr(config.settings, "FIREWORKS_API_KEY", "fw-test-key")
    monkeypatch.setattr(config.settings, "LLM_API_KEY", None)
    monkeypatch.setattr(
        config.settings,
        "EMBEDDING_MODEL",
        "nomic-ai/nomic-embed-text-v1.5",
    )

    embedding = EmbeddingFactory.create()
    assert isinstance(embedding, OpenAICompatibleEmbedding)
    assert embedding._provider == "fireworks"
    assert str(embedding._client.base_url).rstrip("/") == (
        "https://api.fireworks.ai/inference/v1"
    )


def test_nomic_payload_adds_prefix_and_dimensions(monkeypatch):
    from app.core import config

    monkeypatch.setattr(
        config.settings,
        "EMBEDDING_MODEL",
        "nomic-ai/nomic-embed-text-v1.5",
    )
    monkeypatch.setattr(config.settings, "VECTOR_SIZE", 768)

    embedding = OpenAICompatibleEmbedding(provider_name="fireworks")
    payload = embedding._payload(
        texts=["section 417"],
        input_type="search_query",
    )
    assert payload["input"] == "search_query: section 417"
    assert payload["dimensions"] == 768


def test_huggingface_embedding_factory(monkeypatch):
    from app.core import config

    monkeypatch.setattr(config.settings, "EMBEDDING_PROVIDER", "huggingface")
    monkeypatch.setattr(config.settings, "EMBEDDING_ONLINE_ONLY", False)
    monkeypatch.setattr(config.settings, "HUGGINGFACE_API_KEY", "hf-test-key")
    monkeypatch.setattr(
        config.settings,
        "EMBEDDING_MODEL",
        "nomic-ai/nomic-embed-text-v1.5",
    )

    embedding = EmbeddingFactory.create()
    assert isinstance(embedding, HuggingFaceEmbedding)
    assert embedding._model_name == "nomic-ai/nomic-embed-text-v1.5"


def test_nomic_task_prefix():
    assert nomic_task_text("section 417", task_type="search_query") == (
        "search_query: section 417"
    )
    assert nomic_task_text(
        "search_query: already",
        task_type="search_query",
    ) == "search_query: already"


def test_nomic_embedding_factory(monkeypatch):
    from app.core import config

    monkeypatch.setattr(config.settings, "EMBEDDING_PROVIDER", "nomic")
    monkeypatch.setattr(config.settings, "EMBEDDING_URL", None)
    monkeypatch.setattr(config.settings, "NOMIC_API_KEY", "nomic-test-key")
    monkeypatch.setattr(
        config.settings,
        "EMBEDDING_MODEL",
        "nomic-embed-text-v1.5",
    )

    embedding = EmbeddingFactory.create()
    assert isinstance(embedding, NomicEmbedding)
    assert str(embedding._client.base_url).rstrip("/") == (
        "https://api-atlas.nomic.ai/v1"
    )


def test_embedding_api_key_prefers_embedding_api_key(monkeypatch):
    from app.core import config

    monkeypatch.setattr(config.settings, "EMBEDDING_API_KEY", "embed-key")
    monkeypatch.setattr(config.settings, "NOMIC_API_KEY", "nomic-key")
    assert config.settings.embedding_api_key_for_provider("nomic") == "embed-key"


def test_assert_vector_size_mismatch():
    from app.embeddings.validation import assert_vector_size
    from app.core import config

    monkeypatch_dims = config.settings.EMBEDDING_DIMENSIONS
    try:
        config.settings.EMBEDDING_DIMENSIONS = 1536
        try:
            assert_vector_size([0.0] * 1536, provider="openrouter")
        except RuntimeError:
            raise AssertionError("expected success at configured dimension")
        try:
            assert_vector_size([0.0] * 768, provider="openrouter")
        except RuntimeError as exc:
            assert "768 != 1536" in str(exc)
        else:
            raise AssertionError("expected dimension mismatch")
    finally:
        config.settings.EMBEDDING_DIMENSIONS = monkeypatch_dims
