from __future__ import annotations

from app.embeddings.factory import EmbeddingFactory
from app.embeddings.ollama import OllamaEmbedding
from app.llm.factory import LLMFactory
from app.llm.model_capabilities import ModelCapabilityRegistry
from app.llm.openai_compatible import OpenAICompatibleLLM
from app.llm.provider_config import (
    effective_llm_timeout,
    is_online_provider,
    resolve_chat_model,
    resolve_llm_base_url,
)
from app.schemas.llm import ChatCompletionRequest, ChatMessage


def test_openai_uses_hosted_url_when_llm_url_is_still_local():
    url = resolve_llm_base_url("openai", "http://localhost:11434")
    assert url == "https://api.openai.com/v1"
    assert resolve_llm_base_url("deepseek", "http://127.0.0.1:11434") == (
        "https://api.deepseek.com/v1"
    )
    assert resolve_llm_base_url(
        "openai_compatible",
        "https://proxy.example/v1",
    ) == "https://proxy.example/v1"


def test_local_ollama_tag_is_remapped_for_openai():
    assert resolve_chat_model("openai", "deepseek-r1:32b") == "gpt-4o-mini"
    assert resolve_chat_model("deepseek", "deepseek-r1:32b") == "deepseek-reasoner"
    assert resolve_chat_model("ollama", "deepseek-r1:32b") == "deepseek-r1:32b"
    assert resolve_chat_model("openai", "gpt-4o-mini") == "gpt-4o-mini"


def test_online_timeout_does_not_inherit_local_30_minute_wait():
    assert effective_llm_timeout("openai", 1800) == 180
    assert effective_llm_timeout("openai", 90) == 90
    assert effective_llm_timeout("ollama", 1800) == 1800


def test_factory_builds_openai_adapter_against_hosted_api():
    llm = LLMFactory.create_raw(
        provider="openai",
        base_url="http://localhost:11434",
        model="deepseek-r1:32b",
        api_key="sk-test",
    )
    assert isinstance(llm, OpenAICompatibleLLM)
    assert llm.provider_name == "openai"
    assert llm.model_name == "gpt-4o-mini"
    assert str(llm.client.base_url).rstrip("/") == "https://api.openai.com/v1"


def test_embeddings_stay_on_ollama_when_chat_is_online(monkeypatch):
    from app.core import config

    monkeypatch.setattr(config.settings, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(config.settings, "CHAT_MODEL", "gpt-4o-mini")
    monkeypatch.setattr(config.settings, "EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setattr(config.settings, "EMBEDDING_URL", None)
    monkeypatch.setattr(config.settings, "LLM_URL", "https://api.openai.com/v1")

    embedding = EmbeddingFactory.create()
    assert isinstance(embedding, OllamaEmbedding)
    assert str(embedding._client.base_url).rstrip("/") == "http://localhost:11434"


def test_gpt4o_mini_profile_has_cursor_style_window():
    caps = ModelCapabilityRegistry.resolve(
        provider="openai",
        model_name="gpt-4o-mini",
    )
    assert caps.context_window == 128000
    assert caps.tokenizer_id == "o200k_base"
    assert caps.returns_usage is True
    assert is_online_provider(caps.provider)


def test_reasoning_models_use_max_completion_tokens():
    llm = OpenAICompatibleLLM(
        base_url="https://api.openai.com/v1",
        api_key="sk-test",
        model="o4-mini",
        provider_name="openai",
    )
    payload = llm._completion_payload(
        ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="hello")],
            temperature=0.1,
            max_tokens=1024,
        ),
        stream=False,
    )
    assert "max_completion_tokens" in payload
    assert "max_tokens" not in payload
    assert "temperature" not in payload
    assert payload["model"] == "o4-mini"


def test_canonical_llm_api_key_wins_for_any_provider(monkeypatch):
    from app.core import config

    monkeypatch.setattr(config.settings, "LLM_API_KEY", "canonical-online-key")
    monkeypatch.setattr(config.settings, "GOOGLE_API_KEY", "google-alias")
    monkeypatch.setattr(config.settings, "OPENAI_API_KEY", "openai-alias")
    monkeypatch.setattr(config.settings, "ANTHROPIC_API_KEY", "anthropic-alias")
    assert config.settings.api_key_for_provider("gemini") == "canonical-online-key"
    assert config.settings.api_key_for_provider("openai") == "canonical-online-key"
    assert config.settings.api_key_for_provider("anthropic") == "canonical-online-key"
    assert config.settings.api_key_for_provider("deepseek") == "canonical-online-key"


def test_blank_canonical_key_falls_back_to_vendor_alias(monkeypatch):
    from app.core import config

    monkeypatch.setattr(config.settings, "LLM_API_KEY", "  ")
    monkeypatch.setattr(config.settings, "GOOGLE_API_KEY", "google-alias")
    monkeypatch.setattr(config.settings, "GEMINI_API_KEY", None)
    assert config.settings.api_key_for_provider("gemini") == "google-alias"
    assert config.settings.api_key_for_provider("openai") is None


def test_default_chat_model_names_for_online_providers():
    from app.llm.provider_config import default_chat_model, resolve_chat_model

    assert default_chat_model("gemini") == "gemini-3.6-flash"
    assert default_chat_model("openai") == "gpt-4o-mini"
    assert default_chat_model("anthropic") == "claude-sonnet-4-5"
    assert resolve_chat_model("gemini", "") == "gemini-3.6-flash"
    assert resolve_chat_model("groq", "deepseek-r1:32b") == "llama-3.3-70b-versatile"


def test_gpt4o_payload_keeps_max_tokens_and_temperature():
    llm = OpenAICompatibleLLM(
        base_url="https://api.openai.com/v1",
        api_key="sk-test",
        model="gpt-4o-mini",
        provider_name="openai",
    )
    payload = llm._completion_payload(
        ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="hello")],
            temperature=0.1,
            max_tokens=2048,
        ),
        stream=False,
    )
    assert payload["max_tokens"] == 2048
    assert payload["temperature"] == 0.1
    assert "max_completion_tokens" not in payload
