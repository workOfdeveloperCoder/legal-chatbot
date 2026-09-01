from __future__ import annotations

import httpx
import pytest

from app.contracts.json_parse import parse_json_object
from app.llm.errors import LLMPermanentError, LLMTransientError
from app.llm.factory import LLMFactory
from app.llm.gateway import LLMGateway
from app.llm.ollama import OllamaLLM
from app.llm.openrouter import OpenRouterLLM
from app.schemas.llm import ChatCompletionRequest, ChatMessage


def _request(**kwargs) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="What is section 54-C?")],
        temperature=0.1,
        max_tokens=256,
        **kwargs,
    )


def _openrouter(**kwargs) -> OpenRouterLLM:
    return OpenRouterLLM(
        base_url="https://openrouter.ai/api/v1",
        api_key="sk-or-test",
        model="openrouter/free",
        timeout_seconds=5,
        connect_timeout=2,
        **kwargs,
    )


async def _swap_transport(llm, handler) -> None:
    base = str(getattr(llm, "_base_url", "") or llm.client.base_url)
    await llm.aclose()
    llm.client = httpx.AsyncClient(
        base_url=base,
        transport=httpx.MockTransport(handler),
        timeout=httpx.Timeout(5.0),
    )


def test_factory_selects_openrouter_adapter():
    llm = LLMFactory.create_raw(
        provider="openrouter",
        base_url="http://localhost:11434",
        model="openrouter/free",
        api_key="sk-or-test",
    )
    assert isinstance(llm, OpenRouterLLM)
    assert llm.provider_name == "openrouter"
    assert llm.model_name == "openrouter/free"
    assert str(llm.client.base_url).rstrip("/") == "https://openrouter.ai/api/v1"
    assert "openrouter" in LLMFactory.registered_providers()
    assert "ollama" in LLMFactory.registered_providers()


def test_factory_keeps_openrouter_free_model_suffix():
    llm = LLMFactory.create_raw(
        provider="openrouter",
        model="meta-llama/llama-3.2-3b-instruct:free",
        api_key="sk-or-test",
    )
    assert llm.model_name == "meta-llama/llama-3.2-3b-instruct:free"


def test_factory_selects_ollama_adapter():
    llm = LLMFactory.create_raw(
        provider="ollama",
        base_url="http://127.0.0.1:11434",
        model="deepseek-r1:32b",
        api_key=None,
    )
    assert isinstance(llm, OllamaLLM)
    assert llm.provider_name == "ollama"
    assert llm.model_name == "deepseek-r1:32b"


def test_create_wraps_openrouter_in_gateway(monkeypatch):
    from app.core import config

    monkeypatch.setattr(config.settings, "LLM_PROVIDER", "openrouter")
    monkeypatch.setattr(config.settings, "OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setattr(config.settings, "OPENROUTER_MODEL", "openrouter/free")
    monkeypatch.setattr(config.settings, "OPENROUTER_BASE_URL", None)
    monkeypatch.setattr(config.settings, "LLM_API_KEY", None)
    monkeypatch.setattr(config.settings, "LLM_FALLBACK_PROVIDER", None)
    llm = LLMFactory.create()
    assert isinstance(llm, LLMGateway)
    assert isinstance(llm._primary, OpenRouterLLM)
    assert llm._primary.model_name == "openrouter/free"


def test_openrouter_model_env_overrides_chat_model(monkeypatch):
    from app.core import config

    monkeypatch.setattr(config.settings, "LLM_PROVIDER", "openrouter")
    monkeypatch.setattr(config.settings, "CHAT_MODEL", "deepseek-r1:32b")
    monkeypatch.setattr(config.settings, "OPENROUTER_MODEL", "openrouter/free")
    llm = LLMFactory.create_raw(provider="openrouter", api_key="sk-or-test")
    assert llm.model_name == "openrouter/free"


def test_ollama_model_env_overrides_chat_model(monkeypatch):
    from app.core import config

    monkeypatch.setattr(config.settings, "LLM_PROVIDER", "ollama")
    monkeypatch.setattr(config.settings, "CHAT_MODEL", "other-local")
    monkeypatch.setattr(config.settings, "OLLAMA_MODEL", "deepseek-r1:32b")
    llm = LLMFactory.create_raw(provider="ollama")
    assert isinstance(llm, OllamaLLM)
    assert llm.model_name == "deepseek-r1:32b"


def test_openrouter_base_url_env(monkeypatch):
    from app.core import config

    monkeypatch.setattr(
        config.settings,
        "OPENROUTER_BASE_URL",
        "https://openrouter.ai/api/v1",
    )
    llm = LLMFactory.create_raw(
        provider="openrouter",
        model="openrouter/free",
        api_key="sk-or-test",
    )
    assert str(llm.client.base_url).rstrip("/") == "https://openrouter.ai/api/v1"


def test_missing_openrouter_api_key_fails_fast():
    with pytest.raises(LLMPermanentError, match="API key"):
        LLMFactory.create_raw(
            provider="openrouter",
            model="openrouter/free",
            api_key="",
        )


def test_openrouter_constructor_requires_api_key():
    with pytest.raises(LLMPermanentError, match="OPENROUTER_API_KEY"):
        OpenRouterLLM(
            base_url="https://openrouter.ai/api/v1",
            api_key="",
            model="openrouter/free",
        )


def test_invalid_provider():
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        LLMFactory.create_raw(provider="not_a_real_vendor", api_key="k")


def test_openrouter_json_mode_payload_and_no_stream_options():
    llm = _openrouter()
    payload = llm._completion_payload(_request(json_mode=True), stream=True)
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["model"] == "openrouter/free"
    assert payload["temperature"] == 0.1
    assert payload["max_tokens"] == 256
    assert "max_completion_tokens" not in payload
    assert "stream_options" not in payload


def test_openrouter_alias_key_used_when_canonical_blank(monkeypatch):
    from app.core import config

    monkeypatch.setattr(config.settings, "LLM_API_KEY", "  ")
    monkeypatch.setattr(config.settings, "OPENROUTER_API_KEY", "or-alias-key")
    assert config.settings.api_key_for_provider("openrouter") == "or-alias-key"


@pytest.mark.asyncio
async def test_openrouter_generate():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/chat/completions")
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": "Section 54-C restricts injunctions."}}
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 8,
                    "total_tokens": 18,
                },
            },
        )

    llm = _openrouter()
    await _swap_transport(llm, handler)
    try:
        result = await llm.generate(_request())
        assert "54-C" in result.content
        assert result.total_tokens == 18
    finally:
        await llm.aclose()


@pytest.mark.asyncio
async def test_openrouter_stream():
    def handler(request: httpx.Request) -> httpx.Response:
        body = (
            b'data: {"choices":[{"delta":{"content":"Hello "}}]}\n\n'
            b'data: {"choices":[{"delta":{"content":"world"}}]}\n\n'
            b"data: [DONE]\n\n"
        )
        return httpx.Response(200, content=body)

    llm = _openrouter()
    await _swap_transport(llm, handler)
    try:
        chunks: list[str] = []
        async for piece in llm.generate_stream(_request()):
            chunks.append(piece)
        assert "".join(chunks) == "Hello world"
    finally:
        await llm.aclose()


@pytest.mark.asyncio
async def test_openrouter_stream_normalizes_through_gateway():
    def handler(request: httpx.Request) -> httpx.Response:
        body = (
            b'data: {"choices":[{"delta":{"content":"token"}}]}\n\n'
            b"data: [DONE]\n\n"
        )
        return httpx.Response(200, content=body)

    llm = _openrouter()
    await _swap_transport(llm, handler)
    gateway = LLMGateway(llm, max_retries=0, timeout_seconds=5)
    try:
        pieces = []
        async for delta in gateway.generate_stream(_request()):
            pieces.append(delta)
        assert pieces
        assert all(getattr(item, "kind", None) == "token" for item in pieces)
        assert "".join(item.text for item in pieces) == "token"
    finally:
        await gateway.aclose()


@pytest.mark.asyncio
async def test_openrouter_generate_uses_reasoning_when_content_null():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "reasoning": "Section 417 PPC is cheating.",
                        }
                    }
                ]
            },
        )

    llm = _openrouter()
    await _swap_transport(llm, handler)
    try:
        result = await llm.generate(_request())
        assert "417" in result.content
    finally:
        await llm.aclose()


@pytest.mark.asyncio
async def test_openrouter_stream_yields_reasoning_as_thinking():
    def handler(request: httpx.Request) -> httpx.Response:
        body = (
            b'data: {"choices":[{"delta":{"reasoning":"think "}}]}\n\n'
            b'data: {"choices":[{"delta":{"content":"answer"}}]}\n\n'
            b"data: [DONE]\n\n"
        )
        return httpx.Response(200, content=body)

    llm = _openrouter()
    await _swap_transport(llm, handler)
    gateway = LLMGateway(llm, max_retries=0, timeout_seconds=5)
    try:
        pieces = []
        async for delta in gateway.generate_stream(_request()):
            pieces.append(delta)
        kinds = [item.kind for item in pieces]
        texts = [item.text for item in pieces]
        assert "thinking" in kinds
        assert "token" in kinds
        assert "answer" in "".join(texts)
    finally:
        await gateway.aclose()


@pytest.mark.asyncio
async def test_openrouter_timeout():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out")

    llm = _openrouter()
    await _swap_transport(llm, handler)
    try:
        with pytest.raises(LLMTransientError, match="timeout"):
            await llm.generate(_request())
    finally:
        await llm.aclose()


@pytest.mark.asyncio
async def test_openrouter_rate_limit():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="rate limited")

    llm = _openrouter()
    await _swap_transport(llm, handler)
    try:
        with pytest.raises(LLMTransientError, match="429"):
            await llm.generate(_request())
    finally:
        await llm.aclose()


@pytest.mark.asyncio
async def test_openrouter_auth_error_does_not_include_api_key():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="unauthorized")

    llm = _openrouter()
    await _swap_transport(llm, handler)
    try:
        with pytest.raises(LLMPermanentError, match="auth failed") as exc:
            await llm.generate(_request())
        assert "sk-or-test" not in str(exc.value)
    finally:
        await llm.aclose()


@pytest.mark.asyncio
async def test_openrouter_invalid_model():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="model not found")

    llm = _openrouter()
    await _swap_transport(llm, handler)
    try:
        with pytest.raises(LLMPermanentError, match="404"):
            await llm.generate(_request())
    finally:
        await llm.aclose()


@pytest.mark.asyncio
async def test_openrouter_json_generation():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": '{"party": "Buyer", "term": "12 months"}'}}
                ],
            },
        )

    llm = _openrouter()
    await _swap_transport(llm, handler)
    try:
        result = await llm.generate(_request(json_mode=True))
        parsed = parse_json_object(result.content)
        assert parsed["party"] == "Buyer"
        assert parsed["term"] == "12 months"
    finally:
        await llm.aclose()


@pytest.mark.asyncio
async def test_openrouter_empty_response():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": None}}]},
        )

    llm = _openrouter()
    await _swap_transport(llm, handler)
    try:
        with pytest.raises(LLMPermanentError, match="Empty content"):
            await llm.generate(_request())
    finally:
        await llm.aclose()


@pytest.mark.asyncio
async def test_openrouter_malformed_stream_skips_bad_events():
    def handler(request: httpx.Request) -> httpx.Response:
        body = (
            b"data: not-json\n\n"
            b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
            b"data: [DONE]\n\n"
        )
        return httpx.Response(200, content=body)

    llm = _openrouter()
    await _swap_transport(llm, handler)
    try:
        chunks: list[str] = []
        async for piece in llm.generate_stream(_request()):
            chunks.append(piece)
        assert "".join(chunks) == "ok"
    finally:
        await llm.aclose()


@pytest.mark.asyncio
async def test_ollama_generate_still_works():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/api/chat")
        return httpx.Response(
            200,
            json={
                "message": {"content": "Local DeepSeek answer."},
                "prompt_eval_count": 4,
                "eval_count": 6,
            },
        )

    llm = OllamaLLM(
        base_url="http://127.0.0.1:11434",
        model="deepseek-r1:32b",
        timeout_seconds=5,
        connect_timeout=2,
    )
    await _swap_transport(llm, handler)
    try:
        result = await llm.generate(_request())
        assert "DeepSeek" in result.content
        assert result.total_tokens == 10
    finally:
        await llm.aclose()


@pytest.mark.asyncio
async def test_ollama_stream_still_works():
    def handler(request: httpx.Request) -> httpx.Response:
        body = (
            b'{"message":{"content":"Hi"},"done":false}\n'
            b'{"message":{"content":" there"},"done":true}\n'
        )
        return httpx.Response(200, content=body)

    llm = OllamaLLM(
        base_url="http://127.0.0.1:11434",
        model="deepseek-r1:32b",
        timeout_seconds=5,
        connect_timeout=2,
    )
    await _swap_transport(llm, handler)
    try:
        chunks: list[str] = []
        async for piece in llm.generate_stream(_request()):
            text = piece.text if hasattr(piece, "text") else str(piece)
            chunks.append(text)
        assert "".join(chunks) == "Hi there"
    finally:
        await llm.aclose()
