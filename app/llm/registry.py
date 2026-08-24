from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from app.llm.base import BaseLLM

T = TypeVar("T", bound=type[BaseLLM])

ConnectFn = Callable[..., BaseLLM]


class LLMAdapterRegistry:
    """
    Plugin registry for chat LLM adapters.

    Application code never instantiates OpenAI/Ollama/Anthropic directly.
    To add a provider:

        @LLMAdapterRegistry.register(
            "my_provider",
            online=True,
            default_url="https://api.example.com/v1",
        )
        class MyProviderLLM(BaseLLM):
            @classmethod
            def connect(cls, *, provider, base_url, model, api_key, timeout_seconds):
                return cls(...)
    """

    _adapters: dict[str, type[BaseLLM]] = {}
    _online: dict[str, bool] = {}
    _default_urls: dict[str, str] = {}

    @classmethod
    def register(
        cls,
        *names: str,
        online: bool = False,
        default_url: str | None = None,
        default_urls: dict[str, str] | None = None,
    ) -> Callable[[T], T]:
        if not names:
            raise ValueError("LLMAdapterRegistry.register requires at least one name")

        urls = dict(default_urls or {})
        if default_url:
            for name in names:
                urls.setdefault(name, default_url)

        def decorator(adapter_cls: T) -> T:
            if not isinstance(adapter_cls, type) or not issubclass(
                adapter_cls, BaseLLM
            ):
                raise TypeError(
                    "LLM adapters must subclass BaseLLM"
                )
            for name in names:
                key = name.lower().strip()
                cls._adapters[key] = adapter_cls
                cls._online[key] = online
                if key in urls and urls[key]:
                    cls._default_urls[key] = urls[key].rstrip("/")
            return adapter_cls

        return decorator

    @classmethod
    def knows(cls, provider: str | None) -> bool:
        if not provider:
            return False
        return provider.lower().strip() in cls._adapters

    @classmethod
    def is_online(cls, provider: str | None) -> bool | None:
        if not provider:
            return None
        key = provider.lower().strip()
        if key not in cls._online:
            return None
        return cls._online[key]

    @classmethod
    def default_url(cls, provider: str | None) -> str | None:
        if not provider:
            return None
        return cls._default_urls.get(provider.lower().strip())

    @classmethod
    def registered_names(cls) -> list[str]:
        return sorted(cls._adapters)

    @classmethod
    def connect(
        cls,
        *,
        provider: str,
        base_url: str,
        model: str,
        api_key: str | None,
        timeout_seconds: int,
    ) -> BaseLLM:
        key = provider.lower().strip()
        adapter_cls = cls._adapters.get(key)
        if adapter_cls is None:
            known = ", ".join(cls.registered_names()) or "(none loaded)"
            raise ValueError(
                f"Unsupported LLM provider {provider!r}. "
                f"Registered adapters: {known}. "
                "Subclass BaseLLM and decorate with "
                "@LLMAdapterRegistry.register('name', online=True)."
            )
        connect = getattr(adapter_cls, "connect", None)
        if callable(connect) and connect is not BaseLLM.connect:
            return connect(
                provider=key,
                base_url=base_url,
                model=model,
                api_key=api_key,
                timeout_seconds=timeout_seconds,
            )
        return adapter_cls()  # type: ignore[call-arg]
