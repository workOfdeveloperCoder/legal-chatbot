from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


    # ==================================================
    # Application
    # ==================================================

    APP_NAME: str = "Legal Chatbot API"

    DEBUG: bool = False

    # "production" enables startup checks (Qdrant API key, TRUSTED_HOSTS, LLM key).
    # Leave as "development" for local work even when DEBUG=false.
    ENVIRONMENT: str = "development"

    API_V1_PREFIX: str = "/api/v1"

    CORS_ORIGINS: str = (
        "http://localhost:5173,http://127.0.0.1:5173"
    )

    # Comma-separated hosts. "*" or empty = allow all (local dev only).
    TRUSTED_HOSTS: str = "*"

    @property
    def cors_origins(self) -> list[str]:
        return [
            origin.strip()
            for origin in self.CORS_ORIGINS.split(",")
            if origin.strip()
        ]

    @property
    def trusted_hosts(self) -> list[str]:
        raw = (self.TRUSTED_HOSTS or "").strip()
        if not raw or raw == "*":
            return ["*"]
        return [item.strip() for item in raw.split(",") if item.strip()]



    # ==================================================
    # Security
    # ==================================================

    SECRET_KEY: str

    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    REFRESH_TOKEN_EXPIRE_DAYS: int = 30



    # ==================================================
    # Database
    # ==================================================

    DATABASE_URL: str

    ALEMBIC_DATABASE_URL: str

    DB_POOL_SIZE: int = 10

    DB_MAX_OVERFLOW: int = 20



    # ==================================================
    # Redis
    # ==================================================

    REDIS_URL: str | None = None



    # ==================================================
    # Storage
    # ==================================================

    UPLOAD_DIR: str = "storage/documents"

    MAX_UPLOAD_BYTES: int = 25 * 1024 * 1024

    MAX_REQUEST_BYTES: int = 32 * 1024 * 1024



    # ==================================================
    # Qdrant
    # ==================================================

    QDRANT_HOST: str = "localhost"

    QDRANT_PORT: int = 6333

    # Full URL overrides host/port when set (e.g. https://qdrant.internal:6333).
    QDRANT_URL: str | None = None

    QDRANT_API_KEY: str | None = None

    QDRANT_HTTPS: bool = False

    QDRANT_TIMEOUT: int = 30


    # --------------------------------------------------
    # Public Legal Corpus
    #
    # Owned by Legal GPT project
    # Read only
    # --------------------------------------------------

    LEGAL_QDRANT_COLLECTION: str = (
        "legal_documents"
    )


    # --------------------------------------------------
    # User Uploaded Documents
    #
    # Matter documents
    # Conversation documents
    #
    # Security:
    # user_id filter mandatory
    # --------------------------------------------------

    USER_DOCUMENT_COLLECTION: str = (
        "chatbot_documents"
    )


    # --------------------------------------------------
    # AI Memory
    #
    # User memories
    # Matter memories
    # Conversation memories
    # --------------------------------------------------

    USER_MEMORY_COLLECTION: str = (
        "user_memory"
    )



    # ==================================================
    # Vector Configuration
    # ==================================================

    VECTOR_SIZE: int = 768


    CHUNK_SIZE: int = 1200

    CHUNK_OVERLAP: int = 200



    # ==================================================
    # MinIO / Object Storage
    # ==================================================

    MINIO_ENDPOINT: str | None = None

    MINIO_ACCESS_KEY: str | None = None

    MINIO_SECRET_KEY: str | None = None

    MINIO_BUCKET: str = "legal-documents"



    # ==================================================
    # AI / LLM
    # ==================================================
    #
    # Chat is selected by LLM_PROVIDER. RAG depends on BaseLLM, never on a
    # concrete vendor. Swap adapters in config, not in application code.
    #
    # Local (DeepSeek R1 32B via Ollama):
    #   LLM_PROVIDER=ollama
    #   OLLAMA_MODEL=deepseek-r1:32b
    #   CHAT_MODEL=deepseek-r1:32b
    #   LLM_URL=http://localhost:11434
    #   LLM_TIMEOUT=3600
    #
    # Temporary CPU-server testing (OpenRouter, OpenAI-compatible):
    #   LLM_PROVIDER=openrouter
    #   OPENROUTER_API_KEY=...
    #   OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
    #   OPENROUTER_MODEL=<free-or-paid-model-id>
    #   LLM_TIMEOUT=120
    #
    # Canonical hosted-chat names (any vendor):
    #   LLM_PROVIDER   ollama | openrouter | openai | gemini | anthropic | ...
    #   CHAT_MODEL     vendor model id (defaults per provider if omitted)
    #   LLM_API_KEY    one key for whatever LLM_PROVIDER is set to
    #   LLM_URL        optional; hosted URL is used when this is still localhost
    # Embeddings stay on local nomic so existing 768-d Qdrant collections
    # keep working when chat is switched to a hosted LLM.
    #
    # Online example (Gemini, OpenAI, Claude — same key name):
    #   LLM_PROVIDER=gemini
    #   CHAT_MODEL=gemini-3.6-flash
    #   LLM_API_KEY=...
    #   LLM_TIMEOUT=120

    LLM_PROVIDER: str = "ollama"


    LLM_URL: str = (
        "http://localhost:11434"
    )


    CHAT_MODEL: str = (
        "deepseek-r1:32b"
    )

    # Optional alias used when LLM_PROVIDER=ollama. If unset, CHAT_MODEL wins.
    OLLAMA_MODEL: str | None = None

    EMBEDDING_PROVIDER: str = "ollama"

    EMBEDDING_URL: str | None = None

    EMBEDDING_MODEL: str = (
        "nomic-embed-text:latest"
    )

    EMBEDDING_API_KEY: str | None = None

    NOMIC_API_KEY: str | None = None

    HUGGINGFACE_API_KEY: str | None = None

    HF_TOKEN: str | None = None

    EMBEDDING_TIMEOUT: int = 120

    EMBEDDING_CONNECT_TIMEOUT: int = 30

    # Server deploy: block ollama/huggingface (they need local model files).
    EMBEDDING_ONLINE_ONLY: bool = False

    # Output width of the active embedding model (may differ from Qdrant VECTOR_SIZE).
    EMBEDDING_DIMENSIONS: int | None = None


    # Local DeepSeek R1 32B often needs 20–50 minutes (hidden thinking +
    # answer). Hosted APIs should set LLM_TIMEOUT=120. If this is still
    # 1800/3600 when LLM_PROVIDER is online, it is remapped to 120.
    LLM_TIMEOUT: int = 3600

    LLM_CONNECT_TIMEOUT: int = 60

    # Canonical key for ANY hosted chat LLM. Prefer this over vendor aliases.
    LLM_API_KEY: str | None = None

    OPENAI_API_KEY: str | None = None

    ANTHROPIC_API_KEY: str | None = None

    GOOGLE_API_KEY: str | None = None

    GEMINI_API_KEY: str | None = None

    DEEPSEEK_API_KEY: str | None = None

    GROQ_API_KEY: str | None = None

    OPENROUTER_API_KEY: str | None = None

    FIREWORKS_API_KEY: str | None = None

    OPENROUTER_BASE_URL: str | None = None

    # Must be set when LLM_PROVIDER=openrouter. Do not hardcode a model id
    # in application code — switch free → paid by changing this value.
    OPENROUTER_MODEL: str | None = None

    LLM_MAX_RETRIES: int = 1

    LLM_RETRY_BASE_DELAY_SECONDS: float = 1.0

    LLM_FALLBACK_PROVIDER: str | None = None

    LLM_FALLBACK_URL: str | None = None

    LLM_FALLBACK_MODEL: str | None = None

    LLM_FALLBACK_API_KEY: str | None = None

    LLM_TEMPERATURE: float = 0.1

    LLM_ENABLE_STREAMING: bool = True

    # Hosted models do not need the local R1 two-stage path.
    LLM_TWO_STAGE_ONLINE: bool = False

    # Local Voice Mode (no cloud STT/TTS, no API keys).
    STT_ENGINE: str = "auto"

    STT_MODEL: str = "base"

    STT_DEVICE: str = "cpu"

    STT_COMPUTE_TYPE: str = "int8"

    STT_MODEL_CACHE: str = "models/voice/whisper"

    STT_WHISPER_CPP_BIN: str | None = None

    STT_WHISPER_CPP_MODEL: str | None = None

    STT_MAX_AUDIO_BYTES: int = 8_000_000

    STT_MAX_SECONDS: int = 45

    TTS_ENGINE: str = "auto"

    TTS_PIPER_BIN: str | None = None

    TTS_VOICE_EN: str = "models/voice/en_US-lessac-medium.onnx"

    TTS_VOICE_UR: str | None = None

    TTS_VOICE_PA: str | None = None

    TTS_ESPEAK_BIN: str = "espeak-ng"

    PROMPT_VERSION: str = "legal-v3"

    RETRIEVAL_VERSION: str = "tiered-v2"

    RERANKER_VERSION: str = "hybrid-v1"

    EMBEDDING_VERSION: str = "nomic-embed-text-v1"



    # ==================================================
    # RAG
    # ==================================================

    RETRIEVAL_LIMIT: int = 10

    RETRIEVAL_LEGAL_LIMIT: int = 6

    RETRIEVAL_CONVERSATION_LIMIT: int = 2

    RETRIEVAL_MATTER_LIMIT: int = 2

    RETRIEVAL_WEB_LIMIT: int = 4

    MEMORY_LIMIT: int = 5


    ENABLE_RERANKER: bool = True

    ENABLE_TWO_STAGE_REASONING: bool = True

    # Block programming, illegal how-to, and off-topic chat before the LLM.
    LLM_FIREWALL_ENABLED: bool = True

    # Live internet search (Tavily / Brave / DuckDuckGo).
    WEB_SEARCH_ENABLED: bool = True

    # auto | tavily | brave | duckduckgo
    WEB_SEARCH_PROVIDER: str = "auto"

    WEB_SEARCH_API_KEY: str | None = None

    TAVILY_API_KEY: str | None = None

    BRAVE_API_KEY: str | None = None

    WEB_SEARCH_MAX_RESULTS: int = 5

    WEB_SEARCH_TIMEOUT: float = 8.0

    # Off by default: the chat Web toggle (and explicit "search the internet"
    # phrasing) must be on. Empty corpus must not silently pull DuckDuckGo hits.
    WEB_SEARCH_FALLBACK_ON_INSUFFICIENT: bool = False



    # ==================================================
    # Token / Context Budget
    # ==================================================

    TOKEN_DEFAULT_CONTEXT_WINDOW: int = 32768

    TOKEN_RESERVED_OUTPUT_TOKENS: int = 12288

    TOKEN_SAFETY_MARGIN: int = 512

    TOKEN_MAX_CONVERSATION_TOKENS: int = 2048

    TOKEN_MAX_LEGAL_EVIDENCE_TOKENS: int = 8000

    TOKEN_MAX_MATTER_EVIDENCE_TOKENS: int = 6000

    TOKEN_MAX_CONVERSATION_DOCUMENT_TOKENS: int = 4000

    TOKEN_MAX_WEB_EVIDENCE_TOKENS: int = 1800

    TOKEN_SCAFFOLDING_OVERHEAD: int = 600

    TOKEN_TOKENIZER_ID: str = "cl100k_base"

    TOKEN_COUNTING_ENABLED: bool = True

    TOKEN_COST_TRACKING_ENABLED: bool = True

    TOKEN_USAGE_WARNING_PERCENT: float = 80.0

    # Tighter packing when chat uses a hosted LLM (cost + focus).
    # Reasoning hosts (MiniMax, R1) spend a large share of this on thinking.
    TOKEN_ONLINE_RESERVED_OUTPUT: int = 4096

    TOKEN_ONLINE_SAFETY_MARGIN: int = 256

    TOKEN_ONLINE_SCAFFOLDING: int = 180

    TOKEN_ONLINE_MAX_CONVERSATION_TOKENS: int = 1200

    TOKEN_ONLINE_MAX_LEGAL_EVIDENCE_TOKENS: int = 3500

    TOKEN_ONLINE_MAX_MATTER_EVIDENCE_TOKENS: int = 2500

    TOKEN_ONLINE_MAX_CONVERSATION_DOCUMENT_TOKENS: int = 2000

    TOKEN_ONLINE_MAX_WEB_EVIDENCE_TOKENS: int = 900



    # ==================================================
    # Observability / Logging
    # ==================================================

    LOG_ENABLE_REQUEST_LOGGING: bool = True

    # Off by default: chat bodies are client legal data.
    LOG_ENABLE_RESPONSE_BODY: bool = False

    LOG_STORE_CHAT_BODIES: bool = False

    LOG_BODY_MAX_BYTES: int = 16384

    LOG_SKIP_PATHS: str = "/docs,/openapi.json,/redoc,/health,/ready"

    RATE_LIMIT_ENABLED: bool = True

    RATE_LIMIT_AUTH_PER_MINUTE: int = 30

    RATE_LIMIT_CHAT_PER_MINUTE: int = 20

    RATE_LIMIT_UPLOAD_PER_MINUTE: int = 10

    RATE_LIMIT_DEFAULT_PER_MINUTE: int = 60

    # Only honour X-Forwarded-For / X-Real-IP when Nginx is the sole client
    # of Uvicorn (bind 127.0.0.1). Leave false if the API port is reachable
    # directly, otherwise clients can spoof the rate-limit key.
    TRUST_FORWARDED_FOR: bool = False

    TOKEN_FALLBACK_SAFETY_FACTOR: float = 1.2


    @property
    def resolved_llm_api_key(self) -> str | None:
        from app.llm.provider_config import nonempty_secret

        return nonempty_secret(self.LLM_API_KEY) or self.api_key_for_provider(
            self.LLM_PROVIDER
        )

    @property
    def resolved_web_search_api_key(self) -> str | None:
        from app.llm.provider_config import nonempty_secret

        return (
            nonempty_secret(self.WEB_SEARCH_API_KEY)
            or nonempty_secret(self.TAVILY_API_KEY)
            or nonempty_secret(self.BRAVE_API_KEY)
        )

    def api_key_for_provider(self, provider: str | None) -> str | None:
        from app.llm.provider_config import (
            nonempty_secret,
            provider_api_key_aliases,
        )

        canonical = nonempty_secret(self.LLM_API_KEY)
        if canonical:
            return canonical
        for alias in provider_api_key_aliases(provider):
            value = nonempty_secret(getattr(self, alias, None))
            if value:
                return value
        return None

    def chat_model_for_provider(self, provider: str | None = None) -> str:
        from app.llm.provider_config import nonempty_secret, normalize_provider

        name = normalize_provider(provider or self.LLM_PROVIDER)
        if name == "openrouter":
            specific = nonempty_secret(self.OPENROUTER_MODEL)
            if specific:
                return specific
        if name == "ollama":
            specific = nonempty_secret(self.OLLAMA_MODEL)
            if specific:
                return specific
        return self.CHAT_MODEL

    def base_url_for_provider(self, provider: str | None = None) -> str:
        from app.llm.provider_config import nonempty_secret, normalize_provider

        name = normalize_provider(provider or self.LLM_PROVIDER)
        if name == "openrouter":
            specific = nonempty_secret(self.OPENROUTER_BASE_URL)
            if specific:
                return specific.rstrip("/")
        return self.LLM_URL

    @property
    def embedding_output_dimensions(self) -> int:
        if self.EMBEDDING_DIMENSIONS is not None and self.EMBEDDING_DIMENSIONS > 0:
            return self.EMBEDDING_DIMENSIONS
        model = (self.EMBEDDING_MODEL or "").lower()
        if "nemotron" in model and "embed" in model:
            return 2048
        if "text-embedding-3-small" in model:
            return 1536
        if "gemini-embedding" in model:
            return 3072
        return self.VECTOR_SIZE

    @property
    def qdrant_embedding_compatible(self) -> bool:
        """True when embed model matches the 768-d nomic Qdrant index."""
        model = (self.EMBEDDING_MODEL or "").lower()
        return "nomic" in model

    @property
    def embedding_runs_locally(self) -> bool:
        """True when this host loads/runs the embedding model (not a remote HTTP API)."""
        from app.llm.provider_config import _LOCAL_HOST_MARKERS
        from urllib.parse import urlparse

        provider = (self.EMBEDDING_PROVIDER or "ollama").lower().strip()
        if provider in {"fireworks", "nomic", "openrouter"}:
            return False
        if provider in {"openai", "openai_compatible"}:
            host = (urlparse(self.embedding_base_url).hostname or "").lower()
            return host in _LOCAL_HOST_MARKERS
        if provider in {"huggingface", "hf"}:
            return True
        if provider == "ollama":
            host = (urlparse(self.embedding_base_url).hostname or "").lower()
            return host in _LOCAL_HOST_MARKERS
        return True

    @property
    def embedding_base_url(self) -> str:
        if self.EMBEDDING_URL:
            return self.EMBEDDING_URL.rstrip("/")

        provider = (self.EMBEDDING_PROVIDER or "ollama").lower().strip()
        if provider == "openrouter":
            specific = (self.OPENROUTER_BASE_URL or "").strip()
            if specific:
                return specific.rstrip("/")
            return "https://openrouter.ai/api/v1"
        if provider == "fireworks":
            return "https://api.fireworks.ai/inference/v1"
        if provider == "nomic":
            return "https://api-atlas.nomic.ai/v1"
        if provider in {"openai", "openai_compatible"}:
            return "https://api.openai.com/v1"
        return "http://localhost:11434"

    def embedding_api_key_for_provider(
        self,
        provider: str | None = None,
    ) -> str | None:
        from app.llm.provider_config import nonempty_secret

        specific = nonempty_secret(self.EMBEDDING_API_KEY)
        if specific:
            return specific

        name = (provider or self.EMBEDDING_PROVIDER or "ollama").lower().strip()
        if name == "nomic":
            return nonempty_secret(self.NOMIC_API_KEY)
        if name in {"huggingface", "hf"}:
            return (
                nonempty_secret(self.HUGGINGFACE_API_KEY)
                or nonempty_secret(self.HF_TOKEN)
            )
        if name == "openrouter":
            return self.api_key_for_provider("openrouter")
        if name == "fireworks":
            return self.api_key_for_provider("fireworks")
        if name in {"openai", "openai_compatible"}:
            return self.api_key_for_provider("openai")
        return None

    @property
    def is_online_llm(self) -> bool:
        from app.llm.provider_config import is_online_provider

        return is_online_provider(self.LLM_PROVIDER)

    @property
    def is_production(self) -> bool:
        return (self.ENVIRONMENT or "").strip().lower() == "production"



@lru_cache
def get_settings() -> Settings:

    return Settings()



settings = get_settings()