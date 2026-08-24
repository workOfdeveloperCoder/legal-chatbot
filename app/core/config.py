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

    API_V1_PREFIX: str = "/api/v1"

    CORS_ORIGINS: str = (
        "http://localhost:5173,http://127.0.0.1:5173"
    )

    @property
    def cors_origins(self) -> list[str]:
        return [
            origin.strip()
            for origin in self.CORS_ORIGINS.split(",")
            if origin.strip()
        ]



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



    # ==================================================
    # Redis
    # ==================================================

    REDIS_URL: str | None = None



    # ==================================================
    # Storage
    # ==================================================

    UPLOAD_DIR: str = "storage/documents"



    # ==================================================
    # Qdrant
    # ==================================================

    QDRANT_HOST: str = "localhost"

    QDRANT_PORT: int = 6333


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
    # Chat can be local (Ollama) or any hosted LLM.
    # Canonical env names for hosted chat (any vendor):
    #   LLM_PROVIDER   openai | gemini | anthropic | deepseek | groq | ...
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


    EMBEDDING_PROVIDER: str = "ollama"

    EMBEDDING_URL: str | None = None

    EMBEDDING_MODEL: str = (
        "nomic-embed-text:latest"
    )


    # Local DeepSeek R1 32B often needs many minutes per answer
    # (especially with two-stage reasoning). Online APIs can override lower.
    LLM_TIMEOUT: int = 1800

    LLM_CONNECT_TIMEOUT: int = 60

    # Canonical key for ANY hosted chat LLM. Prefer this over vendor aliases.
    LLM_API_KEY: str | None = None

    OPENAI_API_KEY: str | None = None

    ANTHROPIC_API_KEY: str | None = None

    GOOGLE_API_KEY: str | None = None

    GEMINI_API_KEY: str | None = None

    DEEPSEEK_API_KEY: str | None = None

    GROQ_API_KEY: str | None = None

    LLM_MAX_RETRIES: int = 1

    LLM_RETRY_BASE_DELAY_SECONDS: float = 1.0

    LLM_FALLBACK_PROVIDER: str | None = None

    LLM_FALLBACK_URL: str | None = None

    LLM_FALLBACK_MODEL: str | None = None

    LLM_FALLBACK_API_KEY: str | None = None

    LLM_TEMPERATURE: float = 0.1

    LLM_ENABLE_STREAMING: bool = False

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

    MEMORY_LIMIT: int = 5


    ENABLE_RERANKER: bool = True

    ENABLE_TWO_STAGE_REASONING: bool = True



    # ==================================================
    # Token / Context Budget
    # ==================================================

    TOKEN_DEFAULT_CONTEXT_WINDOW: int = 32768

    TOKEN_RESERVED_OUTPUT_TOKENS: int = 4096

    TOKEN_SAFETY_MARGIN: int = 512

    TOKEN_MAX_CONVERSATION_TOKENS: int = 2048

    TOKEN_MAX_LEGAL_EVIDENCE_TOKENS: int = 8000

    TOKEN_MAX_MATTER_EVIDENCE_TOKENS: int = 6000

    TOKEN_MAX_CONVERSATION_DOCUMENT_TOKENS: int = 4000

    TOKEN_SCAFFOLDING_OVERHEAD: int = 600

    TOKEN_TOKENIZER_ID: str = "cl100k_base"

    TOKEN_COUNTING_ENABLED: bool = True

    TOKEN_COST_TRACKING_ENABLED: bool = True

    TOKEN_USAGE_WARNING_PERCENT: float = 80.0

    # Tighter packing when chat uses a hosted LLM (cost + focus).
    TOKEN_ONLINE_RESERVED_OUTPUT: int = 1536

    TOKEN_ONLINE_SAFETY_MARGIN: int = 256

    TOKEN_ONLINE_SCAFFOLDING: int = 180

    TOKEN_ONLINE_MAX_CONVERSATION_TOKENS: int = 1200

    TOKEN_ONLINE_MAX_LEGAL_EVIDENCE_TOKENS: int = 3500

    TOKEN_ONLINE_MAX_MATTER_EVIDENCE_TOKENS: int = 2500

    TOKEN_ONLINE_MAX_CONVERSATION_DOCUMENT_TOKENS: int = 2000



    # ==================================================
    # Observability / Logging
    # ==================================================

    LOG_ENABLE_REQUEST_LOGGING: bool = True

    LOG_ENABLE_RESPONSE_BODY: bool = True

    LOG_BODY_MAX_BYTES: int = 16384

    LOG_SKIP_PATHS: str = "/docs,/openapi.json,/redoc"


    @property
    def resolved_llm_api_key(self) -> str | None:
        from app.llm.provider_config import nonempty_secret

        return nonempty_secret(self.LLM_API_KEY) or self.api_key_for_provider(
            self.LLM_PROVIDER
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

    @property
    def embedding_base_url(self) -> str:
        if self.EMBEDDING_URL:
            return self.EMBEDDING_URL.rstrip("/")
        if (self.LLM_PROVIDER or "").lower() == "ollama":
            return self.LLM_URL.rstrip("/")
        return "http://localhost:11434"

    @property
    def is_online_llm(self) -> bool:
        from app.llm.provider_config import is_online_provider

        return is_online_provider(self.LLM_PROVIDER)



@lru_cache
def get_settings() -> Settings:

    return Settings()



settings = get_settings()