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

    LLM_PROVIDER: str = "ollama"


    LLM_URL: str = (
        "http://localhost:11434"
    )


    CHAT_MODEL: str = (
        "deepseek-r1:32b"
    )


    EMBEDDING_MODEL: str = (
        "nomic-embed-text:latest"
    )


    # Local DeepSeek R1 32B often needs many minutes per answer
    # (especially with two-stage reasoning). Online APIs can override lower.
    LLM_TIMEOUT: int = 1800

    LLM_CONNECT_TIMEOUT: int = 60

    LLM_API_KEY: str | None = None

    LLM_MAX_RETRIES: int = 1

    LLM_RETRY_BASE_DELAY_SECONDS: float = 1.0

    LLM_FALLBACK_PROVIDER: str | None = None

    LLM_FALLBACK_URL: str | None = None

    LLM_FALLBACK_MODEL: str | None = None

    LLM_FALLBACK_API_KEY: str | None = None

    LLM_TEMPERATURE: float = 0.1

    LLM_ENABLE_STREAMING: bool = False

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



    # ==================================================
    # Observability / Logging
    # ==================================================

    LOG_ENABLE_REQUEST_LOGGING: bool = True

    LOG_ENABLE_RESPONSE_BODY: bool = True

    LOG_BODY_MAX_BYTES: int = 16384

    LOG_SKIP_PATHS: str = "/docs,/openapi.json,/redoc"



@lru_cache
def get_settings() -> Settings:

    return Settings()



settings = get_settings()