from fastapi import FastAPI
from fastapi.concurrency import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from app.api.v1.auth import router as auth_router
from app.api.v1.matters import router as matters_router
from app.api.v1.chat import router as chats_router
from app.api.v1.conversations import router as conversations_router
from app.api.v1.documents import router as document_router
from app.api.v1.contracts import router as contracts_router
from app.api.v1.logs import router as logs_router
from app.api.v1.voice import router as voice_router
from app.api.v1.llm import router as llm_router
from app.api.v1.health import router as health_router
from app.core.config import settings
from app.core.rate_limit import RateLimitMiddleware
from app.core.request_size import RequestSizeLimitMiddleware
from app.core.production import validate_production_settings
from app.observability.exception_handlers import register_exception_handlers
from app.observability.middleware import RequestLoggingMiddleware
from app.vector.qdrant import qdrant_service


@asynccontextmanager
async def lifespan(app: FastAPI):

    # Startup
    validate_production_settings(settings)
    await qdrant_service.initialize()

    yield

    # Shutdown
    await qdrant_service.client.close()




app = FastAPI(
    title="LegalGPT API",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None if not settings.DEBUG else "/docs",
    redoc_url=None if not settings.DEBUG else "/redoc",
    openapi_url=None if not settings.DEBUG else "/openapi.json",
)

register_exception_handlers(app)

if settings.trusted_hosts != ["*"]:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.trusted_hosts,
    )

app.add_middleware(RateLimitMiddleware)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(RequestSizeLimitMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "refresh-token"],
)

app.include_router(
    health_router,
)

app.include_router(
    auth_router,
    prefix="/api/v1",
)

app.include_router(
    matters_router,
    prefix="/api/v1",
)

app.include_router(
    chats_router,
    prefix="/api/v1",
)

app.include_router(
    voice_router,
    prefix="/api/v1",
)

app.include_router(
    llm_router,
    prefix="/api/v1",
)

app.include_router(
    conversations_router,
    prefix="/api/v1",
)

app.include_router(
    document_router,
    prefix="/api/v1",
)

app.include_router(
    contracts_router,
    prefix="/api/v1",
)

app.include_router(
    logs_router,
    prefix="/api/v1",
)
