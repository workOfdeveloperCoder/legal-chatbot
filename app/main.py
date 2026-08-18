from fastapi import FastAPI
from fastapi.concurrency import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.auth import router as auth_router
from app.api.v1.matters import router as matters_router
from app.api.v1.chat import router as chats_router
from app.api.v1.conversations import router as conversations_router
from app.api.v1.documents import router as document_router
from app.api.v1.logs import router as logs_router
from app.core.config import settings
from app.observability.exception_handlers import register_exception_handlers
from app.observability.middleware import RequestLoggingMiddleware
from app.vector.qdrant import qdrant_service


@asynccontextmanager
async def lifespan(app: FastAPI):

    # Startup
    await qdrant_service.initialize()

    yield

    # Shutdown
    await qdrant_service.client.close()




app = FastAPI(
    title="LegalGPT API",
    version="1.0.0",
    lifespan=lifespan
)

register_exception_handlers(app)

app.add_middleware(RequestLoggingMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "refresh-token"],
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
    conversations_router,
    prefix="/api/v1",
)

app.include_router(
    document_router,
    prefix="/api/v1",
)

app.include_router(
    logs_router,
    prefix="/api/v1",
)
