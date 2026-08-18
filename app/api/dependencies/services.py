from __future__ import annotations

from functools import lru_cache

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.service import AuthService
from app.database.session import get_db

from app.llm.base import BaseLLM
from app.llm.factory import LLMFactory

from app.rag.answer_guard import AnswerGuard
from app.rag.prompt_builder import PromptBuilder
from app.rag.query_rewriter import QueryRewriter
from app.rag.rag_service import RAGService
from app.rag.repository import BaseRetriever
from app.rag.reranker import HybridReranker
from app.rag.qdrant_retriever import QdrantRetriever

from app.document.extractor import DocumentExtractor
from app.embeddings.service import EmbeddingService
from app.repositories.conversation_repository import ConversationRepository
from app.repositories.document_repository import DocumentRepository
from app.repositories.matter_repository import MatterRepository
from app.repositories.message_repository import MessageRepository
from app.repositories.refresh_token_repository import RefreshTokenRepository
from app.repositories.user_repository import UserRepository

from app.services.activity_log_service import ActivityLogService
from app.services.chat_service import ChatService
from app.services.conversation_service import ConversationService
from app.services.document_service import DocumentService
from app.services.error_log_service import ErrorLogService
from app.services.matter_service import MatterService
from app.services.request_log_service import RequestLogService
from app.services.memory_service import MemoryService
from app.services.prompt_context_service import PromptContextService
from app.services.query_router import QueryRouter
from app.services.response_formatter import ResponseFormatter
from app.vector.service import VectorService


# ---------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------

@lru_cache
def get_llm() -> BaseLLM:
    return LLMFactory.create()


# ---------------------------------------------------------------------
# Shared Stateless Services
# ---------------------------------------------------------------------

@lru_cache
def get_query_router() -> QueryRouter:
    return QueryRouter()


@lru_cache
def get_prompt_builder() -> PromptBuilder:
    return PromptBuilder()


@lru_cache
def get_query_rewriter() -> QueryRewriter:
    return QueryRewriter()


@lru_cache
def get_response_formatter() -> ResponseFormatter:
    return ResponseFormatter()


@lru_cache
def get_memory_service() -> MemoryService:
    return MemoryService()


@lru_cache
def get_retriever() -> BaseRetriever:
    return QdrantRetriever()


@lru_cache
def get_reranker() -> HybridReranker:
    return HybridReranker()

# ---------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------

@lru_cache
def get_activity_log_service() -> ActivityLogService:
    return ActivityLogService()


@lru_cache
def get_request_log_service() -> RequestLogService:
    return RequestLogService()


@lru_cache
def get_error_log_service() -> ErrorLogService:
    return ErrorLogService()


# ---------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------

def get_auth_service(
    db: AsyncSession = Depends(get_db),
    activity_log_service: ActivityLogService = Depends(get_activity_log_service),
) -> AuthService:

    return AuthService(
        db=db,
        user_repository=UserRepository(db),
        refresh_repository=RefreshTokenRepository(db),
        activity_log_service=activity_log_service,
    )


# ---------------------------------------------------------------------
# Matter
# ---------------------------------------------------------------------

def get_matter_service(
    db: AsyncSession = Depends(get_db),
    activity_log_service: ActivityLogService = Depends(get_activity_log_service),
) -> MatterService:

    return MatterService(
        db=db,
        repository=MatterRepository(db),
        activity_log_service=activity_log_service,
    )


# ---------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------

def get_document_service(
    db: AsyncSession = Depends(get_db),
    activity_log_service: ActivityLogService = Depends(get_activity_log_service),
) -> DocumentService:

    return DocumentService(
        db=db,
        repository=DocumentRepository(db),
        matter_repository=MatterRepository(db),
        conversation_repository=ConversationRepository(db),
        vector_service=VectorService(
            embedding_service=EmbeddingService(),
        ),
        extractor=DocumentExtractor(),
        activity_log_service=activity_log_service,
    )


# ---------------------------------------------------------------------
# Conversation
# ---------------------------------------------------------------------

def get_conversation_service(
    db: AsyncSession = Depends(get_db),
    activity_log_service: ActivityLogService = Depends(get_activity_log_service),
) -> ConversationService:

    return ConversationService(
        db=db,
        conversation_repository=ConversationRepository(db),
        message_repository=MessageRepository(db),
        matter_repository=MatterRepository(db),
        activity_log_service=activity_log_service,
    )


# ---------------------------------------------------------------------
# Prompt Context
# ---------------------------------------------------------------------

def get_prompt_context_service(
    db: AsyncSession = Depends(get_db),
    memory_service: MemoryService = Depends(get_memory_service),
) -> PromptContextService:

    return PromptContextService(
        message_repository=MessageRepository(db),
        memory_service=memory_service,
    )


# ---------------------------------------------------------------------
# RAG
# ---------------------------------------------------------------------

def get_rag_service(
    retriever: BaseRetriever = Depends(get_retriever),
    llm: BaseLLM = Depends(get_llm),
    prompt_builder: PromptBuilder = Depends(get_prompt_builder),
    query_rewriter: QueryRewriter = Depends(get_query_rewriter),
    response_formatter: ResponseFormatter = Depends(get_response_formatter),
    db: AsyncSession = Depends(get_db),
) -> RAGService:

    return RAGService(
        llm=llm,
        retriever=retriever,
        query_rewriter=query_rewriter,
        prompt_builder=prompt_builder,
        response_formatter=response_formatter,
        answer_guard=AnswerGuard(),
        document_repository=DocumentRepository(db),
    )


# ---------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------

def get_chat_service(
    rag_service: RAGService = Depends(get_rag_service),
    conversation_service: ConversationService = Depends(get_conversation_service),
    prompt_context_service: PromptContextService = Depends(get_prompt_context_service),
    memory_service: MemoryService = Depends(get_memory_service),
    query_router: QueryRouter = Depends(get_query_router),
    activity_log_service: ActivityLogService = Depends(get_activity_log_service),
    db: AsyncSession = Depends(get_db),
) -> ChatService:

    return ChatService(
        rag_service=rag_service,
        conversation_service=conversation_service,
        memory_service=memory_service,
        prompt_context_service=prompt_context_service,
        query_router=query_router,
        document_repository=DocumentRepository(db),
        activity_log_service=activity_log_service,
    )

