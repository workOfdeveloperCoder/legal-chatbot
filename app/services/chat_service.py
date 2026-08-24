from __future__ import annotations

import logging

from fastapi import HTTPException, status

from app.models.user import User
from app.models.document import Document, DocumentScope

from app.rag.legal_query_planner import LegalQueryPlanner
from app.rag.rag_service import RAGService, DOCUMENT_NOT_READY_DETAIL

from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
)
from app.services.token_usage_service import token_usage_from_budget

from app.services.conversation_service import (
    ConversationService,
)

from app.services.memory_service import (
    MemoryService,
)

from app.services.prompt_context_service import (
    PromptContextService,
)

from app.services.query_router import (
    QueryRouter,
)

from app.repositories.document_repository import (
    DocumentRepository,
)
from app.models.activity_log import ActivityAction
from app.services.activity_log_service import ActivityLogService


logger = logging.getLogger(__name__)


class ChatService:
    """
    Production Chat Orchestrator.
    """

    def __init__(
        self,
        *,
        rag_service: RAGService,
        conversation_service: ConversationService,
        memory_service: MemoryService,
        prompt_context_service: PromptContextService,
        query_router: QueryRouter,
        document_repository: DocumentRepository,
        activity_log_service: ActivityLogService,
        legal_query_planner: LegalQueryPlanner | None = None,
    ) -> None:

        self._rag = rag_service
        self._conversation = conversation_service
        self._memory = memory_service
        self._context = prompt_context_service
        self._router = query_router
        self._planner = legal_query_planner or LegalQueryPlanner(query_router)
        self._documents = document_repository
        self._activity_logs = activity_log_service

    async def chat(
        self,
        *,
        user: User,
        payload: ChatRequest,
    ) -> ChatResponse:

        logger.info(
            "Chat request user=%s conversation=%s matter=%s document_id=%s",
            user.id,
            payload.conversation_id,
            payload.matter_id,
            payload.document_id,
        )

        conversation = await self._conversation.get_or_create(
            user=user,
            conversation_id=payload.conversation_id,
            matter_id=payload.matter_id,
            first_message=payload.message,
        )

        created_new_conversation = payload.conversation_id is None

        # Attach matter when client uploads to a matter after chat started
        if conversation.matter_id is None and payload.matter_id is not None:
            conversation.matter_id = payload.matter_id
            await self._conversation.db.commit()
            await self._conversation.db.refresh(conversation)

        matter_id = (
            str(conversation.matter_id)
            if conversation.matter_id
            else (
                str(payload.matter_id)
                if payload.matter_id
                else None
            )
        )
        conversation_id = str(conversation.id)

        document_id: str | None = None
        if payload.document_id is not None:
            document = await self._verify_document_access(
                user=user,
                document_id=payload.document_id,
                conversation_id=conversation.id,
                matter_id=conversation.matter_id or payload.matter_id,
            )
            document_id = str(document.id)

        has_uploaded_documents = bool(document_id) or (
            await self._documents.has_for_scope(
                owner_id=user.id,
                conversation_id=conversation.id,
                matter_id=conversation.matter_id,
            )
        )

        context = await self._context.build(
            user=user,
            conversation=conversation,
            query=payload.message,
        )

        plan = await self._planner.plan(
            question=payload.message,
            has_uploaded_documents=has_uploaded_documents,
            document_id=document_id,
            matter_id=matter_id,
            conversation_id=conversation_id,
            history=context.history,
        )

        logger.info(
            "Query plan mode=%s strategy=%s task=%s complexity=%s "
            "requires_authority=%s document_primary=%s sections=%s",
            plan.answer_mode.value,
            plan.retrieval_strategy.value,
            plan.task.value,
            plan.complexity.value,
            plan.requires_legal_authority,
            plan.uploaded_document_primary,
            plan.sections,
        )

        result = await self._rag.execute(
            question=payload.message,
            history=context.history,
            memories=context.memories,
            task=plan.task,
            user_id=str(user.id),
            matter_id=matter_id,
            conversation_id=conversation_id,
            document_id=document_id,
            query_plan=plan,
        )

        answer = result["answer"]

        await self._conversation.save_exchange(
            conversation=conversation,
            question=payload.message,
            answer=answer,
            prompt_tokens=result.get("prompt_tokens"),
            completion_tokens=result.get("completion_tokens"),
            total_tokens=result.get("total_tokens"),
        )

        try:
            await self._memory.process_chat(
                user_id=user.id,
                matter_id=conversation.matter_id,
                conversation_id=conversation.id,
                question=payload.message,
                answer=answer,
            )
        except Exception:
            logger.exception(
                "Memory update failed user=%s",
                user.id,
            )

        if created_new_conversation:
            await self._activity_logs.record(
                action=ActivityAction.CONVERSATION_CREATED,
                summary=f'Started conversation "{conversation.title}"',
                user_id=user.id,
                entity_type="conversation",
                entity_id=conversation.id,
                metadata={
                    "title": conversation.title,
                    "matter_id": str(conversation.matter_id)
                    if conversation.matter_id
                    else None,
                    "source": "chat",
                },
            )

        await self._activity_logs.record(
            action=ActivityAction.CHAT_MESSAGE_SENT,
            summary=f'Sent message in conversation "{conversation.title}"',
            user_id=user.id,
            entity_type="conversation",
            entity_id=conversation.id,
            metadata={
                "matter_id": str(conversation.matter_id)
                if conversation.matter_id
                else None,
                "message_preview": payload.message[:120],
            },
        )

        return ChatResponse(
            conversation_id=conversation.id,
            response=answer,
            citations=result.get("citations", []),
            sources=result.get("sources", []),
            resources=result.get("resources", []),
            retrieval_metadata=result.get("retrieval_metadata"),
            grounding_status=result.get("grounding_status"),
            token_usage=token_usage_from_budget(
                (
                    result.get("retrieval_metadata").token_budget
                    if result.get("retrieval_metadata") is not None
                    else None
                ),
            ),
        )

    async def _verify_document_access(
        self,
        *,
        user: User,
        document_id,
        conversation_id,
        matter_id,
    ) -> Document:
        document = await self._documents.get_by_id(document_id)

        if document is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found.",
            )

        if document.owner_id != user.id:
            # Do not leak existence of another user's document
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found.",
            )

        accessible = False
        if (
            document.scope == DocumentScope.CONVERSATION
            and document.conversation_id == conversation_id
        ):
            accessible = True
        elif (
            document.scope == DocumentScope.MATTER
            and matter_id is not None
            and document.matter_id == matter_id
        ):
            accessible = True

        if not accessible:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    "You don't have permission to access "
                    "this document in the current conversation."
                ),
            )

        if (
            not document.vectorized
            or not (document.extracted_text or "").strip()
        ):
            logger.warning(
                "Document not ready for RAG document_id=%s "
                "processed=%s vectorized=%s",
                document.id,
                document.processed,
                document.vectorized,
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=DOCUMENT_NOT_READY_DETAIL,
            )

        return document
