from __future__ import annotations

import inspect
import logging

from fastapi import HTTPException, status
from uuid import UUID

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

from app.repositories.document_repository import (
    DocumentRepository,
)
from app.models.activity_log import ActivityAction
from app.services.activity_log_service import ActivityLogService
from app.services.quick_actions import (
    CARD_OPENING_REPLIES,
    MISSING_DOCUMENT_REPLIES,
    is_card_click,
    resolve_from_message,
    resolve_quick_action,
    should_return_card_opening,
)
from app.services.query_router import (
    QueryRouter,
    Task,
)
from app.services.contract_analysis_service import ContractAnalysisService


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
        contract_analysis: ContractAnalysisService | None = None,
    ) -> None:

        self._rag = rag_service
        self._conversation = conversation_service
        self._memory = memory_service
        self._context = prompt_context_service
        self._router = query_router
        self._planner = legal_query_planner or LegalQueryPlanner(query_router)
        self._documents = document_repository
        self._activity_logs = activity_log_service
        self._contracts = contract_analysis

    async def chat(
        self,
        *,
        user: User,
        payload: ChatRequest,
        token_callback=None,
        started_callback=None,
        thinking_callback=None,
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
            await self._require_matter_access(user=user, matter_id=payload.matter_id)
            conversation.matter_id = payload.matter_id
            await self._conversation.db.commit()
            await self._conversation.db.refresh(conversation)

        if started_callback is not None:
            maybe = started_callback(conversation)
            if inspect.isawaitable(maybe):
                await maybe

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

        action = resolve_quick_action(payload.quick_action) or resolve_from_message(
            payload.message
        )
        question = (
            action.prompt
            if action is not None and is_card_click(payload.message, action)
            else payload.message
        )
        quick_action = action.slug if action is not None else payload.quick_action

        if (
            action is not None
            and action.requires_document
            and not has_uploaded_documents
        ):
            return await self._finish_without_llm(
                user=user,
                conversation=conversation,
                payload=payload,
                answer=MISSING_DOCUMENT_REPLIES[action.slug],
                created_new_conversation=created_new_conversation,
            )

        if (
            action is not None
            and should_return_card_opening(
                payload.message,
                action,
                quick_action=payload.quick_action,
            )
        ):
            return await self._finish_without_llm(
                user=user,
                conversation=conversation,
                payload=payload,
                answer=CARD_OPENING_REPLIES[action.slug],
                created_new_conversation=created_new_conversation,
            )

        context = await self._context.build(
            user=user,
            conversation=conversation,
            query=question,
        )

        logger.info(
            "Chat request web_search=%s quick_action=%s user=%s",
            payload.web_search,
            payload.quick_action,
            user.id,
        )

        plan = await self._planner.plan(
            question=question,
            has_uploaded_documents=has_uploaded_documents,
            document_id=document_id,
            matter_id=matter_id,
            conversation_id=conversation_id,
            history=context.history,
            quick_action=quick_action,
            web_search=payload.web_search,
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

        if self._contracts is not None and plan.task == Task.REVIEW_TABLE:
            if not has_uploaded_documents:
                return await self._finish_without_llm(
                    user=user,
                    conversation=conversation,
                    payload=payload,
                    answer=MISSING_DOCUMENT_REPLIES["review_table"],
                    created_new_conversation=created_new_conversation,
                )
            return await self._review_table_chat(
                user=user,
                conversation=conversation,
                payload=payload,
                created_new_conversation=created_new_conversation,
            )

        if self._contracts is not None and plan.task == Task.PLAYBOOK_REVIEW:
            if not has_uploaded_documents:
                return await self._finish_without_llm(
                    user=user,
                    conversation=conversation,
                    payload=payload,
                    answer=MISSING_DOCUMENT_REPLIES["playbook_review"],
                    created_new_conversation=created_new_conversation,
                )
            return await self._playbook_review_chat(
                user=user,
                conversation=conversation,
                payload=payload,
                document_id=document_id,
                created_new_conversation=created_new_conversation,
            )

        if self._contracts is not None and plan.task == Task.REDLINE:
            if not has_uploaded_documents:
                return await self._finish_without_llm(
                    user=user,
                    conversation=conversation,
                    payload=payload,
                    answer=MISSING_DOCUMENT_REPLIES["redline"],
                    created_new_conversation=created_new_conversation,
                )
            return await self._redline_chat(
                user=user,
                conversation=conversation,
                payload=payload,
                document_id=document_id,
                created_new_conversation=created_new_conversation,
            )

        result = await self._rag.execute(
            question=question,
            history=context.history,
            memories=context.memories,
            task=plan.task,
            user_id=str(user.id),
            matter_id=matter_id,
            conversation_id=conversation_id,
            document_id=document_id,
            query_plan=plan,
            web_search=payload.web_search,
            token_callback=token_callback,
            thinking_callback=thinking_callback,
        )

        answer = result["answer"]
        clause_cards = None
        if (
            self._contracts is not None
            and plan.task == Task.CONTRACT_REVIEW
        ):
            clause_cards = await self._contracts.extract_for_chat(
                user_id=user.id,
                document_id=UUID(document_id) if document_id else None,
                conversation_id=conversation.id,
                matter_id=conversation.matter_id or payload.matter_id,
            )
            if clause_cards is not None:
                answer = (
                    f"{answer.rstrip()}\n\n"
                    f"{self._contracts.cards_markdown(clause_cards)}"
                )

        await self._conversation.save_exchange(
            conversation=conversation,
            question=payload.message,
            answer=answer,
            prompt_tokens=result.get("prompt_tokens"),
            completion_tokens=result.get("completion_tokens"),
            total_tokens=result.get("total_tokens"),
            save_user=not payload.regenerate,
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
            clause_cards=clause_cards,
        )

    async def _finish_without_llm(
        self,
        *,
        user: User,
        conversation,
        payload: ChatRequest,
        answer: str,
        created_new_conversation: bool,
        clause_cards=None,
        review_table=None,
        playbook_review=None,
        redline=None,
    ) -> ChatResponse:
        await self._conversation.save_exchange(
            conversation=conversation,
            question=payload.message,
            answer=answer,
            save_user=not payload.regenerate,
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
            clause_cards=clause_cards,
            review_table=review_table,
            playbook_review=playbook_review,
            redline=redline,
        )

    async def _review_table_chat(
        self,
        *,
        user: User,
        conversation,
        payload: ChatRequest,
        created_new_conversation: bool,
    ) -> ChatResponse:
        table = await self._contracts.review_table(
            user_id=user.id,
            matter_id=conversation.matter_id or payload.matter_id,
            conversation_id=conversation.id,
        )
        if not table.rows:
            answer = MISSING_DOCUMENT_REPLIES["review_table"]
        else:
            answer = self._contracts.table_markdown(table)
        return await self._finish_without_llm(
            user=user,
            conversation=conversation,
            payload=payload,
            answer=answer,
            created_new_conversation=created_new_conversation,
            review_table=table if table.rows else None,
        )

    async def _playbook_review_chat(
        self,
        *,
        user: User,
        conversation,
        payload: ChatRequest,
        document_id: str | None,
        created_new_conversation: bool,
    ) -> ChatResponse:
        review = await self._contracts.playbook_review_for_chat(
            user_id=user.id,
            document_id=UUID(document_id) if document_id else None,
            conversation_id=conversation.id,
            matter_id=conversation.matter_id or payload.matter_id,
        )
        if review is None:
            answer = MISSING_DOCUMENT_REPLIES["playbook_review"]
            return await self._finish_without_llm(
                user=user,
                conversation=conversation,
                payload=payload,
                answer=answer,
                created_new_conversation=created_new_conversation,
            )
        answer = self._contracts.playbook_markdown(review)
        return await self._finish_without_llm(
            user=user,
            conversation=conversation,
            payload=payload,
            answer=answer,
            created_new_conversation=created_new_conversation,
            clause_cards=review.extraction,
            playbook_review=review,
        )

    async def _redline_chat(
        self,
        *,
        user: User,
        conversation,
        payload: ChatRequest,
        document_id: str | None,
        created_new_conversation: bool,
    ) -> ChatResponse:
        redline = await self._contracts.redline_for_chat(
            user_id=user.id,
            document_id=UUID(document_id) if document_id else None,
            conversation_id=conversation.id,
            matter_id=conversation.matter_id or payload.matter_id,
        )
        if redline is None:
            return await self._finish_without_llm(
                user=user,
                conversation=conversation,
                payload=payload,
                answer=MISSING_DOCUMENT_REPLIES["redline"],
                created_new_conversation=created_new_conversation,
            )
        answer = self._contracts.redline_markdown(redline)
        return await self._finish_without_llm(
            user=user,
            conversation=conversation,
            payload=payload,
            answer=answer,
            created_new_conversation=created_new_conversation,
            redline=redline,
        )

    async def chat_stream(
        self,
        *,
        user: User,
        payload: ChatRequest,
    ):
        """Yield SSE-ready dicts: started, status, thinking, token, complete, error."""
        import asyncio

        queue: asyncio.Queue = asyncio.Queue()

        async def on_started(conversation) -> None:
            await queue.put(
                {
                    "event": "started",
                    "conversation_id": str(conversation.id),
                }
            )
            await queue.put(
                {
                    "event": "status",
                    "phase": "retrieving",
                    "detail": (
                        "Searching the legal corpus and the internet…"
                        if payload.web_search
                        else "Searching the legal corpus…"
                    ),
                }
            )

        async def on_token(piece: str) -> None:
            await queue.put({"event": "token", "text": piece})

        async def on_thinking(piece: str) -> None:
            await queue.put({"event": "thinking", "text": piece})

        async def heartbeat() -> None:
            # Keep the SSE queue warm while retrieval/LLM runs with no tokens.
            while True:
                await asyncio.sleep(8)
                await queue.put(
                    {
                        "event": "status",
                        "phase": "thinking",
                        "detail": "Still generating — local model is working…",
                    }
                )

        async def run() -> None:
            pulse = asyncio.create_task(heartbeat())
            try:
                response = await self.chat(
                    user=user,
                    payload=payload,
                    token_callback=on_token,
                    thinking_callback=on_thinking,
                    started_callback=on_started,
                )
                await queue.put(
                    {
                        "event": "complete",
                        "conversation_id": str(response.conversation_id),
                        "response": response.response,
                        "citations": [
                            item.model_dump() for item in response.citations
                        ],
                        "sources": [
                            item.model_dump() for item in response.sources
                        ],
                        "resources": [
                            item.model_dump() for item in response.resources
                        ],
                        "grounding_status": response.grounding_status,
                        "token_usage": (
                            response.token_usage.model_dump()
                            if response.token_usage
                            else None
                        ),
                        "clause_cards": (
                            response.clause_cards.model_dump(by_alias=True)
                            if response.clause_cards
                            else None
                        ),
                        "review_table": (
                            response.review_table.model_dump(by_alias=True)
                            if response.review_table
                            else None
                        ),
                        "playbook_review": (
                            response.playbook_review.model_dump(by_alias=True)
                            if response.playbook_review
                            else None
                        ),
                        "redline": (
                            response.redline.model_dump(by_alias=True)
                            if response.redline
                            else None
                        ),
                    }
                )
            except HTTPException as exc:
                await queue.put(
                    {
                        "event": "error",
                        "status": exc.status_code,
                        "detail": exc.detail,
                    }
                )
            except Exception:
                logger.exception("Streaming chat failed")
                await queue.put(
                    {
                        "event": "error",
                        "status": 503,
                        "detail": "The assistant is temporarily unavailable.",
                    }
                )
            finally:
                pulse.cancel()
                try:
                    await pulse
                except asyncio.CancelledError:
                    pass
                await queue.put(None)

        task = asyncio.create_task(run())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    async def _require_matter_access(self, *, user: User, matter_id) -> None:
        matter = await self._conversation.matters.get(matter_id)
        if matter is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Matter not found.",
            )
        if matter.owner_id != user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to access this matter.",
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
