from __future__ import annotations

import logging
from datetime import datetime, timezone
from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.conversation import Conversation
from app.models.message import Message, MessageRole
from app.models.user import User
from app.repositories.conversation_repository import ConversationRepository
from app.repositories.matter_repository import MatterRepository
from app.repositories.message_repository import MessageRepository
from app.models.activity_log import ActivityAction
from app.services.activity_log_service import ActivityLogService

logger = logging.getLogger(__name__)


class ConversationService:
    """
    Conversation lifecycle.

    Rules
    -----
    • conversation_id provided
        -> load existing conversation

    • conversation_id omitted
        -> create new conversation

    • matter_id is optional
        -> conversation may or may not belong to a Matter
    """

    def __init__(
        self,
        *,
        db: AsyncSession,
        conversation_repository: ConversationRepository,
        message_repository: MessageRepository,
        matter_repository: MatterRepository,
        activity_log_service: ActivityLogService,
    ) -> None:

        self.db = db
        self.conversations = conversation_repository
        self.messages = message_repository
        self.matters = matter_repository
        self.activity_logs = activity_log_service

    # ---------------------------------------------------------
    # Conversation
    # ---------------------------------------------------------

    async def get_or_create(
        self,
        *,
        user: User,
        conversation_id: UUID | None,
        matter_id: UUID | None,
        first_message: str,
    ) -> Conversation:

        # Existing conversation
        if conversation_id:

            conversation = await self.conversations.get(
                conversation_id,
            )

            if conversation is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Conversation not found.",
                )

            await self._validate_ownership(
                conversation,
                user.id,
            )

            return conversation

        # Create new conversation

        if matter_id is not None:

            matter = await self.matters.get(matter_id)

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

        conversation = Conversation(
            user_id=user.id,
            matter_id=matter_id,
            title=self._generate_title(first_message),
        )

        self.db.add(conversation)

        await self.db.commit()

        await self.db.refresh(conversation)

        logger.info(
            "Conversation created %s",
            conversation.id,
        )

        return conversation

    async def create(
        self,
        *,
        user: User,
        title: str = "New Conversation",
        matter_id: UUID | None = None,
    ) -> Conversation:
        """Create an empty conversation (e.g. before document upload)."""

        if matter_id is not None:
            matter = await self.matters.get(matter_id)

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

        cleaned_title = (title or "").strip() or "New Conversation"
        if len(cleaned_title) > 60:
            cleaned_title = cleaned_title[:57] + "..."

        conversation = Conversation(
            user_id=user.id,
            matter_id=matter_id,
            title=cleaned_title,
        )

        self.db.add(conversation)
        await self.db.commit()
        await self.db.refresh(conversation)

        logger.info("Conversation created %s", conversation.id)

        await self.activity_logs.record(
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
            },
        )

        return conversation

    # ---------------------------------------------------------
    # History
    # ---------------------------------------------------------

    async def get_history(
        self,
        *,
        user: User,
        conversation_id: UUID,
    ) -> list[Message]:

        conversation = await self._get_owned(
            user=user,
            conversation_id=conversation_id,
        )

        return await self.messages.list_by_conversation(
            conversation.id,
        )

    # ---------------------------------------------------------
    # Save Messages
    # ---------------------------------------------------------

    async def save_exchange(
        self,
        *,
        conversation: Conversation,
        question: str,
        answer: str,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        total_tokens: int | None = None,
        save_user: bool = True,
    ) -> tuple[Message | None, Message]:

        user_message = None
        if save_user:
            user_message = Message(
                conversation_id=conversation.id,
                role=MessageRole.USER,
                content=question,
            )
            self.db.add(user_message)

        assistant_message = Message(
            conversation_id=conversation.id,
            role=MessageRole.ASSISTANT,
            content=answer,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
        self.db.add(assistant_message)

        conversation.last_message_at = datetime.now(
            timezone.utc,
        )

        await self.db.commit()

        await self.db.refresh(assistant_message)
        await self.db.refresh(conversation)
        if user_message is not None:
            await self.db.refresh(user_message)

        logger.info(
            "Conversation updated %s",
            conversation.id,
        )

        return user_message, assistant_message

    # ---------------------------------------------------------
    # Rename
    # ---------------------------------------------------------

    async def rename(
        self,
        *,
        user: User,
        conversation_id: UUID,
        title: str,
    ) -> Conversation:

        conversation = await self._get_owned(
            user=user,
            conversation_id=conversation_id,
        )

        cleaned = (title or "").strip()
        if not cleaned:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Title cannot be empty.",
            )
        if len(cleaned) > 120:
            cleaned = cleaned[:117] + "..."

        conversation.title = cleaned

        await self.db.commit()

        await self.db.refresh(conversation)

        return conversation

    # ---------------------------------------------------------
    # Delete
    # ---------------------------------------------------------

    async def delete(
        self,
        *,
        user: User,
        conversation_id: UUID,
    ) -> None:

        conversation = await self._get_owned(
            user=user,
            conversation_id=conversation_id,
        )

        await self.db.delete(conversation)

        await self.db.commit()

    # ---------------------------------------------------------
    # List
    # ---------------------------------------------------------

    async def list_by_user(
        self,
        *,
        user: User,
    ) -> list[Conversation]:

        return await self.conversations.list_by_user(
            user.id,
        )

    async def list_by_matter(
        self,
        *,
        user: User,
        matter_id: UUID,
    ) -> list[Conversation]:

        conversations = await self.conversations.list_by_matter(
            matter_id,
        )

        for conversation in conversations:
            await self._validate_ownership(
                conversation,
                user.id,
            )

        return conversations

    async def get_for_user(
        self,
        *,
        user: User,
        conversation_id: UUID,
    ) -> Conversation:

        return await self._get_owned(
            user=user,
            conversation_id=conversation_id,
        )

    async def record_view(
        self,
        *,
        user: User,
        conversation: Conversation,
    ) -> None:
        matter_title = None
        if conversation.matter is not None:
            matter_title = conversation.matter.title

        summary = f'Opened conversation "{conversation.title}"'
        if matter_title:
            summary = (
                f'Opened conversation "{conversation.title}" '
                f'in matter "{matter_title}"'
            )

        await self.activity_logs.record(
            action=ActivityAction.CONVERSATION_VIEWED,
            summary=summary,
            user_id=user.id,
            entity_type="conversation",
            entity_id=conversation.id,
            metadata={
                "title": conversation.title,
                "matter_id": str(conversation.matter_id)
                if conversation.matter_id
                else None,
                "matter_title": matter_title,
            },
        )

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------

    async def _get_owned(
        self,
        *,
        user: User,
        conversation_id: UUID,
    ) -> Conversation:

        conversation = await self.conversations.get(
            conversation_id,
        )

        if conversation is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found.",
            )

        await self._validate_ownership(
            conversation,
            user.id,
        )

        return conversation

    async def _validate_ownership(
        self,
        conversation: Conversation,
        user_id: UUID,
    ) -> None:
        """
        Conversation access requires ownership via conversations.user_id.

        When a matter is linked, matter.owner_id must also match.
        """

        if conversation.user_id != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Forbidden.",
            )

        if conversation.matter is not None:
            if conversation.matter.owner_id != user_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Forbidden.",
                )

    @staticmethod
    def _generate_title(
        message: str,
    ) -> str:

        title = message.strip()

        if not title:
            return "New Conversation"

        if len(title) > 60:
            return title[:57] + "..."

        return title