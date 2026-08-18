from __future__ import annotations

from dataclasses import dataclass

from app.models.user import User
from app.models.conversation import Conversation
from app.models.message import Message as DBMessage

from app.rag.models import Message, Memory

from app.repositories.message_repository import MessageRepository
from app.services.memory_service import MemoryService



@dataclass(slots=True)
class PromptContext:
    """
    Complete AI context.

    history:
        Recent conversation messages
        from PostgreSQL.


    memories:
        Semantic memories
        from Qdrant.

    """

    history: list[Message]

    memories: list[Memory]



class PromptContextService:
    """
    Builds context before RAG.


    Sources:


    PostgreSQL:

        conversations
            |
            +-- messages


    Qdrant:

        user_memory

            |
            +-- user scope
            +-- matter scope
            +-- conversation scope


    Security:

        All queries are scoped
        through authenticated user.
    """



    HISTORY_LIMIT = 10

    MEMORY_LIMIT = 5



    def __init__(
        self,
        *,
        message_repository: MessageRepository,
        memory_service: MemoryService,
    ) -> None:


        self._message_repository = (
            message_repository
        )

        self._memory_service = (
            memory_service
        )



    async def build(
        self,
        *,
        user: User,
        conversation: Conversation,
        query: str,
    ) -> PromptContext:


        # =====================================
        # Conversation History
        # =====================================

        messages = await (
            self._message_repository
            .list_by_conversation(
                conversation.id
            )
        )


        history = self._convert_history(
            messages
        )



        # =====================================
        # Semantic Memory
        # =====================================

        memories = await (
            self._memory_service
            .get_memories(

                user_id=user.id,

                matter_id=(

                    conversation.matter_id

                    if conversation.matter_id

                    else None

                ),

                conversation_id=(

                    conversation.id

                    if conversation.id

                    else None

                ),

                query=query,

                limit=self.MEMORY_LIMIT,

            )
        )



        return PromptContext(

            history=history,

            memories=memories,

        )



    def _convert_history(
        self,
        messages: list[DBMessage],
    ) -> list[Message]:

        """
        Convert DB messages into RAG DTOs.
        """


        recent = messages[
            -self.HISTORY_LIMIT:
        ]


        return [

            Message(

                role=(

                    message.role.value

                    if hasattr(
                        message.role,
                        "value"
                    )

                    else str(
                        message.role
                    )

                ),

                content=message.content,

                created_at=message.created_at,

            )

            for message in recent

        ]