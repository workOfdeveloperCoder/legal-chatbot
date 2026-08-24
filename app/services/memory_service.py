from __future__ import annotations

import logging
import uuid

from uuid import UUID

from qdrant_client.models import PointStruct

from app.core.config import settings
from app.embeddings.service import EmbeddingService

from app.rag.models import Memory

from app.vector.filters import QdrantFilterBuilder
from app.vector.qdrant import qdrant_service


logger = logging.getLogger(__name__)


class MemoryService:
    """
    AI Long Term Memory Service.


    Qdrant:

        user_memory


    Stores:

        user memory
        matter memory
        conversation memory


    NOT document chunks.

    Documents are stored in:

        chatbot_documents


    Security:

        Every query requires user_id.


    Visibility:

        user
            |
            available everywhere


        matter
            |
            same matter


        conversation
            |
            same conversation

    """



    def __init__(self):

        self.client = qdrant_service.client


        self.collection = (
            settings.USER_MEMORY_COLLECTION
        )


        self.embedding = EmbeddingService()



    async def get_memories(
        self,
        *,
        user_id: UUID,
        query: str,
        matter_id: UUID | None = None,
        conversation_id: UUID | None = None,
        limit: int = 5,
    ) -> list[Memory]:


        if not query:

            return []



        vector = await self.embedding.embed_query(
            query
        )



        memory_filter = (
            QdrantFilterBuilder.memory_visibility(

                user_id=str(user_id),

                matter_id=(

                    str(matter_id)

                    if matter_id

                    else None

                ),

                conversation_id=(

                    str(conversation_id)

                    if conversation_id

                    else None

                ),

            )
        )



        result = await self.client.query_points(

            collection_name=self.collection,

            query=vector,

            query_filter=memory_filter,

            limit=limit,

        )



        memories=[]



        for point in result.points:


            payload = point.payload or {}


            memories.append(

                Memory(

                    id=str(point.id),


                    text=payload.get(
                        "text",
                        "",
                    ),


                    score=float(
                        point.score
                    ),


                    memory_type=payload.get(
                        "memory_type"
                    ),


                    scope=payload.get(
                        "scope"
                    ),


                    user_id=payload.get(
                        "user_id"
                    ),


                    matter_id=payload.get(
                        "matter_id"
                    ),


                    conversation_id=payload.get(
                        "conversation_id"
                    ),


                    document_id=payload.get(
                        "document_id"
                    ),

                )

            )



        return memories



    async def process_chat(
        self,
        *,
        user_id: UUID,
        question: str,
        answer: str,
        matter_id: UUID | None = None,
        conversation_id: UUID | None = None,
    ):


        if not question or not answer:

            return



        text=f"""
Question:

{question}


Answer:

{answer}
""".strip()



        await self._store(

            user_id=user_id,

            text=text,

            memory_type="chat",

            matter_id=matter_id,

            conversation_id=conversation_id,

        )



    async def _store(
        self,
        *,
        user_id: UUID,
        text: str,
        memory_type: str,
        matter_id: UUID | None = None,
        conversation_id: UUID | None = None,
    ):


        vector = await self.embedding.embed_query(
            text
        )



        if conversation_id:

            scope="conversation"


        elif matter_id:

            scope="matter"


        else:

            scope="user"



        payload={

            "memory_id":str(
                uuid.uuid4()
            ),


            "user_id":str(
                user_id
            ),


            "scope":scope,


            "memory_type":memory_type,


            "text":text,


            "matter_id":(

                str(matter_id)

                if matter_id

                else None

            ),


            "conversation_id":(

                str(conversation_id)

                if conversation_id

                else None

            ),

        }



        await self.client.upsert(

            collection_name=self.collection,


            points=[

                PointStruct(

                    id=str(
                        uuid.uuid4()
                    ),

                    vector=vector,

                    payload=payload,

                )

            ],

        )



        logger.info(

            "Memory stored user=%s scope=%s type=%s",

            user_id,

            scope,

            memory_type,

        )