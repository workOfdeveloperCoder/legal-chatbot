from __future__ import annotations

import logging

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
)

from app.core.config import settings


logger = logging.getLogger(__name__)


class QdrantService:
    """
    Central Qdrant manager.

    Collections:


    legal_documents
    ---------------
    Public Pakistani legal corpus.

    Owner:
        Legal GPT ingestion project


    chatbot_documents
    -----------------
    User uploaded private documents.

    Contains:

        - Matter documents
        - Conversation documents


    Security:

        user_id


    user_memory
    -----------
    AI memory vectors.

    Contains:

        - User preferences
        - Matter context
        - Conversation memory


    PostgreSQL owns metadata.

    Qdrant owns:
        - embeddings
        - searchable vectors

    """



    def __init__(self) -> None:

        self.client = AsyncQdrantClient(

            host=settings.QDRANT_HOST,

            port=settings.QDRANT_PORT,

        )



    # ==================================================
    # Collection names
    # ==================================================

    @property
    def legal_collection(self) -> str:

        return settings.LEGAL_QDRANT_COLLECTION



    @property
    def document_collection(self) -> str:

        return settings.USER_DOCUMENT_COLLECTION



    @property
    def memory_collection(self) -> str:

        return settings.USER_MEMORY_COLLECTION



    # ==================================================
    # Startup initialization
    # ==================================================

    async def initialize(self) -> None:
        """
        Initialize only Legal Chatbot collections.

        IMPORTANT:

        legal_documents is managed separately
        by Legal GPT ingestion.

        Never recreate it here.
        """


        await self._ensure_collection(
            self.document_collection
        )


        await self._ensure_collection(
            self.memory_collection
        )


        logger.info(
            "Qdrant initialization completed"
        )



    # ==================================================
    # Ensure collection exists
    # ==================================================

    async def _ensure_collection(
        self,
        collection_name: str,
    ) -> None:


        exists = await self.client.collection_exists(

            collection_name=collection_name

        )


        if exists:

            logger.info(
                "Qdrant collection exists %s",
                collection_name,
            )

            return



        await self.client.create_collection(

            collection_name=collection_name,


            vectors_config=VectorParams(

                size=settings.VECTOR_SIZE,

                distance=Distance.COSINE,

            ),

        )


        logger.info(

            "Qdrant collection created %s",

            collection_name,

        )



    # ==================================================
    # Health
    # ==================================================

    async def health(self) -> bool:

        try:

            await self.client.get_collections()

            return True


        except Exception as exc:

            logger.error(
                "Qdrant health failed %s",
                exc,
            )

            return False



    # ==================================================
    # Shutdown
    # ==================================================

    async def close(self) -> None:

        await self.client.close()



# Singleton instance

qdrant_service = QdrantService()