from __future__ import annotations

import hashlib
import logging
import uuid

from qdrant_client.models import (
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

from app.embeddings.service import EmbeddingService
from app.vector.qdrant import qdrant_service


logger = logging.getLogger(__name__)


class VectorService:
    """
    Production User Document Vector Indexer.


    Storage:

    PostgreSQL
    -----------
        documents table

        Stores:
            - filename
            - owner_id
            - matter_id
            - conversation_id
            - storage_path
            - processing status


    Qdrant
    -------
        chatbot_documents collection

        Stores:
            - document chunks
            - embeddings
            - searchable metadata


    Security Boundary:

        Every vector MUST contain:

            user_id


    Visibility Rules:


        USER DOCUMENT

            user_id
            scope=user


        MATTER DOCUMENT

            user_id
            scope=matter
            matter_id


        CONVERSATION DOCUMENT

            user_id
            scope=conversation
            conversation_id


    """


    def __init__(
        self,
        *,
        embedding_service: EmbeddingService,
    ) -> None:

        self.embedding = embedding_service

        self.collection = (
            qdrant_service.document_collection
        )



    # ==================================================
    # Index Document
    # ==================================================

    async def index_document(
        self,
        *,
        document_id: str,
        owner_id: str,
        matter_id: str | None,
        conversation_id: str | None,
        scope: str,
        text: str,
        filename: str | None = None,
        mime_type: str | None = None,
    ) -> None:


        if not text or not text.strip():

            logger.warning(
                "Empty document skipped document=%s",
                document_id,
            )

            return



        # ------------------------------------------
        # Remove old vectors
        # ------------------------------------------

        await self.delete_document(
            document_id=document_id,
            owner_id=owner_id,
        )



        # ------------------------------------------
        # Split document
        # ------------------------------------------

        chunks = self._chunk_text(
            text=text,
        )

        if not chunks:
            return

        chunk_spans = self._chunk_spans(
            text=text,
            chunks=chunks,
        )

        vectors = await self.embedding.embed_documents(
            chunks
        )

        points: list[PointStruct] = []

        for index, (chunk, vector) in enumerate(
            zip(chunks, vectors)
        ):
            span = chunk_spans[index]
            chunk_id = f"{document_id}:{index}"

            points.append(
                PointStruct(
                    id=self._point_id(
                        document_id,
                        index,
                    ),
                    vector=vector,
                    payload={
                        "user_id": owner_id,
                        "document_id": document_id,
                        "chunk_id": chunk_id,
                        "chunk_index": index,
                        "scope": scope,
                        "matter_id": matter_id,
                        "conversation_id": conversation_id,
                        "text": chunk,
                        "type": "document",
                        "filename": filename,
                        "mime_type": mime_type,
                        "start_offset": span["start_offset"],
                        "end_offset": span["end_offset"],
                    },
                )
            )



        if not points:

            return



        await qdrant_service.client.upsert(

            collection_name=self.collection,

            points=points,

        )



        logger.info(

            "Document indexed document=%s chunks=%s user=%s",

            document_id,

            len(points),

            owner_id,

        )



    # ==================================================
    # Delete Document Vectors
    # ==================================================

    async def delete_document(
        self,
        *,
        document_id: str,
        owner_id: str,
    ) -> None:


        await qdrant_service.client.delete(

            collection_name=self.collection,


            points_selector=Filter(

                must=[

                    FieldCondition(

                        key="document_id",

                        match=MatchValue(

                            value=document_id

                        ),

                    ),

                    FieldCondition(

                        key="user_id",

                        match=MatchValue(

                            value=owner_id

                        ),

                    ),

                ]

            ),

        )


        logger.info(

            "Document vectors removed document=%s user=%s",

            document_id,

            owner_id,

        )



    # ==================================================
    # Read merged document text (chunk reassembly)
    # ==================================================

    async def get_merged_document_text(
        self,
        *,
        document_id: str,
        owner_id: str,
    ) -> tuple[str | None, int]:
        """
        Rebuild full document text from indexed Qdrant chunks.

        Used when PostgreSQL extracted_text is unavailable but vectors exist.
        """

        points, _ = await qdrant_service.client.scroll(
            collection_name=self.collection,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id),
                    ),
                    FieldCondition(
                        key="user_id",
                        match=MatchValue(value=owner_id),
                    ),
                ]
            ),
            limit=10_000,
            with_payload=True,
            with_vectors=False,
        )

        if not points:
            return None, 0

        sorted_points = sorted(
            points,
            key=lambda point: (point.payload or {}).get("chunk_index", 0),
        )

        parts: list[str] = []
        for point in sorted_points:
            payload = point.payload or {}
            chunk = payload.get("text")
            if chunk and str(chunk).strip():
                parts.append(str(chunk).strip())

        if not parts:
            return None, len(sorted_points)

        return "\n\n".join(parts), len(sorted_points)



    # ==================================================
    # Chunking
    # ==================================================

    def _chunk_text(
        self,
        *,
        text: str,
        size: int = 1200,
        overlap: int = 200,
    ) -> list[str]:


        chunks: list[str] = []


        start = 0

        length = len(text)



        while start < length:


            end = start + size


            chunk = text[start:end]


            if chunk.strip():

                chunks.append(
                    chunk.strip()
                )


            start = end - overlap


            if start < 0:

                start = 0



        return chunks

    def _chunk_spans(
        self,
        *,
        text: str,
        chunks: list[str],
    ) -> list[dict[str, int]]:
        spans: list[dict[str, int]] = []
        cursor = 0

        for chunk in chunks:
            start = text.find(chunk, cursor)
            if start < 0:
                start = cursor
            end = start + len(chunk)
            spans.append(
                {
                    "start_offset": start,
                    "end_offset": end,
                }
            )
            cursor = max(cursor, end)

        return spans


    def _point_id(
        self,
        document_id: str,
        chunk_index: int,
    ) -> str:


        value = (
            f"{document_id}:{chunk_index}"
        )


        digest = hashlib.md5(
            value.encode()
        ).hexdigest()


        return str(
            uuid.UUID(digest)
        )