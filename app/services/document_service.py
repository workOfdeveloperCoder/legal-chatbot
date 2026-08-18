from __future__ import annotations

import logging
import uuid
from pathlib import Path

from fastapi import UploadFile, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import (
    Document,
    DocumentScope,
)

from app.document.extractor import DocumentExtractor
from app.repositories.conversation_repository import ConversationRepository
from app.repositories.matter_repository import MatterRepository
from app.models.activity_log import ActivityAction
from app.services.activity_log_service import ActivityLogService
from app.schemas.document import (
    MatterDocumentDetail,
    MatterDocumentItem,
    TEXT_PREVIEW_LIMIT,
)
from app.vector.service import VectorService


logger = logging.getLogger(__name__)


class DocumentService:
    """
    Production Document Management Service.


    Upload Pipeline:

        Upload File
             |
             v
        Validate Ownership
             |
             v
        Save File Storage
             |
             v
        Create PostgreSQL Document Record
             |
             v
        Extract Text
             |
             v
        Generate Embeddings
             |
             v
        Store Chunks in Qdrant


    Storage Separation:

    PostgreSQL
    -----------
        Document metadata


    Qdrant
    -------
        Document chunks + embeddings


    Security:

        Every vector contains:

            user_id


    Visibility:

        MATTER DOCUMENT

            User
             |
             Matter
                |
                Conversations


        CONVERSATION DOCUMENT

            User
             |
             Conversation only

    """


    MAX_FILE_SIZE = 25 * 1024 * 1024


    def __init__(
        self,
        *,
        db: AsyncSession,
        repository,
        matter_repository: MatterRepository,
        conversation_repository: ConversationRepository,
        vector_service: VectorService,
        extractor: DocumentExtractor,
        activity_log_service: ActivityLogService,
    ) -> None:

        self.db = db

        self.repo = repository

        self.matters = matter_repository

        self.conversations = conversation_repository

        self.vector_service = vector_service

        self.extractor = extractor

        self.activity_logs = activity_log_service



    async def upload(
        self,
        *,
        user_id,
        file: UploadFile,
        matter_id=None,
        conversation_id=None,
    ) -> Document:


        # ==================================================
        # Validate ownership scope
        # ==================================================

        if matter_id and conversation_id:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Document cannot belong "
                    "to matter and conversation "
                    "at the same time"
                ),
            )


        if not matter_id and not conversation_id:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Document must belong "
                    "to a matter or conversation"
                ),
            )



        if matter_id:

            matter = await self.matters.get(matter_id)

            if matter is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Matter not found.",
                )

            if matter.owner_id != user_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=(
                        "You don't have permission "
                        "to access this matter."
                    ),
                )

            scope = DocumentScope.MATTER

        else:

            conversation = await self.conversations.get(
                conversation_id,
            )

            if conversation is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Conversation not found.",
                )

            if conversation.user_id != user_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Forbidden.",
                )

            scope = DocumentScope.CONVERSATION



        # ==================================================
        # Validate file
        # ==================================================

        if not file.filename:

            raise HTTPException(
                status_code=400,
                detail="Filename missing",
            )



        content = await file.read()


        if len(content) > self.MAX_FILE_SIZE:

            raise HTTPException(
                status_code=413,
                detail="File size exceeds 25MB limit",
            )



        # ==================================================
        # Save file
        # ==================================================

        safe_name = Path(file.filename).name

        if not safe_name or safe_name in {".", ".."}:

            raise HTTPException(
                status_code=400,
                detail="Filename missing",
            )


        upload_dir = Path(
            "storage/documents"
        ).resolve()


        upload_dir.mkdir(
            parents=True,
            exist_ok=True,
        )



        stored_filename = (
            f"{uuid.uuid4()}_{safe_name}"
        )


        storage_path = (
            upload_dir /
            stored_filename
        ).resolve()


        if not storage_path.is_relative_to(upload_dir):

            raise HTTPException(
                status_code=400,
                detail="Invalid filename",
            )


        storage_path.write_bytes(
            content
        )


        logger.info(
            "Document saved user=%s file=%s",
            user_id,
            safe_name,
        )



        # ==================================================
        # Create PostgreSQL record
        # ==================================================

        document = Document(

            owner_id=user_id,

            matter_id=matter_id,

            conversation_id=conversation_id,

            scope=scope,

            filename=file.filename,

            mime_type=file.content_type,

            storage_path=str(storage_path),

            processed=False,

            vectorized=False,

        )


        document = await self.repo.create(
            document
        )



        # ==================================================
        # Extract + Vectorize
        # ==================================================

        try:

            extracted_text = await self.extractor.extract(

                path=str(storage_path),

                mime_type=file.content_type,

            )



            if not extracted_text or not str(extracted_text).strip():
                # Empty extract must NOT be treated as successfully processed.
                # Otherwise summarization falls through to the global corpus.
                logger.warning(
                    "Document extraction empty document=%s mime=%s file=%s",
                    document.id,
                    file.content_type,
                    safe_name,
                )
                document.extracted_text = None
                document.processed = False
                document.vectorized = False
            else:
                document.extracted_text = extracted_text

                await self.vector_service.index_document(

                    document_id=str(
                        document.id
                    ),

                    owner_id=str(
                        user_id
                    ),


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


                    scope=scope.value,


                    text=extracted_text,

                    filename=safe_name,

                    mime_type=file.content_type,

                )


                document.vectorized = True
                document.processed = True



        except Exception:


            logger.exception(
                "Document processing failed document=%s",
                document.id,
            )


            document.processed = False
            document.vectorized = False



        # ==================================================
        # Update DB
        # ==================================================

        document = await self.repo.update(
            document
        )

        await self.db.commit()

        scope_label = "matter" if matter_id else "conversation"
        scope_id = matter_id or conversation_id

        await self.activity_logs.record(
            action=ActivityAction.DOCUMENT_UPLOADED,
            summary=f'Uploaded document "{file.filename}"',
            user_id=user_id,
            entity_type="document",
            entity_id=document.id,
            metadata={
                "filename": file.filename,
                "scope": scope_label,
                "scope_id": str(scope_id),
                "mime_type": file.content_type,
                "processed": document.processed,
                "vectorized": document.vectorized,
            },
        )

        return document

    # ==================================================
    # Matter document library
    # ==================================================

    async def list_matter_documents(
        self,
        *,
        user_id,
        matter_id,
    ) -> list[MatterDocumentItem]:
        matter = await self.matters.get(matter_id)

        if matter is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Matter not found.",
            )

        if matter.owner_id != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to access this matter.",
            )

        documents = await self.repo.list_matter_documents(
            owner_id=user_id,
            matter_id=matter_id,
        )

        items: list[MatterDocumentItem] = []
        for document in documents:
            items.append(self._to_matter_document_item(document))

        return items

    def _to_matter_document_item(
        self,
        document: Document,
    ) -> MatterDocumentItem:
        preview_source = (
            document.extracted_text.strip()
            if document.extracted_text and document.extracted_text.strip()
            else None
        )

        scope_value = (
            document.scope.value
            if hasattr(document.scope, "value")
            else str(document.scope)
        )

        return MatterDocumentItem(
            id=document.id,
            filename=document.filename,
            type="document",
            scope=scope_value,
            matter_id=document.matter_id,
            mime_type=document.mime_type,
            text_preview=self._build_text_preview(preview_source),
            character_count=len(preview_source) if preview_source else 0,
            processed=document.processed,
            vectorized=document.vectorized,
            created_at=document.created_at,
            updated_at=document.updated_at,
        )

    async def _to_matter_document_detail(
        self,
        document: Document,
        *,
        owner_id: str,
    ) -> MatterDocumentDetail:
        normalized_text, chunk_count = await self._resolve_merged_text(
            document,
            owner_id=owner_id,
        )

        scope_value = (
            document.scope.value
            if hasattr(document.scope, "value")
            else str(document.scope)
        )

        return MatterDocumentDetail(
            id=document.id,
            filename=document.filename,
            type="document",
            scope=scope_value,
            matter_id=document.matter_id,
            conversation_id=document.conversation_id,
            mime_type=document.mime_type,
            text=normalized_text,
            text_preview=self._build_text_preview(normalized_text),
            character_count=len(normalized_text) if normalized_text else 0,
            chunk_count=chunk_count,
            processed=document.processed,
            vectorized=document.vectorized,
            created_at=document.created_at,
            updated_at=document.updated_at,
        )

    async def get_matter_document(
        self,
        *,
        user_id,
        matter_id,
        document_id,
    ) -> MatterDocumentDetail:
        document = await self._get_owned_matter_document(
            user_id=user_id,
            matter_id=matter_id,
            document_id=document_id,
        )

        return await self._to_matter_document_detail(
            document,
            owner_id=str(user_id),
        )

    async def get_conversation_document(
        self,
        *,
        user_id,
        conversation_id,
        document_id,
    ) -> MatterDocumentDetail:
        document = await self._get_owned_conversation_document(
            user_id=user_id,
            conversation_id=conversation_id,
            document_id=document_id,
        )

        return await self._to_matter_document_detail(
            document,
            owner_id=str(user_id),
        )

    async def delete_matter_document(
        self,
        *,
        user_id,
        matter_id,
        document_id,
    ) -> None:
        document = await self._get_owned_matter_document(
            user_id=user_id,
            matter_id=matter_id,
            document_id=document_id,
        )

        await self._delete_qdrant_vectors(
            document_id=str(document.id),
            owner_id=str(user_id),
        )

        self._delete_storage_file(document.storage_path)

        await self.repo.delete(document)
        await self.db.commit()

        logger.info(
            "Matter document deleted user=%s matter=%s document=%s file=%s",
            user_id,
            matter_id,
            document.id,
            document.filename,
        )

    async def _get_owned_matter_document(
        self,
        *,
        user_id,
        matter_id,
        document_id,
    ) -> Document:
        matter = await self.matters.get(matter_id)

        if matter is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Matter not found.",
            )

        if matter.owner_id != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to access this matter.",
            )

        document = await self.repo.get_by_id(document_id)

        if document is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found.",
            )

        if document.owner_id != user_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found.",
            )

        if (
            document.scope != DocumentScope.MATTER
            or document.matter_id != matter_id
        ):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found.",
            )

        return document

    async def _get_owned_conversation_document(
        self,
        *,
        user_id,
        conversation_id,
        document_id,
    ) -> Document:
        conversation = await self.conversations.get(conversation_id)

        if conversation is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found.",
            )

        if conversation.user_id != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Forbidden.",
            )

        document = await self.repo.get_by_id(document_id)

        if document is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found.",
            )

        if document.owner_id != user_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found.",
            )

        if (
            document.scope != DocumentScope.CONVERSATION
            or document.conversation_id != conversation_id
        ):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found.",
            )

        return document

    async def _delete_qdrant_vectors(
        self,
        *,
        document_id: str,
        owner_id: str,
    ) -> None:
        try:
            await self.vector_service.delete_document(
                document_id=document_id,
                owner_id=owner_id,
            )
        except Exception:
            logger.warning(
                "Qdrant vector delete skipped or failed document=%s",
                document_id,
                exc_info=True,
            )

    @staticmethod
    def _delete_storage_file(storage_path: str | None) -> None:
        if not storage_path:
            return

        file_path = Path(storage_path)
        if not file_path.is_file():
            return

        try:
            file_path.unlink()
        except OSError:
            logger.warning(
                "Failed to delete document file path=%s",
                storage_path,
                exc_info=True,
            )

    async def _resolve_merged_text(
        self,
        document: Document,
        *,
        owner_id: str,
    ) -> tuple[str | None, int | None]:
        if document.extracted_text and document.extracted_text.strip():
            return document.extracted_text.strip(), None

        if document.vectorized:
            merged_text, chunk_count = (
                await self.vector_service.get_merged_document_text(
                    document_id=str(document.id),
                    owner_id=owner_id,
                )
            )
            if merged_text and merged_text.strip():
                return merged_text.strip(), chunk_count

        return None, None

    @staticmethod
    def _build_text_preview(text: str | None) -> str | None:
        if not text:
            return None

        cleaned = " ".join(text.split())
        if len(cleaned) <= TEXT_PREVIEW_LIMIT:
            return cleaned

        return cleaned[: TEXT_PREVIEW_LIMIT - 1].rstrip() + "…"