from __future__ import annotations

from uuid import UUID

from sqlalchemy import and_, desc, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document, DocumentScope


class DocumentRepository:

    def __init__(
        self,
        db: AsyncSession,
    ):
        self.db = db

    async def create(
        self,
        document: Document,
    ):
        self.db.add(document)
        await self.db.flush()
        await self.db.refresh(document)
        return document

    async def update(
        self,
        document: Document,
    ):
        await self.db.flush()
        await self.db.refresh(document)
        return document

    async def get_by_id(
        self,
        document_id,
    ):
        # Legal-corpus / web chunk ids are often not Postgres UUIDs.
        # Calling db.get with them raises asyncpg DataError and aborts the chat.
        try:
            if isinstance(document_id, UUID):
                uid = document_id
            else:
                uid = UUID(str(document_id).strip())
        except (TypeError, ValueError, AttributeError):
            return None
        return await self.db.get(
            Document,
            uid,
        )

    async def delete(
        self,
        document: Document,
    ) -> None:
        await self.db.delete(document)
        await self.db.flush()

    async def has_for_scope(
        self,
        *,
        owner_id: UUID,
        conversation_id: UUID | None = None,
        matter_id: UUID | None = None,
    ) -> bool:
        """
        True when the user has at least one document ready for Q&A
        in the given conversation and/or matter scope.
        """
        scope_filters = []

        if conversation_id is not None:
            scope_filters.append(
                and_(
                    Document.scope == DocumentScope.CONVERSATION,
                    Document.conversation_id == conversation_id,
                )
            )

        if matter_id is not None:
            scope_filters.append(
                and_(
                    Document.scope == DocumentScope.MATTER,
                    Document.matter_id == matter_id,
                )
            )
            # Conversation uploads stamped with this matter_id (matter-wide).
            scope_filters.append(
                and_(
                    Document.scope == DocumentScope.CONVERSATION,
                    Document.matter_id == matter_id,
                )
            )

        if not scope_filters:
            return False

        result = await self.db.execute(
            select(Document.id)
            .where(
                Document.owner_id == owner_id,
                Document.processed.is_(True),
                Document.vectorized.is_(True),
                or_(*scope_filters),
            )
            .limit(1)
        )
        return result.scalar_one_or_none() is not None

    async def list_matter_documents(
        self,
        *,
        owner_id: UUID,
        matter_id: UUID,
    ) -> list[Document]:
        result = await self.db.execute(
            select(Document)
            .where(
                Document.owner_id == owner_id,
                Document.matter_id == matter_id,
                Document.scope == DocumentScope.MATTER,
            )
            .order_by(desc(Document.created_at))
        )
        return list(result.scalars().all())

    async def list_by_matter(
        self,
        matter_id,
    ):
        result = await self.db.execute(
            select(Document).where(
                Document.matter_id == matter_id
            )
        )
        return list(result.scalars().all())

    async def list_conversation_documents(
        self,
        *,
        owner_id: UUID,
        conversation_id: UUID,
    ) -> list[Document]:
        result = await self.db.execute(
            select(Document)
            .where(
                Document.owner_id == owner_id,
                Document.conversation_id == conversation_id,
                Document.scope == DocumentScope.CONVERSATION,
            )
            .order_by(desc(Document.created_at))
        )
        return list(result.scalars().all())

    async def list_for_scope(
        self,
        *,
        owner_id: UUID,
        conversation_id: UUID | None = None,
        matter_id: UUID | None = None,
    ) -> list[Document]:
        """
        Matter files first (newest), then conversation-only files.
        Dedupes if a row somehow appears in both lists.
        """
        found: list[Document] = []
        seen: set[UUID] = set()
        if matter_id is not None:
            for document in await self.list_matter_documents(
                owner_id=owner_id,
                matter_id=matter_id,
            ):
                found.append(document)
                seen.add(document.id)
        if conversation_id is not None:
            for document in await self.list_conversation_documents(
                owner_id=owner_id,
                conversation_id=conversation_id,
            ):
                if document.id not in seen:
                    found.append(document)
                    seen.add(document.id)
        return found
