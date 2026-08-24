from __future__ import annotations

import logging

from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    PayloadSchemaType,
    VectorParams,
)

from app.core.config import settings
from app.vector.client import build_async_qdrant_client, qdrant_endpoint_label


logger = logging.getLogger(__name__)

# Fields actually used in QdrantFilterBuilder for private collections.
_PRIVATE_PAYLOAD_INDEXES: tuple[str, ...] = (
    "user_id",
    "document_id",
    "scope",
    "matter_id",
    "conversation_id",
)


class QdrantService:
    """
    Central Qdrant manager.

    Collections:

    legal_documents
        Public Pakistani legal corpus. Owned by Legal GPT ingestion.
        This app never creates or recreates it.

    chatbot_documents
        User uploaded private documents (matter + conversation).

    user_memory
        AI memory vectors (not document chunks).
    """

    def __init__(self) -> None:
        self.client = build_async_qdrant_client()

    @property
    def legal_collection(self) -> str:
        return settings.LEGAL_QDRANT_COLLECTION

    @property
    def document_collection(self) -> str:
        return settings.USER_DOCUMENT_COLLECTION

    @property
    def memory_collection(self) -> str:
        return settings.USER_MEMORY_COLLECTION

    async def initialize(self) -> None:
        """
        Ensure chatbot collections exist. Never recreate legal_documents.
        """
        endpoint = qdrant_endpoint_label()
        has_key = bool((settings.QDRANT_API_KEY or "").strip())
        if settings.is_production and not has_key:
            raise RuntimeError(
                "QDRANT_API_KEY is required when ENVIRONMENT=production. "
                "Refusing to connect to Qdrant without authentication."
            )
        try:
            await self._ensure_collection(self.document_collection)
            await self._ensure_payload_indexes(self.document_collection)
            await self._ensure_collection(self.memory_collection)
            await self._ensure_payload_indexes(self.memory_collection)
        except Exception as exc:
            auth_hint = " (check QDRANT_API_KEY)" if has_key else ""
            raise RuntimeError(
                f"Cannot connect to Qdrant at {endpoint}{auth_hint}. "
                "Start the Qdrant service on the private network, then retry."
            ) from exc

        logger.info(
            "Qdrant initialization completed endpoint=%s api_key_configured=%s",
            endpoint,
            has_key,
        )

    async def _ensure_collection(self, collection_name: str) -> None:
        exists = await self.client.collection_exists(
            collection_name=collection_name,
        )
        if exists:
            logger.info("Qdrant collection exists %s", collection_name)
            return

        await self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=settings.VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )
        logger.info("Qdrant collection created %s", collection_name)

    async def _ensure_payload_indexes(self, collection_name: str) -> None:
        for field_name in _PRIVATE_PAYLOAD_INDEXES:
            try:
                await self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            except UnexpectedResponse as exc:
                # Already indexed or older server — do not fail startup.
                logger.debug(
                    "Payload index skip collection=%s field=%s detail=%s",
                    collection_name,
                    field_name,
                    exc,
                )
            except Exception:
                logger.exception(
                    "Payload index failed collection=%s field=%s",
                    collection_name,
                    field_name,
                )

    async def health(self) -> bool:
        try:
            await self.client.get_collections()
            return True
        except Exception as exc:
            logger.error("Qdrant health failed %s", exc)
            return False

    async def collection_info(self, collection_name: str) -> dict[str, object]:
        info = await self.client.get_collection(collection_name)
        vectors = getattr(info.config.params, "vectors", None)
        size = getattr(vectors, "size", None) if vectors is not None else None
        distance = getattr(vectors, "distance", None) if vectors is not None else None
        return {
            "name": collection_name,
            "points_count": getattr(info, "points_count", None),
            "vector_size": size,
            "distance": str(distance) if distance is not None else None,
        }

    async def close(self) -> None:
        await self.client.close()


qdrant_service = QdrantService()
