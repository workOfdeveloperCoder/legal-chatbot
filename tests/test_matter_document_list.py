from __future__ import annotations

import uuid
from datetime import datetime, timezone

from app.schemas.document import MatterDocumentDetail, MatterDocumentItem, TEXT_PREVIEW_LIMIT
from app.services.document_service import DocumentService


class TestMatterDocumentPresentation:
    def test_text_preview_truncates_long_content(self):
        long_text = "A" * (TEXT_PREVIEW_LIMIT + 50)
        preview = DocumentService._build_text_preview(long_text)

        assert preview is not None
        assert len(preview) <= TEXT_PREVIEW_LIMIT
        assert preview.endswith("…")

    def test_matter_document_item_list_shape_has_no_full_text(self):
        now = datetime.now(timezone.utc)
        matter_id = uuid.uuid4()

        item = MatterDocumentItem(
            id=uuid.uuid4(),
            filename="2002J18.txt",
            type="document",
            scope="matter",
            matter_id=matter_id,
            mime_type="text/plain",
            text_preview="Section 497 text content.",
            character_count=26,
            processed=True,
            vectorized=True,
            created_at=now,
            updated_at=now,
        )

        assert item.filename == "2002J18.txt"
        assert item.type == "document"
        assert item.ready_for_qa is True
        assert not hasattr(item, "text") or "text" not in item.model_fields

    def test_matter_document_detail_includes_full_text(self):
        now = datetime.now(timezone.utc)
        matter_id = uuid.uuid4()

        detail = MatterDocumentDetail(
            id=uuid.uuid4(),
            filename="2000J8.txt",
            scope="matter",
            matter_id=matter_id,
            text="Full merged document body.",
            text_preview="Full merged document body.",
            character_count=28,
            chunk_count=4,
            processed=True,
            vectorized=True,
            created_at=now,
            updated_at=now,
        )

        assert detail.text == "Full merged document body."
        assert detail.chunk_count == 4

    def test_two_files_are_two_objects(self):
        now = datetime.now(timezone.utc)
        matter_id = uuid.uuid4()

        items = [
            MatterDocumentItem(
                id=uuid.uuid4(),
                filename="2002J18.txt",
                scope="matter",
                matter_id=matter_id,
                processed=True,
                vectorized=True,
                created_at=now,
                updated_at=now,
            ),
            MatterDocumentItem(
                id=uuid.uuid4(),
                filename="2000J8.txt",
                scope="matter",
                matter_id=matter_id,
                processed=True,
                vectorized=True,
                created_at=now,
                updated_at=now,
            ),
        ]

        assert len(items) == 2
        assert items[0].filename != items[1].filename
