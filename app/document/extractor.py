from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader


class DocumentExtractor:

    async def extract(
        self,
        path: str,
        mime_type: str | None,
    ) -> str:

        file = Path(path)
        suffix = file.suffix.lower()
        mime = (mime_type or "").lower()

        is_pdf = (
            mime == "application/pdf"
            or (mime == "application/octet-stream" and suffix == ".pdf")
            or suffix == ".pdf"
        )

        if is_pdf:
            reader = PdfReader(file)
            pages = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
            return "\n".join(pages)

        is_text = (
            mime.startswith("text/")
            or suffix in {".txt", ".md", ".csv"}
        )

        if is_text:
            return file.read_text(encoding="utf-8", errors="ignore")

        return ""


extractor = DocumentExtractor()
