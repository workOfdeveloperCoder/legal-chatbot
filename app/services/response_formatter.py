from __future__ import annotations

from app.core.config import settings
from app.rag.answer_guard import (
    CitationValidationResult,
    EvidenceStrength,
    GroundingStatus,
)
from app.rag.models import RetrievedChunk, RetrievalMetadata, SourceType
from app.rag.document_identity import (
    chunk_identity_key,
    normalize_document_id,
    recover_document_id,
    resource_identity_key,
)
from app.rag.evidence import (
    offsets_overlap,
    passages_are_near_duplicates,
)
from app.rag.retrieval_pipeline import build_source_reference
from app.rag.section_ids import normalize_hyphens, section_text_needles
from app.schemas.chat import (
    Citation,
    EvidenceHighlight,
    ResourceReference,
    RetrievalMetadataResponse,
    SourceReference,
)

# Resources panel: keep near-top *documents*, then attach their evidence chunks.
_RESOURCE_SCORE_RATIO = 0.90
_RESOURCE_SCORE_GAP = 0.10
_RESOURCE_MAX_DOCUMENTS = 3


class ResponseFormatter:
    """
    Converts RAG output into API response format.

    Separation of concerns:
    - sources[]   = chunk-level evidence (one per retrieved chunk)
    - resources[] = unique documents deduped by document_id
    - citations[] = legal metadata cards for cited chunks
    """

    EXCERPT_LIMIT = 800

    @staticmethod
    def _without_web_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        return [
            chunk
            for chunk in chunks
            if getattr(chunk, "source_type", None) != SourceType.WEB.value
        ]

    def format(
        self,
        *,
        answer: str,
        chunks: list[RetrievedChunk],
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        total_tokens: int | None = None,
        retrieval_metadata: RetrievalMetadata | None = None,
        degraded_sources: list[str] | None = None,
        evidence_strength: EvidenceStrength | None = None,
        grounding_status: GroundingStatus | None = None,
        citation_validation: CitationValidationResult | None = None,
        sources_used: list[int] | None = None,
        used_conversation_context: bool = False,
        build_resources: bool = False,
        query_plan: dict[str, object] | None = None,
        evidence_metadata: dict[str, object] | None = None,
        token_budget: dict[str, object] | None = None,
        pipeline_versions: dict[str, str] | None = None,
        performance: dict[str, object] | None = None,
        llm_execution: dict[str, object] | None = None,
        conversation_context: dict[str, object] | None = None,
        language: dict[str, object] | None = None,
        include_web: bool = False,
    ) -> dict:

        display_chunks = (
            chunks
            if include_web
            else self._without_web_chunks(chunks)
        )

        raw_sources = self._build_sources(display_chunks)
        if build_resources:
            legal_raw = [
                source
                for source in raw_sources
                if (source.source_type or "") == SourceType.LEGAL.value
            ]
            resource_pool = (
                legal_raw
                if legal_raw and not include_web
                else raw_sources
            )
            section_hint = None
            if retrieval_metadata is not None:
                section_hint = (retrieval_metadata.filters_applied or {}).get(
                    "section"
                )
            # Resources: document-level relevance cut + optional section filter.
            # Do not dump the full retrieval limit (weak vector neighbors).
            resource_sources = self._select_resource_sources(
                resource_pool,
                section_hint=section_hint,
            )
            sources = self._prepare_display_sources(
                raw_sources,
                sources_used=sources_used,
                include_all_qdrant=False,
            )
            resources = self._build_resources(
                resource_sources,
                include_web=include_web,
            )
            resources = resources[:_RESOURCE_MAX_DOCUMENTS]
            self._assign_relevance_percents(resources)
        else:
            sources = raw_sources
            resources = []
        if not include_web:
            sources = [
                source
                for source in sources
                if (source.source_type or "") != SourceType.WEB.value
            ]
            resources = [
                resource
                for resource in resources
                if (resource.source_type or "") != SourceType.WEB.value
            ]
        citations = self._build_citations(
            display_chunks,
            sources_used=sources_used,
        )

        metadata_response = None
        if retrieval_metadata is not None:
            legal_used = sum(
                1
                for index in (sources_used or [])
                if 1 <= index <= len(raw_sources)
                and raw_sources[index - 1].source_type
                == SourceType.LEGAL.value
            )
            conversation_used = sum(
                1
                for index in (sources_used or [])
                if 1 <= index <= len(raw_sources)
                and raw_sources[index - 1].source_type
                == SourceType.CONVERSATION.value
            )
            matter_used = sum(
                1
                for index in (sources_used or [])
                if 1 <= index <= len(raw_sources)
                and raw_sources[index - 1].source_type
                == SourceType.MATTER.value
            )
            web_used = (
                sum(
                    1
                    for index in (sources_used or [])
                    if 1 <= index <= len(raw_sources)
                    and raw_sources[index - 1].source_type
                    == SourceType.WEB.value
                )
                if include_web
                else 0
            )

            metadata_response = RetrievalMetadataResponse(
                legal_chunks=retrieval_metadata.legal_chunks,
                conversation_chunks=retrieval_metadata.conversation_chunks,
                matter_chunks=retrieval_metadata.matter_chunks,
                web_chunks=(
                    getattr(retrieval_metadata, "web_chunks", 0)
                    if include_web
                    else 0
                ),
                total_selected=retrieval_metadata.total_selected,
                collections_queried=list(
                    retrieval_metadata.collections_queried
                ),
                degraded_sources=list(
                    degraded_sources
                    or retrieval_metadata.degraded_sources
                ),
                filters_applied=dict(
                    retrieval_metadata.filters_applied
                ),
                evidence_strength=(
                    evidence_strength.value
                    if evidence_strength is not None
                    else None
                ),
                legal_sources_used=legal_used,
                conversation_sources_used=conversation_used,
                matter_sources_used=matter_used,
                web_sources_used=web_used,
                citation_validation_status=(
                    citation_validation.status.value
                    if citation_validation is not None
                    else None
                ),
                used_conversation_context=used_conversation_context,
                query_plan=query_plan,
                evidence_metadata=evidence_metadata,
                token_budget=token_budget,
                pipeline_versions=pipeline_versions,
                performance=performance,
                llm_execution=llm_execution,
                conversation_context=conversation_context,
                language=language,
            )

        return {
            "answer": self._clean_answer(answer),
            "chunks": display_chunks,
            "citations": citations,
            "sources": sources,
            "resources": resources,
            "retrieval_metadata": metadata_response,
            "grounding_status": (
                grounding_status.value
                if grounding_status is not None
                else None
            ),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    def _clean_answer(
        self,
        answer: str,
    ) -> str:
        if not answer:
            return ""
        return answer.strip()

    def _build_sources(
        self,
        chunks: list[RetrievedChunk],
    ) -> list[SourceReference]:
        sources: list[SourceReference] = []

        for index, chunk in enumerate(chunks, 1):
            relevance = chunk.relevance_score
            if relevance is None:
                relevance = chunk.score or 0.0

            document_name = (
                chunk.display_name
                or chunk.filename
                or chunk.title
                or chunk.law_name
                or chunk.document_type
            )

            document_id = recover_document_id(
                document_id=chunk.document_id,
                chunk_id=chunk.chunk_id or chunk.id,
            )

            sources.append(
                SourceReference(
                    id=f"source-{index}",
                    source_number=index,
                    source_type=chunk.source_type or "legal",
                    document_id=document_id,
                    document_name=document_name,
                    display_name=chunk.display_name or document_name,
                    filename=chunk.filename,
                    author=getattr(chunk, "author", None),
                    chunk_id=chunk.chunk_id or chunk.id,
                    chunk_index=chunk.chunk_index,
                    page=chunk.page_number,
                    start_offset=chunk.start_offset,
                    end_offset=chunk.end_offset,
                    paragraph_index=chunk.paragraph_index,
                    section=chunk.section,
                    excerpt=self._excerpt(chunk.text),
                    text=chunk.text,
                    source_reference=build_source_reference(chunk),
                    url=getattr(chunk, "url", None),
                    relevance=round(float(relevance), 4),
                    matter_id=chunk.matter_id,
                    conversation_id=chunk.conversation_id,
                    law_name=chunk.law_name,
                    court=chunk.court,
                    year=chunk.year,
                    sections=list(chunk.sections or []),
                )
            )

        return sources

    def _prepare_display_sources(
        self,
        sources: list[SourceReference],
        *,
        sources_used: list[int] | None = None,
        include_all_qdrant: bool = False,
    ) -> list[SourceReference]:
        filtered = self._filter_sources_for_display(
            sources,
            sources_used=sources_used,
            include_all_qdrant=include_all_qdrant,
        )
        return self._dedupe_sources_by_chunk_identity(filtered)

    def _select_resource_sources(
        self,
        sources: list[SourceReference],
        *,
        section_hint: str | None = None,
    ) -> list[SourceReference]:
        """
        Pick a short document list for the Resources panel.

        - If the user asked about a section and some passages mention it, keep
          only those (any statute — no hardcoding).
        - Rank by best chunk score per document and drop docs far below the top.
        - Keep all evidence chunks for the selected documents.
        """
        pool = [
            source
            for source in sources
            if (source.source_type or "") != SourceType.WEB.value
        ]
        if not pool:
            return []

        if section_hint:
            matched = [
                source
                for source in pool
                if self._source_mentions_section(source, section_hint)
            ]
            if matched:
                pool = matched

        by_doc: dict[str, list[SourceReference]] = {}
        order: list[str] = []
        for source in pool:
            key = (
                normalize_document_id(source.document_id)
                or source.filename
                or source.id
            )
            if key not in by_doc:
                order.append(key)
                by_doc[key] = []
            by_doc[key].append(source)

        ranked_docs = sorted(
            (
                (
                    key,
                    max(float(item.relevance or 0.0) for item in items),
                    items,
                )
                for key, items in by_doc.items()
            ),
            key=lambda row: row[1],
            reverse=True,
        )
        top = ranked_docs[0][1]
        threshold = (
            0.0
            if top <= 0
            else max(top * _RESOURCE_SCORE_RATIO, top - _RESOURCE_SCORE_GAP)
        )

        selected: list[SourceReference] = []
        docs_kept = 0
        for _key, score, items in ranked_docs:
            if score < threshold:
                continue
            selected.extend(
                sorted(items, key=lambda item: item.relevance, reverse=True)
            )
            docs_kept += 1
            if docs_kept >= _RESOURCE_MAX_DOCUMENTS:
                break
        return selected

    def _filter_sources_for_display(
        self,
        sources: list[SourceReference],
        *,
        sources_used: list[int] | None = None,
        include_all_qdrant: bool = False,
    ) -> list[SourceReference]:
        """
        Keep only sources that support the answer.

        Prefer cited sources. If the model cited nothing, keep the strongest
        retrieved evidence. When include_all_qdrant is True (Web off), surface
        all Qdrant tiers up to the retrieval limit — not just the top 3.
        """
        qdrant_only = [
            source
            for source in sources
            if (source.source_type or "") != SourceType.WEB.value
        ]
        pool = qdrant_only if include_all_qdrant else sources

        if sources_used and not include_all_qdrant:
            used = set(sources_used)
            filtered = [
                source
                for source in pool
                if source.source_number in used
            ]
            if filtered:
                return filtered

        if not pool:
            return []

        if include_all_qdrant:
            ranked = sorted(
                pool,
                key=lambda source: source.relevance,
                reverse=True,
            )
            return ranked[: settings.RETRIEVAL_LIMIT]

        ranked = sorted(
            pool,
            key=lambda source: source.relevance,
            reverse=True,
        )
        top = float(ranked[0].relevance or 0.0)
        if top <= 0:
            return ranked[:2]

        # Keep sources close to the top hit; drop weak/irrelevant tails.
        threshold = max(top * 0.72, top - 0.18)
        selected = [
            source
            for source in ranked
            if float(source.relevance or 0.0) >= threshold
        ]
        return selected[:3]

    @staticmethod
    def _source_mentions_section(
        source: SourceReference,
        section_hint: str,
    ) -> bool:
        needles = section_text_needles(section_hint)
        if not needles:
            return False
        blob = normalize_hyphens(
            " ".join(
                [
                    source.text or "",
                    source.excerpt or "",
                    source.section or "",
                    " ".join(source.sections or []),
                    source.law_name or "",
                    source.filename or "",
                    source.document_name or "",
                    source.source_reference or "",
                ]
            )
        ).lower()
        compact = blob.replace("-", "")
        for needle in needles:
            n = needle.lower()
            if n in blob or n.replace("-", "") in compact:
                return True
        return False

    def _dedupe_sources_by_chunk_identity(
        self,
        sources: list[SourceReference],
    ) -> list[SourceReference]:
        seen: dict[str, SourceReference] = {}
        ordered_keys: list[str] = []

        for source in sources:
            key = chunk_identity_key(
                document_id=source.document_id,
                chunk_id=source.chunk_id,
                chunk_index=source.chunk_index,
                source_id=source.id,
            )
            existing = seen.get(key)
            if existing is None:
                seen[key] = source
                ordered_keys.append(key)
                continue

            if source.relevance > existing.relevance:
                seen[key] = source

        return [seen[key] for key in ordered_keys]

    def _dedupe_sources_by_passage(
        self,
        sources: list[SourceReference],
    ) -> list[SourceReference]:
        """Drop overlapping/near-duplicate passages within one document."""
        if len(sources) <= 1:
            return sources

        ranked = sorted(sources, key=lambda item: item.relevance, reverse=True)
        kept: list[SourceReference] = []

        for source in ranked:
            duplicate = False
            for existing in kept:
                if offsets_overlap(
                    left_start=existing.start_offset,
                    left_end=existing.end_offset,
                    right_start=source.start_offset,
                    right_end=source.end_offset,
                ) or passages_are_near_duplicates(
                    existing.text or existing.excerpt,
                    source.text or source.excerpt,
                ):
                    duplicate = True
                    break
            if not duplicate:
                kept.append(source)

        kept_ids = {id(item) for item in kept}
        return [item for item in ranked if id(item) in kept_ids]

    def _build_resources(
        self,
        sources: list[SourceReference],
        *,
        include_web: bool = False,
    ) -> list[ResourceReference]:
        """
        Canonical resource builder.

        - Private docs with the same filename → one resource
        - Otherwise one resource per document_id
        - Multiple chunks → multiple evidence items under that resource
        """
        grouped: dict[str, list[SourceReference]] = {}

        if not include_web:
            sources = [
                source
                for source in sources
                if (source.source_type or "") != SourceType.WEB.value
            ]

        for source in sources:
            key = resource_identity_key(
                document_id=source.document_id,
                chunk_id=source.chunk_id,
                source_id=source.id,
                filename=source.filename,
                source_type=source.source_type,
                display_name=source.display_name or source.document_name,
            )
            grouped.setdefault(key, []).append(source)

        resources: list[ResourceReference] = []

        for key, items in grouped.items():
            items = self._dedupe_sources_by_chunk_identity(items)
            items = self._dedupe_sources_by_passage(items)
            items.sort(key=lambda item: item.relevance, reverse=True)
            if not items:
                continue
            primary = items[0]

            # Prefer a concrete document_id from the strongest evidence item.
            document_id = None
            for item in items:
                document_id = recover_document_id(
                    document_id=item.document_id,
                    chunk_id=item.chunk_id,
                )
                if document_id:
                    break

            evidence = [
                EvidenceHighlight(
                    source_id=item.id,
                    source_number=item.source_number,
                    chunk_id=item.chunk_id,
                    chunk_index=item.chunk_index,
                    excerpt=item.excerpt,
                    text=item.text,
                    start_offset=item.start_offset,
                    end_offset=item.end_offset,
                    page=item.page,
                    relevance=item.relevance,
                )
                for item in items
            ]

            excerpt_parts = [
                item.excerpt for item in items if item.excerpt
            ]
            primary_excerpt = excerpt_parts[0] if excerpt_parts else None
            if len(excerpt_parts) > 1:
                primary_excerpt = self._excerpt(
                    " … ".join(excerpt_parts)
                )

            resource_id = (
                f"resource-{document_id}"
                if document_id
                else f"resource-{key}"
            )

            resources.append(
                ResourceReference(
                    id=resource_id,
                    document_id=document_id,
                    display_name=primary.display_name or primary.document_name,
                    filename=primary.filename,
                    author=primary.author,
                    source_type=primary.source_type,
                    matter_id=primary.matter_id,
                    conversation_id=primary.conversation_id,
                    law_name=primary.law_name,
                    court=primary.court,
                    year=primary.year,
                    url=getattr(primary, "url", None),
                    source_ids=[item.id for item in items],
                    source_numbers=[item.source_number for item in items],
                    evidence=evidence,
                    primary_excerpt=primary_excerpt,
                    relevance=max(item.relevance for item in items),
                )
            )

        resources.sort(key=lambda item: item.relevance, reverse=True)
        resources = self._dedupe_resources(resources)
        for resource in resources:
            resource.evidence = self._collapse_evidence_to_one(
                self._dedupe_evidence_highlights(resource.evidence)
            )
            resource.source_ids = [item.source_id for item in resource.evidence]
            resource.source_numbers = [
                item.source_number for item in resource.evidence
            ]
            if resource.evidence:
                resource.primary_excerpt = resource.evidence[0].excerpt
                resource.relevance = max(
                    item.relevance for item in resource.evidence
                )
        self._assign_relevance_percents(resources)
        return resources

    @staticmethod
    def _assign_relevance_percents(
        resources: list[ResourceReference],
    ) -> None:
        max_rel = max((item.relevance for item in resources), default=0.0)
        for resource in resources:
            if max_rel > 0:
                resource.relevance_percent = round(
                    (resource.relevance / max_rel) * 100
                )
                for highlight in resource.evidence:
                    highlight.relevance_percent = round(
                        (highlight.relevance / max_rel) * 100
                    )
            else:
                resource.relevance_percent = 0

    @staticmethod
    def _collapse_evidence_to_one(
        evidence: list[EvidenceHighlight],
    ) -> list[EvidenceHighlight]:
        """One resource → one evidence blob (no Passage 1 / 2)."""
        if len(evidence) <= 1:
            return evidence

        ranked = sorted(evidence, key=lambda item: item.relevance, reverse=True)
        primary = ranked[0]

        parts: list[str] = []
        for item in ranked:
            text = (item.excerpt or item.text or "").strip()
            if not text:
                continue
            if any(
                passages_are_near_duplicates(text, existing)
                or text in existing
                or existing in text
                for existing in parts
            ):
                # Keep the longer version when one contains the other.
                for idx, existing in enumerate(parts):
                    if text in existing or existing in text or passages_are_near_duplicates(
                        text, existing
                    ):
                        if len(text) > len(existing):
                            parts[idx] = text
                        break
                continue
            parts.append(text)

        combined = "\n\n".join(parts).strip() if parts else (primary.excerpt or primary.text)
        primary.excerpt = combined
        if primary.text:
            primary.text = combined
        return [primary]

    @staticmethod
    def _dedupe_evidence_highlights(
        evidence: list[EvidenceHighlight],
    ) -> list[EvidenceHighlight]:
        if len(evidence) <= 1:
            return evidence

        ranked = sorted(evidence, key=lambda item: item.relevance, reverse=True)
        kept: list[EvidenceHighlight] = []

        for item in ranked:
            duplicate = False
            for existing in kept:
                if offsets_overlap(
                    left_start=existing.start_offset,
                    left_end=existing.end_offset,
                    right_start=item.start_offset,
                    right_end=item.end_offset,
                ) or passages_are_near_duplicates(
                    existing.text or existing.excerpt,
                    item.text or item.excerpt,
                ):
                    duplicate = True
                    break
            if not duplicate:
                kept.append(item)

        kept_ids = {id(item) for item in kept}
        return [item for item in ranked if id(item) in kept_ids]

    def _dedupe_resources(
        self,
        resources: list[ResourceReference],
    ) -> list[ResourceReference]:
        """HARD INVARIANT: one document_id = one resource."""
        return self._dedupe_resources_by_document_id(resources)

    @staticmethod
    def _merge_resource_evidence(
        existing: ResourceReference,
        incoming: ResourceReference,
    ) -> None:
        seen = {
            chunk_identity_key(
                document_id=existing.document_id or incoming.document_id,
                chunk_id=item.chunk_id,
                chunk_index=item.chunk_index,
                source_id=item.source_id,
            )
            for item in existing.evidence
        }

        for item in incoming.evidence:
            item_key = chunk_identity_key(
                document_id=incoming.document_id or existing.document_id,
                chunk_id=item.chunk_id,
                chunk_index=item.chunk_index,
                source_id=item.source_id,
            )
            if item_key in seen:
                continue

            near_duplicate = False
            for current in existing.evidence:
                if offsets_overlap(
                    left_start=current.start_offset,
                    left_end=current.end_offset,
                    right_start=item.start_offset,
                    right_end=item.end_offset,
                ) or passages_are_near_duplicates(
                    current.text or current.excerpt,
                    item.text or item.excerpt,
                ):
                    near_duplicate = True
                    break
            if near_duplicate:
                continue

            existing.evidence.append(item)
            seen.add(item_key)
            if item.source_id not in existing.source_ids:
                existing.source_ids.append(item.source_id)
            if item.source_number not in existing.source_numbers:
                existing.source_numbers.append(item.source_number)

        if incoming.relevance > existing.relevance:
            existing.relevance = incoming.relevance
            if incoming.primary_excerpt:
                existing.primary_excerpt = incoming.primary_excerpt
            if incoming.document_id and (
                not existing.document_id
                or incoming.relevance >= existing.relevance
            ):
                existing.document_id = incoming.document_id
                existing.id = f"resource-{incoming.document_id}"

        if incoming.display_name and not existing.display_name:
            existing.display_name = incoming.display_name
        if incoming.filename and not existing.filename:
            existing.filename = incoming.filename
        if incoming.author and not existing.author:
            existing.author = incoming.author

    def _dedupe_resources_by_document_id(
        self,
        resources: list[ResourceReference],
    ) -> list[ResourceReference]:
        merged: dict[str, ResourceReference] = {}
        order: list[str] = []

        for resource in resources:
            doc_id = normalize_document_id(resource.document_id)
            key = f"doc:{doc_id}" if doc_id else resource.id

            existing = merged.get(key)
            if existing is None:
                if doc_id:
                    resource.document_id = doc_id
                    resource.id = f"resource-{doc_id}"
                merged[key] = resource
                order.append(key)
                continue

            self._merge_resource_evidence(existing, resource)

        return [merged[key] for key in order]

    def _build_citations(
        self,
        chunks: list[RetrievedChunk],
        *,
        sources_used: list[int] | None = None,
    ) -> list[Citation]:
        if sources_used:
            filtered = [
                chunks[index - 1]
                for index in sorted(set(sources_used))
                if 1 <= index <= len(chunks)
            ]
            if filtered:
                chunks = filtered

        merged: dict[str, Citation] = {}
        excerpt_parts: dict[str, list[str]] = {}

        for chunk in chunks:
            key = self._citation_merge_key(chunk)
            excerpt = self._excerpt(chunk.text)
            score = round(
                chunk.relevance_score
                if chunk.relevance_score is not None
                else (chunk.score or 0.0),
                4,
            )

            if key not in merged:
                excerpt_parts[key] = [excerpt] if excerpt else []
                merged[key] = Citation(
                    id=chunk.chunk_id or chunk.id,
                    document_id=chunk.document_id,
                    excerpt=excerpt,
                    filename=chunk.filename,
                    title=chunk.display_name or chunk.title,
                    heading=chunk.heading,
                    law_name=chunk.law_name,
                    document_type=chunk.document_type,
                    category=chunk.category,
                    sub_category=chunk.sub_category,
                    practice_area=chunk.practice_area,
                    court=chunk.court,
                    year=chunk.year,
                    jurisdiction=chunk.jurisdiction,
                    sections=list(chunk.sections or []),
                    keywords=list((chunk.keywords or [])[:12]),
                    summary=chunk.summary,
                    score=score,
                    source_type=chunk.source_type,
                    chunk_id=chunk.chunk_id or chunk.id,
                    page=chunk.page_number,
                    source_reference=build_source_reference(chunk),
                    url=getattr(chunk, "url", None),
                    matter_id=chunk.matter_id,
                    conversation_id=chunk.conversation_id,
                    display_name=chunk.display_name,
                )
                continue

            existing = merged[key]
            if score > existing.score:
                existing.score = score
                existing.id = chunk.chunk_id or chunk.id or existing.id

            if excerpt and excerpt not in excerpt_parts[key]:
                excerpt_parts[key].append(excerpt)

            existing.sections = self._unique_preserve(
                list(existing.sections or []) + list(chunk.sections or [])
            )
            existing.keywords = self._unique_preserve(
                list(existing.keywords or []) + list(chunk.keywords or [])
            )[:12]

            for field in (
                "document_id",
                "filename",
                "title",
                "heading",
                "law_name",
                "document_type",
                "category",
                "sub_category",
                "practice_area",
                "court",
                "year",
                "jurisdiction",
                "summary",
                "source_type",
                "chunk_id",
                "page",
                "source_reference",
                "matter_id",
                "conversation_id",
                "display_name",
            ):
                if getattr(existing, field) in (None, "", []):
                    value = getattr(chunk, field, None)
                    if value not in (None, "", []):
                        setattr(existing, field, value)

        for key, citation in merged.items():
            parts = excerpt_parts.get(key) or []
            if not parts:
                citation.excerpt = None
            elif len(parts) == 1:
                citation.excerpt = parts[0]
            else:
                combined = " … ".join(parts)
                citation.excerpt = self._excerpt(combined)

        citations = list(merged.values())
        citations.sort(key=lambda x: x.score, reverse=True)
        return citations

    @staticmethod
    def _citation_merge_key(chunk: RetrievedChunk) -> str:
        doc_id = recover_document_id(
            document_id=chunk.document_id,
            chunk_id=chunk.chunk_id or chunk.id,
        )
        if doc_id and chunk.chunk_index is not None:
            return f"doc:{doc_id}:{chunk.chunk_index}"

        if doc_id and chunk.chunk_id:
            return f"doc:{doc_id}:chunk:{chunk.chunk_id}"

        if doc_id:
            return f"doc:{doc_id}"

        if chunk.law_name and chunk.sections:
            return (
                f"law:{chunk.law_name}:"
                f"{','.join(chunk.sections)}"
            )

        return f"chunk:{chunk.id or (chunk.text or '')[:40]}"

    @staticmethod
    def _unique_preserve(values: list) -> list:
        seen: set[str] = set()
        out: list = []
        for value in values:
            if value is None:
                continue
            marker = str(value).strip().lower()
            if not marker or marker in seen:
                continue
            seen.add(marker)
            out.append(value)
        return out

    def _excerpt(self, text: str | None) -> str | None:
        if not text:
            return None

        cleaned = " ".join(text.split())
        if len(cleaned) <= self.EXCERPT_LIMIT:
            return cleaned

        return cleaned[: self.EXCERPT_LIMIT - 1].rstrip() + "…"
