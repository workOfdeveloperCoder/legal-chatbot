from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from app.rag.answer_guard import EvidenceStrength
from app.rag.authority import (
    AUTHORITY_RANK,
    AuthorityType,
    classify_authority,
)
from app.rag.evidence_assessment import (
    EvidenceAssessment,
    assess_evidence_for_question,
)
from app.rag.models import RetrievedChunk, SourceType


class EvidenceRole(str, Enum):
    RETRIEVED = "retrieved"
    RELEVANT = "relevant"
    SUPPORTING = "supporting"
    CONTRADICTING = "contradicting"
    WEAK = "weak"


@dataclass(slots=True)
class EvidenceItem:
    chunk: RetrievedChunk
    authority_type: AuthorityType
    role: EvidenceRole = EvidenceRole.RETRIEVED
    support_strength: float = 0.0
    relevance_score: float = 0.0

    @property
    def document_id(self) -> str | None:
        return self.chunk.document_id

    @property
    def chunk_id(self) -> str | None:
        return self.chunk.chunk_id or self.chunk.id

    @property
    def source_type(self) -> str:
        return self.chunk.source_type or SourceType.LEGAL.value

    @property
    def excerpt(self) -> str:
        return (self.chunk.text or "")[:800]


@dataclass(slots=True)
class EvidenceBundle:
    items: list[EvidenceItem] = field(default_factory=list)
    assessment: EvidenceAssessment | None = None
    overall_strength: EvidenceStrength = EvidenceStrength.NONE

    @property
    def chunks(self) -> list[RetrievedChunk]:
        return [item.chunk for item in self.items]

    def supporting(self) -> list[EvidenceItem]:
        return [
            item
            for item in self.items
            if item.role in {EvidenceRole.SUPPORTING, EvidenceRole.RELEVANT}
        ]

    def contradicting(self) -> list[EvidenceItem]:
        return [
            item for item in self.items if item.role == EvidenceRole.CONTRADICTING
        ]

    def to_metadata(self) -> dict[str, object]:
        return {
            "evidence_count": len(self.items),
            "supporting_count": len(self.supporting()),
            "contradicting_count": len(self.contradicting()),
            "overall_strength": self.overall_strength.value,
            "authority_types": sorted(
                {item.authority_type.value for item in self.items}
            ),
            "authority_strength": (
                self.assessment.authority_strength if self.assessment else None
            ),
            "evidence_relevance": (
                self.assessment.evidence_relevance if self.assessment else None
            ),
            "coverage": self.assessment.coverage if self.assessment else None,
            "source_quality": (
                self.assessment.source_quality if self.assessment else None
            ),
            "has_conflicts": (
                self.assessment.has_conflicts if self.assessment else False
            ),
            "missing_concepts": (
                list(self.assessment.missing_concepts) if self.assessment else []
            ),
        }


class EvidenceEngine:
    """
    Converts retrieved chunks into structured evidence.

    Retrieved ≠ relevant ≠ supporting.
    """

    def build(
        self,
        chunks: list[RetrievedChunk],
        *,
        question: str,
        document_task: bool = False,
        document_qa_mode: bool = False,
        requires_legal_authority: bool = True,
    ) -> EvidenceBundle:
        assessment = assess_evidence_for_question(
            chunks,
            question,
            document_task=document_task,
            document_qa_mode=document_qa_mode,
            requires_legal_authority=requires_legal_authority,
        )

        items: list[EvidenceItem] = []
        for chunk in chunks:
            authority = classify_authority(
                source_type=chunk.source_type,
                document_type=chunk.document_type,
                law_name=chunk.law_name,
                court=chunk.court,
                filename=chunk.filename,
                display_name=chunk.display_name,
            )
            relevance = float(chunk.relevance_score or chunk.score or 0.0)
            support = self._support_strength(
                chunk=chunk,
                question=question,
                authority=authority,
                relevance=relevance,
                assessment=assessment,
            )
            role = self._classify_role(
                support=support,
                relevance=relevance,
                authority=authority,
                requires_legal_authority=requires_legal_authority,
                document_qa_mode=document_qa_mode or document_task,
            )
            items.append(
                EvidenceItem(
                    chunk=chunk,
                    authority_type=authority,
                    role=role,
                    support_strength=support,
                    relevance_score=relevance,
                )
            )

        items.sort(
            key=lambda item: (
                item.support_strength,
                AUTHORITY_RANK.get(item.authority_type, 0),
                item.relevance_score,
            ),
            reverse=True,
        )

        return EvidenceBundle(
            items=items,
            assessment=assessment,
            overall_strength=assessment.overall_strength,
        )

    def _support_strength(
        self,
        *,
        chunk: RetrievedChunk,
        question: str,
        authority: AuthorityType,
        relevance: float,
        assessment: EvidenceAssessment,
    ) -> float:
        text = (chunk.text or "").lower()
        if not text:
            return 0.0

        covered = 0
        for concept in assessment.key_concepts:
            if concept.lower() in text:
                covered += 1
        concept_score = (
            covered / len(assessment.key_concepts)
            if assessment.key_concepts
            else relevance
        )

        authority_bonus = AUTHORITY_RANK.get(authority, 10) / 100.0
        return min(
            1.0,
            (relevance * 0.45) + (concept_score * 0.40) + (authority_bonus * 0.15),
        )

    @staticmethod
    def _classify_role(
        *,
        support: float,
        relevance: float,
        authority: AuthorityType,
        requires_legal_authority: bool,
        document_qa_mode: bool,
    ) -> EvidenceRole:
        if support >= 0.62 and relevance >= 0.35:
            if requires_legal_authority and not document_qa_mode:
                if AUTHORITY_RANK.get(authority, 0) < 50 and support < 0.75:
                    return EvidenceRole.WEAK
            return EvidenceRole.SUPPORTING
        if support >= 0.40:
            return EvidenceRole.RELEVANT
        if relevance >= 0.25:
            return EvidenceRole.WEAK
        return EvidenceRole.RETRIEVED
