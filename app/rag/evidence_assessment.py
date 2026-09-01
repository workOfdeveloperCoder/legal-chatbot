from __future__ import annotations

import re
from dataclasses import dataclass

from app.rag.answer_guard import EvidenceStrength
from app.rag.models import RetrievedChunk, SourceType


@dataclass(frozen=True, slots=True)
class EvidenceAssessment:
    """Multi-dimensional evidence quality for a legal question."""

    authority_strength: float
    evidence_relevance: float
    coverage: float
    source_quality: float
    overall_strength: EvidenceStrength
    key_concepts: tuple[str, ...]
    covered_concepts: tuple[str, ...]
    missing_concepts: tuple[str, ...]
    has_conflicts: bool = False

    @property
    def is_sufficient(self) -> bool:
        return self.overall_strength not in {
            EvidenceStrength.NONE,
            EvidenceStrength.WEAK,
        }


def extract_question_concepts(question: str) -> list[str]:
    """Extract legal concepts the answer must address."""
    concepts: list[str] = []
    seen: set[str] = set()

    for match in re.finditer(
        r"\b(?:section|article|u/s)\s+([0-9A-Za-z\-]+)",
        question,
        re.I,
    ):
        key = f"section {match.group(1).lower()}"
        if key not in seen:
            seen.add(key)
            concepts.append(key)

    keywords = (
        "double jeopardy",
        "injunction",
        "temporary injunction",
        "judicial discretion",
        "clog",
        "bail",
        "appeal",
        "limitation",
        "specific performance",
        "negligence",
        "murder",
        "contract",
        "arbitration",
        "maintenance",
        "defamation",
        "possession",
        "recovery",
        "precedent",
        "constitutional petition",
        "contempt",
        "inheritance",
    )
    lower = question.lower()
    for keyword in keywords:
        if keyword in lower and keyword not in seen:
            seen.add(keyword)
            concepts.append(keyword)

    return concepts


def assess_evidence_for_question(
    chunks: list[RetrievedChunk],
    question: str,
    *,
    document_task: bool = False,
    document_qa_mode: bool = False,
    requires_legal_authority: bool = True,
    prefer_web: bool = False,
) -> EvidenceAssessment:
    concepts = tuple(extract_question_concepts(question))
    corpus = _build_corpus(chunks)

    covered = tuple(c for c in concepts if _concept_in_corpus(c, corpus))
    missing = tuple(c for c in concepts if c not in covered)

    if concepts:
        coverage = len(covered) / len(concepts)
    else:
        coverage = 1.0 if chunks else 0.0

    evidence_relevance = _score_relevance(chunks, question, concepts)
    authority_strength = _score_authority(chunks, requires_legal_authority)
    source_quality = _score_source_quality(chunks)
    has_conflicts = detect_evidence_conflicts(chunks)

    overall = _derive_overall_strength(
        chunks=chunks,
        document_task=document_task,
        document_qa_mode=document_qa_mode,
        requires_legal_authority=requires_legal_authority,
        authority_strength=authority_strength,
        evidence_relevance=evidence_relevance,
        coverage=coverage,
        source_quality=source_quality,
        concept_count=len(concepts),
        prefer_web=prefer_web,
    )

    return EvidenceAssessment(
        authority_strength=authority_strength,
        evidence_relevance=evidence_relevance,
        coverage=coverage,
        source_quality=source_quality,
        overall_strength=overall,
        key_concepts=concepts,
        covered_concepts=covered,
        missing_concepts=missing,
        has_conflicts=has_conflicts,
    )


def detect_evidence_conflicts(chunks: list[RetrievedChunk]) -> bool:
    """Lightweight check for opposing signals in legal authority chunks."""
    legal_texts = [
        (chunk.text or "").lower()
        for chunk in chunks
        if (chunk.source_type or SourceType.LEGAL.value) == SourceType.LEGAL.value
        and chunk.text
    ]
    if len(legal_texts) < 2:
        return False

    conflict_pairs = (
        ("shall not", "may"),
        ("prohibited", "permitted"),
        ("mandatory", "discretionary"),
        ("cannot", "can"),
        ("must", "may"),
        ("invalid", "valid"),
        ("allowed", "not allowed"),
    )

    combined = " ".join(legal_texts)
    for negative, positive in conflict_pairs:
        if negative in combined and positive in combined:
            return True

    sections: dict[str, list[str]] = {}
    for chunk in chunks:
        if not chunk.text:
            continue
        for match in re.finditer(
            r"\b(?:section|sec\.?|u/s|article)\s+([0-9A-Za-z\-]+)",
            chunk.text,
            re.I,
        ):
            section = match.group(1).lower()
            sections.setdefault(section, []).append(chunk.text.lower())

    for texts in sections.values():
        if len(texts) < 2:
            continue
        joined = " ".join(texts)
        for negative, positive in conflict_pairs:
            if negative in joined and positive in joined:
                return True

    return False


def _derive_overall_strength(
    *,
    chunks: list[RetrievedChunk],
    document_task: bool,
    document_qa_mode: bool,
    requires_legal_authority: bool,
    authority_strength: float,
    evidence_relevance: float,
    coverage: float,
    source_quality: float,
    concept_count: int,
    prefer_web: bool = False,
) -> EvidenceStrength:
    if not chunks:
        return EvidenceStrength.NONE

    composite = (
        authority_strength * 0.35
        + evidence_relevance * 0.30
        + coverage * 0.25
        + source_quality * 0.10
    )
    web = [
        c for c in chunks
        if getattr(c, "source_type", None) == SourceType.WEB.value
    ]
    web_top = (
        max(float(c.relevance_score or c.score or 0.0) for c in web)
        if web
        else 0.0
    )

    if document_task or document_qa_mode:
        private_score = _score_private_sources(chunks)
        if private_score >= 0.45 and evidence_relevance >= 0.35:
            return EvidenceStrength.STRONG
        if private_score >= 0.25:
            return EvidenceStrength.PARTIAL
        if prefer_web and web_top >= 0.40:
            return EvidenceStrength.PARTIAL
        return EvidenceStrength.NONE

    if requires_legal_authority:
        legal_chunks = [
            c for c in chunks
            if (c.source_type or SourceType.LEGAL.value) == SourceType.LEGAL.value
        ]
        if not legal_chunks:
            if web:
                if web_top >= 0.45 or (prefer_web and web_top >= 0.35):
                    return EvidenceStrength.PARTIAL
                return EvidenceStrength.WEAK
            private = _score_private_sources(chunks)
            if private >= 0.30:
                return EvidenceStrength.WEAK
            return EvidenceStrength.NONE

        if concept_count >= 2 and coverage < 0.50:
            if composite >= 0.55:
                return EvidenceStrength.PARTIAL
            if composite >= 0.35:
                return EvidenceStrength.WEAK
            return EvidenceStrength.NONE

        if composite >= 0.70:
            return EvidenceStrength.STRONG
        if composite >= 0.45:
            return EvidenceStrength.PARTIAL
        if composite >= 0.25:
            return EvidenceStrength.WEAK
        return EvidenceStrength.NONE

    if composite >= 0.60:
        return EvidenceStrength.STRONG
    if composite >= 0.35:
        return EvidenceStrength.PARTIAL
    return EvidenceStrength.WEAK


def _score_authority(
    chunks: list[RetrievedChunk],
    requires_legal_authority: bool,
) -> float:
    if not chunks:
        return 0.0

    legal = [
        c for c in chunks
        if (c.source_type or SourceType.LEGAL.value) == SourceType.LEGAL.value
    ]
    private = [c for c in chunks if c not in legal]

    legal_top = max(
        (float(c.relevance_score or c.score or 0.0) for c in legal),
        default=0.0,
    )
    private_top = max(
        (float(c.relevance_score or c.score or 0.0) for c in private),
        default=0.0,
    )

    if requires_legal_authority:
        if legal:
            return min(1.0, 0.55 + legal_top * 0.45)
        if private:
            return min(0.35, private_top * 0.35)
        return 0.0

    return min(1.0, max(legal_top, private_top))


def _score_relevance(
    chunks: list[RetrievedChunk],
    question: str,
    concepts: tuple[str, ...] | list[str],
) -> float:
    if not chunks:
        return 0.0

    question_terms = _question_terms(question)
    if not question_terms and not concepts:
        top = max(float(c.relevance_score or c.score or 0.0) for c in chunks)
        return min(1.0, top)

    scores: list[float] = []
    for chunk in chunks:
        text = (chunk.text or "").lower()
        if not text:
            continue

        term_hits = sum(1 for term in question_terms if term in text)
        concept_hits = sum(1 for concept in concepts if _concept_in_corpus(concept, text))
        denominator = max(len(question_terms), 1) + max(len(concepts), 1)
        overlap = (term_hits + concept_hits * 2) / denominator
        retrieval = float(chunk.relevance_score or chunk.score or 0.0)
        scores.append(min(1.0, overlap * 0.65 + retrieval * 0.35))

    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def _score_source_quality(chunks: list[RetrievedChunk]) -> float:
    if not chunks:
        return 0.0

    scores = [
        float(c.relevance_score or c.score or 0.0)
        for c in chunks
    ]
    avg = sum(scores) / len(scores)
    diversity = len({c.source_type for c in chunks}) / 4.0
    return min(1.0, avg * 0.75 + diversity * 0.25)


def _score_private_sources(chunks: list[RetrievedChunk]) -> float:
    private = [
        c for c in chunks
        if (c.source_type or SourceType.LEGAL.value) != SourceType.LEGAL.value
    ]
    if not private:
        return 0.0
    return max(float(c.relevance_score or c.score or 0.0) for c in private)


def _build_corpus(chunks: list[RetrievedChunk]) -> str:
    parts: list[str] = []
    for chunk in chunks:
        parts.append(chunk.text or "")
        if chunk.sections:
            parts.extend(str(s) for s in chunk.sections)
        if chunk.law_name:
            parts.append(chunk.law_name)
        if chunk.section:
            parts.append(str(chunk.section))
    return " ".join(parts).lower()


def _question_terms(question: str) -> list[str]:
    stopwords = {
        "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
        "can", "could", "should", "would", "will", "is", "are", "was", "were",
        "the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or",
        "this", "that", "these", "those", "it", "be", "been", "being", "do",
        "does", "did", "under", "about", "with", "from", "by", "as", "if",
        "pakistani", "pakistan", "law", "legal",
    }
    words = re.findall(r"[a-z0-9\-]+", question.lower())
    return [w for w in words if len(w) > 2 and w not in stopwords]


def _concept_in_corpus(concept: str, corpus: str) -> bool:
    normalized = concept.lower()
    if normalized in corpus:
        return True

    if normalized.startswith("section "):
        section_num = normalized.split(" ", 1)[1]
        if re.search(
            rf"\b(?:section|sec\.?|u/s|article)\s+{re.escape(section_num)}\b",
            corpus,
            re.I,
        ):
            return True
        if section_num.replace("-", "") in corpus.replace("-", ""):
            return True

    return False
