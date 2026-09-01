from __future__ import annotations

import re
from enum import Enum

from pydantic import BaseModel, Field


class EvidenceStrength(str, Enum):
    STRONG = "strong"
    PARTIAL = "partial"
    WEAK = "weak"
    NONE = "none"


class GroundingStatus(str, Enum):
    STRONG = "strong"
    PARTIAL = "partial"
    INSUFFICIENT = "insufficient"


class CitationValidationStatus(str, Enum):
    VALID = "valid"
    REPAIRED = "repaired"
    PARTIAL = "partial"
    NONE = "none"


class CitationValidationResult(BaseModel):
    valid: list[int] = Field(default_factory=list)
    invalid: list[int] = Field(default_factory=list)
    unused: list[int] = Field(default_factory=list)
    status: CitationValidationStatus = CitationValidationStatus.NONE


class ClaimSupportStatus(str, Enum):
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"


class ClaimAssessment(BaseModel):
    claim: str
    status: ClaimSupportStatus


class GroundingResult(BaseModel):
    supported_claims: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    partial_claims: list[str] = Field(default_factory=list)
    contradicted_claims: list[str] = Field(default_factory=list)
    claim_assessments: list[ClaimAssessment] = Field(default_factory=list)
    status: GroundingStatus = GroundingStatus.STRONG
    has_conflicts: bool = False


class GuardedAnswerResult(BaseModel):
    answer: str
    citation_validation: CitationValidationResult
    grounding: GroundingResult
    evidence_strength: EvidenceStrength
    grounding_status: GroundingStatus
    sources_used: list[int] = Field(default_factory=list)


INSUFFICIENT_EVIDENCE_ANSWER = (
    "No matching document or corpus passage was found for this question.\n\n"
    "I could not locate sufficient retrieved legal material to give a "
    "document-grounded answer. Please rephrase with a statute name, "
    "section number, or upload a document — or ask again so I can provide "
    "general legal guidance clearly marked as not authority-backed."
)

NO_DOCUMENT_FOUND_PREFIX = (
    "No matching document or corpus passage was found for this question. "
    "The following is general legal guidance only — not grounded in "
    "retrieved authorities or uploaded files — and should be verified by "
    "counsel against primary sources before reliance.\n\n"
)

PARTIAL_EVIDENCE_PREFIX = (
    "Note: The available retrieved sources provide only partial support "
    "for this answer. The following should be verified against primary "
    "legal authority.\n\n"
)

WEAK_EVIDENCE_PREFIX = (
    "Note: Retrieved sources are limited and may not include authoritative "
    "legal material on this point. The following is based on available "
    "documents only and should not be treated as definitive legal authority.\n\n"
)

_CITATION_PATTERN = re.compile(
    r"\[Source\s+(\d+)\]",
    re.IGNORECASE,
)

# Combined forms: [Source 1, Source 2] / [Sources 1, 2] / [Source 1 and 2]
_CITATION_COMBINED_PATTERN = re.compile(
    r"\[Sources?\s+((?:\d+(?:\s*(?:,|and|&)\s*(?:Source\s+)?)*\d+)|(?:\d+))\]",
    re.IGNORECASE,
)

_CITATION_NUMBER_PATTERN = re.compile(r"\d+")

_SECTION_PATTERN = re.compile(
    r"\b(?:(?:section|sec\.?|u/s)\s+([0-9A-Za-z\-]+)|"
    r"article\s+([0-9]+[A-Za-z\-]*|[IVXLCDM]+))\b",
    re.IGNORECASE,
)


def _section_number(match: re.Match[str] | None) -> str:
    if match is None:
        return ""
    for group in match.groups():
        if group:
            return group.lower()
    return ""

_CASE_CITATION_PATTERN = re.compile(
    r"\b(?:PLD|SCMR|MLD|YLR|CLC|PCr\.LJ)\s+\d{4}\s+[A-Za-z]+\s+\d+",
    re.IGNORECASE,
)

_QUOTED_TEXT_PATTERN = re.compile(
    r'"([^"]{20,})"|"([^"]{20,})"|' r"'([^']{20,})'",
)


class CitationValidator:
    """Validate and repair [Source N] references in model answers."""

    MAX_REPAIR_PASSES = 1

    def validate(
        self,
        answer: str,
        *,
        source_count: int,
    ) -> CitationValidationResult:
        if source_count <= 0:
            cited = self._extract_citations(answer)
            return CitationValidationResult(
                valid=[],
                invalid=sorted(set(cited)),
                unused=[],
                status=(
                    CitationValidationStatus.NONE
                    if not cited
                    else CitationValidationStatus.PARTIAL
                ),
            )

        cited = self._extract_citations(answer)
        valid = sorted({n for n in cited if 1 <= n <= source_count})
        invalid = sorted({n for n in cited if n < 1 or n > source_count})
        all_sources = set(range(1, source_count + 1))
        used = set(valid)
        unused = sorted(all_sources - used)

        if not cited:
            status = CitationValidationStatus.NONE
        elif invalid and valid:
            status = CitationValidationStatus.PARTIAL
        elif invalid:
            status = CitationValidationStatus.PARTIAL
        else:
            status = CitationValidationStatus.VALID

        return CitationValidationResult(
            valid=valid,
            invalid=invalid,
            unused=unused,
            status=status,
        )

    @staticmethod
    def _extract_citations(answer: str) -> list[int]:
        numbers: list[int] = []

        for match in _CITATION_COMBINED_PATTERN.finditer(answer):
            numbers.extend(
                int(n) for n in _CITATION_NUMBER_PATTERN.findall(match.group(1))
            )

        for match in _CITATION_PATTERN.finditer(answer):
            numbers.append(int(match.group(1)))

        return numbers

    def repair(
        self,
        answer: str,
        validation: CitationValidationResult,
        *,
        source_count: int,
    ) -> tuple[str, CitationValidationResult]:
        if not validation.invalid:
            return answer, validation

        repaired = answer

        # Remove invalid numbers from combined markers first
        def _rewrite_combined(match: re.Match[str]) -> str:
            nums = [
                int(n) for n in _CITATION_NUMBER_PATTERN.findall(match.group(1))
            ]
            valid_nums = [n for n in nums if 1 <= n <= source_count]
            if not valid_nums:
                return ""
            if len(valid_nums) == 1:
                return f"[Source {valid_nums[0]}]"
            joined = ", ".join(f"Source {n}" for n in valid_nums)
            return f"[{joined}]"

        repaired = _CITATION_COMBINED_PATTERN.sub(_rewrite_combined, repaired)

        for source_num in validation.invalid:
            repaired = _CITATION_PATTERN.sub(
                lambda match, bad=source_num: (
                    "" if int(match.group(1)) == bad else match.group(0)
                ),
                repaired,
            )

        repaired = re.sub(r"\s{2,}", " ", repaired)
        repaired = re.sub(r" \.", ".", repaired)
        repaired = repaired.strip()

        revalidated = self.validate(
            repaired,
            source_count=source_count,
        )
        if validation.invalid:
            revalidated.status = CitationValidationStatus.REPAIRED
        return repaired, revalidated


class ClaimGroundingValidator:
    """Post-generation claim-level grounding against retrieved evidence."""

    _LEGAL_ASSERTION_PATTERN = re.compile(
        r"(?:the (?:statute|law|court|judgment|act|ordinance|section|article)|"
        r"section\s+[0-9A-Za-z\-]+|"
        r"(?:pld|scmr|mld|ylr|clc|pcr\.lj)\s+\d{4})",
        re.IGNORECASE,
    )

    def validate(
        self,
        answer: str,
        *,
        chunks: list,
        sources: list,
        document_evidence_mode: bool = False,
        question: str | None = None,
        has_conflicts: bool = False,
    ) -> GroundingResult:
        claims = self._extract_claims(answer)
        sentence_claims = self._extract_material_sentences(answer)

        all_claims = self._merge_claims(claims, sentence_claims)
        if not all_claims:
            return GroundingResult(
                status=GroundingStatus.STRONG,
                has_conflicts=has_conflicts,
            )

        if document_evidence_mode:
            return self._validate_document_claims(
                all_claims,
                chunks=chunks,
                sources=sources,
                has_conflicts=has_conflicts,
            )

        evidence_corpus = self._build_evidence_corpus(chunks, sources)
        question_terms = self._question_terms(question or "")

        supported: list[str] = []
        partial: list[str] = []
        unsupported: list[str] = []
        contradicted: list[str] = []
        assessments: list[ClaimAssessment] = []

        for claim in all_claims:
            status = self._assess_claim(
                claim,
                evidence_corpus=evidence_corpus,
                question_terms=question_terms,
            )
            assessments.append(ClaimAssessment(claim=claim, status=status))
            if status == ClaimSupportStatus.SUPPORTED:
                supported.append(claim)
            elif status == ClaimSupportStatus.PARTIALLY_SUPPORTED:
                partial.append(claim)
            elif status == ClaimSupportStatus.CONTRADICTED:
                contradicted.append(claim)
                unsupported.append(claim)
            else:
                unsupported.append(claim)

        if (unsupported or contradicted) and not supported and not partial:
            status = GroundingStatus.INSUFFICIENT
        elif unsupported or partial or contradicted:
            status = GroundingStatus.PARTIAL
        else:
            status = GroundingStatus.STRONG

        return GroundingResult(
            supported_claims=supported,
            unsupported_claims=unsupported,
            partial_claims=partial,
            contradicted_claims=contradicted,
            claim_assessments=assessments,
            status=status,
            has_conflicts=has_conflicts or bool(contradicted),
        )

    def _assess_claim(
        self,
        claim: str,
        *,
        evidence_corpus: str,
        question_terms: list[str],
    ) -> ClaimSupportStatus:
        lower_claim = claim.lower()

        # Invented case citations are never allowed without evidence match.
        if _CASE_CITATION_PATTERN.search(claim):
            if self._citation_in_evidence(claim, evidence_corpus):
                return ClaimSupportStatus.SUPPORTED
            return ClaimSupportStatus.UNSUPPORTED

        if _QUOTED_TEXT_PATTERN.search(claim):
            if self._claim_supported(claim, evidence_corpus):
                return ClaimSupportStatus.SUPPORTED
            return ClaimSupportStatus.UNSUPPORTED

        # Explicit negation conflict against evidence (lightweight).
        if self._looks_contradicted(lower_claim, evidence_corpus):
            return ClaimSupportStatus.CONTRADICTED

        relation_terms = self._relation_terms(claim)
        missing_relations = [
            term for term in relation_terms if term not in evidence_corpus
        ]

        section_match = _SECTION_PATTERN.search(claim)
        if section_match:
            section_num = _section_number(section_match)
            section_in_evidence = bool(
                section_num
                and (
                    section_num in evidence_corpus
                    or re.search(
                        rf"\b(?:section|sec\.?|u/s|article)\s+{re.escape(section_num)}\b",
                        evidence_corpus,
                    )
                )
            )
            if not section_in_evidence:
                return ClaimSupportStatus.UNSUPPORTED

            # Section alone does not establish a multi-part legal relationship.
            if missing_relations:
                return ClaimSupportStatus.PARTIALLY_SUPPORTED

            if self._claim_supported(claim, evidence_corpus):
                return ClaimSupportStatus.SUPPORTED
            return ClaimSupportStatus.PARTIALLY_SUPPORTED

        # Holding / judgment assertions require evidence support.
        if self._HOLDING_PATTERN.search(claim):
            if self._claim_supported(claim, evidence_corpus):
                return ClaimSupportStatus.SUPPORTED
            return ClaimSupportStatus.UNSUPPORTED

        if missing_relations and self._LEGAL_ASSERTION_PATTERN.search(claim):
            if self._claim_supported(claim, evidence_corpus):
                return ClaimSupportStatus.PARTIALLY_SUPPORTED
            return ClaimSupportStatus.UNSUPPORTED

        if self._claim_supported(claim, evidence_corpus):
            return ClaimSupportStatus.SUPPORTED

        normalized = " ".join(claim.lower().split())
        if len(normalized) >= 20:
            words = normalized.split()
            if len(words) >= 4:
                snippet = " ".join(words[:6])
                if snippet in evidence_corpus:
                    return ClaimSupportStatus.PARTIALLY_SUPPORTED

        if self._LEGAL_ASSERTION_PATTERN.search(claim):
            return ClaimSupportStatus.UNSUPPORTED

        return ClaimSupportStatus.PARTIALLY_SUPPORTED

    _HOLDING_PATTERN = re.compile(
        r"\b(?:the court held|the supreme court held|the high court held|"
        r"it was held that|the judgment (?:held|established|laid down)|"
        r"binding precedent|ratio decidendi)\b",
        re.IGNORECASE,
    )

    @staticmethod
    def _citation_in_evidence(claim: str, corpus: str) -> bool:
        match = _CASE_CITATION_PATTERN.search(claim)
        if not match:
            return False
        cite = " ".join(match.group(0).lower().split())
        return cite in corpus

    @staticmethod
    def _looks_contradicted(claim_lower: str, corpus: str) -> bool:
        """Detect obvious claim/evidence polarity conflicts for key terms."""
        if "not a clog" in claim_lower and "clog" in corpus and "not a clog" not in corpus:
            if "clog on" in corpus or "clog on discretion" in corpus:
                return True
        if (
            "does not restrict" in claim_lower
            and ("restrict" in corpus or "bar" in corpus)
            and "does not restrict" not in corpus
        ):
            return True
        return False

    @staticmethod
    def _relation_terms(claim: str) -> list[str]:
        """Material legal relation terms that must appear in evidence."""
        keywords = (
            "injunction",
            "interim injunction",
            "double jeopardy",
            "discretion",
            "clog",
            "bail",
            "appeal",
            "limitation",
            "prevent",
            "prohibit",
            "mandatory",
            "immunity",
            "deposit",
            "natural justice",
            "unreasonable",
            "holding",
            "conviction",
            "acquittal",
        )
        lower = claim.lower()
        return [kw for kw in keywords if kw in lower]

    @staticmethod
    def _question_terms(question: str) -> list[str]:
        stopwords = {
            "what", "when", "where", "which", "who", "why", "how", "can", "could",
            "should", "would", "will", "is", "are", "the", "a", "an", "in", "on",
            "to", "for", "of", "and", "or", "this", "that", "under", "about",
            "pakistani", "pakistan", "law", "legal", "does", "do",
        }
        words = re.findall(r"[a-z0-9\-]+", question.lower())
        return [w for w in words if len(w) > 2 and w not in stopwords]

    @staticmethod
    def _merge_claims(*groups: list[str]) -> list[str]:
        seen: set[str] = set()
        merged: list[str] = []
        for group in groups:
            for claim in group:
                key = claim.lower().strip()
                if not key or key in seen:
                    continue
                seen.add(key)
                merged.append(claim)
        return merged

    def _extract_material_sentences(self, answer: str) -> list[str]:
        sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
        material: list[str] = []
        for sentence in sentences:
            cleaned = sentence.strip()
            if len(cleaned) < 25:
                continue
            if not self._LEGAL_ASSERTION_PATTERN.search(cleaned):
                continue
            if cleaned.lower().startswith(("note:", "i don't have enough")):
                continue
            material.append(cleaned)
        return material[:8]

    @staticmethod
    def _document_paraphrase_supported(claim: str, corpus: str) -> bool:
        if ClaimGroundingValidator._claim_supported(claim, corpus):
            return True

        claim_words = {
            w for w in re.findall(r"[a-z0-9\-]+", claim.lower())
            if len(w) > 3
            and w not in {"article", "author", "argues", "according", "document", "uploaded"}
        }
        if not claim_words:
            return False

        hits = sum(1 for word in claim_words if word in corpus)
        return hits >= max(2, len(claim_words) // 2)

    def _validate_document_claims(
        self,
        claims: list[str],
        *,
        chunks: list,
        sources: list,
        has_conflicts: bool = False,
    ) -> GroundingResult:
        """
        Document Q&A: section references and paraphrases are supported
        when the retrieved document text contains the section/topic.
        Only flag long direct quotes not present in evidence.
        """
        evidence_corpus = self._build_evidence_corpus(chunks, sources)
        supported: list[str] = []
        unsupported: list[str] = []

        for claim in claims:
            if _QUOTED_TEXT_PATTERN.search(claim):
                if self._claim_supported(claim, evidence_corpus):
                    supported.append(claim)
                else:
                    unsupported.append(claim)
                continue

            if len(claim) >= 40 and " " in claim:
                if self._document_paraphrase_supported(claim, evidence_corpus):
                    supported.append(claim)
                else:
                    unsupported.append(claim)
                continue

            if _SECTION_PATTERN.search(claim):
                section_match = _SECTION_PATTERN.search(claim)
                section_num = _section_number(section_match) if section_match else ""
                if section_num and (
                    section_num in evidence_corpus
                    or self._claim_supported(claim, evidence_corpus)
                ):
                    supported.append(claim)
                else:
                    unsupported.append(claim)
                continue

            supported.append(claim)

        if unsupported:
            status = GroundingStatus.PARTIAL
        else:
            status = GroundingStatus.STRONG

        return GroundingResult(
            supported_claims=supported,
            unsupported_claims=unsupported,
            claim_assessments=[
                ClaimAssessment(
                    claim=claim,
                    status=(
                        ClaimSupportStatus.UNSUPPORTED
                        if claim in unsupported
                        else ClaimSupportStatus.SUPPORTED
                    ),
                )
                for claim in claims
            ],
            status=status,
            has_conflicts=has_conflicts,
        )

    def _extract_claims(self, answer: str) -> list[str]:
        claims: list[str] = []

        for match in _SECTION_PATTERN.finditer(answer):
            claims.append(match.group(0).strip())

        for match in _CASE_CITATION_PATTERN.finditer(answer):
            claims.append(match.group(0).strip())

        for match in _QUOTED_TEXT_PATTERN.finditer(answer):
            quote = next(g for g in match.groups() if g)
            if quote:
                claims.append(quote.strip())

        seen: set[str] = set()
        unique: list[str] = []
        for claim in claims:
            key = claim.lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append(claim)
        return unique

    @staticmethod
    def _build_evidence_corpus(chunks, sources) -> str:
        parts: list[str] = []
        for chunk in chunks:
            parts.append(chunk.text or "")
            if chunk.sections:
                parts.extend(str(s) for s in chunk.sections)
            if chunk.law_name:
                parts.append(chunk.law_name)
            if chunk.section:
                parts.append(chunk.section)

        for source in sources:
            if source.excerpt:
                parts.append(source.excerpt)
            if source.sections:
                parts.extend(str(s) for s in source.sections)
            if source.law_name:
                parts.append(source.law_name)
            if source.source_reference:
                parts.append(source.source_reference)

        return " ".join(parts).lower()

    @staticmethod
    def _claim_supported(claim: str, corpus: str) -> bool:
        normalized = " ".join(claim.lower().split())
        if normalized in corpus:
            return True

        if _SECTION_PATTERN.search(claim):
            section_match = _SECTION_PATTERN.search(claim)
            if section_match:
                section_num = _section_number(section_match)
                if re.search(
                    rf"\b(?:section|sec\.?|u/s|article)\s+{re.escape(section_num)}\b",
                    corpus,
                ):
                    return True
                if section_num in corpus:
                    return True

        if len(normalized) >= 20:
            words = normalized.split()
            if len(words) >= 4:
                snippet = " ".join(words[:8])
                return snippet in corpus

        return False


def assess_evidence_strength(
    chunks: list,
    *,
    document_task: bool = False,
    document_qa_mode: bool = False,
    requires_legal_authority: bool = True,
    question: str | None = None,
) -> EvidenceStrength:
    if question:
        from app.rag.evidence_assessment import assess_evidence_for_question

        assessment = assess_evidence_for_question(
            chunks,
            question,
            document_task=document_task,
            document_qa_mode=document_qa_mode,
            requires_legal_authority=requires_legal_authority,
        )
        return assessment.overall_strength

    if not chunks:
        return EvidenceStrength.NONE

    legal_chunks = [
        c for c in chunks
        if getattr(c, "source_type", None) == "legal"
    ]
    private_chunks = [
        c for c in chunks
        if getattr(c, "source_type", None) != "legal"
    ]

    def _top_score(items: list) -> float:
        if not items:
            return 0.0
        return max(
            float(c.relevance_score or c.score or 0.0)
            for c in items
        )

    legal_top = _top_score(legal_chunks)
    private_top = _top_score(private_chunks)

    if document_task or document_qa_mode:
        if private_chunks and private_top >= 0.25:
            return EvidenceStrength.STRONG
        if private_chunks:
            return EvidenceStrength.PARTIAL
        return EvidenceStrength.NONE

    if requires_legal_authority:
        if legal_chunks and legal_top >= 0.45:
            return EvidenceStrength.STRONG
        if legal_chunks:
            return EvidenceStrength.PARTIAL
        if private_chunks:
            return EvidenceStrength.WEAK
        return EvidenceStrength.NONE

    if legal_chunks and legal_top >= 0.40:
        return EvidenceStrength.STRONG
    if chunks:
        return EvidenceStrength.PARTIAL
    return EvidenceStrength.NONE


def requires_legal_authority(
    question: str,
    *,
    task: str | None = None,
    document_task: bool = False,
    document_qa_mode: bool = False,
) -> bool:
    if document_task or document_qa_mode:
        return False

    task_name = task.value if hasattr(task, "value") else str(task or "")
    legal_tasks = {
        "legal_qa",
        "case_search",
        "statute_search",
        "legal_research",
    }
    if task_name in legal_tasks:
        return True

    legal_markers = [
        "section",
        "statute",
        "act",
        "law",
        "case",
        "court",
        "appeal",
        "limitation",
        "bail",
        "contract",
        "constitution",
        "plaint",
        "judgment",
        "precedent",
        "ordinance",
        "regulation",
    ]
    lower = question.lower()
    return any(marker in lower for marker in legal_markers)


class AnswerGuard:
    """Post-generation validation, repair, and evidence-limited responses."""

    def __init__(
        self,
        *,
        citation_validator: CitationValidator | None = None,
        claim_validator: ClaimGroundingValidator | None = None,
    ) -> None:
        self._citations = citation_validator or CitationValidator()
        self._claims = claim_validator or ClaimGroundingValidator()

    def build_insufficient_evidence_answer(
        self,
        *,
        evidence_strength: EvidenceStrength,
        has_private_sources: bool = False,
        language=None,
    ) -> str:
        from app.rag.language import insufficient_evidence_answer

        if has_private_sources and evidence_strength == EvidenceStrength.WEAK:
            if language is None:
                return (
                    "I could not find sufficient authoritative legal material "
                    "in the available legal sources to give a definitive answer "
                    "on this point.\n\n"
                    "Some matter or conversation documents were retrieved, but "
                    "they do not establish authoritative law. They may contain "
                    "case-specific facts or party positions only. Please verify "
                    "against applicable statutes, rules, or binding judgments."
                )
        return insufficient_evidence_answer(language)

    def process(
        self,
        *,
        answer: str,
        chunks: list,
        sources: list,
        evidence_strength: EvidenceStrength,
        skip_grounding: bool = False,
        document_evidence_mode: bool = False,
        question: str | None = None,
        has_conflicts: bool = False,
    ) -> GuardedAnswerResult:
        source_count = len(sources)

        validation = self._citations.validate(
            answer,
            source_count=source_count,
        )

        final_answer = answer
        if validation.invalid:
            final_answer, validation = self._citations.repair(
                answer,
                validation,
                source_count=source_count,
            )

        if not (final_answer or "").strip():
            # Never return a blank bubble when the model only emitted thinking.
            fallback = self.build_insufficient_evidence_answer(
                evidence_strength=evidence_strength,
                has_private_sources=bool(chunks),
            )
            return GuardedAnswerResult(
                answer=fallback,
                citation_validation=validation,
                grounding=GroundingResult(status=GroundingStatus.INSUFFICIENT),
                evidence_strength=evidence_strength,
                grounding_status=GroundingStatus.INSUFFICIENT,
                sources_used=[],
            )

        grounding = GroundingResult(status=GroundingStatus.STRONG)
        if not skip_grounding:
            grounding = self._claims.validate(
                final_answer,
                chunks=chunks,
                sources=sources,
                document_evidence_mode=document_evidence_mode,
                question=question,
                has_conflicts=has_conflicts,
            )

        if has_conflicts or grounding.has_conflicts:
            final_answer = self._apply_conflict_notice(final_answer)

        grounding_status = grounding.status

        if document_evidence_mode:
            if grounding_status == GroundingStatus.INSUFFICIENT:
                grounding_status = GroundingStatus.PARTIAL
        elif (
            evidence_strength == EvidenceStrength.WEAK
            and grounding_status == GroundingStatus.STRONG
        ):
            grounding_status = GroundingStatus.PARTIAL

        if document_evidence_mode:
            # Only warn when an explicit long quotation could not be verified.
            quote_mismatches = [
                claim
                for claim in grounding.unsupported_claims
                if _QUOTED_TEXT_PATTERN.search(f'"{claim}"')
                or _QUOTED_TEXT_PATTERN.search(claim)
                or (
                    len(claim) >= 40
                    and claim.strip().startswith(('"', "'", "“", "‘"))
                )
            ]
            if quote_mismatches:
                final_answer = self._apply_document_quote_warning(
                    final_answer,
                    quote_mismatches,
                )
        elif grounding_status == GroundingStatus.INSUFFICIENT:
            final_answer = self._apply_insufficient_grounding(final_answer)
        elif evidence_strength == EvidenceStrength.NONE and not chunks:
            if not final_answer.lstrip().lower().startswith("no matching document"):
                final_answer = NO_DOCUMENT_FOUND_PREFIX + final_answer
        elif grounding_status == GroundingStatus.PARTIAL:
            final_answer = self._strip_invented_authorities(
                final_answer,
                grounding.unsupported_claims,
            )
            final_answer = self._apply_partial_grounding(
                final_answer,
                grounding.unsupported_claims,
                grounding.partial_claims,
            )
        elif evidence_strength == EvidenceStrength.WEAK:
            if not final_answer.startswith(WEAK_EVIDENCE_PREFIX):
                final_answer = WEAK_EVIDENCE_PREFIX + final_answer
        elif evidence_strength == EvidenceStrength.PARTIAL:
            if not final_answer.startswith(PARTIAL_EVIDENCE_PREFIX):
                final_answer = PARTIAL_EVIDENCE_PREFIX + final_answer

        sources_used = sorted(set(validation.valid))

        return GuardedAnswerResult(
            answer=final_answer.strip(),
            citation_validation=validation,
            grounding=grounding,
            evidence_strength=evidence_strength,
            grounding_status=grounding_status,
            sources_used=sources_used,
        )

    @staticmethod
    def _strip_invented_authorities(
        answer: str,
        unsupported_claims: list[str],
    ) -> str:
        """Remove invented case citations that failed grounding."""
        cleaned = answer
        for claim in unsupported_claims:
            if _CASE_CITATION_PATTERN.fullmatch(claim.strip()) or (
                _CASE_CITATION_PATTERN.search(claim)
                and len(claim) < 80
            ):
                cleaned = cleaned.replace(claim, "[citation not verified in evidence]")
        return cleaned

    @staticmethod
    def _apply_document_quote_warning(
        answer: str,
        unsupported_claims: list[str] | None = None,
    ) -> str:
        if unsupported_claims:
            return (
                "Note: Some quoted passages could not be matched exactly to "
                "the retrieved document text. The summary below is based on "
                "available excerpts.\n\n"
                + answer
            )
        return answer

    @staticmethod
    def _apply_insufficient_grounding(answer: str) -> str:
        prefix = (
            "No matching document or corpus passage fully supports every "
            "legal proposition below. Treat this as provisional guidance "
            "and verify against primary authorities before reliance.\n\n"
        )
        if answer.lstrip().lower().startswith("no matching document"):
            return answer
        return prefix + answer

    @staticmethod
    def _apply_conflict_notice(answer: str) -> str:
        notice = (
            "Note: Retrieved authorities may reflect conflicting positions. "
            "The analysis below identifies uncertainty where present.\n\n"
        )
        if "conflicting" in answer.lower() or answer.startswith("Note:"):
            return answer
        return notice + answer

    @staticmethod
    def _apply_partial_grounding(
        answer: str,
        unsupported_claims: list[str],
        partial_claims: list[str] | None = None,
    ) -> str:
        if not unsupported_claims and not partial_claims:
            if answer.startswith(PARTIAL_EVIDENCE_PREFIX):
                return answer
            return PARTIAL_EVIDENCE_PREFIX + answer

        flagged = unsupported_claims[:2] + (partial_claims or [])[:2]
        claims_preview = ", ".join(flagged[:3])
        prefix = (
            "Note: Some legal propositions in this answer could not be fully "
            f"verified against retrieved evidence ({claims_preview}). "
            "Please verify against primary legal authority.\n\n"
        )
        if answer.startswith(PARTIAL_EVIDENCE_PREFIX):
            return answer
        return prefix + answer
