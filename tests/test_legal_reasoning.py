"""Harvey-level legal reasoning upgrade tests."""

from __future__ import annotations

import pytest

from app.rag.answer_guard import (
    AnswerGuard,
    ClaimGroundingValidator,
    EvidenceStrength,
    GroundingStatus,
    assess_evidence_strength,
)
from app.rag.context_resolver import ContextResolver
from app.rag.evidence_assessment import (
    assess_evidence_for_question,
    detect_evidence_conflicts,
    extract_question_concepts,
)
from app.rag.models import Message, RetrievedChunk, SourceType
from app.rag.prompt_builder import PromptBuilder
from app.rag.query_complexity import QueryComplexity, classify_query_complexity
from app.rag.reasoning_cleanup import (
    ReasoningStreamFilter,
    join_stream_text,
    strip_reasoning_output,
)


def _chunk(**kwargs) -> RetrievedChunk:
    defaults = {
        "id": "chunk-1",
        "chunk_id": "chunk-1",
        "score": 0.7,
        "relevance_score": 0.7,
        "text": "Section 54-C restricts judicial discretion.",
        "source_type": SourceType.MATTER.value,
    }
    defaults.update(kwargs)
    return RetrievedChunk(**defaults)


class TestQueryComplexity:
    def test_simple_section_lookup(self):
        complexity = classify_query_complexity("What is section 54-C?")
        assert complexity == QueryComplexity.SIMPLE

    def test_research_applied_question(self):
        complexity = classify_query_complexity(
            "What is the effect of section 54-C on injunctions?"
        )
        assert complexity == QueryComplexity.RESEARCH

    def test_complex_multi_concept(self):
        complexity = classify_query_complexity(
            "Can section 54-C be treated as a clog on judicial discretion "
            "where the consumer alleges double jeopardy?"
        )
        assert complexity == QueryComplexity.COMPLEX

    def test_mixed_qa_is_complex(self):
        complexity = classify_query_complexity(
            "Is the argument in this article legally correct?",
            mixed_qa=True,
        )
        assert complexity == QueryComplexity.COMPLEX


class TestEvidenceAssessment:
    def test_extracts_multiple_concepts(self):
        concepts = extract_question_concepts(
            "Can section 54-C prevent an injunction in a double jeopardy situation?"
        )
        assert "section 54-c" in concepts
        assert "injunction" in concepts
        assert "double jeopardy" in concepts

    def test_section_only_chunk_insufficient_for_multi_concept(self):
        chunks = [
            _chunk(
                text="Section 54-C is discussed in this article.",
                relevance_score=0.85,
            )
        ]
        question = (
            "Can section 54-C prevent an injunction in a double jeopardy situation?"
        )
        assessment = assess_evidence_for_question(
            chunks,
            question,
            requires_legal_authority=True,
        )
        assert assessment.coverage < 1.0
        assert "injunction" in assessment.missing_concepts
        assert assessment.overall_strength in {
            EvidenceStrength.WEAK,
            EvidenceStrength.NONE,
            EvidenceStrength.PARTIAL,
        }

    def test_full_coverage_with_legal_authority(self):
        chunks = [
            _chunk(
                source_type=SourceType.LEGAL.value,
                text=(
                    "Section 54-C limits discretion. Temporary injunction may be "
                    "refused where double jeopardy is alleged."
                ),
                relevance_score=0.82,
            )
        ]
        question = (
            "Can section 54-C prevent an injunction in a double jeopardy situation?"
        )
        assessment = assess_evidence_for_question(
            chunks,
            question,
            requires_legal_authority=True,
        )
        assert assessment.coverage >= 0.66
        assert assessment.overall_strength in {
            EvidenceStrength.STRONG,
            EvidenceStrength.PARTIAL,
        }

    def test_document_qa_does_not_require_legal_corpus(self):
        chunks = [
            _chunk(
                text="The author argues section 54-C operates as a clog on discretion.",
                relevance_score=0.80,
            )
        ]
        assessment = assess_evidence_for_question(
            chunks,
            "What does the article say about section 54-C?",
            document_qa_mode=True,
            requires_legal_authority=False,
        )
        assert assessment.overall_strength in {
            EvidenceStrength.STRONG,
            EvidenceStrength.PARTIAL,
        }

    def test_detect_conflicts(self):
        chunks = [
            _chunk(
                source_type=SourceType.LEGAL.value,
                text="The court held the injunction shall not be granted.",
            ),
            _chunk(
                source_type=SourceType.LEGAL.value,
                text="The court may grant injunction in exceptional cases.",
                chunk_id="chunk-2",
                id="chunk-2",
            ),
        ]
        assert detect_evidence_conflicts(chunks) is True


class TestClaimGrounding:
    def test_unsupported_section_not_in_evidence(self):
        validator = ClaimGroundingValidator()
        chunks = [_chunk(text="Section 54-C is discussed.")]
        result = validator.validate(
            "Section 999 provides absolute immunity from injunction.",
            chunks=chunks,
            sources=[],
            question="Can section 54-C prevent injunction?",
        )
        assert result.unsupported_claims
        assert result.status in {
            GroundingStatus.PARTIAL,
            GroundingStatus.INSUFFICIENT,
        }

    def test_document_mode_supports_section_in_doc(self):
        validator = ClaimGroundingValidator()
        chunks = [_chunk(text="Section 54-C creates a clog on discretion.")]
        result = validator.validate(
            "The article argues section 54-C creates a clog on discretion.",
            chunks=chunks,
            sources=[],
            document_evidence_mode=True,
        )
        assert result.status == GroundingStatus.STRONG


class TestReasoningCleanup:
    def test_strips_think_tags(self):
        think_open = chr(60) + "think" + chr(62)
        think_close = chr(60) + "/" + "think" + chr(62)
        raw = (
            f"{think_open}Internal reasoning about section 54-C.{think_close}"
            "## Short Answer\nSection 54-C restricts discretion."
        )
        cleaned = strip_reasoning_output(raw)
        assert "Internal reasoning" not in cleaned
        assert "Short Answer" in cleaned

    def test_strips_unclosed_think(self):
        think_open = chr(60) + "think" + chr(62)
        raw = f"{think_open}Still reasoning..."
        cleaned = strip_reasoning_output(raw)
        assert cleaned == ""

    def test_stream_filter_keeps_answer_when_think_tags_split(self):
        think_open = chr(60) + "think" + chr(62)
        think_close = chr(60) + "/" + "think" + chr(62)
        filt = ReasoningStreamFilter()
        visible = [
            filt.feed(think_open[:3]),
            filt.feed(think_open[3:] + "internal notes"),
            filt.feed(think_close + "Section 54-C bars an injunction."),
            filt.finish(),
        ]
        joined = "".join(visible)
        assert "internal notes" not in joined
        assert "Section 54-C bars an injunction." in joined

    def test_join_stream_text_does_not_split_subword_crumbs(self):
        # Blind space insertion used to turn BPE crumbs into "rel ati ve s".
        assert join_stream_text(["rel", "ati", "ve", "s"]) == "relatives"

    def test_join_stream_text_keeps_model_provided_spaces(self):
        assert join_stream_text(["The ", "court ", "held"]) == "The court held"

    def test_join_stream_text_spaces_before_capitalized_word(self):
        assert join_stream_text(["held", "The"]) == "held The"

    def test_join_stream_text_does_not_duplicate_existing_spaces(self):
        assert join_stream_text(["The ", "court ", "held."]) == "The court held."

    def test_join_stream_text_keeps_newlines_and_indent(self):
        text = join_stream_text(["Short answer", "\n    ", "Section 54-C applies."])
        assert "\n    Section 54-C applies." in text

    def test_strip_reasoning_does_not_glue_words_across_think_tags(self):
        think_open = chr(60) + "think" + chr(62)
        think_close = chr(60) + "/" + "think" + chr(62)
        cleaned = strip_reasoning_output(
            f"The{think_open}hidden{think_close}court held."
        )
        assert "The court held." == cleaned

    def test_strip_reasoning_preserves_indented_lines(self):
        raw = "Short Answer\n    Section 54-C restricts discretion."
        assert strip_reasoning_output(raw) == raw


class TestFollowUpContext:
    def test_can_it_be_challenged_follow_up(self):
        resolver = ContextResolver()
        history = [
            Message(role="user", content="Explain section 54-C clog on discretion."),
            Message(role="assistant", content="Section 54-C restricts discretion."),
        ]
        resolved = resolver.resolve("Can it be challenged?", history)
        assert "challenged" in resolved.lower()
        assert "54-c" in resolved.lower() or "discretion" in resolved.lower()


class TestPromptBuilder:
    def test_includes_assessment_and_complexity(self):
        from app.rag.evidence_assessment import EvidenceAssessment

        builder = PromptBuilder()
        assessment = EvidenceAssessment(
            authority_strength=0.4,
            evidence_relevance=0.5,
            coverage=0.33,
            source_quality=0.6,
            overall_strength=EvidenceStrength.PARTIAL,
            key_concepts=("section 54-c", "injunction"),
            covered_concepts=("section 54-c",),
            missing_concepts=("injunction",),
            has_conflicts=False,
        )
        prompt = builder.build(
            question="Can section 54-C prevent injunction?",
            history=[],
            chunks=[_chunk()],
            memories=[],
            complexity=QueryComplexity.RESEARCH,
            evidence_assessment=assessment,
        )
        assert "QUESTION COMPLEXITY: RESEARCH" in prompt
        assert "concepts missing from evidence" in prompt
        assert "RETRIEVED EVIDENCE" in prompt


class TestAssessEvidenceStrengthBackwardCompat:
    def test_without_question_uses_legacy_path(self):
        chunks = [
            _chunk(
                source_type=SourceType.LEGAL.value,
                relevance_score=0.82,
            )
        ]
        strength = assess_evidence_strength(
            chunks,
            requires_legal_authority=True,
        )
        assert strength == EvidenceStrength.STRONG

    def test_with_question_uses_relevance(self):
        chunks = [_chunk(text="Section 54-C only.")]
        strength = assess_evidence_strength(
            chunks,
            requires_legal_authority=True,
            question=(
                "Can section 54-C prevent an injunction in double jeopardy?"
            ),
        )
        assert strength in {
            EvidenceStrength.WEAK,
            EvidenceStrength.NONE,
            EvidenceStrength.PARTIAL,
        }


class TestAnswerGuardConflicts:
    def test_conflict_flagged_without_hardcoded_preamble(self):
        guard = AnswerGuard()
        result = guard.process(
            answer="Section 54-C restricts discretion.",
            chunks=[_chunk(source_type=SourceType.LEGAL.value)],
            sources=[],
            evidence_strength=EvidenceStrength.PARTIAL,
            has_conflicts=True,
        )
        assert result.answer == "Section 54-C restricts discretion."
        assert result.grounding.has_conflicts is True
