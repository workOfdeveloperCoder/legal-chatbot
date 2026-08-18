"""
Pakistani legal regression evaluation cases.

Evaluates retrieval/citation properties rather than exact answer text.
Expand this dataset over time.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LegalRegressionCase:
    question: str
    expected_source_type: str
    expected_topics: tuple[str, ...]
    expected_legal_sources: bool
    requires_legal_authority: bool = True
    follow_up: bool = False
    explicit_override: bool = False
    expected_answer_mode: str | None = None
    expected_one_resource: bool = False
    forbid_binding_law_from_article: bool = False
    has_uploaded_documents: bool = False


LEGAL_REGRESSION_CASES: tuple[LegalRegressionCase, ...] = (
    LegalRegressionCase(
        question="What is the punishment for murder under Section 302 PPC?",
        expected_source_type="legal",
        expected_topics=("murder", "section 302"),
        expected_legal_sources=True,
    ),
    LegalRegressionCase(
        question="What is the limitation period for filing a civil suit?",
        expected_source_type="legal",
        expected_topics=("limitation",),
        expected_legal_sources=True,
    ),
    LegalRegressionCase(
        question="When can bail be granted under Section 497 CrPC?",
        expected_source_type="legal",
        expected_topics=("bail", "section 497"),
        expected_legal_sources=True,
    ),
    LegalRegressionCase(
        question="What are the grounds for a constitutional petition?",
        expected_source_type="legal",
        expected_topics=("constitutional petition", "constitution"),
        expected_legal_sources=True,
    ),
    LegalRegressionCase(
        question="What is specific performance under the Contract Act?",
        expected_source_type="legal",
        expected_topics=("specific performance", "contract"),
        expected_legal_sources=True,
    ),
    LegalRegressionCase(
        question="What is the procedure for filing an appeal in civil matters?",
        expected_source_type="legal",
        expected_topics=("appeal", "civil"),
        expected_legal_sources=True,
    ),
    LegalRegressionCase(
        question="What are the requirements for a valid contract?",
        expected_source_type="legal",
        expected_topics=("contract",),
        expected_legal_sources=True,
    ),
    LegalRegressionCase(
        question="What is the standard of proof in criminal cases?",
        expected_source_type="legal",
        expected_topics=("evidence", "criminal"),
        expected_legal_sources=True,
    ),
    LegalRegressionCase(
        question="What is the law on inheritance under Muslim family law?",
        expected_source_type="legal",
        expected_topics=("inheritance", "family"),
        expected_legal_sources=True,
    ),
    LegalRegressionCase(
        question="What is adverse possession in property law?",
        expected_source_type="legal",
        expected_topics=("property", "possession"),
        expected_legal_sources=True,
    ),
    LegalRegressionCase(
        question="What is the procedure for temporary injunction?",
        expected_source_type="legal",
        expected_topics=("injunction",),
        expected_legal_sources=True,
    ),
    LegalRegressionCase(
        question="What is negligence under tort law?",
        expected_source_type="legal",
        expected_topics=("negligence",),
        expected_legal_sources=True,
    ),
    LegalRegressionCase(
        question="What is the limitation period for recovery of money?",
        expected_source_type="legal",
        expected_topics=("limitation", "recovery"),
        expected_legal_sources=True,
    ),
    LegalRegressionCase(
        question="What is the appellate jurisdiction of the Supreme Court?",
        expected_source_type="legal",
        expected_topics=("appeal", "supreme court"),
        expected_legal_sources=True,
    ),
    LegalRegressionCase(
        question="What is the law on arbitration under Pakistani law?",
        expected_source_type="legal",
        expected_topics=("arbitration",),
        expected_legal_sources=True,
    ),
    LegalRegressionCase(
        question="What is the procedure for filing a plaint?",
        expected_source_type="legal",
        expected_topics=("plaint", "civil procedure"),
        expected_legal_sources=True,
    ),
    LegalRegressionCase(
        question="What is the law on defamation?",
        expected_source_type="legal",
        expected_topics=("defamation",),
        expected_legal_sources=True,
    ),
    LegalRegressionCase(
        question="What is the law on maintenance in family cases?",
        expected_source_type="legal",
        expected_topics=("maintenance", "family"),
        expected_legal_sources=True,
    ),
    LegalRegressionCase(
        question="What is the law on transfer of property?",
        expected_source_type="legal",
        expected_topics=("property", "transfer"),
        expected_legal_sources=True,
    ),
    LegalRegressionCase(
        question="What is the law on contempt of court?",
        expected_source_type="legal",
        expected_topics=("contempt", "court"),
        expected_legal_sources=True,
    ),
    LegalRegressionCase(
        question="What about appeals?",
        expected_source_type="legal",
        expected_topics=("appeal",),
        expected_legal_sources=True,
        follow_up=True,
    ),
    LegalRegressionCase(
        question=(
            "What about limitation under the Punjab-specific land revenue law?"
        ),
        expected_source_type="legal",
        expected_topics=("limitation", "punjab"),
        expected_legal_sources=True,
        explicit_override=True,
    ),
    LegalRegressionCase(
        question="Tell me about limitation under the Limitation Act.",
        expected_source_type="legal",
        expected_topics=("limitation act",),
        expected_legal_sources=True,
        explicit_override=True,
    ),
    LegalRegressionCase(
        question="Summarize the termination clause in my contract.",
        expected_source_type="conversation",
        expected_topics=("contract", "termination"),
        expected_legal_sources=False,
        requires_legal_authority=False,
    ),
    LegalRegressionCase(
        question="What facts are pleaded in the plaint?",
        expected_source_type="matter",
        expected_topics=("plaint", "facts"),
        expected_legal_sources=False,
        requires_legal_authority=False,
    ),
    LegalRegressionCase(
        question="What is section 54-C?",
        expected_source_type="legal",
        expected_topics=("section 54-c",),
        expected_legal_sources=True,
    ),
    LegalRegressionCase(
        question=(
            "Can section 54-C prevent an injunction in a double jeopardy situation?"
        ),
        expected_source_type="legal",
        expected_topics=("section 54-c", "injunction", "double jeopardy"),
        expected_legal_sources=True,
    ),
    LegalRegressionCase(
        question=(
            "What does CLOG ON DISCRETION say about section 54-C?"
        ),
        expected_source_type="matter",
        expected_topics=("section 54-c", "discretion", "clog"),
        expected_legal_sources=False,
        requires_legal_authority=False,
        expected_answer_mode="document_qa",
        expected_one_resource=True,
        forbid_binding_law_from_article=True,
        has_uploaded_documents=True,
    ),
    LegalRegressionCase(
        question="What is the problem identified with section 54-C?",
        expected_source_type="matter",
        expected_topics=("section 54-c", "problem", "discretion"),
        expected_legal_sources=False,
        requires_legal_authority=False,
        expected_answer_mode="document_qa",
        expected_one_resource=True,
        forbid_binding_law_from_article=True,
        has_uploaded_documents=True,
    ),
    LegalRegressionCase(
        question="Is the argument in this article legally correct?",
        expected_source_type="legal",
        expected_topics=("article", "legally correct"),
        expected_legal_sources=True,
        expected_answer_mode="mixed_legal_analysis",
        has_uploaded_documents=True,
    ),
    LegalRegressionCase(
        question="Can it be challenged?",
        expected_source_type="legal",
        expected_topics=("challenge",),
        expected_legal_sources=True,
        follow_up=True,
    ),
    LegalRegressionCase(
        question=(
            "Can section 54-C be treated as a clog on judicial discretion "
            "where the consumer alleges double jeopardy?"
        ),
        expected_source_type="legal",
        expected_topics=("section 54-c", "double jeopardy", "discretion"),
        expected_legal_sources=True,
    ),
    LegalRegressionCase(
        question="What is the effect of section 54-C on injunctions?",
        expected_source_type="legal",
        expected_topics=("section 54-c", "injunction"),
        expected_legal_sources=True,
    ),
)
