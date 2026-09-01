from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from app.rag.query_complexity import QueryComplexity, classify_query_complexity
from app.rag.models import Message
from app.services.query_router import QueryRouter, Task


class AnswerMode(str, Enum):
    DOCUMENT_QA = "document_qa"
    LEGAL_RESEARCH = "legal_research"
    MIXED_LEGAL_ANALYSIS = "mixed_legal_analysis"
    GENERAL_LEGAL_QA = "general_legal_qa"
    SUMMARIZATION = "summarization"
    DRAFTING = "drafting"
    HEARING_PREP = "hearing_prep"
    COMPARE_PROVISIONS = "compare_provisions"
    REVIEW_TABLE = "review_table"
    PLAYBOOK_REVIEW = "playbook_review"
    REDLINE = "redline"
    GENERAL_CHAT = "general_chat"


class RetrievalStrategy(str, Enum):
    LEGAL_FIRST = "legal_first"
    DOCUMENT_SCOPED = "document_scoped"
    PRIVATE_DOCS_FIRST = "private_docs_first"
    MIXED_DOC_AND_LAW = "mixed_doc_and_law"
    NONE = "none"


@dataclass(slots=True)
class LegalQueryPlan:
    """Structured plan for retrieval, reasoning, and answer mode."""

    original_question: str
    task: Task
    answer_mode: AnswerMode
    retrieval_strategy: RetrievalStrategy
    complexity: QueryComplexity
    requires_legal_authority: bool
    uploaded_document_primary: bool
    document_scoped: bool
    legal_area: str | None = None
    entities: list[str] = field(default_factory=list)
    statutes: list[str] = field(default_factory=list)
    sections: list[str] = field(default_factory=list)
    cases: list[str] = field(default_factory=list)
    courts: list[str] = field(default_factory=list)
    jurisdiction: str | None = None
    matter_scope: bool = False
    conversation_scope: bool = False
    document_id: str | None = None
    confidence: float = 0.0
    reason: str = ""
    web_search: bool = False
    quick_action: str | None = None

    def to_metadata(self) -> dict[str, object]:
        return {
            "task": self.task.value if hasattr(self.task, "value") else str(self.task),
            "answer_mode": self.answer_mode.value,
            "retrieval_strategy": self.retrieval_strategy.value,
            "complexity": self.complexity.value,
            "requires_legal_authority": self.requires_legal_authority,
            "uploaded_document_primary": self.uploaded_document_primary,
            "document_scoped": self.document_scoped,
            "legal_area": self.legal_area,
            "entities": list(self.entities),
            "statutes": list(self.statutes),
            "sections": list(self.sections),
            "cases": list(self.cases),
            "courts": list(self.courts),
            "jurisdiction": self.jurisdiction,
            "matter_scope": self.matter_scope,
            "conversation_scope": self.conversation_scope,
            "document_id": self.document_id,
            "confidence": self.confidence,
            "reason": self.reason,
            "web_search": self.web_search,
            "quick_action": self.quick_action,
        }


_SECTION_PATTERN = re.compile(
    r"(?:(?:\b(?:section|sec\.?|u/s|article)\s+)|(?:دفعہ\s*)|(?:\bdafaa?\s+))"
    r"([0-9A-Za-z\-]+)",
    re.IGNORECASE,
)

_CASE_PATTERN = re.compile(
    r"\b(?:PLD|SCMR|MLD|YLR|CLC|PCr\.LJ)\s+\d{4}\s+[A-Za-z]+\s+\d+",
    re.IGNORECASE,
)

_COURT_PATTERN = re.compile(
    r"\b(?:supreme court|lahore high court|sindh high court|"
    r"islamabad high court|peshawar high court|balochistan high court|"
    r"federal shariat court)\b",
    re.IGNORECASE,
)

_AREA_KEYWORDS = {
    "criminal": (
        "bail",
        "fir",
        "murder",
        "ppc",
        "crpc",
        "criminal",
        "ضمانت",
        "zamanat",
        "zamaanat",
    ),
    "civil": ("injunction", "specific performance", "cpc", "civil", "recovery"),
    "constitutional": ("constitution", "writ", "fundamental rights"),
    "family": ("maintenance", "inheritance", "family", "nikah"),
    "commercial": ("contract", "arbitration", "company", "banking"),
    "energy": ("electricity", "wapda", "54-c", "54 c"),
}


class LegalQueryPlanner:
    """
    Dedicated planning layer before retrieval and reasoning.

    Reuses QueryRouter for task detection and enriches it into an
    explicit retrieval/answer plan.
    """

    def __init__(self, router: QueryRouter | None = None) -> None:
        self._router = router or QueryRouter()

    async def plan(
        self,
        *,
        question: str,
        has_uploaded_documents: bool = False,
        document_id: str | None = None,
        matter_id: str | None = None,
        conversation_id: str | None = None,
        history: list[Message] | None = None,
        quick_action: str | int | None = None,
        web_search: bool = False,
    ) -> LegalQueryPlan:
        from app.services.quick_actions import (
            resolve_from_message,
            resolve_quick_action,
        )

        action = resolve_quick_action(quick_action) or resolve_from_message(
            question
        )
        route = await self._router.route(
            question,
            has_uploaded_documents=has_uploaded_documents,
            quick_action=quick_action,
        )
        task = route.task

        # Explicit document scope: keep mixed analysis if user asks for law check.
        # Quick actions keep their own workflow even when a file is attached.
        if (
            document_id
            and action is None
            and task not in {
                Task.SUMMARIZATION,
                Task.DOCUMENT_QA,
                Task.MIXED_QA,
                Task.CONTRACT_REVIEW,
                Task.REVIEW_TABLE,
                Task.PLAYBOOK_REVIEW,
                Task.REDLINE,
            }
        ):
            task = Task.DOCUMENT_QA

        answer_mode = self._answer_mode(task, has_uploaded_documents=has_uploaded_documents)
        retrieval_strategy = self._retrieval_strategy(
            task=task,
            document_id=document_id,
            answer_mode=answer_mode,
        )
        uploaded_primary = answer_mode in {
            AnswerMode.DOCUMENT_QA,
            AnswerMode.SUMMARIZATION,
            AnswerMode.REVIEW_TABLE,
            AnswerMode.PLAYBOOK_REVIEW,
            AnswerMode.REDLINE,
        }
        requires_authority = answer_mode in {
            AnswerMode.LEGAL_RESEARCH,
            AnswerMode.MIXED_LEGAL_ANALYSIS,
            AnswerMode.GENERAL_LEGAL_QA,
            AnswerMode.HEARING_PREP,
            AnswerMode.COMPARE_PROVISIONS,
        }
        document_scoped = bool(document_id) and answer_mode in {
            AnswerMode.DOCUMENT_QA,
            AnswerMode.SUMMARIZATION,
            AnswerMode.REVIEW_TABLE,
            AnswerMode.PLAYBOOK_REVIEW,
            AnswerMode.REDLINE,
        }

        follow_up = bool(history) and len(question.split()) <= 12
        complexity = classify_query_complexity(
            question,
            task=task,
            mixed_qa=answer_mode == AnswerMode.MIXED_LEGAL_ANALYSIS,
            document_qa=answer_mode
            in {AnswerMode.DOCUMENT_QA, AnswerMode.SUMMARIZATION},
            follow_up=follow_up,
        )

        sections = [m.group(1) for m in _SECTION_PATTERN.finditer(question)]
        cases = [m.group(0) for m in _CASE_PATTERN.finditer(question)]
        courts = [m.group(0).lower() for m in _COURT_PATTERN.finditer(question)]
        entities = self._extract_entities(question)
        legal_area = self._detect_legal_area(question)
        jurisdiction = self._detect_jurisdiction(question)

        use_web = bool(web_search)
        if document_scoped and not web_search:
            use_web = False
        if use_web and retrieval_strategy == RetrievalStrategy.NONE:
            retrieval_strategy = RetrievalStrategy.LEGAL_FIRST

        return LegalQueryPlan(
            original_question=question,
            task=task,
            answer_mode=answer_mode,
            retrieval_strategy=retrieval_strategy,
            complexity=complexity,
            requires_legal_authority=requires_authority and not uploaded_primary,
            uploaded_document_primary=uploaded_primary,
            document_scoped=document_scoped,
            legal_area=legal_area,
            entities=entities,
            statutes=self._extract_statutes(question),
            sections=sections,
            cases=cases,
            courts=courts,
            jurisdiction=jurisdiction,
            matter_scope=bool(matter_id),
            conversation_scope=bool(conversation_id),
            document_id=document_id,
            confidence=route.confidence,
            reason=route.reason,
            web_search=use_web,
            quick_action=action.slug if action else None,
        )

    @staticmethod
    def _answer_mode(task: Task, *, has_uploaded_documents: bool) -> AnswerMode:
        if task == Task.SUMMARIZATION:
            return AnswerMode.SUMMARIZATION
        if task == Task.MIXED_QA:
            return AnswerMode.MIXED_LEGAL_ANALYSIS
        if task == Task.DOCUMENT_QA:
            return AnswerMode.DOCUMENT_QA
        if task in {
            Task.LEGAL_NOTICE,
            Task.PLAINT,
            Task.CONTRACT_REVIEW,
            Task.DOCUMENT_DRAFT,
        }:
            return AnswerMode.DRAFTING
        if task == Task.HEARING_PREP:
            return AnswerMode.HEARING_PREP
        if task == Task.COMPARE_PROVISIONS:
            return AnswerMode.COMPARE_PROVISIONS
        if task == Task.REVIEW_TABLE:
            return AnswerMode.REVIEW_TABLE
        if task == Task.PLAYBOOK_REVIEW:
            return AnswerMode.PLAYBOOK_REVIEW
        if task == Task.REDLINE:
            return AnswerMode.REDLINE
        if task in {
            Task.LEGAL_QA,
            Task.CASE_SEARCH,
            Task.STATUTE_SEARCH,
            Task.CASE_ANALYSIS,
        }:
            return AnswerMode.LEGAL_RESEARCH
        if has_uploaded_documents and task == Task.GENERAL_CHAT:
            return AnswerMode.GENERAL_LEGAL_QA
        if task == Task.GENERAL_CHAT:
            return AnswerMode.GENERAL_CHAT
        return AnswerMode.GENERAL_LEGAL_QA

    @staticmethod
    def _retrieval_strategy(
        *,
        task: Task,
        document_id: str | None,
        answer_mode: AnswerMode,
    ) -> RetrievalStrategy:
        if document_id and answer_mode in {
            AnswerMode.DOCUMENT_QA,
            AnswerMode.SUMMARIZATION,
        }:
            return RetrievalStrategy.DOCUMENT_SCOPED
        if answer_mode == AnswerMode.MIXED_LEGAL_ANALYSIS:
            return RetrievalStrategy.MIXED_DOC_AND_LAW
        if answer_mode in {AnswerMode.DOCUMENT_QA, AnswerMode.SUMMARIZATION}:
            return RetrievalStrategy.PRIVATE_DOCS_FIRST
        if answer_mode == AnswerMode.GENERAL_CHAT:
            return RetrievalStrategy.NONE
        if answer_mode == AnswerMode.REVIEW_TABLE:
            return RetrievalStrategy.NONE
        if answer_mode in {
            AnswerMode.PLAYBOOK_REVIEW,
            AnswerMode.REDLINE,
        }:
            return RetrievalStrategy.NONE
        return RetrievalStrategy.LEGAL_FIRST

    @staticmethod
    def _extract_entities(question: str) -> list[str]:
        entities: list[str] = []
        for match in _SECTION_PATTERN.finditer(question):
            entities.append(f"section {match.group(1)}")
        lower = question.lower()
        for token in (
            "double jeopardy",
            "injunction",
            "discretion",
            "clog",
            "bail",
            "appeal",
            "limitation",
            "ضمانت",
            "zamanat",
            "zamaanat",
        ):
            if token in lower:
                entities.append("bail" if token in {"ضمانت", "zamanat", "zamaanat"} else token)
        return list(dict.fromkeys(entities))

    @staticmethod
    def _extract_statutes(question: str) -> list[str]:
        lower = question.lower()
        found: list[str] = []
        mapping = {
            "electricity act": "Electricity Act, 1910",
            "pakistan penal code": "Pakistan Penal Code",
            "ppc": "Pakistan Penal Code",
            "crpc": "Code of Criminal Procedure",
            "cpc": "Code of Civil Procedure",
            "constitution": "Constitution of Pakistan",
            "limitation act": "Limitation Act",
            "contract act": "Contract Act",
        }
        for key, value in mapping.items():
            if key in lower:
                found.append(value)
        if re.search(r"\b54[\s\-]?c\b", lower) and "Electricity Act, 1910" not in found:
            found.append("Electricity Act, 1910")
        return found

    @staticmethod
    def _detect_legal_area(question: str) -> str | None:
        lower = question.lower()
        for area, keywords in _AREA_KEYWORDS.items():
            if any(keyword in lower for keyword in keywords):
                return area
        return None

    @staticmethod
    def _detect_jurisdiction(question: str) -> str | None:
        lower = question.lower()
        for province in (
            "punjab",
            "sindh",
            "khyber pakhtunkhwa",
            "balochistan",
            "islamabad",
            "pakistan",
        ):
            if province in lower:
                return province
        return "pakistan"
