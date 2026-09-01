from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class Task(str, Enum):
    LEGAL_QA = "legal_qa"
    CASE_SEARCH = "case_search"
    STATUTE_SEARCH = "statute_search"
    CASE_ANALYSIS = "case_analysis"
    CONTRACT_REVIEW = "contract_review"
    LEGAL_NOTICE = "legal_notice"
    PLAINT = "plaint"
    DOCUMENT_QA = "document_qa"
    SUMMARIZATION = "summarization"
    MIXED_QA = "mixed_qa"
    DOCUMENT_DRAFT = "document_draft"
    HEARING_PREP = "hearing_prep"
    COMPARE_PROVISIONS = "compare_provisions"
    REVIEW_TABLE = "review_table"
    PLAYBOOK_REVIEW = "playbook_review"
    REDLINE = "redline"
    GENERAL_CHAT = "general_chat"


@dataclass(slots=True)
class RouteResult:
    task: Task
    confidence: float
    reason: str
    web_search: bool = False


class QueryRouter:
    """
    Production rule-based router.

    Runs before RAG.

    Responsibilities

    - Detect workflow
    - Avoid unnecessary LLM calls
    - Keep latency very low
    """

    LEGAL_NOTICE_KEYWORDS = {
        "legal notice",
        "notice",
        "demand notice",
        "send notice",
    }

    PLAINT_KEYWORDS = {
        "plaint",
        "recovery suit",
        "civil suit",
        "draft plaint",
        "prepare plaint",
    }

    REVIEW_TABLE_KEYWORDS = {
        "review table",
        "diligence table",
        "extraction table",
        "extract clauses from all",
        "across all documents",
        "across all contracts",
        "compare all contracts",
        "compare these contracts",
        "vault review",
    }

    PLAYBOOK_KEYWORDS = {
        "playbook review",
        "run playbook",
        "apply playbook",
        "against playbook",
        "playbook check",
        "score against playbook",
        "off-playbook",
        "off playbook",
    }

    REDLINE_KEYWORDS = {
        "redline",
        "red line",
        "show redline",
        "compare drafts",
        "compare versions",
        "diff the contracts",
        "markup the contract",
    }

    CONTRACT_KEYWORDS = {
        "contract",
        "agreement",
        "nda",
        "lease",
        "employment contract",
        "sale agreement",
        "review contract",
        "contract risk",
        "contract analysis",
        "agreement review",
        "agreement analysis",
        "extract clauses",
        "clause extraction",
        "key terms",
    }

    CASE_ANALYSIS_KEYWORDS = {
        "analyze case",
        "analyse case",
        "case analysis",
        "legal strategy",
        "strength",
        "weakness",
        "risk",
    }

    CASE_SEARCH_KEYWORDS = {
        "judgment",
        "judgement",
        "case law",
        "precedent",
        "citation",
        "find case",
        "find judgment",
    }

    STATUTE_KEYWORDS = {
        "section",
        "article",
        "ordinance",
        "act",
        "rule",
        "ppc",
        "crpc",
        "cpc",
        "constitution",
        "qanun",
        "دفعہ",
        "دفعه",
        "dafa",
        "dafaa",
    }

    SUMMARY_KEYWORDS = {
        "summarize",
        "summary",
        "summarise",
        "tl;dr",
        "brief",
    }

    MATTER_DOC_CONTENT_KEYWORDS = {
        "problem",
        "issue",
        "identify",
        "identified",
        "author",
        "article",
        "according to",
        "this article",
        "the article",
        "the document",
        "this document",
        "in the document",
        "in this document",
        "uploaded",
        "attached",
        "clog",
        "discretion",
        "discusses",
        "argues",
        "contends",
        "what does",
        "what is the",
        "what are the",
    }

    GENERAL_LAW_PHRASES = {
        "what is the law",
        "what does the law",
        "under pakistan law",
        "under the act",
        "statute provides",
        "legal position",
        "binding authority",
        "case law",
        "precedent",
        "what is section",
        "define section",
    }

    MIXED_LEGAL_DOC_KEYWORDS = {
        "reflect pakistani law",
        "reflect the law",
        "consistent with the law",
        "consistent with pakistani law",
        "compare with the law",
        "compare to the law",
        "compare with pakistani",
        "does the argument",
        "does this argument",
        "is this argument",
        "is the argument",
        "is the author's argument",
        "author's argument",
        "legally correct",
        "legally accurate",
        "binding law",
        "authoritative law",
        "actual law",
        "what the law says",
        "under pakistani law",
        "under the law",
    }

    DRAFT_KEYWORDS = {
        "draft a",
        "draft the",
        "drafting",
        "prepare a document",
        "prepare document",
        "write a notice",
        "write a plaint",
        "write an affidavit",
        "draft the document",
    }

    HEARING_KEYWORDS = {
        "prepare for hearing",
        "hearing note",
        "hearing notes",
        "oral submissions",
        "arguments for hearing",
        "prepare arguments",
        "court hearing",
        "hearing preparation",
    }

    COMPARE_KEYWORDS = {
        "compare section",
        "compare article",
        "compare provisions",
        "compare the provisions",
        "compare legal provisions",
        "compare laws",
        "side by side",
        "difference between section",
        "distinction between section",
    }

    WEB_SEARCH_KEYWORDS = {
        "search the internet",
        "search internet",
        "search online",
        "search the web",
        "web search",
        "look online",
        "look it up online",
        "from the internet",
        "on the internet",
        "google this",
        "latest judgment",
        "recent judgment",
        "recent case law",
        "current law",
        "latest amendment",
    }

    DOCUMENT_QA_KEYWORDS = {
        "this document",
        "uploaded document",
        "uploaded file",
        "attached file",
        "attached document",
        "this agreement",
        "this contract",
        "this pdf",
        "attached agreement",
        "attached contract",
        "review this",
        "analyze this",
        "analyse this",
        "what are the issues",
        "find problems",
    }

    async def detect(
        self,
        question: str,
        has_uploaded_documents: bool = False,
    ) -> Task:
        return (await self.route(question, has_uploaded_documents)).task

    async def route(
        self,
        question: str,
        has_uploaded_documents: bool = False,
        quick_action: str | int | None = None,
    ) -> RouteResult:

        q = self._normalize(question)
        wants_web = self._wants_web_search(q)

        forced = self._from_quick_action(quick_action)
        if forced is None:
            forced = self._from_quick_action(question)
        if forced is not None:
            return RouteResult(
                task=self._task_from_name(forced.task),
                confidence=0.99,
                reason=f"quick action {forced.slug}",
                web_search=wants_web or forced.enables_web_search,
            )

        if self._contains(q, self.REVIEW_TABLE_KEYWORDS):
            return RouteResult(
                Task.REVIEW_TABLE,
                0.98,
                "review table",
                wants_web,
            )

        if self._contains(q, self.PLAYBOOK_KEYWORDS):
            return RouteResult(
                Task.PLAYBOOK_REVIEW,
                0.98,
                "playbook review",
                wants_web,
            )

        if self._contains(q, self.REDLINE_KEYWORDS):
            return RouteResult(
                Task.REDLINE,
                0.98,
                "redline",
                wants_web,
            )

        if has_uploaded_documents:
            if self._contains(q, self.SUMMARY_KEYWORDS):
                return RouteResult(
                    task=Task.SUMMARIZATION,
                    confidence=0.99,
                    reason="uploaded document summary",
                    web_search=wants_web,
                )

            if self._is_mixed_legal_document_query(q):
                return RouteResult(
                    task=Task.MIXED_QA,
                    confidence=0.96,
                    reason="mixed document and legal authority",
                    web_search=wants_web,
                )

            if self._contains(q, self.DOCUMENT_QA_KEYWORDS):
                return RouteResult(
                    task=Task.DOCUMENT_QA,
                    confidence=0.99,
                    reason="document question",
                    web_search=wants_web,
                )

            if self._is_matter_document_content_query(q):
                return RouteResult(
                    task=Task.DOCUMENT_QA,
                    confidence=0.95,
                    reason="matter document content analysis",
                    web_search=wants_web,
                )

        if self._contains(q, self.HEARING_KEYWORDS):
            return RouteResult(
                Task.HEARING_PREP,
                0.98,
                "hearing preparation",
                wants_web,
            )

        if self._contains(q, self.COMPARE_KEYWORDS):
            return RouteResult(
                Task.COMPARE_PROVISIONS,
                0.97,
                "compare provisions",
                True,
            )

        if self._contains(q, self.LEGAL_NOTICE_KEYWORDS):
            return RouteResult(
                Task.LEGAL_NOTICE,
                0.98,
                "legal notice keywords",
                wants_web,
            )

        if self._contains(q, self.PLAINT_KEYWORDS):
            return RouteResult(
                Task.PLAINT,
                0.98,
                "plaint keywords",
                wants_web,
            )

        if self._contains(q, self.DRAFT_KEYWORDS) and not has_uploaded_documents:
            return RouteResult(
                Task.DOCUMENT_DRAFT,
                0.96,
                "document drafting",
                wants_web,
            )

        if self._contains(q, self.CONTRACT_KEYWORDS):
            return RouteResult(
                Task.CONTRACT_REVIEW,
                0.97,
                "contract keywords",
                wants_web,
            )

        if self._contains(q, self.CASE_ANALYSIS_KEYWORDS):
            return RouteResult(
                Task.CASE_ANALYSIS,
                0.96,
                "case analysis keywords",
                wants_web,
            )

        if self._contains(q, self.CASE_SEARCH_KEYWORDS):
            return RouteResult(
                Task.CASE_SEARCH,
                0.95,
                "case law search",
                wants_web,
            )

        if self._is_statute_query(q):
            return RouteResult(
                Task.STATUTE_SEARCH,
                0.94,
                "statute lookup",
                wants_web,
            )

        if self._looks_like_legal_question(q):
            return RouteResult(
                Task.LEGAL_QA,
                0.90,
                "general legal question",
                wants_web,
            )

        if wants_web:
            return RouteResult(
                Task.LEGAL_QA,
                0.80,
                "explicit internet search",
                True,
            )

        return RouteResult(
            Task.GENERAL_CHAT,
            0.70,
            "fallback",
            False,
        )

    def _wants_web_search(self, text: str) -> bool:
        return self._contains(text, self.WEB_SEARCH_KEYWORDS)

    @staticmethod
    def _from_quick_action(value: str | int | None):
        from app.services.quick_actions import resolve_quick_action

        return resolve_quick_action(value)

    @staticmethod
    def _task_from_name(name: str) -> Task:
        try:
            return Task(name)
        except ValueError:
            return Task.LEGAL_QA

    @staticmethod
    def _normalize(text: str) -> str:
        text = text.lower()
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def _contains(
        text: str,
        keywords: Iterable[str],
    ) -> bool:
        return any(keyword in text for keyword in keywords)

    def _is_statute_query(self, text: str) -> bool:

        if self._contains(text, self.STATUTE_KEYWORDS):
            return True

        if re.search(r"\bsection\s+\d+", text):
            return True

        if re.search(r"\barticle\s+\d+", text):
            return True

        if re.search(r"\bs\.\s*\d+", text):
            return True

        if re.search(r"دفعہ\s*[0-9A-Za-z\-]+", text):
            return True

        if re.search(r"\bdafaa?\s+[0-9A-Za-z\-]+", text):
            return True

        return False

    def _is_matter_document_content_query(self, text: str) -> bool:
        """
        Questions about problems/issues/authors in uploaded material
        should use document Q&A even when they mention sections.
        """
        if self._contains(text, self.GENERAL_LAW_PHRASES):
            return False

        if self._contains(text, self.MATTER_DOC_CONTENT_KEYWORDS):
            return True

        if re.search(r"\bsection\s+[0-9A-Za-z\-]+", text):
            if any(
                word in text
                for word in (
                    "problem",
                    "issue",
                    "identify",
                    "author",
                    "article",
                    "document",
                )
            ):
                return True

        return False

    def _is_mixed_legal_document_query(self, text: str) -> bool:
        if self._contains(text, self.MIXED_LEGAL_DOC_KEYWORDS):
            return True

        if "document" in text or "article" in text or "uploaded" in text or "author" in text:
            if any(
                phrase in text
                for phrase in (
                    "pakistani law",
                    "the law",
                    "statute",
                    "binding",
                    "precedent",
                    "case law",
                    "legally correct",
                    "legally accurate",
                    "correct under",
                )
            ):
                return True

        return False

    @staticmethod
    def _looks_like_legal_question(text: str) -> bool:

        legal_words = (
            # General legal terms
            "law",
            "legal",
            "court",
            "judge",
            "lawyer",
            "advocate",
            "justice",
            "hearing",
            "judgment",
            "judgement",

            # Criminal law Pakistan
            "fir",
            "fard",
            "challan",
            "remand",
            "bail",
            "anticipatory bail",
            "pre arrest bail",
            "post arrest bail",
            "arrest",
            "offence",
            "crime",
            "criminal",
            "murder",
            "theft",
            "fraud",
            "ضمانت",
            "مقدمہ",
            "عدالت",
            "قانون",
            "فوجداری",
            "zamanat",
            "zamaanat",
            "qanoon",
            "muqadma",
            "muqadama",

            # Civil law
            "civil",
            "recovery",
            "injunction",
            "specific performance",
            "damages",
            "negligence",
            "limitation",
            "property",
            "tenant",
            "landlord",
            "inheritance",
            "succession",

            # Courts / procedure
            "appeal",
            "petition",
            "writ",
            "revision",
            "stay order",
            "katcheri",
            "thana",
            "wakalatnama",

            # Evidence / dispute resolution
            "evidence",
            "arbitration",
            "contract",
            "agreement",
        )

        if any(word in text for word in legal_words):
            return True

        if text.startswith(("what", "how", "when", "whether", "can", "is")):
            return True

        return False