from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.rag.context_resolver import ContextResolver
from app.rag.models import Message
from app.rag.section_ids import normalize_section_id
from app.services.query_router import Task


@dataclass(slots=True)
class RewriteResult:
    original_query: str
    rewritten_query: str
    resolved_query: str
    legal_terms: list[str]
    filters: dict[str, str]
    intent: str
    used_conversation_context: bool = False
    active_context: dict[str, object] | None = None


class BaseQueryRewriter(ABC):

    @abstractmethod
    async def rewrite(
        self,
        *,
        question: str,
        task: str | None = None,
        history: list[Message] | None = None,
    ) -> RewriteResult:
        raise NotImplementedError


class QueryRewriter(BaseQueryRewriter):
    """
    Query preprocessing layer before RAG retrieval.
    """

    LAW_MAP = {
        "ppc": "Pakistan Penal Code",
        "crpc": "Code of Criminal Procedure",
        "cpc": "Code of Civil Procedure",
        "qso": "Qanun-e-Shahadat Order",
        "ica": "Contract Act",
        "limitation act": "Limitation Act",
        "constitution": "Constitution of Pakistan",
        "cp": "Constitution Petition",
    }

    # Intentionally empty: do not hard-map bare section numbers to a statute.
    # Corpus hits must come from embeddings + metadata the user already indexed.
    SECTION_EXPANSIONS: tuple[tuple[re.Pattern[str], str], ...] = ()

    COURTS = [
        "supreme court",
        "lahore high court",
        "sindh high court",
        "islamabad high court",
        "balochistan high court",
        "peshawar high court",
        "federal shariat court",
    ]

    def __init__(
        self,
        *,
        context_resolver: ContextResolver | None = None,
    ) -> None:
        self._context_resolver = (
            context_resolver or ContextResolver()
        )

    async def rewrite(
        self,
        *,
        question: str,
        task: Task | str | None = None,
        history: list[Message] | None = None,
    ) -> RewriteResult:

        original = self._normalize(question)
        active = self._context_resolver.build_active_context(
            original,
            history=history,
        )
        resolved = self._context_resolver.resolve(
            original,
            history=history,
            active_context=active,
        )
        used_context = resolved.strip().lower() != original.strip().lower()

        # Filters from the user's resolved question only — never from rewrite
        # enrichment text (years/statute names must not become hard Qdrant filters).
        filters = self._extract_filters(resolved)

        query = self._expand_abbreviations(resolved)
        query = self._expand_section_aliases(query)
        query = self._enrich_multilingual_legal_terms(query)
        legal_terms = self._extract_legal_terms(query)
        query = self._append_keywords(query=query, task=task)

        return RewriteResult(
            original_query=question,
            rewritten_query=query,
            resolved_query=resolved,
            legal_terms=legal_terms,
            filters=filters,
            intent=task or "unknown",
            used_conversation_context=used_context,
            active_context=active.to_metadata(),
        )

    def _normalize(self, query: str) -> str:
        return re.sub(r"\s+", " ", query.strip())

    def _expand_abbreviations(self, query: str) -> str:
        for short, full in self.LAW_MAP.items():
            query = re.sub(
                rf"\b{short}\b",
                full,
                query,
                flags=re.IGNORECASE,
            )
        return query

    def _expand_section_aliases(self, query: str) -> str:
        extras: list[str] = []
        seen: set[str] = set()
        for pattern, expansion in self.SECTION_EXPANSIONS:
            key = expansion.lower()
            if key in seen or key in query.lower():
                continue
            if pattern.search(query):
                extras.append(expansion)
                seen.add(key)
        if not extras:
            return query
        return f"{query} {' '.join(extras)}"

    def _enrich_multilingual_legal_terms(self, query: str) -> str:
        """
        Append English legal anchors for retrieval without translating the query.

        Original wording, citations, and section numbers stay intact.
        """
        extras: list[str] = []
        seen: set[str] = set()

        for match in re.finditer(r"دفعہ\s*([0-9A-Za-z\-]+)", query):
            term = f"section {match.group(1)}"
            if term.lower() not in seen and term.lower() not in query.lower():
                extras.append(term)
                seen.add(term.lower())

        for match in re.finditer(r"\bdafaa?\s+([0-9A-Za-z\-]+)", query, re.I):
            term = f"section {match.group(1)}"
            if term.lower() not in seen and term.lower() not in query.lower():
                extras.append(term)
                seen.add(term.lower())

        if re.search(r"ضمانت", query) and "bail" not in query.lower():
            extras.append("bail")
        if re.search(r"\bzamaan?at\b", query, re.I) and "bail" not in query.lower():
            extras.append("bail")

        if extras:
            return f"{query} {' '.join(extras)}"
        return query

    def _extract_filters(self, query: str) -> dict[str, str]:
        filters: dict[str, str] = {}

        year = re.search(r"\b(19\d{2}|20\d{2})\b", query)
        if year:
            filters["year"] = year.group(1)

        lower_query = query.lower()

        for court in self.COURTS:
            if court in lower_query:
                filters["court"] = court
                break

        section = re.search(
            r"(section|sec\.?|u/s|dafaa?)\s*([0-9A-Za-z\-]+)",
            lower_query,
        )
        if section:
            filters["section"] = normalize_section_id(section.group(2))
        else:
            arabic_section = re.search(r"دفعہ\s*([0-9A-Za-z\-]+)", query)
            if arabic_section:
                filters["section"] = normalize_section_id(
                    arabic_section.group(1)
                )
            else:
                bare = re.search(r"\b(\d{1,4}[\s\-]?[a-z])\b", lower_query)
                if bare and re.search(r"\bsection\b|\bsec\.?\b|\bu/s\b", lower_query):
                    filters["section"] = normalize_section_id(bare.group(1))

        return filters

    def _extract_legal_terms(self, query: str) -> list[str]:
        terms: list[str] = []
        keywords = [
            "murder",
            "theft",
            "bail",
            "contract",
            "negligence",
            "inheritance",
            "appeal",
            "injunction",
            "recovery",
            "specific performance",
            "arbitration",
            "limitation",
            "constitutional petition",
        ]

        lower = query.lower()
        for keyword in keywords:
            if keyword in lower:
                terms.append(keyword)

        if "ضمانت" in query and "bail" not in terms:
            terms.append("bail")
        if re.search(r"دفعہ", query) and not any(
            item.startswith("section") or item == "section" for item in terms
        ):
            terms.append("section")

        return terms

    def _append_keywords(
        self,
        *,
        query: str,
        task: str | None,
    ) -> str:
        enrichment = {
            "legal_qa": (
                "Pakistan law statutes legal provisions case law"
            ),
            "case_search": (
                "judgment precedent court decision case law"
            ),
            "statute_search": (
                "act ordinance statute section legal provision"
            ),
            "plaint": (
                "cause of action jurisdiction relief prayer"
            ),
            "legal_notice": (
                "formal notice demand notice legal communication"
            ),
            "document_draft": (
                "legal draft format parties facts relief prayer"
            ),
            "hearing_prep": (
                "hearing issues oral submissions authorities objections"
            ),
            "compare_provisions": (
                "compare sections articles provisions difference"
            ),
            "contract_review": (
                "contract clause risk obligation termination liability"
            ),
            "summarization": (
                "key points summary issues parties"
            ),
        }

        task_name = task.value if hasattr(task, "value") else str(task or "")
        if task_name in enrichment:
            return f"{query} {enrichment[task_name]}"

        return query
