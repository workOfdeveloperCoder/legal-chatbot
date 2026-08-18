from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import HTTPException, status

from app.core.config import settings
from app.llm.base import BaseLLM
from app.llm.errors import LLMAllProvidersFailed, LLMCancelledError, LLMError
from app.llm.versions import current_pipeline_versions

from app.rag.prompt_builder import PromptBuilder
from app.rag.models import (
    Memory,
    Message,
)

from app.rag.repository import BaseRetriever
from app.rag.query_rewriter import QueryRewriter
from app.services.query_router import Task

from app.rag.evidence_engine import EvidenceEngine
from app.rag.legal_query_planner import (
    AnswerMode,
    LegalQueryPlan,
    LegalQueryPlanner,
)
from app.rag.reasoning_pipeline import ReasoningPipeline

from app.services.response_formatter import ResponseFormatter

from app.rag.answer_guard import (
    AnswerGuard,
    EvidenceStrength,
)
from app.rag.answer_cleanup import clean_answer_for_display
from app.rag.document_metadata import (
    enrich_chunks_from_documents,
    load_documents_for_chunks,
)
from app.rag.prompts import LEGAL_SYSTEM_PROMPT
from app.rag.token_budget import TokenBudgetManager
from app.rag.language import detect_language


logger = logging.getLogger(__name__)

DOCUMENT_NOT_READY_DETAIL = (
    "The requested document could not be read or indexed. "
    "Please re-upload or process the document before asking "
    "questions about it."
)


def _as_task(task: Task | str) -> Task | str:
    if isinstance(task, Task):
        return task
    try:
        return Task(str(task))
    except ValueError:
        return task


def _safe_str_attr(obj: object, name: str, default: str) -> str:
    value = getattr(obj, name, None)
    if value is None:
        return default
    # AsyncMock / MagicMock attribute access can yield awaitables.
    try:
        from unittest.mock import Mock

        if isinstance(value, Mock):
            return default
    except Exception:  # noqa: BLE001
        pass
    if hasattr(type(value), "__await__") or type(value).__name__ in {
        "coroutine",
        "Coroutine",
    }:
        return default
    try:
        text = str(value)
    except Exception:  # noqa: BLE001
        return default
    if "coroutine" in text.lower():
        return default
    return text or default


class RAGService:
    """
    Production Legal Chatbot RAG Pipeline.

    Flow:
      plan → rewrite → retrieve → evidence engine → token budget →
      prompt → reason → claim guard → format resources → clean answer
    """

    def __init__(
        self,
        *,
        llm: BaseLLM,
        retriever: BaseRetriever,
        query_rewriter: QueryRewriter,
        prompt_builder: PromptBuilder,
        response_formatter: ResponseFormatter,
        answer_guard: AnswerGuard | None = None,
        document_repository=None,
        evidence_engine: EvidenceEngine | None = None,
        legal_query_planner: LegalQueryPlanner | None = None,
        token_budget_manager: TokenBudgetManager | None = None,
    ) -> None:

        self._llm = llm
        self._retriever = retriever
        self._query_rewriter = query_rewriter
        self._prompt_builder = prompt_builder
        self._response_formatter = response_formatter
        self._answer_guard = answer_guard or AnswerGuard()
        self._documents = document_repository
        self._reasoning = ReasoningPipeline(llm)
        self._evidence_engine = evidence_engine or EvidenceEngine()
        self._planner = legal_query_planner or LegalQueryPlanner()
        self._token_budget = token_budget_manager or TokenBudgetManager()

    async def execute(
        self,
        *,
        question: str,
        history: list[Message],
        memories: list[Memory],
        task: Task | str,
        user_id: str,
        matter_id: str | None = None,
        conversation_id: str | None = None,
        document_id: str | None = None,
        query_plan: LegalQueryPlan | None = None,
    ) -> dict[str, Any]:

        t0 = time.perf_counter()
        perf: dict[str, float] = {}
        llm_execution: dict[str, object] = {
            "retry_count": 0,
            "fallback_used": False,
            "error_type": None,
        }
        versions = current_pipeline_versions(
            model=_safe_str_attr(self._llm, "model_name", settings.CHAT_MODEL),
            provider=_safe_str_attr(self._llm, "provider_name", settings.LLM_PROVIDER),
        )
        language = detect_language(question)
        language_meta = language.to_metadata()

        task_obj = _as_task(task)
        plan = query_plan
        if plan is None:
            has_uploads = bool(document_id) or (
                isinstance(task_obj, Task)
                and task_obj
                in {Task.DOCUMENT_QA, Task.SUMMARIZATION, Task.MIXED_QA}
            )
            plan = await self._planner.plan(
                question=question,
                has_uploaded_documents=has_uploads,
                document_id=document_id,
                matter_id=matter_id,
                conversation_id=conversation_id,
                history=history,
            )
            if isinstance(task_obj, Task):
                # Caller-selected task wins (ChatService sends a plan; tests may
                # pass only task).
                plan.task = task_obj
                plan.answer_mode = self._planner._answer_mode(
                    task_obj,
                    has_uploaded_documents=has_uploads,
                )
                plan.retrieval_strategy = self._planner._retrieval_strategy(
                    task=task_obj,
                    document_id=document_id,
                    answer_mode=plan.answer_mode,
                )
                plan.uploaded_document_primary = plan.answer_mode in {
                    AnswerMode.DOCUMENT_QA,
                    AnswerMode.SUMMARIZATION,
                }
                plan.requires_legal_authority = (
                    plan.answer_mode
                    in {
                        AnswerMode.LEGAL_RESEARCH,
                        AnswerMode.MIXED_LEGAL_ANALYSIS,
                        AnswerMode.GENERAL_LEGAL_QA,
                    }
                    and not plan.uploaded_document_primary
                )
                plan.document_scoped = bool(document_id) and plan.answer_mode in {
                    AnswerMode.DOCUMENT_QA,
                    AnswerMode.SUMMARIZATION,
                }
        else:
            task_obj = plan.task

        mixed_qa_mode = plan.answer_mode == AnswerMode.MIXED_LEGAL_ANALYSIS
        document_qa_mode = plan.answer_mode in {
            AnswerMode.DOCUMENT_QA,
            AnswerMode.SUMMARIZATION,
            AnswerMode.MIXED_LEGAL_ANALYSIS,
        }
        document_task = plan.document_scoped
        needs_legal_authority = plan.requires_legal_authority
        complexity = plan.complexity

        logger.info(
            "RAG started user=%s matter=%s conversation=%s "
            "document_id=%s task=%s mode=%s strategy=%s "
            "language=%s response_language=%s style=%s override=%s",
            user_id,
            matter_id,
            conversation_id,
            document_id,
            task_obj,
            plan.answer_mode.value,
            plan.retrieval_strategy.value,
            language.language.value,
            language.response_language.value,
            language.response_style.value,
            language.explicit_override,
        )

        rewrite = await self._query_rewriter.rewrite(
            question=question,
            task=task_obj,
            history=history,
        )
        search_query = rewrite.rewritten_query
        conversation_context = rewrite.active_context

        t_retrieve = time.perf_counter()
        try:
            retrieval = await self._retriever.search(
                query=search_query,
                user_id=user_id,
                matter_id=matter_id,
                conversation_id=conversation_id,
                task=task_obj,
                document_id=document_id,
                limit=settings.RETRIEVAL_LIMIT,
                filters=rewrite.filters,
            )
        except Exception:
            logger.exception("Retrieval failed")
            if document_task:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=(
                        "Document retrieval is temporarily unavailable. "
                        "Please try again shortly."
                    ),
                )
            retrieval = None
        perf["retrieval_latency_ms"] = round(
            (time.perf_counter() - t_retrieve) * 1000,
            2,
        )

        if retrieval is None:
            from app.rag.models import RetrievalMetadata, RetrievalOutcome

            retrieval = RetrievalOutcome(
                chunks=[],
                metadata=RetrievalMetadata(
                    degraded_sources=["all"],
                ),
            )

        chunks = retrieval.chunks
        retrieval_metadata = retrieval.metadata

        if self._documents is not None and chunks:
            documents = await load_documents_for_chunks(
                self._documents,
                chunks,
            )
            chunks = enrich_chunks_from_documents(chunks, documents)

        if document_task:
            if not chunks:
                logger.warning(
                    "Document-scoped retrieval empty "
                    "document_id=%s user=%s",
                    document_id,
                    user_id,
                )
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=DOCUMENT_NOT_READY_DETAIL,
                )

            pure = [
                c
                for c in chunks
                if c.document_id == document_id
            ]
            dropped = len(chunks) - len(pure)
            if dropped:
                logger.error(
                    "Retrieval integrity failure: dropped %s "
                    "foreign chunks for document_id=%s "
                    "found_ids=%s",
                    dropped,
                    document_id,
                    sorted(
                        {
                            c.document_id
                            for c in chunks
                            if c.document_id
                        }
                    ),
                )
            chunks = pure
            if not chunks:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=DOCUMENT_NOT_READY_DETAIL,
                )

        evidence_bundle = self._evidence_engine.build(
            chunks,
            question=rewrite.resolved_query,
            document_task=document_task,
            document_qa_mode=document_qa_mode,
            requires_legal_authority=needs_legal_authority,
        )
        evidence_assessment = evidence_bundle.assessment
        evidence_strength = evidence_bundle.overall_strength
        chunks = evidence_bundle.chunks

        logger.info(
            "Retrieved chunks=%s legal=%s conversation=%s matter=%s "
            "degraded=%s evidence=%s document_qa=%s used_context=%s "
            "complexity=%s coverage=%.2f supporting=%s",
            len(chunks),
            retrieval_metadata.legal_chunks,
            retrieval_metadata.conversation_chunks,
            retrieval_metadata.matter_chunks,
            retrieval_metadata.degraded_sources,
            evidence_strength.value,
            document_qa_mode,
            rewrite.used_conversation_context,
            complexity.value,
            evidence_assessment.coverage if evidence_assessment else 0.0,
            len(evidence_bundle.supporting()),
        )

        if (
            evidence_strength == EvidenceStrength.NONE
            and needs_legal_authority
            and not document_qa_mode
        ):
            logger.info(
                "Insufficient evidence; skipping LLM user=%s",
                user_id,
            )
            insufficient_answer = (
                self._answer_guard.build_insufficient_evidence_answer(
                    evidence_strength=evidence_strength,
                    has_private_sources=bool(chunks),
                    language=language,
                )
            )
            return self._finalize_response(
                self._response_formatter.format(
                    answer=insufficient_answer,
                    chunks=chunks,
                    retrieval_metadata=retrieval_metadata,
                    degraded_sources=retrieval_metadata.degraded_sources,
                    evidence_strength=evidence_strength,
                    grounding_status=None,
                    citation_validation=None,
                    used_conversation_context=rewrite.used_conversation_context,
                    query_plan=plan.to_metadata(),
                    evidence_metadata=evidence_bundle.to_metadata(),
                    token_budget=None,
                    pipeline_versions=versions.to_dict(),
                    performance={
                        **perf,
                        "total_latency_ms": round(
                            (time.perf_counter() - t0) * 1000, 2
                        ),
                    },
                    llm_execution=llm_execution,
                    conversation_context=conversation_context,
                    language=language_meta,
                )
            )

        t_budget = time.perf_counter()
        packed = self._token_budget.prepare(
            question=question,
            history=history,
            chunks=chunks,
            memories=memories,
            system_prompt=LEGAL_SYSTEM_PROMPT,
            document_qa_mode=document_qa_mode or document_task,
        )
        chunks = packed.chunks
        history_for_prompt = packed.history
        token_budget_meta = (
            packed.metadata.to_dict() if packed.metadata else None
        )

        prompt = self._prompt_builder.build(
            question=question,
            history=history_for_prompt,
            chunks=chunks,
            memories=memories,
            task=task_obj,
            document_scoped=document_task,
            document_qa_mode=(
                document_qa_mode and not document_task and not mixed_qa_mode
            ),
            mixed_qa_mode=mixed_qa_mode,
            evidence_strength=evidence_strength,
            evidence_assessment=evidence_assessment,
            complexity=complexity,
            conversation_summary=packed.conversation_summary,
            active_legal_context=packed.active_legal_context,
            language=language,
        )

        # Post-format safety: drop evidence if scaffolding pushed over budget.
        packed = self._token_budget.ensure_prompt_within_budget(
            prompt,
            system_prompt=LEGAL_SYSTEM_PROMPT,
            packed=packed,
        )
        if packed.chunks is not chunks:
            chunks = packed.chunks
            prompt = self._prompt_builder.build(
                question=question,
                history=history_for_prompt,
                chunks=chunks,
                memories=memories,
                task=task_obj,
                document_scoped=document_task,
                document_qa_mode=(
                    document_qa_mode and not document_task and not mixed_qa_mode
                ),
                mixed_qa_mode=mixed_qa_mode,
                evidence_strength=evidence_strength,
                evidence_assessment=evidence_assessment,
                complexity=complexity,
                conversation_summary=packed.conversation_summary,
                active_legal_context=packed.active_legal_context,
                language=language,
            )
            token_budget_meta = (
                packed.metadata.to_dict() if packed.metadata else token_budget_meta
            )
        perf["token_budget_latency_ms"] = round(
            (time.perf_counter() - t_budget) * 1000,
            2,
        )
        perf["prompt_construction_latency_ms"] = perf["token_budget_latency_ms"]

        t_llm = time.perf_counter()
        try:
            generation = await self._reasoning.generate(
                user_prompt=prompt,
                complexity=complexity,
                temperature=settings.LLM_TEMPERATURE,
                max_tokens=packed.reserved_output_tokens,
            )
        except LLMCancelledError:
            raise HTTPException(
                status_code=499,
                detail="The request was cancelled.",
            )
        except (LLMAllProvidersFailed, LLMError) as exc:
            logger.exception("LLM generation failed: %s", exc)
            llm_execution["error_type"] = type(exc).__name__
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "The assistant is temporarily unavailable. "
                    "Please try again shortly."
                ),
            )
        except Exception:
            logger.exception("LLM generation failed")
            llm_execution["error_type"] = "unexpected"
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "The assistant is temporarily unavailable. "
                    "Please try again shortly."
                ),
            )
        perf["llm_latency_ms"] = round((time.perf_counter() - t_llm) * 1000, 2)

        gateway_stats = getattr(self._llm, "last_stats", None)
        if gateway_stats is not None:
            llm_execution["retry_count"] = gateway_stats.retry_count
            llm_execution["fallback_used"] = gateway_stats.fallback_used
            llm_execution["provider"] = gateway_stats.provider
            llm_execution["model"] = gateway_stats.model
        else:
            llm_execution["provider"] = versions.provider
            llm_execution["model"] = versions.model

        # Prefer provider-reported usage when available; keep budget estimate otherwise.
        if generation.prompt_tokens is not None and token_budget_meta is not None:
            token_budget_meta["input_tokens"] = generation.prompt_tokens
            if generation.completion_tokens is not None:
                token_budget_meta["output_tokens"] = generation.completion_tokens
            if generation.total_tokens is not None:
                token_budget_meta["total_tokens"] = generation.total_tokens

        logger.info(
            "LLM completed tokens=%s two_stage=%s complexity=%s "
            "degraded_sources=%s retries=%s fallback=%s",
            generation.total_tokens,
            generation.two_stage,
            complexity.value,
            retrieval_metadata.degraded_sources,
            llm_execution.get("retry_count"),
            llm_execution.get("fallback_used"),
        )

        t_guard = time.perf_counter()
        preliminary = self._response_formatter.format(
            answer=generation.content,
            chunks=chunks,
            prompt_tokens=generation.prompt_tokens,
            completion_tokens=generation.completion_tokens,
            total_tokens=generation.total_tokens,
            retrieval_metadata=retrieval_metadata,
            degraded_sources=retrieval_metadata.degraded_sources,
            evidence_strength=evidence_strength,
            used_conversation_context=rewrite.used_conversation_context,
            query_plan=plan.to_metadata(),
            evidence_metadata=evidence_bundle.to_metadata(),
            token_budget=token_budget_meta,
            pipeline_versions=versions.to_dict(),
            performance=None,
            llm_execution=llm_execution,
            conversation_context=conversation_context,
            language=language_meta,
        )

        guarded = self._answer_guard.process(
            answer=preliminary["answer"],
            chunks=chunks,
            sources=preliminary["sources"],
            evidence_strength=evidence_strength,
            skip_grounding=document_task,
            document_evidence_mode=document_qa_mode and not mixed_qa_mode,
            question=rewrite.resolved_query,
            has_conflicts=bool(
                evidence_assessment and evidence_assessment.has_conflicts
            ),
        )
        perf["answer_guard_latency_ms"] = round(
            (time.perf_counter() - t_guard) * 1000,
            2,
        )
        perf["total_latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        logger.info(
            "Answer guarded citation_status=%s grounding=%s "
            "invalid_citations=%s unsupported_claims=%s document_qa=%s "
            "total_ms=%s",
            guarded.citation_validation.status.value,
            guarded.grounding_status.value,
            guarded.citation_validation.invalid,
            guarded.grounding.unsupported_claims,
            document_qa_mode,
            perf["total_latency_ms"],
        )

        return self._finalize_response(
            self._response_formatter.format(
                answer=guarded.answer,
                chunks=chunks,
                prompt_tokens=generation.prompt_tokens,
                completion_tokens=generation.completion_tokens,
                total_tokens=generation.total_tokens,
                retrieval_metadata=retrieval_metadata,
                degraded_sources=retrieval_metadata.degraded_sources,
                evidence_strength=guarded.evidence_strength,
                grounding_status=guarded.grounding_status,
                citation_validation=guarded.citation_validation,
                sources_used=guarded.sources_used,
                used_conversation_context=rewrite.used_conversation_context,
                build_resources=True,
                query_plan=plan.to_metadata(),
                evidence_metadata=evidence_bundle.to_metadata(),
                token_budget=token_budget_meta,
                pipeline_versions=versions.to_dict(),
                performance=perf,
                llm_execution=llm_execution,
                conversation_context=conversation_context,
                language=language_meta,
            )
        )

    @staticmethod
    def _finalize_response(payload: dict[str, Any]) -> dict[str, Any]:
        """Strip LLM source markers from answer text for API display."""
        payload["answer"] = clean_answer_for_display(payload.get("answer", ""))
        return payload
