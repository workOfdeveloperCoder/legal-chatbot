from __future__ import annotations

import inspect
import logging
import re
import time
from dataclasses import replace
from typing import Any

from fastapi import HTTPException, status

from app.core.config import settings
from app.llm.base import BaseLLM
from app.llm.errors import LLMAllProvidersFailed, LLMCancelledError, LLMError
from app.llm.execution_profile import ExecutionProfile
from app.llm.firewall import LLMFirewall
from app.llm.model_capabilities import ModelCapabilities, ModelCapabilityRegistry
from app.llm.versions import current_pipeline_versions

from app.rag.prompt_builder import PromptBuilder
from app.rag.models import (
    Memory,
    Message,
    RetrievalMetadata,
    SourceType,
)

from app.rag.repository import BaseRetriever
from app.rag.query_rewriter import QueryRewriter
from app.services.query_router import Task

from app.rag.evidence_engine import EvidenceEngine
from app.rag.legal_query_planner import (
    AnswerMode,
    LegalQueryPlan,
    LegalQueryPlanner,
    RetrievalStrategy,
)
from app.rag.reasoning_pipeline import GenerationResult, ReasoningPipeline

from app.search.web_search import WebSearchService

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
from app.rag.token_budget import TokenBudgetManager
from app.rag.token_budget.manager import TokenBudgetExceeded
from app.rag.language import detect_language
from app.services.token_usage_service import apply_provider_usage


logger = logging.getLogger(__name__)

DOCUMENT_NOT_READY_DETAIL = (
    "The requested document could not be read or indexed. "
    "Please re-upload or process the document before asking "
    "questions about it."
)

RETRIEVAL_EMBEDDINGS_OFFLINE_NOTE = (
    "**Library search unavailable.** Qdrant `legal_documents` uses 768-d nomic "
    "vectors. Your embedding model must be nomic-compatible (remote Ollama, "
    "Fireworks, or Nomic API). OpenRouter free chat still works.\n\n"
)

RETRIEVAL_EMBEDDING_MODEL_NOTE = (
    "**Library search skipped.** The configured embedding model does not match "
    "the nomic 768-d Qdrant index. Switch to a nomic embed provider for "
    "corpus Resources, or use OpenRouter chat without library hits.\n\n"
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


def _capabilities_from_llm(llm: BaseLLM) -> ModelCapabilities | None:
    getter = getattr(llm, "capabilities", None)
    if getter is None:
        return None
    try:
        from unittest.mock import Mock

        if isinstance(llm, Mock):
            configured = getattr(getter, "return_value", None)
            if isinstance(configured, ModelCapabilities):
                return configured
            return None
        caps = getter() if callable(getter) else getter
    except Exception:  # noqa: BLE001 — mocks / missing adapters
        return None
    if inspect.iscoroutine(caps):
        caps.close()
        return None
    if isinstance(caps, ModelCapabilities):
        return caps
    return None


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
        firewall: LLMFirewall | None = None,
        web_searcher: WebSearchService | None = None,
    ) -> None:

        self._llm = llm
        self._retriever = retriever
        self._query_rewriter = query_rewriter
        self._prompt_builder = prompt_builder
        self._response_formatter = response_formatter
        self._answer_guard = answer_guard or AnswerGuard()
        self._documents = document_repository
        self._firewall = firewall or LLMFirewall()
        self._web_search = web_searcher or WebSearchService()
        caps = _capabilities_from_llm(llm)
        self._profile = ExecutionProfile.for_capabilities(caps)
        self._reasoning = ReasoningPipeline(llm, profile=self._profile)
        self._evidence_engine = evidence_engine or EvidenceEngine()
        self._planner = legal_query_planner or LegalQueryPlanner()
        resolved_caps = caps or ModelCapabilityRegistry.resolve()
        self._token_budget = token_budget_manager or TokenBudgetManager(
            capabilities=caps,
            limits=self._profile.token_limits(resolved_caps),
        )

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
        web_search: bool = False,
        token_callback=None,
        thinking_callback=None,
    ) -> dict[str, Any]:

        t0 = time.perf_counter()
        perf: dict[str, float] = {}
        llm_execution: dict[str, object] = {
            "retry_count": 0,
            "fallback_used": False,
            "error_type": None,
            "compact_prompts": self._profile.compact_prompts,
            "two_stage_enabled": self._profile.two_stage_reasoning,
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
                web_search=web_search,
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
                        AnswerMode.HEARING_PREP,
                        AnswerMode.COMPARE_PROVISIONS,
                    }
                    and not plan.uploaded_document_primary
                )
                plan.document_scoped = bool(document_id) and plan.answer_mode in {
                    AnswerMode.DOCUMENT_QA,
                    AnswerMode.SUMMARIZATION,
                }
        else:
            task_obj = plan.task

        if web_search:
            plan.web_search = True
            if plan.retrieval_strategy == RetrievalStrategy.NONE:
                plan.retrieval_strategy = RetrievalStrategy.LEGAL_FIRST

        mixed_qa_mode = plan.answer_mode == AnswerMode.MIXED_LEGAL_ANALYSIS
        document_qa_mode = plan.answer_mode in {
            AnswerMode.DOCUMENT_QA,
            AnswerMode.SUMMARIZATION,
            AnswerMode.MIXED_LEGAL_ANALYSIS,
        }
        document_task = plan.document_scoped
        needs_legal_authority = plan.requires_legal_authority
        complexity = plan.complexity

        if settings.LLM_FIREWALL_ENABLED:
            has_uploads = bool(document_id) or bool(
                plan.uploaded_document_primary
            )
            firewall_result = self._firewall.screen(
                question,
                history=history,
                matter_id=matter_id,
                document_id=document_id,
                has_uploaded_documents=has_uploads,
                language=language,
            )
            llm_execution["firewall"] = firewall_result.to_metadata()
            if not firewall_result.allowed:
                logger.info(
                    "LLM firewall blocked user=%s decision=%s reason=%s",
                    user_id,
                    firewall_result.decision.value,
                    firewall_result.reason,
                )
                return self._finalize_response(
                    self._response_formatter.format(
                        answer=firewall_result.refusal_message,
                        chunks=[],
                        retrieval_metadata=RetrievalMetadata(),
                        evidence_strength=None,
                        grounding_status=None,
                        citation_validation=None,
                        used_conversation_context=False,
                        query_plan=plan.to_metadata(),
                        pipeline_versions=versions.to_dict(),
                        performance={
                            "total_latency_ms": round(
                                (time.perf_counter() - t0) * 1000, 2
                            ),
                        },
                        llm_execution=llm_execution,
                        language=language_meta,
                    )
                )

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

        prefer_legal_corpus = (
            plan.retrieval_strategy == RetrievalStrategy.LEGAL_FIRST
        )

        t_retrieve = time.perf_counter()
        retrieval_unavailable = False
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
                prefer_legal_corpus=prefer_legal_corpus,
            )
        except Exception:
            logger.exception("Retrieval failed")
            retrieval_unavailable = True
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
            from app.rag.models import RetrievalOutcome

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

        web_requested = bool(web_search)
        should_web = self._should_run_web_search(
            web_search=web_requested,
            document_task=document_task,
        )
        if should_web:
            t_web = time.perf_counter()
            web_chunks = await self._retrieve_web(search_query)
            perf["web_search_latency_ms"] = round(
                (time.perf_counter() - t_web) * 1000,
                2,
            )
            if web_chunks:
                # Keep explicit web hits above evidence / packing floors.
                for chunk in web_chunks:
                    boosted = max(float(chunk.score or 0.0), 0.55)
                    chunk.score = boosted
                    chunk.relevance_score = max(
                        float(chunk.relevance_score or 0.0),
                        boosted,
                    )
                chunks = list(chunks) + web_chunks
                retrieval_metadata.web_chunks = len(web_chunks)
                if "web" not in retrieval_metadata.collections_queried:
                    retrieval_metadata.collections_queried.append("web")
                logger.info(
                    "Web search returned hits=%s query=%r explicit=%s",
                    len(web_chunks),
                    (search_query or "")[:120],
                    web_requested,
                )
            else:
                if "web" not in retrieval_metadata.degraded_sources:
                    retrieval_metadata.degraded_sources.append("web")
                logger.warning(
                    "Web search returned no hits query=%r explicit=%s",
                    (search_query or "")[:120],
                    web_requested,
                )

        if not web_requested:
            chunks = self._response_formatter._without_web_chunks(chunks)
            retrieval_metadata.web_chunks = 0
            retrieval_metadata.collections_queried = [
                name
                for name in retrieval_metadata.collections_queried
                if name != "web"
            ]
            retrieval_metadata.degraded_sources = [
                name
                for name in retrieval_metadata.degraded_sources
                if name != "web"
            ]

        evidence_bundle = self._evidence_engine.build(
            chunks,
            question=rewrite.resolved_query,
            document_task=document_task,
            document_qa_mode=document_qa_mode,
            requires_legal_authority=needs_legal_authority,
            prefer_web=web_requested,
        )
        evidence_assessment = evidence_bundle.assessment
        evidence_strength = evidence_bundle.overall_strength
        chunks = evidence_bundle.chunks
        if (
            web_requested
            and retrieval_metadata.web_chunks > 0
            and evidence_strength == EvidenceStrength.NONE
        ):
            evidence_strength = EvidenceStrength.PARTIAL
            if evidence_assessment is not None:
                evidence_assessment = replace(
                    evidence_assessment,
                    overall_strength=EvidenceStrength.PARTIAL,
                )
                evidence_bundle.assessment = evidence_assessment
                evidence_bundle.overall_strength = EvidenceStrength.PARTIAL

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
            and not chunks
        ):
            # Still call the LLM for in-scope legal questions. Lawyers need a
            # usable answer even when the corpus/uploads miss the point — with
            # an explicit "not document-grounded" disclaimer after generation.
            logger.info(
                "No retrieved evidence; continuing with ungrounded legal "
                "answer path user=%s",
                user_id,
            )

        # Prefer polished LLM synthesis over dumping source text. Only enter
        # document Q&A when private passages are actually about THIS question.
        # Otherwise unrelated conversation uploads (e.g. "VOID ORDERS") hijack
        # conceptual / web-backed answers like "what are sections and articles".
        private_chunks = [
            c
            for c in chunks
            if getattr(c, "source_type", None)
            in {
                SourceType.CONVERSATION.value,
                SourceType.MATTER.value,
            }
        ]
        legal_chunks = [
            c
            for c in chunks
            if getattr(c, "source_type", None) == SourceType.LEGAL.value
        ]
        web_chunks_present = [
            c
            for c in chunks
            if getattr(c, "source_type", None) == SourceType.WEB.value
        ]

        question_about_uploads = self._question_targets_uploaded_docs(
            rewrite.resolved_query or question
        )
        relevant_private = [
            c
            for c in private_chunks
            if self._chunk_matches_question(c, rewrite.resolved_query or question)
        ]

        if private_chunks and not document_task and not document_id:
            # Library hits win for legal-research questions — conversation
            # uploads must not hijack corpus passages (e.g. VOID ORDERS vs Cornelius).
            if (
                legal_chunks
                and needs_legal_authority
                and not question_about_uploads
            ):
                drop_ids = {id(c) for c in private_chunks}
                chunks = [c for c in chunks if id(c) not in drop_ids]
                private_chunks = []
                logger.info(
                    "Dropped %s private chunks; legal corpus preferred user=%s",
                    len(drop_ids),
                    user_id,
                )
            elif not question_about_uploads and not relevant_private:
                drop_ids = {id(c) for c in private_chunks}
                chunks = [c for c in chunks if id(c) not in drop_ids]
                private_chunks = []
                logger.info(
                    "Dropped %s unrelated private chunks for conceptual/"
                    "web question user=%s",
                    len(drop_ids),
                    user_id,
                )
            elif relevant_private and len(relevant_private) < len(private_chunks):
                keep = {id(c) for c in relevant_private}
                drop_n = len(private_chunks) - len(relevant_private)
                chunks = [
                    c
                    for c in chunks
                    if getattr(c, "source_type", None)
                    not in {
                        SourceType.CONVERSATION.value,
                        SourceType.MATTER.value,
                    }
                    or id(c) in keep
                ]
                private_chunks = relevant_private
                logger.info(
                    "Kept %s relevant private chunks, dropped %s unrelated",
                    len(relevant_private),
                    drop_n,
                )

        if private_chunks and not document_task:
            if web_requested and web_chunks_present:
                # Web toggle wins over unrelated-doc summarization.
                mixed_qa_mode = bool(legal_chunks or private_chunks)
                document_qa_mode = False
            elif legal_chunks:
                mixed_qa_mode = True
                document_qa_mode = False
            elif question_about_uploads or relevant_private or document_id:
                document_qa_mode = True
            else:
                document_qa_mode = False

        t_budget = time.perf_counter()
        try:
            packed = self._token_budget.prepare(
                question=question,
                history=history,
                chunks=chunks,
                memories=memories,
                system_prompt=self._profile.system_prompt,
                document_qa_mode=document_qa_mode or document_task,
                duplicate_system_in_user=self._profile.embed_system_in_user_prompt,
                prefer_web=web_requested,
            )
        except TokenBudgetExceeded as exc:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=str(exc),
            ) from exc
        chunks = packed.chunks
        history_for_prompt = packed.history
        question_for_prompt = packed.question if packed.question is not None else question
        token_budget_meta = (
            packed.metadata.to_dict() if packed.metadata else None
        )

        prompt = self._format_user_prompt(
            question=question_for_prompt,
            history=history_for_prompt,
            chunks=chunks,
            memories=memories,
            task_obj=task_obj,
            document_task=document_task,
            document_qa_mode=document_qa_mode,
            mixed_qa_mode=mixed_qa_mode,
            evidence_strength=evidence_strength,
            evidence_assessment=evidence_assessment,
            complexity=complexity,
            packed=packed,
            language=language,
            prefer_web=web_requested,
        )

        # Post-format safety: drop evidence if scaffolding pushed over budget.
        try:
            packed = self._token_budget.ensure_prompt_within_budget(
                prompt,
                system_prompt=self._profile.system_prompt,
                packed=packed,
            )
        except TokenBudgetExceeded as exc:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=str(exc),
            ) from exc
        if packed.chunks is not chunks:
            chunks = packed.chunks
            prompt = self._format_user_prompt(
                question=question_for_prompt,
                history=history_for_prompt,
                chunks=chunks,
                memories=memories,
                task_obj=task_obj,
                document_task=document_task,
                document_qa_mode=document_qa_mode,
                mixed_qa_mode=mixed_qa_mode,
                evidence_strength=evidence_strength,
                evidence_assessment=evidence_assessment,
                complexity=complexity,
                packed=packed,
                language=language,
                prefer_web=web_requested,
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
        # Local R1 often stalls on streaming (thinking-only / proxy drops).
        # For document-grounded answers, use non-stream generate then emit once.
        prefer_nonstream = bool(document_qa_mode or mixed_qa_mode)
        try:
            if token_callback is not None and not prefer_nonstream:
                from app.llm.delta import coerce_delta
                from app.rag.reasoning_cleanup import (
                    join_stream_text,
                    strip_reasoning_output,
                )
                from app.schemas.llm import ChatCompletionRequest, ChatMessage

                request = ChatCompletionRequest(
                    messages=[
                        ChatMessage(
                            role="system",
                            content=self._profile.system_prompt,
                        ),
                        ChatMessage(role="user", content=prompt),
                    ],
                    temperature=settings.LLM_TEMPERATURE,
                    max_tokens=packed.reserved_output_tokens,
                )
                pieces: list[str] = []
                async for piece in self._llm.generate_stream(request):
                    delta = coerce_delta(piece)
                    if not delta.text:
                        continue
                    if delta.kind == "thinking":
                        if thinking_callback is not None:
                            maybe = thinking_callback(delta.text)
                            if inspect.isawaitable(maybe):
                                await maybe
                        continue
                    pieces.append(delta.text)
                    maybe = token_callback(delta.text)
                    if inspect.isawaitable(maybe):
                        await maybe
                generation = GenerationResult(
                    content=strip_reasoning_output(join_stream_text(pieces)),
                    two_stage=False,
                )
                if not (generation.content or "").strip():
                    logger.warning(
                        "Stream produced no visible answer; "
                        "retrying non-stream generate user=%s",
                        user_id,
                    )
                    try:
                        generation = await self._reasoning.generate(
                            user_prompt=prompt,
                            complexity=complexity,
                            temperature=settings.LLM_TEMPERATURE,
                            max_tokens=max(
                                packed.reserved_output_tokens,
                                12288,
                            ),
                        )
                    except (LLMAllProvidersFailed, LLMError) as exc:
                        logger.warning(
                            "Non-stream retry after empty stream failed: %s",
                            exc,
                        )
                        generation = GenerationResult(
                            content="",
                            two_stage=False,
                        )
            else:
                generation = await self._reasoning.generate(
                    user_prompt=prompt,
                    complexity=complexity,
                    temperature=settings.LLM_TEMPERATURE,
                    max_tokens=max(
                        packed.reserved_output_tokens,
                        12288 if prefer_nonstream else packed.reserved_output_tokens,
                    ),
                )
                if (
                    token_callback is not None
                    and (generation.content or "").strip()
                ):
                    maybe = token_callback(generation.content)
                    if inspect.isawaitable(maybe):
                        await maybe
            if not (generation.content or "").strip():
                logger.warning(
                    "LLM returned blank content after retries user=%s "
                    "chunks=%s",
                    user_id,
                    len(chunks),
                )
                if chunks:
                    generation = GenerationResult(
                        content=self._blank_llm_chunk_fallback(
                            question=rewrite.resolved_query or question,
                            chunks=chunks,
                        ),
                        two_stage=False,
                    )
                else:
                    generation = GenerationResult(
                        content=self._answer_guard.build_insufficient_evidence_answer(
                            evidence_strength=evidence_strength,
                            has_private_sources=False,
                            language=language,
                        ),
                        two_stage=False,
                    )
        except LLMCancelledError:
            raise HTTPException(
                status_code=499,
                detail="The request was cancelled.",
            )
        except (LLMAllProvidersFailed, LLMError) as exc:
            logger.exception("LLM generation failed: %s", exc)
            llm_execution["error_type"] = type(exc).__name__
            detail = str(exc).strip()
            # Prefer retrieved passages over a hard 503 when the model only
            # spent tokens on hidden reasoning / timed out empty.
            if chunks and (
                "empty" in detail.lower()
                or "reasoning" in detail.lower()
                or "timeout" in detail.lower()
            ):
                logger.warning(
                    "Using passage fallback after LLM failure user=%s",
                    user_id,
                )
                generation = GenerationResult(
                    content=self._blank_llm_chunk_fallback(
                        question=rewrite.resolved_query or question,
                        chunks=chunks,
                    ),
                    two_stage=False,
                )
            else:
                if "empty" in detail.lower() or "timeout" in detail.lower():
                    user_detail = (
                        "The local language model did not finish in time "
                        "(or returned only hidden reasoning). "
                        "Wait a moment and try again — avoid running two "
                        "chats against Ollama at once."
                    )
                elif "404" in detail or "bad request" in detail.lower():
                    user_detail = (
                        "The configured chat model is not available. "
                        "Set OPENROUTER_MODEL or CHAT_MODEL to a model id "
                        "your OpenRouter account can use."
                    )
                elif "429" in detail or "rate" in detail.lower():
                    user_detail = (
                        "The chat provider rate-limited this request. "
                        "Wait a few seconds and try again."
                    )
                else:
                    user_detail = (
                        "The assistant is temporarily unavailable. "
                        "Please try again shortly."
                    )
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=user_detail,
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

        if settings.LLM_FIREWALL_ENABLED:
            output_guard = self._firewall.screen_output(
                generation.content,
                language=language,
            )
            if not output_guard.allowed:
                logger.warning(
                    "LLM firewall blocked generated output user=%s reason=%s",
                    user_id,
                    output_guard.reason,
                )
                llm_execution["firewall"] = output_guard.to_metadata()
                generation.content = output_guard.refusal_message

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
        if token_budget_meta is not None:
            token_budget_meta = apply_provider_usage(
                token_budget_meta,
                prompt_tokens=generation.prompt_tokens,
                completion_tokens=generation.completion_tokens,
                total_tokens=generation.total_tokens,
                capabilities=self._token_budget.capabilities,
            )

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
            include_web=web_requested,
        )

        guarded = self._answer_guard.process(
            answer=preliminary["answer"],
            chunks=chunks,
            sources=preliminary["sources"],
            evidence_strength=evidence_strength,
            skip_grounding=document_task
            or (
                evidence_strength == EvidenceStrength.NONE and not chunks
            ),
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

        final_answer = guarded.answer
        if (
            "qdrant_embedding_model" in retrieval_metadata.degraded_sources
            and not chunks
            and not web_requested
        ):
            final_answer = RETRIEVAL_EMBEDDING_MODEL_NOTE + final_answer
        elif retrieval_unavailable and not web_requested and not chunks:
            final_answer = RETRIEVAL_EMBEDDINGS_OFFLINE_NOTE + final_answer

        return self._finalize_response(
            self._response_formatter.format(
                answer=final_answer,
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
                include_web=web_requested,
            )
        )

    @staticmethod
    def _blank_llm_chunk_fallback(*, question: str, chunks: list) -> str:
        """Last-resort when the model returns only hidden reasoning."""
        titles: list[str] = []
        for chunk in chunks[:4]:
            title = (
                getattr(chunk, "filename", None)
                or getattr(chunk, "law_name", None)
                or getattr(chunk, "title", None)
            )
            if title and str(title) not in titles:
                titles.append(str(title))
        title_bit = (
            ", ".join(titles)
            if titles
            else "the retrieved resources listed below"
        )
        return (
            f"I retrieved relevant material for **{question.strip()}** "
            f"(including {title_bit}), but the local model did not return a "
            "readable synthesized answer this time.\n\n"
            "Please open the Resources below for the source passages, or "
            "try the question again. Prefer a specific prompt such as "
            "`section 54-C Electricity Act 1910` if the answer stays empty."
        )

    def _should_run_web_search(
        self,
        *,
        web_search: bool,
        document_task: bool = False,
    ) -> bool:
        """Internet search runs only when the client Web toggle is on."""
        if not settings.WEB_SEARCH_ENABLED:
            return False
        if document_task and not web_search:
            return False
        return bool(web_search)

    async def _retrieve_web(self, query: str) -> list:
        try:
            return await self._web_search.search_as_chunks(
                query,
                limit=settings.RETRIEVAL_WEB_LIMIT,
            )
        except Exception:
            logger.exception("Web search failed")
            return []

    @staticmethod
    def _question_targets_uploaded_docs(question: str) -> bool:
        lower = (question or "").lower()
        return bool(
            re.search(
                r"\b(?:this|the|my|uploaded|attached)\s+"
                r"(?:document|doc|file|article|paper|pdf|upload)\b|"
                r"\b(?:summarize|summary of|key (?:points|issues) in|"
                r"according to (?:the )?(?:author|article|document)|"
                r"what does (?:the )?(?:author|article|document))\b|"
                r"\bvoid orders\b",
                lower,
            )
        )

    @staticmethod
    def _chunk_matches_question(chunk, question: str) -> bool:
        """Cheap lexical relevance so unrelated uploads don't dominate."""
        text = " ".join(
            part
            for part in (
                getattr(chunk, "text", None) or "",
                getattr(chunk, "filename", None) or "",
                getattr(chunk, "title", None) or "",
                getattr(chunk, "display_name", None) or "",
            )
            if part
        ).lower()
        if not text:
            return False

        score = float(
            getattr(chunk, "relevance_score", None)
            or getattr(chunk, "score", None)
            or 0.0
        )

        terms = [
            t
            for t in re.findall(r"[a-z0-9\-]{3,}", (question or "").lower())
            if t
            not in {
                "the",
                "and",
                "for",
                "what",
                "with",
                "from",
                "this",
                "that",
                "are",
                "is",
                "law",
                "legal",
                "please",
                "about",
                "under",
                "pakistan",
                "pakistani",
                "does",
                "how",
                "why",
                "can",
                "any",
                "into",
                "who",
                "when",
                "where",
                "which",
                "have",
                "has",
                "was",
                "were",
                "been",
                "being",
                "will",
                "would",
                "should",
                "could",
                "their",
                "they",
                "them",
                "your",
                "you",
                "his",
                "her",
                "its",
                "our",
            }
        ]
        if not terms:
            return score >= 0.70

        generic_legal = {
            "justice",
            "government",
            "independent",
            "establish",
            "established",
            "primary",
            "task",
            "newly",
            "identify",
            "court",
            "order",
            "orders",
            "section",
            "sections",
            "article",
            "articles",
            "act",
            "statute",
            "provision",
            "provisions",
            "judgment",
            "judgement",
            "case",
            "lawyer",
            "advocate",
            "petition",
            "appeal",
            "bail",
            "fir",
            "crpc",
            "cpc",
            "ppc",
        }
        discriminative = [t for t in terms if t not in generic_legal]
        if discriminative:
            hits = sum(1 for term in discriminative if term in text)
            required = max(1, min(2, (len(discriminative) + 1) // 2))
            return hits >= required

        hits = sum(1 for term in terms if term in text)
        return hits >= max(2, (len(terms) + 2) // 3)

    def _format_user_prompt(
        self,
        *,
        question: str,
        history,
        chunks,
        memories,
        task_obj,
        document_task: bool,
        document_qa_mode: bool,
        mixed_qa_mode: bool,
        evidence_strength,
        evidence_assessment,
        complexity,
        packed,
        language,
        prefer_web: bool = False,
    ) -> str:
        return self._prompt_builder.build(
            question=question,
            history=history,
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
            include_system_in_user=self._profile.embed_system_in_user_prompt,
            compact=self._profile.compact_prompts,
            prefer_web=prefer_web,
        )

    @staticmethod
    def _finalize_response(payload: dict[str, Any]) -> dict[str, Any]:
        """Strip LLM source markers from answer text for API display."""
        raw = payload.get("answer", "") or ""
        cleaned = clean_answer_for_display(raw)
        if not cleaned.strip():
            # Never ship an empty bubble (e.g. answer was only [Source N] markers).
            cleaned = (
                "No visible answer text remained after cleanup. "
                "Check Resources below for retrieved passages, or try again "
                "with a more specific question "
                "(for example: `section 54-C Electricity Act 1910`)."
                if raw.strip()
                else (
                    "No visible answer was returned. Check Resources below "
                    "if any passages were retrieved, or try again."
                )
            )
        payload["answer"] = cleaned
        return payload
