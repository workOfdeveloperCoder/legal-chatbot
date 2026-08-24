"""One real Gemini generation test. Does not change process LLM_PROVIDER."""

from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path

from app.core.config import settings
from app.embeddings.factory import EmbeddingFactory
from app.embeddings.ollama import OllamaEmbedding
from app.llm.errors import LLMPermanentError, LLMTransientError
from app.llm.execution_profile import ExecutionProfile
from app.llm.factory import LLMFactory
from app.llm.model_capabilities import ModelCapabilityRegistry, estimate_cost_usd
from app.llm.openai_compatible import OpenAICompatibleLLM
from app.llm.provider_config import resolve_chat_model, resolve_llm_base_url
from app.rag.models import RetrievedChunk, SourceType
from app.rag.prompt_builder import PromptBuilder
from app.rag.query_complexity import QueryComplexity
from app.rag.token_budget.manager import TokenBudgetManager
from app.schemas.llm import ChatCompletionRequest, ChatMessage

QUESTION = (
    "Under Section 54-C of the Pakistan Income Tax Ordinance, what is the "
    "basic tax treatment provided by the section? Answer briefly and identify "
    "the relevant legal basis from the supplied evidence."
)

# Same synthetic evidence used by the previous online LLM benchmark
# (Specific Relief Act 54-C). This is packed into the prompt; Qdrant is not used.
STATUTE = (
    "Section 54-C of the Specific Relief Act, 1877, as applicable in Pakistan, "
    "restricts the grant of injunctions in certain consumer and contractual "
    "disputes. Courts have treated it as a limitation on judicial discretion "
    "where the statute's conditions are met. The provision does not abolish "
    "injunctions altogether; it channels when an injunction may be refused. "
) * 8

REPORT_PATH = Path("tmp_test/gemini_llm_report.json")
PROVIDER = "gemini"
RECOMMENDED_MODEL = "gemini-3.6-flash"
EXPECTED_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/openai"


def _resolve_api_key() -> tuple[bool, str | None, str]:
    """Return (configured, key, source_label). Never log the key."""
    canonical = settings.api_key_for_provider("gemini")
    if canonical:
        return True, canonical, "LLM_API_KEY"
    return False, None, "none"


def _redact(text: str | None, secret: str | None) -> str | None:
    if text is None:
        return None
    redacted = text
    if secret:
        redacted = redacted.replace(secret, "[REDACTED]")
    redacted = re.sub(r"(AIza[0-9A-Za-z_-]{10,})", "[REDACTED]", redacted)
    redacted = re.sub(r"(sk-[A-Za-z0-9_-]{8,})", "[REDACTED]", redacted)
    redacted = re.sub(
        r"(Bearer\s+)[A-Za-z0-9._\-]+",
        r"\1[REDACTED]",
        redacted,
        flags=re.IGNORECASE,
    )
    return redacted


def _chunk(i: int, source: str) -> RetrievedChunk:
    return RetrievedChunk(
        id=f"c{i}",
        chunk_id=f"c{i}",
        score=0.9 - (i * 0.02),
        relevance_score=0.88 - (i * 0.02),
        text=f"[Chunk {i}] {STATUTE}",
        source_type=source,
        title=f"Authority {i}",
        law_name="Specific Relief Act, 1877",
        document_id=f"doc-{i}",
    )


def _build_user_prompt(profile: ExecutionProfile, caps) -> str:
    chunks = [
        _chunk(i, SourceType.LEGAL.value if i < 7 else SourceType.MATTER.value)
        for i in range(1, 9)
    ]
    mgr = TokenBudgetManager(
        capabilities=caps,
        limits=profile.token_limits(caps),
    )
    packed = mgr.prepare(
        question=QUESTION,
        history=[],
        chunks=chunks,
        memories=[],
        system_prompt=profile.system_prompt,
        duplicate_system_in_user=profile.embed_system_in_user_prompt,
    )
    return PromptBuilder().build(
        question=QUESTION,
        history=packed.history,
        chunks=packed.chunks,
        memories=[],
        complexity=QueryComplexity.RESEARCH,
        include_system_in_user=profile.embed_system_in_user_prompt,
        compact=profile.compact_prompts,
    )


def _empty_report(**overrides) -> dict:
    base = {
        "test_type": "direct_generation_with_packed_benchmark_evidence",
        "full_rag": False,
        "two_stage": False,
        "provider": PROVIDER,
        "model": RECOMMENDED_MODEL,
        "adapter": "OpenAICompatibleLLM",
        "endpoint": EXPECTED_ENDPOINT,
        "adapter_url_ok": None,
        "adapter_model_ok": None,
        "api_configured": False,
        "endpoint_reachable": None,
        "live_generation": "BLOCKED",
        "verdict": "BLOCKED BY MISSING API KEY",
        "http_status": None,
        "latency_ms": None,
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
        "finish_reason": None,
        "estimated_cost_usd": None,
        "answer": None,
        "error": None,
        "embeddings_unchanged": True,
        "embedding_provider": settings.EMBEDDING_PROVIDER,
        "embedding_url": settings.embedding_base_url,
        "process_llm_provider_unchanged": settings.LLM_PROVIDER,
        "question": QUESTION,
    }
    base.update(overrides)
    return base


def _print_report(report: dict) -> None:
    print("============================================================")
    print("GEMINI REAL LLM TEST")
    print("============================================================")
    print()
    print(f"Provider: {report['provider']}")
    print(f"Model: {report['model']}")
    print(f"API configured: {'yes' if report['api_configured'] else 'no'}")
    print(f"Adapter: {report['adapter']}")
    reachable = report["endpoint_reachable"]
    if reachable is True:
        reachable_s = "yes"
    elif reachable is False:
        reachable_s = "no"
    else:
        reachable_s = "not probed"
    print(f"Endpoint reachable: {reachable_s}")
    print(f"Live generation: {report['live_generation']}")
    latency = report["latency_ms"]
    print(f"Latency: {latency if latency is None else f'{latency} ms'}")
    print(f"Input tokens: {report['input_tokens']}")
    print(f"Output tokens: {report['output_tokens']}")
    print(f"Total tokens: {report['total_tokens']}")
    print(f"Finish reason: {report['finish_reason']}")
    print()
    print("Answer:")
    print(report["answer"] or report.get("error") or "(none)")
    print()
    print("============================================================")
    print("VERDICT")
    print("============================================================")
    print()
    print(report["verdict"])
    if report.get("http_status") is not None:
        print(f"HTTP status: {report['http_status']}")
    if report.get("estimated_cost_usd") is not None:
        print(f"Estimated cost: ${report['estimated_cost_usd']:.8f}")
    print()
    print(
        "Test type: direct generation with packed benchmark evidence "
        "(not a full RAG + Qdrant retrieval test)."
    )
    print(f"Process LLM_PROVIDER left as: {report['process_llm_provider_unchanged']}")
    print(
        "Embeddings unchanged: "
        f"{report['embedding_provider']} @ {report['embedding_url']}"
    )


async def run() -> dict:
    configured, api_key, _source = _resolve_api_key()
    resolved_url = resolve_llm_base_url(PROVIDER, settings.LLM_URL)
    resolved_model = resolve_chat_model(PROVIDER, settings.CHAT_MODEL)
    if resolved_model != RECOMMENDED_MODEL and ":" in (settings.CHAT_MODEL or ""):
        resolved_model = resolve_chat_model(PROVIDER, settings.CHAT_MODEL)

    llm_probe = LLMFactory.create_raw(
        provider=PROVIDER,
        base_url=settings.LLM_URL,
        model=RECOMMENDED_MODEL,
        api_key=api_key or "missing-key-not-used",
    )
    adapter_ok = isinstance(llm_probe, OpenAICompatibleLLM)
    actual_url = str(llm_probe.client.base_url).rstrip("/")
    url_ok = actual_url == EXPECTED_ENDPOINT
    model_ok = llm_probe.model_name == RECOMMENDED_MODEL
    await llm_probe.aclose()

    embedding = EmbeddingFactory.create()
    embeddings_local = isinstance(embedding, OllamaEmbedding) and (
        str(embedding._client.base_url).rstrip("/") == "http://localhost:11434"
    )

    report = _empty_report(
        model=RECOMMENDED_MODEL,
        endpoint=actual_url,
        adapter="OpenAICompatibleLLM" if adapter_ok else type(llm_probe).__name__,
        adapter_url_ok=url_ok,
        adapter_model_ok=model_ok,
        resolved_url=resolved_url,
        resolved_model_from_local_tag=resolve_chat_model(PROVIDER, settings.CHAT_MODEL),
        recommended_model=RECOMMENDED_MODEL,
        api_configured=configured,
        embeddings_unchanged=embeddings_local,
        two_stage=False,
    )

    if not configured:
        report["live_generation"] = "BLOCKED"
        report["verdict"] = "BLOCKED BY MISSING API KEY"
        report["error"] = (
            "No LLM_API_KEY (canonical) or Gemini alias "
            "(GOOGLE_API_KEY / GEMINI_API_KEY). Did not send a Gemini request."
        )
        report["answer"] = report["error"]
        return report

    caps = ModelCapabilityRegistry.resolve(
        provider=PROVIDER,
        model_name=RECOMMENDED_MODEL,
    )
    profile = ExecutionProfile.online(caps)
    if profile.two_stage_reasoning:
        report["live_generation"] = "FAILED"
        report["verdict"] = "FAILED API REQUEST"
        report["error"] = "Online profile unexpectedly enabled two-stage reasoning."
        report["two_stage"] = True
        return report

    user_prompt = _build_user_prompt(profile, caps)
    request = ChatCompletionRequest(
        messages=[
            ChatMessage(role="system", content=profile.system_prompt),
            ChatMessage(role="user", content=user_prompt),
        ],
        temperature=0.1,
        max_tokens=min(512, profile.reserved_output_tokens),
    )

    llm = LLMFactory.create_raw(
        provider=PROVIDER,
        base_url=settings.LLM_URL,
        model=RECOMMENDED_MODEL,
        api_key=api_key,
    )
    captured: dict[str, object] = {}
    original_post = llm.client.post

    async def capturing_post(*args, **kwargs):
        response = await original_post(*args, **kwargs)
        captured["http_status"] = response.status_code
        try:
            payload = response.json()
            choices = payload.get("choices") or [{}]
            captured["finish_reason"] = choices[0].get("finish_reason")
        except Exception:  # noqa: BLE001
            captured["finish_reason"] = None
        return response

    llm.client.post = capturing_post  # type: ignore[method-assign]
    t0 = time.perf_counter()
    try:
        response = await llm.generate(request)
    except (LLMPermanentError, LLMTransientError) as exc:
        latency = round((time.perf_counter() - t0) * 1000, 1)
        report["latency_ms"] = latency
        report["http_status"] = captured.get("http_status")
        report["endpoint_reachable"] = captured.get("http_status") is not None
        report["live_generation"] = "FAILED"
        report["verdict"] = "FAILED API REQUEST"
        report["error"] = _redact(f"{type(exc).__name__}: {exc}", api_key)
        report["answer"] = report["error"]
        await llm.aclose()
        return report
    except Exception as exc:  # noqa: BLE001
        latency = round((time.perf_counter() - t0) * 1000, 1)
        report["latency_ms"] = latency
        report["http_status"] = captured.get("http_status")
        report["endpoint_reachable"] = captured.get("http_status") is not None
        report["live_generation"] = "FAILED"
        report["verdict"] = "FAILED API REQUEST"
        report["error"] = _redact(f"{type(exc).__name__}: {exc}", api_key)
        report["answer"] = report["error"]
        await llm.aclose()
        return report
    finally:
        try:
            await llm.aclose()
        except Exception:  # noqa: BLE001
            pass

    latency = round((time.perf_counter() - t0) * 1000, 1)
    cost = estimate_cost_usd(
        capabilities=caps,
        input_tokens=response.prompt_tokens or 0,
        output_tokens=response.completion_tokens or 0,
    )
    report.update(
        {
            "api_configured": True,
            "endpoint_reachable": True,
            "live_generation": "SUCCESS",
            "verdict": "REAL LIVE GEMINI GENERATION",
            "http_status": captured.get("http_status") or 200,
            "latency_ms": latency,
            "input_tokens": response.prompt_tokens,
            "output_tokens": response.completion_tokens,
            "total_tokens": response.total_tokens,
            "finish_reason": captured.get("finish_reason"),
            "estimated_cost_usd": cost,
            "answer": _redact(response.content, api_key),
            "two_stage": False,
        }
    )
    return report


def main() -> int:
    report = asyncio.run(run())
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))
    # Guard: never persist a key-shaped string
    dumped = REPORT_PATH.read_text()
    for candidate in (
        settings.LLM_API_KEY,
        settings.GOOGLE_API_KEY,
        settings.GEMINI_API_KEY,
    ):
        if candidate and candidate.strip() and candidate in dumped:
            raise RuntimeError("Refusing to write report containing API key")
    _print_report(report)
    print(f"JSON: {REPORT_PATH}")
    if report["verdict"] == "BLOCKED BY MISSING API KEY":
        return 2
    if report["verdict"] == "REAL LIVE GEMINI GENERATION":
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
