"""One-shot online LLM verification. Prints a report; does not persist secrets."""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass

import httpx

from app.core.config import settings
from app.llm.anthropic import AnthropicLLM
from app.llm.errors import LLMPermanentError, LLMTransientError
from app.llm.execution_profile import ExecutionProfile
from app.llm.factory import LLMFactory
from app.llm.model_capabilities import ModelCapabilityRegistry
from app.llm.openai_compatible import OpenAICompatibleLLM
from app.llm.provider_config import (
    is_online_provider,
    resolve_chat_model,
    resolve_llm_base_url,
)
from app.llm.registry import LLMAdapterRegistry
from app.llm.token_counter import build_token_counter
from app.rag.models import RetrievedChunk, SourceType
from app.rag.prompt_builder import PromptBuilder
from app.rag.prompts import LEGAL_SYSTEM_PROMPT, LEGAL_SYSTEM_PROMPT_COMPACT
from app.rag.query_complexity import QueryComplexity
from app.rag.token_budget.manager import TokenBudgetManager
from app.schemas.llm import ChatCompletionRequest, ChatMessage
from app.services.token_usage_service import TokenUsageService


QUESTION = (
    "What is the effect of section 54-C of the Specific Relief Act on "
    "the grant of an injunction in a consumer matter?"
)

STATUTE = (
    "Section 54-C of the Specific Relief Act, 1877, as applicable in Pakistan, "
    "restricts the grant of injunctions in certain consumer and contractual "
    "disputes. Courts have treated it as a limitation on judicial discretion "
    "where the statute's conditions are met. The provision does not abolish "
    "injunctions altogether; it channels when an injunction may be refused. "
    * 8
)


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


def _mask(value: str | None) -> str:
    if not value:
        return "<unset>"
    if len(value) <= 8:
        return f"set(len={len(value)})"
    return f"{value[:4]}…{value[-4:]} (len={len(value)})"


@dataclass
class Probe:
    name: str
    ok: bool
    detail: str
    latency_ms: float | None = None


def config_section() -> dict:
    return {
        "process_LLM_PROVIDER": settings.LLM_PROVIDER,
        "process_CHAT_MODEL": settings.CHAT_MODEL,
        "process_online": settings.is_online_llm,
        "env_LLM_PROVIDER": os.environ.get("LLM_PROVIDER") or "<unset>",
        "LLM_API_KEY": _mask(settings.LLM_API_KEY),
        "OPENAI_API_KEY": _mask(settings.OPENAI_API_KEY),
        "ANTHROPIC_API_KEY": _mask(settings.ANTHROPIC_API_KEY),
        "GOOGLE_API_KEY": _mask(settings.GOOGLE_API_KEY),
        "registered_adapters": LLMFactory.registered_providers(),
    }


def adapter_wiring() -> list[dict]:
    rows = []
    cases = [
        ("openai", "deepseek-r1:32b", OpenAICompatibleLLM, "https://api.openai.com/v1"),
        ("deepseek", "deepseek-r1:32b", OpenAICompatibleLLM, "https://api.deepseek.com/v1"),
        ("anthropic", "deepseek-r1:32b", AnthropicLLM, "https://api.anthropic.com/v1"),
        ("gemini", "gemini-2.0-flash", OpenAICompatibleLLM, None),
    ]
    for provider, model, cls, expected_url in cases:
        llm = LLMFactory.create_raw(
            provider=provider,
            base_url="http://localhost:11434",
            model=model,
            api_key="sk-test-not-real",
        )
        url = str(llm.client.base_url).rstrip("/")
        rows.append(
            {
                "provider": provider,
                "requested_model": model,
                "resolved_model": llm.model_name,
                "adapter": type(llm).__name__,
                "adapter_ok": isinstance(llm, cls),
                "base_url": url,
                "url_ok": (expected_url is None) or (url == expected_url),
                "online": is_online_provider(provider),
                "mapped_model": resolve_chat_model(provider, model),
                "hosted_url": resolve_llm_base_url(provider, "http://localhost:11434"),
            }
        )
    return rows


def token_savings() -> dict:
    local_caps = ModelCapabilityRegistry.resolve(
        provider="ollama", model_name="deepseek-r1:32b"
    )
    online_caps = ModelCapabilityRegistry.resolve(
        provider="openai", model_name="gpt-4o-mini"
    )
    local = ExecutionProfile.local(local_caps)
    online = ExecutionProfile.online(online_caps)
    counter = build_token_counter(online_caps.tokenizer_id)

    chunks = [
        _chunk(1, SourceType.LEGAL.value),
        _chunk(2, SourceType.LEGAL.value),
        _chunk(3, SourceType.LEGAL.value),
        _chunk(4, SourceType.LEGAL.value),
        _chunk(5, SourceType.MATTER.value),
        _chunk(6, SourceType.MATTER.value),
        _chunk(7, SourceType.LEGAL.value),
        _chunk(8, SourceType.LEGAL.value),
    ]

    def pack_and_build(profile, caps):
        mgr = TokenBudgetManager(
            capabilities=caps,
            limits=profile.token_limits(caps),
            counter=counter,
        )
        packed = mgr.prepare(
            question=QUESTION,
            history=[],
            chunks=chunks,
            memories=[],
            system_prompt=profile.system_prompt,
            duplicate_system_in_user=profile.embed_system_in_user_prompt,
        )
        prompt = PromptBuilder().build(
            question=QUESTION,
            history=packed.history,
            chunks=packed.chunks,
            memories=[],
            complexity=QueryComplexity.RESEARCH,
            include_system_in_user=profile.embed_system_in_user_prompt,
            compact=profile.compact_prompts,
        )
        sys_tokens = counter.count(profile.system_prompt)
        user_tokens = counter.count(prompt)
        billed_input = sys_tokens + user_tokens
        if profile.two_stage_reasoning:
            billed_input *= 2
        return {
            "chunks_kept": len(packed.chunks),
            "chunks_dropped": packed.metadata.chunks_dropped if packed.metadata else 0,
            "reserved_output": packed.reserved_output_tokens,
            "system_tokens": sys_tokens,
            "user_prompt_tokens": user_tokens,
            "estimated_billed_input": billed_input,
            "two_stage": profile.two_stage_reasoning,
            "compact": profile.compact_prompts,
            "embed_system_in_user": profile.embed_system_in_user_prompt,
            "legal_cap": profile.max_legal_evidence_tokens,
            "tokenizer": counter.tokenizer_id,
            "tokenizer_exact": counter.is_exact,
        }

    local_stats = pack_and_build(local, local_caps)
    online_stats = pack_and_build(online, online_caps)
    saved = (
        local_stats["estimated_billed_input"] - online_stats["estimated_billed_input"]
    )
    pct = round(100.0 * saved / max(local_stats["estimated_billed_input"], 1), 1)
    return {
        "system_prompt_words": {
            "local": len(LEGAL_SYSTEM_PROMPT.split()),
            "online": len(LEGAL_SYSTEM_PROMPT_COMPACT.split()),
        },
        "local": local_stats,
        "online": online_stats,
        "input_tokens_saved": saved,
        "input_tokens_saved_percent": pct,
    }


def status_for_online() -> dict:
    class Stub:
        provider_name = "openai"
        model_name = "gpt-4o-mini"

        def capabilities(self):
            return ModelCapabilityRegistry.resolve(
                provider="openai", model_name="gpt-4o-mini"
            )

    return TokenUsageService(Stub()).status().model_dump()


async def live_generate() -> Probe:
    key = settings.api_key_for_provider("openai") or settings.resolved_llm_api_key
    if not key:
        return Probe(
            "openai_generate",
            False,
            "No OPENAI_API_KEY / LLM_API_KEY in .env or process environment. "
            "Live generation was not sent.",
        )
    llm = LLMFactory.create_raw(
        provider="openai",
        model="gpt-4o-mini",
        api_key=key,
    )
    request = ChatCompletionRequest(
        messages=[
            ChatMessage(role="system", content=LEGAL_SYSTEM_PROMPT_COMPACT),
            ChatMessage(
                role="user",
                content=(
                    "In one short paragraph, what is section 54-C of the "
                    "Specific Relief Act in Pakistani legal practice? "
                    "If you lack retrieved evidence, say so."
                ),
            ),
        ],
        temperature=0.1,
        max_tokens=180,
    )
    t0 = time.perf_counter()
    try:
        response = await llm.generate(request)
    except (LLMPermanentError, LLMTransientError) as exc:
        return Probe(
            "openai_generate",
            False,
            f"{type(exc).__name__}: {exc}",
            round((time.perf_counter() - t0) * 1000, 1),
        )
    finally:
        await llm.aclose()
    ms = round((time.perf_counter() - t0) * 1000, 1)
    preview = (response.content or "").replace("\n", " ")[:240]
    return Probe(
        "openai_generate",
        True,
        (
            f"model={llm.model_name} prompt={response.prompt_tokens} "
            f"completion={response.completion_tokens} total={response.total_tokens} "
            f"preview={preview!r}"
        ),
        ms,
    )


async def http_probes() -> list[Probe]:
    targets = [
        ("openai", "https://api.openai.com/v1/models", {"Authorization": "Bearer invalid"}),
        ("anthropic", "https://api.anthropic.com/v1/messages", {
            "x-api-key": "invalid",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }),
        ("deepseek", "https://api.deepseek.com/v1/models", {"Authorization": "Bearer invalid"}),
    ]
    out: list[Probe] = []
    async with httpx.AsyncClient(timeout=12.0) as client:
        for name, url, headers in targets:
            t0 = time.perf_counter()
            try:
                if name == "anthropic":
                    response = await client.post(
                        url,
                        headers=headers,
                        json={"model": "claude-sonnet-4-5", "max_tokens": 8, "messages": []},
                    )
                else:
                    response = await client.get(url, headers=headers)
                ms = round((time.perf_counter() - t0) * 1000, 1)
                reachable = response.status_code in {400, 401, 403, 404, 422}
                out.append(
                    Probe(
                        f"{name}_http",
                        reachable,
                        f"HTTP {response.status_code} (host reachable)",
                        ms,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                out.append(
                    Probe(
                        f"{name}_http",
                        False,
                        f"{type(exc).__name__}: {exc}",
                        round((time.perf_counter() - t0) * 1000, 1),
                    )
                )
    return out


def print_report(
    *,
    config: dict,
    wiring: list[dict],
    tokens: dict,
    status: dict,
    probes: list[Probe],
    generate: Probe,
    pytest_line: str,
) -> None:
    print("=" * 72)
    print("ONLINE LLM TEST REPORT")
    print("=" * 72)
    print("\n1) Current process config")
    for k, v in config.items():
        print(f"   {k}: {v}")

    print("\n2) Adapter wiring (local LLM_URL remapped to hosted APIs)")
    for row in wiring:
        mark = "PASS" if row["adapter_ok"] and row["url_ok"] else "FAIL"
        print(
            f"   [{mark}] {row['provider']}: {row['requested_model']} -> "
            f"{row['resolved_model']} via {row['adapter']} @ {row['base_url']}"
        )

    print("\n3) Token efficiency (same legal question + 8 evidence chunks)")
    print(
        f"   system prompt words: local={tokens['system_prompt_words']['local']} "
        f"online={tokens['system_prompt_words']['online']}"
    )
    for label in ("local", "online"):
        s = tokens[label]
        print(
            f"   {label}: billed_input≈{s['estimated_billed_input']} "
            f"system={s['system_tokens']} user={s['user_prompt_tokens']} "
            f"chunks_kept={s['chunks_kept']} reserved_out={s['reserved_output']} "
            f"two_stage={s['two_stage']} tokenizer={s['tokenizer']}"
            f"{'' if s['tokenizer_exact'] else ' (estimate)'}"
        )
    print(
        f"   saved ≈ {tokens['input_tokens_saved']} input tokens "
        f"({tokens['input_tokens_saved_percent']}%) vs local 32B path"
    )

    print("\n4) GET-equivalent /llm/status for gpt-4o-mini")
    print(
        f"   online={status['online']} model={status['model']} "
        f"window={status['context_window']} available_input="
        f"{status['available_input_tokens']} reserved_out="
        f"{status['reserved_output_tokens']}"
    )

    print("\n5) Hosted API reachability (invalid key on purpose)")
    for probe in probes:
        mark = "PASS" if probe.ok else "FAIL"
        print(f"   [{mark}] {probe.name}: {probe.detail} ({probe.latency_ms} ms)")

    print("\n6) Live generation")
    mark = "PASS" if generate.ok else "BLOCKED" if "No OPENAI" in generate.detail else "FAIL"
    print(f"   [{mark}] {generate.detail}")
    if generate.latency_ms is not None:
        print(f"   latency: {generate.latency_ms} ms")

    print("\n7) Unit tests")
    print(f"   {pytest_line}")
    print("=" * 72)


async def main() -> int:
    config = config_section()
    wiring = adapter_wiring()
    tokens = token_savings()
    status = status_for_online()
    probes = await http_probes()
    generate = await live_generate()
    print_report(
        config=config,
        wiring=wiring,
        tokens=tokens,
        status=status,
        probes=probes,
        generate=generate,
        pytest_line="see pytest output",
    )
    # Also dump JSON for the assistant to quote accurately
    payload = {
        "config": config,
        "wiring": wiring,
        "tokens": tokens,
        "status": {
            k: status[k]
            for k in (
                "online",
                "provider",
                "model",
                "context_window",
                "available_input_tokens",
                "reserved_output_tokens",
                "safety_margin_tokens",
                "tokenizer_id",
                "tokenizer_exact",
            )
        },
        "probes": [probe.__dict__ for probe in probes],
        "generate": generate.__dict__,
        "registry_online": {
            name: LLMAdapterRegistry.is_online(name)
            for name in LLMAdapterRegistry.registered_names()
        },
    }
    Path = __import__("pathlib").Path
    out = Path("tmp_test/online_llm_report.json")
    out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"JSON written to {out}")
    if not generate.ok and "No OPENAI" in generate.detail:
        return 2
    return 0 if generate.ok and all(p.ok for p in probes) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
