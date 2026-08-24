from __future__ import annotations

from dataclasses import dataclass

from app.llm.base import BaseLLM
from app.llm.execution_profile import ExecutionProfile
from app.llm.model_capabilities import ModelCapabilities
from app.rag.prompts import REASONING_SYSTEM_PROMPT
from app.rag.query_complexity import QueryComplexity
from app.rag.reasoning_cleanup import strip_reasoning_output
from app.schemas.llm import ChatCompletionRequest, ChatCompletionResponse, ChatMessage


@dataclass(slots=True)
class GenerationResult:
    content: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    two_stage: bool = False


STAGE1_INSTRUCTION = """
INTERNAL EVIDENCE ANALYSIS (NOT FOR END USER — DO NOT OUTPUT TO USER)

Analyze the retrieved evidence carefully before any final answer is written.

Work through these steps internally:
1. Identify the precise legal issue(s) in the question.
2. List which retrieved sources are LEGAL AUTHORITY vs uploaded documents vs context.
3. For each source type, note what propositions it actually supports.
4. Identify governing provisions/rules where the evidence establishes them.
5. Note exceptions, limitations, or gaps in the evidence.
6. Identify any conflicting authorities or document-vs-law tension.
7. Determine what conclusion is actually supported — not merely keyword overlap.
8. Mark each material proposition as SUPPORTED, PARTIALLY_SUPPORTED, or UNSUPPORTED
   by the retrieved evidence.

Output a concise structured internal memo with sections:
- LEGAL ISSUE
- APPLICABLE AUTHORITY
- DOCUMENT POSITION (if any uploaded material)
- ANALYSIS
- CONFLICTS / UNCERTAINTY
- SUPPORTED CONCLUSION
- LIMITATIONS

Do NOT write the user-facing answer yet.
Do NOT expose chain-of-thought narrative.
Be rigorous: a chunk mentioning a section alone does not establish a multi-part legal test.
"""


STAGE2_INSTRUCTION = """
FINAL ANSWER GENERATION

Using ONLY the internal evidence analysis above and the retrieved source blocks,
write the professional legal answer for the user.

Rules:
- Return ONLY the final answer — no analysis memo, no chain-of-thought.
- Ground every legal proposition in retrieved evidence.
- Distinguish "The law provides..." from "The article argues..." from "The court held..."
- If evidence is partial or conflicting, say so clearly.
- Use [Source N] inline citations matching the numbered evidence blocks.
- Do NOT add a trailing Sources/References section.
- Do NOT invent authorities, sections, cases, quotes, dates, or holdings.
- Follow any RESPONSE LANGUAGE instruction in the user prompt.
- Preserve legal terms, section numbers, and citations exactly.
"""


class ReasoningPipeline:
    """Optional two-stage generation for research/complex legal questions."""

    def __init__(
        self,
        llm: BaseLLM,
        profile: ExecutionProfile | None = None,
    ) -> None:
        self._llm = llm
        if profile is not None:
            self._profile = profile
        else:
            caps = None
            getter = getattr(llm, "capabilities", None)
            try:
                raw = getter() if callable(getter) else None
                if isinstance(raw, ModelCapabilities):
                    caps = raw
            except Exception:  # noqa: BLE001
                caps = None
            self._profile = ExecutionProfile.for_capabilities(caps)

    async def generate(
        self,
        *,
        user_prompt: str,
        complexity: QueryComplexity,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> GenerationResult:
        use_two_stage = (
            self._profile.two_stage_reasoning
            and complexity in {
                QueryComplexity.RESEARCH,
                QueryComplexity.COMPLEX,
            }
        )

        if not use_two_stage:
            response = await self._single_call(
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return GenerationResult(
                content=strip_reasoning_output(response.content),
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                total_tokens=response.total_tokens,
                two_stage=False,
            )

        stage1 = await self._stage1_analysis(
            user_prompt=user_prompt,
            temperature=temperature,
            # R1 needs headroom for hidden thinking before the memo body.
            max_tokens=max(max_tokens, 4096),
        )
        stage2 = await self._stage2_answer(
            user_prompt=user_prompt,
            analysis=stage1.content,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        total_prompt = (stage1.prompt_tokens or 0) + (stage2.prompt_tokens or 0)
        total_completion = (stage1.completion_tokens or 0) + (stage2.completion_tokens or 0)

        return GenerationResult(
            content=strip_reasoning_output(stage2.content),
            prompt_tokens=total_prompt or None,
            completion_tokens=total_completion or None,
            total_tokens=(total_prompt + total_completion) or None,
            two_stage=True,
        )

    async def _single_call(
        self,
        *,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> ChatCompletionResponse:
        request = ChatCompletionRequest(
            messages=[
                ChatMessage(role="system", content=self._profile.system_prompt),
                ChatMessage(role="user", content=user_prompt),
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        response = await self._llm.generate(request)
        return ChatCompletionResponse(
            content=strip_reasoning_output(response.content),
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
        )

    async def _stage1_analysis(
        self,
        *,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> ChatCompletionResponse:
        request = ChatCompletionRequest(
            messages=[
                ChatMessage(role="system", content=REASONING_SYSTEM_PROMPT),
                ChatMessage(
                    role="user",
                    content=f"{user_prompt}\n\n{STAGE1_INSTRUCTION}",
                ),
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        response = await self._llm.generate(request)
        return ChatCompletionResponse(
            content=strip_reasoning_output(response.content),
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
        )

    async def _stage2_answer(
        self,
        *,
        user_prompt: str,
        analysis: str,
        temperature: float,
        max_tokens: int,
    ) -> ChatCompletionResponse:
        request = ChatCompletionRequest(
            messages=[
                ChatMessage(role="system", content=self._profile.system_prompt),
                ChatMessage(
                    role="user",
                    content=(
                        f"{user_prompt}\n\n"
                        f"INTERNAL EVIDENCE ANALYSIS (verified — do not repeat verbatim):\n"
                        f"{analysis}\n\n"
                        f"{STAGE2_INSTRUCTION}"
                    ),
                ),
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return await self._llm.generate(request)
