from app.rag.models import (
    Message,
    RetrievedChunk,
    Memory,
    SourceType,
)

from app.rag.prompts import LEGAL_SYSTEM_PROMPT
from app.rag.answer_guard import EvidenceStrength
from app.rag.evidence_assessment import EvidenceAssessment
from app.rag.query_complexity import QueryComplexity
from app.rag.retrieval_pipeline import build_source_reference
from app.rag.language import LanguageDetection


class PromptBuilder:

    def build(
        self,
        question: str,
        history: list[Message],
        chunks: list[RetrievedChunk],
        memories: list[Memory],
        task: str | None = None,
        document_scoped: bool = False,
        document_qa_mode: bool = False,
        mixed_qa_mode: bool = False,
        evidence_strength: EvidenceStrength | None = None,
        evidence_assessment: EvidenceAssessment | None = None,
        complexity: QueryComplexity | None = None,
        conversation_summary: str | None = None,
        active_legal_context: str | None = None,
        language: LanguageDetection | None = None,
        include_system_in_user: bool = True,
        compact: bool = False,
    ) -> str:

        prompt_parts = []
        if include_system_in_user:
            prompt_parts.append(LEGAL_SYSTEM_PROMPT)

        if complexity is not None:
            prompt_parts.append(self._build_complexity_guidance(complexity))

        if language is not None:
            prompt_parts.append(self._build_language_instruction(language))

        if task:
            prompt_parts.append(
                f"""
LEGAL TASK TYPE:

{task}
"""
            )

        if document_scoped:
            prompt_parts.append(
                """
DOCUMENT SCOPE INSTRUCTION:

Summarize or answer using ONLY the supplied document context below.
Do not use outside knowledge or unrelated retrieved documents.
Base the answer solely on the retrieved chunks from the requested document.
Attribute all legal propositions to the document:
"The article argues..." / "According to the document..."
"""
            )
        elif mixed_qa_mode:
            prompt_parts.append(
                """
MIXED DOCUMENT + LEGAL AUTHORITY INSTRUCTION:

The user wants to compare uploaded material against authoritative law.

Answer in two distinct layers:

1. DOCUMENT'S POSITION — what the uploaded document/article argues.
2. LEGAL ASSESSMENT — whether available legal authorities support,
   contradict, or do not establish that argument.

Rules:
- Use uploaded documents for author/article arguments only.
- Use LEGAL AUTHORITY for what statute, rule, or judgment establishes.
- Clearly separate:
  "The article argues..." vs "The statute provides..." vs "The court held..."
- Do NOT treat the uploaded document as binding law.
- If legal authorities do not support the document's argument, say so.
- Cite [Source N] inline only where needed.
- Do NOT add a Sources/References section at the end.
"""
            )
        elif document_qa_mode:
            prompt_parts.append(
                """
DOCUMENT Q&A INSTRUCTION:

The user is asking about content in uploaded matter or conversation documents.
The document itself is the primary evidence.

Rules:
- Write in a direct, professional tone for legal practitioners.
- ALWAYS attribute legal propositions to the document/author:
  "According to the article..." / "The author argues..." /
  "The uploaded document identifies..."
- Do NOT present document arguments as independently verified current law.
  Wrong: "Section 54-C is a clog on judicial discretion."
  Right: "According to the article, Section 54-C can operate as a clog
  on judicial discretion."
- Do NOT reject an answer merely because legal corpus evidence is missing.
- Use LEGAL AUTHORITY sources only if retrieved and directly relevant.
- Cite [Source N] inline only where a specific claim needs attribution.
- Do NOT add a Sources, References, or Citations section.
- Avoid meta phrases such as "The retrieved evidence highlights".
- Answer the question directly in the opening sentences.
"""
            )
        else:
            prompt_parts.append(
                """
LEGAL RESEARCH INSTRUCTION:

Reason over the retrieved evidence — do not merely summarize chunks.

Internally determine:
- The precise legal issue
- Governing provision/rule (if established by evidence)
- Supporting authorities and any exceptions
- Conflicting authorities (if any)
- How retrieved material applies to THIS question
- What conclusion is actually supported

A chunk mentioning a keyword alone is NOT sufficient for a multi-part legal test.
"""
            )

        if memories:
            prompt_parts.append(self._build_memory(memories))

        if conversation_summary:
            prompt_parts.append(
                f"CONVERSATION SUMMARY:\n\n{conversation_summary}"
            )

        if active_legal_context:
            prompt_parts.append(active_legal_context)

        if history:
            prompt_parts.append(self._build_history(history))

        prompt_parts.append(self._build_context(chunks))

        if evidence_assessment is not None:
            prompt_parts.append(
                self._build_assessment_guidance(evidence_assessment)
            )
        elif evidence_strength is not None:
            prompt_parts.append(
                self._build_evidence_guidance(
                    evidence_strength,
                    document_scoped=document_scoped,
                    document_qa_mode=document_qa_mode,
                )
            )

        prompt_parts.append(self._build_answer_contract(question, complexity, compact=compact))

        return "\n\n".join(prompt_parts)

    def _build_complexity_guidance(
        self,
        complexity: QueryComplexity,
    ) -> str:
        if complexity == QueryComplexity.SIMPLE:
            return """
QUESTION COMPLEXITY: SIMPLE

Provide a concise, direct answer. Do not over-structure.
"""
        if complexity == QueryComplexity.RESEARCH:
            return """
QUESTION COMPLEXITY: RESEARCH

This requires applied legal analysis, not a definition.
Reason carefully over the evidence before concluding.
Lead with a direct answer, then explain briefly.
Use a natural structure — do not force rigid section headings.
"""
        return """
QUESTION COMPLEXITY: COMPLEX

This question involves multiple legal concepts and/or document-law comparison.
Synthesize across sources. Identify conflicts and limitations explicitly.
Lead with a direct answer. Use clear structure only when it helps readability.
"""

    def _build_language_instruction(
        self,
        language: LanguageDetection,
    ) -> str:
        style_guide = {
            "english": (
                "Write the answer in professional Pakistani legal English."
            ),
            "urdu_script": (
                "Write the answer in Urdu using Arabic/Nastaliq script."
            ),
            "roman_urdu": (
                "Write the answer in Roman Urdu (Urdu in Latin characters). "
                "Do not switch to English prose."
            ),
            "punjabi_roman": (
                "Write the answer in Pakistani Punjabi using Latin/Roman "
                "script (for example eh/ae/di/kithay). Do not answer in "
                "Urdu or assume Indian Punjabi."
            ),
            "punjabi_shahmukhi": (
                "Write the answer in Pakistani Punjabi using Shahmukhi "
                "(Arabic) script. Do not answer in Urdu."
            ),
            "mixed": (
                "The user is code-switching. Reply in the same mixed style, "
                "keeping the dominant register and leaving English legal "
                "terms in English."
            ),
        }
        style_key = language.response_style.value
        guide = style_guide.get(style_key, style_guide["english"])
        override = (
            "The user explicitly requested this response language. "
            "Follow that request over automatic detection."
            if language.explicit_override
            else "No explicit language request; follow the detected style."
        )
        return f"""
RESPONSE LANGUAGE:

- user_language: {language.user_language.value}
- response_language: {language.response_language.value}
- response_style: {style_key}
- confidence: {language.confidence:.2f}
- explicit_override: {str(language.explicit_override).lower()}

{guide}
{override}

Keep section numbers, case citations (PLD/SCMR/etc.), statute names
(PPC, CrPC, CPC, QSO), FIR, and other Pakistani legal terms unchanged.
Do not fabricate translations of legal terminology.
"""

    def _build_assessment_guidance(
        self,
        assessment: EvidenceAssessment,
    ) -> str:
        missing = ", ".join(assessment.missing_concepts) or "none identified"
        covered = ", ".join(assessment.covered_concepts) or "none matched"

        lines = [
            "EVIDENCE QUALITY ASSESSMENT:",
            f"- authority_strength: {assessment.authority_strength:.2f}",
            f"- evidence_relevance: {assessment.evidence_relevance:.2f}",
            f"- coverage: {assessment.coverage:.2f}",
            f"- source_quality: {assessment.source_quality:.2f}",
            f"- overall: {assessment.overall_strength.value}",
            f"- concepts covered by evidence: {covered}",
            f"- concepts missing from evidence: {missing}",
        ]

        if assessment.missing_concepts:
            lines.append(
                "IMPORTANT: Retrieved evidence does not fully cover all concepts "
                "in the question. Do not state a definitive conclusion on missing "
                "elements unless the evidence actually supports the relationship."
            )

        if assessment.has_conflicts:
            lines.append(
                "CONFLICT WARNING: Retrieved authorities may contain conflicting "
                "positions. Identify the conflict rather than merging into one rule."
            )

        if assessment.overall_strength == EvidenceStrength.NONE:
            lines.append(
                "Do not provide a confident legal answer. State that available "
                "sources are insufficient."
            )
        elif assessment.overall_strength == EvidenceStrength.WEAK:
            lines.append(
                "Qualify the response heavily. Do not present private documents "
                "as authoritative law."
            )
        elif assessment.overall_strength == EvidenceStrength.PARTIAL:
            lines.append(
                "Qualify conclusions where evidence is incomplete."
            )

        return "\n".join(lines)

    def _build_answer_contract(
        self,
        question: str,
        complexity: QueryComplexity | None,
        compact: bool = False,
    ) -> str:
        structure = ""
        if complexity == QueryComplexity.COMPLEX:
            structure = """
Write a clear professional answer:
1. Direct answer first.
2. Explain the relevant position from the evidence.
3. Apply the evidence to the issue; distinguish document claims from law.
4. State limitations where evidence is incomplete or conflicting.

Do not force rigid markdown section headings unless they improve clarity.
"""
        elif complexity == QueryComplexity.RESEARCH:
            structure = """
Lead with a direct answer, then explain the supporting evidence briefly.
Keep the structure natural and professional.
"""

        brevity = ""
        if compact:
            brevity = """
Be token-efficient: default to 1–3 short paragraphs. No preamble, no
restating the question, no heading template unless the question is complex.
"""

        return f"""
USER QUESTION:

{question}

Provide a professional legal research answer grounded ONLY in the retrieved
evidence above. Reason over the evidence — do not merely restate it.
{structure}{brevity}
Attribution language (use precisely):
- Statute/law: "The statute provides...", "Section X states..."
- Judgment: "The court held...", "The judgment established..."
- Uploaded document/article: "The article argues...", "According to the
  uploaded document...", "The author contends..."
- Analysis: "This suggests...", "On the available evidence..."
- Never present an author's argument or secondary material as binding law.

Rules:
- Follow the RESPONSE LANGUAGE instruction above when present.
- Cite sources using [Source N] matching the numbered blocks.
- Do not include a separate Sources, References, or Citations section.
- Distinguish legal authority from matter/conversation documents.
- Do not invent statutes, sections, cases, citations, or quotes.
- If evidence is insufficient, say clearly:
  "I don't have enough reliable evidence in the available sources to
  answer that conclusively."
- Be concise for simple questions; provide detail only when required.
- Return ONLY the final answer — no chain-of-thought or internal analysis.
"""
        structure = ""
        if complexity == QueryComplexity.COMPLEX:
            structure = """
Write a clear professional answer:
1. Direct answer first.
2. Explain the relevant position from the evidence.
3. Apply the evidence to the issue; distinguish document claims from law.
4. State limitations where evidence is incomplete or conflicting.

Do not force rigid markdown section headings unless they improve clarity.
"""
        elif complexity == QueryComplexity.RESEARCH:
            structure = """
Lead with a direct answer, then explain the supporting evidence briefly.
Keep the structure natural and professional.
"""

        return f"""
USER QUESTION:

{question}

Provide a professional legal research answer grounded ONLY in the retrieved
evidence above. Reason over the evidence — do not merely restate it.
{structure}
Attribution language (use precisely):
- Statute/law: "The statute provides...", "Section X states..."
- Judgment: "The court held...", "The judgment established..."
- Uploaded document/article: "The article argues...", "According to the
  uploaded document...", "The author contends..."
- Analysis: "This suggests...", "On the available evidence..."
- Never present an author's argument or secondary material as binding law.

Rules:
- Follow the RESPONSE LANGUAGE instruction above when present.
- Cite sources using [Source N] matching the numbered blocks.
- Do not include a separate Sources, References, or Citations section.
- Distinguish legal authority from matter/conversation documents.
- Do not invent statutes, sections, cases, citations, or quotes.
- If evidence is insufficient, say clearly:
  "I don't have enough reliable evidence in the available sources to
  answer that conclusively."
- Be concise for simple questions; provide detail only when required.
- Return ONLY the final answer — no chain-of-thought or internal analysis.
"""

    def _build_context(
        self,
        chunks: list[RetrievedChunk],
    ) -> str:

        if not chunks:
            return """
RETRIEVED EVIDENCE:

No legal documents were retrieved.

Do not fabricate legal information.
State clearly that available sources are insufficient.
"""

        grouped: dict[str, list[tuple[int, RetrievedChunk]]] = {
            SourceType.LEGAL.value: [],
            SourceType.CONVERSATION.value: [],
            SourceType.MATTER.value: [],
        }

        for index, chunk in enumerate(chunks, 1):
            source_type = chunk.source_type or SourceType.LEGAL.value
            if source_type not in grouped:
                source_type = SourceType.LEGAL.value
            grouped[source_type].append((index, chunk))

        sections: list[str] = []
        section_titles = {
            SourceType.LEGAL.value: (
                "LEGAL AUTHORITY (establishes what the law is — priority 1-3)"
            ),
            SourceType.CONVERSATION.value: (
                "CONVERSATION DOCUMENTS (uploaded research/articles — priority 5)"
            ),
            SourceType.MATTER.value: (
                "MATTER DOCUMENTS (case-specific facts/content — priority 4)"
            ),
        }

        for source_type in (
            SourceType.LEGAL.value,
            SourceType.CONVERSATION.value,
            SourceType.MATTER.value,
        ):
            items = grouped[source_type]
            if not items:
                continue

            blocks = [
                self._format_source_block(index, chunk)
                for index, chunk in items
            ]

            sections.append(
                f"{section_titles[source_type]}\n\n"
                + "\n".join(blocks)
            )

        return (
            "RETRIEVED EVIDENCE (for reasoning — grouped by authority hierarchy):\n\n"
            + "\n\n".join(sections)
        )

    def _format_source_block(
        self,
        index: int,
        chunk: RetrievedChunk,
    ) -> str:
        meta_parts: list[str] = []

        ref = build_source_reference(chunk)
        display = chunk.display_name or chunk.filename or ref
        if display:
            meta_parts.append(f"Document: {display}")
        if chunk.filename and chunk.filename != display:
            meta_parts.append(f"Filename: {chunk.filename}")
        if getattr(chunk, "author", None):
            meta_parts.append(f"Author: {chunk.author}")

        if chunk.document_id:
            meta_parts.append(f"Document ID: {chunk.document_id}")

        if chunk.chunk_id:
            meta_parts.append(f"Chunk ID: {chunk.chunk_id}")

        if chunk.page_number:
            meta_parts.append(f"Page: {chunk.page_number}")

        meta = " | ".join(meta_parts)

        return f"""
[Source {index}]
Type: {chunk.source_type or SourceType.LEGAL.value}
{meta}

Relevant text:

{chunk.text}
"""

    def _build_evidence_guidance(
        self,
        evidence_strength: EvidenceStrength,
        *,
        document_scoped: bool,
        document_qa_mode: bool = False,
    ) -> str:
        if document_scoped or document_qa_mode:
            return """
EVIDENCE STATUS:

Document-focused task. Ground answers in retrieved document content.
Present author/document positions as document claims.
"""

        if evidence_strength == EvidenceStrength.NONE:
            return """
EVIDENCE STATUS: INSUFFICIENT

No sufficient authoritative legal material was retrieved.
Do not provide a confident legal answer.
"""

        if evidence_strength == EvidenceStrength.WEAK:
            return """
EVIDENCE STATUS: WEAK

Retrieved material is limited and may lack authoritative legal sources.
Qualify the response and do not present private documents as authoritative law.
"""

        if evidence_strength == EvidenceStrength.PARTIAL:
            return """
EVIDENCE STATUS: PARTIAL

Some relevant material was retrieved, but support may be incomplete.
Qualify conclusions where necessary.
"""

        return """
EVIDENCE STATUS: ADEQUATE

Sufficient retrieved material is available. Ground legal propositions in
LEGAL AUTHORITY sources; use matter/conversation documents for facts only.
"""

    def _build_history(
        self,
        history: list[Message],
    ) -> str:

        messages = []

        # History is expected to already be budget-fitted by TokenBudgetManager.
        for msg in history:
            messages.append(
                f"""
{msg.role.upper()}:

{msg.content}
"""
            )

        return """

PREVIOUS CONVERSATION:

""" + "\n".join(messages)

    def _build_memory(
        self,
        memories: list[Memory],
    ) -> str:

        items = []

        for memory in memories:
            items.append(
                f"""
{memory.text}
"""
            )

        return """

USER LEGAL MEMORY:

""" + "\n".join(items)
