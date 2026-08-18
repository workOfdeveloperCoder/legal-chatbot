---
# Project Progress Log

## Last Updated
Monday, Aug 17, 2026 — 11:10 PM (UTC+5)

## Current State
- Production Phase 1 chat core remains in place
- Automatic language detection is now part of the RAG chat path
- Supported user languages: English (Pakistan), Urdu (Pakistan), Pakistani Punjabi
- Mixed/code-switched input and Roman Urdu are handled without a language picker
- **235+ tests passing** (1 deselected tiktoken network test, unrelated)
- **735 bulk ingest not started** (awaiting legal-gpt completed/ migration)

## What Was Done This Session
- Inspected ChatRequest, ChatService, QueryRouter, RAGService, PromptBuilder, LLM/reasoning, API schemas
- Confirmed this repo has no frontend and no speech-to-text provider
- Added a lightweight heuristic language detector (`app/rag/language.py`) returning `en-PK` / `ur-PK` / `pa-PK` / `mixed` plus confidence, response style, and explicit-override handling
- Wired detection into `RAGService.execute` before rewrite/retrieval; original query is preserved
- Updated legal prompts so the LLM answers in the detected/dominant language and keeps Pakistani legal terms intact
- Extended QueryRouter, QueryRewriter, and LegalQueryPlanner so Urdu `دفعہ` / ضمانت still route and retrieve as legal questions
- Rewriter appends English legal anchors (`section 497`, `bail`) for retrieval without translating the user query
- Exposed detection on `retrieval_metadata.language` (no ChatRequest language field)
- Added focused tests in `tests/test_language_detection.py` covering the required English, Urdu, Roman Urdu, Punjabi, mixed, override, and citation cases

## In Progress / Half Done
- Speech-to-text multilingual configuration is pending a frontend/STT provider; this backend detects language on the resulting text/transcript
- Live LLM response-language quality still depends on the chat model following the new prompt instructions

## Next Steps (Do This First When You Return)
1. If a frontend/STT layer is added, configure multilingual recognition (en-PK, ur-PK, pa-PK) without forcing a single language, and keep chatbot detection on the transcript
2. After legal-gpt migrates more completed docs, re-test `/chat` on Section 54-C / PIL queries
3. Align frontend token_budget field mapping
4. Optional hybrid lexical search

## Known Issues / Blockers
- Most legal_documents points still pre-v4 (parent expansion no-ops)
- Claim grounding remains lexical
- No STT implementation exists in this repository, so microphone language auto-detect cannot be configured here
- `tests/test_token_budget.py::test_exact_tiktoken_counting` can fail without network access to the tiktoken encoding file (pre-existing)

## Key Decisions & Context
- Do not ask the user to select a language
- Do not translate the full query before retrieval; keep `retrieval_language=original`
- English legal loanwords (bail, FIR, PPC, section, etc.) must not force English classification
- Roman Urdu is treated as Urdu-style input via lexicon, not Unicode script alone
- Punjabi detection targets Pakistani Punjabi (eh/ae/kithay/menu/sakda), not Indian Gurmukhi assumptions
- Explicit user language requests override automatic detection
- legal-chatbot never writes corpus Qdrant
- document_id is canonical for resources
---
