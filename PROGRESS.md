---
# Project Progress Log

## Last Updated
Friday, Aug 21, 2026 — 10:00 PM (UTC+5)

## Current State
- Chat LLM is still local Ollama by default; switching to a hosted model is env-only
- New adapter registry: add OpenAI-compatible vendors or a new wire protocol (Anthropic/Claude) without changing RAG
- Online execution profile spends fewer tokens: compact system prompt, no duplicated system text in the user message, no two-stage reasoning, tighter evidence caps
- Embeddings stay on local Ollama `nomic-embed-text` (768-d)
- Token budget packing still runs before every LLM call
- Composer token meter in legal-ai-ui is local + one `GET /llm/status` per login
- Voice Mode remains local (faster-whisper + Piper/espeak-ng)
- Backend **265 tests passed**

## What Was Done This Session
- Live Gemini generation **succeeded** on `gemini-3.6-flash` (HTTP 200, 4.5s, 5790/95/6271 tokens). `gemini-2.0-flash` is retired (404). Default Gemini model updated. `.env` is currently `LLM_PROVIDER=gemini` with `LLM_API_KEY` set; embeddings still local Ollama.
- Added `LLMAdapterRegistry` so providers self-register (`@LLMAdapterRegistry.register(...)`)
- Factory no longer has per-vendor if/elif; it only calls `LLMAdapterRegistry.connect`
- Added Anthropic/Claude adapter (`app/llm/anthropic.py`) as a second wire protocol
- Registered extra OpenAI-compatible vendors: gemini, mistral, together, fireworks
- Added `ExecutionProfile`: local (full prompt, optional two-stage) vs online (compact, single call, smaller reservations)
- Compact system prompt (~153 words vs ~853); hosted models do not re-embed the system prompt in the user message
- Online packing caps: reserved output 1536 (was 4096), legal evidence 3500 (was 8000)
- `ANTHROPIC_API_KEY` / `GOOGLE_API_KEY` aliases; `api_key_for_provider()`
- Tests for registry plugin, Anthropic payload, Gemini URL, compact prompts, skipped two-stage on hosted models

## In Progress / Half Done
- Online chat is **not** the process default until env is set (keeps local Ollama working without an API key)

## Next Steps (Do This First When You Return)
1. Put any hosted key in `.env` as `LLM_API_KEY=` (same name for Gemini/OpenAI/Claude). Then set:
   ```
   LLM_PROVIDER=gemini
   CHAT_MODEL=gemini-3.6-flash
   LLM_API_KEY=...
   LLM_TIMEOUT=120
   EMBEDDING_PROVIDER=ollama
   EMBEDDING_URL=http://localhost:11434
   ```
2. Restart FastAPI and confirm `GET /api/v1/llm/status` shows `online: true`
3. In legal-ai-ui, confirm the composer chip shows `used / window` while typing
4. After legal-gpt migrates more completed docs, re-test `/chat` on Section 54-C / PIL queries

## Known Issues / Blockers
- Most legal_documents points still pre-v4 (parent expansion no-ops)
- Claim grounding remains lexical
- Voice Mode returns 503 until faster-whisper and Piper/espeak-ng are installed
- Official Piper voices are weak for Urdu/Punjabi; espeak-ng is the local fallback
- Switching embeddings to OpenAI would require re-indexing Qdrant (do not do this)
- tiktoken `cl100k_base` / `o200k_base` files may need a one-time download; fallback estimator is used if missing

## Key Decisions & Context
- Do not ask the user to select a language
- Do not translate the full query before retrieval; keep `retrieval_language=original`
- Chat LLM and embeddings are independent: hosted chat, local nomic embeddings
- Voice stays local; no OpenAI Realtime / Deepgram
- Token limits follow the **chat** model window, not the embedding model
- Hosted models skip two-stage reasoning (`LLM_TWO_STAGE_ONLINE=false`) because the extra call doubles cost
- Default process settings remain Ollama so the app still boots without an API key
- document_id is canonical for resources
- legal-chatbot never writes corpus Qdrant
---
