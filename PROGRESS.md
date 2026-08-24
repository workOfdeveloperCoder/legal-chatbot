---
# Project Progress Log

## Last Updated
Monday, Aug 24, 2026 — 6:05 PM (UTC+5)

## Current State
- Production-hardening code is in the repo: Qdrant API key on the shared client, production fail-closed checks, snapshot recover URIs, token overflow rejection, matter attach authorization, request-size middleware, firewall tightening
- The application is **not production-ready** until a real server restore of `legal_documents`, private Qdrant + API key, `/ready`, and a Gemini live chat drill are verified
- Unit tests: **370 passed**, 1 skipped, 0 failed
- Deploy is **no Docker**: systemd + Nginx + loopback Qdrant/Postgres/Ollama (see `docs/PRODUCTION.md` and `deploy/legal-chatbot.service`)
- Embeddings remain local `nomic-embed-text` 768-d cosine; `legal_documents` is never recreated

## What Was Done This Session
- Production `ENVIRONMENT=production` validation: require Qdrant API key, non-wildcard `TRUSTED_HOSTS`, hosted LLM key
- Qdrant initialize fails closed without API key in production; logs `api_key_configured` never the secret
- Snapshot recover builds a real Qdrant `location` (HTTP snapshot URL or `file://`)
- Token budget truncates an oversized current question after shedding scaffolding; raises `TokenBudgetExceeded` (HTTP 413) instead of sending an oversized prompt
- Gemini cl100k counts remain approximate (`tokenizer_native=false`; `/api/v1/llm/status` `tokenizer_exact` is false)
- Online LLM timeout stays 120s when config is still the local 1800s sentinel; local DeepSeek 1800s preserved
- Firewall: Roman Urdu / Urdu how-tos, dotted obfuscation, roleplay jailbreaks, multi-turn “continue” after an illegal turn; retrieved evidence stays untrusted
- `ChatService` re-checks matter ownership before attaching `payload.matter_id` to an existing conversation
- `RequestSizeLimitMiddleware` enforces `MAX_REQUEST_BYTES` via Content-Length (413)
- Upload types limited to PDF/txt/md/csv (no broad `text/*`)
- Rate limiter ignores `X-Forwarded-For` unless `TRUST_FORWARDED_FOR=true`
- Docs/systemd template/`.env.example` updated for Ubuntu/CloudPanel/Nginx
- Tests added for ops recover URIs, overflow, Gemini vs DeepSeek profiles, matter 403, request size, production config, firewall/injection

## In Progress / Half Done
- Live Qdrant API-key handshake: **NOT VERIFIED**
- `legal_documents` snapshot restore on a production host: **NOT VERIFIED**
- Live SSE against Gemini: **NOT VERIFIED**
- Backup/restore drill: **NOT VERIFIED**
- Document ingestion still runs inside the upload HTTP request
- Rate limiter is in-process (use 1 uvicorn worker)

## Next Steps (Do This First When You Return)
1. On the target server: bind Qdrant to 127.0.0.1, set API key, restore `legal_documents`, run `python scripts/qdrant_ops.py verify legal_documents --vector-size 768 --payload`
2. Copy `.env.example` → `/etc/legal-chatbot.env`; `alembic upgrade head`; enable `deploy/legal-chatbot.service` + Nginx from `docs/PRODUCTION.md`
3. Confirm `curl /health` and `curl /ready` (postgres, qdrant, embeddings)
4. Confirm `GET /api/v1/llm/status` reports Gemini `tokenizer_exact: false` (approximation) or fallback id if tiktoken is missing
5. Exercise `POST /api/v1/chat/stream` with a real lawyer query, then Stop
6. Only then consider background ingestion / Redis limiter / object storage

## Known Issues / Blockers
- Rule-based firewall can still miss paraphrases or over-block unusual wording
- Most `legal_documents` points still pre-v4 (parent expansion no-ops) — do not destroy the corpus
- Claim grounding remains lexical
- Large uploads can block API workers (sync embed)
- Chunked requests without Content-Length are not size-checked in-app (Nginx `client_max_body_size` is the backstop)
- Voice Mode returns 503 until faster-whisper and Piper are usable
- Switching embeddings to OpenAI would require re-indexing Qdrant (do not)
- MinIO/Redis settings exist but are unused; uploads are local disk

## Key Decisions & Context
- No Docker / Compose — direct Linux services
- Do not ask the user to select a language
- Chat LLM and embeddings are independent: hosted Gemini chat, local nomic embeddings
- legal-chatbot never writes corpus Qdrant (`legal_documents` is legal-gpt owned)
- Criminal-law Q&A is allowed; operational "how to commit X" is not
- Programming/code is always out of scope
- Qdrant must not be exposed on a public IP
- JWT `user_id` is the only tenant key for private vectors — never trust a client-supplied user id
- Gemini tokenization is cl100k approximation, not native
---
