# Project Progress Log

## Last Updated
Monday, Aug 31, 2026 — ~9:50 PM (UTC+5)

## Current State
- Chat LLM: **OpenRouter** `minimax/minimax-m2.7:free`
- Qdrant search: **Hugging Face fastembed** `nomic-ai/nomic-embed-text-v1.5` (768-d, matches `legal_documents`)
- Web search: only when UI Web toggle is on (`payload.web_search`)
- `.env` switched away from OpenRouter Nemotron embed (2048-d — incompatible with Qdrant index)

## What Was Done This Session
- Diagnosed missing library Resources: `EMBEDDING_MODEL=nvidia/nemotron-3-embed-1b:free` → `qdrant_embedding_compatible=False` → retriever returns zero legal chunks
- Restored `.env` to `EMBEDDING_PROVIDER=huggingface` + nomic v1.5 + HF token; set `EMBEDDING_ONLINE_ONLY=false`
- Verified end-to-end: 768-d embed → 3 Qdrant hits for "section 417 PPC"
- Hardened `response_formatter.py`: strip `source_type=web` from sources/resources when `include_web=False`
- Added chat request log: `web_search` + `quick_action` for debugging UI toggle issues

## In Progress / Half Done
- API on `:8000` may be a **different app** (404 on `/api/v1/health/ready`); legal-chatbot needs restart on correct port
- Remote server (`172.16.112.17:8000`?) may still run old code + old `.env`

## Next Steps (Do This First When You Return)
1. **Restart legal-chatbot** so it loads the new `.env` (huggingface nomic, not OpenRouter embed)
2. Confirm `/ready` shows `"embedding_provider": "huggingface"`, `"qdrant_embedding_compatible": true` (via stack.search.model contains nomic)
3. Retry a legal query with **Web toggle OFF** — Resources should show `legal` corpus docs, no `web` badge
4. If UI still shows web junk, check API logs for `Chat request web_search=True` (UI may be sending toggle on)
5. For strict no-download server: use remote Ollama sidecar or Fireworks nomic API instead of fastembed

## Known Issues / Blockers
- OpenRouter free embed models are **not nomic 768-d** — cannot search existing Qdrant without full re-index
- fastembed downloads ~300MB once on first embed (not ideal for zero-download production)
- Port 8000 currently occupied by another service

## Key Decisions & Context
- UI Web toggle is the single source of truth for internet search
- Qdrant `legal_documents` index is 768-d nomic — embedding model name must contain "nomic"
- Chat (OpenRouter) and search (nomic embed) are intentionally separate providers
