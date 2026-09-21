# Project Progress Log

## Last Updated
Tuesday, Sep 22, 2026 — ~1:54 AM (UTC+5)

## Current State
- Chat LLM: **Gemini** `gemini-flash-latest`
- Backend: `http://127.0.0.1:8001` (uvicorn --reload)
- Frontend: `http://127.0.0.1:5173` (Vite); proxy → `:8001`
- Qdrant: local Docker on `:6333` (`legal_documents`)
- Opening a Resources card for a legal corpus file now loads **full merged document text** via `GET /api/v1/library/documents/{document_id}`

## What Was Done This Session
- Added `LibraryDocumentService` — scrolls Qdrant by `document_id` / filename, merges child `chunk_text` in order
- Added `GET /api/v1/library/documents/{document_id}` (+ `LibraryDocumentDetail` schema)
- Wired UI `resolveDocumentGetPath` so `library` / `legal` cards hit that endpoint (non-UUID corpus ids allowed)
- Verified `a4fdab6fd0bcdf55` (`2000J8.txt`): ~4047 chars, includes Section 54-C; API 200; UI unit tests pass

## In Progress / Half Done
- None for full-text Resources open

## Next Steps (Do This First When You Return)
1. In UI, open Resources → `2000J8` and confirm **Full document** section (not only Retrieved evidence)
2. If a longer statute appears truncated, check whether Qdrant only indexed partial chunks for that `document_id`

## Known Issues / Blockers
- Docker Desktop must be running for local Qdrant
- Full text is only as complete as indexed Qdrant chunks (parents skipped when children exist)

## Key Decisions & Context
- Corpus ids are hex hashes (e.g. `a4fdab6fd0bcdf55`), not Postgres UUIDs — separate `/library/documents` route
- Prefer child chunks over parents when merging (parents duplicate children)
- Local stack: UI `:5173` → LegalGPT `:8001` → Gemini + Ollama embed + Qdrant `:6333`
