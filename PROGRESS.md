# Project Progress Log

## Last Updated
Tuesday, Sep 22, 2026 — ~3:45 AM (UTC+5)

## Current State
- Chat LLM: **Gemini** `gemini-flash-latest`
- Backend: `http://127.0.0.1:8001` (uvicorn --reload)
- Frontend: `http://127.0.0.1:5173` (Vite); proxy → `:8001`
- Qdrant: local Docker on `:6333` (`legal_documents`)
- Resources full text: `GET /api/v1/library/documents/{document_id}` merges **all** Qdrant chunks with overlap collapse

## What Was Done This Session
- Library document merge: overlap-aware stitch + clean document-start seed; paginated Qdrant scroll
- UI display reflow fills width but keeps headings / blank lines
- Verified `2000J8` / `a4fdab6fd0bcdf55` → full text starts at `CLOG ON DISCRETION`, single `one kilowatt`

## In Progress / Half Done
- Deploy/restart **server** so production picks up `library_document_service` merge fix

## Next Steps (Do This First When You Return)
1. Deploy/restart remote API with latest `app/services/library_document_service.py`
2. Re-open Resources for 2000J8 and confirm text does not start with `erson`

## Known Issues / Blockers
- Server on old merge code shows overlapping child windows (mid-word `erson…`) as the file
- Docker Desktop must be running for local Qdrant

## Key Decisions & Context
- Child chunks are sliding windows; never return a single mid-doc child as the file
- Stitch chunk-by-chunk with overlap; parent is a candidate; longest clean start wins
- Corpus ids are hex hashes (e.g. `a4fdab6fd0bcdf55`), not Postgres UUIDs
- Local stack: UI `:5173` → LegalGPT `:8001` → Gemini + Ollama embed + Qdrant `:6333`
