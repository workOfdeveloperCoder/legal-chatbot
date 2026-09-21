# Project Progress Log

## Last Updated
Tuesday, Sep 22, 2026 — ~12:46 AM (UTC+5)

## Current State
- Chat LLM: **Gemini** `gemini-3.6-flash` (live E2E PASS)
- Backend: `http://127.0.0.1:8001` (uvicorn --reload)
- Frontend: `http://127.0.0.1:5173` (Vite); `API_PROXY_TARGET=http://127.0.0.1:8001`
- Qdrant: local Docker on `:6333` (collections present)
- `:8000` still occupied by unrelated **Site Chatbot / dental-chatbot-v2** — do not bind LegalGPT there
- Voice: STT faster-whisper + TTS piper ready; speak E2E PASS

## What Was Done This Session
- Switched chat to Gemini; verified live generation
- Started Docker Desktop + existing `qdrant` container
- Pointed legal-ai-ui proxy to local `:8001`
- Started UI + backend; E2E via Vite proxy: login → chat/stream → speak
- Login user `yasir@example.com` succeeded
- Report: `tmp_test/gemini_ui_e2e_report.json`

## In Progress / Half Done
- Chat answered Section 54-C but noted no corpus hit (general guidance path) — embeddings/Qdrant retrieval may need a follow-up query check

## Next Steps (Do This First When You Return)
1. Open UI at http://127.0.0.1:5173 and chat as yasir@example.com
2. If legal Resources are empty, diagnose Ollama nomic embed → Qdrant `legal_documents` hits
3. Keep LegalGPT on **8001**; leave Site Chatbot on **8000**

## Known Issues / Blockers
- Docker Desktop must be running for local Qdrant
- Leaving `API_PROXY_TARGET` on a dead LAN host breaks the UI proxy

## Key Decisions & Context
- Local stack: UI `:5173` → proxy → LegalGPT `:8001` → Gemini + Ollama embed + Qdrant `:6333`
- Speak is local (Piper/espeak), independent of Gemini
