---
# Project Progress Log

## Last Updated
Wednesday, Aug 19, 2026 — 7:40 PM (UTC+5)

## Current State
- Production Phase 1 chat core remains in place (Ollama `deepseek-r1:32b`, Qdrant RAG)
- Automatic language detection is part of the RAG chat path on text transcripts
- Supported user languages: English (Pakistan), Urdu (Pakistan), Pakistani Punjabi
- Mixed/code-switched input and Roman Urdu are handled without a language picker
- `POST /api/v1/chat` is unchanged (text-only RAG)
- Voice Mode is **local/self-hosted**: faster-whisper STT + Piper/espeak-ng TTS
- No OpenAI Realtime, no `OPENAI_API_KEY`, no paid speech APIs
- Spoken legal answers still go through existing `/api/v1/chat` from the frontend
- **Voice tests: 5 passed**; language tests still passing (35 total with language suite)
- **735 bulk ingest not started** (awaiting legal-gpt completed/ migration)

## What Was Done This Session
- Inspected chat API, auth, language detection, RAG, frontend chat UI, Python 3.14.6, Apple M1 Pro 32GB, no Docker
- Removed OpenAI Realtime session minting (`voice_session_service`, `OPENAI_*` settings)
- Added local `GET /api/v1/voice/status`, `POST /api/v1/voice/transcribe`, `POST /api/v1/voice/speak`
- STT language is metadata only; `/chat` still runs `detect_language()` on the transcript
- TTS voice selection reuses `detect_language()` on the **answer** text (espeak `ur`/`pa` for Urdu/Punjabi)
- Frontend Voice Mode rewritten: mic VAD → WAV → local STT → existing `/chat` → local TTS → playback, with barge-in
- Frontend Voice Mode is now on legal-ai-ui `feature/waqar` (mic on PromptBar; Discard / Pause / Send)
- Document upload endpoints are unchanged and still used by legal-ai-ui

## In Progress / Half Done
- Local STT/TTS Python packages are installed (`faster-whisper` 1.2.1, `piper-tts`); Piper English voice and espeak-ng are present. Restart FastAPI so `/voice/status` picks them up.

## Next Steps (Do This First When You Return)
1. Finish voice deps if needed: `.venv/bin/pip install -r requirements-voice.txt`
2. `python scripts/setup_voice.py --download-piper-en`
3. `brew install espeak-ng` (Urdu/Punjabi TTS)
4. Restart FastAPI; in legal-ai-ui click the mic and test English/Urdu/Punjabi
5. Confirm chat paperclip still uploads via `/documents/conversations/{id}/upload` and matter Upload via `/documents/matters/{id}/upload`
6. After legal-gpt migrates more completed docs, re-test `/chat` on Section 54-C / PIL queries

## Known Issues / Blockers
- Most legal_documents points still pre-v4 (parent expansion no-ops)
- Claim grounding remains lexical
- Voice Mode returns 503 until faster-whisper and Piper/espeak-ng are installed
- Whisper multilingual `base` can miss Roman Urdu / mixed Punjabi; `STT_MODEL=small` is more accurate but slower
- Official Piper voices are weak for Urdu/Punjabi; espeak-ng is the local fallback
- Local 32B LLM (`LLM_TIMEOUT=1800`) means Voice Mode “thinking” can take minutes
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
- Chatbot language detection stays on the transcript after STT, not the STT engine’s language code
- Voice is a speech interface only; RAG remains the legal answer source
- No OpenAI / Deepgram / cloud speech API keys
- Default STT: `base` on CPU int8 (M1 Pro has no CUDA; do not auto-download large models)
---
