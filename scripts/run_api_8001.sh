#!/usr/bin/env bash
# Start LegalGPT API on :8001 (avoids legal-gpt on :8000).
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/uvicorn app.main:app --host=127.0.0.1 --port=8001 --reload
