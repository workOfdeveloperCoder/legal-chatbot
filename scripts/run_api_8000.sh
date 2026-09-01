#!/usr/bin/env bash
# LegalGPT chat API on :8000 (stop legal-gpt on 8000 first if both bind).
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/uvicorn app.main:app --host=127.0.0.1 --port=8000 --reload
