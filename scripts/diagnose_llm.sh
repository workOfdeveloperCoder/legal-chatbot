#!/usr/bin/env bash
# Diagnose local LegalGPT chat stack: Ollama R1 + legal-chatbot + FE proxy target.
set -euo pipefail

echo "=== 1) Ollama ==="
curl -sS -m 5 http://127.0.0.1:11434/api/version || { echo "FAIL: Ollama not running"; exit 1; }
echo
echo "Loaded models:"
curl -sS -m 5 http://127.0.0.1:11434/api/ps
echo
echo "Generate smoke (30s max)..."
GEN=$(curl -sS -m 30 http://127.0.0.1:11434/api/generate \
  -d '{"model":"deepseek-r1:32b","prompt":"Say OK","stream":false,"options":{"num_predict":6}}' \
  -w '|HTTP:%{http_code}|TIME:%{time_total}' || true)
if [[ "$GEN" == *'|HTTP:200|'* ]]; then
  echo "PASS: generate HTTP 200"
else
  echo "FAIL: generate hung or failed (common cause of UI 503)"
  echo "$GEN" | tail -c 300
  echo
  echo "Fix: brew services restart ollama"
  exit 2
fi

echo
echo "=== 2) Port 8000 owners ==="
lsof -nP -iTCP:8000 -sTCP:LISTEN 2>/dev/null || echo "(none listening)"
echo
echo "127.0.0.1:8000 openapi title:"
curl -sS -m 3 http://127.0.0.1:8000/openapi.json 2>/dev/null \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('info',{}).get('title'), 'paths', len(d.get('paths',{})))" \
  2>/dev/null || echo "(no openapi)"
LAN=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || true)
if [[ -n "${LAN:-}" ]]; then
  echo "LAN $LAN:8000 openapi title:"
  curl -sS -m 3 "http://$LAN:8000/openapi.json" 2>/dev/null \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('info',{}).get('title'), 'chat', any('/chat' in p for p in d.get('paths',{})))" \
    2>/dev/null || echo "(no openapi on LAN)"
fi

echo
echo "=== 3) Gateway stream (legal-chatbot → Ollama) ==="
cd "$(dirname "$0")/.."
.venv/bin/python - <<'PY'
import asyncio
from app.llm.factory import LLMFactory
from app.schemas.llm import ChatCompletionRequest, ChatMessage

async def main():
    llm = LLMFactory.create()
    req = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="Reply with exactly: DIAG_OK")],
        temperature=0.1,
        max_tokens=128,
    )
    parts = []
    async for piece in llm.generate_stream(req):
        kind = getattr(piece, "kind", "token")
        text = getattr(piece, "text", "") or ""
        if kind != "thinking":
            parts.append(text)
    out = "".join(parts).strip()
    print("adapter", type(llm._primary).__name__, llm._primary.model_name)
    print("stream_out", repr(out[:120]), "len", len(out))
    if "DIAG_OK" not in out and not out:
        raise SystemExit("FAIL: empty stream content")
    await llm.aclose()

asyncio.run(main())
print("PASS: gateway stream")
PY

echo
echo "=== Summary ==="
echo "If generate/stream FAIL → UI shows 'The assistant is temporarily unavailable.'"
echo "If 127.0.0.1 openapi is 'Legal AI OS' (no chat) but FE uses localhost → wrong app."
echo "FE should proxy to LegalGPT (LAN :8000) via API_PROXY_TARGET, not Legal AI OS."
