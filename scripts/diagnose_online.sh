#!/usr/bin/env bash
# Verify fully-online chat + Qdrant search stack (OpenRouter + Fireworks, etc.).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

API="${API_BASE:-http://127.0.0.1:8000}"

echo "=== Online stack (/ready) ==="
READY=$(curl -sS -m 10 "$API/ready" || true)
echo "$READY" | python3 - <<'PY'
import json, sys
raw = sys.stdin.read().strip()
if not raw:
    print("FAIL: API not reachable at", "$API")
    sys.exit(1)
d = json.loads(raw)
stack = d.get("stack") or {}
chat = stack.get("chat") or {}
search = stack.get("search") or {}
qdrant = stack.get("qdrant") or {}
print("status:", d.get("status"))
print("chat  :", chat.get("provider"), chat.get("model"), "online=", chat.get("online"), "key=", chat.get("api_key_configured"))
print("search:", search.get("provider"), search.get("model"), "online=", search.get("online"), "key=", search.get("api_key_configured"))
print("qdrant:", qdrant.get("endpoint"), "collection=", qdrant.get("collection"), "ok=", qdrant.get("ok"))
for err in d.get("errors") or []:
    print("error :", err)
if d.get("status") != "ready":
    sys.exit(2)
PY

echo
echo "=== Embed + Qdrant smoke ==="
PYTHONPATH=. .venv/bin/python - <<'PY'
import asyncio
from app.embeddings.factory import EmbeddingFactory
from app.vector.qdrant import qdrant_service

async def main():
    emb = EmbeddingFactory.create()
    vector = await emb.embed_query("section 417 PPC")
    print("embedding dim:", len(vector))
    result = await qdrant_service.client.query_points(
        collection_name=qdrant_service.legal_collection,
        query=vector,
        limit=2,
        with_payload=True,
    )
    print("qdrant hits:", len(result.points))
    for point in result.points:
        payload = point.payload or {}
        title = payload.get("title") or payload.get("source") or "?"
        print(f"  score={point.score:.3f} {str(title)[:60]}")
    close = getattr(emb, "aclose", None)
    if callable(close):
        await close()

asyncio.run(main())
print("PASS: online search pipeline")
PY

echo
echo "=== Chat stream smoke ==="
PYTHONPATH=. .venv/bin/python - <<'PY'
import asyncio
from app.llm.factory import LLMFactory
from app.schemas.llm import ChatCompletionRequest, ChatMessage

async def main():
    llm = LLMFactory.create()
    req = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="Reply with exactly: ONLINE_OK")],
        temperature=0.1,
        max_tokens=64,
    )
    parts = []
    async for piece in llm.generate_stream(req):
        if getattr(piece, "kind", "token") != "thinking":
            parts.append(getattr(piece, "text", "") or "")
    out = "".join(parts).strip()
    print("model:", llm._primary.model_name)
    print("out:", repr(out[:80]))
    if not out:
        raise SystemExit("FAIL: empty chat response")
    await llm.aclose()

asyncio.run(main())
print("PASS: online chat")
PY

echo
echo "=== Summary ==="
echo "Fully online server = OpenRouter (chat) + Fireworks (768-d embed API) + Qdrant."
echo "Do not use EMBEDDING_PROVIDER=huggingface on servers that cannot store models."
