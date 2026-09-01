# Production deployment (no Docker)

This application is deployed as a Linux service behind Nginx on Ubuntu 24.04
with CloudPanel. Qdrant, PostgreSQL, and Ollama stay on the private host
network. Do not publish Qdrant port 6333, Ollama port 11434, or Uvicorn.

Architecture:

```
Browser
  → Nginx / HTTPS
    → FastAPI (127.0.0.1:8000, 1 worker)
      ├── PostgreSQL
      ├── Qdrant (127.0.0.1:6333, API key required)
      └── Ollama (127.0.0.1:11434, nomic-embed-text 768-d)
FastAPI → hosted Gemini (chat only)
```

Do not re-index `legal_documents`. Do not change vector size 768 or Cosine
distance. Do not expose Uvicorn on a public interface.

## 1. Linux packages

```bash
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3-pip \
  postgresql nginx certbot python3-certbot-nginx \
  build-essential libpq-dev
```

Install Qdrant from the official binary or systemd unit:
https://qdrant.tech/documentation/guides/installation/

Persist `/var/lib/qdrant/storage`. Bind Qdrant to `127.0.0.1:6333`.

```yaml
# /etc/qdrant/config.yaml (excerpt)
service:
  host: 127.0.0.1
  http_port: 6333
  api_key: "REPLACE_WITH_LONG_RANDOM_KEY"
storage:
  storage_path: /var/lib/qdrant/storage
```

```bash
sudo systemctl enable --now qdrant
```

## 2. Python application

```bash
sudo useradd -r -m -d /opt/legal-chatbot legalbot
sudo -u legalbot python3.12 -m venv /opt/legal-chatbot/venv
# Clone this repository to /opt/legal-chatbot/app
sudo -u legalbot /opt/legal-chatbot/venv/bin/pip install -r /opt/legal-chatbot/app/requirements.txt
```

`requirements.txt` is pinned from the development venv. Voice extras
(`faster-whisper`, `piper-tts`) are included; omit them only if you
disable the voice router.

```bash
cd /opt/legal-chatbot/app
sudo -u legalbot /opt/legal-chatbot/venv/bin/alembic upgrade head
sudo mkdir -p /var/lib/legal-chatbot/documents
sudo chown legalbot:legalbot /var/lib/legal-chatbot/documents
sudo chmod 750 /var/lib/legal-chatbot/documents
```

## 3. Environment (`/etc/legal-chatbot.env`)

Copy `.env.example`. Required production values:

- `ENVIRONMENT=production`
- `DEBUG=false`
- `TRUSTED_HOSTS=api.your-firm.tld`
- `CORS_ORIGINS=https://app.your-firm.tld`
- `QDRANT_URL=http://127.0.0.1:6333`
- `QDRANT_API_KEY=...` (required; startup fails without it)
- `LOG_ENABLE_RESPONSE_BODY=false`
- `LOG_STORE_CHAT_BODIES=false`
- Hosted chat: `LLM_PROVIDER=gemini`, `CHAT_MODEL=gemini-3.6-flash`,
  `LLM_API_KEY=...`, `LLM_TIMEOUT=120`
- Local chat: `LLM_PROVIDER=ollama`, `OLLAMA_MODEL=deepseek-r1:32b`
  (or `CHAT_MODEL=deepseek-r1:32b`), `LLM_TIMEOUT=3600`
- Temporary CPU-server testing: `LLM_PROVIDER=openrouter`,
  `OPENROUTER_API_KEY=...`, `OPENROUTER_BASE_URL=https://openrouter.ai/api/v1`,
  `OPENROUTER_MODEL=openrouter/free`, `LLM_TIMEOUT=120`. Switch to a paid
  OpenRouter model by changing `OPENROUTER_MODEL` only.
- **Server without local models** (no Ollama, no HF/fastembed download):
  `EMBEDDING_ONLINE_ONLY=true`, `EMBEDDING_PROVIDER=fireworks`,
  `EMBEDDING_MODEL=nomic-ai/nomic-embed-text-v1.5`, `FIREWORKS_API_KEY=...`
  (768-d cloud API; matches existing `legal_documents` index). Chat stays on
  OpenRouter or another online `LLM_PROVIDER`. Do **not** use
  `EMBEDDING_PROVIDER=huggingface` on that host — it downloads the ONNX model.
- Embeddings with local Ollama (legacy): `EMBEDDING_PROVIDER=ollama`,
  `EMBEDDING_MODEL=nomic-embed-text:latest`, `EMBEDDING_URL=http://127.0.0.1:11434`
- `TRUST_FORWARDED_FOR=true` only because Uvicorn binds to `127.0.0.1`
- `UPLOAD_DIR=/var/lib/legal-chatbot/documents`
- Optional internet search: `WEB_SEARCH_ENABLED=true`,
  `WEB_SEARCH_PROVIDER=auto`. Prefer `TAVILY_API_KEY` or `BRAVE_API_KEY`;
  without a key the app falls back to DuckDuckGo HTML search.

Do not set `TRUSTED_HOSTS=*`. Do not put real secrets in git.

Local development should keep `ENVIRONMENT=development` so a missing Qdrant
API key does not block the app.

## 4. Restore legal corpus (do not re-index)

`legal_documents` is owned by the Legal GPT ingestion project. This app never
creates or recreates it. The normal recovery path is Qdrant snapshot restore,
not embedding the corpus again.

Qdrant recover requires a URL or `file://` URI, not a bare filename.

### 4.1 Create snapshot (source host)

```bash
export QDRANT_URL=http://127.0.0.1:6333
export QDRANT_API_KEY=...
cd /opt/legal-chatbot/app
python scripts/qdrant_ops.py snapshot legal_documents
python scripts/qdrant_ops.py list-snapshots legal_documents
```

Note the snapshot name from the JSON (`name` field, often ending in `.snapshot`).
On a default Qdrant data dir the file lives under:

```text
/var/lib/qdrant/storage/snapshots/legal_documents/<SNAPSHOT_NAME>
```

### 4.2 Transfer snapshot

```bash
scp /var/lib/qdrant/storage/snapshots/legal_documents/<SNAPSHOT_NAME> \
  production:/var/lib/qdrant/storage/snapshots/legal_documents/
```

Keep the file owned by the Qdrant service user.

### 4.3 Restore snapshot (production host)

Prefer `file://` when the snapshot is already on the production Qdrant host:

```bash
export QDRANT_URL=http://127.0.0.1:6333
export QDRANT_API_KEY=...
python scripts/qdrant_ops.py recover legal_documents \
  file:///var/lib/qdrant/storage/snapshots/legal_documents/<SNAPSHOT_NAME>
```

If recovering from this node's own snapshot API:

```bash
python scripts/qdrant_ops.py recover legal_documents <SNAPSHOT_NAME>
```

That expands to:

```text
http://127.0.0.1:6333/collections/legal_documents/snapshots/<SNAPSHOT_NAME>
```

### 4.4 Verify legal_documents

```bash
python scripts/qdrant_ops.py verify legal_documents --vector-size 768 --payload
```

Confirm:

1. Collection exists
2. `vector_size` is `768`
3. `points_count` is greater than 0 and matches the source
4. `distance` is Cosine (reported by Qdrant as `Cosine`)
5. `sample_payload_keys` is a non-empty legal-corpus payload (not user-upload fields)

Do not recreate the collection if verification fails. Restore again from a
known-good snapshot.

Also snapshot private collections after go-live:

```bash
python scripts/qdrant_ops.py snapshot chatbot_documents
python scripts/qdrant_ops.py snapshot user_memory
```

## 5. systemd

Template: `deploy/legal-chatbot.service`

```bash
sudo cp deploy/legal-chatbot.service /etc/systemd/system/legal-chatbot.service
sudo systemctl daemon-reload
sudo systemctl enable --now legal-chatbot
sudo systemctl status legal-chatbot
```

Uvicorn must listen on `127.0.0.1:8000` with **one worker**. The in-process
rate limiter does not share state across workers. CloudPanel does not start
FastAPI for you; this unit does.

Ollama embeddings (same host, loopback only):

```bash
sudo systemctl enable --now ollama
ollama pull nomic-embed-text
```

Local DeepSeek chat remains supported for offline development
(`LLM_PROVIDER=ollama`, `OLLAMA_MODEL=deepseek-r1:32b` or
`CHAT_MODEL=deepseek-r1:32b`, `LLM_TIMEOUT=3600`).
Do not use that 3600-second timeout for hosted Gemini. R1 32B often
thinks for many minutes before the first answer token; keep Nginx
`proxy_read_timeout` at least as high, and prefer `/chat/stream` so
SSE keepalives hold the connection.

## 6. Nginx + HTTPS (CloudPanel)

Terminate TLS on Nginx. Proxy to loopback only. Enable WebSocket/SSE by
disabling buffering.

```nginx
server {
    listen 443 ssl http2;
    server_name api.your-firm.tld;
    client_max_body_size 32m;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_buffering off;
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;
    }
}
```

```bash
sudo certbot --nginx -d api.your-firm.tld
```

Because Uvicorn is not public, set `TRUST_FORWARDED_FOR=true` so rate limits
use the Nginx client IP. If the API port is ever reachable directly, leave
`TRUST_FORWARDED_FOR=false` or clients can spoof `X-Forwarded-For`.

## 7. Host firewall

```bash
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw deny 6333/tcp
sudo ufw deny 8000/tcp
sudo ufw deny 11434/tcp
sudo ufw enable
```

## 8. Backups

A backup is not complete until you restore onto a staging instance.

### PostgreSQL

```bash
sudo -u postgres mkdir -p /var/backups/legalchatbot
sudo -u postgres pg_dump -Fc legalchatbot \
  > /var/backups/legalchatbot/$(date +%F).dump
```

Restore:

```bash
sudo -u postgres pg_restore --clean --if-exists -d legalchatbot \
  /var/backups/legalchatbot/YYYY-MM-DD.dump
cd /opt/legal-chatbot/app
sudo -u legalbot /opt/legal-chatbot/venv/bin/alembic upgrade head
```

### Qdrant

`legal_documents` **must** be backed up. Do not treat re-embedding the legal
corpus as the backup strategy.

```bash
export QDRANT_URL=http://127.0.0.1:6333 QDRANT_API_KEY=...
python scripts/qdrant_ops.py snapshot legal_documents
python scripts/qdrant_ops.py snapshot chatbot_documents
python scripts/qdrant_ops.py snapshot user_memory
# copy snapshot files off-box, then on restore:
python scripts/qdrant_ops.py recover legal_documents \
  file:///path/to/legal_documents.snapshot
python scripts/qdrant_ops.py verify legal_documents --vector-size 768 --payload
```

### Uploads

```bash
rsync -a /var/lib/legal-chatbot/documents/ backup-host:/backups/legal-chatbot/documents/
```

## 9. Health

```bash
curl -sS https://api.your-firm.tld/health
curl -sS https://api.your-firm.tld/ready
```

`/health` is process liveness only.
`/ready` checks PostgreSQL, Qdrant, and the Ollama embedding daemon (`/api/tags`).
It does not call Gemini. Gemini is validated at startup when
`ENVIRONMENT=production` and `LLM_PROVIDER` is a hosted provider (API key
required). `GET /api/v1/llm/status` reports `tokenizer_exact` (false for Gemini
cl100k approximation and for the fallback estimator).

## 10. Chat streaming

`POST /api/v1/chat/stream` is SSE (`event: started|token|complete|error`).
Set `LLM_ENABLE_STREAMING=true`. Closing the client aborts generation.

## 11. Rate limiting

Limits are in-process and IP-based. Use one Uvicorn worker. Multiple workers
each keep their own counters, so effective limits multiply. Redis is not
required for the initial single-worker deployment.

## 12. Tokenizer note

Gemini token counts use the `cl100k_base` approximation. They are not a native
Gemini tokenizer. If tiktoken cannot load, `/api/v1/llm/status` reports
`tokenizer_exact: false` and a fallback tokenizer id.
