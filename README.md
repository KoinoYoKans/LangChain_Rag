# LangChain_Rag

Enterprise knowledge-base Q&A service with FastAPI, React, pgvector, Redis-backed ingestion jobs, local Qwen3 embedding/rerank models, and an OpenAI-compatible chat model.

## What Is Included

- Account/password login with JWT.
- Organization, department, user, role, and knowledge-base ACLs.
- File ingestion for `.txt`, `.md`, `.pdf`, `.docx`.
- Web URL ingestion with HTML text extraction.
- Redis queue plus a dedicated worker for parsing, chunking, embedding, and pgvector writes.
- PostgreSQL tables for files, chunks, conversations, ingest jobs, audit logs, and chat logs.
- React + Vite management console for knowledge bases, ingestion, chat, users/departments, and audit logs.

## Local Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn core.main:app --host 127.0.0.1 --port 8000
```

Run the worker in another terminal:

```bash
source .venv/bin/activate
python worker.py
```

Run the frontend:

```bash
cd frontend
npm install
VITE_API_BASE=http://127.0.0.1:8000 npm run dev
```

## Required Configuration

The service reads `.env`.

- `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`
- `LOCAL_EMBEDDING_MODEL_PATH`, `EMBEDDING_DIMENSION`
- `LOCAL_RERANK_MODEL_PATH`, `RERANK_TOP_N`
- `POSTGRES_DSN`
- `REDIS_URL`
- `JWT_SECRET`
- `DEFAULT_ADMIN_EMAIL`, `DEFAULT_ADMIN_PASSWORD`

On startup, the API creates the default organization, department, and admin user if they do not exist.

## API Overview

- `POST /auth/login`
- `GET /auth/me`
- `GET/POST /departments`
- `GET/POST /users`
- `GET/POST /knowledge-bases`
- `POST /knowledge-bases/{id}/documents`
- `POST /knowledge-bases/{id}/urls`
- `GET /knowledge-bases/{id}/ingest-jobs`
- `GET /knowledge-bases/{id}/documents`
- `POST /chat`
- `GET /audit-logs`
- `GET /health`

`POST /chat` requires `knowledge_base_id` and uses the authenticated user's ACL to filter access.

## Docker Compose

```bash
docker compose up --build
```

The Compose file starts:

- `api` on port `8000`
- `worker`
- `frontend` on port `8080`

It expects PostgreSQL and Redis connection strings in `.env`. Model paths are mounted into the containers as `/models/embedding` and `/models/reranker`.

## Security Notes

- Replace `JWT_SECRET` and default admin password before exposing the service.
- Do not commit `.env`.
- PostgreSQL and Redis should not be open to `0.0.0.0/0` in production; restrict access to the API/worker host or private network.
