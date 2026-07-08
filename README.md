# LangChain_Rag

FastAPI RAG Agent with:

- OpenAI-compatible chat model from `.env`
- Local Qwen embedding model via `sentence-transformers`
- Local Qwen3 rerank model via `transformers`
- PostgreSQL + pgvector persistence
- SQLAlchemy async ORM with asyncpg for files, chunks, conversations, and messages

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
uvicorn core.main:app --host 127.0.0.1 --port 8000
```

Required `.env` keys:

- `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`
- `LOCAL_EMBEDDING_MODEL_PATH`, `EMBEDDING_DIMENSION`
- `LOCAL_RERANK_MODEL_PATH` when `RERANK_TOP_N > 0` and `RERANK_PROVIDER=local`
- `POSTGRES_DSN` or `PG_HOST`, `PG_USER`, `PG_PWD`, `PG_DATABASE`
- `PGVECTOR_TABLE`, `RAG_FILE_TABLE`, `RAG_CHUNK_TABLE`
- `RAG_CONVERSATION_TABLE`, `RAG_MESSAGE_TABLE`, `CHAT_HISTORY_LIMIT`

APIs:

- `GET /health`
- `POST /documents` multipart form: `user_id` plus `.txt`, `.md`, `.pdf`, `.docx` file
- `GET /documents?user_id=...`
- `GET /documents/{file_id}?user_id=...`
- `DELETE /documents/{file_id}?user_id=...`
- `GET /conversations?user_id=...`
- `POST /chat` JSON body: `{"user_id": "...", "message": "...", "conversation_id": null, "top_k": 8, "rerank_top_n": 4}`

Document content is deduplicated per `user_id`. File records include `chunk_ids`, and chat retrieval is filtered by `user_id`.
