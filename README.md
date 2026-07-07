# LangChain_Rag

FastAPI RAG Agent with:

- OpenAI-compatible chat model from `.env`
- Local Qwen embedding model via `sentence-transformers`
- Local Qwen3 rerank model via `transformers`
- PostgreSQL + pgvector persistence

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
- `DASHSCOPE_API_KEY` when using DashScope embedding or rerank providers
- `POSTGRES_DSN`, `PGVECTOR_TABLE`

APIs:

- `GET /health`
- `POST /documents` multipart file upload for `.txt`, `.md`, `.pdf`, `.docx`
- `POST /chat` JSON body: `{"message": "...", "top_k": 8, "rerank_top_n": 4}`
