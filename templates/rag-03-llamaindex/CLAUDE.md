# RAG Server - LlamaIndex

## Overview
RAG server built with LlamaIndex framework. Uses VectorStoreIndex, query engine, and LlamaIndex readers for the most abstracted approach.

## Tech Stack
- **API**: FastAPI (port 8000)
- **Framework**: LlamaIndex (llama-index-core + integrations)
- **LLM**: llama-index-llms-openai (gpt-4o-mini default)
- **Embedding**: llama-index-embeddings-openai (text-embedding-3-small)
- **Vector DB**: llama-index-vector-stores-chroma (cosine distance)
- **PDF**: llama-index-readers-file (PyMuPDFReader)
- **Web scraping**: llama-index-readers-web (SimpleWebPageReader)
- **MCP**: FastMCP SSE server (port 8001)

## Project Structure
```
├── main.py                 # FastAPI entrypoint
├── config.py               # Settings from .env (pydantic-settings)
├── models.py               # Pydantic request/response schemas
├── mcp_server.py           # MCP SSE server wrapping RAG API
├── routers/
│   ├── documents.py        # POST /upload, GET /documents, DELETE /documents/{id}
│   └── query.py            # POST /query, POST /ingest-url
├── services/
│   ├── index.py            # LlamaIndex Settings, ChromaVectorStore, VectorStoreIndex
│   ├── loaders.py          # LlamaIndex readers (PyMuPDFReader, SimpleWebPageReader)
│   ├── vector_store.py     # ChromaDB direct access for list/delete/get_all operations
│   ├── keyword_search.py   # BM25 keyword search (rank-bm25)
│   ├── hybrid_search.py    # Reciprocal Rank Fusion (RRF) merger
│   └── rag.py              # Hybrid retrieval (vector/keyword/hybrid) + LLM generation
├── docker-compose.yml      # rag-server + mcp-server
└── .env                    # OPENAI_API_KEY and settings
```

## API Endpoints
- `POST /upload` — Upload PDF/TXT/MD file
- `POST /ingest-url` — Ingest web page by URL
- `POST /query` — Ask a question (RAG) with search_mode: vector/keyword/hybrid
- `GET /documents` — List all documents
- `DELETE /documents/{id}` — Delete a document
- `GET /health` — Health check

## MCP Tools (port 8001)
- `rag_query` — Ask a question (supports search_mode: vector/keyword/hybrid)
- `rag_upload_text` — Upload text content
- `rag_ingest_url` — Ingest from URL
- `rag_list_documents` — List documents
- `rag_delete_document` — Delete document

## Configuration (.env)
- `OPENAI_API_KEY` — Required
- `OPENAI_CHAT_MODEL` — Default: gpt-4o-mini
- `OPENAI_EMBEDDING_MODEL` — Default: text-embedding-3-small
- `CHUNK_SIZE` — Default: 1000
- `CHUNK_OVERLAP` — Default: 200
- `TOP_K` — Default: 5
- `RRF_K` — RRF constant for hybrid search. Default: 60

## Running
```bash
# Docker (recommended)
docker compose up --build

# Local
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
python mcp_server.py  # separate terminal

# MCP for Claude Code
claude mcp add ragforge --transport sse http://localhost:8001/sse
```

## Hybrid Search
- **Vector search**: Cosine similarity via ChromaDB HNSW index (semantic matching)
- **Keyword search**: BM25 via rank-bm25 library (exact term matching)
- **Hybrid search** (default): Runs both, merges results using Reciprocal Rank Fusion (RRF)
- BM25 index is built on startup and rebuilt after each ingest/delete
- Query API accepts `search_mode`: `"vector"`, `"keyword"`, or `"hybrid"`

## Key Design Decisions
- Global LlamaIndex `Settings` configures LLM, embed model, and node parser once
- `services/index.py` is the central module managing ChromaVectorStore and StorageContext
- Ingestion uses `VectorStoreIndex.from_documents()` which handles chunking + embedding + storing
- Retrieval decoupled from generation: uses `index.as_retriever()` + `LlamaSettings.llm.complete()` to support hybrid search modes
- List/delete operations go through ChromaDB collection directly (LlamaIndex has no built-in delete-by-metadata)
- MCP server is a separate process that calls RAG API via HTTP internally
