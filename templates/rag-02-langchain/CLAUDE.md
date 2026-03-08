# RAG Server - LangChain

## Overview
RAG server built with LangChain framework. Uses LCEL chains, LangChain loaders, text splitters, and vector store integrations.

## Tech Stack
- **API**: FastAPI (port 8000)
- **Framework**: LangChain + LangChain-OpenAI + LangChain-Chroma + LangChain-Community
- **LLM**: ChatOpenAI (gpt-4o-mini default)
- **Embedding**: OpenAIEmbeddings (text-embedding-3-small)
- **Vector DB**: Chroma via langchain-chroma (cosine distance)
- **PDF**: PyMuPDFLoader (via langchain-community)
- **Web scraping**: WebBaseLoader (via langchain-community)
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
│   ├── loaders.py          # LangChain loaders (PyMuPDFLoader, TextLoader, WebBaseLoader)
│   ├── vector_store.py     # LangChain Chroma wrapper (add, query, list, delete)
│   ├── keyword_search.py   # BM25 keyword search (rank-bm25)
│   ├── hybrid_search.py    # Reciprocal Rank Fusion (RRF) merger
│   └── rag.py              # LCEL chain with hybrid retrieval (vector/keyword/hybrid)
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
- Text splitting uses LangChain `RecursiveCharacterTextSplitter`
- Embedding is handled automatically by `langchain-chroma` (no manual embed calls)
- RAG pipeline uses LCEL: `ChatPromptTemplate | ChatOpenAI | StrOutputParser`
- Vector query uses `similarity_search_with_score()` for ranked results
- Document delete goes through ChromaDB collection directly (no LangChain abstraction for delete)
- MCP server is a separate process that calls RAG API via HTTP internally
