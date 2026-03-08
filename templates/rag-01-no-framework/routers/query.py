import logging
import uuid

from fastapi import APIRouter, HTTPException

from config import settings
from models import IngestResponse, IngestURLRequest, QueryRequest, QueryResponse, SourceInfo
from services import chunker, embeddings, keyword_search, loaders, rag, vector_store

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query_documents(req: QueryRequest):
    """Ask a question using RAG."""
    try:
        result = rag.ask(question=req.question, top_k=req.top_k, search_mode=req.search_mode)
        return QueryResponse(
            answer=result["answer"],
            sources=[SourceInfo(**s) for s in result["sources"]],
        )
    except Exception:
        logger.exception("Query failed for: %s", req.question[:100])
        raise HTTPException(status_code=500, detail="Failed to process query")


@router.post("/ingest-url", response_model=IngestResponse)
async def ingest_url(req: IngestURLRequest):
    """Ingest content from a URL."""
    try:
        text, metadata = loaders.load_url(req.url)

        if not text.strip():
            raise HTTPException(status_code=400, detail="No text content found at URL")

        doc_id = str(uuid.uuid4())

        chunks = chunker.split_text(
            text,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

        all_embeddings = []
        for i in range(0, len(chunks), 100):
            batch = chunks[i : i + 100]
            all_embeddings.extend(embeddings.embed_texts(batch))

        chunk_count = vector_store.add_document(
            doc_id=doc_id,
            chunks=chunks,
            embeddings=all_embeddings,
            metadata=metadata,
        )

        keyword_search.build_index()
        logger.info("Ingested URL '%s' (%d chunks) as %s", req.url, chunk_count, doc_id)
        return IngestResponse(
            document_id=doc_id,
            filename=metadata.get("filename", req.url),
            chunk_count=chunk_count,
            message="URL content ingested successfully",
        )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("Failed to ingest URL: %s", req.url)
        raise HTTPException(status_code=500, detail="Failed to ingest URL")
