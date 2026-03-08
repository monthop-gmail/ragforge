import logging
import uuid

from fastapi import APIRouter, HTTPException
from llama_index.core import VectorStoreIndex

from config import settings
from models import IngestResponse, IngestURLRequest, QueryRequest, QueryResponse, SourceInfo
from services import loaders, rag, vector_store
from services.index import get_storage_context

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query_documents(req: QueryRequest):
    """Ask a question using RAG."""
    try:
        result = rag.ask(question=req.question, top_k=req.top_k)
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
        docs, metadata = loaders.load_url(req.url)

        if not docs:
            raise HTTPException(status_code=400, detail="No text content found at URL")

        doc_id = str(uuid.uuid4())

        for doc in docs:
            doc.metadata.update(metadata)
            doc.metadata["document_id"] = doc_id

        storage_context = get_storage_context()
        VectorStoreIndex.from_documents(
            documents=docs,
            storage_context=storage_context,
        )

        logger.info("Ingested URL '%s' (%d docs) as %s", req.url, len(docs), doc_id)
        return IngestResponse(
            document_id=doc_id,
            filename=metadata.get("filename", req.url),
            chunk_count=len(docs),
            message="URL content ingested successfully",
        )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("Failed to ingest URL: %s", req.url)
        raise HTTPException(status_code=500, detail="Failed to ingest URL")
