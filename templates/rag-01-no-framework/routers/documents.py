import logging
import os
import uuid

from fastapi import APIRouter, HTTPException, UploadFile

from config import settings
from models import DeleteResponse, DocumentInfo, IngestResponse
from services import chunker, embeddings, keyword_search, loaders, vector_store

logger = logging.getLogger(__name__)
router = APIRouter()

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}
MAX_UPLOAD_BYTES = settings.max_upload_size_mb * 1024 * 1024


@router.post("/upload", response_model=IngestResponse)
async def upload_document(file: UploadFile):
    """Upload and ingest a document (PDF, TXT, MD)."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # Read with size limit
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size: {settings.max_upload_size_mb}MB",
        )

    doc_id = str(uuid.uuid4())
    os.makedirs(settings.upload_dir, exist_ok=True)
    file_path = os.path.join(settings.upload_dir, f"{doc_id}_{file.filename}")

    with open(file_path, "wb") as f:
        f.write(content)

    try:
        if ext == ".pdf":
            text, metadata = loaders.load_pdf(file_path)
        else:
            text, metadata = loaders.load_text(file_path)

        if not text.strip():
            raise HTTPException(status_code=400, detail="Document contains no text")

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
        logger.info("Ingested '%s' (%d chunks) as %s", file.filename, chunk_count, doc_id)
        return IngestResponse(
            document_id=doc_id,
            filename=file.filename,
            chunk_count=chunk_count,
            message="Document ingested successfully",
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to ingest document '%s'", file.filename)
        raise HTTPException(status_code=500, detail="Failed to process document")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


@router.get("/documents", response_model=list[DocumentInfo])
async def list_documents():
    """List all ingested documents."""
    docs = vector_store.list_documents()
    return [DocumentInfo(**d) for d in docs]


@router.delete("/documents/{doc_id}", response_model=DeleteResponse)
async def delete_document(doc_id: str):
    """Delete a document and all its chunks."""
    deleted = vector_store.delete_document(doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")
    keyword_search.build_index()
    logger.info("Deleted document %s", doc_id)
    return DeleteResponse(message=f"Document {doc_id} deleted successfully")
