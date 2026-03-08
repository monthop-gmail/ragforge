import chromadb

from config import settings

_client = chromadb.PersistentClient(path=settings.chroma_path)
_collection = _client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"},
)

BATCH_SIZE = 100


def add_document(
    doc_id: str,
    chunks: list[str],
    embeddings: list[list[float]],
    metadata: dict,
) -> int:
    """Add document chunks to vector store. Returns chunk count."""
    ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "document_id": doc_id,
            "chunk_index": i,
            **metadata,
        }
        for i in range(len(chunks))
    ]

    # Batch upsert
    for start in range(0, len(chunks), BATCH_SIZE):
        end = start + BATCH_SIZE
        _collection.upsert(
            ids=ids[start:end],
            documents=chunks[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
        )

    return len(chunks)


def query(query_embedding: list[float], top_k: int = 5) -> list[dict]:
    """Query similar chunks. Returns list of results with scores."""
    results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    items = []
    if results["ids"] and results["ids"][0]:
        for i, chunk_id in enumerate(results["ids"][0]):
            items.append(
                {
                    "id": chunk_id,
                    "chunk_text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": results["distances"][0][i] if results["distances"] else None,
                }
            )
    return items


def get_all_chunks() -> list[dict]:
    """Get all chunks from the store for BM25 indexing."""
    all_data = _collection.get()
    if not all_data["ids"]:
        return []
    chunks = []
    for i, chunk_id in enumerate(all_data["ids"]):
        chunks.append({
            "id": chunk_id,
            "chunk_text": all_data["documents"][i],
            "metadata": all_data["metadatas"][i],
        })
    return chunks


def list_documents() -> list[dict]:
    """List all unique documents in the store."""
    all_data = _collection.get()
    if not all_data["ids"]:
        return []

    docs = {}
    for i, meta in enumerate(all_data["metadatas"]):
        doc_id = meta.get("document_id", "unknown")
        if doc_id not in docs:
            docs[doc_id] = {
                "id": doc_id,
                "filename": meta.get("filename", "unknown"),
                "source_type": meta.get("source_type", "unknown"),
                "chunk_count": 0,
            }
        docs[doc_id]["chunk_count"] += 1

    return list(docs.values())


def delete_document(doc_id: str) -> bool:
    """Delete all chunks for a document. Returns True if any deleted."""
    results = _collection.get(where={"document_id": doc_id})
    if not results["ids"]:
        return False
    _collection.delete(ids=results["ids"])
    return True
