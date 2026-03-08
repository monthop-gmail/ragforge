from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from config import settings

_embeddings = OpenAIEmbeddings(
    model=settings.openai_embedding_model,
    openai_api_key=settings.openai_api_key,
)

_vectorstore = Chroma(
    collection_name="documents",
    embedding_function=_embeddings,
    persist_directory=settings.chroma_path,
    collection_metadata={"hnsw:space": "cosine"},
)


def add_document(doc_id: str, chunks: list[Document], metadata: dict) -> int:
    """Add document chunks to vector store. Returns chunk count."""
    ids = []
    for i, chunk in enumerate(chunks):
        chunk.metadata.update(metadata)
        chunk.metadata["document_id"] = doc_id
        chunk.metadata["chunk_index"] = i
        ids.append(f"{doc_id}_chunk_{i}")

    _vectorstore.add_documents(documents=chunks, ids=ids)
    return len(chunks)


def query(query_text: str, top_k: int = 5) -> list[dict]:
    """Query similar chunks with scores."""
    results = _vectorstore.similarity_search_with_score(query_text, k=top_k)

    items = []
    for doc, score in results:
        items.append(
            {
                "id": doc.metadata.get("document_id", "") + "_chunk_" + str(doc.metadata.get("chunk_index", 0)),
                "chunk_text": doc.page_content,
                "metadata": doc.metadata,
                "score": score,
            }
        )
    return items


def get_all_chunks() -> list[dict]:
    """Get all chunks from the store for BM25 indexing."""
    collection = _vectorstore._collection
    all_data = collection.get()
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
    collection = _vectorstore._collection
    all_data = collection.get()
    if not all_data["ids"]:
        return []

    docs = {}
    for meta in all_data["metadatas"]:
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
    """Delete all chunks for a document."""
    collection = _vectorstore._collection
    results = collection.get(where={"document_id": doc_id})
    if not results["ids"]:
        return False
    collection.delete(ids=results["ids"])
    return True
