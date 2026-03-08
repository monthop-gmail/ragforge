import logging
import threading

from rank_bm25 import BM25Okapi

from services.vector_store import get_all_chunks

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_bm25: BM25Okapi | None = None
_chunks: list[dict] = []


def build_index() -> int:
    """Build BM25 index from all chunks in vector store."""
    global _bm25, _chunks

    all_chunks = get_all_chunks()
    if not all_chunks:
        with _lock:
            _bm25 = None
            _chunks = []
        return 0

    tokenized = [chunk["chunk_text"].lower().split() for chunk in all_chunks]

    with _lock:
        _bm25 = BM25Okapi(tokenized)
        _chunks = all_chunks

    logger.info("BM25 index built with %d chunks", len(all_chunks))
    return len(all_chunks)


def search(query: str, top_k: int = 5) -> list[dict]:
    """Search using BM25 keyword matching."""
    with _lock:
        bm25 = _bm25
        chunks = _chunks

    if bm25 is None or not chunks:
        return []

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    scored_indices = sorted(
        range(len(scores)), key=lambda i: scores[i], reverse=True
    )[:top_k]

    results = []
    for idx in scored_indices:
        if scores[idx] > 0:
            results.append({
                "id": chunks[idx]["id"],
                "chunk_text": chunks[idx]["chunk_text"],
                "metadata": chunks[idx]["metadata"],
                "score": float(scores[idx]),
            })
    return results
