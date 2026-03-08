from openai import OpenAI

from config import settings
from models import SearchMode
from services.embeddings import embed_query
from services.hybrid_search import reciprocal_rank_fusion
from services.keyword_search import search as keyword_search
from services.vector_store import query as vector_query

_client = OpenAI(api_key=settings.openai_api_key)

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.
Use ONLY the context below to answer. If the context doesn't contain enough information, say so.
Do not make up information."""


def _retrieve(question: str, top_k: int, search_mode: SearchMode) -> list[dict]:
    """Retrieve relevant chunks using the specified search mode."""
    if search_mode == SearchMode.vector:
        query_embedding = embed_query(question)
        return vector_query(query_embedding, top_k)

    if search_mode == SearchMode.keyword:
        return keyword_search(question, top_k)

    # hybrid: run both and merge with RRF
    query_embedding = embed_query(question)
    vector_results = vector_query(query_embedding, top_k)
    keyword_results = keyword_search(question, top_k)
    return reciprocal_rank_fusion(vector_results, keyword_results)[:top_k]


def ask(question: str, top_k: int | None = None, search_mode: SearchMode = SearchMode.hybrid) -> dict:
    """Run the full RAG pipeline: retrieve -> generate."""
    k = top_k or settings.top_k

    results = _retrieve(question, k, search_mode)

    if not results:
        return {
            "answer": "No relevant documents found. Please upload some documents first.",
            "sources": [],
        }

    # Build context
    context_parts = []
    for i, r in enumerate(results, 1):
        filename = r["metadata"].get("filename", "unknown")
        context_parts.append(f"[Source {i}] ({filename}):\n{r['chunk_text']}")
    context = "\n\n".join(context_parts)

    # Call chat completion
    response = _client.chat.completions.create(
        model=settings.openai_chat_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0.2,
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": results,
    }
