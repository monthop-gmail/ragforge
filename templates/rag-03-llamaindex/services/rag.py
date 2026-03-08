from llama_index.core import Settings as LlamaSettings
from llama_index.core.prompts import PromptTemplate

from config import settings
from models import SearchMode
from services.hybrid_search import reciprocal_rank_fusion
from services.index import get_index
from services.keyword_search import search as keyword_search

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions based on the provided context.\n"
    "Use ONLY the context below to answer. If the context doesn't contain enough information, say so.\n"
    "Do not make up information.\n\n"
    "Context:\n{context_str}\n\n"
    "Question: {query_str}\n"
    "Answer: "
)

_qa_template = PromptTemplate(SYSTEM_PROMPT)


def _vector_retrieve(question: str, top_k: int) -> list[dict]:
    """Retrieve using LlamaIndex vector retriever."""
    index = get_index()
    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(question)

    results = []
    for node in nodes:
        results.append({
            "id": node.node_id,
            "chunk_text": node.text,
            "score": node.score,
            "metadata": node.metadata,
        })
    return results


def _retrieve(question: str, top_k: int, search_mode: SearchMode) -> list[dict]:
    """Retrieve relevant chunks using the specified search mode."""
    if search_mode == SearchMode.vector:
        return _vector_retrieve(question, top_k)

    if search_mode == SearchMode.keyword:
        return keyword_search(question, top_k)

    # hybrid
    vector_results = _vector_retrieve(question, top_k)
    keyword_results = keyword_search(question, top_k)
    return reciprocal_rank_fusion(vector_results, keyword_results)[:top_k]


def ask(question: str, top_k: int | None = None, search_mode: SearchMode = SearchMode.hybrid) -> dict:
    """Run the full RAG pipeline using LlamaIndex."""
    k = top_k or settings.top_k

    results = _retrieve(question, k, search_mode)

    if not results:
        return {
            "answer": "No relevant documents found. Please upload some documents first.",
            "sources": [],
        }

    # Build context and use LLM to generate answer
    context_parts = []
    for i, r in enumerate(results, 1):
        filename = r["metadata"].get("filename", "unknown")
        context_parts.append(f"[Source {i}] ({filename}):\n{r['chunk_text']}")
    context = "\n\n".join(context_parts)

    llm = LlamaSettings.llm
    prompt = _qa_template.format(context_str=context, query_str=question)
    response = llm.complete(prompt)

    return {
        "answer": str(response),
        "sources": results,
    }
