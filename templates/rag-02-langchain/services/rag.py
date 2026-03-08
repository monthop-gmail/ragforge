from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from config import settings
from models import SearchMode
from services.hybrid_search import reciprocal_rank_fusion
from services.keyword_search import search as keyword_search
from services.vector_store import query as vector_query

_llm = ChatOpenAI(
    model=settings.openai_chat_model,
    openai_api_key=settings.openai_api_key,
    temperature=0.2,
)

_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that answers questions based on the provided context.\n"
            "Use ONLY the context below to answer. If the context doesn't contain enough information, say so.\n"
            "Do not make up information.",
        ),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)

_chain = _prompt | _llm | StrOutputParser()


def _retrieve(question: str, top_k: int, search_mode: SearchMode) -> list[dict]:
    """Retrieve relevant chunks using the specified search mode."""
    if search_mode == SearchMode.vector:
        return vector_query(question, top_k)

    if search_mode == SearchMode.keyword:
        return keyword_search(question, top_k)

    # hybrid
    vector_results = vector_query(question, top_k)
    keyword_results = keyword_search(question, top_k)
    return reciprocal_rank_fusion(vector_results, keyword_results)[:top_k]


def ask(question: str, top_k: int | None = None, search_mode: SearchMode = SearchMode.hybrid) -> dict:
    """Run the full RAG pipeline using LangChain LCEL."""
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

    # Run chain
    answer = _chain.invoke({"context": context, "question": question})

    return {
        "answer": answer,
        "sources": results,
    }
