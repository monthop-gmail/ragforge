"""MCP Server for RAG - exposes RAG API as MCP tools via SSE transport."""

import io
import os

import httpx
from mcp.server.fastmcp import FastMCP

RAG_API_URL = os.environ.get("RAG_API_URL", "http://localhost:8000")
MCP_PORT = int(os.environ.get("MCP_PORT", "8001"))

mcp = FastMCP("RagForge", host="0.0.0.0", port=MCP_PORT)


def _url(path: str) -> str:
    return f"{RAG_API_URL}{path}"


@mcp.tool()
async def rag_query(question: str, top_k: int = 5) -> str:
    """Ask a question to the RAG knowledge base. Returns an answer based on ingested documents."""
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(_url("/query"), json={"question": question, "top_k": top_k})
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as e:
        return f"Error: RAG service unavailable - {e}"

    answer = data["answer"]
    sources = data.get("sources", [])

    result = f"Answer: {answer}\n"
    if sources:
        result += "\nSources:\n"
        for i, s in enumerate(sources, 1):
            filename = s.get("metadata", {}).get("filename", "unknown")
            score = s.get("score", "N/A")
            text_preview = s.get("chunk_text", "")[:200]
            result += f"  [{i}] {filename} (score: {score})\n      {text_preview}...\n"

    return result


@mcp.tool()
async def rag_upload_text(content: str, filename: str = "document.txt") -> str:
    """Upload text content to the RAG knowledge base. Use this to add new knowledge."""
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            files = {"file": (filename, io.BytesIO(content.encode("utf-8")), "text/plain")}
            resp = await client.post(_url("/upload"), files=files)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as e:
        return f"Error: Failed to upload - {e}"

    return f"Uploaded '{data['filename']}' - {data['chunk_count']} chunks indexed (ID: {data['document_id']})"


@mcp.tool()
async def rag_ingest_url(url: str) -> str:
    """Ingest content from a URL into the RAG knowledge base."""
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(_url("/ingest-url"), json={"url": url})
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as e:
        return f"Error: Failed to ingest URL - {e}"

    return f"Ingested '{data['filename']}' - {data['chunk_count']} chunks indexed (ID: {data['document_id']})"


@mcp.tool()
async def rag_list_documents() -> str:
    """List all documents in the RAG knowledge base."""
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(_url("/documents"))
            resp.raise_for_status()
            docs = resp.json()
    except httpx.HTTPError as e:
        return f"Error: RAG service unavailable - {e}"

    if not docs:
        return "No documents in the knowledge base."

    lines = [f"Documents ({len(docs)}):\n"]
    for d in docs:
        lines.append(f"  - [{d['source_type']}] {d['filename']} ({d['chunk_count']} chunks) ID: {d['id']}")

    return "\n".join(lines)


@mcp.tool()
async def rag_delete_document(document_id: str) -> str:
    """Delete a document from the RAG knowledge base by its ID."""
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.delete(_url(f"/documents/{document_id}"))
            if resp.status_code == 404:
                return f"Document {document_id} not found."
            resp.raise_for_status()
    except httpx.HTTPError as e:
        return f"Error: Failed to delete - {e}"

    return f"Document {document_id} deleted successfully."


if __name__ == "__main__":
    mcp.run(transport="sse")
