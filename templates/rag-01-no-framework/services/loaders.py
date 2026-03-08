import logging
import os
from urllib.parse import urlparse

import fitz  # PyMuPDF
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

BLOCKED_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "169.254.169.254", "[::1]"}
BLOCKED_PREFIXES = ("10.", "172.16.", "172.17.", "172.18.", "172.19.", "172.20.",
                    "172.21.", "172.22.", "172.23.", "172.24.", "172.25.", "172.26.",
                    "172.27.", "172.28.", "172.29.", "172.30.", "172.31.", "192.168.")


def _validate_url(url: str) -> None:
    """Validate URL to prevent SSRF attacks."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Invalid URL scheme: {parsed.scheme}. Only http/https allowed.")
    hostname = parsed.hostname or ""
    if hostname in BLOCKED_HOSTS or hostname.startswith(BLOCKED_PREFIXES):
        raise ValueError("URLs pointing to internal/private networks are not allowed.")


def load_pdf(file_path: str) -> tuple[str, dict]:
    """Extract text from a PDF file."""
    try:
        doc = fitz.open(file_path)
        pages = [page.get_text() for page in doc]
        doc.close()
    except Exception as e:
        logger.error("Failed to parse PDF %s: %s", file_path, e)
        raise ValueError(f"Failed to parse PDF: {e}") from e
    text = "\n".join(pages)
    return text, {
        "source_type": "pdf",
        "filename": os.path.basename(file_path),
    }


def load_text(file_path: str) -> tuple[str, dict]:
    """Read a text/markdown file."""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text, {
        "source_type": "text",
        "filename": os.path.basename(file_path),
    }


def load_url(url: str) -> tuple[str, dict]:
    """Fetch and extract text from a web page."""
    _validate_url(url)

    headers = {"User-Agent": "Mozilla/5.0 (compatible; RAGBot/1.0)"}
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    title = soup.title.string if soup.title else url

    return text, {
        "source_type": "url",
        "filename": title,
        "url": url,
    }
