"""Parsers for turning raw source files into clean plain text.

Each parser takes a file path and returns normalized text. Real production
usage would wire in `pypdf`/`pdfplumber` for PDFs, `BeautifulSoup` for HTML,
and `markdown-it-py` for Markdown; interfaces are kept simple so backends can
be swapped without touching the rest of the pipeline.
"""
from __future__ import annotations

import re
from pathlib import Path


def clean_text(text: str) -> str:
    """Normalize whitespace and strip control characters from parsed text."""
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_markdown(path: str | Path) -> str:
    """Parse a Markdown file into clean plain text (headings/formatting kept)."""
    raw = Path(path).read_text(encoding="utf-8")
    return clean_text(raw)


def parse_pdf(path: str | Path) -> str:
    """Parse a PDF file into clean plain text.

    Requires `pypdf` at runtime:
        pip install pypdf
    """
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "parse_pdf requires the 'pypdf' package: pip install pypdf"
        ) from exc

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return clean_text("\n\n".join(pages))


def parse_html(path: str | Path) -> str:
    """Parse an HTML file into clean plain text.

    Requires `beautifulsoup4` at runtime:
        pip install beautifulsoup4
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "parse_html requires 'beautifulsoup4': pip install beautifulsoup4"
        ) from exc

    raw = Path(path).read_text(encoding="utf-8")
    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    return clean_text(soup.get_text(separator="\n"))


PARSERS = {
    ".pdf": parse_pdf,
    ".html": parse_html,
    ".htm": parse_html,
    ".md": parse_markdown,
    ".markdown": parse_markdown,
}


def parse_document(path: str | Path) -> str:
    """Dispatch to the right parser based on file extension."""
    path = Path(path)
    parser = PARSERS.get(path.suffix.lower())
    if parser is None:
        return clean_text(path.read_text(encoding="utf-8"))
    return parser(path)
