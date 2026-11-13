"""Document ingestion utilities.

Loads raw documents (JSON knowledge base, PDFs, HTML, Markdown) from `data/raw/`
or a bundled sample knowledge base, and returns a normalized list of `Document`
records ready for parsing and chunking.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Document:
    """A single source document before chunking."""

    doc_id: str
    title: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def load_sample_knowledge_base(
    path: str | Path = "data/sample/knowledge_base.json",
) -> list[Document]:
    """Load the bundled synthetic knowledge base used for demos/tests."""
    path = Path(path)
    records = json.loads(path.read_text(encoding="utf-8"))
    documents = []
    for record in records:
        documents.append(
            Document(
                doc_id=record["doc_id"],
                title=record.get("title", record["doc_id"]),
                text=record["text"],
                metadata={"page": record.get("page", 1)},
            )
        )
    return documents


def load_documents_from_directory(directory: str | Path) -> list[Document]:
    """Load every `.json`/`.txt`/`.md` file in `directory` as a `Document`.

    Real PDF/HTML ingestion should plug a parser (e.g. `pypdf`, `unstructured`,
    `BeautifulSoup`) into `document_parser.py` and feed its output here.
    """
    directory = Path(directory)
    documents: list[Document] = []
    for file_path in sorted(directory.glob("**/*")):
        if not file_path.is_file():
            continue
        if file_path.suffix == ".json":
            payload = json.loads(file_path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                for record in payload:
                    documents.append(
                        Document(
                            doc_id=record.get("doc_id", file_path.stem),
                            title=record.get("title", file_path.stem),
                            text=record["text"],
                            metadata=record.get("metadata", {}),
                        )
                    )
            continue
        if file_path.suffix in {".txt", ".md"}:
            documents.append(
                Document(
                    doc_id=file_path.stem,
                    title=file_path.stem,
                    text=file_path.read_text(encoding="utf-8"),
                    metadata={"source_path": str(file_path)},
                )
            )
    return documents
