"""Chunking strategies for splitting documents before embedding.

Implements the four strategies compared in the project write-up:
fixed-size, overlapping, recursive (structure-aware) and semantic chunking.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from src.data_engineering.ingestion import Document


@dataclass
class Chunk:
    """A chunk of text ready for embedding, with traceable metadata."""

    chunk_id: str
    doc_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _word_tokenize(text: str) -> list[str]:
    return text.split()


def fixed_size_chunking(document: Document, chunk_size: int = 256) -> list[Chunk]:
    """Split a document into non-overlapping chunks of `chunk_size` tokens."""
    words = _word_tokenize(document.text)
    chunks = []
    for i in range(0, len(words), chunk_size):
        piece = " ".join(words[i : i + chunk_size])
        chunks.append(
            Chunk(
                chunk_id=f"{document.doc_id}::fixed::{i // chunk_size}",
                doc_id=document.doc_id,
                text=piece,
                metadata={**document.metadata, "strategy": "fixed-size"},
            )
        )
    return chunks


def overlapping_chunking(
    document: Document, chunk_size: int = 512, overlap: int = 64
) -> list[Chunk]:
    """Split a document into overlapping chunks to avoid losing context at boundaries."""
    words = _word_tokenize(document.text)
    chunks = []
    step = max(chunk_size - overlap, 1)
    idx = 0
    pos = 0
    while pos < len(words):
        piece = " ".join(words[pos : pos + chunk_size])
        chunks.append(
            Chunk(
                chunk_id=f"{document.doc_id}::overlap::{idx}",
                doc_id=document.doc_id,
                text=piece,
                metadata={**document.metadata, "strategy": "overlapping"},
            )
        )
        pos += step
        idx += 1
    return chunks


def recursive_chunking(
    document: Document,
    separators: tuple[str, ...] = ("\n\n", "\n", ". "),
    chunk_size: int = 512,
) -> list[Chunk]:
    """Recursively split on structural separators (paragraphs, lines, sentences)
    before falling back to fixed-size splitting, preserving document structure.
    """

    def _split(text: str, seps: tuple[str, ...]) -> list[str]:
        if len(text.split()) <= chunk_size or not seps:
            return [text]
        sep, rest = seps[0], seps[1:]
        parts = [p for p in text.split(sep) if p.strip()]
        if len(parts) <= 1:
            return _split(text, rest)
        out: list[str] = []
        for part in parts:
            out.extend(_split(part, rest))
        return out

    pieces = _split(document.text, separators)
    chunks = []
    for idx, piece in enumerate(pieces):
        chunks.append(
            Chunk(
                chunk_id=f"{document.doc_id}::recursive::{idx}",
                doc_id=document.doc_id,
                text=piece.strip(),
                metadata={**document.metadata, "strategy": "recursive"},
            )
        )
    return chunks


def semantic_chunking(
    document: Document,
    embed_fn: Callable[[list[str]], Any],
    similarity_threshold: float = 0.6,
) -> list[Chunk]:
    """Group consecutive sentences into a chunk while embedding similarity stays
    above `similarity_threshold`; start a new chunk when the topic shifts.

    `embed_fn` must accept a list of sentences and return a 2D array of vectors
    (e.g. `src.embeddings.inference.EmbeddingModel.encode`).
    """
    import numpy as np

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", document.text) if s.strip()]
    if not sentences:
        return []

    vectors = np.asarray(embed_fn(sentences))
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    unit_vectors = vectors / norms

    groups: list[list[str]] = [[sentences[0]]]
    for i in range(1, len(sentences)):
        similarity = float(np.dot(unit_vectors[i], unit_vectors[i - 1]))
        if similarity >= similarity_threshold:
            groups[-1].append(sentences[i])
        else:
            groups.append([sentences[i]])

    chunks = []
    for idx, group in enumerate(groups):
        chunks.append(
            Chunk(
                chunk_id=f"{document.doc_id}::semantic::{idx}",
                doc_id=document.doc_id,
                text=" ".join(group),
                metadata={**document.metadata, "strategy": "semantic"},
            )
        )
    return chunks


CHUNKING_STRATEGIES: dict[str, Callable[..., list[Chunk]]] = {
    "fixed-size": fixed_size_chunking,
    "overlapping": overlapping_chunking,
    "recursive": recursive_chunking,
    "semantic": semantic_chunking,
}
