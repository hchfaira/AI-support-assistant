import pytest

from src.data_engineering.chunking import (
    fixed_size_chunking,
    overlapping_chunking,
    recursive_chunking,
)
from src.data_engineering.ingestion import Document, load_sample_knowledge_base


def test_load_sample_knowledge_base_returns_documents():
    documents = load_sample_knowledge_base()
    assert len(documents) > 0
    assert all(isinstance(d, Document) for d in documents)
    assert all(d.text for d in documents)


def test_fixed_size_chunking_respects_chunk_size():
    doc = Document(doc_id="d1", title="t", text=" ".join(f"word{i}" for i in range(100)))
    chunks = fixed_size_chunking(doc, chunk_size=10)
    assert len(chunks) == 10
    assert all(len(c.text.split()) <= 10 for c in chunks)


def test_overlapping_chunking_produces_overlap():
    doc = Document(doc_id="d1", title="t", text=" ".join(f"word{i}" for i in range(100)))
    chunks = overlapping_chunking(doc, chunk_size=20, overlap=5)
    assert len(chunks) > 1
    first_words = chunks[0].text.split()[-5:]
    second_words = chunks[1].text.split()[:5]
    assert first_words == second_words


def test_recursive_chunking_preserves_all_text_content():
    doc = Document(doc_id="d1", title="t", text="Para one.\n\nPara two.\n\nPara three.")
    chunks = recursive_chunking(doc, chunk_size=3)
    assert len(chunks) == 3
    assert {c.text for c in chunks} == {"Para one.", "Para two.", "Para three."}
