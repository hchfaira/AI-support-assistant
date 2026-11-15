"""BM25 lexical retriever wrapper (rank_bm25-based)."""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


@dataclass
class BM25Index:
    doc_ids: list[str]
    bm25: BM25Okapi

    @classmethod
    def build(cls, doc_ids: list[str], texts: list[str]) -> "BM25Index":
        tokenized_corpus = [_tokenize(t) for t in texts]
        return cls(doc_ids=doc_ids, bm25=BM25Okapi(tokenized_corpus))

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        scores = self.bm25.get_scores(_tokenize(query))
        ranked_idx = np.argsort(-scores)[:top_k]
        return [(self.doc_ids[i], float(scores[i])) for i in ranked_idx]
