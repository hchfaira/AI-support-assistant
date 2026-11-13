"""Cross-encoder reranker wrapper.

Defaults to a small CPU-friendly cross-encoder
(`cross-encoder/ms-marco-MiniLM-L-6-v2`). Swap `model_name` for
`BAAI/bge-reranker-v2-m3` in production. Falls back to a lexical-overlap
scorer when the cross-encoder backend is unavailable, so the pipeline keeps
running end-to-end in constrained environments.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class RerankedDocument:
    doc_id: str
    text: str
    score: float


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._backend = None
        try:
            from sentence_transformers import CrossEncoder

            self._backend = CrossEncoder(model_name)
        except Exception:
            self._backend = None

    def rerank(
        self, query: str, candidates: list[tuple[str, str]], top_k: int | None = None
    ) -> list[RerankedDocument]:
        """`candidates` is a list of (doc_id, text) pairs. Returns them reordered
        by relevance score, highest first.
        """
        if not candidates:
            return []

        if self._backend is not None:
            pairs = [[query, text] for _, text in candidates]
            scores = self._backend.predict(pairs)
        else:
            scores = [self._lexical_overlap_score(query, text) for _, text in candidates]

        ranked = sorted(zip(candidates, scores), key=lambda pair: -pair[1])
        results = [
            RerankedDocument(doc_id=doc_id, text=text, score=float(score))
            for (doc_id, text), score in ranked
        ]
        return results[:top_k] if top_k else results

    @staticmethod
    def _lexical_overlap_score(query: str, text: str) -> float:
        query_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
        text_tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
        if not query_tokens:
            return 0.0
        return len(query_tokens & text_tokens) / len(query_tokens)

    @property
    def is_real_backend(self) -> bool:
        return self._backend is not None
