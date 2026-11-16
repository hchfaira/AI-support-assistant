"""Builds this repo's RAG knowledge base and LLM SFT dataset from the
open-source Bitext customer-support dataset
(`bitext/Bitext-customer-support-llm-chatbot-training-dataset`).

This dataset pairs a customer instruction with an agent response per
intent, which is exactly what both halves of the RAG+LLM pipeline need:

- A **knowledge base** (one canonical document per intent, built from its
  agent response) consumable by `src.data_engineering.ingestion
  .load_documents_from_directory` for hybrid retrieval/reranking -- same
  role as `data/sample/knowledge_base.json`.
- An **SFT dataset** (instruction/context/query/response) in the exact
  schema `src.llm.finetuning.load_sft_dataset` expects -- same role as
  `data/sample/sft_dataset.json`.
- **Eval tickets** (query/relevant_docs) for the embedding/reranker
  comparison notebooks -- same role as `data/sample/eval_tickets.json`.

Usage:
    python -m src.data_engineering.support_dataset_builder --max-samples 500
    python -m src.data_engineering.support_dataset_builder --output-dir data/processed
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data_engineering.hf_loader import load_support_dataset

SFT_INSTRUCTION = (
    "You are a customer support assistant. Answer using only the provided "
    "context and respond in the required JSON schema."
)

# Heuristic priority/escalation mapping (same spirit as
# `src.rag.generation.heuristic_generate`'s keyword rules) -- adjust the
# keyword lists to match your own production definition of priority and
# escalation criteria.
_HIGH_PRIORITY_KEYWORDS = ("refund", "payment", "cancel")
_MEDIUM_PRIORITY_KEYWORDS = ("delivery", "shipping", "order", "invoice")
_ESCALATION_KEYWORDS = ("complaint", "human_agent")


def derive_priority(category: str, intent: str) -> str:
    """Heuristically map a Bitext `category`/`intent` to a priority level."""
    haystack = f"{category} {intent}".lower()
    if any(kw in haystack for kw in _HIGH_PRIORITY_KEYWORDS):
        return "high"
    if any(kw in haystack for kw in _MEDIUM_PRIORITY_KEYWORDS):
        return "medium"
    return "low"


def derive_escalation(intent: str) -> bool:
    """Heuristically flag intents that should be escalated to a human agent."""
    lower_intent = intent.lower()
    return any(kw in lower_intent for kw in _ESCALATION_KEYWORDS)


def build_knowledge_base(records: list[dict]) -> list[dict]:
    """Build one canonical knowledge-base document per unique intent.

    Picks the longest response seen for each intent as the canonical
    document text (a reasonable proxy for "most complete answer").
    Returns records in the generic schema
    `src.data_engineering.ingestion.load_documents_from_directory` expects:
    `{"doc_id", "title", "text", "metadata"}`.
    """
    best_response_by_intent: dict[str, dict] = {}
    for record in records:
        intent = record["intent"]
        response = record["response"]
        existing = best_response_by_intent.get(intent)
        if existing is None or len(response) > len(existing["response"]):
            best_response_by_intent[intent] = {
                "category": record["category"],
                "response": response,
            }

    documents = []
    for intent, data in sorted(best_response_by_intent.items()):
        documents.append(
            {
                "doc_id": intent,
                "title": intent.replace("_", " ").title(),
                "text": data["response"],
                "metadata": {"category": data["category"], "intent": intent},
            }
        )
    return documents


def build_sft_dataset(records: list[dict], knowledge_base: list[dict]) -> list[dict]:
    """Build (instruction, context, query, response) SFT pairs, grounding each
    example's context in its matching knowledge-base document -- so training
    teaches the model to answer *from the retrieved context*, matching how
    `src.rag.generation.answer_query` calls the LLM at inference time.
    """
    context_by_intent = {doc["doc_id"]: doc["text"] for doc in knowledge_base}

    sft_examples = []
    for record in records:
        intent = record["intent"]
        context = context_by_intent.get(intent, record["response"])
        structured_response = {
            "intent": intent,
            "priority": derive_priority(record["category"], intent),
            "answer": record["response"],
            "escalation_required": derive_escalation(intent),
        }
        sft_examples.append(
            {
                "instruction": SFT_INSTRUCTION,
                "context": context,
                "query": record["instruction"],
                "response": json.dumps(structured_response),
            }
        )
    return sft_examples


def build_eval_tickets(records: list[dict], n: int = 50, seed: int = 42) -> list[dict]:
    """Build (query, relevant_docs) retrieval/reranker eval tickets, in the
    same schema as `data/sample/eval_tickets.json`, for the embedding/reranker
    comparison notebooks (`02_embedding_model_comparison.ipynb`,
    `03_reranker_comparison.ipynb`).

    Each ticket's `relevant_docs` is the single knowledge-base doc_id (the
    record's intent) that `build_knowledge_base` produces for it, so these
    tickets are only meaningful when evaluated against the knowledge base
    built from the *same* records (or a superset of them).
    """
    import random

    rng = random.Random(seed)
    sample = records if len(records) <= n else rng.sample(records, n)
    return [{"query": r["instruction"], "relevant_docs": [r["intent"]]} for r in sample]


def build(max_samples: int | None, output_dir: str | Path, seed: int = 42) -> None:
    """Pull the Bitext dataset and write the knowledge base, SFT dataset and
    eval tickets to `output_dir` as `bitext_knowledge_base.json`,
    `bitext_sft_dataset.json` and `bitext_eval_tickets.json`.
    """
    dataset = load_support_dataset(max_samples=max_samples, seed=seed)
    records = [dict(r) for r in dataset]

    knowledge_base = build_knowledge_base(records)
    sft_dataset = build_sft_dataset(records, knowledge_base)
    eval_tickets = build_eval_tickets(records, seed=seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "bitext_knowledge_base.json").write_text(
        json.dumps(knowledge_base, indent=2), encoding="utf-8"
    )
    (output_dir / "bitext_sft_dataset.json").write_text(
        json.dumps(sft_dataset, indent=2), encoding="utf-8"
    )
    (output_dir / "bitext_eval_tickets.json").write_text(
        json.dumps(eval_tickets, indent=2), encoding="utf-8"
    )
    print(
        f"Wrote {len(knowledge_base)} knowledge-base docs, "
        f"{len(sft_dataset)} SFT examples and {len(eval_tickets)} eval "
        f"tickets to {output_dir}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Subsample the raw dataset for a fast, low-compute run.",
    )
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build(max_samples=args.max_samples, output_dir=args.output_dir, seed=args.seed)


if __name__ == "__main__":
    main()
