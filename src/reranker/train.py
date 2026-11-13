"""Pairwise fine-tuning of a cross-encoder reranker on (query, positive, negative)
triples, using Sentence-Transformers' `CrossEncoder` with a ranking loss.

Usage:
    python -m src.reranker.train --data data/processed/reranker_triplets.jsonl \
        --base-model BAAI/bge-reranker-base --output checkpoints/reranker-finetuned

Each line of the training file must be a JSON object:
    {"query": "...", "positive": "...", "negative": "..."}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_triplets(path: str | Path) -> list[dict]:
    path = Path(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def train(
    data_path: str,
    base_model: str = "BAAI/bge-reranker-base",
    output_dir: str = "checkpoints/reranker-finetuned",
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
) -> None:
    """Fine-tune a cross-encoder reranker with pairwise/contrastive ranking loss.

    Each triplet (query, positive, negative) is turned into two labeled pairs:
    (query, positive) -> 1.0 and (query, negative) -> 0.0, trained with binary
    cross-entropy, which approximates pairwise ranking preference learning.

    Requires `sentence-transformers` + `torch` installed; intended for GPU use.
    """
    from sentence_transformers import InputExample
    from sentence_transformers.cross_encoder import CrossEncoder
    from torch.utils.data import DataLoader

    triplets = load_triplets(data_path)
    examples = []
    for t in triplets:
        examples.append(InputExample(texts=[t["query"], t["positive"]], label=1.0))
        examples.append(InputExample(texts=[t["query"], t["negative"]], label=0.0))

    model = CrossEncoder(base_model, num_labels=1)
    train_dataloader = DataLoader(examples, shuffle=True, batch_size=batch_size)

    model.fit(
        train_dataloader=train_dataloader,
        epochs=epochs,
        optimizer_params={"lr": learning_rate},
        output_path=output_dir,
        show_progress_bar=True,
    )
    print(f"Saved fine-tuned reranker to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to JSONL triplets file")
    parser.add_argument("--base-model", default="BAAI/bge-reranker-base")
    parser.add_argument("--output", default="checkpoints/reranker-finetuned")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    train(
        data_path=args.data,
        base_model=args.base_model,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )


if __name__ == "__main__":
    main()
