"""Contrastive fine-tuning of an embedding model on (query, positive, negative)
triplets using Sentence-Transformers' `MultipleNegativesRankingLoss`.

Usage:
    python -m src.embeddings.train --data data/processed/embedding_triplets.jsonl \
        --base-model BAAI/bge-base-en-v1.5 --output checkpoints/embedding-finetuned

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
    base_model: str = "BAAI/bge-base-en-v1.5",
    output_dir: str = "checkpoints/embedding-finetuned",
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
) -> None:
    """Fine-tune `base_model` with contrastive ranking loss on query/pos/neg triplets.

    Requires `sentence-transformers` + `torch` installed. Intended to run on a
    GPU instance; falls back gracefully with a clear error on CPU-only/offline
    environments.
    """
    from sentence_transformers import InputExample, SentenceTransformer, losses
    from torch.utils.data import DataLoader

    triplets = load_triplets(data_path)
    examples = [
        InputExample(texts=[t["query"], t["positive"], t["negative"]]) for t in triplets
    ]

    model = SentenceTransformer(base_model)
    train_dataloader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        optimizer_params={"lr": learning_rate},
        output_path=output_dir,
        show_progress_bar=True,
    )
    print(f"Saved fine-tuned embedding model to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to JSONL triplets file")
    parser.add_argument("--base-model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--output", default="checkpoints/embedding-finetuned")
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
