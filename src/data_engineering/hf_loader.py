"""Loaders for open-source datasets hosted on the Hugging Face Hub used to
build this repo's RAG knowledge base and LLM SFT dataset from real data
(see `src.data_engineering.support_dataset_builder`).
"""
from __future__ import annotations


def load_support_dataset(
    split: str = "train",
    max_samples: int | None = None,
    seed: int = 42,
):
    """Load the Bitext customer-support instruction/response dataset.

    This dataset pairs a customer `instruction` with an agent `response` and
    an `intent`/`category` -- exactly what the RAG knowledge base and LLM
    SFT dataset need (see `src.data_engineering.support_dataset_builder`).

    Args:
        split: the HF dataset split to load (this dataset only ships "train").
        max_samples: if set, randomly subsample to this size -- use this on
            CPU-only/low-memory machines to smoke-test the pipeline before
            building the full knowledge base / SFT dataset on a GPU runtime.
        seed: shuffle seed used when subsampling, for reproducibility.

    Returns:
        A HF `Dataset` with "instruction", "category", "intent", "response" columns.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "load_support_dataset requires the 'datasets' package: pip install datasets"
        ) from exc

    dataset = load_dataset(
        "bitext/Bitext-customer-support-llm-chatbot-training-dataset", split=split
    )
    dataset = dataset.select_columns(["instruction", "category", "intent", "response"])

    if max_samples is not None and max_samples < len(dataset):
        dataset = dataset.shuffle(seed=seed).select(range(max_samples))

    return dataset
