"""Evaluate and compare a base LLM vs its QLoRA fine-tuned counterpart on the
customer-support instruction dataset: Exact Match, ROUGE-L, BERTScore and
JSON format compliance.
"""
from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass
class LLMEvalResult:
    model_name: str
    exact_match: float
    rouge_l: float
    format_compliance: float
    bertscore_f1: float | None = None


REQUIRED_KEYS = {"intent", "priority", "answer", "escalation_required"}


def _is_format_compliant(text: str) -> bool:
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return False
    return isinstance(payload, dict) and REQUIRED_KEYS.issubset(payload.keys())


def evaluate_generations(
    model_name: str,
    predictions: list[str],
    references: list[str],
    compute_bertscore: bool = False,
) -> LLMEvalResult:
    """Score a list of generated JSON responses against ground-truth responses."""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    exact_matches = [1.0 if p.strip() == r.strip() else 0.0 for p, r in zip(predictions, references)]
    rouge_l_scores = [
        scorer.score(r, p)["rougeL"].fmeasure for p, r in zip(predictions, references)
    ]
    format_scores = [1.0 if _is_format_compliant(p) else 0.0 for p in predictions]

    bertscore_f1 = None
    if compute_bertscore:
        from bert_score import score as bert_score

        _, _, f1 = bert_score(predictions, references, lang="en", verbose=False)
        bertscore_f1 = float(f1.mean())

    return LLMEvalResult(
        model_name=model_name,
        exact_match=sum(exact_matches) / len(exact_matches) if exact_matches else 0.0,
        rouge_l=sum(rouge_l_scores) / len(rouge_l_scores) if rouge_l_scores else 0.0,
        format_compliance=sum(format_scores) / len(format_scores) if format_scores else 0.0,
        bertscore_f1=bertscore_f1,
    )
