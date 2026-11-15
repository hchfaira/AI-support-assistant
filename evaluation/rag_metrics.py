"""RAG-level evaluation (faithfulness, answer relevancy, context precision /
recall, citation accuracy) using the RAGAS framework.

Requires `ragas` + `datasets` installed and an LLM judge configured (e.g. via
an OpenAI-compatible endpoint or a local model) -- RAGAS needs an LLM to score
faithfulness/relevancy. Citation accuracy is computed locally without an LLM
judge since it just checks that every citation in the answer is one of the
retrieved sources.

Usage:
    python -m evaluation.rag_metrics
"""
from __future__ import annotations


def citation_accuracy(answers: list[str], cited_sources: list[list[str]], retrieved_docs: list[list[str]]) -> float:
    """Fraction of answers whose cited sources are all contained in the
    documents that were actually retrieved for that query (no hallucinated
    citations).
    """
    if not answers:
        return 0.0
    correct = 0
    for cited, retrieved in zip(cited_sources, retrieved_docs):
        if set(cited).issubset(set(retrieved)):
            correct += 1
    return correct / len(answers)


def run_ragas_evaluation(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str] | None = None,
) -> dict:
    """Compute Faithfulness, Answer Relevancy, Context Precision and Context
    Recall with RAGAS. Requires network access to an LLM judge at eval time.
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

    data = {"question": questions, "answer": answers, "contexts": contexts}
    if ground_truths:
        data["ground_truth"] = ground_truths

    dataset = Dataset.from_dict(data)
    metrics = [faithfulness, answer_relevancy, context_precision]
    if ground_truths:
        metrics.append(context_recall)

    result = evaluate(dataset, metrics=metrics)
    return result.to_pandas().mean(numeric_only=True).to_dict()


if __name__ == "__main__":
    # Local, LLM-judge-free sanity check for citation accuracy.
    example_answers = ["We refunded the duplicate charge per billing_policy."]
    example_cited = [["billing_policy.pdf"]]
    example_retrieved = [["billing_policy.pdf", "duplicate_payment_procedure.pdf"]]
    print("citation_accuracy:", citation_accuracy(example_answers, example_cited, example_retrieved))
