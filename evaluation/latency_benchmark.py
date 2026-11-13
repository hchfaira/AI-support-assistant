"""Latency/throughput benchmarking utility for serving experiments: measure
P50/P95/P99 latency and throughput across batch sizes and concurrency levels
for any callable (embedding model, reranker, or LLM generate function).

Usage:
    python -m evaluation.latency_benchmark
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable


@dataclass
class BenchmarkResult:
    label: str
    concurrency: int
    num_requests: int
    throughput_rps: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    idx = min(int(round(pct / 100 * (len(sorted_values) - 1))), len(sorted_values) - 1)
    return sorted_values[idx]


def benchmark(
    fn: Callable[[], None],
    label: str,
    num_requests: int = 50,
    concurrency: int = 1,
) -> BenchmarkResult:
    """Run `fn` `num_requests` times across `concurrency` worker threads and
    report throughput + latency percentiles.
    """
    latencies: list[float] = []

    def _timed_call(_: int) -> float:
        start = time.perf_counter()
        fn()
        return (time.perf_counter() - start) * 1000

    start_time = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        latencies = list(executor.map(_timed_call, range(num_requests)))
    total_time = time.perf_counter() - start_time

    return BenchmarkResult(
        label=label,
        concurrency=concurrency,
        num_requests=num_requests,
        throughput_rps=num_requests / total_time if total_time > 0 else 0.0,
        p50_latency_ms=_percentile(latencies, 50),
        p95_latency_ms=_percentile(latencies, 95),
        p99_latency_ms=_percentile(latencies, 99),
    )


if __name__ == "__main__":
    from src.data_engineering.ingestion import load_sample_knowledge_base
    from src.rag.generation import answer_query, build_default_pipeline

    retriever, reranker = build_default_pipeline()
    documents = load_sample_knowledge_base()
    sample_query = "How can I request a refund?"

    for concurrency in (1, 4, 8):
        result = benchmark(
            lambda: answer_query(sample_query, retriever, reranker),
            label="rag_pipeline",
            num_requests=20,
            concurrency=concurrency,
        )
        print(result)
