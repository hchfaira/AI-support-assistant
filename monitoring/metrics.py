"""Serving metrics collection: request counters, latency histograms and
token-throughput tracking for the RAG/LLM pipeline, exposed in a Prometheus-
compatible format.

Usage (FastAPI integration):
    from monitoring.metrics import MetricsMiddleware, metrics_endpoint
    app.add_middleware(MetricsMiddleware)
    app.add_route("/metrics", metrics_endpoint)
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class MetricsRegistry:
    """In-process metrics store (swap for `prometheus_client` in production
    for multi-process/multi-replica aggregation)."""

    request_count: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    latency_samples_ms: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    tokens_generated: int = 0
    errors: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def record_request(self, route: str, latency_ms: float, error: bool = False) -> None:
        self.request_count[route] += 1
        self.latency_samples_ms[route].append(latency_ms)
        if error:
            self.errors[route] += 1

    def record_tokens(self, n_tokens: int) -> None:
        self.tokens_generated += n_tokens

    def percentile(self, route: str, pct: float) -> float:
        samples = sorted(self.latency_samples_ms.get(route, []))
        if not samples:
            return 0.0
        idx = min(int(round(pct / 100 * (len(samples) - 1))), len(samples) - 1)
        return samples[idx]

    def summary(self) -> dict:
        return {
            route: {
                "count": self.request_count[route],
                "errors": self.errors.get(route, 0),
                "p50_ms": self.percentile(route, 50),
                "p95_ms": self.percentile(route, 95),
                "p99_ms": self.percentile(route, 99),
            }
            for route in self.request_count
        }


REGISTRY = MetricsRegistry()


class RequestTimer:
    """Context manager that records request latency into `REGISTRY`."""

    def __init__(self, route: str):
        self.route = route
        self._start = 0.0
        self.error = False

    def __enter__(self) -> "RequestTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        latency_ms = (time.perf_counter() - self._start) * 1000
        REGISTRY.record_request(self.route, latency_ms, error=exc_type is not None)
