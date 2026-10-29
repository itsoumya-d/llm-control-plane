"""Local micro-benchmarks for routing overhead, audit verification, and evals."""

from __future__ import annotations

import json
import statistics
import time
from typing import Dict, List

from .evals import ContainsScorer, Dataset, EvalCase, EvalRunner
from .gateway import LLMGateway
from .providers import FakeProvider, provider_map
from .rate_limit import TenantRateLimiter
from .routing import RoutingTable
from .types import ChatMessage, CompletionRequest


def percentile(values: List[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = max(0, min(len(ordered) - 1, int(fraction * len(ordered)) - 1))
    return ordered[index]


def bench_gateway(iterations: int = 5000) -> Dict[str, float]:
    provider = FakeProvider("fake-primary", latency_ms=0.0)
    routing = RoutingTable.from_dict({"smart": [{"provider": "fake-primary", "model": "fake-large"}]})
    gateway = LLMGateway(
        provider_map(provider),
        routing,
        limiter=TenantRateLimiter(capacity=10**9, refill_per_second=10**9),
    )
    request = CompletionRequest(
        tenant_id="bench",
        model="smart",
        messages=[ChatMessage(role="user", content="Summarize the audit evidence in one sentence.")],
    )
    latencies: List[float] = []
    started = time.perf_counter()
    for _ in range(iterations):
        before = time.perf_counter()
        gateway.complete(request)
        latencies.append((time.perf_counter() - before) * 1000.0)
    elapsed = time.perf_counter() - started
    return {
        "iterations": float(iterations),
        "total_ms": elapsed * 1000.0,
        "throughput_rps": iterations / elapsed if elapsed else 0.0,
        "mean_ms": statistics.fmean(latencies),
        "p50_ms": percentile(latencies, 0.50),
        "p95_ms": percentile(latencies, 0.95),
        "p99_ms": percentile(latencies, 0.99),
        "audit_events": float(len(gateway.audit)),
    }


def bench_audit(iterations: int = 5000) -> Dict[str, float]:
    provider = FakeProvider("fake-primary", latency_ms=0.0)
    routing = RoutingTable.from_dict({"smart": [{"provider": "fake-primary", "model": "fake-large"}]})
    gateway = LLMGateway(
        provider_map(provider),
        routing,
        limiter=TenantRateLimiter(capacity=10**9, refill_per_second=10**9),
    )
    request = CompletionRequest(
        tenant_id="bench",
        model="smart",
        messages=[ChatMessage(role="user", content="Audit test")],
    )
    for _ in range(iterations):
        gateway.complete(request)
    started = time.perf_counter()
    valid = gateway.audit.verify()
    elapsed = time.perf_counter() - started
    return {
        "events": float(len(gateway.audit)),
        "verify_ms": elapsed * 1000.0,
        "valid": 1.0 if valid else 0.0,
        "head_hash": gateway.audit.head_hash(),
    }


def bench_evals(iterations: int = 100, runs: int = 5) -> Dict[str, float]:
    provider = FakeProvider("fake-primary", responses=["Paris"] * iterations, latency_ms=0.0)
    routing = RoutingTable.from_dict({"smart": [{"provider": "fake-primary", "model": "fake-large"}]})
    gateway = LLMGateway(
        provider_map(provider),
        routing,
        limiter=TenantRateLimiter(capacity=10**9, refill_per_second=10**9),
    )
    dataset = Dataset(
        name="bench",
        version="v1",
        cases=[
            EvalCase(id=f"case-{index}", input="What is the capital of France?", expected="Paris")
            for index in range(iterations)
        ],
    )
    runner = EvalRunner(gateway, [ContainsScorer()])
    started = time.perf_counter()
    for _ in range(runs):
        runner.run(dataset, model_alias="smart", tenant_id="bench")
    elapsed = time.perf_counter() - started
    total_cases = iterations * runs
    return {
        "cases": float(total_cases),
        "total_ms": elapsed * 1000.0,
        "cases_per_second": total_cases / elapsed if elapsed else 0.0,
    }


def main() -> int:
    results = {
        "gateway": bench_gateway(),
        "audit": bench_audit(),
        "evals": bench_evals(),
    }
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
