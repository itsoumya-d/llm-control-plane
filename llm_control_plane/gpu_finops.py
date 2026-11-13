"""GPU cluster FinOps telemetry, KV-cache throughput profiling, and reliability SLO instrumentation.

Positioned for AI-native production clusters running LLM inference services,
providing exact $/1M token unit economics, node right-sizing, KV-cache utilization,
and multi-window SRE SLO error budget tracking.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class GPUModelSpec:
    """Hardware specifications and cost profile for an accelerator node."""

    gpu_type: str
    vram_gb: float
    hourly_cost_usd: float
    max_memory_bandwidth_gbps: float


STANDARD_GPU_SPECS: Dict[str, GPUModelSpec] = {
    "h100-sxm": GPUModelSpec(
        gpu_type="h100-sxm",
        vram_gb=80.0,
        hourly_cost_usd=3.35,
        max_memory_bandwidth_gbps=3350.0,
    ),
    "a100-80gb": GPUModelSpec(
        gpu_type="a100-80gb",
        vram_gb=80.0,
        hourly_cost_usd=2.20,
        max_memory_bandwidth_gbps=2039.0,
    ),
    "a100-40gb": GPUModelSpec(
        gpu_type="a100-40gb",
        vram_gb=40.0,
        hourly_cost_usd=1.50,
        max_memory_bandwidth_gbps=1555.0,
    ),
    "l4": GPUModelSpec(
        gpu_type="l4",
        vram_gb=24.0,
        hourly_cost_usd=0.70,
        max_memory_bandwidth_gbps=300.0,
    ),
    "t4": GPUModelSpec(
        gpu_type="t4",
        vram_gb=16.0,
        hourly_cost_usd=0.35,
        max_memory_bandwidth_gbps=320.0,
    ),
}


@dataclass
class InferenceBatchRecord:
    """Telemetry capture for a completed or failed inference batch execution."""

    batch_id: str
    timestamp: float
    node_id: str
    gpu_type: str
    model_id: str
    concurrency: int
    prompt_tokens: int
    generation_tokens: int
    ttft_ms: float
    tpot_ms: float
    kv_cache_allocated_mb: float
    kv_cache_used_mb: float
    prefix_cache_hit_tokens: int = 0
    is_oom_error: bool = False
    is_success: bool = True

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.generation_tokens

    @property
    def kv_cache_utilization_pct(self) -> float:
        if self.kv_cache_allocated_mb <= 0:
            return 0.0
        return min(100.0, (self.kv_cache_used_mb / self.kv_cache_allocated_mb) * 100.0)


@dataclass(frozen=True)
class RightSizingRecommendation:
    """Kubernetes pod/node pool right-sizing recommendation from FinOps telemetry."""

    node_id: str
    current_gpu_type: str
    recommended_action: str
    peak_kv_cache_utilization_pct: float
    avg_kv_cache_utilization_pct: float
    potential_monthly_savings_usd: float
    rationale: str


@dataclass(frozen=True)
class SLISnapshot:
    """Service Level Indicator metrics for latency, throughput, and error budget."""

    total_requests: int
    successful_requests: int
    oom_errors: int
    success_rate: float
    ttft_p95_ms: float
    tpot_p99_ms: float
    target_success_rate: float
    error_budget_remaining_pct: float
    burn_rate: float
    is_slo_breached: bool


def _percentile(values: List[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[idx]


class GPUClusterFinOpsTracker:
    """Tracks GPU memory, throughput, and token unit costs across heterogeneous nodes."""

    def __init__(
        self,
        specs: Optional[Dict[str, GPUModelSpec]] = None,
    ) -> None:
        self._specs = dict(specs or STANDARD_GPU_SPECS)
        self._records: List[InferenceBatchRecord] = []
        self._lock = threading.Lock()

    def record_batch(self, record: InferenceBatchRecord) -> None:
        """Record an inference batch telemetry record in thread-safe storage."""
        with self._lock:
            self._records.append(record)

    def get_records(
        self,
        since_timestamp: Optional[float] = None,
        node_id: Optional[str] = None,
        gpu_type: Optional[str] = None,
    ) -> List[InferenceBatchRecord]:
        """Query stored records with optional time and node filters."""
        with self._lock:
            res = list(self._records)

        if since_timestamp is not None:
            res = [r for r in res if r.timestamp >= since_timestamp]
        if node_id is not None:
            res = [r for r in res if r.node_id == node_id]
        if gpu_type is not None:
            res = [r for r in res if r.gpu_type == gpu_type]
        return res

    def calculate_cost_per_million_tokens(
        self,
        gpu_type: str,
        since_timestamp: Optional[float] = None,
    ) -> float:
        """Calculate amortized compute cost per 1M processed tokens on a GPU type.

        Cost = (Active GPU Hours * Hourly Rate) / (Total Processed Tokens / 1,000,000)
        """
        records = self.get_records(since_timestamp=since_timestamp, gpu_type=gpu_type)
        if not records:
            return 0.0

        spec = self._specs.get(gpu_type)
        if not spec:
            return 0.0

        total_tokens = sum(r.total_tokens for r in records if r.is_success)
        if total_tokens <= 0:
            return 0.0

        # Compute duration span covered by the batch run
        timestamps = [r.timestamp for r in records]
        duration_hours = max(0.001, (max(timestamps) - min(timestamps)) / 3600.0)
        # If all executed simultaneously or within short window, approximate by total processing time
        if duration_hours <= 0.001:
            total_duration_sec = sum((r.ttft_ms + (r.tpot_ms * r.generation_tokens)) / 1000.0 for r in records)
            duration_hours = max(0.0001, total_duration_sec / 3600.0)

        total_compute_cost = duration_hours * spec.hourly_cost_usd
        return (total_compute_cost / total_tokens) * 1_000_000.0

    def calculate_prefix_cache_savings_pct(
        self,
        since_timestamp: Optional[float] = None,
    ) -> float:
        """Calculate token re-computation compute avoidance via KV-cache prefix hits."""
        records = self.get_records(since_timestamp=since_timestamp)
        if not records:
            return 0.0

        total_prompt_tokens = sum(r.prompt_tokens for r in records)
        if total_prompt_tokens <= 0:
            return 0.0

        total_hit_tokens = sum(r.prefix_cache_hit_tokens for r in records)
        return min(100.0, (total_hit_tokens / total_prompt_tokens) * 100.0)

    def evaluate_right_sizing(
        self,
        overprovision_threshold_pct: float = 40.0,
        underprovision_oom_threshold: int = 1,
    ) -> List[RightSizingRecommendation]:
        """Analyze KV-cache memory headroom and recommend node right-sizing."""
        with self._lock:
            records = list(self._records)

        node_groups: Dict[str, List[InferenceBatchRecord]] = {}
        for r in records:
            node_groups.setdefault(r.node_id, []).append(r)

        recommendations: List[RightSizingRecommendation] = []
        hours_per_month = 730.0

        for node_id, batch_list in node_groups.items():
            if not batch_list:
                continue

            current_gpu = batch_list[0].gpu_type
            spec = self._specs.get(current_gpu)
            current_monthly_cost = (spec.hourly_cost_usd if spec else 0.0) * hours_per_month

            utils = [r.kv_cache_utilization_pct for r in batch_list]
            peak_util = max(utils)
            avg_util = sum(utils) / len(utils)
            oom_count = sum(1 for r in batch_list if r.is_oom_error)

            if oom_count >= underprovision_oom_threshold:
                recommendations.append(
                    RightSizingRecommendation(
                        node_id=node_id,
                        current_gpu_type=current_gpu,
                        recommended_action="SCALE_UP_VRAM",
                        peak_kv_cache_utilization_pct=peak_util,
                        avg_kv_cache_utilization_pct=avg_util,
                        potential_monthly_savings_usd=0.0,
                        rationale=(
                            f"Observed {oom_count} OOM errors; KV-cache exceeded capacity. "
                            "Scale node pool to higher VRAM tier."
                        ),
                    )
                )
            elif peak_util < overprovision_threshold_pct and current_gpu in ("h100-sxm", "a100-80gb"):
                # Candidate for down-tiering to A100-40GB or L4
                target_gpu = "a100-40gb" if current_gpu == "h100-sxm" else "l4"
                target_spec = self._specs.get(target_gpu)
                target_cost = (target_spec.hourly_cost_usd if target_spec else 0.0) * hours_per_month
                savings = max(0.0, current_monthly_cost - target_cost)

                recommendations.append(
                    RightSizingRecommendation(
                        node_id=node_id,
                        current_gpu_type=current_gpu,
                        recommended_action=f"DOWNSIZE_TO_{target_gpu.upper()}",
                        peak_kv_cache_utilization_pct=peak_util,
                        avg_kv_cache_utilization_pct=avg_util,
                        potential_monthly_savings_usd=round(savings, 2),
                        rationale=(
                            f"Peak KV-cache usage is only {peak_util:.1f}% (<{overprovision_threshold_pct}%). "
                            f"Safe to consolidate into {target_gpu}."
                        ),
                    )
                )
            else:
                recommendations.append(
                    RightSizingRecommendation(
                        node_id=node_id,
                        current_gpu_type=current_gpu,
                        recommended_action="RIGHT_SIZED",
                        peak_kv_cache_utilization_pct=peak_util,
                        avg_kv_cache_utilization_pct=avg_util,
                        potential_monthly_savings_usd=0.0,
                        rationale=(
                            f"KV-cache utilization healthy (peak {peak_util:.1f}%, avg {avg_util:.1f}%). "
                            "Node pool right-sized."
                        ),
                    )
                )

        return recommendations


class ReliabilitySLOMonitor:
    """Tracks SLIs (TTFT, TPOT, Success Rate) against SRE Service Level Objectives."""

    def __init__(
        self,
        target_ttft_p95_ms: float = 250.0,
        target_tpot_p99_ms: float = 35.0,
        target_success_rate: float = 0.999,
    ) -> None:
        self.target_ttft_p95_ms = target_ttft_p95_ms
        self.target_tpot_p99_ms = target_tpot_p99_ms
        self.target_success_rate = target_success_rate
        self._samples: List[InferenceBatchRecord] = []
        self._lock = threading.Lock()

    def record(self, sample: InferenceBatchRecord) -> None:
        """Record an inference batch observation for SLO compliance."""
        with self._lock:
            self._samples.append(sample)

    def evaluate_slo(
        self,
        since_timestamp: Optional[float] = None,
    ) -> SLISnapshot:
        """Compute SLI percentiles and remaining error budget burn rate."""
        with self._lock:
            samples = list(self._samples)

        if since_timestamp is not None:
            samples = [s for s in samples if s.timestamp >= since_timestamp]

        total = len(samples)
        if total == 0:
            return SLISnapshot(
                total_requests=0,
                successful_requests=0,
                oom_errors=0,
                success_rate=1.0,
                ttft_p95_ms=0.0,
                tpot_p99_ms=0.0,
                target_success_rate=self.target_success_rate,
                error_budget_remaining_pct=100.0,
                burn_rate=0.0,
                is_slo_breached=False,
            )

        successes = sum(1 for s in samples if s.is_success)
        oom_errors = sum(1 for s in samples if s.is_oom_error)
        actual_success_rate = successes / total

        successful_samples = [s for s in samples if s.is_success]
        ttft_values = [s.ttft_ms for s in successful_samples]
        tpot_values = [s.tpot_ms for s in successful_samples]

        ttft_p95 = _percentile(ttft_values, 0.95)
        tpot_p99 = _percentile(tpot_values, 0.99)

        # Allowed failure rate: 1.0 - 0.999 = 0.001
        allowed_error_fraction = 1.0 - self.target_success_rate
        actual_error_fraction = 1.0 - actual_success_rate

        if allowed_error_fraction > 0:
            burn_rate = actual_error_fraction / allowed_error_fraction
            error_budget_used_pct = min(100.0, burn_rate * 100.0)
            remaining_pct = max(0.0, 100.0 - error_budget_used_pct)
        else:
            burn_rate = 0.0 if actual_error_fraction == 0 else 999.0
            remaining_pct = 100.0 if actual_error_fraction == 0 else 0.0

        is_breached = (
            actual_success_rate < self.target_success_rate
            or ttft_p95 > self.target_ttft_p95_ms
            or tpot_p99 > self.target_tpot_p99_ms
            or remaining_pct < 0.0
        )

        return SLISnapshot(
            total_requests=total,
            successful_requests=successes,
            oom_errors=oom_errors,
            success_rate=actual_success_rate,
            ttft_p95_ms=ttft_p95,
            tpot_p99_ms=tpot_p99,
            target_success_rate=self.target_success_rate,
            error_budget_remaining_pct=round(remaining_pct, 2),
            burn_rate=round(burn_rate, 3),
            is_slo_breached=is_breached,
        )

    def render_prometheus_metrics(self) -> str:
        """Render Prometheus exposition lines for telemetry scrapers."""
        snapshot = self.evaluate_slo()
        lines = [
            "# HELP inference_requests_total Total count of batch inference requests",
            "# TYPE inference_requests_total counter",
            f"inference_requests_total {snapshot.total_requests}",
            "# HELP inference_requests_successful_total Total count of successful inference requests",
            "# TYPE inference_requests_successful_total counter",
            f"inference_requests_successful_total {snapshot.successful_requests}",
            "# HELP inference_oom_errors_total Total count of GPU out-of-memory errors",
            "# TYPE inference_oom_errors_total counter",
            f"inference_oom_errors_total {snapshot.oom_errors}",
            "# HELP inference_ttft_p95_milliseconds P95 time to first token",
            "# TYPE inference_ttft_p95_milliseconds gauge",
            f"inference_ttft_p95_milliseconds {snapshot.ttft_p95_ms:.2f}",
            "# HELP inference_tpot_p99_milliseconds P99 time per output token",
            "# TYPE inference_tpot_p99_milliseconds gauge",
            f"inference_tpot_p99_milliseconds {snapshot.tpot_p99_ms:.2f}",
            "# HELP inference_slo_error_budget_burn_rate Error budget burn rate multiple (1.0 = normal consumption)",
            "# TYPE inference_slo_error_budget_burn_rate gauge",
            f"inference_slo_error_budget_burn_rate {snapshot.burn_rate:.3f}",
        ]
        return "\n".join(lines) + "\n"
