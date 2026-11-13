"""Tests for GPU cluster FinOps, KV-cache telemetry, and reliability SLO monitoring."""

import threading

from llm_control_plane import (
    STANDARD_GPU_SPECS,
    GPUClusterFinOpsTracker,
    InferenceBatchRecord,
    ReliabilitySLOMonitor,
)


def test_gpu_model_specs_default() -> None:
    assert "h100-sxm" in STANDARD_GPU_SPECS
    assert "a100-80gb" in STANDARD_GPU_SPECS
    h100 = STANDARD_GPU_SPECS["h100-sxm"]
    assert h100.vram_gb == 80.0
    assert h100.hourly_cost_usd == 3.35


def test_inference_batch_record_properties() -> None:
    rec = InferenceBatchRecord(
        batch_id="b-1",
        timestamp=1700000000.0,
        node_id="node-gpu-1",
        gpu_type="h100-sxm",
        model_id="llama-3-70b-instruct",
        concurrency=16,
        prompt_tokens=4000,
        generation_tokens=1000,
        ttft_ms=120.0,
        tpot_ms=22.0,
        kv_cache_allocated_mb=65536.0,
        kv_cache_used_mb=32768.0,
        prefix_cache_hit_tokens=800,
    )
    assert rec.total_tokens == 5000
    assert rec.kv_cache_utilization_pct == 50.0

    empty_cache = InferenceBatchRecord(
        batch_id="b-0",
        timestamp=1700000000.0,
        node_id="node-gpu-0",
        gpu_type="t4",
        model_id="slm",
        concurrency=1,
        prompt_tokens=100,
        generation_tokens=50,
        ttft_ms=50.0,
        tpot_ms=10.0,
        kv_cache_allocated_mb=0.0,
        kv_cache_used_mb=0.0,
    )
    assert empty_cache.kv_cache_utilization_pct == 0.0


def test_finops_tracker_cost_and_prefix_cache() -> None:
    tracker = GPUClusterFinOpsTracker()
    now = 1700000000.0

    # Add 2 batches on an A100-80GB node ($2.20/hr)
    # Total tokens = 10,000 + 15,000 = 25,000
    tracker.record_batch(
        InferenceBatchRecord(
            batch_id="b-1",
            timestamp=now,
            node_id="node-a100-1",
            gpu_type="a100-80gb",
            model_id="mixtral-8x7b",
            concurrency=8,
            prompt_tokens=8000,
            generation_tokens=2000,
            ttft_ms=150.0,
            tpot_ms=25.0,
            kv_cache_allocated_mb=80000.0,
            kv_cache_used_mb=40000.0,
            prefix_cache_hit_tokens=2000,
        )
    )
    tracker.record_batch(
        InferenceBatchRecord(
            batch_id="b-2",
            timestamp=now + 1800.0,  # 0.5 hour later
            node_id="node-a100-1",
            gpu_type="a100-80gb",
            model_id="mixtral-8x7b",
            concurrency=8,
            prompt_tokens=12000,
            generation_tokens=3000,
            ttft_ms=160.0,
            tpot_ms=26.0,
            kv_cache_allocated_mb=80000.0,
            kv_cache_used_mb=44000.0,
            prefix_cache_hit_tokens=4000,
        )
    )

    records = tracker.get_records(node_id="node-a100-1")
    assert len(records) == 2

    # Total prompt tokens = 20,000; hits = 6,000 => 30.0%
    prefix_savings = tracker.calculate_prefix_cache_savings_pct()
    assert prefix_savings == 30.0

    cost_per_m = tracker.calculate_cost_per_million_tokens(gpu_type="a100-80gb")
    assert cost_per_m > 0.0

    # Edge cases
    assert tracker.calculate_cost_per_million_tokens(gpu_type="nonexistent") == 0.0
    empty_tracker = GPUClusterFinOpsTracker()
    assert empty_tracker.calculate_prefix_cache_savings_pct() == 0.0
    assert empty_tracker.calculate_cost_per_million_tokens("a100-80gb") == 0.0


def test_finops_right_sizing_evaluation() -> None:
    tracker = GPUClusterFinOpsTracker()
    now = 1700000000.0

    # Overprovisioned node: H100 with only 15% KV-cache usage
    tracker.record_batch(
        InferenceBatchRecord(
            batch_id="b-h100",
            timestamp=now,
            node_id="node-overprovisioned",
            gpu_type="h100-sxm",
            model_id="contact-agent-v1",
            concurrency=4,
            prompt_tokens=1000,
            generation_tokens=200,
            ttft_ms=90.0,
            tpot_ms=18.0,
            kv_cache_allocated_mb=80000.0,
            kv_cache_used_mb=12000.0,  # 15%
        )
    )

    # Underprovisioned node: T4 experiencing OOM
    tracker.record_batch(
        InferenceBatchRecord(
            batch_id="b-oom",
            timestamp=now,
            node_id="node-underprovisioned",
            gpu_type="t4",
            model_id="large-agent",
            concurrency=8,
            prompt_tokens=5000,
            generation_tokens=0,
            ttft_ms=500.0,
            tpot_ms=0.0,
            kv_cache_allocated_mb=16000.0,
            kv_cache_used_mb=16000.0,
            is_oom_error=True,
            is_success=False,
        )
    )

    # Healthy node: L4 with 70% KV-cache usage
    tracker.record_batch(
        InferenceBatchRecord(
            batch_id="b-healthy",
            timestamp=now,
            node_id="node-healthy",
            gpu_type="l4",
            model_id="analytics-embedder",
            concurrency=8,
            prompt_tokens=3000,
            generation_tokens=800,
            ttft_ms=110.0,
            tpot_ms=20.0,
            kv_cache_allocated_mb=24000.0,
            kv_cache_used_mb=16800.0,  # 70%
        )
    )

    recs = tracker.evaluate_right_sizing()
    rec_by_node = {r.node_id: r for r in recs}

    assert rec_by_node["node-overprovisioned"].recommended_action.startswith("DOWNSIZE")
    assert rec_by_node["node-overprovisioned"].potential_monthly_savings_usd > 500.0

    assert rec_by_node["node-underprovisioned"].recommended_action == "SCALE_UP_VRAM"
    assert "OOM" in rec_by_node["node-underprovisioned"].rationale

    assert rec_by_node["node-healthy"].recommended_action == "RIGHT_SIZED"


def test_reliability_slo_monitor() -> None:
    monitor = ReliabilitySLOMonitor(
        target_ttft_p95_ms=200.0,
        target_tpot_p99_ms=30.0,
        target_success_rate=0.99,
    )

    # Initial empty snapshot
    empty_snap = monitor.evaluate_slo()
    assert empty_snap.total_requests == 0
    assert empty_snap.success_rate == 1.0
    assert not empty_snap.is_slo_breached

    # Record 99 successful requests well within latency budgets
    now = 1700000000.0
    for i in range(99):
        monitor.record(
            InferenceBatchRecord(
                batch_id=f"req-{i}",
                timestamp=now + i,
                node_id="node-1",
                gpu_type="h100-sxm",
                model_id="level-llm",
                concurrency=4,
                prompt_tokens=500,
                generation_tokens=150,
                ttft_ms=120.0 + (i % 20),
                tpot_ms=18.0 + (i % 5),
                kv_cache_allocated_mb=80000.0,
                kv_cache_used_mb=35000.0,
            )
        )

    # 1 failure
    monitor.record(
        InferenceBatchRecord(
            batch_id="req-failed",
            timestamp=now + 100,
            node_id="node-1",
            gpu_type="h100-sxm",
            model_id="level-llm",
            concurrency=4,
            prompt_tokens=500,
            generation_tokens=0,
            ttft_ms=0.0,
            tpot_ms=0.0,
            kv_cache_allocated_mb=80000.0,
            kv_cache_used_mb=0.0,
            is_oom_error=True,
            is_success=False,
        )
    )

    snap = monitor.evaluate_slo()
    assert snap.total_requests == 100
    assert snap.successful_requests == 99
    assert snap.oom_errors == 1
    assert snap.success_rate == 0.99
    assert snap.ttft_p95_ms > 100.0
    assert snap.tpot_p99_ms > 15.0
    assert not snap.is_slo_breached

    # Prometheus exposition output check
    prom_text = monitor.render_prometheus_metrics()
    assert "inference_requests_total 100" in prom_text
    assert "inference_requests_successful_total 99" in prom_text
    assert "inference_oom_errors_total 1" in prom_text
    assert "inference_slo_error_budget_burn_rate" in prom_text


def test_slo_monitor_breach_detection() -> None:
    monitor = ReliabilitySLOMonitor(
        target_ttft_p95_ms=100.0,
        target_tpot_p99_ms=20.0,
        target_success_rate=0.99,
    )
    now = 1700000000.0
    # Add requests with high latency breaching TTFT
    for i in range(10):
        monitor.record(
            InferenceBatchRecord(
                batch_id=f"lat-{i}",
                timestamp=now + i,
                node_id="node-slow",
                gpu_type="t4",
                model_id="heavy-model",
                concurrency=1,
                prompt_tokens=2000,
                generation_tokens=500,
                ttft_ms=350.0,  # breaches 100.0ms target
                tpot_ms=45.0,  # breaches 20.0ms target
                kv_cache_allocated_mb=16000.0,
                kv_cache_used_mb=14000.0,
            )
        )
    snap = monitor.evaluate_slo()
    assert snap.is_slo_breached


def test_thread_safety_finops() -> None:
    tracker = GPUClusterFinOpsTracker()
    threads = []
    now = 1700000000.0

    def worker(worker_id: int) -> None:
        for i in range(50):
            tracker.record_batch(
                InferenceBatchRecord(
                    batch_id=f"w{worker_id}-{i}",
                    timestamp=now + i,
                    node_id=f"node-{worker_id}",
                    gpu_type="l4",
                    model_id="agent",
                    concurrency=2,
                    prompt_tokens=100,
                    generation_tokens=50,
                    ttft_ms=80.0,
                    tpot_ms=15.0,
                    kv_cache_allocated_mb=24000.0,
                    kv_cache_used_mb=12000.0,
                )
            )

    for w in range(4):
        t = threading.Thread(target=worker, args=(w,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    records = tracker.get_records()
    assert len(records) == 200
