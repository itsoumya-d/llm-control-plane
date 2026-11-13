"""Tests for Poolside Inference Infrastructure: GPU Workload Scheduler & K8s Controller."""

import pytest
from llm_control_plane import (
    GPUClusterNode,
    GPUNodeState,
    GPUWorkloadType,
    InferenceWorkload,
    PoolsideInferenceScheduler,
    PoolsideKubernetesController,
)


def test_scheduler_prefix_affinity_and_kv_hit():
    """Verify prefix-aware routing directs matching prompt contexts to the same GPU worker."""
    node_a = GPUClusterNode(node_id="gpu-node-01", gpu_count=8, total_vram_gb=640.0)
    node_b = GPUClusterNode(node_id="gpu-node-02", gpu_count=8, total_vram_gb=640.0)
    scheduler = PoolsideInferenceScheduler(nodes=[node_a, node_b])

    # Workload 1 with specific repo prefix
    shared_prefix = "import torch\nimport triton\n# Common poolside frontier kernel context"
    w1 = InferenceWorkload(
        workload_id="req-1",
        model_name="poolside-code-frontier-70b",
        workload_type=GPUWorkloadType.INTERACTIVE_ASSISTANT,
        required_vram_gb=160.0,
        prompt_tokens=4096,
        max_output_tokens=512,
        prompt_prefix=shared_prefix,
    )
    scheduler.submit_workload(w1)
    res1 = scheduler.schedule_next()
    assert res1 == ("req-1", "gpu-node-01")

    # Complete workload 1, node_a now has the prefix cached
    scheduler.complete_workload("req-1", tokens_generated=512, ttft_ms=10.2, itl_ms=6.8)

    # Workload 2 has the same prefix
    w2 = InferenceWorkload(
        workload_id="req-2",
        model_name="poolside-code-frontier-70b",
        workload_type=GPUWorkloadType.INTERACTIVE_ASSISTANT,
        required_vram_gb=160.0,
        prompt_tokens=4200,
        max_output_tokens=256,
        prompt_prefix=shared_prefix,
    )
    scheduler.submit_workload(w2)
    res2 = scheduler.schedule_next()
    # Must route to node_a because of prefix affinity
    assert res2 == ("req-2", "gpu-node-01")

    metric = scheduler.complete_workload("req-2", tokens_generated=256, ttft_ms=3.4, itl_ms=6.5)
    assert metric is not None
    assert metric.kv_cache_hit is True
    assert metric.ttft_ms < 5.0  # accelerated TTFT on KV cache hit


def test_priority_preemption_interactive_over_evals():
    """Verify interactive assistant workloads preempt background batch evals on saturated nodes."""
    node = GPUClusterNode(node_id="gpu-node-solo", gpu_count=8, total_vram_gb=160.0)
    scheduler = PoolsideInferenceScheduler(nodes=[node])

    # Fill node with low-priority EVALS workload
    eval_workload = InferenceWorkload(
        workload_id="eval-batch-01",
        model_name="poolside-code-frontier-70b",
        workload_type=GPUWorkloadType.EVALS,
        required_vram_gb=140.0,
        prompt_tokens=8192,
        max_output_tokens=1024,
        priority=20,
    )
    scheduler.submit_workload(eval_workload)
    sched_eval = scheduler.schedule_next()
    assert sched_eval == ("eval-batch-01", "gpu-node-solo")
    assert node.free_vram_gb == 20.0  # Node saturated

    # High-priority interactive assistant arrives needing 120 GB
    interactive_workload = InferenceWorkload(
        workload_id="user-assistant-01",
        model_name="poolside-code-frontier-70b",
        workload_type=GPUWorkloadType.INTERACTIVE_ASSISTANT,
        required_vram_gb=120.0,
        prompt_tokens=2048,
        max_output_tokens=512,
        priority=95,
    )
    scheduler.submit_workload(interactive_workload)

    # Scheduling next should trigger preemption of eval-batch-01 and schedule user-assistant-01
    sched_interactive = scheduler.schedule_next()
    assert sched_interactive == ("user-assistant-01", "gpu-node-solo")
    assert "user-assistant-01" in scheduler.active_workloads
    assert "eval-batch-01" not in scheduler.active_workloads
    # The preempted eval workload is back in pending queue
    assert any(w.workload_id == "eval-batch-01" for w in scheduler.pending_queue)


def test_kubernetes_controller_autoscaling_reconcile():
    """Verify Kubernetes Custom Controller reconcile loop handles queue-depth driven replica scaling."""
    node = GPUClusterNode(node_id="node-k8s-01", gpu_count=8, total_vram_gb=640.0)
    scheduler = PoolsideInferenceScheduler(nodes=[node])
    controller = PoolsideKubernetesController(scheduler, min_replicas=2, max_replicas=8)

    # Empty queue reconcile -> NOOP at min_replicas
    rec1 = controller.reconcile()
    assert rec1["action"] == "NOOP"
    assert rec1["current_replicas"] == 2

    # Inject 16 pending workloads
    for i in range(16):
        scheduler.pending_queue.append(
            InferenceWorkload(
                workload_id=f"queued-{i}",
                model_name="poolside-code-frontier-70b",
                workload_type=GPUWorkloadType.INTERACTIVE_ASSISTANT,
                required_vram_gb=80.0,
                prompt_tokens=1024,
                max_output_tokens=256,
            )
        )

    # Reconcile -> should scale up (2 + 16//4 = 6 replicas)
    rec2 = controller.reconcile()
    assert rec2["current_replicas"] == 6
    assert "SCALE_UP_2_TO_6" in rec2["action"]


def test_node_ecc_alert_drain_and_workload_requeue():
    """Verify hardware ECC error triggers automatic node drain and graceful workload migration."""
    node_failing = GPUClusterNode(node_id="gpu-node-ecc-fail", gpu_count=8, total_vram_gb=640.0)
    node_healthy = GPUClusterNode(node_id="gpu-node-healthy-02", gpu_count=8, total_vram_gb=640.0)
    scheduler = PoolsideInferenceScheduler(nodes=[node_failing, node_healthy])
    controller = PoolsideKubernetesController(scheduler, min_replicas=2, max_replicas=4)

    # Allocate workload to failing node
    w = InferenceWorkload(
        workload_id="active-task-migration",
        model_name="poolside-code-frontier-70b",
        workload_type=GPUWorkloadType.RL_ROLLOUT,
        required_vram_gb=160.0,
        prompt_tokens=2048,
        max_output_tokens=512,
    )
    scheduler.submit_workload(w)
    scheduler.schedule_next()
    assert w.assigned_node_id == "gpu-node-ecc-fail"

    # Simulate DCGM alert with 15 uncorrectable ECC errors
    alert_handled = controller.handle_node_ecc_alert("gpu-node-ecc-fail", ecc_count=15)
    assert alert_handled is True
    assert node_failing.state == GPUNodeState.DEGRADED_ECC
    assert "gpu-node-ecc-fail" in controller.drained_nodes

    # The evicted workload should now be requeued at front of queue
    assert scheduler.pending_queue[0].workload_id == "active-task-migration"
    assert scheduler.pending_queue[0].assigned_node_id is None

    # Reschedule moves it onto healthy node
    new_placement = scheduler.schedule_next()
    assert new_placement == ("active-task-migration", "gpu-node-healthy-02")


def test_cluster_stats_and_telemetry_metrics():
    """Verify cluster stats reporting and telemetry history aggregation."""
    node = GPUClusterNode(node_id="gpu-stat-node", gpu_count=8, total_vram_gb=640.0)
    scheduler = PoolsideInferenceScheduler(nodes=[node])

    w = InferenceWorkload(
        workload_id="stat-req-1",
        model_name="poolside-code-frontier-70b",
        workload_type=GPUWorkloadType.SPECULATIVE_VERIFICATION,
        required_vram_gb=128.0,
        prompt_tokens=1000,
        max_output_tokens=200,
    )
    scheduler.submit_workload(w)
    scheduler.schedule_next()

    stats = scheduler.cluster_stats()
    assert stats["node_count"] == 1
    assert stats["healthy_nodes"] == 1
    assert stats["total_gpus"] == 8
    assert stats["total_vram_gb"] == 640.0
    assert stats["allocated_vram_gb"] == 128.0
    assert stats["cluster_utilization_pct"] == 20.0
    assert stats["active_workloads_count"] == 1

    metric = scheduler.complete_workload("stat-req-1", tokens_generated=200, ttft_ms=11.4, itl_ms=7.1)
    assert metric.tokens_generated == 200
    assert len(scheduler.metrics_history) == 1
    post_stats = scheduler.cluster_stats()
    assert post_stats["allocated_vram_gb"] == 0.0
    assert post_stats["completed_workloads_count"] == 1
