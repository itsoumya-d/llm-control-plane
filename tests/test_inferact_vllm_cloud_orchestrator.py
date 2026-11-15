"""Tests for Inferact vLLM Cloud Orchestrator & Distributed GPU Cluster Operator."""

from llm_control_plane.inferact_vllm_cloud_orchestrator import (
    ClusterNode,
    DeploymentPhase,
    GPUDeviceState,
    GPUHardwareEvent,
    GPUHealthWatchdog,
    HardwareDegradationType,
    InferactVLLMCloudOrchestrator,
    PagedKVCacheState,
    VLLMInferenceClusterSpec,
    VLLMNodeLifecycleState,
)


def _build_test_node(node_id: str, gpu_count: int = 8) -> ClusterNode:
    node = ClusterNode(
        node_id=node_id,
        hostname=f"{node_id}.infra.inferact.internal",
        rack_id="rack-apac-sg-01",
    )
    for idx in range(gpu_count):
        node.gpus[idx] = GPUDeviceState(
            gpu_index=idx,
            model_name="NVIDIA H100 SXM5 80GB",
            temperature_c=58.0,
        )
    return node


def test_paged_kv_cache_state() -> None:
    """Verifies PagedAttention KV-cache block calculation and utilization metrics."""
    state = PagedKVCacheState.create(
        total_blocks=10000,
        used_blocks=8500,
        prefix_cache_hit_rate=0.74,
        queue_depth=65,
    )
    assert state.total_blocks == 10000
    assert state.used_blocks == 8500
    assert state.free_blocks == 1500
    assert state.utilization_ratio == 0.85
    assert state.pending_token_queue_depth == 65


def test_gpu_watchdog_degradation_fencing() -> None:
    """Verifies watchdog fences nodes reporting uncorrectable ECC or thermal violations."""
    watchdog = GPUHealthWatchdog()
    node = _build_test_node("sg-node-01", gpu_count=4)

    is_healthy, reason = watchdog.evaluate_node(node)
    assert is_healthy is True
    assert reason is None
    assert node.state == VLLMNodeLifecycleState.READY

    # Inject ECC error on GPU 2
    node.gpus[2].ecc_uncorrectable_count = 1
    is_healthy, reason = watchdog.evaluate_node(node)
    assert is_healthy is False
    assert "ECC error" in str(reason)
    assert node.state == VLLMNodeLifecycleState.FENCED_ISOLATED
    assert "nvidia.com/gpu-degraded:NoSchedule" in node.taints
    assert "sg-node-01" in watchdog.fenced_nodes


def test_orchestrator_placement_group_allocation() -> None:
    """Verifies Ray actor placement group satisfies Tensor Parallel co-locality."""
    spec = VLLMInferenceClusterSpec(
        model_id="deepseek-ai/DeepSeek-V3",
        revision="v1.0.0",
        tensor_parallel_size=4,
        pipeline_parallel_size=1,
        min_replicas=1,
        max_replicas=4,
    )
    orchestrator = InferactVLLMCloudOrchestrator(spec)
    node1 = _build_test_node("sg-node-01", gpu_count=8)
    orchestrator.register_node(node1)

    group = orchestrator.allocate_placement_group("group-01", revision="v1.0.0")
    assert group.group_id == "group-01"
    assert len(group.workers) == 4
    for rank, worker in enumerate(group.workers):
        assert worker.tp_rank == rank
        assert worker.node_id == "sg-node-01"
        assert node1.gpus[worker.gpu_index].assigned_actor_id == worker.actor_id


def test_autoscaling_demand_computation() -> None:
    """Verifies dynamic autoscaling demand responds to KV-cache pressure and queue depth."""
    spec = VLLMInferenceClusterSpec(
        model_id="meta-llama/Llama-3.1-70B-Instruct",
        revision="v1.0.0",
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
        min_replicas=2,
        max_replicas=8,
        target_kv_cache_utilization=0.80,
    )
    orchestrator = InferactVLLMCloudOrchestrator(spec)

    # Moderate state -> stays at min_replicas
    moderate_kv = PagedKVCacheState.create(
        total_blocks=10000, used_blocks=5000, prefix_cache_hit_rate=0.5, queue_depth=10
    )
    desired = orchestrator.compute_autoscaling_demand(moderate_kv)
    assert desired == 2

    # High pressure state -> scales up
    burst_kv = PagedKVCacheState.create(
        total_blocks=10000, used_blocks=8800, prefix_cache_hit_rate=0.3, queue_depth=120
    )
    desired = orchestrator.compute_autoscaling_demand(burst_kv)
    assert desired > 2
    assert desired <= 8


def test_reconciliation_and_node_failure_recovery() -> None:
    """Tests cluster reconciliation and automated failover when a node degrades."""
    spec = VLLMInferenceClusterSpec(
        model_id="Qwen/Qwen2.5-Coder-32B",
        revision="v1.0.0",
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
        min_replicas=2,
        max_replicas=6,
    )
    orchestrator = InferactVLLMCloudOrchestrator(spec)
    node1 = _build_test_node("sg-node-01", gpu_count=4)
    node2 = _build_test_node("sg-node-02", gpu_count=4)
    orchestrator.register_node(node1)
    orchestrator.register_node(node2)

    # Reconcile up to 2 replicas
    res = orchestrator.reconcile_replicas(desired_replicas=2)
    assert res["scaled_up"] == 2
    assert orchestrator.status.active_replicas == 2

    # Simulate hardware degradation event on node1
    event = GPUHardwareEvent(
        node_id="sg-node-01",
        gpu_index=0,
        degradation_type=HardwareDegradationType.XID_ECC_UNCORRECTABLE,
        error_code=48,
        severity="CRITICAL",
        message="Double-bit uncorrectable ECC error detected in HBM3",
    )
    recovery_result = orchestrator.handle_node_failure(node_id="sg-node-01", event=event)
    assert recovery_result["status"] == "node_fenced"
    assert recovery_result["fenced_nodes_total"] == 1
    assert orchestrator.status.total_fenced_nodes == 1


def test_blue_green_canary_rollout_and_promotion() -> None:
    """Verifies zero-downtime blue/green rollout, traffic shift, and promotion."""
    spec = VLLMInferenceClusterSpec(
        model_id="vllm-engine-v2",
        revision="v2.1.0",
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
        min_replicas=2,
        max_replicas=6,
    )
    orchestrator = InferactVLLMCloudOrchestrator(spec)
    node1 = _build_test_node("sg-node-01", gpu_count=8)
    orchestrator.register_node(node1)
    orchestrator.reconcile_replicas(desired_replicas=2)

    # Initiate canary rollout
    rollout = orchestrator.initiate_blue_green_rollout(target_revision="v2.2.0-rc1", canary_weight=0.25)
    assert rollout["status"] == "canary_active"
    assert rollout["traffic_split"]["canary_revision"] == 0.25
    assert orchestrator.status.current_phase == DeploymentPhase.CANARY_PROGRESSING

    # Promote canary
    promotion = orchestrator.promote_canary_to_stable(new_revision="v2.2.0-rc1")
    assert promotion["status"] == "promoted_stable"
    assert orchestrator.status.current_phase == DeploymentPhase.STABLE
    assert orchestrator.spec.revision == "v2.2.0-rc1"


def test_telemetry_and_prometheus_metrics_export() -> None:
    """Verifies p95 SLA telemetry calculations and Prometheus metric formatting."""
    spec = VLLMInferenceClusterSpec(
        model_id="mistralai/Mistral-Large-2407",
        revision="v1.0.0",
        tensor_parallel_size=4,
        pipeline_parallel_size=1,
        min_replicas=1,
        max_replicas=4,
    )
    orchestrator = InferactVLLMCloudOrchestrator(spec)
    for latency in [85.0, 92.0, 110.0, 140.0, 95.0]:
        orchestrator.record_telemetry(ttft_ms=latency, tpot_ms=22.0)

    assert orchestrator.status.p95_ttft_ms > 0
    assert orchestrator.status.p95_tpot_ms == 22.0

    metrics = orchestrator.generate_prometheus_metrics()
    assert "vllm_active_replicas" in metrics
    assert "vllm_kv_cache_utilization_ratio" in metrics
    assert "vllm_p95_ttft_seconds" in metrics
    assert 'model="mistralai/Mistral-Large-2407"' in metrics
