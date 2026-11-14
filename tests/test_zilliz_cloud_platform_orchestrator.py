"""Unit tests for Zilliz Cloud Platform and Vector Lakebase Orchestrator."""

import pytest

from llm_control_plane.zilliz_cloud_platform_orchestrator import (
    CloudProvider,
    ClusterState,
    IndexType,
    MilvusClusterSpec,
    TenantQuota,
    ZillizCloudPlatformOrchestrator,
)


@pytest.fixture
def orchestrator() -> ZillizCloudPlatformOrchestrator:
    return ZillizCloudPlatformOrchestrator(
        slo_p99_latency_threshold_ms=15.0,
        slo_error_rate_threshold=0.001,
    )


@pytest.fixture
def default_spec() -> MilvusClusterSpec:
    return MilvusClusterSpec(
        cluster_id="milvus-cluster-prod-01",
        tenant_id="tenant-ai-frontier",
        cloud_provider=CloudProvider.AWS,
        region="us-west-2",
        milvus_version="v2.4.5",
        query_nodes=2,
        index_nodes=1,
        data_nodes=2,
        cpu_per_node=4,
        memory_gb_per_node=16,
        storage_gb=500,
        vector_dimension=1536,
        index_type=IndexType.HNSW,
    )


def test_tenant_registration_and_quota(orchestrator: ZillizCloudPlatformOrchestrator) -> None:
    custom_quota = TenantQuota(max_clusters=2, max_cpu_cores=32, max_memory_gb=128)
    orchestrator.register_tenant("tenant-custom", custom_quota)
    quota = orchestrator.get_tenant_quota("tenant-custom")
    assert quota.max_clusters == 2
    assert quota.max_cpu_cores == 32
    assert quota.max_memory_gb == 128


def test_cluster_provisioning_success(
    orchestrator: ZillizCloudPlatformOrchestrator, default_spec: MilvusClusterSpec
) -> None:
    runtime = orchestrator.provision_cluster(default_spec)
    assert runtime.state == ClusterState.RUNNING
    assert runtime.active_query_nodes == 2
    assert runtime.active_index_nodes == 1
    assert runtime.active_data_nodes == 2
    assert runtime.spec.total_cpu_cores == 20  # (2 + 1 + 2) * 4
    assert runtime.spec.total_memory_gb == 80  # (2 + 1 + 2) * 16
    assert runtime.version_history == ["v2.4.5"]


def test_cluster_provisioning_duplicate_id_fails(
    orchestrator: ZillizCloudPlatformOrchestrator, default_spec: MilvusClusterSpec
) -> None:
    orchestrator.provision_cluster(default_spec)
    with pytest.raises(ValueError, match="already exists"):
        orchestrator.provision_cluster(default_spec)


def test_cluster_provisioning_exceeds_quota(orchestrator: ZillizCloudPlatformOrchestrator) -> None:
    orchestrator.register_tenant("tenant-limited", TenantQuota(max_clusters=1, max_cpu_cores=16, max_memory_gb=64))
    spec1 = MilvusClusterSpec(
        cluster_id="cluster-01",
        tenant_id="tenant-limited",
        cloud_provider=CloudProvider.GCP,
        region="us-central1",
        query_nodes=1,
        index_nodes=1,
        data_nodes=1,
        cpu_per_node=4,
        memory_gb_per_node=16,
    )
    orchestrator.provision_cluster(spec1)

    spec2 = MilvusClusterSpec(
        cluster_id="cluster-02",
        tenant_id="tenant-limited",
        cloud_provider=CloudProvider.GCP,
        region="us-central1",
    )
    with pytest.raises(PermissionError, match="exceeds max clusters quota"):
        orchestrator.provision_cluster(spec2)


def test_manual_cluster_scaling(
    orchestrator: ZillizCloudPlatformOrchestrator, default_spec: MilvusClusterSpec
) -> None:
    orchestrator.provision_cluster(default_spec)
    scaled = orchestrator.scale_cluster(default_spec.cluster_id, target_query_nodes=4, target_index_nodes=2)
    assert scaled.active_query_nodes == 4
    assert scaled.active_index_nodes == 2
    assert scaled.state == ClusterState.RUNNING


def test_manual_cluster_scaling_quota_rejection(
    orchestrator: ZillizCloudPlatformOrchestrator, default_spec: MilvusClusterSpec
) -> None:
    orchestrator.register_tenant(default_spec.tenant_id, TenantQuota(max_cpu_cores=24, max_memory_gb=100))
    orchestrator.provision_cluster(default_spec)
    # Target scale requires (8 + 2 + 2) * 4 = 48 cores > 24 quota
    with pytest.raises(PermissionError, match="exceeds tenant CPU"):
        orchestrator.scale_cluster(default_spec.cluster_id, target_query_nodes=8, target_index_nodes=2)


def test_auto_scaling_heuristics(
    orchestrator: ZillizCloudPlatformOrchestrator, default_spec: MilvusClusterSpec
) -> None:
    orchestrator.provision_cluster(default_spec)

    # Scale UP triggered by high P99 latency
    triggered, msg = orchestrator.evaluate_auto_scaling(default_spec.cluster_id, current_qps=800.0, p99_latency_ms=22.5)
    assert triggered is True
    assert "Scaled UP" in msg
    assert orchestrator._clusters[default_spec.cluster_id].active_query_nodes == 3

    # Scale DOWN triggered by low latency and minimal QPS
    # First scale to 4 nodes
    orchestrator.scale_cluster(default_spec.cluster_id, target_query_nodes=4, target_index_nodes=1)
    triggered_down, msg_down = orchestrator.evaluate_auto_scaling(
        default_spec.cluster_id, current_qps=50.0, p99_latency_ms=4.0
    )
    assert triggered_down is True
    assert "Scaled DOWN" in msg_down
    assert orchestrator._clusters[default_spec.cluster_id].active_query_nodes == 3


def test_canary_rolling_upgrade_success(
    orchestrator: ZillizCloudPlatformOrchestrator, default_spec: MilvusClusterSpec
) -> None:
    orchestrator.provision_cluster(default_spec)
    success, msg = orchestrator.execute_rolling_upgrade(
        default_spec.cluster_id, target_version="v2.4.8", canary_steps=3, simulated_error_rate=0.0001
    )
    assert success is True
    assert "Successfully upgraded" in msg
    assert orchestrator._clusters[default_spec.cluster_id].spec.milvus_version == "v2.4.8"
    assert orchestrator._clusters[default_spec.cluster_id].version_history == ["v2.4.5", "v2.4.8"]


def test_canary_rolling_upgrade_rollback_on_error(
    orchestrator: ZillizCloudPlatformOrchestrator, default_spec: MilvusClusterSpec
) -> None:
    orchestrator.provision_cluster(default_spec)
    success, msg = orchestrator.execute_rolling_upgrade(
        default_spec.cluster_id, target_version="v2.5.0-rc1", canary_steps=2, simulated_error_rate=0.05
    )
    assert success is False
    assert "Rolled back to v2.4.5" in msg
    # Cluster remains on original version
    assert orchestrator._clusters[default_spec.cluster_id].spec.milvus_version == "v2.4.5"
    assert orchestrator._clusters[default_spec.cluster_id].state == ClusterState.RUNNING


def test_backup_and_restore(
    orchestrator: ZillizCloudPlatformOrchestrator, default_spec: MilvusClusterSpec
) -> None:
    orchestrator.provision_cluster(default_spec)
    orchestrator.record_telemetry(
        default_spec.cluster_id,
        query_qps=500.0,
        p99_latency_ms=8.0,
        error_rate=0.0001,
        current_vectors_millions=25.5,
    )

    snapshot = orchestrator.create_backup_snapshot(default_spec.cluster_id, "snap-20261115-01")
    assert snapshot.vector_count_millions == 25.5
    assert snapshot.milvus_version == "v2.4.5"

    # Modify cluster state
    orchestrator._clusters[default_spec.cluster_id].spec.milvus_version = "v2.4.9"
    orchestrator._clusters[default_spec.cluster_id].current_vectors_millions = 50.0

    # Restore from snapshot
    restored = orchestrator.restore_from_backup(default_spec.cluster_id, "snap-20261115-01")
    assert restored is True
    assert orchestrator._clusters[default_spec.cluster_id].spec.milvus_version == "v2.4.5"
    assert orchestrator._clusters[default_spec.cluster_id].current_vectors_millions == 25.5


def test_slo_telemetry_reporting(
    orchestrator: ZillizCloudPlatformOrchestrator, default_spec: MilvusClusterSpec
) -> None:
    orchestrator.provision_cluster(default_spec)

    # 1. Healthy telemetry
    orchestrator.record_telemetry(
        default_spec.cluster_id,
        query_qps=350.0,
        p99_latency_ms=9.2,
        error_rate=0.00005,
        current_vectors_millions=10.0,
    )
    report_healthy = orchestrator.get_slo_report(default_spec.cluster_id)
    assert report_healthy["slo_compliance"]["overall_compliant"] is True
    assert report_healthy["health_score"] == 100.0

    # 2. Spiked latency and error rate
    orchestrator.record_telemetry(
        default_spec.cluster_id,
        query_qps=1200.0,
        p99_latency_ms=28.4,
        error_rate=0.005,
    )
    report_degraded = orchestrator.get_slo_report(default_spec.cluster_id)
    assert report_degraded["slo_compliance"]["overall_compliant"] is False
    assert report_degraded["slo_compliance"]["p99_latency_compliant"] is False
    assert report_degraded["slo_compliance"]["error_rate_compliant"] is False
    assert report_degraded["slo_compliance"]["breach_count"] >= 2
    assert report_degraded["health_score"] < 50.0
