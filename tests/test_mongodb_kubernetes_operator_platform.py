"""Comprehensive test suite for MongoDB Kubernetes Operator and Core Platforms architecture."""

import pytest

from llm_control_plane.mongodb_kubernetes_operator_platform import (
    AtlasSearchSidecarOrchestrator,
    ClusterPhase,
    MongoDBClusterSpec,
    MongoDBKubernetesController,
    MongoDBOperatorTelemetryAndSLOTracker,
    RollingUpgradeAndFCVCoordinator,
    SplitBrainException,
)


@pytest.fixture
def default_spec() -> MongoDBClusterSpec:
    return MongoDBClusterSpec(
        cluster_name="analytics-core",
        namespace="data-platform",
        members_count=3,
        replica_set_name="rs0",
        version="7.0.8",
        feature_compatibility_version="7.0",
        storage_size_gb=250,
        enable_vector_search=True,
    )


def test_cluster_spec_initialization(default_spec: MongoDBClusterSpec) -> None:
    assert default_spec.cluster_name == "analytics-core"
    assert default_spec.members_count == 3
    assert default_spec.version == "7.0.8"
    assert default_spec.feature_compatibility_version == "7.0"
    assert default_spec.enable_vector_search is True


def test_reconcile_healthy_topology(default_spec: MongoDBClusterSpec) -> None:
    controller = MongoDBKubernetesController(default_spec)
    pods = [
        {"pod_name": "analytics-core-0", "phase": "Running", "is_primary": True, "lag_sec": 0.0},
        {"pod_name": "analytics-core-1", "phase": "Running", "is_primary": False, "lag_sec": 0.2},
        {"pod_name": "analytics-core-2", "phase": "Running", "is_primary": False, "lag_sec": 0.1},
    ]

    status = controller.reconcile(pods)
    assert status.phase == ClusterPhase.RUNNING
    assert status.primary_pod == "analytics-core-0"
    assert len(status.ready_members) == 3
    assert len(status.unhealthy_members) == 0
    assert status.oplog_lag_seconds == 0.2
    assert status.vector_search_ready is True


def test_quorum_loss_triggers_degradation(default_spec: MongoDBClusterSpec) -> None:
    controller = MongoDBKubernetesController(default_spec)
    pods = [
        {"pod_name": "analytics-core-0", "phase": "Running", "is_primary": True, "lag_sec": 0.0},
        {"pod_name": "analytics-core-1", "phase": "Pending", "is_primary": False, "lag_sec": 0.0},
        {"pod_name": "analytics-core-2", "phase": "Failed", "is_primary": False, "lag_sec": 0.0},
    ]

    status = controller.reconcile(pods)
    assert status.phase == ClusterPhase.DEGRADED
    assert status.primary_pod is None
    assert "Quorum loss" in status.message


def test_split_brain_raises_exception(default_spec: MongoDBClusterSpec) -> None:
    controller = MongoDBKubernetesController(default_spec)
    pods = [
        {"pod_name": "analytics-core-0", "phase": "Running", "is_primary": True, "lag_sec": 0.0},
        {"pod_name": "analytics-core-1", "phase": "Running", "is_primary": True, "lag_sec": 0.0},
        {"pod_name": "analytics-core-2", "phase": "Running", "is_primary": False, "lag_sec": 0.1},
    ]

    with pytest.raises(SplitBrainException, match="Split-brain detected"):
        controller.reconcile(pods)


def test_automatic_primary_election(default_spec: MongoDBClusterSpec) -> None:
    controller = MongoDBKubernetesController(default_spec)
    pods = [
        {"pod_name": "analytics-core-0", "phase": "Running", "is_primary": False, "lag_sec": 1.2},
        {"pod_name": "analytics-core-1", "phase": "Running", "is_primary": False, "lag_sec": 0.05},
        {"pod_name": "analytics-core-2", "phase": "Running", "is_primary": False, "lag_sec": 0.8},
    ]

    status = controller.reconcile(pods)
    assert status.phase == ClusterPhase.RUNNING
    # Should elect pod-1 because it has minimum lag (0.05s)
    assert status.primary_pod == "analytics-core-1"


def test_fence_partitioned_pod(default_spec: MongoDBClusterSpec) -> None:
    controller = MongoDBKubernetesController(default_spec)
    pods = [
        {"pod_name": "analytics-core-0", "phase": "Running", "is_primary": True, "lag_sec": 0.0},
        {"pod_name": "analytics-core-1", "phase": "Running", "is_primary": False, "lag_sec": 0.1},
        {"pod_name": "analytics-core-2", "phase": "Running", "is_primary": False, "lag_sec": 45.0},
    ]

    updated = controller.fence_partitioned_pod("analytics-core-2", pods)
    fenced = next(p for p in updated if p["pod_name"] == "analytics-core-2")
    assert fenced["phase"] == "Failed"
    assert fenced["is_primary"] is False


def test_atlas_search_sidecar_spec(default_spec: MongoDBClusterSpec) -> None:
    orchestrator = AtlasSearchSidecarOrchestrator(default_spec)
    spec = orchestrator.provision_sidecar_spec()
    assert spec["name"] == "mongodb-vector-search"
    assert spec["resources"]["limits"]["memory"] == "16Gi"
    assert spec["resources"]["limits"]["cpu"] == "4"
    assert spec["readinessProbe"]["httpGet"]["port"] == 27027


def test_vector_search_index_registration(default_spec: MongoDBClusterSpec) -> None:
    orchestrator = AtlasSearchSidecarOrchestrator(default_spec)
    idx = orchestrator.register_vector_index(
        database="knowledge",
        collection="embeddings",
        index_name="vector_cosine_1536",
        dimensions=1536,
        similarity="cosine",
    )

    assert idx["status"] == "READY"
    assert idx["dimensions"] == 1536
    fetched = orchestrator.get_index_status("knowledge", "embeddings", "vector_cosine_1536")
    assert fetched is not None
    assert fetched["similarity"] == "cosine"


def test_rolling_upgrade_requires_healthy_cluster(default_spec: MongoDBClusterSpec) -> None:
    coordinator = RollingUpgradeAndFCVCoordinator(default_spec)
    degraded_pods = [
        {"pod_name": "analytics-core-0", "phase": "Running", "is_primary": True},
        {"pod_name": "analytics-core-1", "phase": "Failed", "is_primary": False},
        {"pod_name": "analytics-core-2", "phase": "Running", "is_primary": False},
    ]

    with pytest.raises(RuntimeError, match="Cannot initiate upgrade"):
        coordinator.execute_rolling_upgrade("8.0.0", "8.0", degraded_pods)


def test_successful_rolling_upgrade_and_fcv_transition(default_spec: MongoDBClusterSpec) -> None:
    coordinator = RollingUpgradeAndFCVCoordinator(default_spec)
    pods = [
        {"pod_name": "analytics-core-0", "phase": "Running", "is_primary": True},
        {"pod_name": "analytics-core-1", "phase": "Running", "is_primary": False},
        {"pod_name": "analytics-core-2", "phase": "Running", "is_primary": False},
    ]

    logs = coordinator.execute_rolling_upgrade("7.2.0", "7.2", pods)
    assert any("Cordoning secondary pod analytics-core-1" in entry for entry in logs)
    assert any("Issuing replSetStepDown to primary pod analytics-core-0" in entry for entry in logs)
    assert any("Advancing Feature Compatibility Version" in entry for entry in logs)
    assert default_spec.version == "7.2.0"
    assert default_spec.feature_compatibility_version == "7.2"


def test_operator_telemetry_snapshot(default_spec: MongoDBClusterSpec) -> None:
    controller = MongoDBKubernetesController(default_spec)
    tracker = MongoDBOperatorTelemetryAndSLOTracker("analytics-core")

    tracker.record_reconciliation(12.5)
    tracker.record_reconciliation(18.0)

    pods = [
        {"pod_name": "analytics-core-0", "phase": "Running", "is_primary": True, "lag_sec": 0.05},
        {"pod_name": "analytics-core-1", "phase": "Running", "is_primary": False, "lag_sec": 0.08},
        {"pod_name": "analytics-core-2", "phase": "Running", "is_primary": False, "lag_sec": 0.04},
    ]
    status = controller.reconcile(pods)
    snapshot = tracker.generate_snapshot(status, search_sync_ratio=1.0)

    assert snapshot.is_healthy is True
    assert snapshot.reconcile_duration_ms == 18.0
    assert snapshot.primary_elected is True
    assert snapshot.metrics["mongodb_controller_health_status"] == 1.0
    assert snapshot.metrics["mongodb_operator_ready_members"] == 3.0
