"""Unit tests for Flow Traders Infrastructure Platform Orchestrator."""

import pytest

from llm_control_plane.flowtraders_infra_platform_orchestrator import (
    CiCdGuardrailEngine,
    ClusterType,
    HostHealthState,
    HybridK8sClusterManager,
    KafkaPartitionMetric,
    KafkaStreamingWatchdog,
    LinuxFleetHostAuditor,
    LinuxHostSpec,
    NodeCondition,
    TerraformModuleReconciler,
    TerraformModuleSpec,
)


def test_terraform_module_validation_pass() -> None:
    reconciler = TerraformModuleReconciler()
    spec = TerraformModuleSpec(
        name="trading-k8s-cluster",
        version="1.4.2",
        source_path="modules/trading-k8s-cluster",
        required_providers={"google": "~> 5.0", "kubernetes": "~> 2.26"},
        variables=["project_id", "region", "node_count"],
        outputs=["cluster_id", "endpoint", "ca_certificate"],
        mandatory_tags=["Environment", "Owner", "CostCenter", "TradingVenue"],
    )
    result = reconciler.validate_module_spec(spec)
    assert result.passed is True
    assert len(result.violations) == 0
    assert len(result.warnings) == 0


def test_terraform_module_validation_violations() -> None:
    reconciler = TerraformModuleReconciler()
    spec = TerraformModuleSpec(
        name="",
        version="",
        source_path="modules/untrusted",
        required_providers={"unapproved_cloud": "1.0"},
        variables=[],
        outputs=[],
        mandatory_tags=["SomeTag"],
    )
    result = reconciler.validate_module_spec(spec)
    assert result.passed is False
    assert any("strictly required" in v for v in result.violations)
    assert any("not in approved provider set" in v for v in result.violations)
    assert any("missing from module spec" in v for v in result.violations)
    assert len(result.warnings) == 1


def test_terraform_drift_detection() -> None:
    reconciler = TerraformModuleReconciler()
    declared = {
        "google_compute_instance.trading_gw": {"machine_type": "c2-standard-16", "zone": "europe-west4-a"},
        "google_storage_bucket.market_archive": {"location": "EU", "versioning_enabled": True},
    }
    actual = {
        "google_compute_instance.trading_gw": {"machine_type": "c2-standard-8", "zone": "europe-west4-a"},
    }
    drift = reconciler.detect_resource_drift(declared, actual)
    assert drift["has_drift"] is True
    assert drift["drift_count"] == 2
    types = [d["drift_type"] for d in drift["drifted_resources"]]
    assert "ATTRIBUTE_MISMATCH" in types
    assert "MISSING_IN_ACTUAL" in types


def test_cicd_guardrail_pass() -> None:
    engine = CiCdGuardrailEngine()
    pr_context = {
        "base_branch": "main",
        "has_peer_approval": True,
        "commits_signed": True,
        "changed_resources_count": 4,
        "detected_unpinned_actions": False,
        "secrets_leak_detected": False,
    }
    result = engine.evaluate_pull_request_gate(pr_context)
    assert result.passed is True
    assert len(result.violations) == 0


def test_cicd_guardrail_violations() -> None:
    engine = CiCdGuardrailEngine(max_blast_radius=10)
    pr_context = {
        "base_branch": "main",
        "has_peer_approval": False,
        "commits_signed": False,
        "changed_resources_count": 45,
        "detected_unpinned_actions": True,
        "secrets_leak_detected": True,
    }
    result = engine.evaluate_pull_request_gate(pr_context)
    assert result.passed is False
    assert len(result.violations) == 5


def test_cicd_canary_rollback() -> None:
    engine = CiCdGuardrailEngine()
    healthy_canary = engine.evaluate_canary_rollback(error_rate_pct=0.05, p99_latency_ms=120.0)
    assert healthy_canary["rollback_triggered"] is False
    assert healthy_canary["action"] == "PROCEED_ROLLOUT"

    degraded_canary = engine.evaluate_canary_rollback(error_rate_pct=1.8, p99_latency_ms=450.0)
    assert degraded_canary["rollback_triggered"] is True
    assert degraded_canary["action"] == "EXECUTE_AUTOMATED_ROLLBACK"
    assert len(degraded_canary["reasons"]) == 2


def test_hybrid_k8s_cluster_health() -> None:
    mgr = HybridK8sClusterManager()
    mgr.register_node(NodeCondition("gke-node-1", ClusterType.CLOUD_GKE, True, True, False, False))
    mgr.register_node(NodeCondition("k3s-edge-1", ClusterType.ON_PREM_K3S, True, True, False, False))

    health = mgr.audit_cluster_health()
    assert health["is_cluster_healthy"] is True
    assert health["total_nodes"] == 2
    assert health["unhealthy_count"] == 0

    # Degrade one node
    mgr.register_node(NodeCondition("k3s-edge-2", ClusterType.ON_PREM_K3S, True, False, True, False))
    health_degraded = mgr.audit_cluster_health()
    assert health_degraded["is_cluster_healthy"] is False
    assert "k3s-edge-2" in health_degraded["unhealthy_nodes"]


def test_hybrid_k8s_node_drain_protection() -> None:
    mgr = HybridK8sClusterManager()
    mgr.register_node(NodeCondition("gke-node-prod-1", ClusterType.CLOUD_GKE, True, True, False, False))

    # Reject drain when PDB is violated
    res_blocked = mgr.prepare_node_drain("gke-node-prod-1", has_pdb_violations=True)
    assert res_blocked["can_drain"] is False
    assert res_blocked["action"] == "REJECT_DRAIN"

    # Allow drain when clean
    res_allowed = mgr.prepare_node_drain("gke-node-prod-1", has_pdb_violations=False)
    assert res_allowed["can_drain"] is True
    assert res_allowed["action"] == "CORDON_AND_DRAIN"

    with pytest.raises(KeyError):
        mgr.prepare_node_drain("non-existent-node")


def test_kafka_watchdog_healthy() -> None:
    watchdog = KafkaStreamingWatchdog(max_allowed_lag=500)
    partitions = [
        KafkaPartitionMetric("market_ticks", 0, 101, [101, 102], [101, 102], 50),
        KafkaPartitionMetric("market_ticks", 1, 102, [102, 103], [102, 103], 45),
        KafkaPartitionMetric("orders_reconciled", 0, 103, [103, 101], [103, 101], 10),
    ]
    report = watchdog.audit_topic_partitions(partitions)
    assert report["health_status"] == "HEALTHY"
    assert report["under_replicated_count"] == 0
    assert report["high_lag_count"] == 0
    assert report["partition_skew_ratio"] == 1.0


def test_kafka_watchdog_under_replicated_and_lag() -> None:
    watchdog = KafkaStreamingWatchdog(max_allowed_lag=100)
    partitions = [
        KafkaPartitionMetric("market_ticks", 0, 101, [101, 102, 103], [101], 25),  # Under-replicated
        KafkaPartitionMetric("market_ticks", 1, 101, [101, 102, 103], [101, 102, 103], 850),  # High lag
    ]
    report = watchdog.audit_topic_partitions(partitions)
    assert report["health_status"] == "CRITICAL"
    assert report["under_replicated_count"] == 1
    assert report["high_lag_count"] == 1


def test_linux_host_auditor_healthy() -> None:
    auditor = LinuxFleetHostAuditor()
    spec = LinuxHostSpec(
        hostname="ams-trade-srv-01",
        kernel_version="6.6.15-amd64",
        active_systemd_services=["sshd", "chronyd", "node_exporter", "trading_daemon"],
        sysctl_params={"net.core.rmem_max": 33554432, "net.core.wmem_max": 33554432, "fs.file-max": 4194304},
        disk_usage_pct=42.5,
        load_average=4.2,
    )
    res = auditor.audit_host(spec)
    assert res["health_state"] == HostHealthState.HEALTHY.value
    assert res["is_ready_for_trading"] is True
    assert len(res["missing_critical_services"]) == 0
    assert len(res["sysctl_mismatches"]) == 0


def test_linux_host_auditor_degraded_and_critical() -> None:
    auditor = LinuxFleetHostAuditor()
    # Degraded due to missing service & un-tuned sysctl
    degraded_spec = LinuxHostSpec(
        hostname="ams-srv-bad-sysctl",
        kernel_version="6.6.15-amd64",
        active_systemd_services=["sshd"],  # missing chronyd, node_exporter
        sysctl_params={"net.core.rmem_max": 4096},  # below 16MB minimum
        disk_usage_pct=50.0,
        load_average=2.0,
    )
    res_degraded = auditor.audit_host(degraded_spec)
    assert res_degraded["health_state"] == HostHealthState.DEGRADED.value
    assert res_degraded["is_ready_for_trading"] is False
    assert "chronyd" in res_degraded["missing_critical_services"]
    assert "net.core.rmem_max" in res_degraded["sysctl_mismatches"]

    # Critical due to disk usage
    critical_spec = LinuxHostSpec(
        hostname="ams-srv-critical-disk",
        kernel_version="6.6.15-amd64",
        active_systemd_services=["sshd", "chronyd", "node_exporter"],
        sysctl_params={"net.core.rmem_max": 16777216, "net.core.wmem_max": 16777216, "fs.file-max": 2097152},
        disk_usage_pct=98.5,
        load_average=80.0,
    )
    res_critical = auditor.audit_host(critical_spec)
    assert res_critical["health_state"] == HostHealthState.CRITICAL.value
