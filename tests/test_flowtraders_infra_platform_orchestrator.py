"""Tests for Flow Traders Infrastructure Platform Orchestrator."""

from llm_control_plane.flowtraders_infra_platform_orchestrator import (
    ChangeRiskLevel,
    CiCdGuardrailEngine,
    ClusterType,
    HybridK8sClusterManager,
    K8sNode,
    KafkaStreamingWatchdog,
    LinuxFleetHostAuditor,
    NodeCondition,
    NodeConditionStatus,
    PlannedChange,
    PodDisruptionBudget,
    TerraformModuleReconciler,
    TerraformResource,
    TopicPartition,
)


def test_terraform_policy_evaluation_success_and_violations():
    reconciler = TerraformModuleReconciler(module_name="flowtraders-vpc-core", version="1.4.0")

    # 1. Compliant resource
    compliant_res = TerraformResource(
        resource_type="google_compute_disk",
        resource_name="trading_engine_nvme",
        attributes={"encryption": {"enabled": True, "kms_key_id": "projects/flowtraders/kms/keys/key1"}},
        tags={"team": "infra-platform", "environment": "production", "cost-center": "amsterdam-trading"},
    )
    res_eval = reconciler.evaluate_policies(compliant_res)
    assert res_eval.allowed is True
    assert len(res_eval.violations) == 0

    # 2. Resource missing mandatory tags
    missing_tags_res = TerraformResource(
        resource_type="google_compute_disk",
        resource_name="ad_hoc_disk",
        attributes={"encryption": {"enabled": True, "kms_key_id": "kms1"}},
        tags={"team": "quant"},
    )
    eval_missing = reconciler.evaluate_policies(missing_tags_res)
    assert eval_missing.allowed is False
    assert any("missing mandatory tags" in v for v in eval_missing.violations)

    # 3. Security group opening sensitive port (22 SSH) to 0.0.0.0/0
    bad_sg = TerraformResource(
        resource_type="aws_security_group",
        resource_name="trading_ingress",
        attributes={
            "ingress": [
                {"cidr_blocks": ["0.0.0.0/0"], "from_port": 22, "to_port": 22},
                {"cidr_blocks": ["10.0.0.0/8"], "from_port": 443, "to_port": 443},
            ]
        },
        tags={"team": "infra-platform", "environment": "production", "cost-center": "amsterdam-trading"},
    )
    eval_bad_sg = reconciler.evaluate_policies(bad_sg)
    assert eval_bad_sg.allowed is False
    assert any("prohibited port 22" in v for v in eval_bad_sg.violations)

    # 4. Storage without KMS encryption
    unencrypted_bucket = TerraformResource(
        resource_type="google_storage_bucket",
        resource_name="market_data_lake",
        attributes={"encryption": {"enabled": False}},
        tags={"team": "infra-platform", "environment": "production", "cost-center": "amsterdam-trading"},
    )
    eval_bucket = reconciler.evaluate_policies(unencrypted_bucket)
    assert eval_bucket.allowed is False
    assert any("KMS encryption" in v for v in eval_bucket.violations)


def test_terraform_drift_detection():
    reconciler = TerraformModuleReconciler(module_name="trading-cluster-gke", version="2.1.0")
    res = TerraformResource(
        resource_type="google_container_cluster",
        resource_name="ams_prod_cluster",
        attributes={"node_count": 8, "machine_type": "c2-standard-16", "release_channel": "REGULAR"},
        tags={"team": "infra-platform", "environment": "production", "cost-center": "ams-trading"},
    )
    reconciler.register_resource(res)

    observed_attrs = {
        "node_count": 12,  # modified
        "machine_type": "c2-standard-16",
        # release_channel removed
        "auto_repair": True,  # added out-of-band
    }

    drift = reconciler.detect_drift(res.address, observed_attrs)
    assert drift.has_drift is True
    assert drift.added_attributes == {"auto_repair": True}
    assert drift.removed_attributes == ["release_channel"]
    assert drift.modified_attributes["node_count"] == (8, 12)


def test_cicd_guardrail_blast_radius_and_emergency_gate():
    guardrail = CiCdGuardrailEngine(max_allowed_risk_score=75.0)

    # Low risk changes (standard app updates)
    low_risk_changes = [
        PlannedChange(resource_address="k8s_deployment.trader", resource_type="k8s_deployment", action="UPDATE"),
        PlannedChange(resource_address="k8s_service.trader", resource_type="k8s_service", action="CREATE"),
    ]
    assessment_low = guardrail.assess_blast_radius(low_risk_changes)
    assert assessment_low.risk_level == ChangeRiskLevel.LOW
    assert assessment_low.risk_score < 20.0
    approved, msg = guardrail.evaluate_gate(assessment_low)
    assert approved is True
    assert "Deployment approved" in msg

    # Critical risk changes (slated deletion of core network VPC and Kafka cluster)
    critical_changes = [
        PlannedChange(
            resource_address="google_compute_network.flowtraders_ams_vpc",
            resource_type="google_compute_network",
            action="DELETE",
            is_sensitive=True,
        ),
        PlannedChange(
            resource_address="kafka_cluster.market_data_backbone",
            resource_type="kafka_cluster",
            action="DELETE",
            is_sensitive=True,
        ),
    ]
    assessment_crit = guardrail.assess_blast_radius(critical_changes)
    assert assessment_crit.risk_level == ChangeRiskLevel.CRITICAL
    assert assessment_crit.sensitive_deletes == 2
    assert assessment_crit.risk_score >= 70.0

    # Gate blocked without bypass
    approved_crit, blocked_msg = guardrail.evaluate_gate(assessment_crit)
    assert approved_crit is False
    assert "Emergency gate blocked deployment" in blocked_msg

    # Gate passed with break-glass token
    approved_with_token, token_msg = guardrail.evaluate_gate(
        assessment_crit, bypass_token="BREAKGLASS-AMS-ONCALL-2026"
    )
    assert approved_with_token is True


def test_hybrid_k8s_cluster_node_health_and_pdb_drain():
    mgr = HybridK8sClusterManager()

    # Register healthy GKE node
    gke_node = K8sNode(
        name="gke-pool-ams-1",
        cluster_type=ClusterType.CLOUD_GKE,
        conditions={
            "Ready": NodeCondition("Ready", NodeConditionStatus.TRUE),
            "DiskPressure": NodeCondition("DiskPressure", NodeConditionStatus.FALSE),
            "MemoryPressure": NodeCondition("MemoryPressure", NodeConditionStatus.FALSE),
            "NetworkUnavailable": NodeCondition("NetworkUnavailable", NodeConditionStatus.FALSE),
        },
        pods=["order-executor-1", "order-executor-2"],
    )
    mgr.register_node(gke_node)
    assert mgr.is_node_healthy("gke-pool-ams-1") is True

    # Register degraded bare-metal k3s node
    k3s_node = K8sNode(
        name="k3s-colocation-node-04",
        cluster_type=ClusterType.ON_PREM_K3S,
        conditions={
            "Ready": NodeCondition("Ready", NodeConditionStatus.TRUE),
            "DiskPressure": NodeCondition("DiskPressure", NodeConditionStatus.TRUE, reason="DiskFullAbove95Percent"),
        },
    )
    mgr.register_node(k3s_node)
    assert mgr.is_node_healthy("k3s-colocation-node-04") is False

    # Setup PodDisruptionBudget for order-executor
    pdb = PodDisruptionBudget(app_label="order-executor", min_available=1, current_available=2)
    mgr.register_pdb(pdb)

    app_labels = {"order-executor-1": "order-executor", "order-executor-2": "order-executor"}

    # Drain node: evicting 1 pod leaves 1 available (>= min 1), next pod evict blocked by PDB
    success, details = mgr.drain_node("gke-pool-ams-1", app_labels)
    assert success is False
    assert any("blocked by PDB" in d for d in details)
    assert gke_node.cordoned is True
    assert len(gke_node.pods) == 1  # 1 pod evicted, 1 remaining


def test_kafka_streaming_watchdog_lag_and_rebalance():
    watchdog = KafkaStreamingWatchdog(lag_threshold=5000)

    # Record normal partition
    watchdog.record_partition(
        TopicPartition(
            topic="market-data-etf-l1",
            partition=0,
            leader_broker_id=101,
            replicas=[101, 102, 103],
            in_sync_replicas=[101, 102, 103],
            current_offset=150000,
            log_end_offset=150200,  # lag = 200
        )
    )

    # Record under-replicated partition with high lag
    watchdog.record_partition(
        TopicPartition(
            topic="market-data-equities-l2",
            partition=1,
            leader_broker_id=101,
            replicas=[101, 102, 103],
            in_sync_replicas=[101],  # 102 and 103 offline
            current_offset=200000,
            log_end_offset=215000,  # lag = 15000 >= threshold 5000
        )
    )

    # Record third partition also led by broker 101 to create leadership skew
    watchdog.record_partition(
        TopicPartition(
            topic="trades-settlement-ams",
            partition=0,
            leader_broker_id=101,
            replicas=[101, 102, 103],
            in_sync_replicas=[101, 102, 103],
            current_offset=50000,
            log_end_offset=50050,
        )
    )

    urp = watchdog.detect_under_replicated_partitions()
    assert len(urp) == 1
    assert urp[0].topic == "market-data-equities-l2"

    high_lag = watchdog.detect_high_lag_partitions()
    assert len(high_lag) == 1
    assert high_lag[0].lag == 15000

    broker_dist = watchdog.compute_broker_distribution([101, 102, 103])
    assert broker_dist[101].lead_partitions == 3
    assert broker_dist[102].healthy is False  # missed replica

    recs = watchdog.generate_rebalance_recommendations([101, 102, 103])
    assert len(recs) > 0
    assert recs[0]["action"] == "SHED_LEADERSHIP"
    assert recs[0]["broker_id"] == 101


def test_linux_fleet_host_auditor_services_and_sysctl():
    auditor = LinuxFleetHostAuditor(hostname="ams-trade-bm-01.flowtraders.internal")

    # Service auditing
    services = {
        "kubelet.service": {"active": True, "enabled": True},
        "containerd.service": {"SubState": "running", "UnitFileState": "enabled"},
        "node-exporter.service": {"active": False, "enabled": True},
    }
    svc_results = auditor.audit_services(services)
    assert len(svc_results) == 3
    active_map = {r.service_name: r.active for r in svc_results}
    assert active_map["kubelet.service"] is True
    assert active_map["node-exporter.service"] is False

    # Kernel sysctl tuning audit
    sysctl_compliant = {
        "net.core.rmem_max": 33554432,  # >= 16777216
        "net.core.wmem_max": 33554432,
        "net.core.somaxconn": 65535,
        "fs.file-max": 4194304,
    }
    compliant, evals = auditor.audit_kernel_sysctl(sysctl_compliant)
    assert compliant is True
    assert all(e.is_compliant for e in evals)

    sysctl_non_compliant = {
        "net.core.rmem_max": 212992,  # default Linux low buffer
        "net.core.wmem_max": 212992,
        "net.core.somaxconn": 128,  # too low for trading sockets
        "fs.file-max": 100000,
    }
    compliant_bad, evals_bad = auditor.audit_kernel_sysctl(sysctl_non_compliant)
    assert compliant_bad is False
    assert any(not e.is_compliant for e in evals_bad)
