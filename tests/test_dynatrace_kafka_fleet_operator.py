import pytest
from llm_control_plane.dynatrace_kafka_fleet_operator import (
    AIOpsToilAutomator,
    AclOperation,
    AclPermission,
    ArgoCDGitOpsValidator,
    BrokerResourceSpec,
    BrokerRuntimeStatus,
    CloudProvider,
    ClusterState,
    CruiseControlPartitionRebalancer,
    KafkaClusterSpec,
    KafkaClusterStatus,
    KafkaFleetReconciler,
    KafkaObservabilityAndLagMonitor,
    KafkaTopicSpec,
    KafkaUserAclSpec,
)


@pytest.fixture
def base_cluster_spec():
    return KafkaClusterSpec(
        cluster_id="dt-kafka-prod-01",
        name="telemetry-kafka-fleet",
        namespace="streaming-platform",
        cloud_provider=CloudProvider.AWS,
        region="eu-central-1",
        kafka_version="3.8.0",
        broker_replicas=3,
        resources=BrokerResourceSpec(cpu_cores=8.0, memory_gb=32.0, storage_gb=2000),
        topics={
            "telemetry.traces": KafkaTopicSpec(
                name="telemetry.traces", partitions=24, replication_factor=3, min_insync_replicas=2
            ),
            "telemetry.metrics": KafkaTopicSpec(
                name="telemetry.metrics", partitions=48, replication_factor=3, min_insync_replicas=2
            ),
        },
        acls=[
            KafkaUserAclSpec(
                principal="User:davis-ai-worker",
                resource_type="topic",
                resource_name="telemetry.traces",
                operation=AclOperation.READ,
            )
        ],
    )


@pytest.fixture
def base_cluster_status(base_cluster_spec):
    return KafkaClusterStatus(
        cluster_id=base_cluster_spec.cluster_id,
        current_state=ClusterState.READY,
        active_version=base_cluster_spec.kafka_version,
        active_replicas=base_cluster_spec.broker_replicas,
        brokers=[
            BrokerRuntimeStatus(
                broker_id=0,
                pod_name="telemetry-kafka-fleet-kafka-0",
                rack="zone-1",
                kafka_version="3.8.0",
                in_sync=True,
                disk_used_gb=400.0,
                disk_capacity_gb=2000.0,
            ),
            BrokerRuntimeStatus(
                broker_id=1,
                pod_name="telemetry-kafka-fleet-kafka-1",
                rack="zone-2",
                kafka_version="3.8.0",
                in_sync=True,
                disk_used_gb=420.0,
                disk_capacity_gb=2000.0,
            ),
            BrokerRuntimeStatus(
                broker_id=2,
                pod_name="telemetry-kafka-fleet-kafka-2",
                rack="zone-3",
                kafka_version="3.8.0",
                in_sync=True,
                disk_used_gb=390.0,
                disk_capacity_gb=2000.0,
            ),
        ],
        applied_spec_hash=base_cluster_spec.spec_hash(),
        under_replicated_partitions=0,
        active_controller_count=1,
        max_consumer_lag=50,
    )


def test_topic_validation():
    # Valid topic
    t_valid = KafkaTopicSpec("events", partitions=12, replication_factor=3, min_insync_replicas=2)
    assert t_valid.validate(3) == []

    # Invalid: partitions <= 0
    t_invalid_part = KafkaTopicSpec("events", partitions=0, replication_factor=3)
    assert any("partitions must be > 0" in e for e in t_invalid_part.validate(3))

    # Invalid: replication factor exceeds broker count
    t_invalid_rf = KafkaTopicSpec("events", partitions=12, replication_factor=5)
    assert any("replication_factor 5 must be in 1..3" in e for e in t_invalid_rf.validate(3))

    # Invalid: min.insync.replicas > replication factor
    t_invalid_isr = KafkaTopicSpec("events", partitions=12, replication_factor=2, min_insync_replicas=3)
    assert any("min_insync_replicas 3 cannot exceed" in e for e in t_invalid_isr.validate(3))


def test_strimzi_cr_generation(base_cluster_spec):
    cr = base_cluster_spec.to_strimzi_cr()
    assert cr["apiVersion"] == "kafka.strimzi.io/v1beta2"
    assert cr["kind"] == "Kafka"
    assert cr["metadata"]["name"] == "telemetry-kafka-fleet"
    assert cr["metadata"]["namespace"] == "streaming-platform"
    assert cr["spec"]["kafka"]["version"] == "3.8.0"
    assert cr["spec"]["kafka"]["replicas"] == 3
    assert cr["spec"]["kafka"]["storage"]["size"] == "2000Gi"
    assert cr["spec"]["kafka"]["rack"]["topologyKey"] == "topology.kubernetes.io/zone"
    assert cr["spec"]["cruiseControl"]["autoRebalance"] is True


def test_pre_release_gate_pass(base_cluster_spec, base_cluster_status):
    reconciler = KafkaFleetReconciler()
    gate = reconciler.evaluate_pre_release_gate(base_cluster_spec, base_cluster_status)
    assert gate.passed is True
    assert len(gate.reasons) == 0


def test_pre_release_gate_blocks_under_replicated(base_cluster_spec, base_cluster_status):
    base_cluster_status.under_replicated_partitions = 4
    reconciler = KafkaFleetReconciler()
    gate = reconciler.evaluate_pre_release_gate(base_cluster_spec, base_cluster_status)
    assert gate.passed is False
    assert any("under-replicated" in r for r in gate.reasons)


def test_pre_release_gate_blocks_missing_controller(base_cluster_spec, base_cluster_status):
    base_cluster_status.active_controller_count = 0
    reconciler = KafkaFleetReconciler()
    gate = reconciler.evaluate_pre_release_gate(base_cluster_spec, base_cluster_status)
    assert gate.passed is False
    assert any("active controller count" in r for r in gate.reasons)


def test_pre_release_gate_blocks_excessive_lag(base_cluster_spec, base_cluster_status):
    base_cluster_status.max_consumer_lag = 12000
    reconciler = KafkaFleetReconciler(max_allowed_lag_for_upgrade=5000)
    gate = reconciler.evaluate_pre_release_gate(base_cluster_spec, base_cluster_status)
    assert gate.passed is False
    assert any("exceeds SLA threshold" in r for r in gate.reasons)


def test_configuration_drift_detection(base_cluster_spec, base_cluster_status):
    reconciler = KafkaFleetReconciler()
    # No drift initially
    assert reconciler.detect_configuration_drift(base_cluster_spec, base_cluster_status) == []

    # Drift version
    base_cluster_spec.kafka_version = "3.8.1"
    drifts = reconciler.detect_configuration_drift(base_cluster_spec, base_cluster_status)
    assert any("Version drift" in d for d in drifts)

    # Drift replicas
    base_cluster_spec.broker_replicas = 5
    drifts = reconciler.detect_configuration_drift(base_cluster_spec, base_cluster_status)
    assert any("Replica drift" in d for d in drifts)


def test_reconcile_rolling_upgrade(base_cluster_spec, base_cluster_status):
    reconciler = KafkaFleetReconciler()
    # Desired version upgrade
    base_cluster_spec.kafka_version = "3.8.1"
    updated_status, logs = reconciler.reconcile_cluster(base_cluster_spec, base_cluster_status)

    assert updated_status.current_state == ClusterState.READY
    assert updated_status.active_version == "3.8.1"
    assert all(b.kafka_version == "3.8.1" for b in updated_status.brokers)
    assert any("Rolling upgrade completed successfully" in l for l in logs)


def test_reconcile_scale_out(base_cluster_spec, base_cluster_status):
    reconciler = KafkaFleetReconciler()
    base_cluster_spec.broker_replicas = 5
    updated_status, logs = reconciler.reconcile_cluster(base_cluster_spec, base_cluster_status)

    assert updated_status.current_state == ClusterState.READY
    assert updated_status.active_replicas == 5
    assert len(updated_status.brokers) == 5
    assert any("Scaling out cluster by +2 brokers" in l for l in logs)


def test_cruise_control_partition_rebalancing(base_cluster_spec, base_cluster_status):
    rebalancer = CruiseControlPartitionRebalancer(disk_imbalance_threshold=0.15)

    # Create artificial disk skew on broker 0
    base_cluster_status.brokers[0].disk_used_gb = 1800.0  # 90% full
    base_cluster_status.brokers[1].disk_used_gb = 400.0   # 20% full
    base_cluster_status.brokers[2].disk_used_gb = 400.0   # 20% full

    imbalance = rebalancer.compute_imbalance_ratio(base_cluster_status.brokers)
    assert imbalance > 0.15

    proposals = rebalancer.generate_rebalance_proposals(
        base_cluster_status.brokers, base_cluster_spec.topics
    )
    assert len(proposals) > 0
    first = proposals[0]
    assert first.source_broker == 0
    assert first.target_broker in (1, 2)
    assert first.throttle_bytes_per_sec == 52428800


def test_prometheus_observability_exporter(base_cluster_status):
    monitor = KafkaObservabilityAndLagMonitor("telemetry-kafka-fleet", "streaming-platform")
    lags = {"davis-ai-workers": {"telemetry.traces": 150, "telemetry.metrics": 80}}
    metrics_text = monitor.export_prometheus_metrics(base_cluster_status, lags)

    assert "kafka_server_brokermonitoring_underreplicatedpartitions" in metrics_text
    assert 'cluster="telemetry-kafka-fleet"' in metrics_text
    assert 'group="davis-ai-workers",topic="telemetry.traces"} 150' in metrics_text
    assert "kafka_broker_disk_used_ratio" in metrics_text


def test_consumer_lag_calculation():
    monitor = KafkaObservabilityAndLagMonitor("test-cluster", "test-ns")
    topic_offsets = {0: 1000, 1: 2500, 2: 8000}
    consumer_offsets = {0: 950, 1: 2500, 2: 7000}
    lag = monitor.calculate_consumer_lag(topic_offsets, consumer_offsets)

    assert lag[0] == 50
    assert lag[1] == 0
    assert lag[2] == 1000


def test_argocd_helm_synthesis(base_cluster_spec):
    helm_values = ArgoCDGitOpsValidator.generate_helm_values(base_cluster_spec)
    assert helm_values["cluster"]["name"] == "telemetry-kafka-fleet"
    assert helm_values["cluster"]["provider"] == "aws"
    assert helm_values["cluster"]["replicas"] == 3
    assert len(helm_values["topics"]) == 2

    app = ArgoCDGitOpsValidator.generate_argocd_application(
        base_cluster_spec, "https://github.com/dynatrace/gitops-fleet", "charts/kafka"
    )
    assert app["apiVersion"] == "argoproj.io/v1alpha1"
    assert app["spec"]["destination"]["namespace"] == "streaming-platform"
    assert app["spec"]["syncPolicy"]["automated"]["selfHeal"] is True


def test_aiops_incident_diagnostics(base_cluster_status):
    # Healthy cluster -> no incident
    diag = AIOpsToilAutomator.diagnose_cluster(base_cluster_status)
    assert diag is None

    # Disk exhaustion scenario
    base_cluster_status.brokers[1].disk_used_gb = 1950.0  # > 90%
    diag_disk = AIOpsToilAutomator.diagnose_cluster(base_cluster_status)
    assert diag_disk is not None
    assert diag_disk.incident_type == "BROKER_DISK_EXHAUSTION"
    assert diag_disk.severity == "P1"
    assert diag_disk.automated_remediation_applicable is True

    # Controller loss
    base_cluster_status.active_controller_count = 0
    diag_ctrl = AIOpsToilAutomator.diagnose_cluster(base_cluster_status)
    assert diag_ctrl is not None
    assert diag_ctrl.incident_type == "CONTROLLER_ELECTION_FAILURE"
    assert diag_ctrl.severity == "P1"


def test_multi_cloud_provider_support():
    for provider, storage_class in [
        (CloudProvider.AWS, "gp3"),
        (CloudProvider.AZURE, "managed-premium"),
        (CloudProvider.GCP, "pd-ssd"),
    ]:
        spec = KafkaClusterSpec(
            cluster_id=f"dt-{provider.value}-01",
            name=f"kafka-{provider.value}",
            namespace="streaming",
            cloud_provider=provider,
            region="europe-west1",
            resources=BrokerResourceSpec(storage_type=storage_class),
        )
        cr = spec.to_strimzi_cr()
        assert cr["spec"]["kafka"]["storage"]["class"] == f"{provider.value}-fast"
