"""Tests for Together AI Infra Agent Systems (GPU fleet knowledge graph, anomaly detection, and remediation)."""

import pytest

from llm_control_plane import (
    CANONICAL_XID_CATALOG,
    AutonomousRemediationEngine,
    GPUAnomalyDetector,
    GPUFailureSeverity,
    GPUFleetKnowledgeGraph,
    GPUHealthTelemetry,
    HardwareResetType,
    InfraInvestigatorAgent,
    NodeLifecycleState,
)


def test_canonical_xid_catalog() -> None:
    """Verify standard NVIDIA Xid error catalog contains critical AI cluster fault definitions."""
    assert 79 in CANONICAL_XID_CATALOG
    xid_79 = CANONICAL_XID_CATALOG[79]
    assert xid_79.name == "GPU Fallen Off the Bus"
    assert xid_79.severity == GPUFailureSeverity.FATAL
    assert xid_79.requires_power_cycle is True
    assert xid_79.recommended_action == HardwareResetType.BMC_POWER_CYCLE

    assert 31 in CANONICAL_XID_CATALOG
    xid_31 = CANONICAL_XID_CATALOG[31]
    assert xid_31.severity == GPUFailureSeverity.WARNING
    assert xid_31.requires_power_cycle is False


def test_fleet_knowledge_graph_topology() -> None:
    """Verify fleet knowledge graph tracks hosts, GPUs, racks, and distributed jobs."""
    kg = GPUFleetKnowledgeGraph()
    kg.register_host(
        host_id="node-us-west-01",
        datacenter="us-west-1a",
        rack_id="rack-42",
        gpu_count=8,
        gpu_model="NVIDIA-H100-SXM5-80GB",
    )
    kg.register_host(
        host_id="node-us-west-02",
        datacenter="us-west-1a",
        rack_id="rack-42",
        gpu_count=8,
        gpu_model="NVIDIA-H100-SXM5-80GB",
    )

    assert "node-us-west-01" in kg.hosts
    assert len(kg.gpus) == 16
    assert kg.get_host_state("node-us-west-01") == NodeLifecycleState.ACTIVE

    # Allocate multi-node training job
    kg.allocate_job("job-llama3-70b-train", ["node-us-west-01", "node-us-west-02"])

    blast = kg.compute_blast_radius("node-us-west-01")
    assert blast["datacenter"] == "us-west-1a"
    assert blast["rack_id"] == "rack-42"
    assert blast["rack_neighbor_count"] == 1
    assert blast["affected_job_count"] == 1
    assert "job-llama3-70b-train" in blast["affected_jobs"]


def test_anomaly_detector_healthy_telemetry() -> None:
    """Verify healthy GPU telemetry produces no anomalies."""
    detector = GPUAnomalyDetector()
    telemetry = GPUHealthTelemetry(
        host_id="node-01",
        gpu_index=0,
        temperature_celsius=62.0,
        power_watts=350.0,
        pcie_generation=5,
        pcie_link_width=16,
        correctable_ecc_errors=0,
        uncorrectable_ecc_errors=0,
        nvlink_packet_drop_rate=0.001,
        active_xid_codes=[],
    )
    result = detector.evaluate_telemetry(telemetry)
    assert not result.is_anomalous
    assert result.severity == GPUFailureSeverity.TRANSIENT
    assert len(result.triggered_rules) == 0


def test_anomaly_detector_xid_79_fatal() -> None:
    """Verify NVIDIA Xid 79 triggers fatal anomaly requiring BMC power cycle."""
    detector = GPUAnomalyDetector()
    telemetry = GPUHealthTelemetry(
        host_id="node-01",
        gpu_index=3,
        temperature_celsius=75.0,
        power_watts=400.0,
        pcie_generation=5,
        pcie_link_width=16,
        active_xid_codes=[79],
    )
    result = detector.evaluate_telemetry(telemetry)
    assert result.is_anomalous
    assert result.severity == GPUFailureSeverity.FATAL
    assert result.recommended_action == HardwareResetType.BMC_POWER_CYCLE
    assert any("Xid 79" in r for r in result.triggered_rules)
    assert result.confidence >= 0.95


def test_anomaly_detector_pcie_degradation_and_thermal() -> None:
    """Verify degraded PCIe bus width and high thermal throttling are identified."""
    detector = GPUAnomalyDetector(max_temp_celsius=85.0, expected_pcie_gen=5, expected_pcie_width=16)
    telemetry = GPUHealthTelemetry(
        host_id="node-02",
        gpu_index=1,
        temperature_celsius=89.5,
        power_watts=550.0,
        pcie_generation=4,  # Degraded from Gen5
        pcie_link_width=8,   # Degraded from x16
        active_xid_codes=[45],
    )
    result = detector.evaluate_telemetry(telemetry)
    assert result.is_anomalous
    assert result.severity in (GPUFailureSeverity.CRITICAL, GPUFailureSeverity.FATAL)
    assert any("PCIe link degraded" in r for r in result.triggered_rules)
    assert any("Thermal threshold exceeded" in r for r in result.triggered_rules)


def test_infra_investigator_agent_correlation() -> None:
    """Verify investigator agent synthesizes incident diagnosis across topology and anomalies."""
    kg = GPUFleetKnowledgeGraph()
    kg.register_host("host-cluster-04", datacenter="eu-central-1", rack_id="rack-11", gpu_count=8)
    kg.allocate_job("job-speculative-inference-fleet", ["host-cluster-04"])

    detector = GPUAnomalyDetector()
    telemetry_gpu2 = GPUHealthTelemetry(
        host_id="host-cluster-04",
        gpu_index=2,
        temperature_celsius=70.0,
        power_watts=300.0,
        pcie_generation=5,
        pcie_link_width=16,
        active_xid_codes=[79],
    )
    anomalies = [detector.evaluate_telemetry(telemetry_gpu2)]

    agent = InfraInvestigatorAgent(kg)
    diagnosis = agent.investigate("host-cluster-04", anomalies)

    assert diagnosis.host_id == "host-cluster-04"
    assert diagnosis.datacenter == "eu-central-1"
    assert diagnosis.rack_id == "rack-11"
    assert diagnosis.severity == GPUFailureSeverity.FATAL
    assert diagnosis.affected_gpus == [2]
    assert "job-speculative-inference-fleet" in diagnosis.affected_jobs
    assert diagnosis.recommended_remediation == HardwareResetType.BMC_POWER_CYCLE
    assert diagnosis.confidence_score >= 0.95
    assert "Xid 79" in diagnosis.root_cause_hypothesis


def test_autonomous_remediation_engine_successful_flow() -> None:
    """Verify cordon, drain, reset, and revalidation restores node to active compute pool."""
    kg = GPUFleetKnowledgeGraph()
    kg.register_host("host-remedy-01", datacenter="us-east-1", rack_id="rack-01", gpu_count=8)
    kg.allocate_job("job-batch-evals", ["host-remedy-01"])

    agent = InfraInvestigatorAgent(kg)
    detector = GPUAnomalyDetector()
    telemetry = GPUHealthTelemetry(
        host_id="host-remedy-01",
        gpu_index=0,
        temperature_celsius=65.0,
        power_watts=350.0,
        pcie_generation=5,
        pcie_link_width=16,
        active_xid_codes=[43],
    )
    diagnosis = agent.investigate("host-remedy-01", [detector.evaluate_telemetry(telemetry)])

    engine = AutonomousRemediationEngine(kg)
    result = engine.execute_remediation(diagnosis, simulate_hardware_success=True)

    assert result.host_id == "host-remedy-01"
    assert result.drained_job_count == 1
    assert result.revalidation.passed is True
    assert result.node_final_state == NodeLifecycleState.ACTIVE
    assert result.escalated_to_oncall is False
    assert kg.get_host_state("host-remedy-01") == NodeLifecycleState.ACTIVE
    assert "returned to ACTIVE pool" in result.resolution_summary


def test_autonomous_remediation_failure_escalation() -> None:
    """Verify persistent hardware failure triggers cordon, drain, and escalation to datacenter on-call."""
    kg = GPUFleetKnowledgeGraph()
    kg.register_host("host-broken-01", datacenter="us-east-1", rack_id="rack-99", gpu_count=8)
    kg.allocate_job("job-live-serving", ["host-broken-01"])

    agent = InfraInvestigatorAgent(kg)
    detector = GPUAnomalyDetector()
    telemetry = GPUHealthTelemetry(
        host_id="host-broken-01",
        gpu_index=7,
        temperature_celsius=92.0,
        power_watts=600.0,
        pcie_generation=1,
        pcie_link_width=1,
        uncorrectable_ecc_errors=4,
        active_xid_codes=[92],
    )
    diagnosis = agent.investigate("host-broken-01", [detector.evaluate_telemetry(telemetry)])

    engine = AutonomousRemediationEngine(kg)
    result = engine.execute_remediation(diagnosis, simulate_hardware_success=False)

    assert result.host_id == "host-broken-01"
    assert result.revalidation.passed is False
    assert result.node_final_state == NodeLifecycleState.MAINTENANCE
    assert result.escalated_to_oncall is True
    assert kg.get_host_state("host-broken-01") == NodeLifecycleState.MAINTENANCE
    assert "marked for datacenter on-call MAINTENANCE" in result.resolution_summary


def test_knowledge_graph_missing_host_error_handling() -> None:
    """Verify knowledge graph handles unregistered host errors gracefully."""
    kg = GPUFleetKnowledgeGraph()
    with pytest.raises(KeyError):
        kg.get_host_state("non-existent-host")
    with pytest.raises(KeyError):
        kg.compute_blast_radius("non-existent-host")
    with pytest.raises(ValueError):
        kg.allocate_job("job-err", ["unregistered-node"])
