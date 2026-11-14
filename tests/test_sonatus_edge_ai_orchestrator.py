"""Unit tests for Sonatus Edge AI Orchestrator."""

from __future__ import annotations

from llm_control_plane.sonatus_edge_ai_orchestrator import (
    CANBusSignalAnomalyDetector,
    CANSignalReading,
    EdgeModelSpec,
    EdgeQuantizationTier,
    EdgeResourceBudgetManager,
    HybridEdgeCloudTelemetryTriager,
    IncidentSeverity,
    KernelLogClusterEngine,
    KernelLogEntry,
    ThermalState,
)


def test_edge_resource_budget_manager() -> None:
    """Verifies RAM and latency budget optimization across FP32, INT8, and INT4 tiers."""
    mgr = EdgeResourceBudgetManager(max_ram_bytes=32 * 1024 * 1024)  # 32 MB limit

    # Register 2 models with combined 48 MB FP32 footprint (> 32 MB budget)
    m1 = EdgeModelSpec(model_id="can_sequence_lstm", base_ram_bytes=24 * 1024 * 1024, base_latency_ms=12.0)
    m2 = EdgeModelSpec(model_id="log_transformer_slm", base_ram_bytes=24 * 1024 * 1024, base_latency_ms=18.0)
    mgr.register_model(m1)
    mgr.register_model(m2)

    assert mgr.total_ram_used() == 48 * 1024 * 1024

    # Optimize for normal thermal state: should quantize to INT8 (12 MB total <= 32 MB)
    res_normal = mgr.optimize_budgets(ThermalState.NORMAL)
    assert res_normal["within_budget"] is True
    assert res_normal["total_ram_bytes"] == 12 * 1024 * 1024
    assert m1.current_tier == EdgeQuantizationTier.INT8
    assert m1.current_latency_ms < 12.0

    # Optimize under thermal throttling: should aggressively quantize to INT4
    res_throttled = mgr.optimize_budgets(ThermalState.THROTTLED)
    assert res_throttled["within_budget"] is True
    assert res_throttled["thermal_state"] == "THROTTLED"
    assert m1.current_tier == EdgeQuantizationTier.INT4
    assert res_throttled["total_ram_bytes"] == 6 * 1024 * 1024  # 48 MB * 0.125 = 6 MB


def test_can_bus_signal_anomaly_detector_streaming() -> None:
    """Verifies streaming CAN signal anomaly detection and rolling z-scores."""
    detector = CANBusSignalAnomalyDetector(window_size=30, z_threshold=3.0)

    # Ingest 25 normal battery voltage readings (around 380V +/- 1V)
    for _ in range(25):
        reading = CANSignalReading(signal_name="battery_cell_voltage_v", value=380.0)
        res = detector.ingest_reading(reading)

    assert res["status"] == "NORMAL"
    assert res["is_anomaly"] is False
    assert res["confidence"] > 0.8

    # Inject severe voltage drop (anomaly)
    anom_reading = CANSignalReading(signal_name="battery_cell_voltage_v", value=340.0)
    anom_res = detector.ingest_reading(anom_reading)

    assert anom_res["is_anomaly"] is True
    assert anom_res["status"] == "ANOMALY"
    assert anom_res["z_score"] >= 3.0


def test_can_bus_multi_signal_correlation() -> None:
    """Verifies multi-modal correlation of compound powertrain faults."""
    detector = CANBusSignalAnomalyDetector(window_size=20, z_threshold=2.5)

    # Train baselines for both signals
    for _ in range(20):
        detector.ingest_reading(CANSignalReading("battery_cell_voltage_v", 400.0))
        detector.ingest_reading(CANSignalReading("motor_inverter_temp_c", 65.0))

    # Send simultaneous anomalies: battery voltage plunge + motor inverter overheating
    anom_readings = [
        CANSignalReading("battery_cell_voltage_v", 320.0),
        CANSignalReading("motor_inverter_temp_c", 115.0),
    ]

    eval_res = detector.evaluate_multi_signal_correlation(anom_readings)
    assert eval_res["severity"] == IncidentSeverity.CRITICAL.value
    assert eval_res["fault_type"] == "POWERTRAIN_THERMAL_ELECTRICAL_FAULT"
    assert eval_res["recommended_action"] == "DERATE_MOTOR_AND_ISOLATE_PACK"
    assert eval_res["anomalous_signals_count"] == 2


def test_kernel_log_cluster_engine() -> None:
    """Verifies abstraction of volatile tokens, entropy calculation, and semantic clustering."""
    engine = KernelLogClusterEngine()

    log1 = KernelLogEntry(message="[dmesg] DMA error at 0x7ffd9820 for PID 1042 on interface eth0")
    log2 = KernelLogEntry(message="[dmesg] DMA error at 0x7ffd9940 for PID 1089 on interface eth0")
    log3 = KernelLogEntry(message="[dmesg] PCIe link down on slot 2 with UUID 12345678-1234-5678-1234-567812345678")

    res1 = engine.ingest_log(log1)
    res2 = engine.ingest_log(log2)
    res3 = engine.ingest_log(log3)

    assert res1["template"] == "[dmesg] DMA error at <ADDR> for PID <NUM> on interface eth0"
    assert res2["template"] == "[dmesg] DMA error at <ADDR> for PID <NUM> on interface eth0"
    assert res2["frequency"] == 2
    assert res3["template"] == "[dmesg] PCIe link down on slot <NUM> with UUID <UUID>"

    clusters = engine.cluster_templates()
    assert len(clusters) == 2
    assert clusters[0]["total_occurrences"] >= 1


def test_hybrid_edge_cloud_telemetry_triager_closed_loop() -> None:
    """Verifies closed-loop triage: high compression in normal mode vs uncompressed incident escalation."""
    detector = CANBusSignalAnomalyDetector(window_size=20)
    log_engine = KernelLogClusterEngine()
    budget_mgr = EdgeResourceBudgetManager(max_ram_bytes=32 * 1024 * 1024)
    model = EdgeModelSpec(model_id="sdv_health_net", base_ram_bytes=16 * 1024 * 1024, base_latency_ms=10.0)
    budget_mgr.register_model(model)

    triager = HybridEdgeCloudTelemetryTriager(detector, log_engine, budget_mgr, compression_ratio=28.0)

    # Establish baseline for both signals
    for _ in range(15):
        detector.ingest_reading(CANSignalReading("motor_inverter_temp_c", 60.0))
        detector.ingest_reading(CANSignalReading("battery_cell_voltage_v", 400.0))

    # Cycle 1: Normal operations -> should compress data via TSDB 28x
    normal_can = [CANSignalReading("motor_inverter_temp_c", 60.1) for _ in range(10)]
    normal_logs = [KernelLogEntry("Routine sensor polling tick completed") for _ in range(5)]

    cycle_normal = triager.triage_cycle(normal_can, normal_logs, ThermalState.NORMAL)
    assert cycle_normal["transmission_mode"] == "COMPRESSED_TSDB_BATCH"
    assert cycle_normal["bandwidth_reduction_pct"] > 90.0
    assert cycle_normal["overall_severity"] in [IncidentSeverity.NORMAL.value, IncidentSeverity.INFO.value]

    # Cycle 2: Incur severe compound anomaly -> should escalate immediately without lossy compression
    anom_can = [
        CANSignalReading("battery_cell_voltage_v", 310.0),
        CANSignalReading("motor_inverter_temp_c", 120.0),
    ]
    anom_logs = [KernelLogEntry("CRITICAL: Inverter hardware thermal trip at 0xdeadbeef")]

    cycle_anom = triager.triage_cycle(anom_can, anom_logs, ThermalState.WARMED)
    assert cycle_anom["transmission_mode"] == "IMMEDIATE_INCIDENT_ESCALATION"
    assert cycle_anom["bandwidth_reduction_pct"] == 0.0
    assert cycle_anom["overall_severity"] == IncidentSeverity.CRITICAL.value
    assert cycle_anom["can_analysis"]["fault_type"] == "POWERTRAIN_THERMAL_ELECTRICAL_FAULT"
    assert cycle_anom["edge_budgets"]["within_budget"] is True
