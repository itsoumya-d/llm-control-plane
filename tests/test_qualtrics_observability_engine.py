"""Unit tests for Qualtrics Observability Engine."""

import time

from llm_control_plane.qualtrics_observability_engine import (
    CustomerExperienceCorrelator,
    EksObservabilityAutomator,
    MTTRReductionRootCauseAnalyzer,
    TelemetryEvent,
    TelemetryIngestionStream,
    TelemetryType,
)


def test_telemetry_event_hash_and_deduplication():
    ev1 = TelemetryEvent(
        event_id="ev-1",
        telemetry_type=TelemetryType.METRIC,
        service_name="survey-runtime",
        cluster_id="eks-eu-west-1-prod",
        timestamp=1000.0,
        metric_name="cpu_usage_pct",
        metric_value=84.5,
    )
    ev2 = TelemetryEvent(
        event_id="ev-2",
        telemetry_type=TelemetryType.METRIC,
        service_name="survey-runtime",
        cluster_id="eks-eu-west-1-prod",
        timestamp=1000.0,
        metric_name="cpu_usage_pct",
        metric_value=84.5,
    )
    assert ev1.compute_hash() == ev2.compute_hash()

    stream = TelemetryIngestionStream(max_buffer_size=10)
    assert stream.ingest(ev1) is True
    assert stream.ingest(ev2) is False  # Deduplicated
    assert stream.total_deduplicated == 1
    assert stream.buffer_depth == 1


def test_telemetry_ingestion_stream_eviction_and_flush():
    stream = TelemetryIngestionStream(max_buffer_size=3)
    for i in range(5):
        ev = TelemetryEvent(
            event_id=f"ev-{i}",
            telemetry_type=TelemetryType.LOG,
            service_name="survey-engine",
            cluster_id="eks-dublin",
            timestamp=float(i),
            log_message=f"Processing survey {i}",
        )
        stream.ingest(ev)

    assert stream.buffer_depth == 3
    batch = stream.flush_batch(batch_size=2)
    assert len(batch) == 2
    assert stream.buffer_depth == 1


def test_customer_experience_correlator_sla_breach():
    correlator = CustomerExperienceCorrelator(latency_sla_ms=200.0, error_rate_sla=0.05)
    now = time.time()
    events = [
        TelemetryEvent(
            event_id=f"e-{i}",
            telemetry_type=TelemetryType.TRACE,
            service_name="response-collector",
            cluster_id="eks-dublin",
            timestamp=now - 5,
            latency_ms=300.0 if i % 2 == 0 else 150.0,
            http_status=500 if i == 0 else 200,
        )
        for i in range(10)
    ]

    signal = correlator.evaluate_customer_impact("response-collector", events)
    assert signal.is_breached is True
    assert signal.p99_latency_ms >= 200.0
    assert signal.error_rate == 0.10
    assert "p99 latency" in (signal.breach_reason or "")
    assert "error rate" in (signal.breach_reason or "")


def test_customer_experience_correlator_empty():
    correlator = CustomerExperienceCorrelator()
    signal = correlator.evaluate_customer_impact("empty-service", [])
    assert signal.is_breached is False
    assert signal.p99_latency_ms == 0.0
    assert signal.error_rate == 0.0


def test_pearson_correlation_calculation():
    correlator = CustomerExperienceCorrelator()
    infra_metrics = [10.0, 20.0, 30.0, 40.0, 50.0]
    customer_latencies = [100.0, 200.0, 300.0, 400.0, 500.0]

    r = correlator.correlate_infrastructure_metric(infra_metrics, customer_latencies)
    assert r == 1.0

    # Negative correlation
    r_neg = correlator.correlate_infrastructure_metric(infra_metrics, [500.0, 400.0, 300.0, 200.0, 100.0])
    assert r_neg == -1.0

    # Zero variance returns 0.0
    r_zero = correlator.correlate_infrastructure_metric([10.0, 10.0, 10.0], [1.0, 2.0, 3.0])
    assert r_zero == 0.0


def test_mttr_reduction_root_cause_analysis():
    analyzer = MTTRReductionRootCauseAnalyzer()
    correlator = CustomerExperienceCorrelator()
    signal = correlator.evaluate_customer_impact("survey-api", [])

    correlations = {
        "container_cpu_throttled_seconds": 0.94,
        "container_memory_working_set_bytes": 0.35,
        "dns_lookup_latency_seconds": 0.12,
    }

    brief = analyzer.analyze_incident(
        service_name="survey-api",
        cluster_id="eks-eu-west-1",
        impact_signal=signal,
        metric_correlations=correlations,
    )

    assert brief.culprit_component == "container_cpu_throttled_seconds"
    assert brief.correlation_score == 0.94
    assert "cpu-scaling" in brief.suggested_runbook
    assert "CPU throttling" in brief.root_cause_summary


def test_eks_observability_automator():
    automator = EksObservabilityAutomator(cluster_id="eks-dublin-core")
    manifest = automator.generate_service_monitor_manifest(
        service_name="telemetry-aggregator",
        namespace="observability",
    )

    assert manifest["kind"] == "ServiceMonitor"
    assert manifest["spec"]["endpoints"][0]["interval"] == "15s"

    # Perfect match reconciliation
    match, drifts = automator.reconcile_monitor_drift("telemetry-aggregator", manifest)
    assert match is True
    assert drifts == []

    # Drifted match
    drifted_manifest = {
        "spec": {
            "endpoints": [
                {
                    "interval": "60s",  # drifted
                    "path": "/custom-metrics",  # drifted
                }
            ]
        }
    }
    match, drifts = automator.reconcile_monitor_drift("telemetry-aggregator", drifted_manifest)
    assert match is False
    assert len(drifts) == 2
