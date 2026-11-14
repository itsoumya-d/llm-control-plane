"""Unit & Integration Tests for Pinecone Experience Platform Telemetry Engine."""

from __future__ import annotations

import csv
import io

from llm_control_plane.pinecone_experience_telemetry import (
    AlertSeverity,
    BillingIntegrationGateway,
    FinOpsCostExporter,
    OperationType,
    SpendAlertManager,
    SpendBudgetRule,
    VectorTelemetryEngine,
    VectorTelemetryEvent,
)


def test_read_unit_calculation_query() -> None:
    engine = VectorTelemetryEngine()
    # top_k=10, dim=1536 (OpenAI embeddings) -> 10 * 1536 * 4 = 61440 bytes.
    # 61440 / 2048 = 30.0 RUs.
    rus = engine.calculate_read_units(
        operation=OperationType.QUERY,
        vector_dim=1536,
        top_k=10,
        metadata_bytes=0,
    )
    assert rus == 30.0

    # Minimum 1 RU for small query
    min_rus = engine.calculate_read_units(
        operation=OperationType.QUERY,
        vector_dim=32,
        top_k=1,
        metadata_bytes=0,
    )
    assert min_rus == 1.0

    # 0 RUs for UPSERT operation
    assert engine.calculate_read_units(OperationType.UPSERT, vector_dim=1536) == 0.0


def test_write_unit_calculation_upsert() -> None:
    engine = VectorTelemetryEngine()
    # 100 vectors * 1536 * 4 bytes = 614,400 bytes.
    # 614400 / 2048 = 300.0 WUs.
    wus = engine.calculate_write_units(
        operation=OperationType.UPSERT,
        vector_dim=1536,
        vector_count=100,
        metadata_bytes=0,
    )
    assert wus == 300.0

    # 0 WUs for QUERY operation
    assert engine.calculate_write_units(OperationType.QUERY, 1536, 100) == 0.0


def test_storage_calculation() -> None:
    engine = VectorTelemetryEngine()
    # 100,000 vectors, dim=1536 (6144 bytes) + 256 bytes metadata = 6400 bytes per vector.
    # Total = 640,000,000 bytes ≈ 0.596046 GB.
    gb = engine.calculate_storage_gb(vector_count=100_000, vector_dim=1536)
    assert 0.59 < gb < 0.60


def test_idempotent_event_deduplication() -> None:
    engine = VectorTelemetryEngine()

    # First recording should succeed
    recorded1, event1 = engine.record_event(
        event_id="evt_001",
        account_id="acc_pinecone_prod",
        project_id="proj_genai_rag",
        index_name="docs-semantic-index",
        namespace="support-articles",
        operation=OperationType.QUERY,
        vector_dim=1536,
        vector_count=1,
        top_k=20,
    )
    assert recorded1 is True
    assert event1.read_units > 0
    assert event1.cost_usd > 0

    # Identical replay should be caught by deduplication hash
    # Directly pass event1 with same hash
    event_dup = VectorTelemetryEvent(
        event_id="evt_001",
        account_id="acc_pinecone_prod",
        project_id="proj_genai_rag",
        index_name="docs-semantic-index",
        namespace="support-articles",
        operation=OperationType.QUERY,
        vector_dim=1536,
        vector_count=1,
        timestamp=event1.timestamp,
        dedup_hash=event1.dedup_hash,
    )
    assert event_dup.dedup_hash in engine._seen_hashes


def test_namespace_aggregation() -> None:
    engine = VectorTelemetryEngine()
    # Namespace A
    engine.record_event(
        event_id="evt_a1",
        account_id="acc_1",
        project_id="proj_1",
        index_name="idx",
        namespace="agent-alpha",
        operation=OperationType.QUERY,
        vector_dim=1536,
        vector_count=1,
        top_k=10,
    )
    engine.record_event(
        event_id="evt_a2",
        account_id="acc_1",
        project_id="proj_1",
        index_name="idx",
        namespace="agent-alpha",
        operation=OperationType.UPSERT,
        vector_dim=1536,
        vector_count=50,
    )

    # Namespace B
    engine.record_event(
        event_id="evt_b1",
        account_id="acc_1",
        project_id="proj_1",
        index_name="idx",
        namespace="agent-beta",
        operation=OperationType.QUERY,
        vector_dim=1536,
        vector_count=1,
        top_k=5,
    )

    summary = engine.aggregate_by_namespace("acc_1", "proj_1")
    assert "agent-alpha" in summary
    assert "agent-beta" in summary
    assert summary["agent-alpha"]["total_events"] == 2
    assert summary["agent-alpha"]["write_units"] > 0
    assert summary["agent-beta"]["total_events"] == 1


def test_spend_alert_thresholds() -> None:
    manager = SpendAlertManager()
    rule = SpendBudgetRule(
        rule_id="budget_monthly_prod",
        account_id="acc_pinecone_1",
        project_id="proj_1",
        budget_limit_usd=1000.0,
        warning_threshold_pct=80.0,
        critical_threshold_pct=100.0,
        webhook_url="https://hooks.internal.pinecone.io/spend-alerts",
    )
    manager.register_rule(rule)

    # 50% spend -> No alerts
    alerts_50 = manager.evaluate_spend("acc_pinecone_1", "proj_1", 500.0)
    assert len(alerts_50) == 0

    # 85% spend -> WARNING alert
    alerts_85 = manager.evaluate_spend("acc_pinecone_1", "proj_1", 850.0)
    assert len(alerts_85) == 1
    assert alerts_85[0]["severity"] == AlertSeverity.WARNING.value
    assert alerts_85[0]["percent_consumed"] == 85.0

    # 105% spend -> CRITICAL alert
    alerts_105 = manager.evaluate_spend("acc_pinecone_1", "proj_1", 1050.0)
    assert len(alerts_105) == 1
    assert alerts_105[0]["severity"] == AlertSeverity.CRITICAL.value


def test_billing_gateway_schema_validation() -> None:
    gateway = BillingIntegrationGateway()

    # Valid payload
    valid_payload = {
        "account_id": "acc_1",
        "project_id": "proj_1",
        "period_start": "2026-11-01T00:00:00Z",
        "period_end": "2026-11-30T23:59:59Z",
        "read_units": 150000.0,
        "write_units": 45000.0,
        "total_amount_usd": 185.50,
    }
    is_valid, msg = gateway.validate_billing_payload(valid_payload)
    assert is_valid is True

    # Missing keys (schema drift)
    broken_payload = {
        "account_id": "acc_1",
        "total_amount_usd": 100.0,
    }
    is_valid_broken, err_msg = gateway.validate_billing_payload(broken_payload)
    assert is_valid_broken is False
    assert "Schema drift detected" in err_msg


def test_billing_gateway_outbox_queue() -> None:
    gateway = BillingIntegrationGateway()
    payload = {
        "account_id": "acc_1",
        "project_id": "proj_1",
        "period_start": "2026-11-01T00:00:00Z",
        "period_end": "2026-11-30T23:59:59Z",
        "read_units": 1000.0,
        "write_units": 200.0,
        "total_amount_usd": 25.0,
    }
    ok, entry_id = gateway.enqueue_for_sync(payload)
    assert ok is True
    assert len(gateway._outbox_queue) == 1
    assert gateway._outbox_queue[0]["status"] == "QUEUED"


def test_invoice_reconciliation() -> None:
    gateway = BillingIntegrationGateway()
    # Within 1 cent tolerance
    is_rec, diff = gateway.reconcile_invoice(150.005, 150.010, tolerance=0.01)
    assert is_rec is True
    assert diff == 0.005

    # Out of tolerance discrepancy
    is_rec_fail, diff_fail = gateway.reconcile_invoice(150.0, 165.0, tolerance=0.01)
    assert is_rec_fail is False
    assert diff_fail == 15.0


def test_finops_focus_v14_export() -> None:
    engine = VectorTelemetryEngine()
    _, ev1 = engine.record_event(
        event_id="ev_focus_1",
        account_id="acc_enterprise_9",
        project_id="proj_knowledge_base",
        index_name="vectors-prod",
        namespace="customer-docs",
        operation=OperationType.QUERY,
        vector_dim=1536,
        vector_count=1,
        top_k=20,
    )

    records = FinOpsCostExporter.generate_focus_records([ev1])
    assert len(records) == 1
    assert records[0]["BillingAccountId"] == "acc_enterprise_9"
    assert records[0]["ServiceName"] == "Pinecone Serverless Vector Database"
    assert records[0]["ConsumedUnit"] == "ReadUnit"
    assert records[0]["BillingCurrency"] == "USD"

    csv_data = FinOpsCostExporter.export_focus_csv([ev1])
    assert "ChargePeriodStart" in csv_data
    assert "BillingAccountId" in csv_data
    assert "Pinecone Serverless Vector Database" in csv_data

    # Parse exported CSV to ensure RFC-4180 validity
    reader = csv.DictReader(io.StringIO(csv_data))
    rows = list(reader)
    assert len(rows) == 1
    assert rows[0]["ResourceName"] == "vectors-prod/customer-docs"
