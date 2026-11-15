"""Unit and integration tests for Superchat Data Platform Orchestrator."""

import datetime as dt

from llm_control_plane.superchat_data_platform_orchestrator import (
    ChannelSource,
    DagsterOrchestrationEngine,
    DbtLayer,
    DbtModelReconciler,
    DbtModelSpec,
    OmnichannelIngestionPipeline,
    RawEventRecord,
    RedshiftWarehouseManager,
    ReverseEtlSyncAuditor,
    SCD2SnapshotEngine,
)


def test_omnichannel_idempotent_ingestion() -> None:
    pipeline = OmnichannelIngestionPipeline()
    now = dt.datetime(2026, 11, 15, 8, 0, 0, tzinfo=dt.timezone.utc)

    rec1 = RawEventRecord(
        event_id="msg_001",
        channel=ChannelSource.WHATSAPP,
        workspace_id="ws_123",
        payload={"from": "+49151234567", "text": "Hello Superchat"},
        received_at=now,
        cursor_position=101,
    )
    # First ingest must pass
    assert pipeline.ingest_event(rec1) is True
    # Duplicate ingest must be rejected
    assert pipeline.ingest_event(rec1) is False
    assert pipeline.get_highest_cursor(ChannelSource.WHATSAPP) == 101

    batch = pipeline.flush_microbatch(10)
    assert len(batch) == 1
    assert batch[0].event_id == "msg_001"
    # Flushed buffer is empty
    assert len(pipeline.flush_microbatch(10)) == 0


def test_scd2_subscription_snapshot_history() -> None:
    engine = SCD2SnapshotEngine()
    t1 = dt.datetime(2026, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc)
    t2 = dt.datetime(2026, 6, 1, 0, 0, 0, tzinfo=dt.timezone.utc)
    t3 = dt.datetime(2026, 11, 1, 0, 0, 0, tzinfo=dt.timezone.utc)

    # Initial Starter plan (€99/mo = 9900 cents)
    engine.record_change("sub_abc", "ws_1", "starter", 9900, t1)
    # Upgrade to Pro (€299/mo = 29900 cents)
    engine.record_change("sub_abc", "ws_1", "pro", 29900, t2)

    # In March 2026, user was on Starter
    snap_march = engine.get_snapshot_at("sub_abc", dt.datetime(2026, 3, 15, 0, 0, 0, tzinfo=dt.timezone.utc))
    assert snap_march is not None
    assert snap_march.plan_tier == "starter"
    assert snap_march.mrr_cents == 9900

    # In August 2026, user was on Pro
    snap_aug = engine.get_snapshot_at("sub_abc", dt.datetime(2026, 8, 1, 0, 0, 0, tzinfo=dt.timezone.utc))
    assert snap_aug is not None
    assert snap_aug.plan_tier == "pro"
    assert snap_aug.mrr_cents == 29900

    # Total active MRR at t3
    assert engine.calculate_active_mrr_at(t3) == 29900


def test_dbt_dag_execution_order_and_tests() -> None:
    reconciler = DbtModelReconciler()

    m_stg_whatsapp = DbtModelSpec(
        name="stg_whatsapp_messages",
        layer=DbtLayer.STAGING,
        sql_query="SELECT * FROM raw_whatsapp",
    )
    m_int_conversations = DbtModelSpec(
        name="int_conversations",
        layer=DbtLayer.INTERMEDIATE,
        sql_query="SELECT * FROM stg_whatsapp_messages",
        upstream_dependencies=["stg_whatsapp_messages"],
    )
    m_fct_mrr = DbtModelSpec(
        name="fct_workspace_mrr",
        layer=DbtLayer.GOLD,
        sql_query="SELECT * FROM int_conversations",
        upstream_dependencies=["int_conversations"],
        tests=["not_null", "unique", "positive_mrr"],
    )

    reconciler.register_model(m_fct_mrr)
    reconciler.register_model(m_int_conversations)
    reconciler.register_model(m_stg_whatsapp)

    order = reconciler.resolve_execution_order()
    assert order == ["stg_whatsapp_messages", "int_conversations", "fct_workspace_mrr"]

    # Test runner verification
    sample_rows = [
        {"id": "1", "workspace_id": "ws_1", "mrr_cents": 15000},
        {"id": "2", "workspace_id": "ws_2", "mrr_cents": 25000},
    ]
    test_results = reconciler.run_tests("fct_workspace_mrr", sample_rows)
    assert test_results["not_null"] is True
    assert test_results["unique"] is True
    assert test_results["positive_mrr"] is True


def test_dagster_freshness_alerting() -> None:
    orchestrator = DagsterOrchestrationEngine()
    t_start = dt.datetime(2026, 11, 15, 4, 0, 0, tzinfo=dt.timezone.utc)
    t_now = dt.datetime(2026, 11, 15, 9, 0, 0, tzinfo=dt.timezone.utc)  # 5 hours later

    orchestrator.register_asset("airbyte_chargebee_invoices", max_lag_seconds=14400.0)  # 4h SLA
    orchestrator.record_materialization("airbyte_chargebee_invoices", t_start)

    # Stale because 5 hours elapsed > 4 hours SLA
    stale = orchestrator.evaluate_freshness(t_now)
    assert "airbyte_chargebee_invoices" in stale
    assert len(orchestrator.alert_log) == 1
    assert "lag_exceeded" in orchestrator.alert_log[0]["reason"]


def test_redshift_dist_keys_and_vacuum_hygiene() -> None:
    manager = RedshiftWarehouseManager()
    assert manager.recommend_dist_key("fct_conversation_messages") == "workspace_id"
    assert manager.recommend_dist_key("dim_subscriptions") == "workspace_id"
    assert manager.recommend_dist_key("audit_log") == "AUTO"

    # Vacuum recommendations
    assert manager.requires_vacuum(unsorted_percent=20.0, deleted_rows_percent=5.0) is True
    assert manager.requires_vacuum(unsorted_percent=5.0, deleted_rows_percent=12.0) is True
    assert manager.requires_vacuum(unsorted_percent=5.0, deleted_rows_percent=2.0) is False


def test_reverse_etl_drift_detection() -> None:
    auditor = ReverseEtlSyncAuditor()
    # Healthy sync (< 1% drift)
    res_healthy = auditor.audit_sync("sync_01", source_gold_rows=10000, destination_target_rows=9995)
    assert res_healthy["status"] == "HEALTHY"

    # Drift alert (> 1% drift)
    res_drift = auditor.audit_sync("sync_02", source_gold_rows=10000, destination_target_rows=9700)
    assert res_drift["status"] == "DRIFT_ALERT"
    assert res_drift["drift_rate"] == 0.03
