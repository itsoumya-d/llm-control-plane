"""
Tests for CrateDB Ecosystem Driver Adapter & Telemetry Store.
"""

import pytest
from llm_control_plane.cratedb_ecosystem_driver_adapter import (
    CrateDBDriverConfig,
    CrateDBNode,
    CrateDBSQLDialect,
    CrateDBConnectionPool,
    CrateDBTelemetryStore,
)


def test_dialect_create_table():
    cols = {
        "id": "TEXT",
        "timestamp": "TIMESTAMP WITH TIME ZONE",
        "prompt_tokens": "INTEGER",
        "embedding": "FLOAT_VECTOR(1536)",
    }
    ddl = CrateDBSQLDialect.build_create_table_statement(
        table_name="llm_telemetry",
        columns=cols,
        primary_key=["id", "timestamp"],
        partition_by="timestamp",
        shards=6,
        replicas="1-2"
    )
    assert 'CREATE TABLE IF NOT EXISTS "llm_telemetry"' in ddl
    assert 'PRIMARY KEY ("id", "timestamp")' in ddl
    assert 'PARTITIONED BY ("timestamp")' in ddl
    assert "CLUSTERED INTO 6 SHARDS" in ddl
    assert "number_of_replicas = '1-2'" in ddl


def test_dialect_knn_vector_query():
    query_vec = [0.1, 0.2, 0.3]
    sql, params = CrateDBSQLDialect.rewrite_knn_vector_query(
        table_name="prompt_vectors",
        vector_column="embedding",
        query_vector=query_vec,
        k=5,
        filters={"model": "gpt-4o", "tenant_id": "tenant-123"},
        fields=["id", "model", "content"]
    )
    assert 'SELECT "id", "model", "content", _score AS similarity FROM "prompt_vectors"' in sql
    assert "knn_match(embedding, $1, $2)" in sql
    assert '"model" = $3' in sql
    assert '"tenant_id" = $4' in sql
    assert params[0] == query_vec
    assert params[1] == 5
    assert params[2] == "gpt-4o"
    assert params[3] == "tenant-123"


def test_dialect_upsert_query():
    cols = ["id", "latency_ms", "status"]
    sql = CrateDBSQLDialect.rewrite_upsert_query(
        table_name="task_status",
        conflict_keys=["id"],
        update_columns=["latency_ms", "status"],
        columns=cols
    )
    assert 'INSERT INTO "task_status" ("id", "latency_ms", "status")' in sql
    assert 'VALUES ($1, $2, $3)' in sql
    assert 'ON CONFLICT ("id") DO UPDATE SET "latency_ms" = excluded."latency_ms", "status" = excluded."status"' in sql


def test_dialect_time_series_aggregation():
    sql, params = CrateDBSQLDialect.build_time_series_aggregation_query(
        table_name="metrics",
        timestamp_column="recorded_at",
        metric_column="latency_ms",
        interval="hour",
        start_time_iso="2026-11-01T00:00:00Z",
        end_time_iso="2026-11-16T00:00:00Z"
    )
    assert "date_trunc('hour', \"recorded_at\") AS time_bucket" in sql
    assert 'AVG("latency_ms") AS avg_metric' in sql
    assert '"recorded_at" >= $1' in sql
    assert '"recorded_at" <= $2' in sql
    assert params == ["2026-11-01T00:00:00Z", "2026-11-16T00:00:00Z"]


def test_connection_pool_round_robin():
    nodes = [
        CrateDBNode(host="node1.cratedb.internal"),
        CrateDBNode(host="node2.cratedb.internal"),
    ]
    cfg = CrateDBDriverConfig(nodes=nodes)
    pool = CrateDBConnectionPool(cfg)

    n1 = pool.get_healthy_node()
    n2 = pool.get_healthy_node()
    n3 = pool.get_healthy_node()

    assert n1.host == "node1.cratedb.internal"
    assert n2.host == "node2.cratedb.internal"
    assert n3.host == "node1.cratedb.internal"


def test_connection_pool_failover():
    node1 = CrateDBNode(host="failing-node.internal")
    node2 = CrateDBNode(host="healthy-node.internal")
    cfg = CrateDBDriverConfig(nodes=[node1, node2], max_retries=2, retry_backoff_sec=0.01)
    pool = CrateDBConnectionPool(cfg)

    call_records = []

    def mock_executor(node, sql, params):
        call_records.append(node.host)
        if node.host == "failing-node.internal":
            raise ConnectionError("Node unreachable")
        return [{"status": "success", "rows": 1}]

    pool.set_executor(mock_executor)

    res = pool.execute_with_failover("SELECT 1")
    assert res == [{"status": "success", "rows": 1}]
    assert "failing-node.internal" in call_records
    assert "healthy-node.internal" in call_records
    assert not node1.is_healthy
    assert node2.is_healthy


def test_telemetry_store_buffering_and_flush():
    cfg = CrateDBDriverConfig(max_batch_size=2)
    pool = CrateDBConnectionPool(cfg)
    store = CrateDBTelemetryStore(cfg, pool)

    flushed_records = []
    pool.set_executor(lambda node, sql, params: flushed_records.append(params) or [{"ok": 1}])

    # 1st event -> buffered
    store.record_telemetry_event(
        request_id="req-1",
        provider="anthropic",
        model="claude-3-7-sonnet",
        latency_ms=120.5,
        prompt_tokens=50,
        completion_tokens=20,
        total_tokens=70,
        cost_usd=0.0015
    )
    assert store.get_metrics()["buffered_records"] == 1
    assert store.get_metrics()["total_flushed"] == 0

    # 2nd event -> reaches batch size (2) -> automatic flush
    store.record_telemetry_event(
        request_id="req-2",
        provider="openai",
        model="gpt-4o",
        latency_ms=85.0,
        prompt_tokens=40,
        completion_tokens=15,
        total_tokens=55,
        cost_usd=0.0009
    )
    assert store.get_metrics()["buffered_records"] == 0
    assert store.get_metrics()["total_flushed"] == 2
    assert len(flushed_records) == 2


def test_telemetry_store_error_and_dead_letter():
    cfg = CrateDBDriverConfig(max_batch_size=10, max_retries=1, retry_backoff_sec=0.01)
    pool = CrateDBConnectionPool(cfg)
    store = CrateDBTelemetryStore(cfg, pool)

    def failing_executor(node, sql, params):
        raise TimeoutError("Cluster busy")

    pool.set_executor(failing_executor)

    store.record_telemetry_event(
        request_id="req-fail",
        provider="bedrock",
        model="claude-3",
        latency_ms=200.0,
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        cost_usd=0.0001
    )

    count = store.flush()
    assert count == 0
    m = store.get_metrics()
    assert m["flush_errors"] == 1
    assert m["dead_letter_count"] == 1


def test_grafana_metrics_export():
    cfg = CrateDBDriverConfig()
    pool = CrateDBConnectionPool(cfg)
    store = CrateDBTelemetryStore(cfg, pool)

    store.record_telemetry_event("req-m", "provider", "model", 10.0, 1, 1, 2, 0.0001)
    text = store.export_grafana_metrics()
    assert "cratedb_telemetry_buffered_count 1" in text
    assert "cratedb_telemetry_total_ingested 1" in text
    assert "cratedb_telemetry_total_flushed 0" in text
    assert "cratedb_telemetry_flush_errors 0" in text
