"""Tests for DuckDB analytical execution engine, dual-execution planner, and columnar caching."""

import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from llm_control_plane import (
    ColumnarBatch,
    ColumnarCache,
    ColumnarQueryResult,
    DualExecutionPlanner,
    ExecutionTarget,
    MotherDuckHybridEngine,
    QueryComplexity,
    QueryCostProfiler,
    QueryExecutionError,
    UnsafeQueryError,
)


def test_columnar_batch_basics() -> None:
    batch = ColumnarBatch(
        columns=["id", "metric"],
        column_types={"id": "INTEGER", "metric": "REAL"},
        data={"id": [1, 2, 3], "metric": [10.5, 20.0, 30.2]},
    )
    assert batch.row_count == 3

    empty_batch = ColumnarBatch(columns=[], column_types={}, data={})
    assert empty_batch.row_count == 0


def test_columnar_query_result_to_rows() -> None:
    batch = ColumnarBatch(
        columns=["event", "value"],
        column_types={"event": "TEXT", "value": "INTEGER"},
        data={"event": ["click", "view"], "value": [10, 20]},
    )
    result = ColumnarQueryResult(
        columns=["event", "value"],
        column_types={"event": "TEXT", "value": "INTEGER"},
        batches=[batch],
        row_count=2,
        execution_target=ExecutionTarget.LOCAL,
        execution_time_ms=1.5,
    )
    rows = result.to_rows()
    assert len(rows) == 2
    assert rows[0] == {"event": "click", "value": 10}
    assert rows[1] == {"event": "view", "value": 20}


def test_dual_execution_planner_routing() -> None:
    planner = DualExecutionPlanner(max_local_rows=50_000, heavy_query_cost_threshold=20.0)

    # Small local query
    sql_local = "SELECT id, count(*) FROM analytics_events GROUP BY id"
    target, complexity, reason = planner.route_query(sql_local, table_stats={"analytics_events": 5000})
    assert target == ExecutionTarget.LOCAL
    assert "fit local DuckDB footprint" in reason
    assert complexity.has_aggregates is True

    # High row count offload to cloud
    sql_cloud = "SELECT * FROM large_log_lake"
    target_cloud, complexity_cloud, reason_cloud = planner.route_query(
        sql_cloud, table_stats={"large_log_lake": 500_000}
    )
    assert target_cloud == ExecutionTarget.MOTHERDUCK_CLOUD
    assert "exceeds local threshold" in reason_cloud
    assert complexity_cloud.estimated_rows == 500_000

    # Complex join offload to cloud
    sql_heavy = "SELECT * FROM a JOIN b ON a.id = b.id JOIN c ON b.id = c.id"
    target_heavy, complexity_heavy, _ = planner.route_query(
        sql_heavy, table_stats={"a": 40_000, "b": 40_000, "c": 40_000}
    )
    assert target_heavy == ExecutionTarget.MOTHERDUCK_CLOUD
    assert complexity_heavy.has_joins is True


def test_dual_execution_planner_force_target() -> None:
    planner = DualExecutionPlanner()
    target, _, reason = planner.route_query(
        "SELECT 1",
        force_target=ExecutionTarget.MOTHERDUCK_CLOUD,
    )
    assert target == ExecutionTarget.MOTHERDUCK_CLOUD
    assert "Forced routing" in reason


def test_columnar_cache_operations() -> None:
    cache = ColumnarCache(max_entries=2, default_ttl_seconds=0.5)

    batch = ColumnarBatch(columns=["x"], column_types={"x": "INTEGER"}, data={"x": [42]})
    res = ColumnarQueryResult(
        columns=["x"],
        column_types={"x": "INTEGER"},
        batches=[batch],
        row_count=1,
        execution_target=ExecutionTarget.LOCAL,
        execution_time_ms=0.5,
    )

    sql = "SELECT 42 as x"
    cache.put(sql, res, tables=["numbers"])

    hit = cache.get(sql)
    assert hit is not None
    assert hit.cached is True
    assert hit.row_count == 1

    # Invalidate by table
    evicted = cache.invalidate_tables(["numbers"])
    assert evicted == 1
    assert cache.get(sql) is None

    # Test TTL expiry
    cache.put(sql, res, ttl_seconds=0.1)
    time.sleep(0.15)
    assert cache.get(sql) is None

    # Test LRU eviction
    cache.put("SELECT 1", res)
    cache.put("SELECT 2", res)
    cache.put("SELECT 3", res)
    cache.clear()
    assert cache.get("SELECT 1") is None


def test_query_cost_profiler() -> None:
    profiler = QueryCostProfiler()
    complexity = QueryComplexity(
        estimated_rows=1000,
        has_joins=False,
        has_window_functions=False,
        has_aggregates=True,
        estimated_cost=1.5,
        table_scans=["orders"],
    )

    p = profiler.record_profile(
        sql="SELECT count(*) FROM orders",
        target=ExecutionTarget.LOCAL,
        duration_ms=10.0,
        row_count=1000,
        complexity=complexity,
    )
    assert p["target"] == "LOCAL"
    assert p["throughput_rows_sec"] == 100_000.0

    summary = profiler.get_summary()
    assert summary["total_queries"] == 1
    assert summary["local_queries"] == 1
    assert summary["cloud_queries"] == 0


def test_motherduck_hybrid_engine_local_execution() -> None:
    engine = MotherDuckHybridEngine(database=":memory:")

    engine.register_table(
        name="telemetry",
        columns={"device_id": "TEXT", "latency_ms": "REAL", "errors": "INTEGER"},
        rows=[
            {"device_id": "dev_1", "latency_ms": 12.4, "errors": 0},
            {"device_id": "dev_2", "latency_ms": 45.1, "errors": 2},
            {"device_id": "dev_3", "latency_ms": 18.9, "errors": 0},
        ],
    )

    # Execute aggregation query
    res = engine.execute_query("SELECT count(*) as total, sum(errors) as total_errors FROM telemetry")
    assert res.row_count == 1
    assert res.execution_target == ExecutionTarget.LOCAL
    rows = res.to_rows()
    assert rows[0]["total"] == 3
    assert rows[0]["total_errors"] == 2

    # Verify cache hit on immediate second run
    cached_res = engine.execute_query("SELECT count(*) as total, sum(errors) as total_errors FROM telemetry")
    assert cached_res.cached is True

    engine.close()


def test_motherduck_hybrid_engine_cloud_offload() -> None:
    engine = MotherDuckHybridEngine(motherduck_token="md_test_token_123")

    engine.register_table(
        name="cloud_events",
        columns={"id": "INTEGER", "payload": "TEXT"},
        rows=[{"id": i, "payload": f"event_{i}"} for i in range(10)],
    )

    # Force cloud offload
    res = engine.execute_query(
        "SELECT * FROM cloud_events",
        force_target=ExecutionTarget.MOTHERDUCK_CLOUD,
    )
    assert res.execution_target == ExecutionTarget.MOTHERDUCK_CLOUD
    assert res.cost_metrics.get("cloud_offloaded") is True
    assert res.cost_metrics.get("motherduck_token_configured") is True
    assert res.row_count == 10

    engine.close()


def test_motherduck_guardrails_prevent_unsafe_queries() -> None:
    engine = MotherDuckHybridEngine()
    engine.register_table("safe_table", {"id": "INTEGER"})

    with pytest.raises(UnsafeQueryError, match="Prohibited mutation"):
        engine.execute_query("DROP TABLE safe_table")

    with pytest.raises(UnsafeQueryError, match="Prohibited mutation"):
        engine.execute_query("TRUNCATE safe_table")

    with pytest.raises(UnsafeQueryError, match="Prohibited mutation"):
        engine.execute_query("DELETE FROM safe_table WHERE 1=1")

    with pytest.raises(UnsafeQueryError, match="Invalid table identifier"):
        engine.register_table("bad-table-name!", {"id": "INTEGER"})

    engine.close()


def test_motherduck_query_execution_error() -> None:
    engine = MotherDuckHybridEngine()
    with pytest.raises(QueryExecutionError, match="Local query execution failed"):
        engine.execute_query("SELECT * FROM non_existent_table_xyz")

    with pytest.raises(QueryExecutionError, match="MotherDuck cloud query offload failed"):
        engine.execute_query("SELECT * FROM invalid_table", force_target=ExecutionTarget.MOTHERDUCK_CLOUD)

    engine.close()


def test_concurrent_query_execution() -> None:
    engine = MotherDuckHybridEngine()
    engine.register_table(
        "concurrent_data",
        {"val": "INTEGER"},
        [{"val": i} for i in range(50)],
    )

    def run_query(val_mult: int) -> int:
        res = engine.execute_query(f"SELECT sum(val) as total FROM concurrent_data WHERE val >= {val_mult}")
        return res.to_rows()[0]["total"]

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(run_query, i * 10) for i in range(4)]
        results = [f.result() for f in futures]

    assert len(results) == 4
    assert results[0] == sum(range(50))

    engine.close()
