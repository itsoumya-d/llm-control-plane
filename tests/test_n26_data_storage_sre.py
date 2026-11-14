"""Unit tests for N26 Data Storage SRE Platform module."""

import time

from llm_control_plane.n26_data_storage_sre import (
    AuroraPostgresConnectionPoolOrchestrator,
    BaFinAuditSnapshotManager,
    ConnectionRequest,
    DatabaseNode,
    DatabasePlatformSLOTracker,
    ReplicationLagAndFailoverMonitor,
)


def test_aurora_connection_pool_routing():
    pool = AuroraPostgresConnectionPoolOrchestrator(cluster_id="n26-test-cluster")
    writer = DatabaseNode(
        node_id="aurora-writer-01", role="writer", host="writer.aurora.internal", max_connections=50
    )
    reader1 = DatabaseNode(
        node_id="aurora-reader-01", role="reader", host="reader1.aurora.internal", max_connections=50
    )
    reader2 = DatabaseNode(
        node_id="aurora-reader-02", role="reader", host="reader2.aurora.internal", max_connections=50
    )

    pool.register_node(writer)
    pool.register_node(reader1)
    pool.register_node(reader2)

    read_req = ConnectionRequest(
        client_id="transfers-service", query_type="read", sql_statement="SELECT * FROM transactions WHERE user_id = 42"
    )
    node, decision = pool.acquire_connection(read_req)
    assert decision == "reader_replica"
    assert node.role == "reader"
    assert node.active_connections == 1

    write_req = ConnectionRequest(
        client_id="payments-service",
        query_type="write",
        sql_statement="INSERT INTO ledgers (account, amount) VALUES ('DE123', 100)",
    )
    w_node, w_decision = pool.acquire_connection(write_req)
    assert w_decision == "writer_primary"
    assert w_node.role == "writer"
    assert w_node.active_connections == 1

    pool.release_connection(node.node_id)
    assert node.active_connections == 0

    saturation = pool.get_cluster_saturation()
    assert saturation["cluster_id"] == "n26-test-cluster"
    assert saturation["total_nodes"] == 3


def test_replication_lag_and_failover_evaluation():
    primary_id = "aurora-master-eu-central-1"
    monitor = ReplicationLagAndFailoverMonitor(primary_node_id=primary_id)

    base_time = time.time()
    monitor.record_heartbeat(primary_id, lsn=100000, current_timestamp=base_time)
    monitor.record_heartbeat("replica-01", lsn=100000, current_timestamp=base_time - 0.01)
    monitor.record_heartbeat("replica-02", lsn=95000, current_timestamp=base_time - 2.0)

    lag1 = monitor.compute_replication_lag("replica-01")
    assert lag1["status"] == "synchronized"
    assert lag1["lag_lsn_bytes"] == 0

    lag2 = monitor.compute_replication_lag("replica-02")
    assert lag2["status"] == "critical_lag"
    assert lag2["lag_lsn_bytes"] == 5000

    eval_res = monitor.evaluate_failover_candidates(max_tolerable_lag_ms=500.0)
    assert eval_res["failover_recommended"] is True
    assert eval_res["selected_candidate"] == "replica-01"
    assert eval_res["zero_data_loss_guaranteed"] is True


def test_bafin_audit_snapshot_integrity_and_export():
    manager = BaFinAuditSnapshotManager(institution_code="N26-BANK-BERLIN")
    snap = manager.create_immutable_snapshot(
        database_name="core_banking_ledger",
        lsn=8472910,
        transaction_count=150000,
        ledger_hash_root="a4f5b7c89e1d2c3b4a5f6e7d8c9b0a1f2e3d4c5b6a7f8e9d0c1b2a3f4e5d6c7",
    )
    assert snap["snapshot_id"].startswith("snap-core_banking_ledger-")
    assert snap["retention_period_years"] == 10
    assert len(snap["checksum_sha256"]) == 64

    assert manager.verify_snapshot_integrity(snap["snapshot_id"]) is True

    manager.snapshots[snap["snapshot_id"]]["transaction_count"] = 999999
    assert manager.verify_snapshot_integrity(snap["snapshot_id"]) is False

    manager.snapshots[snap["snapshot_id"]]["transaction_count"] = 150000
    export_res = manager.export_audit_archive_to_s3(snap["snapshot_id"], s3_bucket="n26-bafin-compliance-vault")
    assert export_res["status"] == "COMPLETED"
    assert "s3://n26-bafin-compliance-vault/audit_ledgers/core_banking_ledger/" in export_res["destination"]


def test_database_slo_tracker_and_prometheus_export():
    tracker = DatabasePlatformSLOTracker(target_availability=0.999, p99_latency_slo_ms=30.0)

    for i in range(99):
        q_type = "read" if i % 2 == 0 else "write"
        tracker.record_query(query_type=q_type, latency_ms=10.0 + (i % 10), success=True)
    tracker.record_query(query_type="read", latency_ms=100.0, success=False)

    metrics = tracker.calculate_slo_metrics()
    assert metrics["total_queries"] == 100
    assert metrics["failed_queries"] == 1
    assert metrics["availability"] == 0.99
    assert metrics["p99_latency_ms"] <= 30.0
    assert metrics["p99_slo_met"] is True

    prom_text = tracker.export_prometheus_metrics()
    assert "n26_db_query_total 100" in prom_text
    assert "n26_db_query_failures_total 1" in prom_text
    assert "n26_db_availability 0.99" in prom_text
