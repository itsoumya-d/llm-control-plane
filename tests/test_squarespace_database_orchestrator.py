"""Unit tests for Squarespace Distributed Database Multi-Engine Replication & Connection Orchestrator."""

import time

from llm_control_plane.squarespace_database_orchestrator import (
    CDCEvent,
    CircuitState,
    ConnectionPool,
    DatabaseEngineType,
    KafkaEventPublisher,
    MigrationState,
    SchemaMigrationOrchestrator,
    SquarespaceDatabaseOrchestrator,
    TenantTier,
)


def test_connection_pool_tier_shedding():
    pool = ConnectionPool(engine=DatabaseEngineType.COCKROACHDB, max_connections=10)

    # Fill to 80% (8 connections)
    for _ in range(8):
        assert pool.acquire(TenantTier.ENTERPRISE) is True

    # Standard tier should now be shed (utilization >= 80%)
    assert pool.can_acquire(TenantTier.STANDARD) is False
    assert pool.acquire(TenantTier.STANDARD) is False

    # Growth tier should still be allowed (< 95%)
    assert pool.can_acquire(TenantTier.GROWTH) is True
    assert pool.acquire(TenantTier.GROWTH) is True  # 9 connections (90%)

    # Enterprise tier should be allowed until 100%
    assert pool.acquire(TenantTier.ENTERPRISE) is True  # 10 connections (100%)
    assert pool.can_acquire(TenantTier.ENTERPRISE) is False

    # Release one connection
    pool.release(success=True)
    assert pool.active_connections == 9


def test_circuit_breaker_failure_and_recovery():
    pool = ConnectionPool(
        engine=DatabaseEngineType.POSTGRESQL,
        max_connections=10,
        circuit_recovery_timeout_sec=0.1,
    )

    # Record 5 consecutive failures to trip circuit
    for _ in range(5):
        assert pool.acquire(TenantTier.ENTERPRISE) is True
        pool.release(success=False)

    assert pool.circuit_state == CircuitState.OPEN
    assert pool.can_acquire(TenantTier.ENTERPRISE) is False

    # Wait for recovery timeout
    time.sleep(0.12)
    assert pool.can_acquire(TenantTier.ENTERPRISE) is True

    # Successful acquisition in half-open state heals the circuit
    assert pool.acquire(TenantTier.ENTERPRISE) is True
    assert pool.circuit_state == CircuitState.HALF_OPEN
    pool.release(success=True)
    assert pool.circuit_state == CircuitState.CLOSED
    assert pool.failure_count == 0


def test_kafka_cdc_publisher_partitioning():
    publisher = KafkaEventPublisher(num_partitions=8)

    tenant_a = "tenant-squarespace-ireland-101"
    tenant_b = "tenant-squarespace-nyc-202"

    p_a = publisher.get_partition(tenant_a)
    p_b = publisher.get_partition(tenant_b)

    assert 0 <= p_a < 8
    assert 0 <= p_b < 8
    # Consistent hashing guarantee
    assert publisher.get_partition(tenant_a) == p_a

    # Publish multiple events for tenant A, verifying sequential partition offsets
    event1 = CDCEvent(
        event_id="evt-1",
        tenant_id=tenant_a,
        engine=DatabaseEngineType.MONGODB,
        table_name="storefront_products",
        operation="INSERT",
        lsn_sequence=1001,
        before_state=None,
        after_state={"sku": "prod-1", "price_eur": 49.99},
        schema_version=1,
    )
    part1, off1 = publisher.publish_cdc(event1)
    assert part1 == p_a
    assert off1 == 0

    event2 = CDCEvent(
        event_id="evt-2",
        tenant_id=tenant_a,
        engine=DatabaseEngineType.MONGODB,
        table_name="storefront_products",
        operation="UPDATE",
        lsn_sequence=1002,
        before_state={"sku": "prod-1", "price_eur": 49.99},
        after_state={"sku": "prod-1", "price_eur": 39.99},
        schema_version=1,
    )
    part2, off2 = publisher.publish_cdc(event2)
    assert part2 == p_a
    assert off2 == 1
    assert publisher.published_count == 2


def test_schema_migration_lifecycle():
    migrator = SchemaMigrationOrchestrator(
        migration_id="mig-sqsp-001",
        table_name="billing_invoices",
        target_engine=DatabaseEngineType.COCKROACHDB,
    )
    assert migrator.state == MigrationState.IDLE

    # Step through valid migration stages
    assert migrator.advance_state(MigrationState.PRE_SYNC) is True
    assert migrator.advance_state(MigrationState.DUAL_WRITE) is True
    assert migrator.advance_state(MigrationState.BACKFILL) is True
    assert migrator.advance_state(MigrationState.VALIDATE_PARITY) is True

    # Parity check
    parity_ok = migrator.record_parity_checksums("hash-abc-123", "hash-abc-123")
    assert parity_ok is True

    assert migrator.advance_state(MigrationState.CUTOVER) is True
    assert migrator.advance_state(MigrationState.COMPLETE) is True


def test_schema_migration_rollback():
    migrator = SchemaMigrationOrchestrator(
        migration_id="mig-sqsp-002",
        table_name="domains_catalog",
        target_engine=DatabaseEngineType.COCKROACHDB,
    )
    assert migrator.advance_state(MigrationState.PRE_SYNC) is True
    assert migrator.advance_state(MigrationState.DUAL_WRITE) is True

    # Trigger emergency rollback
    migrator.trigger_rollback(reason="Replication lag exceeded 5000ms SLA")
    assert migrator.state == MigrationState.ROLLED_BACK
    assert migrator.rollback_reason == "Replication lag exceeded 5000ms SLA"

    # Cannot advance from rolled back
    assert migrator.advance_state(MigrationState.BACKFILL) is False


def test_orchestrator_dual_write_and_query():
    orch = SquarespaceDatabaseOrchestrator()

    # Query execution
    res = orch.execute_query(
        engine=DatabaseEngineType.COCKROACHDB,
        tier=TenantTier.ENTERPRISE,
        query_payload={"sql": "SELECT * FROM site_pages WHERE site_id = 'sqsp-1'"},
    )
    assert res["success"] is True
    assert res["engine"] == "cockroachdb"
    assert "latency_ms" in res

    # Dual-write execution
    dw_res = orch.execute_dual_write(
        tenant_id="tenant-acuity-scheduling-505",
        table_name="appointments",
        record={"appointment_id": "apt-99", "service": "consultation", "status": "CONFIRMED"},
        tier=TenantTier.GROWTH,
    )
    assert dw_res["success"] is True
    assert dw_res["transaction"]["status"] == "COMMITTED"
    assert dw_res["cdc_event"]["op"] == "INSERT"
    assert dw_res["cdc_event"]["table"] == "appointments"
    assert len(orch.transaction_history) == 1
