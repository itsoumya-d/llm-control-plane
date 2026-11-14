"""Squarespace Distributed Database Multi-Engine Replication & Connection Orchestrator.

Provides high-throughput database connection multiplexing, multi-engine replication
coordination (CockroachDB, MongoDB, PostgreSQL), real-time Change Data Capture (CDC)
to Kafka streaming with total partition ordering, tenant-tiered priority queuing,
and automated zero-downtime schema migration workflows.
"""

from __future__ import annotations

import enum
import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class DatabaseEngineType(str, enum.Enum):
    """Supported database engines across Squarespace infrastructure."""
    COCKROACHDB = "cockroachdb"
    MONGODB = "mongodb"
    POSTGRESQL = "postgresql"


class TenantTier(str, enum.Enum):
    """Tenant priority tiers for database connection multiplexing."""
    ENTERPRISE = "enterprise"
    GROWTH = "growth"
    STANDARD = "standard"


class CircuitState(str, enum.Enum):
    """Circuit breaker states for database engine resilience."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class MigrationState(str, enum.Enum):
    """Zero-downtime online schema migration lifecycle states."""
    IDLE = "idle"
    PRE_SYNC = "pre_sync"
    DUAL_WRITE = "dual_write"
    BACKFILL = "backfill"
    VALIDATE_PARITY = "validate_parity"
    CUTOVER = "cutover"
    COMPLETE = "complete"
    ROLLED_BACK = "rolled_back"


@dataclass
class CDCEvent:
    """Change Data Capture (CDC) event captured from database write-ahead log."""
    event_id: str
    tenant_id: str
    engine: DatabaseEngineType
    table_name: str
    operation: str  # INSERT, UPDATE, DELETE
    lsn_sequence: int
    before_state: Optional[Dict[str, Any]]
    after_state: Optional[Dict[str, Any]]
    schema_version: int
    timestamp: float = field(default_factory=time.time)

    def to_kafka_payload(self) -> Dict[str, Any]:
        """Serialize CDC record to Avro-compatible Kafka message envelope."""
        return {
            "eventId": self.event_id,
            "tenantId": self.tenant_id,
            "engine": self.engine.value,
            "table": self.table_name,
            "op": self.operation,
            "lsn": self.lsn_sequence,
            "before": self.before_state,
            "after": self.after_state,
            "schemaVersion": self.schema_version,
            "ts": int(self.timestamp * 1000),
        }


@dataclass
class ConnectionPool:
    """Manages active connection pools and acquisition leases for a database engine."""
    engine: DatabaseEngineType
    max_connections: int = 100
    active_connections: int = 0
    timeout_ms: float = 500.0
    failure_count: int = 0
    success_count: int = 0
    circuit_state: CircuitState = CircuitState.CLOSED
    circuit_opened_at: Optional[float] = None
    circuit_recovery_timeout_sec: float = 30.0

    def can_acquire(self, tier: TenantTier) -> bool:
        """Determines if a connection lease can be granted given load and priority."""
        now = time.time()
        if self.circuit_state == CircuitState.OPEN:
            if self.circuit_opened_at and (now - self.circuit_opened_at) >= self.circuit_recovery_timeout_sec:
                self.circuit_state = CircuitState.HALF_OPEN
            else:
                return False

        # Load shedding thresholds based on tenant tier
        utilization = self.active_connections / max(1, self.max_connections)
        if tier == TenantTier.STANDARD and utilization >= 0.80:
            return False
        if tier == TenantTier.GROWTH and utilization >= 0.95:
            return False
        if self.active_connections >= self.max_connections:
            return False

        return True

    def acquire(self, tier: TenantTier) -> bool:
        """Acquires a connection lease."""
        if not self.can_acquire(tier):
            return False
        self.active_connections += 1
        return True

    def release(self, success: bool = True) -> None:
        """Releases an active connection lease and records outcome metrics."""
        if self.active_connections > 0:
            self.active_connections -= 1

        if success:
            self.success_count += 1
            if self.circuit_state == CircuitState.HALF_OPEN:
                self.circuit_state = CircuitState.CLOSED
                self.failure_count = 0
        else:
            self.failure_count += 1
            if self.failure_count >= 5 and self.circuit_state != CircuitState.OPEN:
                self.circuit_state = CircuitState.OPEN
                self.circuit_opened_at = time.time()


class KafkaEventPublisher:
    """Deterministic partition-aligned Kafka publisher for CDC events."""

    def __init__(self, num_partitions: int = 16) -> None:
        self.num_partitions = num_partitions
        self.partition_queues: Dict[int, List[Dict[str, Any]]] = {
            i: [] for i in range(num_partitions)
        }
        self.published_count: int = 0

    def get_partition(self, tenant_id: str) -> int:
        """Computes deterministic partition index via stable hash of tenant ID."""
        digest = hashlib.sha256(tenant_id.encode("utf-8")).hexdigest()
        return int(digest, 16) % self.num_partitions

    def publish_cdc(self, event: CDCEvent) -> Tuple[int, int]:
        """Publishes CDC event to its assigned partition queue ensuring total order.
        
        Returns:
            Tuple of (partition_id, offset)
        """
        partition = self.get_partition(event.tenant_id)
        payload = event.to_kafka_payload()
        offset = len(self.partition_queues[partition])
        self.partition_queues[partition].append(payload)
        self.published_count += 1
        return partition, offset


class SchemaMigrationOrchestrator:
    """Manages zero-downtime online database schema migration workflows."""

    def __init__(self, migration_id: str, table_name: str, target_engine: DatabaseEngineType) -> None:
        self.migration_id = migration_id
        self.table_name = table_name
        self.target_engine = target_engine
        self.state: MigrationState = MigrationState.IDLE
        self.source_checksum: Optional[str] = None
        self.target_checksum: Optional[str] = None
        self.backfill_progress_percent: float = 0.0
        self.rollback_reason: Optional[str] = None

    def advance_state(self, new_state: MigrationState) -> bool:
        """Transitions migration state following valid lifecycle paths."""
        valid_transitions = {
            MigrationState.IDLE: [MigrationState.PRE_SYNC],
            MigrationState.PRE_SYNC: [MigrationState.DUAL_WRITE, MigrationState.ROLLED_BACK],
            MigrationState.DUAL_WRITE: [MigrationState.BACKFILL, MigrationState.ROLLED_BACK],
            MigrationState.BACKFILL: [MigrationState.VALIDATE_PARITY, MigrationState.ROLLED_BACK],
            MigrationState.VALIDATE_PARITY: [MigrationState.CUTOVER, MigrationState.ROLLED_BACK],
            MigrationState.CUTOVER: [MigrationState.COMPLETE, MigrationState.ROLLED_BACK],
            MigrationState.COMPLETE: [],
            MigrationState.ROLLED_BACK: [],
        }

        if new_state not in valid_transitions.get(self.state, []):
            logger.warning("Invalid migration state transition: %s -> %s", self.state, new_state)
            return False

        self.state = new_state
        return True

    def record_parity_checksums(self, source_hash: str, target_hash: str) -> bool:
        """Validates schema and data checksum parity between source and target."""
        self.source_checksum = source_hash
        self.target_checksum = target_hash
        return source_hash == target_hash

    def trigger_rollback(self, reason: str) -> None:
        """Executes emergency rollback to protect production availability."""
        self.state = MigrationState.ROLLED_BACK
        self.rollback_reason = reason
        logger.error("Migration %s aborted: %s", self.migration_id, reason)


class SquarespaceDatabaseOrchestrator:
    """Central orchestrator for Squarespace multi-engine database operations."""

    def __init__(self) -> None:
        self.pools: Dict[DatabaseEngineType, ConnectionPool] = {
            DatabaseEngineType.COCKROACHDB: ConnectionPool(engine=DatabaseEngineType.COCKROACHDB, max_connections=120),
            DatabaseEngineType.MONGODB: ConnectionPool(engine=DatabaseEngineType.MONGODB, max_connections=80),
            DatabaseEngineType.POSTGRESQL: ConnectionPool(engine=DatabaseEngineType.POSTGRESQL, max_connections=50),
        }
        self.kafka_publisher = KafkaEventPublisher(num_partitions=16)
        self.active_migrations: Dict[str, SchemaMigrationOrchestrator] = {}
        self.transaction_history: List[Dict[str, Any]] = []

    def execute_query(
        self,
        engine: DatabaseEngineType,
        tier: TenantTier,
        query_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Executes a database query with connection multiplexing and circuit protection."""
        pool = self.pools[engine]
        if not pool.acquire(tier):
            err_msg = (
                f"Connection rejected for engine {engine.value} under tier {tier.value}. "
                "Pool saturated or circuit open."
            )
            return {
                "success": False,
                "error": err_msg,
                "circuit_state": pool.circuit_state.value,
            }

        start_time = time.time()
        # Simulated execution logic
        latency_ms = (time.time() - start_time) * 1000.0
        pool.release(success=True)

        return {
            "success": True,
            "engine": engine.value,
            "latency_ms": latency_ms,
            "active_connections": pool.active_connections,
        }

    def execute_dual_write(
        self,
        tenant_id: str,
        table_name: str,
        record: Dict[str, Any],
        tier: TenantTier = TenantTier.STANDARD,
    ) -> Dict[str, Any]:
        """Coordinates atomic dual-write between document (MongoDB) and SQL (CockroachDB)."""
        mongo_pool = self.pools[DatabaseEngineType.MONGODB]
        crdb_pool = self.pools[DatabaseEngineType.COCKROACHDB]

        if not mongo_pool.acquire(tier):
            return {"success": False, "error": "MongoDB pool unavailable for dual-write"}
        if not crdb_pool.acquire(tier):
            mongo_pool.release(success=True)
            return {"success": False, "error": "CockroachDB pool unavailable for dual-write"}

        # Simulate dual-write success
        mongo_pool.release(success=True)
        crdb_pool.release(success=True)

        # Emit CDC event to Kafka
        cdc_event = CDCEvent(
            event_id=f"evt-{int(time.time()*1000)}",
            tenant_id=tenant_id,
            engine=DatabaseEngineType.COCKROACHDB,
            table_name=table_name,
            operation="INSERT",
            lsn_sequence=len(self.transaction_history) + 1,
            before_state=None,
            after_state=record,
            schema_version=2,
        )
        partition, offset = self.kafka_publisher.publish_cdc(cdc_event)

        tx_record = {
            "tenant_id": tenant_id,
            "table": table_name,
            "kafka_partition": partition,
            "kafka_offset": offset,
            "status": "COMMITTED",
        }
        self.transaction_history.append(tx_record)

        return {
            "success": True,
            "transaction": tx_record,
            "cdc_event": cdc_event.to_kafka_payload(),
        }

    def start_migration(
        self,
        migration_id: str,
        table_name: str,
        target_engine: DatabaseEngineType = DatabaseEngineType.COCKROACHDB,
    ) -> SchemaMigrationOrchestrator:
        """Initializes and registers a new zero-downtime schema migration."""
        migrator = SchemaMigrationOrchestrator(
            migration_id=migration_id,
            table_name=table_name,
            target_engine=target_engine,
        )
        self.active_migrations[migration_id] = migrator
        return migrator
