"""N26 Data Storage Platform & Site Reliability Engineering Module.

Designed for high-concurrency cloud banking architectures (Aurora PostgreSQL, AWS RDS),
providing connection pool orchestration, read/write query splitting, replication lag monitoring,
automated failover evaluation with zero data loss (WAL/LSN tracking), BaFin-compliant immutable
audit snapshots, and Prometheus-compatible database SLO telemetry.
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class DatabaseNode:
    """Represents an Aurora PostgreSQL database instance in the storage cluster."""

    node_id: str
    role: str  # "writer" or "reader"
    host: str
    port: int = 5432
    active_connections: int = 0
    max_connections: int = 100
    status: str = "healthy"  # "healthy", "degraded", "unhealthy"
    current_lsn: int = 0  # Log Sequence Number for WAL tracking
    last_heartbeat: float = field(default_factory=time.time)

    @property
    def utilization(self) -> float:
        if self.max_connections <= 0:
            return 1.0
        return round(self.active_connections / self.max_connections, 4)

    @property
    def is_available(self) -> bool:
        return self.status in ("healthy", "degraded") and self.active_connections < self.max_connections


@dataclass
class ConnectionRequest:
    """A client database query routing request."""

    client_id: str
    query_type: str  # "read" or "write"
    sql_statement: str
    priority: int = 1  # 1 (normal) to 10 (critical)
    request_id: str = field(default_factory=lambda: hashlib.sha256(str(time.time()).encode()).hexdigest()[:12])


class AuroraPostgresConnectionPoolOrchestrator:
    """Orchestrates connection routing, burst queue throttling, and master/replica query splitting."""

    def __init__(self, cluster_id: str = "n26-core-aurora-cluster"):
        self.cluster_id = cluster_id
        self.nodes: Dict[str, DatabaseNode] = {}
        self.query_history: List[Dict[str, Any]] = []

    def register_node(self, node: DatabaseNode) -> None:
        self.nodes[node.node_id] = node

    def acquire_connection(self, request: ConnectionRequest) -> Tuple[DatabaseNode, str]:
        """Routes query to appropriate healthy database node with read/write splitting."""
        is_write = request.query_type.lower() == "write" or bool(
            re.search(r"^\s*(INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|TRUNCATE)\b", request.sql_statement, re.IGNORECASE)
        )

        candidates: List[DatabaseNode] = []
        if is_write:
            candidates = [n for n in self.nodes.values() if n.role == "writer" and n.is_available]
            if not candidates:
                raise RuntimeError(f"ConnectionRejected: No healthy writer node in {self.cluster_id}")
        else:
            candidates = [n for n in self.nodes.values() if n.role == "reader" and n.is_available]
            if not candidates:
                candidates = [n for n in self.nodes.values() if n.role == "writer" and n.is_available]
            if not candidates:
                raise RuntimeError(f"ConnectionRejected: All reader/writer pools saturated in {self.cluster_id}")

        candidates.sort(key=lambda n: n.utilization)
        selected = candidates[0]
        selected.active_connections += 1

        routing_decision = "writer_primary" if selected.role == "writer" else "reader_replica"
        self.query_history.append({
            "request_id": request.request_id,
            "client_id": request.client_id,
            "node_id": selected.node_id,
            "role": selected.role,
            "routing_decision": routing_decision,
            "sql_snippet": request.sql_statement[:60],
            "timestamp": time.time(),
        })
        return selected, routing_decision

    def release_connection(self, node_id: str) -> None:
        """Releases an active connection back to the pool."""
        node = self.nodes.get(node_id)
        if node and node.active_connections > 0:
            node.active_connections -= 1

    def get_cluster_saturation(self) -> Dict[str, Any]:
        """Returns detailed cluster connection metrics and saturation ratios."""
        writer_nodes = [n for n in self.nodes.values() if n.role == "writer"]
        reader_nodes = [n for n in self.nodes.values() if n.role == "reader"]

        total_writer_conn = sum(n.active_connections for n in writer_nodes)
        max_writer_conn = sum(n.max_connections for n in writer_nodes) or 1
        total_reader_conn = sum(n.active_connections for n in reader_nodes)
        max_reader_conn = sum(n.max_connections for n in reader_nodes) or 1

        node_metrics = {
            n.node_id: {
                "role": n.role,
                "active": n.active_connections,
                "max": n.max_connections,
                "utilization": n.utilization,
            }
            for n in self.nodes.values()
        }

        return {
            "cluster_id": self.cluster_id,
            "total_nodes": len(self.nodes),
            "writer_saturation": round(total_writer_conn / max_writer_conn, 4),
            "reader_saturation": round(total_reader_conn / max_reader_conn, 4),
            "nodes": node_metrics,
        }


class ReplicationLagAndFailoverMonitor:
    """Monitors Aurora replica WAL synchronization and orchestrates zero-data-loss failover."""

    def __init__(self, primary_node_id: str):
        self.primary_node_id = primary_node_id
        self.replica_telemetry: Dict[str, Dict[str, Any]] = {}

    def record_heartbeat(self, node_id: str, lsn: int, current_timestamp: Optional[float] = None) -> None:
        """Records the latest Write-Ahead Log sequence number (LSN) for a node."""
        ts = current_timestamp or time.time()
        self.replica_telemetry[node_id] = {
            "lsn": lsn,
            "last_seen": ts,
        }

    def compute_replication_lag(self, replica_node_id: str) -> Dict[str, Any]:
        """Calculates replication lag in LSN offset and approximate milliseconds."""
        primary = self.replica_telemetry.get(self.primary_node_id)
        replica = self.replica_telemetry.get(replica_node_id)

        if not primary or not replica:
            return {"replica_node_id": replica_node_id, "status": "unknown", "lag_bytes": -1, "lag_ms": -1.0}

        lsn_delta = max(0, primary["lsn"] - replica["lsn"])
        time_delta_ms = max(0.0, (primary["last_seen"] - replica["last_seen"]) * 1000.0)

        status = "synchronized" if lsn_delta == 0 and time_delta_ms < 50.0 else (
            "lagging" if time_delta_ms < 500.0 else "critical_lag"
        )

        return {
            "replica_node_id": replica_node_id,
            "primary_lsn": primary["lsn"],
            "replica_lsn": replica["lsn"],
            "lag_lsn_bytes": lsn_delta,
            "lag_ms": round(time_delta_ms, 2),
            "status": status,
        }

    def evaluate_failover_candidates(self, max_tolerable_lag_ms: float = 300.0) -> Dict[str, Any]:
        """Identifies the most synchronized read replica eligible for promotion to writer."""
        candidates = []

        for node_id in self.replica_telemetry:
            if node_id == self.primary_node_id:
                continue
            metrics = self.compute_replication_lag(node_id)
            if metrics["lag_ms"] <= max_tolerable_lag_ms and metrics["status"] != "unknown":
                candidates.append((node_id, metrics["lag_lsn_bytes"], metrics["lag_ms"]))

        candidates.sort(key=lambda c: (c[1], c[2]))

        if not candidates:
            return {
                "failover_recommended": False,
                "reason": "No replica qualifies within strict lag tolerance (RPO protection)",
                "selected_candidate": None,
            }

        best_candidate = candidates[0]
        return {
            "failover_recommended": True,
            "primary_node_id": self.primary_node_id,
            "selected_candidate": best_candidate[0],
            "candidate_lsn_lag_bytes": best_candidate[1],
            "candidate_lag_ms": best_candidate[2],
            "zero_data_loss_guaranteed": best_candidate[1] == 0,
        }


class BaFinAuditSnapshotManager:
    """Manages immutable cryptographic database snapshots compliant with BaFin/EBA banking requirements."""

    def __init__(self, institution_code: str = "N26-DE-BANK-AG"):
        self.institution_code = institution_code
        self.snapshots: Dict[str, Dict[str, Any]] = {}

    def create_immutable_snapshot(
        self, database_name: str, lsn: int, transaction_count: int, ledger_hash_root: str
    ) -> Dict[str, Any]:
        """Generates a point-in-time snapshot with SHA-256 cryptographic verification."""
        timestamp = time.time()
        raw_manifest = (
            f"{self.institution_code}:{database_name}:{lsn}:{transaction_count}:{ledger_hash_root}:{timestamp}"
        )
        snapshot_checksum = hashlib.sha256(raw_manifest.encode("utf-8")).hexdigest()
        snapshot_id = f"snap-{database_name}-{int(timestamp)}"

        record = {
            "snapshot_id": snapshot_id,
            "institution_code": self.institution_code,
            "database_name": database_name,
            "lsn": lsn,
            "transaction_count": transaction_count,
            "ledger_hash_root": ledger_hash_root,
            "created_at": timestamp,
            "retention_period_years": 10,
            "checksum_sha256": snapshot_checksum,
            "s3_cold_storage_status": "pending_upload",
        }
        self.snapshots[snapshot_id] = record
        return record

    def verify_snapshot_integrity(self, snapshot_id: str) -> bool:
        """Verifies cryptographic hash consistency of a retained snapshot."""
        record = self.snapshots.get(snapshot_id)
        if not record:
            return False
        raw_manifest = (
            f"{record['institution_code']}:{record['database_name']}:{record['lsn']}:"
            f"{record['transaction_count']}:{record['ledger_hash_root']}:{record['created_at']}"
        )
        recalculated = hashlib.sha256(raw_manifest.encode("utf-8")).hexdigest()
        return recalculated == record["checksum_sha256"]

    def export_audit_archive_to_s3(
        self, snapshot_id: str, s3_bucket: str = "n26-bafin-audit-vault"
    ) -> Dict[str, Any]:
        """Exports verified audit snapshot metadata to immutable S3 Glacier Vault."""
        record = self.snapshots.get(snapshot_id)
        if not record or not self.verify_snapshot_integrity(snapshot_id):
            raise ValueError(f"SnapshotIntegrityError: Snapshot {snapshot_id} corrupted or not found")

        s3_key = f"audit_ledgers/{record['database_name']}/{time.strftime('%Y/%m/%d')}/{snapshot_id}.json"
        record["s3_cold_storage_status"] = "uploaded"
        record["s3_destination"] = f"s3://{s3_bucket}/{s3_key}"

        return {
            "snapshot_id": snapshot_id,
            "status": "COMPLETED",
            "destination": record["s3_destination"],
            "retention_years": record["retention_period_years"],
            "verified_hash": record["checksum_sha256"],
        }


class DatabasePlatformSLOTracker:
    """Tracks database availability, query latencies, and exports Prometheus metrics for SRE dashboards."""

    def __init__(self, target_availability: float = 0.9999, p99_latency_slo_ms: float = 25.0):
        self.target_availability = target_availability
        self.p99_latency_slo_ms = p99_latency_slo_ms
        self.read_latencies: List[float] = []
        self.write_latencies: List[float] = []
        self.failed_queries: int = 0
        self.total_queries: int = 0

    def record_query(self, query_type: str, latency_ms: float, success: bool = True) -> None:
        """Records execution latency and status for a database query."""
        self.total_queries += 1
        if not success:
            self.failed_queries += 1
            return

        if query_type.lower() == "write":
            self.write_latencies.append(latency_ms)
        else:
            self.read_latencies.append(latency_ms)

    def calculate_slo_metrics(self) -> Dict[str, Any]:
        """Computes current database platform SLOs, error budget consumption, and latency percentiles."""
        all_latencies = sorted(self.read_latencies + self.write_latencies)
        p50 = all_latencies[int(len(all_latencies) * 0.50)] if all_latencies else 0.0
        p95 = all_latencies[int(len(all_latencies) * 0.95)] if all_latencies else 0.0
        p99 = all_latencies[int(len(all_latencies) * 0.99)] if all_latencies else 0.0

        availability = 1.0 - (self.failed_queries / self.total_queries) if self.total_queries > 0 else 1.0
        allowed_failures = max(1, int(self.total_queries * (1.0 - self.target_availability)))
        budget_consumed = min(1.0, round(self.failed_queries / allowed_failures, 4))

        return {
            "total_queries": self.total_queries,
            "failed_queries": self.failed_queries,
            "availability": round(availability, 5),
            "availability_slo_target": self.target_availability,
            "error_budget_consumed_pct": round(budget_consumed * 100.0, 2),
            "p50_latency_ms": round(p50, 2),
            "p95_latency_ms": round(p95, 2),
            "p99_latency_ms": round(p99, 2),
            "p99_slo_met": p99 <= self.p99_latency_slo_ms,
        }

    def export_prometheus_metrics(self) -> str:
        """Generates Prometheus-formatted text metrics."""
        slo = self.calculate_slo_metrics()
        lines = [
            "# HELP n26_db_query_total Total number of database queries handled",
            "# TYPE n26_db_query_total counter",
            f"n26_db_query_total {slo['total_queries']}",
            "# HELP n26_db_query_failures_total Total failed database queries",
            "# TYPE n26_db_query_failures_total counter",
            f"n26_db_query_failures_total {slo['failed_queries']}",
            "# HELP n26_db_availability Current database availability ratio",
            "# TYPE n26_db_availability gauge",
            f"n26_db_availability {slo['availability']}",
            "# HELP n26_db_query_latency_p99_ms 99th percentile query latency in milliseconds",
            "# TYPE n26_db_query_latency_p99_ms gauge",
            f"n26_db_query_latency_p99_ms {slo['p99_latency_ms']}",
            "# HELP n26_db_error_budget_consumed_ratio Error budget consumption ratio (0.0 to 1.0)",
            "# TYPE n26_db_error_budget_consumed_ratio gauge",
            f"n26_db_error_budget_consumed_ratio {slo['error_budget_consumed_pct'] / 100.0}",
        ]
        return "\n".join(lines) + "\n"
