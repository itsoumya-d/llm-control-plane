"""
CrateDB Ecosystem Driver Adapter & Telemetry Store.

Provides PostgreSQL Wire Protocol connection pooling, CrateDB-specific SQL dialect translation,
high-throughput vector/time-series batch ingestion, and Grafana/OpenTelemetry observability metrics
for LLM control plane event streams.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger("cratedb_ecosystem_adapter")


@dataclasses.dataclass
class CrateDBNode:
    """Represents an individual CrateDB cluster node."""
    host: str
    pgwire_port: int = 5432
    http_port: int = 4200
    is_healthy: bool = True
    last_error: Optional[str] = None
    last_latency_ms: float = 0.0


@dataclasses.dataclass
class CrateDBDriverConfig:
    """Configuration for CrateDB ecosystem connection and dialect."""
    nodes: List[CrateDBNode] = dataclasses.field(default_factory=lambda: [CrateDBNode("localhost")])
    schema: str = "doc"
    username: str = "crate"
    password: Optional[str] = None
    ssl: bool = False
    connection_timeout_sec: float = 5.0
    query_timeout_sec: float = 30.0
    max_batch_size: int = 1000
    flush_interval_sec: float = 2.0
    max_retries: int = 3
    retry_backoff_sec: float = 0.5


class CrateDBSQLDialect:
    """
    Translates and normalizes SQL queries for CrateDB's distributed architecture,
    supporting vector KNN search, fulltext matching, and time-series date partitioning.
    """

    @staticmethod
    def build_create_table_statement(
        table_name: str,
        columns: Dict[str, str],
        primary_key: Optional[Sequence[str]] = None,
        partition_by: Optional[str] = None,
        shards: int = 4,
        replicas: str = "0-1"
    ) -> str:
        """Generates a CREATE TABLE DDL with CrateDB partitioning and routing parameters."""
        col_defs = [f'"{col}" {dtype}' for col, dtype in columns.items()]
        if primary_key:
            pk_cols = ", ".join(f'"{pk}"' for pk in primary_key)
            col_defs.append(f"PRIMARY KEY ({pk_cols})")

        sql = f'CREATE TABLE IF NOT EXISTS "{table_name}" (\n  ' + ",\n  ".join(col_defs) + "\n)"
        if partition_by:
            sql += f' PARTITIONED BY ("{partition_by}")'
        sql += f" CLUSTERED INTO {shards} SHARDS"
        sql += f" WITH (number_of_replicas = '{replicas}')"
        return sql

    @staticmethod
    def rewrite_knn_vector_query(
        table_name: str,
        vector_column: str,
        query_vector: Sequence[float],
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        fields: Optional[Sequence[str]] = None
    ) -> Tuple[str, List[Any]]:
        """
        Builds a CrateDB vector similarity query using the KNN operator.
        CrateDB syntax: _score or knn_match(vector_col, target_vector, k).
        """
        select_fields = ", ".join(f'"{f}"' for f in fields) if fields else "*"
        params: List[Any] = [list(query_vector), k]

        where_clauses = [f"knn_match({vector_column}, $1, $2)"]
        param_idx = 3

        if filters:
            for col, val in filters.items():
                where_clauses.append(f'"{col}" = ${param_idx}')
                params.append(val)
                param_idx += 1

        sql = (
            f'SELECT {select_fields}, _score AS similarity '
            f'FROM "{table_name}" '
            f'WHERE {" AND ".join(where_clauses)} '
            f'ORDER BY _score DESC '
            f'LIMIT {k}'
        )
        return sql, params

    @staticmethod
    def rewrite_upsert_query(
        table_name: str,
        conflict_keys: Sequence[str],
        update_columns: Sequence[str],
        columns: Sequence[str]
    ) -> str:
        """
        Generates an idempotent upsert query compatible with CrateDB (PostgreSQL ON CONFLICT DO UPDATE).
        """
        col_list = ", ".join(f'"{c}"' for c in columns)
        val_placeholders = ", ".join(f"${i+1}" for i in range(len(columns)))
        conflict_target = ", ".join(f'"{k}"' for k in conflict_keys)

        update_set = ", ".join(f'"{c}" = excluded."{c}"' for c in update_columns)

        return (
            f'INSERT INTO "{table_name}" ({col_list}) '
            f'VALUES ({val_placeholders}) '
            f'ON CONFLICT ({conflict_target}) DO UPDATE SET {update_set}'
        )

    @staticmethod
    def build_time_series_aggregation_query(
        table_name: str,
        timestamp_column: str,
        metric_column: str,
        interval: str = "minute",
        start_time_iso: Optional[str] = None,
        end_time_iso: Optional[str] = None
    ) -> Tuple[str, List[Any]]:
        """Generates date_trunc time-series aggregation query with percentiles and count."""
        where_clauses = []
        params: List[Any] = []
        p_idx = 1

        if start_time_iso:
            where_clauses.append(f'"{timestamp_column}" >= ${p_idx}')
            params.append(start_time_iso)
            p_idx += 1
        if end_time_iso:
            where_clauses.append(f'"{timestamp_column}" <= ${p_idx}')
            params.append(end_time_iso)
            p_idx += 1

        where_sql = f'WHERE {" AND ".join(where_clauses)} ' if where_clauses else ""

        sql = (
            f'SELECT date_trunc(\'{interval}\', "{timestamp_column}") AS time_bucket, '
            f'COUNT(*) AS event_count, '
            f'AVG("{metric_column}") AS avg_metric, '
            f'MAX("{metric_column}") AS max_metric, '
            f'MIN("{metric_column}") AS min_metric '
            f'FROM "{table_name}" '
            f'{where_sql}'
            f'GROUP BY 1 '
            f'ORDER BY 1 ASC'
        )
        return sql, params


class CrateDBConnectionPool:
    """Manages connections and failover across CrateDB nodes."""

    def __init__(self, config: CrateDBDriverConfig):
        self.config = config
        self._current_node_index = 0
        self._query_executor: Optional[Callable[[CrateDBNode, str, Sequence[Any]], List[Dict[str, Any]]]] = None

    def set_executor(self, executor: Callable[[CrateDBNode, str, Sequence[Any]], List[Dict[str, Any]]]) -> None:
        """Sets the underlying network/driver query execution hook."""
        self._query_executor = executor

    def get_healthy_node(self) -> CrateDBNode:
        """Selects a healthy node using round-robin with fallback."""
        num_nodes = len(self.config.nodes)
        for _ in range(num_nodes):
            node = self.config.nodes[self._current_node_index]
            self._current_node_index = (self._current_node_index + 1) % num_nodes
            if node.is_healthy:
                return node

        # If all marked unhealthy, attempt recovery with the primary node
        primary = self.config.nodes[0]
        primary.is_healthy = True
        return primary

    def execute_with_failover(self, sql: str, params: Sequence[Any] = ()) -> List[Dict[str, Any]]:
        """Executes a query with automatic retry and node failover."""
        attempts = 0
        last_exception: Optional[Exception] = None

        while attempts < self.config.max_retries:
            node = self.get_healthy_node()
            start_t = time.perf_counter()
            try:
                if self._query_executor:
                    result = self._query_executor(node, sql, params)
                else:
                    # Simulated execution for decoupled testing
                    result = [{"status": "simulated_ok", "node": node.host, "rows_affected": 1}]
                node.last_latency_ms = (time.perf_counter() - start_t) * 1000.0
                node.is_healthy = True
                node.last_error = None
                return result
            except Exception as ex:
                attempts += 1
                node.is_healthy = False
                node.last_error = str(ex)
                last_exception = ex
                logger.warning("Query failed on CrateDB node %s: %s (attempt %d/%d)",
                               node.host, ex, attempts, self.config.max_retries)
                time.sleep(self.config.retry_backoff_sec * attempts)

        err_msg = f"CrateDB query failed across nodes after {attempts} attempts: {last_exception}"
        raise RuntimeError(err_msg)


class CrateDBTelemetryStore:
    """
    High-throughput ingestion buffer for LLM observability telemetry, audit logs,
    and vector embeddings targeted for CrateDB time-series storage.
    """

    def __init__(self, config: CrateDBDriverConfig, pool: CrateDBConnectionPool):
        self.config = config
        self.pool = pool
        self._buffer: List[Dict[str, Any]] = []
        self._total_ingested = 0
        self._total_flushed = 0
        self._flush_errors = 0
        self._dead_letter_queue: List[Dict[str, Any]] = []

    def record_telemetry_event(
        self,
        request_id: str,
        provider: str,
        model: str,
        latency_ms: float,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        cost_usd: float,
        status_code: int = 200,
        embedding: Optional[Sequence[float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Appends a telemetry record to the in-memory batch buffer."""
        record = {
            "id": request_id,
            "timestamp": time.time(),
            "provider": provider,
            "model": model,
            "latency_ms": latency_ms,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost_usd": cost_usd,
            "status_code": status_code,
            "embedding": list(embedding) if embedding else None,
            "metadata": json.dumps(metadata or {}),
        }
        self._buffer.append(record)
        self._total_ingested += 1

        if len(self._buffer) >= self.config.max_batch_size:
            self.flush()

    def flush(self) -> int:
        """Flushes the buffered records to CrateDB."""
        if not self._buffer:
            return 0

        batch = self._buffer[:]
        self._buffer.clear()

        table = f"{self.config.schema}.llm_telemetry"
        cols = list(batch[0].keys())
        conflict_keys = ["id"]
        update_cols = [c for c in cols if c != "id"]

        sql = CrateDBSQLDialect.rewrite_upsert_query(table, conflict_keys, update_cols, cols)

        try:
            for item in batch:
                params = [item[c] for c in cols]
                self.pool.execute_with_failover(sql, params)
            self._total_flushed += len(batch)
            return len(batch)
        except Exception as ex:
            self._flush_errors += 1
            self._dead_letter_queue.extend(batch)
            logger.error("Failed to flush telemetry batch to CrateDB: %s", ex)
            return 0

    def get_metrics(self) -> Dict[str, Any]:
        """Returns internal store metrics."""
        return {
            "buffered_records": len(self._buffer),
            "total_ingested": self._total_ingested,
            "total_flushed": self._total_flushed,
            "flush_errors": self._flush_errors,
            "dead_letter_count": len(self._dead_letter_queue),
        }

    def export_grafana_metrics(self) -> str:
        """Exports Prometheus/Grafana readable metric format."""
        m = self.get_metrics()
        lines = [
            "# HELP cratedb_telemetry_buffered_count Currently buffered telemetry events",
            "# TYPE cratedb_telemetry_buffered_count gauge",
            f"cratedb_telemetry_buffered_count {m['buffered_records']}",
            "# HELP cratedb_telemetry_total_ingested Total telemetry events recorded",
            "# TYPE cratedb_telemetry_total_ingested counter",
            f"cratedb_telemetry_total_ingested {m['total_ingested']}",
            "# HELP cratedb_telemetry_total_flushed Total telemetry events successfully persisted in CrateDB",
            "# TYPE cratedb_telemetry_total_flushed counter",
            f"cratedb_telemetry_total_flushed {m['total_flushed']}",
            "# HELP cratedb_telemetry_flush_errors Total flush failure count",
            "# TYPE cratedb_telemetry_flush_errors counter",
            f"cratedb_telemetry_flush_errors {m['flush_errors']}",
        ]
        return "\n".join(lines) + "\n"
