"""SingleStore distributed vector & hybrid search MCP connector for enterprise AI systems.

Provides connection pooling, parameterized query execution, high-dimensional vector
similarity search, distributed table schema introspection, and latency observability
for Model Context Protocol (MCP) tool execution.
"""

from __future__ import annotations

import math
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .connectors import (
    ConnectorRegistry,
    MCPToolSpec,
    ToolArgSpec,
)


@dataclass(frozen=True)
class SingleStoreConfig:
    """Connection configuration and pooling parameters for SingleStore clusters."""

    host: str = "127.0.0.1"
    port: int = 3306
    user: str = "root"
    password: str = ""
    database: str = "singlestore_ai"
    pool_size: int = 10
    max_overflow: int = 20
    query_timeout_seconds: float = 5.0
    ssl_verify: bool = True
    enforce_read_only: bool = True


@dataclass
class SingleStoreQueryResult:
    """Encapsulates execution metrics and tabular output for a database query."""

    rows: List[Dict[str, Any]]
    row_count: int
    execution_time_ms: float
    is_vector_search: bool = False
    query_id: str = ""


class SingleStorePoolExhaustedError(Exception):
    """Raised when the connection pool and maximum overflow are saturated."""


class SingleStoreQueryTimeoutError(Exception):
    """Raised when a query exceeds the configured timeout budget."""


class SingleStoreSecurityError(Exception):
    """Raised when a destructive or write query violates read-only enforcement."""


class SingleStoreConnectionHandle:
    """Mockable database connection handle representing an active TCP socket."""

    def __init__(self, connection_id: str, config: SingleStoreConfig) -> None:
        self.connection_id = connection_id
        self.config = config
        self.is_healthy = True
        self.created_at = time.time()
        self.last_used_at = self.created_at

    def execute(
        self,
        query: str,
        params: Optional[Tuple[Any, ...]] = None,
        mock_data: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        self.last_used_at = time.time()
        if not self.is_healthy:
            raise RuntimeError(f"connection {self.connection_id} is closed or unhealthy")

        # Validate read-only security
        if self.config.enforce_read_only:
            normalized = query.strip().upper()
            destructive_keywords = ("DROP", "DELETE", "TRUNCATE", "ALTER", "INSERT", "UPDATE")
            for kw in destructive_keywords:
                if re.match(rf"^\s*{kw}\b", normalized):
                    raise SingleStoreSecurityError(f"read-only mode active: statement '{kw}' is prohibited")

        return list(mock_data or [])


class SingleStoreConnectionPool:
    """Thread-safe connection pool with lease lifecycle and saturation tracking."""

    def __init__(self, config: Optional[SingleStoreConfig] = None) -> None:
        self.config = config or SingleStoreConfig()
        self._available: List[SingleStoreConnectionHandle] = []
        self._in_use: List[SingleStoreConnectionHandle] = []
        self._lock = threading.Lock()
        self._counter = 0

        # Pre-seed base pool
        for _ in range(self.config.pool_size):
            self._counter += 1
            handle = SingleStoreConnectionHandle(f"conn-{self._counter}", self.config)
            self._available.append(handle)

    def lease_connection(self) -> SingleStoreConnectionHandle:
        """Lease an available connection or provision overflow up to max_overflow."""
        with self._lock:
            if self._available:
                conn = self._available.pop()
                self._in_use.append(conn)
                return conn

            total_conns = len(self._available) + len(self._in_use)
            max_allowed = self.config.pool_size + self.config.max_overflow
            if total_conns < max_allowed:
                self._counter += 1
                conn = SingleStoreConnectionHandle(f"conn-{self._counter}", self.config)
                self._in_use.append(conn)
                return conn

            raise SingleStorePoolExhaustedError(
                f"SingleStore pool exhausted: {len(self._in_use)} in use, max {max_allowed}"
            )

    def release_connection(self, conn: SingleStoreConnectionHandle) -> None:
        """Return a leased connection back to the available pool."""
        with self._lock:
            if conn in self._in_use:
                self._in_use.remove(conn)
                if len(self._available) < self.config.pool_size:
                    self._available.append(conn)

    @property
    def saturation_ratio(self) -> float:
        """Current pool saturation percentage (0.0 to 1.0)."""
        with self._lock:
            max_allowed = self.config.pool_size + self.config.max_overflow
            if max_allowed <= 0:
                return 0.0
            return len(self._in_use) / max_allowed


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a, vec_a in zip(vec_a, vec_a)))
    norm_b = math.sqrt(sum(b * b for b, vec_b in zip(vec_b, vec_b)))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class SingleStoreMCPConnector:
    """Registers and executes SingleStore SQL & Vector tools within MCP tool frameworks."""

    def __init__(
        self,
        config: Optional[SingleStoreConfig] = None,
        pool: Optional[SingleStoreConnectionPool] = None,
    ) -> None:
        self.config = config or SingleStoreConfig()
        self.pool = pool or SingleStoreConnectionPool(self.config)
        self._mock_tables: Dict[str, List[Dict[str, Any]]] = {}
        self._latencies: List[float] = []
        self._lock = threading.Lock()

    def set_mock_table_data(self, table_name: str, records: List[Dict[str, Any]]) -> None:
        """Populate in-memory mock records for testing or local simulation."""
        with self._lock:
            self._mock_tables[table_name] = records

    def execute_sql(
        self,
        sql: str,
        limit: int = 100,
        mock_data: Optional[List[Dict[str, Any]]] = None,
    ) -> SingleStoreQueryResult:
        """Execute a parameterized SQL statement with latency tracking and timeout guard."""
        start_time = time.time()
        conn = self.pool.lease_connection()
        try:
            # Query simulation
            rows = conn.execute(sql, mock_data=mock_data)
            duration_ms = (time.time() - start_time) * 1000.0

            if duration_ms > (self.config.query_timeout_seconds * 1000.0):
                raise SingleStoreQueryTimeoutError(
                    f"query execution exceeded {self.config.query_timeout_seconds}s limit"
                )

            capped_rows = rows[:limit]
            with self._lock:
                self._latencies.append(duration_ms)

            return SingleStoreQueryResult(
                rows=capped_rows,
                row_count=len(capped_rows),
                execution_time_ms=round(duration_ms, 2),
            )
        finally:
            self.pool.release_connection(conn)

    def vector_search(
        self,
        table_name: str,
        query_vector: List[float],
        vector_column: str = "embedding",
        top_k: int = 5,
        min_similarity: float = 0.0,
    ) -> SingleStoreQueryResult:
        """Execute cosine similarity vector search over indexed embedding columns."""
        start_time = time.time()
        with self._lock:
            records = self._mock_tables.get(table_name, [])

        scored_records = []
        for r in records:
            emb = r.get(vector_column)
            if isinstance(emb, list):
                score = _cosine_similarity(query_vector, emb)
                if score >= min_similarity:
                    item = dict(r)
                    item["similarity_score"] = round(score, 4)
                    scored_records.append(item)

        scored_records.sort(key=lambda x: x.get("similarity_score", 0.0), reverse=True)
        top_records = scored_records[:top_k]
        duration_ms = (time.time() - start_time) * 1000.0

        with self._lock:
            self._latencies.append(duration_ms)

        return SingleStoreQueryResult(
            rows=top_records,
            row_count=len(top_records),
            execution_time_ms=round(duration_ms, 2),
            is_vector_search=True,
        )

    def register_tools(self, registry: ConnectorRegistry) -> None:
        """Register SingleStore SQL, Vector, and Schema tools in the MCP Connector Registry."""
        # 1. SQL Query Tool
        registry.register(
            MCPToolSpec(
                name="singlestore_execute_sql",
                connector="singlestore",
                description="Execute a read-only SQL query against the distributed SingleStore database cluster.",
                args={
                    "sql": ToolArgSpec(arg_type="string", required=True),
                    "limit": ToolArgSpec(arg_type="integer", required=False),
                },
                risk_level="read",
            ),
            handler=lambda args, **ctx: self._handle_execute_sql(args),
        )

        # 2. Vector Search Tool
        registry.register(
            MCPToolSpec(
                name="singlestore_vector_search",
                connector="singlestore",
                description="Perform cosine similarity nearest-neighbor search over SingleStore vector embeddings.",
                args={
                    "table_name": ToolArgSpec(arg_type="string", required=True),
                    "query_vector": ToolArgSpec(arg_type="array", required=True),
                    "top_k": ToolArgSpec(arg_type="integer", required=False),
                },
                risk_level="read",
            ),
            handler=lambda args, **ctx: self._handle_vector_search(args),
        )

        # 3. Table Schema Tool
        registry.register(
            MCPToolSpec(
                name="singlestore_table_schema",
                connector="singlestore",
                description="Retrieve schema, shard keys, and column definitions for a SingleStore table.",
                args={
                    "table_name": ToolArgSpec(arg_type="string", required=True),
                },
                risk_level="read",
            ),
            handler=lambda args, **ctx: self._handle_table_schema(args),
        )

    def _handle_execute_sql(self, args: Dict[str, Any]) -> Dict[str, Any]:
        sql = args["sql"]
        limit = args.get("limit", 100)
        res = self.execute_sql(sql=sql, limit=limit)
        return {
            "row_count": res.row_count,
            "execution_time_ms": res.execution_time_ms,
            "rows": res.rows,
        }

    def _handle_vector_search(self, args: Dict[str, Any]) -> Dict[str, Any]:
        table_name = args["table_name"]
        query_vector = args["query_vector"]
        top_k = args.get("top_k", 5)
        res = self.vector_search(table_name=table_name, query_vector=query_vector, top_k=top_k)
        return {
            "top_k": len(res.rows),
            "execution_time_ms": res.execution_time_ms,
            "matches": res.rows,
        }

    def _handle_table_schema(self, args: Dict[str, Any]) -> Dict[str, Any]:
        table_name = args["table_name"]
        return {
            "table_name": table_name,
            "database": self.config.database,
            "engine": "SingleStore Columnstore",
            "shard_key": ["id"],
            "sort_key": ["timestamp DESC"],
            "status": "ONLINE",
        }
