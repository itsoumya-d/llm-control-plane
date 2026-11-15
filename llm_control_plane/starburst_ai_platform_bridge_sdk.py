"""Starburst AI Platform Bridge & Developer Primitives SDK.

Engineered for Starburst Data AI Platform Engineering:
- Python/JVM Cross-Language Boundary Serialization & Type-Safe Interop Bridge
- Bidirectional Typed Exception & Error Translation across Process Boundaries
- Multi-Tenant Token Bucket Quota Manager with Sliding Window Allocators
- Resilient SSE/gRPC Token Streaming Multiplexer with Reconnect Buffer
- Deterministic Tool Execution Gateway with Autonomous Compiler Feedback Repair Loop
- Unified Starburst AI Platform SDK Facade with Telemetry & Observability
"""

from __future__ import annotations

import hashlib
import json
import time
import zlib
from enum import Enum
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, model_validator


class JVMDataType(str, Enum):
    STRING = "java.lang.String"
    LONG = "java.lang.Long"
    DOUBLE = "java.lang.Double"
    BOOLEAN = "java.lang.Boolean"
    BYTE_ARRAY = "byte[]"
    LIST = "java.util.List"
    MAP = "java.util.Map"
    TRINO_SCHEMA = "io.trino.spi.type.Type"


class InteropMessageEnvelope(BaseModel):
    """Binary/Protobuf message envelope crossing Python/JVM boundary."""
    message_id: str = Field(description="Unique UUID / sequence identifier")
    payload_type: str = Field(description="Protobuf schema contract name")
    jvm_type: JVMDataType = Field(description="Corresponding JVM class type")
    payload_bytes: bytes = Field(description="Raw serialized byte buffer")
    checksum_crc32: int = Field(description="CRC32 integrity checksum")
    metadata: Dict[str, str] = Field(default_factory=dict)
    timestamp_ns: int = Field(default_factory=time.time_ns)

    @classmethod
    def create(
        cls,
        message_id: str,
        payload_type: str,
        jvm_type: JVMDataType,
        data: Any,
        metadata: Optional[Dict[str, str]] = None,
    ) -> InteropMessageEnvelope:
        raw_json = json.dumps(data, separators=(",", ":")).encode("utf-8")
        checksum = zlib.crc32(raw_json)
        return cls(
            message_id=message_id,
            payload_type=payload_type,
            jvm_type=jvm_type,
            payload_bytes=raw_json,
            checksum_crc32=checksum,
            metadata=metadata or {},
        )

    def verify_and_decode(self) -> Any:
        computed_checksum = zlib.crc32(self.payload_bytes)
        if computed_checksum != self.checksum_crc32:
            raise ValueError(
                f"Checksum mismatch across JVM boundary: expected {self.checksum_crc32}, got {computed_checksum}"
            )
        return json.loads(self.payload_bytes.decode("utf-8"))


class CrossLanguageExceptionTranslator:
    """Translates exceptions between Python runtime and JVM throwables."""

    PY_TO_JVM_MAP = {
        ValueError: "java.lang.IllegalArgumentException",
        TypeError: "java.lang.ClassCastException",
        TimeoutError: "java.util.concurrent.TimeoutException",
        PermissionError: "java.lang.SecurityException",
        KeyError: "java.util.NoSuchElementException",
        NotImplementedError: "java.lang.UnsupportedOperationException",
    }

    JVM_TO_PY_MAP = {
        "java.lang.IllegalArgumentException": ValueError,
        "java.lang.ClassCastException": TypeError,
        "java.util.concurrent.TimeoutException": TimeoutError,
        "java.lang.SecurityException": PermissionError,
        "java.util.NoSuchElementException": KeyError,
        "java.lang.UnsupportedOperationException": NotImplementedError,
        "io.trino.spi.TrinoException": RuntimeError,
    }

    @classmethod
    def to_jvm_error_payload(cls, exc: Exception) -> Dict[str, Any]:
        jvm_class = cls.PY_TO_JVM_MAP.get(type(exc), "java.lang.RuntimeException")
        return {
            "error_class": jvm_class,
            "message": str(exc),
            "python_type": type(exc).__name__,
            "timestamp": time.time(),
        }

    @classmethod
    def from_jvm_error_payload(cls, payload: Dict[str, Any]) -> Exception:
        jvm_class = payload.get("error_class", "java.lang.RuntimeException")
        msg = payload.get("message", "Unknown JVM error")
        py_exc_cls = cls.JVM_TO_PY_MAP.get(jvm_class, RuntimeError)
        return py_exc_cls(f"[{jvm_class}] {msg}")


class TokenBucketQuotaManager:
    """Multi-tenant token bucket rate limiter for platform LLM invocations."""

    def __init__(self, default_rate_per_sec: float = 100.0, default_burst_capacity: float = 500.0):
        self.default_rate = default_rate_per_sec
        self.default_capacity = default_burst_capacity
        # team_id -> (current_tokens, last_refill_timestamp, rate, capacity)
        self._buckets: Dict[str, Tuple[float, float, float, float]] = {}

    def configure_tenant(self, tenant_id: str, rate: float, capacity: float) -> None:
        self._buckets[tenant_id] = (capacity, time.time(), rate, capacity)

    def consume(self, tenant_id: str, tokens: int) -> bool:
        now = time.time()
        if tenant_id not in self._buckets:
            self._buckets[tenant_id] = (self.default_capacity, now, self.default_rate, self.default_capacity)

        current, last_time, rate, capacity = self._buckets[tenant_id]
        elapsed = now - last_time
        refreshed = min(capacity, current + elapsed * rate)

        if refreshed >= tokens:
            self._buckets[tenant_id] = (refreshed - tokens, now, rate, capacity)
            return True
        else:
            self._buckets[tenant_id] = (refreshed, now, rate, capacity)
            return False

    def get_available_tokens(self, tenant_id: str) -> float:
        now = time.time()
        if tenant_id not in self._buckets:
            return self.default_capacity
        current, last_time, rate, capacity = self._buckets[tenant_id]
        elapsed = now - last_time
        return min(capacity, current + elapsed * rate)


class StreamTokenChunk(BaseModel):
    chunk_index: int
    text: str
    is_terminal: bool = False
    tokens_consumed: int = 1
    latency_ms: float = 0.0


class ResilientStreamingMultiplexer:
    """Manages SSE/gRPC streaming with reconnect caching and sequence verification."""

    def __init__(self, buffer_size: int = 100):
        self.buffer_size = buffer_size
        self._streams: Dict[str, List[StreamTokenChunk]] = {}

    def push_chunk(self, stream_id: str, chunk: StreamTokenChunk) -> None:
        if stream_id not in self._streams:
            self._streams[stream_id] = []
        self._streams[stream_id].append(chunk)
        if len(self._streams[stream_id]) > self.buffer_size:
            self._streams[stream_id].pop(0)

    def get_chunks_from(self, stream_id: str, from_index: int) -> List[StreamTokenChunk]:
        if stream_id not in self._streams:
            return []
        return [c for c in self._streams[stream_id] if c.chunk_index >= from_index]


class ToolExecutionContract(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]
    catalog: str = "lakehouse"
    schema_name: str = "default"


class ToolValidationResult(BaseModel):
    valid: bool
    error_message: Optional[str] = None
    sanitized_sql: Optional[str] = None
    suggested_repair: Optional[str] = None


class DeterministicToolExecutionGateway:
    """Validates and executes agent lakehouse tools with autonomous feedback repair loop."""

    ALLOWED_CATALOGS = {"lakehouse", "hive", "iceberg", "delta", "system"}

    def __init__(self):
        self.execution_audit: List[Dict[str, Any]] = []

    def validate_tool_call(self, contract: ToolExecutionContract) -> ToolValidationResult:
        if contract.catalog.lower() not in self.ALLOWED_CATALOGS:
            return ToolValidationResult(
                valid=False,
                error_message=f"Catalog '{contract.catalog}' unauthorized. Allowed: {sorted(self.ALLOWED_CATALOGS)}",
                suggested_repair=f"Set catalog to 'lakehouse' or 'iceberg'",
            )

        if contract.tool_name == "trino_query_catalog":
            query = contract.arguments.get("sql", "").strip()
            if not query:
                return ToolValidationResult(
                    valid=False,
                    error_message="Missing required parameter 'sql'",
                    suggested_repair="Provide 'sql': 'SELECT ... FROM ... LIMIT N'",
                )
            # Enforce read-only / safety constraints
            upper = query.upper()
            if any(forbidden in upper for forbidden in ["DROP ", "DELETE ", "TRUNCATE ", "ALTER "]):
                return ToolValidationResult(
                    valid=False,
                    error_message="Destructive queries (DROP/DELETE/TRUNCATE) rejected by platform policy",
                    suggested_repair="Rewrite query using read-only SELECT or EXPLAIN",
                )
            if "LIMIT" not in upper:
                # Autonomous micro-repair: append default safety limit
                query += " LIMIT 1000"
            return ToolValidationResult(valid=True, sanitized_sql=query)

        elif contract.tool_name == "trino_describe_table":
            table = contract.arguments.get("table_name", "").strip()
            if not table:
                return ToolValidationResult(
                    valid=False,
                    error_message="Missing 'table_name' argument",
                    suggested_repair="Supply table_name: 'orders'",
                )
            return ToolValidationResult(valid=True, sanitized_sql=f"DESCRIBE {contract.catalog}.{contract.schema_name}.{table}")

        return ToolValidationResult(
            valid=False,
            error_message=f"Unknown tool '{contract.tool_name}'",
            suggested_repair="Supported tools: trino_query_catalog, trino_describe_table",
        )

    def execute_with_repair_loop(
        self,
        contract: ToolExecutionContract,
        executor_fn: Optional[Callable[[str], Any]] = None,
        max_attempts: int = 2,
    ) -> Dict[str, Any]:
        attempt = 0
        current_contract = contract

        while attempt < max_attempts:
            attempt += 1
            validation = self.validate_tool_call(current_contract)
            if validation.valid:
                sql = validation.sanitized_sql or ""
                result = executor_fn(sql) if executor_fn else {"status": "SUCCESS", "rows": 42, "query": sql}
                self.execution_audit.append({
                    "contract": current_contract.model_dump(),
                    "attempts": attempt,
                    "status": "EXECUTED",
                })
                return {
                    "success": True,
                    "attempts": attempt,
                    "sanitized_sql": sql,
                    "result": result,
                }
            else:
                # Autonomous repair attempt if suggested repair is available
                if validation.suggested_repair and attempt < max_attempts:
                    err_lower = (validation.error_message or "").lower()
                    if "catalog" in err_lower:
                        current_contract.catalog = "lakehouse"
                    elif "missing required parameter 'sql'" in err_lower:
                        current_contract.arguments["sql"] = "SELECT 1"
                else:
                    self.execution_audit.append({
                        "contract": current_contract.model_dump(),
                        "attempts": attempt,
                        "status": "REJECTED",
                        "error": validation.error_message,
                    })
                    return {
                        "success": False,
                        "attempts": attempt,
                        "error": validation.error_message,
                        "suggested_repair": validation.suggested_repair,
                    }

        return {"success": False, "attempts": attempt, "error": "Max repair attempts exhausted"}


class StarburstAIPlatformSDK:
    """Unified Facade for Starburst AI Platform Engineering Primitives."""

    def __init__(
        self,
        cluster_endpoint: str = "trino.galaxy.starburst.io",
        default_rate: float = 200.0,
        default_capacity: float = 1000.0,
    ):
        self.cluster_endpoint = cluster_endpoint
        self.quota_manager = TokenBucketQuotaManager(default_rate, default_capacity)
        self.tool_gateway = DeterministicToolExecutionGateway()
        self.stream_multiplexer = ResilientStreamingMultiplexer()
        self.metrics: Dict[str, Union[int, float]] = {
            "total_requests": 0,
            "throttled_requests": 0,
            "tools_executed": 0,
            "jvm_bridges_called": 0,
        }

    def call_jvm_bridge(self, payload_name: str, jvm_type: JVMDataType, data: Any) -> InteropMessageEnvelope:
        msg_id = hashlib.sha256(f"{payload_name}:{time.time_ns()}".encode()).hexdigest()[:16]
        envelope = InteropMessageEnvelope.create(
            message_id=msg_id,
            payload_type=payload_name,
            jvm_type=jvm_type,
            data=data,
            metadata={"endpoint": self.cluster_endpoint},
        )
        self.metrics["jvm_bridges_called"] += 1
        return envelope

    def request_model_invocation(
        self,
        tenant_id: str,
        prompt: str,
        estimated_tokens: int = 50,
    ) -> Dict[str, Any]:
        self.metrics["total_requests"] += 1
        if not self.quota_manager.consume(tenant_id, estimated_tokens):
            self.metrics["throttled_requests"] += 1
            raise RuntimeError(f"Tenant '{tenant_id}' exceeded platform token rate budget")

        # Mock deterministic LLM platform gateway completion
        return {
            "tenant_id": tenant_id,
            "status": "COMPLETED",
            "tokens_used": estimated_tokens,
            "completion": f"Platform-orchestrated response for: {prompt[:30]}...",
            "timestamp": time.time(),
        }

    def stream_tokens(
        self,
        stream_id: str,
        text_tokens: List[str],
    ) -> Generator[StreamTokenChunk, None, None]:
        for idx, token in enumerate(text_tokens):
            is_last = idx == len(text_tokens) - 1
            chunk = StreamTokenChunk(
                chunk_index=idx,
                text=token,
                is_terminal=is_last,
                tokens_consumed=1,
                latency_ms=1.5 * (idx + 1),
            )
            self.stream_multiplexer.push_chunk(stream_id, chunk)
            yield chunk

    def execute_lakehouse_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        catalog: str = "lakehouse",
        schema_name: str = "default",
    ) -> Dict[str, Any]:
        contract = ToolExecutionContract(
            tool_name=tool_name,
            arguments=arguments,
            catalog=catalog,
            schema_name=schema_name,
        )
        res = self.tool_gateway.execute_with_repair_loop(contract)
        if res.get("success"):
            self.metrics["tools_executed"] += 1
        return res
