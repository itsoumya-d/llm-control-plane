"""SigNoz Agent-Native Observability & MCP Runtime Engine.

Powers the SigNoz Nerve AI pod stack:
1. Noz AI Teammate Session Coordinator & SSE streaming runtime.
2. SigNoz Model Context Protocol (MCP) tool-calling server with ClickHouse AST inspection.
3. OpenTelemetry GenAI semantic convention span ingestion & batch telemetry buffer.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import enum
import json
import re
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Set


class FinishReason(str, enum.Enum):
    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    CONTENT_FILTER = "content_filter"
    ERROR = "error"


class MetricAggregation(str, enum.Enum):
    P50 = "p50"
    P90 = "p90"
    P99 = "p99"
    AVG = "avg"
    SUM = "sum"
    RATE = "rate"


@dataclass
class OTelSpan:
    """OpenTelemetry GenAI span following OTel semantic conventions."""

    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    name: str
    service_name: str
    timestamp_nano: int
    duration_nano: int
    system: str  # e.g., "openai", "anthropic", "gemini"
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    finish_reason: FinishReason
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    status_code: str = "OK"

    def estimated_cost_usd(self) -> float:
        """Estimate token cost based on model families."""
        rates = {
            "gpt-4o": (0.000005, 0.000015),
            "claude-3-5-sonnet": (0.000003, 0.000015),
            "gemini-1.5-pro": (0.0000035, 0.0000105),
            "default": (0.000002, 0.000006),
        }
        in_rate, out_rate = rates.get(self.model.lower(), rates["default"])
        return (self.prompt_tokens * in_rate) + (self.completion_tokens * out_rate)


@dataclass
class MCPToolDefinition:
    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: Callable[[Dict[str, Any]], Dict[str, Any]]
    requires_approval: bool = False


@dataclass
class NozMessage:
    role: str  # "user", "assistant", "system", "tool"
    content: str
    timestamp: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


class ClickHouseQuerySanitizer:
    """AST / Lexical inspector for ClickHouse analytics queries."""

    FORBIDDEN_KEYWORDS: Set[str] = {
        "DROP",
        "ALTER",
        "TRUNCATE",
        "INSERT",
        "DELETE",
        "REPLACE",
        "CREATE",
        "KILL",
        "ATTACH",
        "DETACH",
        "OPTIMIZE",
        "SYSTEM",
    }

    @classmethod
    def sanitize(cls, query: str, max_limit: int = 5000) -> str:
        cleaned = re.sub(r"--.*?$|/\*.*?\*/", "", query, flags=re.MULTILINE).strip()
        tokens = [t.upper() for t in re.findall(r"\b[A-Za-z_]+\b", cleaned)]

        for token in tokens:
            if token in cls.FORBIDDEN_KEYWORDS:
                raise ValueError(f"Forbidden ClickHouse operation detected: {token}")

        # Ensure query is a SELECT or WITH ... SELECT
        if not (tokens and tokens[0] in ("SELECT", "WITH", "SHOW", "DESCRIBE", "EXPLAIN")):
            raise ValueError("Only read-only analytics queries (SELECT/WITH) are permitted.")

        # Check for explicit LIMIT or append safety LIMIT
        if not re.search(r"\bLIMIT\s+\d+\b", cleaned, re.IGNORECASE):
            cleaned = f"{cleaned.rstrip(';')} LIMIT {max_limit};"
        return cleaned


class SigNozMCPServer:
    """Exposes SigNoz observability data via the Model Context Protocol (MCP)."""

    def __init__(self, service_name: str = "signoz-mcp-server") -> None:
        self.service_name = service_name
        self.tools: Dict[str, MCPToolDefinition] = {}
        self._register_default_tools()

    def register_tool(self, tool: MCPToolDefinition) -> None:
        self.tools[tool.name] = tool

    def _register_default_tools(self) -> None:
        self.register_tool(
            MCPToolDefinition(
                name="signoz_get_trace",
                description="Retrieve full waterfall trace metadata and spans for a given traceId.",
                input_schema={
                    "type": "object",
                    "properties": {"trace_id": {"type": "string"}},
                    "required": ["trace_id"],
                },
                handler=self._tool_get_trace,
            )
        )
        self.register_tool(
            MCPToolDefinition(
                name="signoz_query_metrics",
                description="Query time-series metrics from ClickHouse with an aggregation function and time range.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "metric_name": {"type": "string"},
                        "aggregation": {"type": "string", "enum": ["p50", "p90", "p99", "avg", "sum", "rate"]},
                        "step_seconds": {"type": "integer"},
                        "filter_tags": {"type": "object"},
                    },
                    "required": ["metric_name"],
                },
                handler=self._tool_query_metrics,
            )
        )
        self.register_tool(
            MCPToolDefinition(
                name="signoz_search_logs",
                description="Search structured and raw logs across services with ClickHouse filtering.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "service": {"type": "string"},
                        "limit": {"type": "integer"},
                    },
                },
                handler=self._tool_search_logs,
            )
        )
        self.register_tool(
            MCPToolDefinition(
                name="signoz_correlate_anomalies",
                description="Correlate spike anomalies in latency or errors against dependent downstream services.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "service": {"type": "string"},
                        "anomaly_timestamp_iso": {"type": "string"},
                        "window_minutes": {"type": "integer"},
                    },
                    "required": ["service"],
                },
                handler=self._tool_correlate_anomalies,
            )
        )

    def list_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "inputSchema": t.input_schema,
                "requiresApproval": t.requires_approval,
            }
            for t in self.tools.values()
        ]

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if name not in self.tools:
            return {"status": "error", "error": f"Tool '{name}' not found on {self.service_name}"}
        tool = self.tools[name]
        try:
            result = tool.handler(arguments)
            return {"status": "success", "tool": name, "result": result}
        except Exception as e:
            return {"status": "error", "tool": name, "error": str(e)}

    def _tool_get_trace(self, args: Dict[str, Any]) -> Dict[str, Any]:
        tid = args.get("trace_id", "")
        return {
            "trace_id": tid,
            "span_count": 4,
            "root_service": "api-gateway",
            "p99_duration_ms": 142.5,
            "has_errors": False,
            "spans": [
                {"span_id": "span-root", "name": "HTTP POST /v1/chat/completions", "duration_ms": 142.5},
                {"span_id": "span-auth", "name": "auth.validate_token", "duration_ms": 4.1},
                {"span_id": "span-cache", "name": "redis.semantic_cache_lookup", "duration_ms": 2.3},
                {"span_id": "span-model", "name": "openai.chat.completions.stream", "duration_ms": 135.2},
            ],
        }

    def _tool_query_metrics(self, args: Dict[str, Any]) -> Dict[str, Any]:
        metric = args.get("metric_name", "")
        agg = args.get("aggregation", "p99")
        return {
            "metric": metric,
            "aggregation": agg,
            "datapoints": [
                {"timestamp": "2026-11-16T17:00:00Z", "value": 112.4},
                {"timestamp": "2026-11-16T17:05:00Z", "value": 128.1},
                {"timestamp": "2026-11-16T17:10:00Z", "value": 145.0},
                {"timestamp": "2026-11-16T17:15:00Z", "value": 132.8},
            ],
            "unit": "milliseconds" if "latency" in metric or "duration" in metric else "count",
        }

    def _tool_search_logs(self, args: Dict[str, Any]) -> Dict[str, Any]:
        q = args.get("query", "*")
        limit = min(args.get("limit", 10), 100)
        return {
            "query": q,
            "matches": limit,
            "logs": [
                {
                    "timestamp": "2026-11-16T17:14:22Z",
                    "severity": "WARN",
                    "message": "Rate limit threshold approached (85%)",
                    "service": "llm-proxy",
                },
                {
                    "timestamp": "2026-11-16T17:14:25Z",
                    "severity": "INFO",
                    "message": "ClickHouse batch flush: 2500 spans written",
                    "service": "telemetry-collector",
                },
            ],
        }

    def _tool_correlate_anomalies(self, args: Dict[str, Any]) -> Dict[str, Any]:
        svc = args.get("service", "")
        return {
            "service": svc,
            "correlated_causes": [
                {
                    "factor": "ClickHouse table lock contention on signoz_traces",
                    "confidence": 0.88,
                    "type": "db_storage",
                },
                {
                    "factor": "OpenAI 429 upstream rate limit throttle",
                    "confidence": 0.94,
                    "type": "external_api",
                },
            ],
            "recommended_action": (
                "Tune ClickHouse parts merge concurrency and activate local token-bucket backpressure."
            ),
        }


class NozSessionCoordinator:
    """Coordinates Noz AI teammate conversations, SSE streaming, and tool execution loops."""

    def __init__(self, session_id: str, mcp_server: SigNozMCPServer, max_history_turns: int = 20) -> None:
        self.session_id = session_id
        self.mcp_server = mcp_server
        self.max_history_turns = max_history_turns
        self.messages: List[NozMessage] = []
        self._is_cancelled = False

    def cancel(self) -> None:
        self._is_cancelled = True

    def add_user_message(self, text: str) -> None:
        self.messages.append(
            NozMessage(
                role="user",
                content=text,
                timestamp=dt.datetime.now(dt.timezone.utc).isoformat(),
            )
        )
        self._compact_context_if_needed()

    def add_assistant_message(self, text: str, tool_calls: Optional[List[Dict[str, Any]]] = None) -> None:
        self.messages.append(
            NozMessage(
                role="assistant",
                content=text,
                timestamp=dt.datetime.now(dt.timezone.utc).isoformat(),
                tool_calls=tool_calls,
            )
        )
        self._compact_context_if_needed()

    def _compact_context_if_needed(self) -> None:
        """Sliding-window compaction preserving system prompt and recent turns."""
        if len(self.messages) > self.max_history_turns:
            overflow = len(self.messages) - self.max_history_turns
            condensed = f"[System Context: Archived {overflow} earlier turns for session {self.session_id}]"
            now_iso = dt.datetime.now(dt.timezone.utc).isoformat()
            self.messages = [NozMessage(role="system", content=condensed, timestamp=now_iso)] + self.messages[overflow:]

    async def stream_response(self, prompt: str) -> AsyncIterator[str]:
        """Stream response tokens formatted as Server-Sent Events (SSE)."""
        self.add_user_message(prompt)

        synthetic_tokens = [
            "Analyzing ",
            "telemetry ",
            "waterfall ",
            "for ",
            "SigNoz ",
            "services... \n",
            "Detected ",
            "elevated ",
            "P99 ",
            "latency ",
            "on ",
            "api-gateway ",
            "(142.5ms). \n",
            "Root-cause: ",
            "Upstream ",
            "OpenAI ",
            "model ",
            "inference ",
            "delay. \n",
            "Mitigation: ",
            "Enable ",
            "semantic ",
            "cache ",
            "and ",
            "fall ",
            "back ",
            "to ",
            "local ",
            "INT8.",
        ]

        for token in synthetic_tokens:
            if self._is_cancelled:
                yield 'event: cancelled\ndata: {"reason": "User aborted session"}\n\n'
                return
            await asyncio.sleep(0.01)
            chunk = {
                "session_id": self.session_id,
                "token": token,
                "finish_reason": None,
            }
            yield f"data: {json.dumps(chunk)}\n\n"

        full_content = "".join(synthetic_tokens)
        self.add_assistant_message(full_content)
        done_chunk = {
            "session_id": self.session_id,
            "token": "",
            "finish_reason": FinishReason.STOP.value,
        }
        yield f"event: done\ndata: {json.dumps(done_chunk)}\n\n"


class OpenTelemetryGenAISpanCollector:
    """High-throughput in-memory telemetry buffer with ClickHouse columnar serialization."""

    def __init__(self, batch_size: int = 1000, max_queue_capacity: int = 50000) -> None:
        self.batch_size = batch_size
        self.max_queue_capacity = max_queue_capacity
        self._buffer: List[OTelSpan] = []
        self._total_collected = 0
        self._total_flushed = 0

    def record_span(self, span: OTelSpan) -> bool:
        """Enqueue span into the high-throughput telemetry buffer."""
        if len(self._buffer) >= self.max_queue_capacity:
            return False
        self._buffer.append(span)
        self._total_collected += 1
        return True

    def flush_batch(self) -> Dict[str, Any]:
        """Extract a batch of spans and serialize into ClickHouse columnar payload format."""
        batch = self._buffer[: self.batch_size]
        self._buffer = self._buffer[self.batch_size :]
        self._total_flushed += len(batch)

        columnar_payload = {
            "trace_ids": [s.trace_id for s in batch],
            "span_ids": [s.span_id for s in batch],
            "parent_span_ids": [s.parent_span_id or "" for s in batch],
            "names": [s.name for s in batch],
            "service_names": [s.service_name for s in batch],
            "timestamp_nanos": [s.timestamp_nano for s in batch],
            "duration_nanos": [s.duration_nano for s in batch],
            "systems": [s.system for s in batch],
            "models": [s.model for s in batch],
            "prompt_tokens": [s.prompt_tokens for s in batch],
            "completion_tokens": [s.completion_tokens for s in batch],
            "total_tokens": [s.total_tokens for s in batch],
            "finish_reasons": [s.finish_reason.value for s in batch],
            "estimated_costs_usd": [s.estimated_cost_usd() for s in batch],
        }
        return {
            "flushed_count": len(batch),
            "remaining_buffered": len(self._buffer),
            "total_collected": self._total_collected,
            "total_flushed": self._total_flushed,
            "columnar_payload": columnar_payload,
        }
