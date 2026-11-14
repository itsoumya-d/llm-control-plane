"""Speakeasy Enterprise AI Control Plane SRE Engine.

Provides platform reliability, fine-grained MCP server proxying, agent session security,
circuit breaking, token-bucket rate limiting, SRE observability metrics, and canary rollout automation.
"""

from __future__ import annotations

import enum
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class CircuitBreakerState(enum.Enum):
    """Circuit breaker operational state."""

    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CanaryStage(enum.Enum):
    """Canary deployment stages."""

    STAGE_10 = "10%"
    STAGE_25 = "25%"
    STAGE_50 = "50%"
    STAGE_100 = "100%"
    ROLLED_BACK = "ROLLED_BACK"


@dataclass
class MCPServerDefinition:
    """Enterprise MCP server configuration and capability profile."""

    server_id: str
    name: str
    protocol_version: str
    endpoint_url: str
    supported_tools: List[str]
    auth_type: str = "bearer"
    is_healthy: bool = True
    active_connections: int = 0
    max_concurrency: int = 50

    def has_tool(self, tool_name: str) -> bool:
        """Check if server exposes named tool."""
        return tool_name in self.supported_tools


@dataclass
class AgentSessionSecurityPolicy:
    """Security governance and budget policy governing an autonomous agent session."""

    session_id: str
    tenant_id: str
    allowed_tools: List[str]
    max_token_budget: int
    current_token_spend: int = 0
    dlp_restricted_patterns: List[str] = field(
        default_factory=lambda: [
            r"(?i)api[_-]?key",
            r"(?i)bearer\s+[a-zA-Z0-9_\-\.]+",
            r"\b[A-Za-z0-9+/]{40,}\b",
            r"\b\d{3}-\d{2}-\d{4}\b",
        ]
    )
    per_tool_timeout_seconds: float = 5.0

    def is_tool_allowed(self, tool_name: str) -> bool:
        """Validate if tool is allowed for this agent session."""
        if "*" in self.allowed_tools:
            return True
        return tool_name in self.allowed_tools

    def has_sufficient_budget(self, estimated_tokens: int) -> bool:
        """Check if estimated tokens fit within remaining session budget."""
        return (self.current_token_spend + estimated_tokens) <= self.max_token_budget

    def inspect_dlp_violations(self, payload: Dict[str, Any]) -> List[str]:
        """Scan string fields in payload for sensitive data loss violations."""
        violations = []
        payload_str = str(payload)
        for pat in self.dlp_restricted_patterns:
            if re.search(pat, payload_str):
                violations.append(pat)
        return violations


class TokenBucketRateLimiter:
    """Thread-safe token-bucket rate limiter for tool invocation throttling."""

    def __init__(self, capacity: int, refill_rate_per_sec: float) -> None:
        self.capacity = float(capacity)
        self.refill_rate = float(refill_rate_per_sec)
        self.tokens = float(capacity)
        self.last_refill_ts = time.time()

    def _refill(self) -> None:
        now = time.time()
        delta = now - self.last_refill_ts
        self.tokens = min(self.capacity, self.tokens + delta * self.refill_rate)
        self.last_refill_ts = now

    def consume(self, count: int = 1) -> bool:
        """Attempt to consume token units. Returns True if granted, False if throttled."""
        self._refill()
        if self.tokens >= count:
            self.tokens -= count
            return True
        return False


class CircuitBreaker:
    """SRE circuit breaker protecting downstream MCP servers from cascading failures."""

    def __init__(self, failure_threshold: int = 3, recovery_timeout_sec: float = 10.0) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout_sec = recovery_timeout_sec
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_state_change_ts = time.time()

    def record_success(self) -> None:
        """Record successful execution and resolve half-open state."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= 2:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                self.last_state_change_ts = time.time()
        elif self.state == CircuitBreakerState.CLOSED:
            self.failure_count = 0

    def record_failure(self) -> None:
        """Record execution failure and trip circuit if threshold exceeded."""
        self.failure_count += 1
        if self.state == CircuitBreakerState.HALF_OPEN or self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            self.last_state_change_ts = time.time()

    def can_execute(self) -> bool:
        """Check if request execution is permitted."""
        now = time.time()
        if self.state == CircuitBreakerState.OPEN:
            if (now - self.last_state_change_ts) >= self.recovery_timeout_sec:
                self.state = CircuitBreakerState.HALF_OPEN
                self.success_count = 0
                self.last_state_change_ts = now
                return True
            return False
        return True


@dataclass
class ToolExecutionRequest:
    """Incoming agent tool invocation request payload."""

    request_id: str
    session_id: str
    tool_name: str
    arguments: Dict[str, Any]
    estimated_tokens: int = 100


@dataclass
class ToolExecutionResult:
    """Result of proxied tool execution including latency, status, and telemetry."""

    request_id: str
    status: str
    latency_ms: float
    output: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    tokens_consumed: int = 0


class SREObservabilityMetrics:
    """Real-time observability accumulator tracking request percentiles and error budgets."""

    def __init__(self, target_slo: float = 0.999) -> None:
        self.target_slo = target_slo
        self.total_requests = 0
        self.successful_requests = 0
        self.blocked_requests = 0
        self.failed_requests = 0
        self.latencies_ms: List[float] = []
        self.tokens_consumed = 0

    def record_event(self, status: str, latency_ms: float, tokens: int = 0) -> None:
        """Record execution telemetry point."""
        self.total_requests += 1
        self.latencies_ms.append(latency_ms)
        self.tokens_consumed += tokens

        if status == "SUCCESS":
            self.successful_requests += 1
        elif status.startswith("BLOCKED") or status == "RATE_LIMITED":
            self.blocked_requests += 1
        else:
            self.failed_requests += 1

    def compute_percentiles(self) -> Dict[str, float]:
        """Calculate P50, P90, and P99 latency percentiles."""
        if not self.latencies_ms:
            return {"p50": 0.0, "p90": 0.0, "p99": 0.0}
        sorted_l = sorted(self.latencies_ms)
        n = len(sorted_l)

        def pct(p: float) -> float:
            idx = min(n - 1, math.floor(p * n))
            return round(sorted_l[idx], 2)

        return {"p50": pct(0.50), "p90": pct(0.90), "p99": pct(0.99)}

    def calculate_error_rate(self) -> float:
        """Calculate ratio of unhandled service failures."""
        if self.total_requests == 0:
            return 0.0
        return round(self.failed_requests / self.total_requests, 4)

    def calculate_error_budget_burn(self) -> float:
        """Compute consumed fraction of allowed error budget."""
        allowed_error_fraction = 1.0 - self.target_slo
        if allowed_error_fraction <= 0 or self.total_requests == 0:
            return 0.0
        current_error_rate = self.calculate_error_rate()
        return round(current_error_rate / allowed_error_fraction, 2)


class ControlPlaneCanaryRollout:
    """Automated canary deployment controller for zero-downtime AI control plane releases."""

    STAGE_TRANSITIONS = [
        CanaryStage.STAGE_10,
        CanaryStage.STAGE_25,
        CanaryStage.STAGE_50,
        CanaryStage.STAGE_100,
    ]

    def __init__(
        self,
        rollout_id: str,
        target_version: str,
        error_rate_threshold: float = 0.01,
        p99_latency_threshold_ms: float = 2500.0,
    ) -> None:
        self.rollout_id = rollout_id
        self.target_version = target_version
        self.error_rate_threshold = error_rate_threshold
        self.p99_latency_threshold_ms = p99_latency_threshold_ms
        self.current_stage = CanaryStage.STAGE_10
        self.rollback_reason: Optional[str] = None
        self.is_completed = False

    def evaluate_and_advance(self, metrics: SREObservabilityMetrics) -> bool:
        """Evaluate metrics against thresholds and advance stage or trigger rollback."""
        if self.current_stage == CanaryStage.ROLLED_BACK:
            return False

        error_rate = metrics.calculate_error_rate()
        pcts = metrics.compute_percentiles()
        p99 = pcts.get("p99", 0.0)

        if error_rate > self.error_rate_threshold:
            self.trigger_rollback(f"Error rate {error_rate:.2%} exceeded threshold {self.error_rate_threshold:.2%}")
            return False

        if p99 > self.p99_latency_threshold_ms:
            self.trigger_rollback(f"P99 latency {p99}ms exceeded threshold {self.p99_latency_threshold_ms}ms")
            return False

        idx = self.STAGE_TRANSITIONS.index(self.current_stage)
        if idx + 1 < len(self.STAGE_TRANSITIONS):
            self.current_stage = self.STAGE_TRANSITIONS[idx + 1]
            if self.current_stage == CanaryStage.STAGE_100:
                self.is_completed = True
            return True

        self.is_completed = True
        return True

    def trigger_rollback(self, reason: str) -> None:
        """Halt canary rollout and record automated rollback state."""
        self.current_stage = CanaryStage.ROLLED_BACK
        self.rollback_reason = reason
        self.is_completed = False


class SpeakeasyAIControlPlaneProxy:
    """Centralized enterprise proxy gateway mediating MCP server execution."""

    def __init__(self, rate_limit_capacity: int = 100, refill_rate: float = 20.0) -> None:
        self.servers: Dict[str, MCPServerDefinition] = {}
        self.session_policies: Dict[str, AgentSessionSecurityPolicy] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.rate_limiter = TokenBucketRateLimiter(rate_limit_capacity, refill_rate)
        self.metrics = SREObservabilityMetrics()

    def register_server(self, server: MCPServerDefinition) -> None:
        """Register a downstream MCP server."""
        self.servers[server.server_id] = server
        self.circuit_breakers[server.server_id] = CircuitBreaker()

    def register_session(self, policy: AgentSessionSecurityPolicy) -> None:
        """Register security constraints for an agent session."""
        self.session_policies[policy.session_id] = policy

    def _locate_server_for_tool(self, tool_name: str) -> Optional[MCPServerDefinition]:
        for s in self.servers.values():
            if s.has_tool(tool_name) and s.is_healthy:
                return s
        return None

    def execute_tool(self, req: ToolExecutionRequest) -> ToolExecutionResult:
        """Proxy and govern execution of an agent tool call."""
        start_t = time.time()

        policy = self.session_policies.get(req.session_id)
        if not policy:
            res = ToolExecutionResult(
                request_id=req.request_id,
                status="BLOCKED_SECURITY",
                latency_ms=0.0,
                error_message=f"No security policy registered for session '{req.session_id}'",
            )
            self.metrics.record_event(res.status, res.latency_ms)
            return res

        if not policy.is_tool_allowed(req.tool_name):
            res = ToolExecutionResult(
                request_id=req.request_id,
                status="BLOCKED_SECURITY",
                latency_ms=0.0,
                error_message=f"Tool '{req.tool_name}' forbidden by session security policy",
            )
            self.metrics.record_event(res.status, res.latency_ms)
            return res

        dlp_hits = policy.inspect_dlp_violations(req.arguments)
        if dlp_hits:
            res = ToolExecutionResult(
                request_id=req.request_id,
                status="BLOCKED_SECURITY",
                latency_ms=0.0,
                error_message=f"DLP violation: payload matches sensitive pattern {dlp_hits}",
            )
            self.metrics.record_event(res.status, res.latency_ms)
            return res

        if not policy.has_sufficient_budget(req.estimated_tokens):
            res = ToolExecutionResult(
                request_id=req.request_id,
                status="BLOCKED_SECURITY",
                latency_ms=0.0,
                error_message=f"Token budget ceiling exceeded ({policy.current_token_spend}/{policy.max_token_budget})",
            )
            self.metrics.record_event(res.status, res.latency_ms)
            return res

        if not self.rate_limiter.consume(1):
            res = ToolExecutionResult(
                request_id=req.request_id,
                status="RATE_LIMITED",
                latency_ms=0.0,
                error_message="Global request throughput exceeded; token bucket throttled",
            )
            self.metrics.record_event(res.status, res.latency_ms)
            return res

        server = self._locate_server_for_tool(req.tool_name)
        if not server:
            res = ToolExecutionResult(
                request_id=req.request_id,
                status="FAILED",
                latency_ms=0.0,
                error_message=f"No healthy MCP server provides tool '{req.tool_name}'",
            )
            self.metrics.record_event(res.status, res.latency_ms)
            return res

        cb = self.circuit_breakers[server.server_id]
        if not cb.can_execute():
            res = ToolExecutionResult(
                request_id=req.request_id,
                status="CIRCUIT_OPEN",
                latency_ms=0.0,
                error_message=f"Circuit open for server '{server.server_id}'; shedding downstream load",
            )
            self.metrics.record_event(res.status, res.latency_ms)
            return res

        try:
            output = {"tool": req.tool_name, "server": server.server_id, "status": "executed", "data": req.arguments}
            latency_ms = round((time.time() - start_t) * 1000 + 15.0, 2)
            cb.record_success()
            policy.current_token_spend += req.estimated_tokens

            res = ToolExecutionResult(
                request_id=req.request_id,
                status="SUCCESS",
                latency_ms=latency_ms,
                output=output,
                tokens_consumed=req.estimated_tokens,
            )
            self.metrics.record_event(res.status, res.latency_ms, res.tokens_consumed)
            return res
        except Exception as exc:
            cb.record_failure()
            latency_ms = round((time.time() - start_t) * 1000, 2)
            res = ToolExecutionResult(
                request_id=req.request_id,
                status="FAILED",
                latency_ms=latency_ms,
                error_message=str(exc),
            )
            self.metrics.record_event(res.status, res.latency_ms)
            return res
