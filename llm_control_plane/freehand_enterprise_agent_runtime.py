"""Freehand Enterprise Agent Execution Runtime & Policy Enforcement Engine.

Provides high-throughput agent execution runtimes, write-ahead logging (WAL) state
checkpointing, hierarchical policy enforcement layers, and replay evaluation harnesses
for autonomous enterprise AI workers operating across ERP, procurement, and financial systems.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class ActionRiskTier(str, Enum):
    """Hierarchical risk classification for autonomous enterprise agent actions."""
    TIER_1_AUTO_APPROVED = "tier_1_auto_approved"     # Read-only queries, status checks, catalog lookups
    TIER_2_BOUNDED_MUTATION = "tier_2_bounded_mutation" # Draft POs, invoice reconciliations under threshold
    TIER_3_HUMAN_CONSENSUS = "tier_3_human_consensus"   # Payments > threshold, bank changes, new vendors


@dataclass
class WALEvent:
    """Immutable Write-Ahead Log event record with cryptographic tamper-evident hash."""
    event_id: str
    sequence_num: int
    session_id: str
    event_type: str
    payload: Dict[str, Any]
    timestamp_ns: int
    prev_hash: str
    event_hash: str

    @staticmethod
    def compute_hash(prev_hash: str, sequence_num: int, event_type: str, payload_json: str, timestamp_ns: int) -> str:
        digest_input = f"{prev_hash}:{sequence_num}:{event_type}:{payload_json}:{timestamp_ns}".encode("utf-8")
        return hashlib.sha256(digest_input).hexdigest()


class WriteAheadLogStateCheckpoint:
    """Durable state persistence engine using cryptographic write-ahead logging and idempotency checking."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self._events: List[WALEvent] = []
        self._idempotency_keys: Dict[str, Any] = {}
        self._lock = asyncio.Lock()
        self._last_hash = "0" * 64
        self._current_seq = 0

    @property
    def event_count(self) -> int:
        return len(self._events)

    @property
    def head_hash(self) -> str:
        return self._last_hash

    async def append_event(
        self,
        event_type: str,
        payload: Dict[str, Any],
        idempotency_key: Optional[str] = None
    ) -> WALEvent:
        async with self._lock:
            if idempotency_key and idempotency_key in self._idempotency_keys:
                return self._idempotency_keys[idempotency_key]

            self._current_seq += 1
            now_ns = time.time_ns()
            payload_str = json.dumps(payload, sort_keys=True)
            ev_hash = WALEvent.compute_hash(self._last_hash, self._current_seq, event_type, payload_str, now_ns)

            event = WALEvent(
                event_id=str(uuid.uuid4()),
                sequence_num=self._current_seq,
                session_id=self.session_id,
                event_type=event_type,
                payload=payload,
                timestamp_ns=now_ns,
                prev_hash=self._last_hash,
                event_hash=ev_hash,
            )
            self._events.append(event)
            self._last_hash = ev_hash

            if idempotency_key:
                self._idempotency_keys[idempotency_key] = event

            return event

    def verify_integrity(self) -> bool:
        """Verifies cryptographic chain integrity across all committed WAL events."""
        curr_hash = "0" * 64
        for event in self._events:
            if event.prev_hash != curr_hash:
                return False
            payload_str = json.dumps(event.payload, sort_keys=True)
            expected = WALEvent.compute_hash(
                event.prev_hash,
                event.sequence_num,
                event.event_type,
                payload_str,
                event.timestamp_ns
            )
            if event.event_hash != expected:
                return False
            curr_hash = event.event_hash
        return True

    def replay_state(self) -> Dict[str, Any]:
        """Reconstructs state dict by replaying sequential log events."""
        state: Dict[str, Any] = {
            "session_id": self.session_id,
            "variables": {},
            "executed_actions": [],
            "last_sequence": self._current_seq
        }
        for ev in self._events:
            if ev.event_type == "set_variable":
                state["variables"][ev.payload["key"]] = ev.payload["value"]
            elif ev.event_type == "execute_action":
                state["executed_actions"].append(ev.payload)
        return state


@dataclass
class PolicyEvaluationResult:
    allowed: bool
    risk_tier: ActionRiskTier
    requires_escalation: bool
    reason: str


class EnterprisePolicyEnforcer:
    """Enforces compliance, financial bounds, and human-in-the-loop escalation rules."""

    def __init__(self, max_auto_spend_usd: float = 50000.0):
        self.max_auto_spend_usd = max_auto_spend_usd
        self._circuit_broken = False
        self._consecutive_failures = 0
        self._failure_threshold = 3

    def evaluate_action(self, action_type: str, parameters: Dict[str, Any]) -> PolicyEvaluationResult:
        if self._circuit_broken:
            return PolicyEvaluationResult(
                allowed=False,
                risk_tier=ActionRiskTier.TIER_3_HUMAN_CONSENSUS,
                requires_escalation=True,
                reason="Policy circuit breaker active due to repeated validation failures."
            )

        # Read-only or catalog lookups
        if action_type in ("read_catalog", "query_inventory", "check_order_status"):
            return PolicyEvaluationResult(
                allowed=True,
                risk_tier=ActionRiskTier.TIER_1_AUTO_APPROVED,
                requires_escalation=False,
                reason="Low-risk read action automatically approved."
            )

        # Restricted critical operations
        if action_type in ("modify_bank_details", "delete_vendor_record", "purge_audit_trail"):
            return PolicyEvaluationResult(
                allowed=False,
                risk_tier=ActionRiskTier.TIER_3_HUMAN_CONSENSUS,
                requires_escalation=True,
                reason=f"Restricted action {action_type} strictly requires human consensus."
            )

        # Financial transactions
        amount = float(parameters.get("amount_usd", 0.0))
        if amount > self.max_auto_spend_usd:
            return PolicyEvaluationResult(
                allowed=False,
                risk_tier=ActionRiskTier.TIER_3_HUMAN_CONSENSUS,
                requires_escalation=True,
                reason=f"Amount ${amount:.2f} exceeds auto-approval threshold ${self.max_auto_spend_usd:.2f}."
            )

        return PolicyEvaluationResult(
            allowed=True,
            risk_tier=ActionRiskTier.TIER_2_BOUNDED_MUTATION,
            requires_escalation=False,
            reason=f"Bounded mutation (${amount:.2f}) within permissible limits."
        )

    def record_outcome(self, success: bool):
        if success:
            self._consecutive_failures = 0
        else:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._failure_threshold:
                self._circuit_broken = True

    def reset_circuit_breaker(self):
        self._circuit_broken = False
        self._consecutive_failures = 0


@dataclass
class AgentTraceSpan:
    trace_id: str
    span_id: str
    action_type: str
    duration_ms: float
    token_usage: int
    status: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class FreehandEnterpriseAgentRuntime:
    """High-throughput distributed runtime for enterprise autonomous agents."""

    def __init__(
        self,
        session_id: Optional[str] = None,
        policy_enforcer: Optional[EnterprisePolicyEnforcer] = None,
        max_auto_spend_usd: float = 50000.0,
    ):
        self.session_id = session_id or str(uuid.uuid4())
        self.wal = WriteAheadLogStateCheckpoint(self.session_id)
        self.policy = policy_enforcer or EnterprisePolicyEnforcer(max_auto_spend_usd)
        self.trace_spans: List[AgentTraceSpan] = []
        self._tools: Dict[str, Callable[[Dict[str, Any]], Any]] = {}

    def register_tool(self, name: str, handler: Callable[[Dict[str, Any]], Any]):
        self._tools[name] = handler

    async def execute_step(
        self,
        action_type: str,
        params: Dict[str, Any],
        idempotency_key: Optional[str] = None
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()
        trace_id = self.session_id
        span_id = str(uuid.uuid4())

        # Evaluate policy
        eval_result = self.policy.evaluate_action(action_type, params)
        if not eval_result.allowed:
            await self.wal.append_event(
                "policy_blocked",
                {"action": action_type, "reason": eval_result.reason, "tier": eval_result.risk_tier.value},
                idempotency_key=idempotency_key
            )
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            self.trace_spans.append(
                AgentTraceSpan(
                    trace_id=trace_id,
                    span_id=span_id,
                    action_type=action_type,
                    duration_ms=duration_ms,
                    token_usage=120,
                    status="BLOCKED",
                    metadata={"reason": eval_result.reason}
                )
            )
            return {
                "success": False,
                "status": "BLOCKED",
                "risk_tier": eval_result.risk_tier.value,
                "requires_escalation": eval_result.requires_escalation,
                "reason": eval_result.reason
            }

        # Dispatch tool
        handler = self._tools.get(action_type)
        if not handler:
            reason = f"Tool {action_type} not registered in runtime."
            self.policy.record_outcome(False)
            return {"success": False, "status": "ERROR", "reason": reason}

        try:
            if inspect.iscoroutinefunction(handler):
                result = await handler(params)
            else:
                result = handler(params)

            # Persist to WAL
            await self.wal.append_event(
                "execute_action",
                {"action": action_type, "params": params, "result": result},
                idempotency_key=idempotency_key
            )

            self.policy.record_outcome(True)
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            self.trace_spans.append(
                AgentTraceSpan(
                    trace_id=trace_id,
                    span_id=span_id,
                    action_type=action_type,
                    duration_ms=duration_ms,
                    token_usage=250,
                    status="SUCCESS",
                    metadata={"result_keys": list(result.keys()) if isinstance(result, dict) else []}
                )
            )
            return {
                "success": True,
                "status": "COMPLETED",
                "risk_tier": eval_result.risk_tier.value,
                "result": result
            }
        except Exception as e:
            self.policy.record_outcome(False)
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            self.trace_spans.append(
                AgentTraceSpan(
                    trace_id=trace_id,
                    span_id=span_id,
                    action_type=action_type,
                    duration_ms=duration_ms,
                    token_usage=100,
                    status="FAILED",
                    metadata={"error": str(e)}
                )
            )
            return {"success": False, "status": "FAILED", "error": str(e)}


class AgentReplayEvalHarness:
    """Automated evaluation and regression testing harness for enterprise agent workflows."""

    def __init__(self, golden_scenarios: List[Dict[str, Any]]):
        self.golden_scenarios = golden_scenarios

    async def run_benchmark(self, runtime_factory: Callable[[], FreehandEnterpriseAgentRuntime]) -> Dict[str, Any]:
        total_scenarios = len(self.golden_scenarios)
        successful_runs = 0
        total_latency_ms = 0.0
        policy_violations_prevented = 0

        for sc in self.golden_scenarios:
            runtime = runtime_factory()
            action = sc["action"]
            params = sc.get("params", {})
            should_succeed = sc.get("expected_success", True)

            res = await runtime.execute_step(action, params)
            if res["success"] == should_succeed:
                successful_runs += 1
            if not res["success"] and res.get("status") == "BLOCKED":
                policy_violations_prevented += 1

            for span in runtime.trace_spans:
                total_latency_ms += span.duration_ms

        completion_rate = successful_runs / max(1, total_scenarios)
        avg_latency = total_latency_ms / max(1, total_scenarios)

        return {
            "total_scenarios": total_scenarios,
            "successful_runs": successful_runs,
            "completion_rate": completion_rate,
            "policy_violations_prevented": policy_violations_prevented,
            "avg_latency_ms": round(avg_latency, 2),
            "benchmark_status": "PASS" if completion_rate >= 0.95 else "FAIL"
        }
