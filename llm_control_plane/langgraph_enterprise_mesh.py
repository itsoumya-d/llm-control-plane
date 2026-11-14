"""LangGraph Enterprise Mesh & Eval-Driven Development (EDD) Runtime.

Provides enterprise-scale multi-agent orchestration, stateful channel synchronization,
durable checkpointing (with short and long-term memory brokers), cyclic graph validation,
human-in-the-loop interruption gates, LangSmith-compatible trace mining, and
eval-driven benchmark harnesses.
"""

from __future__ import annotations

import copy
import enum
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple


class ChannelType(str, enum.Enum):
    """Supported state channel synchronization behaviors."""

    EPHEMERAL_VALUE = "ephemeral_value"
    APPEND_ONLY_LIST = "append_only_list"
    MERGE_DICT = "merge_dict"
    LAST_WRITE_WINS = "last_write_wins"
    CHECKPOINTED_STATE = "checkpointed_state"


class AgentRole(str, enum.Enum):
    """Specialized agent roles within the enterprise orchestration mesh."""

    SUPERVISOR = "supervisor"
    RESEARCHER = "researcher"
    CODE_EXECUTOR = "code_executor"
    COMPLIANCE_GATE = "compliance_gate"
    CRITIC_REVIEWER = "critic_reviewer"
    DEPLOYED_SPECIALIST = "deployed_specialist"


class RunSpanType(str, enum.Enum):
    """Execution span types for LangSmith-compatible trace mining."""

    AGENT = "agent"
    CHAIN = "chain"
    LLM = "llm"
    TOOL = "tool"
    RETRIEVER = "retriever"


@dataclass
class ChannelDefinition:
    """Configuration for an individual state channel within a multi-agent graph."""

    name: str
    channel_type: ChannelType
    default_value: Any
    reducer: Optional[Callable[[Any, Any], Any]] = None


@dataclass
class StateSnapshot:
    """Point-in-time snapshot of multi-agent state with audit metadata."""

    thread_id: str
    checkpoint_id: str
    step_index: int
    active_node: str
    channels: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    parent_checkpoint_id: Optional[str] = None
    hash_id: str = ""

    def __post_init__(self) -> None:
        if not self.hash_id:
            raw = f"{self.thread_id}:{self.checkpoint_id}:{self.step_index}:{sorted(self.channels.keys())}"
            self.hash_id = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


@dataclass
class TraceSpan:
    """Individual execution span recording runtime telemetry for evaluation."""

    span_id: str
    name: str
    span_type: RunSpanType
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    start_time: float
    end_time: float
    parent_span_id: Optional[str] = None
    status: str = "success"
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def latency_ms(self) -> float:
        return max(0.0, (self.end_time - self.start_time) * 1000.0)


class DurableCheckpointBroker:
    """Manages multi-tier checkpoint storage, state compaction, and time-travel forks."""

    def __init__(self, max_history_per_thread: int = 50) -> None:
        self.max_history = max_history_per_thread
        # thread_id -> list of StateSnapshot
        self._checkpoints: Dict[str, List[StateSnapshot]] = {}

    def save_checkpoint(
        self,
        thread_id: str,
        active_node: str,
        channels: Dict[str, Any],
        step_index: int,
        parent_id: Optional[str] = None,
    ) -> StateSnapshot:
        """Persists an immutable state snapshot into thread history with compaction."""
        if thread_id not in self._checkpoints:
            self._checkpoints[thread_id] = []

        checkpoint_id = f"chk_{step_index}_{int(time.time() * 1000)}"
        snapshot = StateSnapshot(
            thread_id=thread_id,
            checkpoint_id=checkpoint_id,
            step_index=step_index,
            active_node=active_node,
            channels=copy.deepcopy(channels),
            parent_checkpoint_id=parent_id,
        )

        history = self._checkpoints[thread_id]
        history.append(snapshot)

        # Compaction: retain the initial checkpoint and recent N checkpoints
        if len(history) > self.max_history:
            self._checkpoints[thread_id] = [history[0]] + history[-(self.max_history - 1) :]

        return snapshot

    def get_latest_checkpoint(self, thread_id: str) -> Optional[StateSnapshot]:
        history = self._checkpoints.get(thread_id, [])
        return history[-1] if history else None

    def get_checkpoint(self, thread_id: str, checkpoint_id: str) -> Optional[StateSnapshot]:
        for chk in self._checkpoints.get(thread_id, []):
            if chk.checkpoint_id == checkpoint_id:
                return chk
        return None

    def fork_thread(self, source_thread_id: str, checkpoint_id: str, target_thread_id: str) -> StateSnapshot:
        """Forks an existing thread from a prior checkpoint into a new independent thread."""
        source = self.get_checkpoint(source_thread_id, checkpoint_id)
        if not source:
            raise ValueError(f"Checkpoint {checkpoint_id} not found in thread {source_thread_id}")

        forked = StateSnapshot(
            thread_id=target_thread_id,
            checkpoint_id=f"fork_{int(time.time() * 1000)}",
            step_index=source.step_index,
            active_node=source.active_node,
            channels=copy.deepcopy(source.channels),
            parent_checkpoint_id=source.checkpoint_id,
        )
        self._checkpoints[target_thread_id] = [forked]
        return forked


class CyclicGraphValidator:
    """Validates multi-agent graph topologies for unhandled cycles, deadlocks, and recursion limits."""

    def __init__(self, max_recursion_limit: int = 25) -> None:
        self.max_recursion = max_recursion_limit

    def validate_topology(
        self,
        nodes: Sequence[str],
        edges: Sequence[Tuple[str, str]],
        conditional_edges: Dict[str, Sequence[str]],
        terminal_nodes: Sequence[str],
    ) -> Dict[str, Any]:
        """Analyzes graph reachability, terminal convergence, and detects invalid loops."""
        node_set = set(nodes)
        term_set = set(terminal_nodes)

        # Build adjacency graph
        adj: Dict[str, Set[str]] = {n: set() for n in nodes}
        for u, v in edges:
            if u in adj and v in node_set:
                adj[u].add(v)

        for u, branches in conditional_edges.items():
            if u in adj:
                for b in branches:
                    if b in node_set:
                        adj[u].add(b)

        # Verify all nodes can reach at least one terminal node
        unreachable_nodes = []
        for n in nodes:
            visited = set()
            stack = [n]
            reaches_term = False
            while stack:
                curr = stack.pop()
                if curr in term_set:
                    reaches_term = True
                    break
                if curr not in visited:
                    visited.add(curr)
                    stack.extend(adj.get(curr, set()) - visited)
            if not reaches_term:
                unreachable_nodes.append(n)

        # Detect cycles
        cycles = self._find_cycles(nodes, adj)

        return {
            "valid": len(unreachable_nodes) == 0,
            "total_nodes": len(nodes),
            "unreachable_from_terminal": unreachable_nodes,
            "detected_cycles": cycles,
            "cycle_count": len(cycles),
            "max_recursion_limit": self.max_recursion,
        }

    def _find_cycles(self, nodes: Sequence[str], adj: Dict[str, Set[str]]) -> List[List[str]]:
        visited: Dict[str, int] = {n: 0 for n in nodes}  # 0: unvisited, 1: visiting, 2: visited
        cycles: List[List[str]] = []
        path: List[str] = []

        def dfs(node: str) -> None:
            visited[node] = 1
            path.append(node)
            for nxt in adj.get(node, set()):
                if visited.get(nxt) == 1:
                    idx = path.index(nxt)
                    cycles.append(path[idx:] + [nxt])
                elif visited.get(nxt) == 0:
                    dfs(nxt)
            path.pop()
            visited[node] = 2

        for n in nodes:
            if visited.get(n) == 0:
                dfs(n)

        return cycles


class HumanInTheLoopInterrupter:
    """Manages safe execution halts, compliance review gates, and human approval resumption."""

    def __init__(self) -> None:
        # node_name -> condition function: (state_dict) -> bool
        self._interrupt_rules: Dict[str, Callable[[Dict[str, Any]], bool]] = {}
        # thread_id -> paused context dict
        self._paused_threads: Dict[str, Dict[str, Any]] = {}

    def register_interrupt(self, node_name: str, condition: Optional[Callable[[Dict[str, Any]], bool]] = None) -> None:
        """Registers a node that triggers execution interruption before running."""
        self._interrupt_rules[node_name] = condition or (lambda _: True)

    def should_interrupt(self, node_name: str, current_state: Dict[str, Any]) -> bool:
        if node_name not in self._interrupt_rules:
            return False
        return self._interrupt_rules[node_name](current_state)

    def pause_execution(
        self,
        thread_id: str,
        node_name: str,
        reason: str,
        state_snapshot: StateSnapshot,
    ) -> Dict[str, Any]:
        """Pauses a thread and stores resumption context."""
        context = {
            "thread_id": thread_id,
            "node_name": node_name,
            "reason": reason,
            "paused_at": time.time(),
            "snapshot_id": state_snapshot.checkpoint_id,
            "state_channels": copy.deepcopy(state_snapshot.channels),
        }
        self._paused_threads[thread_id] = context
        return context

    def resume_execution(
        self,
        thread_id: str,
        approver: str,
        modifications: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Resumes a paused thread with optional state overrides and audit trail."""
        if thread_id not in self._paused_threads:
            raise KeyError(f"No paused execution found for thread {thread_id}")

        paused = self._paused_threads.pop(thread_id)
        resumed_channels = copy.deepcopy(paused["state_channels"])
        if modifications:
            resumed_channels.update(modifications)

        return {
            "thread_id": thread_id,
            "node_name": paused["node_name"],
            "approver": approver,
            "resumed_at": time.time(),
            "updated_channels": resumed_channels,
        }

    def is_paused(self, thread_id: str) -> bool:
        return thread_id in self._paused_threads


class TraceMiningEvaluator:
    """Ingests LangSmith-compatible execution traces and computes quantitative trajectory metrics."""

    def __init__(self) -> None:
        self._spans: List[TraceSpan] = []

    def record_span(
        self,
        name: str,
        span_type: RunSpanType,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        start_time: float,
        end_time: float,
        parent_span_id: Optional[str] = None,
        status: str = "success",
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TraceSpan:
        span_id = f"span_{len(self._spans) + 1}_{int(time.time() * 1000)}"
        span = TraceSpan(
            span_id=span_id,
            name=name,
            span_type=span_type,
            inputs=copy.deepcopy(inputs),
            outputs=copy.deepcopy(outputs),
            start_time=start_time,
            end_time=end_time,
            parent_span_id=parent_span_id,
            status=status,
            error_message=error_message,
            metadata=metadata or {},
        )
        self._spans.append(span)
        return span

    def compute_trajectory_metrics(self) -> Dict[str, Any]:
        """Computes tool precision, step efficiency, latency percentiles, and error rate."""
        if not self._spans:
            return {"total_spans": 0, "error_rate": 0.0}

        total = len(self._spans)
        tool_spans = [s for s in self._spans if s.span_type == RunSpanType.TOOL]
        errors = [s for s in self._spans if s.status != "success"]
        latencies = sorted(s.latency_ms for s in self._spans)

        p50 = latencies[int(len(latencies) * 0.5)]
        p95 = latencies[int(len(latencies) * 0.95)] if len(latencies) >= 20 else latencies[-1]
        p99 = latencies[int(len(latencies) * 0.99)] if len(latencies) >= 100 else latencies[-1]

        # Tool precision: ratio of tool spans without errors to total tool spans
        tool_precision = 1.0
        if tool_spans:
            success_tools = [t for t in tool_spans if t.status == "success"]
            tool_precision = len(success_tools) / len(tool_spans)

        return {
            "total_spans": total,
            "tool_spans_count": len(tool_spans),
            "tool_precision": round(tool_precision, 4),
            "error_rate": round(len(errors) / total, 4),
            "latency_p50_ms": round(p50, 2),
            "latency_p95_ms": round(p95, 2),
            "latency_p99_ms": round(p99, 2),
            "avg_latency_ms": round(sum(latencies) / total, 2),
        }

    def export_langsmith_dataset(self) -> List[Dict[str, Any]]:
        """Formats collected traces as LangSmith evaluation benchmark dataset rows."""
        rows = []
        for s in self._spans:
            rows.append(
                {
                    "id": s.span_id,
                    "name": s.name,
                    "run_type": s.span_type.value,
                    "inputs": s.inputs,
                    "outputs": s.outputs,
                    "latency_ms": s.latency_ms,
                    "status": s.status,
                    "metadata": s.metadata,
                }
            )
        return rows


class EnterpriseCompanyHarness:
    """End-to-end multi-agent mesh coordinator implementing enterprise state channels and governance."""

    def __init__(
        self,
        checkpoint_broker: Optional[DurableCheckpointBroker] = None,
        interrupter: Optional[HumanInTheLoopInterrupter] = None,
        validator: Optional[CyclicGraphValidator] = None,
        trace_evaluator: Optional[TraceMiningEvaluator] = None,
    ) -> None:
        self.broker = checkpoint_broker or DurableCheckpointBroker()
        self.interrupter = interrupter or HumanInTheLoopInterrupter()
        self.validator = validator or CyclicGraphValidator()
        self.evaluator = trace_evaluator or TraceMiningEvaluator()

        self._channels: Dict[str, ChannelDefinition] = {}
        self._agent_nodes: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {}
        self._rbac_tools: Dict[str, Set[str]] = {}

    def register_channel(
        self,
        name: str,
        channel_type: ChannelType,
        default_value: Any,
        reducer: Optional[Callable[[Any, Any], Any]] = None,
    ) -> None:
        self._channels[name] = ChannelDefinition(
            name=name, channel_type=channel_type, default_value=default_value, reducer=reducer
        )

    def register_agent(
        self,
        role: AgentRole,
        name: str,
        handler: Callable[[Dict[str, Any]], Dict[str, Any]],
        allowed_tools: Sequence[str],
    ) -> None:
        self._agent_nodes[name] = handler
        self._rbac_tools[name] = set(allowed_tools)

    def initialize_state(self) -> Dict[str, Any]:
        """Initializes default channel state dictionary."""
        return {name: copy.deepcopy(ch.default_value) for name, ch in self._channels.items()}

    def apply_channel_update(self, current_state: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
        """Applies channel-specific merge/reducer logic."""
        new_state = copy.deepcopy(current_state)
        for key, val in updates.items():
            if key not in self._channels:
                new_state[key] = val
                continue

            ch = self._channels[key]
            if ch.reducer:
                new_state[key] = ch.reducer(new_state.get(key, ch.default_value), val)
            elif ch.channel_type == ChannelType.APPEND_ONLY_LIST:
                existing = new_state.get(key, [])
                if isinstance(existing, list):
                    if isinstance(val, list):
                        new_state[key] = existing + val
                    else:
                        new_state[key] = existing + [val]
            elif ch.channel_type == ChannelType.MERGE_DICT:
                existing = new_state.get(key, {})
                if isinstance(existing, dict) and isinstance(val, dict):
                    merged = dict(existing)
                    merged.update(val)
                    new_state[key] = merged
            else:
                new_state[key] = val

        return new_state

    def execute_turn(
        self,
        thread_id: str,
        active_node: str,
        step_index: int,
    ) -> Dict[str, Any]:
        """Executes a single node turn with checkpointing, interrupt checks, and trace recording."""
        if active_node not in self._agent_nodes:
            raise KeyError(f"Node {active_node} not registered in company harness")

        latest = self.broker.get_latest_checkpoint(thread_id)
        current_state = latest.channels if latest else self.initialize_state()

        # Check human-in-the-loop interruption gate
        if self.interrupter.should_interrupt(active_node, current_state):
            snap = self.broker.save_checkpoint(thread_id, active_node, current_state, step_index)
            pause_ctx = self.interrupter.pause_execution(
                thread_id, active_node, f"Compliance interrupt triggered on {active_node}", snap
            )
            return {"status": "paused", "context": pause_ctx}

        t_start = time.time()
        handler = self._agent_nodes[active_node]
        try:
            delta = handler(current_state)
            t_end = time.time()
            new_state = self.apply_channel_update(current_state, delta)
            parent_id = latest.checkpoint_id if latest else None
            snapshot = self.broker.save_checkpoint(thread_id, active_node, new_state, step_index, parent_id)

            self.evaluator.record_span(
                name=active_node,
                span_type=RunSpanType.AGENT,
                inputs=current_state,
                outputs=delta,
                start_time=t_start,
                end_time=t_end,
                status="success",
            )
            return {
                "status": "success",
                "checkpoint": snapshot,
                "state": new_state,
                "active_node": active_node,
            }
        except Exception as e:
            t_end = time.time()
            self.evaluator.record_span(
                name=active_node,
                span_type=RunSpanType.AGENT,
                inputs=current_state,
                outputs={},
                start_time=t_start,
                end_time=t_end,
                status="error",
                error_message=str(e),
            )
            raise


class EvalDrivenBenchmarkHarness:
    """Executes offline evaluation suites with LLM-as-a-judge scorers and metric benchmarking."""

    def __init__(self, harness: EnterpriseCompanyHarness) -> None:
        self.harness = harness

    def run_eval_dataset(
        self,
        dataset: Sequence[Dict[str, Any]],
        evaluator_fn: Callable[[Dict[str, Any], Dict[str, Any]], float],
    ) -> Dict[str, Any]:
        """Runs a batch evaluation over golden test cases and computes aggregate benchmark scores."""
        scores = []
        latencies = []
        passed_count = 0

        for idx, item in enumerate(dataset):
            thread_id = f"eval_thread_{idx}_{int(time.time())}"
            expected_criteria = item.get("criteria", {})

            t0 = time.time()
            # Initialize state with test input
            self.harness.register_channel("query", ChannelType.LAST_WRITE_WINS, "")
            self.harness.register_channel("history", ChannelType.APPEND_ONLY_LIST, [])
            self.harness.register_channel("final_output", ChannelType.LAST_WRITE_WINS, "")

            res = self.harness.execute_turn(thread_id, item.get("start_node", "supervisor"), step_index=1)
            t1 = time.time()

            latency_ms = (t1 - t0) * 1000.0
            latencies.append(latency_ms)

            state = res.get("state", {})
            score = evaluator_fn(state, expected_criteria)
            scores.append(score)
            if score >= 0.80:
                passed_count += 1

        avg_score = sum(scores) / len(scores) if scores else 0.0
        pass_rate = passed_count / len(dataset) if dataset else 0.0

        return {
            "total_eval_samples": len(dataset),
            "passed_samples": passed_count,
            "pass_rate": round(pass_rate, 4),
            "mean_eval_score": round(avg_score, 4),
            "mean_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
            "benchmark_status": "PASS" if pass_rate >= 0.85 else "FAIL",
        }
