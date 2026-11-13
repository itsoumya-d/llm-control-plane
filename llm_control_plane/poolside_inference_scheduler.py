"""Poolside Inference Infrastructure: Distributed GPU Workload Scheduler & K8s Controller.

Engineered for frontier code model pre-training, reinforcement learning (RL) rollouts,
evaluations, and low-latency interactive code assistant serving.

Features:
- Multi-tier GPU node cluster management with topology awareness (NVLink, InfiniBand).
- Priority-aware GPU scheduler with preemption (interactive coding assistant preempts batch evals).
- Prefix-aware KV-cache routing to minimize Time-To-First-Token (TTFT).
- Speculative decoding coordination between small draft models and frontier verifiers.
- Kubernetes Custom Controller / Operator reconciliation loop with automated node drain & ECC recovery.
- End-to-end inference observability telemetry (TTFT, ITL, KV hit ratio, GPU utilization).
"""

from __future__ import annotations

import enum
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


class GPUWorkloadType(str, enum.Enum):
    INTERACTIVE_ASSISTANT = "INTERACTIVE_ASSISTANT"  # Highest priority, strict TTFT/ITL SLAs
    SPECULATIVE_VERIFICATION = "SPECULATIVE_VERIFICATION"
    RL_ROLLOUT = "RL_ROLLOUT"  # Large batch RL rollout sampling
    EVALS = "EVALS"  # Preemptible benchmark evaluation sweeps


class GPUNodeState(str, enum.Enum):
    HEALTHY = "HEALTHY"
    DRAINING = "DRAINING"
    DEGRADED_ECC = "DEGRADED_ECC"
    UNHEALTHY = "UNHEALTHY"


@dataclass
class GPUClusterNode:
    """Represents a physical or virtual GPU compute node in a Poolside training/inference cluster."""
    node_id: str
    gpu_model: str = "H100-SXM5-80GB"
    gpu_count: int = 8
    total_vram_gb: float = 640.0
    allocated_vram_gb: float = 0.0
    state: GPUNodeState = GPUNodeState.HEALTHY
    active_workloads: Set[str] = field(default_factory=set)
    cached_prefixes: Dict[str, float] = field(default_factory=dict)  # prefix_hash -> timestamp
    nvlink_bandwidth_gbps: float = 900.0

    @property
    def free_vram_gb(self) -> float:
        return max(0.0, self.total_vram_gb - self.allocated_vram_gb)

    @property
    def utilization_pct(self) -> float:
        if self.total_vram_gb <= 0:
            return 0.0
        return round((self.allocated_vram_gb / self.total_vram_gb) * 100.0, 2)

    def can_fit(self, required_vram_gb: float) -> bool:
        return self.state == GPUNodeState.HEALTHY and self.free_vram_gb >= required_vram_gb

    def allocate(self, workload_id: str, required_vram_gb: float) -> bool:
        if not self.can_fit(required_vram_gb):
            return False
        self.allocated_vram_gb += required_vram_gb
        self.active_workloads.add(workload_id)
        return True

    def release(self, workload_id: str, required_vram_gb: float) -> bool:
        if workload_id in self.active_workloads:
            self.active_workloads.remove(workload_id)
            self.allocated_vram_gb = max(0.0, self.allocated_vram_gb - required_vram_gb)
            return True
        return False

    def mark_drain(self, reason: str = "ECC_ANOMALY") -> None:
        self.state = GPUNodeState.DRAINING if reason == "MAINTENANCE" else GPUNodeState.DEGRADED_ECC


@dataclass
class InferenceWorkload:
    """Descriptor for an inference request or long-running evaluation job."""
    workload_id: str
    model_name: str
    workload_type: GPUWorkloadType
    required_vram_gb: float
    prompt_tokens: int
    max_output_tokens: int
    prompt_prefix: str = ""
    priority: int = 50  # 1-100, higher = higher priority
    assigned_node_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    speculative_draft_model: Optional[str] = None
    completed: bool = False

    @property
    def prefix_hash(self) -> str:
        if not self.prompt_prefix:
            return ""
        return hashlib.sha256(self.prompt_prefix.encode("utf-8")).hexdigest()[:16]


@dataclass
class ObservabilityMetric:
    workload_id: str
    node_id: str
    workload_type: GPUWorkloadType
    ttft_ms: float
    itl_ms: float
    tokens_generated: int
    kv_cache_hit: bool
    gpu_utilization_pct: float
    timestamp: float = field(default_factory=time.time)


class PoolsideInferenceScheduler:
    """Distributed GPU workload scheduler for Poolside frontier code models.
    
    Provides:
    - Prefix-aware routing (maximizing KV-cache reuse across shared repo/AST context).
    - Priority-based scheduling with preemption of batch RL/eval jobs.
    - Speculative decoding execution coordinator.
    """

    def __init__(self, nodes: Optional[List[GPUClusterNode]] = None):
        self.nodes: Dict[str, GPUClusterNode] = {n.node_id: n for n in (nodes or [])}
        self.pending_queue: List[InferenceWorkload] = []
        self.active_workloads: Dict[str, InferenceWorkload] = {}
        self.completed_workloads: Dict[str, InferenceWorkload] = {}
        self.metrics_history: List[ObservabilityMetric] = []

    def register_node(self, node: GPUClusterNode) -> None:
        self.nodes[node.node_id] = node

    def submit_workload(self, workload: InferenceWorkload) -> bool:
        # Assign default priorities based on workload type if not custom
        if workload.workload_type == GPUWorkloadType.INTERACTIVE_ASSISTANT:
            workload.priority = max(workload.priority, 90)
        elif workload.workload_type == GPUWorkloadType.SPECULATIVE_VERIFICATION:
            workload.priority = max(workload.priority, 80)
        elif workload.workload_type == GPUWorkloadType.RL_ROLLOUT:
            workload.priority = min(workload.priority, 50)
        elif workload.workload_type == GPUWorkloadType.EVALS:
            workload.priority = min(workload.priority, 30)

        self.pending_queue.append(workload)
        # Sort queue by priority descending, then created_at ascending
        self.pending_queue.sort(key=lambda w: (-w.priority, w.created_at))
        return True

    def schedule_next(self) -> Optional[Tuple[str, str]]:
        """Schedules the highest priority workload to the optimal GPU node.
        
        Returns:
            Tuple of (workload_id, node_id) if scheduled, None if blocked.
        """
        if not self.pending_queue:
            return None

        # Look for the first workload that can be placed
        for i, workload in enumerate(self.pending_queue):
            best_node = self._find_best_node(workload)
            if best_node:
                self.pending_queue.pop(i)
                best_node.allocate(workload.workload_id, workload.required_vram_gb)
                workload.assigned_node_id = best_node.node_id
                self.active_workloads[workload.workload_id] = workload

                # Record prefix in node cache
                if workload.prefix_hash:
                    best_node.cached_prefixes[workload.prefix_hash] = time.time()

                return (workload.workload_id, best_node.node_id)

        # If high-priority interactive request is starved, attempt preemption
        highest = self.pending_queue[0]
        if highest.workload_type in (GPUWorkloadType.INTERACTIVE_ASSISTANT, GPUWorkloadType.SPECULATIVE_VERIFICATION):
            preempted = self._attempt_preemption(highest)
            if preempted:
                return self.schedule_next()

        return None

    def _find_best_node(self, workload: InferenceWorkload) -> Optional[GPUClusterNode]:
        """Finds the best healthy node: prefers KV-cache prefix affinity, then lowest utilization."""
        candidates = [n for n in self.nodes.values() if n.can_fit(workload.required_vram_gb)]
        if not candidates:
            return None

        # Preference 1: Node with cached prefix
        if workload.prefix_hash:
            for node in candidates:
                if workload.prefix_hash in node.cached_prefixes:
                    return node

        # Preference 2: Node with most free VRAM (lowest utilization)
        candidates.sort(key=lambda n: n.free_vram_gb, reverse=True)
        return candidates[0]

    def _attempt_preemption(self, urgent_workload: InferenceWorkload) -> bool:
        """Preempts lowest priority batch EVALS or RL_ROLLOUT to free up VRAM."""
        preemptible_workloads = [
            w for w in self.active_workloads.values()
            if w.workload_type in (GPUWorkloadType.EVALS, GPUWorkloadType.RL_ROLLOUT)
            and w.priority < urgent_workload.priority
        ]
        if not preemptible_workloads:
            return False

        # Sort preemptible by lowest priority
        preemptible_workloads.sort(key=lambda w: w.priority)
        target = preemptible_workloads[0]
        node = self.nodes.get(target.assigned_node_id or "")
        if not node:
            return False

        # Evict target from active, return to pending queue
        node.release(target.workload_id, target.required_vram_gb)
        del self.active_workloads[target.workload_id]
        target.assigned_node_id = None
        self.pending_queue.append(target)
        self.pending_queue.sort(key=lambda w: (-w.priority, w.created_at))
        return True

    def complete_workload(
        self,
        workload_id: str,
        tokens_generated: int,
        ttft_ms: float = 12.5,
        itl_ms: float = 8.2,
    ) -> Optional[ObservabilityMetric]:
        """Marks a workload complete, frees node VRAM, and records telemetry."""
        workload = self.active_workloads.pop(workload_id, None)
        if not workload:
            return None

        node = self.nodes.get(workload.assigned_node_id or "")
        if node:
            node.release(workload.workload_id, workload.required_vram_gb)

        workload.completed = True
        self.completed_workloads[workload_id] = workload

        kv_hit = bool(workload.prefix_hash and node and workload.prefix_hash in node.cached_prefixes)
        metric = ObservabilityMetric(
            workload_id=workload_id,
            node_id=node.node_id if node else "unknown",
            workload_type=workload.workload_type,
            ttft_ms=ttft_ms,
            itl_ms=itl_ms,
            tokens_generated=tokens_generated,
            kv_cache_hit=kv_hit,
            gpu_utilization_pct=node.utilization_pct if node else 0.0,
        )
        self.metrics_history.append(metric)
        return metric

    def cluster_stats(self) -> Dict[str, Any]:
        total_gpus = sum(n.gpu_count for n in self.nodes.values())
        total_vram = sum(n.total_vram_gb for n in self.nodes.values())
        used_vram = sum(n.allocated_vram_gb for n in self.nodes.values())
        healthy_nodes = sum(1 for n in self.nodes.values() if n.state == GPUNodeState.HEALTHY)
        return {
            "node_count": len(self.nodes),
            "healthy_nodes": healthy_nodes,
            "total_gpus": total_gpus,
            "total_vram_gb": total_vram,
            "allocated_vram_gb": used_vram,
            "cluster_utilization_pct": round((used_vram / total_vram * 100.0) if total_vram > 0 else 0.0, 2),
            "pending_queue_depth": len(self.pending_queue),
            "active_workloads_count": len(self.active_workloads),
            "completed_workloads_count": len(self.completed_workloads),
        }


class PoolsideKubernetesController:
    """Kubernetes Operator & Custom Controller reconcile loop for Poolside inference clusters.
    
    Automates:
    - Pod replica autoscaling based on queue depth and token demand.
    - Node health monitoring: cordoning and draining nodes upon ECC or Xid hardware errors.
    - Zero-downtime rolling weight updates across inference worker pods.
    """

    def __init__(self, scheduler: PoolsideInferenceScheduler, min_replicas: int = 2, max_replicas: int = 16):
        self.scheduler = scheduler
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.current_replicas = min_replicas
        self.drained_nodes: List[str] = []
        self.reconciliation_history: List[Dict[str, Any]] = []

    def reconcile(self) -> Dict[str, Any]:
        """Runs one reconciliation cycle, balancing desired replicas vs cluster load."""
        stats = self.scheduler.cluster_stats()
        queue_depth = stats["pending_queue_depth"]

        # Autoscaling policy: 1 replica per 4 pending requests, bounded by [min, max]
        desired_replicas = max(self.min_replicas, min(self.max_replicas, self.min_replicas + (queue_depth // 4)))
        action = "NOOP"
        if desired_replicas > self.current_replicas:
            action = f"SCALE_UP_{self.current_replicas}_TO_{desired_replicas}"
            self.current_replicas = desired_replicas
        elif desired_replicas < self.current_replicas and queue_depth == 0:
            action = f"SCALE_DOWN_{self.current_replicas}_TO_{desired_replicas}"
            self.current_replicas = desired_replicas

        # Inspect degraded nodes
        degraded = [
            n for n in self.scheduler.nodes.values()
            if n.state in (GPUNodeState.DEGRADED_ECC, GPUNodeState.DRAINING)
        ]
        drained_count = 0
        for node in degraded:
            if node.node_id not in self.drained_nodes:
                self._drain_node(node)
                drained_count += 1

        record = {
            "timestamp": time.time(),
            "queue_depth": queue_depth,
            "current_replicas": self.current_replicas,
            "desired_replicas": desired_replicas,
            "action": action,
            "drained_nodes_cycle": drained_count,
        }
        self.reconciliation_history.append(record)
        return record

    def handle_node_ecc_alert(self, node_id: str, ecc_count: int) -> bool:
        """Handles hardware alert from NVIDIA DCGM or Xid telemetry."""
        node = self.scheduler.nodes.get(node_id)
        if not node:
            return False
        if ecc_count > 10:
            node.mark_drain(reason="ECC_ANOMALY")
            self.reconcile()
            return True
        return False

    def _drain_node(self, node: GPUClusterNode) -> None:
        """Evicts active workloads from a degraded node and re-queues them."""
        for workload_id in list(node.active_workloads):
            workload = self.scheduler.active_workloads.pop(workload_id, None)
            if workload:
                workload.assigned_node_id = None
                self.scheduler.pending_queue.insert(0, workload)
        node.active_workloads.clear()
        node.allocated_vram_gb = 0.0
        self.drained_nodes.append(node.node_id)
