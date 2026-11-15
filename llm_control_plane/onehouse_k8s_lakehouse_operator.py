"""Onehouse Kubernetes Lakehouse Infrastructure Operator & Controller.

Implements declarative Kubernetes custom controllers, Hudi table service reconciliation loops,
dynamic compaction worker autoscaling, and spot node preemption checkpointing
for large-scale multi-tenant data lakehouse clusters.
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class TableServiceType(str, enum.Enum):
    COMPACTION = "compaction"
    CLUSTERING = "clustering"
    CLEANING = "cleaning"
    METADATA_INDEXING = "metadata_indexing"


class ReconciliationState(str, enum.Enum):
    PENDING = "Pending"
    PROVISIONING = "Provisioning"
    RUNNING = "Running"
    COMPACTING = "Compacting"
    SUCCEEDED = "Succeeded"
    DEGRADED = "Degraded"
    DRAINING = "Draining"
    TERMINATING = "Terminating"


@dataclass
class LakehouseResourceLimits:
    cpu_cores: float = 4.0
    memory_gib: float = 16.0
    ephemeral_storage_gib: float = 50.0
    gpu_count: int = 0


@dataclass
class LakehouseServiceSpec:
    service_id: str
    table_name: str
    database: str
    service_type: TableServiceType
    min_workers: int = 2
    max_workers: int = 16
    resource_limits: LakehouseResourceLimits = field(default_factory=LakehouseResourceLimits)
    commit_backlog_threshold: int = 100
    parquet_skew_threshold_mb: float = 512.0
    target_compaction_ratio: float = 0.85
    checkpoint_storage_uri: str = "s3://onehouse-lakehouse-checkpoints/hudi/"
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class WorkerPodInstance:
    pod_id: str
    node_name: str
    status: str
    ip_address: str
    assigned_partition: str
    allocated_memory_gib: float
    active_compaction_tasks: int = 0
    is_draining: bool = False
    registered_at: float = field(default_factory=time.time)


@dataclass
class LakehouseReconciliationStatus:
    service_id: str
    state: ReconciliationState
    active_workers: int
    pending_tasks: int
    completed_commits: int
    reconciliation_epoch: int
    last_reconciled_at: float
    conditions: List[Dict[str, Any]] = field(default_factory=list)
    error_message: Optional[str] = None


class OnehouseK8sLakehouseReconciler:
    """Kubernetes controller reconciler for Apache Hudi table services and workers."""

    def __init__(self, spec: LakehouseServiceSpec):
        self.spec = spec
        self.workers: Dict[str, WorkerPodInstance] = {}
        self.current_state = ReconciliationState.PENDING
        self.reconciliation_epoch = 0
        self.completed_commits = 0
        self.pending_tasks = 0
        self.checkpoints: Dict[str, Dict[str, Any]] = {}
        self.conditions: List[Dict[str, Any]] = []
        self._pod_counter = 0

    def reconcile(self, current_backlog: int, file_skew_mb: float) -> LakehouseReconciliationStatus:
        """Execute a controller reconciliation cycle against current cluster metrics."""
        self.reconciliation_epoch += 1
        now = time.time()
        self.pending_tasks = current_backlog

        # 1. State transition from PENDING -> PROVISIONING -> RUNNING
        if self.current_state in (ReconciliationState.PENDING, ReconciliationState.DEGRADED):
            self.current_state = ReconciliationState.PROVISIONING
            self._ensure_minimum_workers()
            self.current_state = ReconciliationState.RUNNING

        # 2. Check for compaction triggers
        if current_backlog >= self.spec.commit_backlog_threshold or file_skew_mb >= self.spec.parquet_skew_threshold_mb:
            self.current_state = ReconciliationState.COMPACTING
            desired = self.compute_autoscaling_demand(current_backlog, file_skew_mb)
            self._scale_to(desired)
        elif self.current_state == ReconciliationState.COMPACTING and current_backlog == 0:
            self.current_state = ReconciliationState.SUCCEEDED
            self.completed_commits += 1

        # 3. Purge dead or completed workers
        self._prune_drained_workers()

        status = LakehouseReconciliationStatus(
            service_id=self.spec.service_id,
            state=self.current_state,
            active_workers=len([w for w in self.workers.values() if not w.is_draining]),
            pending_tasks=self.pending_tasks,
            completed_commits=self.completed_commits,
            reconciliation_epoch=self.reconciliation_epoch,
            last_reconciled_at=now,
            conditions=self._generate_conditions(),
            error_message=None,
        )
        return status

    def compute_autoscaling_demand(self, backlog: int, file_skew_mb: float) -> int:
        """Computes optimal worker pod count based on commit backpressure and file fragmentation."""
        backlog_factor = max(1.0, backlog / max(1, self.spec.commit_backlog_threshold))
        skew_factor = max(1.0, file_skew_mb / max(1.0, self.spec.parquet_skew_threshold_mb))

        active_count = len([w for w in self.workers.values() if not w.is_draining])
        base = max(self.spec.min_workers, active_count)
        target = int(round(base * max(backlog_factor, skew_factor)))
        bounded = max(self.spec.min_workers, min(self.spec.max_workers, target))
        return bounded

    def handle_spot_preemption(self, node_name: str) -> Dict[str, Any]:
        """Handles spot node interruption notice, drains pods, and checkpoints active transactions."""
        drained_pods = []
        checkpoint_summary = {}

        for pod_id, worker in list(self.workers.items()):
            if worker.node_name == node_name and not worker.is_draining:
                worker.is_draining = True
                drained_pods.append(pod_id)

                # Commit checkpoint for inflight table transactions
                checkpoint_id = f"chk_{self.spec.table_name}_{pod_id}_{int(time.time())}"
                checkpoint_data = {
                    "checkpoint_id": checkpoint_id,
                    "pod_id": pod_id,
                    "partition": worker.assigned_partition,
                    "active_tasks": worker.active_compaction_tasks,
                    "storage_uri": f"{self.spec.checkpoint_storage_uri}{checkpoint_id}.json",
                    "checkpointed_at": time.time(),
                }
                self.checkpoints[checkpoint_id] = checkpoint_data
                checkpoint_summary[pod_id] = checkpoint_data

        # Compensate for lost pods immediately to maintain min replicas
        self._ensure_minimum_workers()

        return {
            "node_name": node_name,
            "drained_pod_count": len(drained_pods),
            "drained_pods": drained_pods,
            "checkpoints": checkpoint_summary,
            "reconciliation_state": self.current_state.value,
        }

    def _ensure_minimum_workers(self) -> None:
        active = len([w for w in self.workers.values() if not w.is_draining])
        needed = self.spec.min_workers - active
        for _ in range(max(0, needed)):
            self._pod_counter += 1
            pod_id = f"onehouse-{self.spec.service_id}-worker-{self._pod_counter}"
            self.workers[pod_id] = WorkerPodInstance(
                pod_id=pod_id,
                node_name=f"k8s-node-zone-a-{self._pod_counter % 3}",
                status="Running",
                ip_address=f"10.244.{self._pod_counter % 250}.{10 + (self._pod_counter % 200)}",
                assigned_partition=f"part_year=2026/part_month=11/chunk_{self._pod_counter}",
                allocated_memory_gib=self.spec.resource_limits.memory_gib,
                active_compaction_tasks=0,
            )

    def _scale_to(self, desired_count: int) -> None:
        active = [w for w in self.workers.values() if not w.is_draining]
        current_count = len(active)

        if desired_count > current_count:
            for _ in range(desired_count - current_count):
                self._pod_counter += 1
                pod_id = f"onehouse-{self.spec.service_id}-worker-{self._pod_counter}"
                self.workers[pod_id] = WorkerPodInstance(
                    pod_id=pod_id,
                    node_name=f"k8s-node-zone-b-{self._pod_counter % 4}",
                    status="Running",
                    ip_address=f"10.244.2.{self._pod_counter % 250}",
                    assigned_partition=f"part_year=2026/chunk_{self._pod_counter}",
                    allocated_memory_gib=self.spec.resource_limits.memory_gib,
                    active_compaction_tasks=2,
                )
        elif desired_count < current_count:
            # Scale down idle or least loaded pods
            remove_count = current_count - desired_count
            candidates = sorted(active, key=lambda w: w.active_compaction_tasks)
            for w in candidates[:remove_count]:
                w.is_draining = True

    def _prune_drained_workers(self) -> None:
        to_remove = [k for k, w in self.workers.items() if w.is_draining and w.active_compaction_tasks == 0]
        for k in to_remove:
            del self.workers[k]

    def _generate_conditions(self) -> List[Dict[str, Any]]:
        ready_states = (ReconciliationState.RUNNING, ReconciliationState.SUCCEEDED, ReconciliationState.COMPACTING)
        return [
            {
                "type": "Ready",
                "status": "True" if self.current_state in ready_states else "False",
                "lastTransitionTime": time.time(),
                "reason": f"StateIs{self.current_state.value}",
            },
            {
                "type": "Compacting",
                "status": "True" if self.current_state == ReconciliationState.COMPACTING else "False",
                "lastTransitionTime": time.time(),
                "reason": "DeltaCommitsOrSkewActive",
            },
        ]
