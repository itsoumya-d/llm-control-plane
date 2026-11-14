"""Railway Infrastructure Orchestrator & Durable Workflow Engine.

Engineered for high-leverage product engineering:
- Interactive 2D canvas infrastructure topology resolution (DAG dependency ordering, cycle detection).
- Nixpacks build plan generation and containerization contract validation.
- Temporal-style durable regional volume migration workflow with automated compensation rollback (Saga pattern).
- High-throughput ClickHouse telemetry buffer for 1B+ logs/day micro-batching.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple


class NodeType(str, Enum):
    SERVICE = "SERVICE"
    DATABASE = "DATABASE"
    VOLUME = "VOLUME"
    CRON_JOB = "CRON_JOB"


class WorkflowStatus(str, Enum):
    PENDING = "PENDING"
    PLANNING = "PLANNING"
    SNAPSHOT_TAKEN = "SNAPSHOT_TAKEN"
    VOLUME_DETACHED = "VOLUME_DETACHED"
    DATA_REPLICATED = "DATA_REPLICATED"
    VOLUME_ATTACHED = "VOLUME_ATTACHED"
    HEALTHCHECK_PASSED = "HEALTHCHECK_PASSED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    COMPENSATED = "COMPENSATED"


@dataclass
class CanvasCoordinate:
    x: float
    y: float

    def distance_to(self, other: CanvasCoordinate) -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5


@dataclass
class InfrastructureNode:
    id: str
    name: str
    node_type: NodeType
    region: str
    environment_variables: Dict[str, str] = field(default_factory=dict)
    dependencies: Set[str] = field(default_factory=set)
    canvas_pos: CanvasCoordinate = field(default_factory=lambda: CanvasCoordinate(0.0, 0.0))

    def mask_secrets(self) -> Dict[str, str]:
        """Mask sensitive credentials for dashboard presentation."""
        masked = {}
        for k, v in self.environment_variables.items():
            k_upper = k.upper()
            if any(term in k_upper for term in ("KEY", "SECRET", "PASSWORD", "TOKEN", "AUTH", "DATABASE", "URL")):
                if "://" in v and "@" in v:
                    proto, rest = v.split("://", 1)
                    creds, host = rest.split("@", 1)
                    masked[k] = f"{proto}://••••••••@{host}"
                elif len(v) >= 4:
                    masked[k] = f"••••••••{v[-4:]}"
                else:
                    masked[k] = "••••••••"
            else:
                masked[k] = v
        return masked


class InfrastructureTopologyGraph:
    """Represents a project canvas topology with DAG dependency resolution."""

    def __init__(self, project_id: str) -> None:
        self.project_id = project_id
        self.nodes: Dict[str, InfrastructureNode] = {}

    def add_node(self, node: InfrastructureNode) -> None:
        if node.id in self.nodes:
            raise ValueError(f"Node with id {node.id} already exists")
        self.nodes[node.id] = node

    def add_dependency(self, source_id: str, depends_on_id: str) -> None:
        if source_id not in self.nodes or depends_on_id not in self.nodes:
            raise KeyError("Both source and target nodes must exist in topology")
        if source_id == depends_on_id:
            raise ValueError("A node cannot depend on itself")
        self.nodes[source_id].dependencies.add(depends_on_id)
        if self.has_cycle():
            self.nodes[source_id].dependencies.remove(depends_on_id)
            raise ValueError(f"Adding dependency from {source_id} to {depends_on_id} creates a circular cycle")

    def has_cycle(self) -> bool:
        visited: Dict[str, int] = {}  # 0: visiting, 1: visited

        def dfs(node_id: str) -> bool:
            visited[node_id] = 0
            for dep in self.nodes[node_id].dependencies:
                if visited.get(dep) == 0:
                    return True
                if dep not in visited and dfs(dep):
                    return True
            visited[node_id] = 1
            return False

        for node_id in self.nodes:
            if node_id not in visited:
                if dfs(node_id):
                    return True
        return False

    def get_deployment_order(self) -> List[str]:
        """Topological sort returning deterministic startup sequence."""
        if self.has_cycle():
            raise ValueError("Cannot compute deployment order for cyclical topology")

        in_degree: Dict[str, int] = {k: 0 for k in self.nodes}
        adj: Dict[str, List[str]] = {k: [] for k in self.nodes}

        for node_id, node in self.nodes.items():
            for dep in node.dependencies:
                adj[dep].append(node_id)
                in_degree[node_id] += 1

        queue = [k for k, deg in in_degree.items() if deg == 0]
        queue.sort()  # deterministic tie-breaking
        order = []

        while queue:
            curr = queue.pop(0)
            order.append(curr)
            for neighbor in adj[curr]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
            queue.sort()

        if len(order) != len(self.nodes):
            raise RuntimeError("Topological order mismatch — unresolved dependencies")
        return order


@dataclass
class NixpacksBuildPlan:
    name: str
    providers: List[str]
    pkgs: List[str]
    apt_pkgs: List[str]
    build_cmd: str
    start_cmd: str

    def generate_plan_hash(self) -> str:
        payload = json.dumps(
            {
                "name": self.name,
                "providers": sorted(self.providers),
                "pkgs": sorted(self.pkgs),
                "apt_pkgs": sorted(self.apt_pkgs),
                "build_cmd": self.build_cmd,
                "start_cmd": self.start_cmd,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def validate(self) -> bool:
        if not self.start_cmd.strip():
            raise ValueError("Nixpacks plan must specify a non-empty start_cmd")
        if not self.providers:
            raise ValueError("At least one language/framework provider must be detected")
        return True


@dataclass
class VolumeMigrationSpec:
    service_id: str
    volume_id: str
    source_region: str
    target_region: str
    volume_size_gb: int


class TemporalVolumeMigrationWorkflow:
    """Durable multi-region volume migration workflow with automated compensation rollback."""

    def __init__(self, spec: VolumeMigrationSpec) -> None:
        self.spec = spec
        self.workflow_id = f"wflow-mig-{spec.volume_id}-{int(datetime.now(timezone.utc).timestamp())}"
        self.status = WorkflowStatus.PENDING
        self.execution_log: List[Tuple[WorkflowStatus, str]] = []
        self.snapshot_id: Optional[str] = None
        self.replicated_volume_id: Optional[str] = None

    def _record(self, status: WorkflowStatus, message: str) -> None:
        self.status = status
        self.execution_log.append((status, message))

    async def execute(self, simulate_failure_step: Optional[WorkflowStatus] = None) -> bool:
        """Executes the migration saga. Automatically rolls back if any step fails."""
        try:
            self._record(
                WorkflowStatus.PLANNING,
                f"Validating regions: {self.spec.source_region} -> {self.spec.target_region}",
            )
            if self.spec.source_region == self.spec.target_region:
                raise ValueError("Source and target regions must differ for cross-region migration")

            # Step 1: Snapshot
            hash_prefix = hashlib.md5(self.workflow_id.encode()).hexdigest()[:8]
            self.snapshot_id = f"snap-{self.spec.volume_id}-{hash_prefix}"
            self._record(
                WorkflowStatus.SNAPSHOT_TAKEN,
                f"Snapshot {self.snapshot_id} created in {self.spec.source_region}",
            )
            if simulate_failure_step == WorkflowStatus.SNAPSHOT_TAKEN:
                raise RuntimeError("Simulated snapshot IO failure")

            # Step 2: Detach
            self._record(
                WorkflowStatus.VOLUME_DETACHED,
                f"Volume {self.spec.volume_id} unmounted from {self.spec.service_id}",
            )
            if simulate_failure_step == WorkflowStatus.VOLUME_DETACHED:
                raise RuntimeError("Simulated volume unmount timeout")

            # Step 3: Replicate
            target_hash = hashlib.md5(self.snapshot_id.encode()).hexdigest()[:8]
            self.replicated_volume_id = f"vol-tgt-{target_hash}"
            self._record(
                WorkflowStatus.DATA_REPLICATED,
                f"Replicated {self.spec.volume_size_gb}GB to {self.spec.target_region} as {self.replicated_volume_id}",
            )
            if simulate_failure_step == WorkflowStatus.DATA_REPLICATED:
                raise RuntimeError("Simulated cross-region network partition")

            # Step 4: Attach
            self._record(
                WorkflowStatus.VOLUME_ATTACHED,
                f"Attached {self.replicated_volume_id} to service {self.spec.service_id} in {self.spec.target_region}",
            )
            if simulate_failure_step == WorkflowStatus.VOLUME_ATTACHED:
                raise RuntimeError("Simulated target mount failure")

            # Step 5: Health Check
            self._record(WorkflowStatus.HEALTHCHECK_PASSED, "Container filesystem verified readable & writeable")
            self._record(WorkflowStatus.COMPLETED, "Migration workflow succeeded; promoting new region primary")
            return True

        except Exception as exc:
            self._record(WorkflowStatus.FAILED, f"Workflow failed at {self.status.value}: {str(exc)}")
            await self._compensate()
            return False

    async def _compensate(self) -> None:
        """Saga compensation rollback logic restoring previous cluster state."""
        # Clean up partially created target volume if exists
        if self.replicated_volume_id:
            self._record(WorkflowStatus.COMPENSATED, f"Cleaning up orphaned target volume {self.replicated_volume_id}")
            self.replicated_volume_id = None
        # Re-attach original source volume if detached
        self._record(
            WorkflowStatus.COMPENSATED,
            f"Re-attaching source volume {self.spec.volume_id} in {self.spec.source_region}",
        )
        self.status = WorkflowStatus.COMPENSATED


@dataclass
class ClickHouseLogEvent:
    timestamp_ns: int
    project_id: str
    service_id: str
    deployment_id: str
    level: str
    message: str


class ClickHouseLogBuffer:
    """High-throughput micro-batching log buffer simulating Railway's 1B+ logs/day engine."""

    def __init__(self, batch_size: int = 500, flush_interval_sec: float = 1.0) -> None:
        self.batch_size = batch_size
        self.flush_interval_sec = flush_interval_sec
        self.buffer: List[ClickHouseLogEvent] = []
        self.flushed_batches: List[List[ClickHouseLogEvent]] = []
        self._lock = asyncio.Lock()

    async def emit(self, event: ClickHouseLogEvent) -> Optional[int]:
        async with self._lock:
            self.buffer.append(event)
            if len(self.buffer) >= self.batch_size:
                return await self._flush_internal()
            return None

    async def flush(self) -> int:
        async with self._lock:
            return await self._flush_internal()

    async def _flush_internal(self) -> int:
        if not self.buffer:
            return 0
        batch = list(self.buffer)
        self.buffer.clear()
        self.flushed_batches.append(batch)
        return len(batch)

    def query(self, service_id: str, level: Optional[str] = None) -> List[ClickHouseLogEvent]:
        results = []
        for batch in self.flushed_batches:
            for ev in batch:
                if ev.service_id == service_id:
                    if level is None or ev.level.upper() == level.upper():
                        results.append(ev)
        return results
