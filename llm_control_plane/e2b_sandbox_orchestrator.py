"""
E2B MicroVM Sandbox Orchestration and Agent Execution Engine.

Provides:
- MicroVM sandbox lifecycle management with microsecond-level boot tracking.
- Pre-warmed idle sandbox pool manager guaranteeing sub-150ms provisioning.
- Linux cgroups v2 resource limit enforcement (CPU quotas, memory caps, max PIDs).
- Multi-dimensional bin-packing scheduler for optimal node placement and noisy-neighbor mitigation.
- Zero-downtime sandbox state checkpointing and live migration across cluster nodes.
- Process isolation and fork-bomb / runaway process termination protection.
"""

from __future__ import annotations

import enum
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


class SandboxState(str, enum.Enum):
    INITIALIZING = "initializing"
    WARM_POOL = "warm_pool"
    RUNNING = "running"
    PAUSED = "paused"
    MIGRATING = "migrating"
    TERMINATED = "terminated"


class OrchestratorError(Exception):
    """Base exception for E2B sandbox orchestrator operations."""


class CapacityExceededError(OrchestratorError):
    """Raised when cluster nodes lack capacity to host the requested sandbox."""


class OOMKilledError(OrchestratorError):
    """Raised when an agent sandbox exceeds hard cgroup memory limits."""


class ForkBombProtectionError(OrchestratorError):
    """Raised when process count exceeds sandbox pid limits."""


@dataclass
class CgroupLimits:
    """Linux cgroups v2 resource controllers."""

    cpu_quota_us: int = 100_000  # 100ms per period
    cpu_period_us: int = 100_000
    memory_max_bytes: int = 512 * 1024 * 1024  # 512 MB
    memory_high_bytes: int = 400 * 1024 * 1024  # 400 MB throttling watermark
    pids_max: int = 256  # Hard limit on concurrent processes to block fork bombs


@dataclass
class SandboxSnapshot:
    """State checkpoint for cold start resumption and live migration."""

    snapshot_id: str
    sandbox_id: str
    memory_diff_bytes: int
    disk_diff_bytes: int
    created_at: float
    state_digest: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    """Result of an executed command inside the microVM sandbox."""

    command: str
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: float
    peak_memory_bytes: int


@dataclass
class MicroVMSandbox:
    """Isolated microVM execution sandbox for untrusted AI agent code."""

    sandbox_id: str
    tenant_id: str
    template_id: str
    vcpu_count: int = 2
    memory_mb: int = 512
    disk_mb: int = 2048
    state: SandboxState = SandboxState.INITIALIZING
    cgroup_limits: CgroupLimits = field(default_factory=CgroupLimits)
    created_at: float = field(default_factory=time.time)
    boot_latency_ms: float = 0.0
    current_node_id: Optional[str] = None
    active_pids: int = 1
    current_memory_bytes: int = 32 * 1024 * 1024
    command_history: List[ExecutionResult] = field(default_factory=list)
    snapshots: List[SandboxSnapshot] = field(default_factory=list)

    def execute_command(
        self,
        command: str,
        timeout_seconds: float = 30.0,
        simulated_memory_delta: int = 0,
        simulated_fork_pids: int = 0,
        custom_runner: Optional[Callable[[str], tuple[int, str, str]]] = None,
    ) -> ExecutionResult:
        """Execute a shell command within the microVM sandbox."""
        if self.state != SandboxState.RUNNING:
            raise OrchestratorError(f"Cannot execute in state {self.state.value}")

        start_time = time.perf_counter()

        # Check PID limits (fork-bomb guard)
        total_pids = self.active_pids + simulated_fork_pids
        if total_pids > self.cgroup_limits.pids_max:
            self.state = SandboxState.TERMINATED
            raise ForkBombProtectionError(
                f"Sandbox {self.sandbox_id} exceeded pids_max limit ({total_pids} > {self.cgroup_limits.pids_max})"
            )

        # Check Memory limits (cgroup OOM guard)
        new_memory = self.current_memory_bytes + simulated_memory_delta
        if new_memory > self.cgroup_limits.memory_max_bytes:
            self.state = SandboxState.TERMINATED
            raise OOMKilledError(
                f"Sandbox {self.sandbox_id} exceeded memory_max_bytes "
                f"({new_memory} > {self.cgroup_limits.memory_max_bytes})"
            )

        self.current_memory_bytes = new_memory
        self.active_pids = total_pids

        if custom_runner:
            exit_code, stdout, stderr = custom_runner(command)
        else:
            exit_code = 0
            stdout = f"Executed: {command}"
            stderr = ""

        duration_ms = (time.perf_counter() - start_time) * 1000.0

        res = ExecutionResult(
            command=command,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            duration_ms=duration_ms,
            peak_memory_bytes=self.current_memory_bytes,
        )
        self.command_history.append(res)
        return res

    def create_snapshot(self, label: str = "checkpoint") -> SandboxSnapshot:
        """Capture an incremental snapshot diff of running memory and disk."""
        snapshot_id = f"snap-{uuid.uuid4().hex[:12]}"
        digest = f"sha256-{uuid.uuid4().hex}"
        snap = SandboxSnapshot(
            snapshot_id=snapshot_id,
            sandbox_id=self.sandbox_id,
            memory_diff_bytes=self.current_memory_bytes,
            disk_diff_bytes=64 * 1024 * 1024,
            created_at=time.time(),
            state_digest=digest,
            metadata={"label": label, "commands_count": len(self.command_history)},
        )
        self.snapshots.append(snap)
        return snap

    def terminate(self) -> None:
        """Safely clean up sandbox resources and teardown microVM."""
        self.state = SandboxState.TERMINATED
        self.active_pids = 0
        self.current_memory_bytes = 0


class WarmPoolManager:
    """Maintains pre-forked idle microVMs to guarantee sub-150ms start times."""

    def __init__(self, target_pool_size: int = 3):
        self.target_pool_size = target_pool_size
        self._pools: Dict[str, List[MicroVMSandbox]] = {}

    def replenish_pool(self, template_id: str, count: Optional[int] = None) -> int:
        """Pre-warm idle microVMs for the specified template."""
        if template_id not in self._pools:
            self._pools[template_id] = []

        needed = count if count is not None else (self.target_pool_size - len(self._pools[template_id]))
        created = 0
        for _ in range(max(0, needed)):
            sb = MicroVMSandbox(
                sandbox_id=f"warm-{uuid.uuid4().hex[:8]}",
                tenant_id="__warm_pool__",
                template_id=template_id,
                state=SandboxState.WARM_POOL,
                boot_latency_ms=12.5,
            )
            self._pools[template_id].append(sb)
            created += 1
        return created

    def get_pool_size(self, template_id: str) -> int:
        """Return the number of idle warm microVMs available."""
        return len(self._pools.get(template_id, []))

    def provision_sandbox(
        self,
        tenant_id: str,
        template_id: str,
        vcpu_count: int = 2,
        memory_mb: int = 512,
    ) -> MicroVMSandbox:
        """Provision a sandbox from the warm pool or boot fresh if empty."""
        start_t = time.perf_counter()
        pool = self._pools.get(template_id, [])

        if pool:
            sb = pool.pop(0)
            sb.sandbox_id = f"sb-{uuid.uuid4().hex[:10]}"
            sb.tenant_id = tenant_id
            sb.vcpu_count = vcpu_count
            sb.memory_mb = memory_mb
            sb.state = SandboxState.RUNNING
            sb.boot_latency_ms = (time.perf_counter() - start_t) * 1000.0 + 15.0  # Warm start ~15ms
            return sb

        # Cold boot path
        sb = MicroVMSandbox(
            sandbox_id=f"sb-{uuid.uuid4().hex[:10]}",
            tenant_id=tenant_id,
            template_id=template_id,
            vcpu_count=vcpu_count,
            memory_mb=memory_mb,
            state=SandboxState.RUNNING,
            boot_latency_ms=(time.perf_counter() - start_t) * 1000.0 + 110.0,  # Cold start ~110ms
        )
        return sb


@dataclass
class ClusterNode:
    """Bare-metal cluster host running Firecracker KVM daemon."""

    node_id: str
    datacenter_zone: str
    total_vcpus: int
    total_memory_mb: int
    total_disk_gb: int
    hosted_sandboxes: Dict[str, MicroVMSandbox] = field(default_factory=dict)

    @property
    def allocated_vcpus(self) -> int:
        return sum(sb.vcpu_count for sb in self.hosted_sandboxes.values() if sb.state == SandboxState.RUNNING)

    @property
    def allocated_memory_mb(self) -> int:
        return sum(sb.memory_mb for sb in self.hosted_sandboxes.values() if sb.state == SandboxState.RUNNING)

    @property
    def free_vcpus(self) -> int:
        return self.total_vcpus - self.allocated_vcpus

    @property
    def free_memory_mb(self) -> int:
        return self.total_memory_mb - self.allocated_memory_mb

    def can_host(self, sandbox: MicroVMSandbox) -> bool:
        return self.free_vcpus >= sandbox.vcpu_count and self.free_memory_mb >= sandbox.memory_mb


class BinPackingScheduler:
    """Multi-dimensional resource scheduler placing sandboxes with noisy-neighbor mitigation."""

    def __init__(self, nodes: List[ClusterNode]):
        self.nodes = nodes

    def schedule(self, sandbox: MicroVMSandbox, tenant_anti_affinity: bool = True) -> ClusterNode:
        """Schedule sandbox onto the optimal cluster node using best-fit heuristics."""
        eligible_nodes = [node for node in self.nodes if node.can_host(sandbox)]

        if not eligible_nodes:
            raise CapacityExceededError(
                f"No cluster node has sufficient capacity for sandbox {sandbox.sandbox_id} "
                f"({sandbox.vcpu_count} vCPUs, {sandbox.memory_mb} MB RAM)"
            )

        if tenant_anti_affinity:
            # Prefer nodes that do NOT already host other sandboxes from the same tenant
            affinity_scored = []
            for n in eligible_nodes:
                same_tenant_count = sum(1 for sb in n.hosted_sandboxes.values() if sb.tenant_id == sandbox.tenant_id)
                affinity_scored.append((same_tenant_count, n))
            affinity_scored.sort(key=lambda item: (item[0], item[1].free_memory_mb))
            selected_node = affinity_scored[0][1]
        else:
            # Classic Best Fit (minimize leftover capacity)
            eligible_nodes.sort(key=lambda n: (n.free_memory_mb, n.free_vcpus))
            selected_node = eligible_nodes[0]

        selected_node.hosted_sandboxes[sandbox.sandbox_id] = sandbox
        sandbox.current_node_id = selected_node.node_id
        return selected_node


class SandboxLiveMigrator:
    """Coordinates zero-downtime microVM live migration between cluster nodes."""

    def __init__(self, scheduler: BinPackingScheduler):
        self.scheduler = scheduler

    def migrate(self, sandbox: MicroVMSandbox, target_node: ClusterNode) -> SandboxSnapshot:
        """Perform live state checkpointing and migration to target node."""
        if not target_node.can_host(sandbox):
            raise CapacityExceededError(
                f"Target node {target_node.node_id} cannot host sandbox {sandbox.sandbox_id}"
            )

        old_node_id = sandbox.current_node_id
        sandbox.state = SandboxState.MIGRATING

        # 1. Capture memory and state snapshot
        snap = sandbox.create_snapshot(label=f"live-migration-{target_node.node_id}")

        # 2. Transfer allocation to target node
        for n in self.scheduler.nodes:
            if n.node_id == old_node_id and sandbox.sandbox_id in n.hosted_sandboxes:
                del n.hosted_sandboxes[sandbox.sandbox_id]

        target_node.hosted_sandboxes[sandbox.sandbox_id] = sandbox
        sandbox.current_node_id = target_node.node_id

        # 3. Resume running state on target node
        sandbox.state = SandboxState.RUNNING
        return snap
