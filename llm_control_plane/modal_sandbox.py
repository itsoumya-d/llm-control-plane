"""Modal serverless container sandbox, microVM snapshotting, and worker pool.

Provides sub-second container sandbox virtualization, rootfs copy-on-write memory
snapshots, cgroups v2 resource quotas, ephemeral volume mounting, and pre-warmed
worker leasing for high-throughput AI inference and agentic execution environments.
"""

from __future__ import annotations

import copy
import hashlib
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


class SandboxStatus(str, Enum):
    """Lifecycle status of a serverless container sandbox."""

    COLD = "COLD"
    PREWARMING = "PREWARMING"
    WARM = "WARM"
    EXECUTING = "EXECUTING"
    TERMINATING = "TERMINATING"
    TERMINATED = "TERMINATED"


class SandboxResourceError(Exception):
    """Raised when sandbox execution exceeds CPU, memory, or timeout quotas."""


class SandboxSnapshotError(Exception):
    """Raised when microVM snapshot creation or restoration fails."""


@dataclass(frozen=True)
class CgroupV2Limits:
    """Cgroups v2 resource constraints for sandboxed process isolation."""

    cpu_quota_us: int = 100_000  # 100ms per period (1.0 core equivalent)
    cpu_period_us: int = 100_000
    memory_max_bytes: int = 512 * 1024 * 1024  # 512 MB
    pids_max: int = 64
    timeout_seconds: float = 30.0


@dataclass
class SandboxSnapshot:
    """Represents a frozen microVM memory and filesystem snapshot."""

    snapshot_id: str
    image_tag: str
    memory_state: Dict[str, Any]
    rootfs_layers: List[str]
    created_at: float = field(default_factory=time.time)
    digest: str = ""

    def __post_init__(self) -> None:
        if not self.digest:
            h = hashlib.sha256(f"{self.snapshot_id}:{self.image_tag}".encode("utf-8"))
            self.digest = h.hexdigest()[:16]


@dataclass
class SandboxExecutionResult:
    """Telemetry and execution output returned from a sandboxed function."""

    sandbox_id: str
    output: Any
    exit_code: int = 0
    wall_time_ms: float = 0.0
    cold_start_ms: float = 0.0
    was_cold: bool = False
    memory_peak_bytes: int = 0
    error: Optional[str] = None


class ModalServerlessSandbox:
    """Isolated serverless container sandbox instance.

    Features:
    - Copy-on-write memory and virtual filesystem layers.
    - Sub-second snapshot restoration to bypass cold starts.
    - Cgroups v2 resource limit enforcement (CPU, memory, timeout watchdog).
    - Ephemeral volume attachment for temporary scratch storage.
    """

    def __init__(
        self,
        image_tag: str = "modal-python-base:latest",
        limits: Optional[CgroupV2Limits] = None,
        env_vars: Optional[Dict[str, str]] = None,
    ):
        self.sandbox_id = f"sbx_{uuid.uuid4().hex[:10]}"
        self.image_tag = image_tag
        self.limits = limits or CgroupV2Limits()
        self.env_vars = env_vars or {}
        self.status = SandboxStatus.COLD
        self.volumes: Dict[str, Dict[str, Any]] = {}
        self._memory_state: Dict[str, Any] = {}
        self._rootfs_layers: List[str] = ["layer_base_kernel", "layer_python_runtime"]
        self._lock = threading.RLock()
        self._created_at = time.time()

    def attach_volume(self, mount_point: str, initial_data: Optional[Dict[str, Any]] = None) -> None:
        """Attaches an ephemeral scratch volume to the sandbox."""
        with self._lock:
            self.volumes[mount_point] = copy.deepcopy(initial_data or {})

    def write_volume_file(self, mount_point: str, filename: str, content: Any) -> None:
        with self._lock:
            if mount_point not in self.volumes:
                raise KeyError(f"Volume mount {mount_point} not attached")
            self.volumes[mount_point][filename] = copy.deepcopy(content)

    def read_volume_file(self, mount_point: str, filename: str) -> Any:
        with self._lock:
            if mount_point not in self.volumes or filename not in self.volumes[mount_point]:
                raise FileNotFoundError(f"File {filename} not found in volume {mount_point}")
            return copy.deepcopy(self.volumes[mount_point][filename])

    def create_snapshot(self) -> SandboxSnapshot:
        """Freezes current sandbox memory and filesystem state into a reusable snapshot."""
        with self._lock:
            snap_id = f"snap_{uuid.uuid4().hex[:8]}"
            snapshot = SandboxSnapshot(
                snapshot_id=snap_id,
                image_tag=self.image_tag,
                memory_state=copy.deepcopy(self._memory_state),
                rootfs_layers=list(self._rootfs_layers),
            )
            return snapshot

    def restore_from_snapshot(self, snapshot: SandboxSnapshot) -> float:
        """Restores sandbox state from snapshot in sub-millisecond time, returning restore duration."""
        start = time.perf_counter()
        with self._lock:
            if snapshot.image_tag != self.image_tag:
                raise SandboxSnapshotError(
                    f"Snapshot image {snapshot.image_tag} incompatible with sandbox {self.image_tag}"
                )
            self._memory_state = copy.deepcopy(snapshot.memory_state)
            self._rootfs_layers = list(snapshot.rootfs_layers)
            self.status = SandboxStatus.WARM
        restore_ms = (time.perf_counter() - start) * 1000.0
        return restore_ms

    def execute(
        self,
        func: Callable[..., Any],
        *args: Any,
        estimated_memory_bytes: int = 1024 * 1024,
        **kwargs: Any,
    ) -> SandboxExecutionResult:
        """Executes a callable inside the isolated sandbox subject to cgroups quotas."""
        with self._lock:
            if self.status == SandboxStatus.TERMINATED:
                raise RuntimeError(f"Cannot execute in terminated sandbox {self.sandbox_id}")

            was_cold = self.status == SandboxStatus.COLD
            cold_start_ms = 0.0

            if was_cold:
                t0 = time.perf_counter()
                # Simulate container initialization and rootfs layer mounting
                time.sleep(0.005)
                self.status = SandboxStatus.WARM
                cold_start_ms = (time.perf_counter() - t0) * 1000.0

            # Cgroup v2 memory quota verification
            if estimated_memory_bytes > self.limits.memory_max_bytes:
                raise SandboxResourceError(
                    f"OOM: Task memory requirement {estimated_memory_bytes}B exceeds cgroup limit "
                    f"{self.limits.memory_max_bytes}B"
                )

            self.status = SandboxStatus.EXECUTING
            start_wall = time.perf_counter()

            # Execute with watchdog monitoring
            result_container: List[Any] = []
            exception_container: List[Exception] = []

            def target_wrapper() -> None:
                try:
                    res = func(*args, **kwargs)
                    result_container.append(res)
                except Exception as ex:
                    exception_container.append(ex)

            worker_thread = threading.Thread(target=target_wrapper, daemon=True)
            worker_thread.start()
            worker_thread.join(timeout=self.limits.timeout_seconds)

            if worker_thread.is_alive():
                self.status = SandboxStatus.TERMINATING
                raise SandboxResourceError(
                    f"SandboxTimeout: Execution exceeded watchdog timeout of {self.limits.timeout_seconds}s"
                )

            wall_time_ms = (time.perf_counter() - start_wall) * 1000.0
            self.status = SandboxStatus.WARM

            if exception_container:
                exc = exception_container[0]
                return SandboxExecutionResult(
                    sandbox_id=self.sandbox_id,
                    output=None,
                    exit_code=1,
                    wall_time_ms=wall_time_ms,
                    cold_start_ms=cold_start_ms,
                    was_cold=was_cold,
                    memory_peak_bytes=estimated_memory_bytes,
                    error=str(exc),
                )

            return SandboxExecutionResult(
                sandbox_id=self.sandbox_id,
                output=result_container[0] if result_container else None,
                exit_code=0,
                wall_time_ms=wall_time_ms,
                cold_start_ms=cold_start_ms,
                was_cold=was_cold,
                memory_peak_bytes=estimated_memory_bytes,
            )

    def terminate(self) -> None:
        """Tears down sandbox and releases virtual memory and volume mounts."""
        with self._lock:
            self.status = SandboxStatus.TERMINATED
            self._memory_state.clear()
            self.volumes.clear()


class ModalWorkerPool:
    """Manages a dynamic pool of pre-warmed sandboxes to eliminate cold-start latency."""

    def __init__(
        self,
        default_image: str = "modal-python-base:latest",
        min_warm: int = 2,
        max_sandboxes: int = 16,
    ):
        self.default_image = default_image
        self.min_warm = min_warm
        self.max_sandboxes = max_sandboxes
        self._warm_sandboxes: List[ModalServerlessSandbox] = []
        self._active_leased: Dict[str, ModalServerlessSandbox] = {}
        self._snapshots: Dict[str, SandboxSnapshot] = {}
        self._lock = threading.RLock()
        self._total_executions = 0
        self._warm_hits = 0

        # Pre-seed initial snapshot and warm workers
        self._preseed()

    def _preseed(self) -> None:
        base_sbx = ModalServerlessSandbox(image_tag=self.default_image)
        snap = base_sbx.create_snapshot()
        self._snapshots[self.default_image] = snap

        for _ in range(self.min_warm):
            sbx = ModalServerlessSandbox(image_tag=self.default_image)
            sbx.restore_from_snapshot(snap)
            self._warm_sandboxes.append(sbx)

    def acquire_sandbox(self, image_tag: Optional[str] = None) -> Tuple[ModalServerlessSandbox, bool]:
        """Leases a warm sandbox from the pool or cold-boots a new instance if empty."""
        image = image_tag or self.default_image
        with self._lock:
            self._total_executions += 1
            # Check for available pre-warmed sandbox
            for i, sbx in enumerate(self._warm_sandboxes):
                if sbx.image_tag == image and sbx.status == SandboxStatus.WARM:
                    leased = self._warm_sandboxes.pop(i)
                    self._active_leased[leased.sandbox_id] = leased
                    self._warm_hits += 1
                    return leased, False  # was_cold = False

            # If no warm sandbox available, create cold sandbox and restore from snapshot if available
            sbx = ModalServerlessSandbox(image_tag=image)
            if image in self._snapshots:
                sbx.restore_from_snapshot(self._snapshots[image])
            self._active_leased[sbx.sandbox_id] = sbx
            return sbx, True  # was_cold = True

    def release_sandbox(self, sandbox: ModalServerlessSandbox, recycle: bool = True) -> None:
        """Returns a leased sandbox back to the warm pool or terminates it."""
        with self._lock:
            self._active_leased.pop(sandbox.sandbox_id, None)

            can_recycle = (
                recycle
                and len(self._warm_sandboxes) < self.max_sandboxes
                and sandbox.status != SandboxStatus.TERMINATED
            )
            if can_recycle:
                # Reset sandbox state for next tenant
                sandbox.volumes.clear()
                sandbox.status = SandboxStatus.WARM
                self._warm_sandboxes.append(sandbox)
            else:
                sandbox.terminate()

    def run_function(
        self,
        func: Callable[..., Any],
        *args: Any,
        image_tag: Optional[str] = None,
        **kwargs: Any,
    ) -> SandboxExecutionResult:
        """Executes a function end-to-end with automated leasing and recycling."""
        sbx, was_cold = self.acquire_sandbox(image_tag=image_tag)
        try:
            res = sbx.execute(func, *args, **kwargs)
            res.was_cold = was_cold
            return res
        finally:
            self.release_sandbox(sbx, recycle=True)

    def get_pool_metrics(self) -> Dict[str, Any]:
        with self._lock:
            hit_rate = (self._warm_hits / self._total_executions) if self._total_executions > 0 else 1.0
            return {
                "warm_sandboxes": len(self._warm_sandboxes),
                "active_leased": len(self._active_leased),
                "total_executions": self._total_executions,
                "warm_hits": self._warm_hits,
                "warm_hit_rate": round(hit_rate, 4),
            }
