"""Simular Agent Infrastructure & Cloud Orchestrator.

Provides production-grade microVM sandbox lifecycle management, deterministic
action trajectory validation, hierarchical planner recovery, and asynchronous
multimodal telemetry buffering for autonomous computer-use agents (Agent S / OSWorld).
"""

from __future__ import annotations

import enum
import hashlib
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable


class SandboxStatus(str, enum.Enum):
    """Lifecycle states of an ephemeral microVM sandbox."""

    INITIALIZING = "INITIALIZING"
    WARM_READY = "WARM_READY"
    ACQUIRED = "ACQUIRED"
    SNAPSHOT_SAVED = "SNAPSHOT_SAVED"
    RESTORED = "RESTORED"
    TERMINATING = "TERMINATING"
    TERMINATED = "TERMINATED"


class ActionType(str, enum.Enum):
    """Supported low-level GUI and OS interaction actions."""

    CLICK = "CLICK"
    DOUBLE_CLICK = "DOUBLE_CLICK"
    DRAG_AND_DROP = "DRAG_AND_DROP"
    TYPE_TEXT = "TYPE_TEXT"
    KEY_PRESS = "KEY_PRESS"
    HOTKEY = "HOTKEY"
    SCROLL = "SCROLL"
    WAIT = "WAIT"


@dataclass(frozen=True)
class SandboxConfig:
    """Configuration for an isolated microVM sandbox instance."""

    sandbox_id: str
    tenant_id: str
    os_type: str = "Ubuntu 24.04 LTS"
    memory_mb: int = 2048
    vcpu_count: int = 2
    screen_width: int = 1920
    screen_height: int = 1080
    enable_network_egress: bool = False


@dataclass
class MicroVMSandbox:
    """Represents an active ephemeral microVM sandbox instance."""

    config: SandboxConfig
    status: SandboxStatus = SandboxStatus.INITIALIZING
    active_session_id: str | None = None
    created_at: float = field(default_factory=time.time)
    last_activity_at: float = field(default_factory=time.time)
    snapshots: dict[str, dict[str, Any]] = field(default_factory=dict)
    current_perceptual_hash: str = "0000000000000000"

    def provision(self) -> None:
        """Provisions virtual display, TAP networking, and transitions to WARM_READY."""
        if self.status != SandboxStatus.INITIALIZING:
            raise RuntimeError(f"Cannot provision sandbox in status {self.status}")
        self.status = SandboxStatus.WARM_READY
        self.last_activity_at = time.time()

    def acquire(self, session_id: str) -> None:
        """Acquires a pre-warmed sandbox for an active agent task session."""
        if self.status not in (SandboxStatus.WARM_READY, SandboxStatus.RESTORED):
            raise RuntimeError(f"Cannot acquire sandbox in status {self.status}")
        self.active_session_id = session_id
        self.status = SandboxStatus.ACQUIRED
        self.last_activity_at = time.time()

    def take_snapshot(self, snapshot_id: str, state_metadata: dict[str, Any] | None = None) -> str:
        """Captures a deterministic memory and screen snapshot for instant rollback."""
        if self.status not in (SandboxStatus.ACQUIRED, SandboxStatus.RESTORED):
            raise RuntimeError(f"Cannot snapshot sandbox in status {self.status}")
        meta = state_metadata or {}
        snapshot_record = {
            "snapshot_id": snapshot_id,
            "session_id": self.active_session_id,
            "captured_at": time.time(),
            "perceptual_hash": self.current_perceptual_hash,
            "metadata": meta,
        }
        self.snapshots[snapshot_id] = snapshot_record
        self.status = SandboxStatus.SNAPSHOT_SAVED
        self.last_activity_at = time.time()
        return snapshot_id

    def restore_snapshot(self, snapshot_id: str) -> bool:
        """Rolls back the sandbox state to a previously saved checkpoint."""
        if snapshot_id not in self.snapshots:
            return False
        rec = self.snapshots[snapshot_id]
        self.current_perceptual_hash = rec["perceptual_hash"]
        self.status = SandboxStatus.RESTORED
        self.last_activity_at = time.time()
        return True

    def teardown(self) -> None:
        """Wipes ephemeral disk and shuts down the microVM."""
        self.status = SandboxStatus.TERMINATING
        self.snapshots.clear()
        self.active_session_id = None
        self.status = SandboxStatus.TERMINATED


class MicroVMSandboxPool:
    """Maintains a warm pool of microVM sandboxes for sub-second agent starts."""

    def __init__(self, tenant_id: str, pool_size: int = 5) -> None:
        self.tenant_id = tenant_id
        self.pool_size = pool_size
        self._sandboxes: dict[str, MicroVMSandbox] = {}
        self._lock = threading.Lock()
        self._initialize_pool()

    def _initialize_pool(self) -> None:
        with self._lock:
            for i in range(self.pool_size):
                sid = f"vm-{self.tenant_id[:6]}-{i:03d}"
                cfg = SandboxConfig(sandbox_id=sid, tenant_id=self.tenant_id)
                vm = MicroVMSandbox(config=cfg)
                vm.provision()
                self._sandboxes[sid] = vm

    def acquire_sandbox(self, session_id: str) -> MicroVMSandbox:
        """Acquires an available warm sandbox from the pool."""
        with self._lock:
            for vm in self._sandboxes.values():
                if vm.status in (SandboxStatus.WARM_READY, SandboxStatus.RESTORED):
                    vm.acquire(session_id)
                    return vm
            # Pool exhausted: spawn an on-demand sandbox
            sid = f"vm-{self.tenant_id[:6]}-ondemand-{len(self._sandboxes):03d}"
            cfg = SandboxConfig(sandbox_id=sid, tenant_id=self.tenant_id)
            vm = MicroVMSandbox(config=cfg)
            vm.provision()
            vm.acquire(session_id)
            self._sandboxes[sid] = vm
            return vm

    def release_sandbox(self, sandbox_id: str) -> None:
        """Releases and recycles a sandbox back into warm status."""
        with self._lock:
            if sandbox_id in self._sandboxes:
                vm = self._sandboxes[sandbox_id]
                vm.teardown()
                # Re-provision for warm reuse
                vm.status = SandboxStatus.INITIALIZING
                vm.provision()

    @property
    def total_count(self) -> int:
        return len(self._sandboxes)

    @property
    def available_count(self) -> int:
        return sum(
            1
            for v in self._sandboxes.values()
            if v.status in (SandboxStatus.WARM_READY, SandboxStatus.RESTORED)
        )


@dataclass(frozen=True)
class ActionCommand:
    """Low-level GUI action command emitted by visual grounding planners."""

    action_type: ActionType
    x: int | None = None
    y: int | None = None
    drag_target_x: int | None = None
    drag_target_y: int | None = None
    text_content: str | None = None
    key_sequence: str | None = None
    scroll_amount: int | None = None


@dataclass(frozen=True)
class ValidationResult:
    """Result of action trajectory validation."""

    is_valid: bool
    reason: str = "PASS"
    safe_command: ActionCommand | None = None


class ActionTrajectoryValidator:
    """Validates action coordinates, bounds, click jitter, and security invariants."""

    def __init__(self, screen_width: int = 1920, screen_height: int = 1080) -> None:
        self.screen_width = screen_width
        self.screen_height = screen_height
        self._last_x: int | None = None
        self._last_y: int | None = None

    def validate(self, action: ActionCommand) -> ValidationResult:
        """Inspects action parameters against screen geometry and safety rules."""
        if action.action_type in (ActionType.CLICK, ActionType.DOUBLE_CLICK):
            if action.x is None or action.y is None:
                return ValidationResult(is_valid=False, reason="Click action missing coordinates")
            if not (0 <= action.x < self.screen_width and 0 <= action.y < self.screen_height):
                return ValidationResult(
                    is_valid=False,
                    reason=(
                        f"Coordinates ({action.x}, {action.y}) out of screen bounds "
                        f"({self.screen_width}x{self.screen_height})"
                    ),
                )
            self._last_x = action.x
            self._last_y = action.y
            return ValidationResult(is_valid=True, safe_command=action)

        if action.action_type == ActionType.DRAG_AND_DROP:
            if any(v is None for v in (action.x, action.y, action.drag_target_x, action.drag_target_y)):
                return ValidationResult(is_valid=False, reason="Drag-and-drop missing start or target coordinates")
            if not (0 <= action.x < self.screen_width and 0 <= action.y < self.screen_height):  # type: ignore[operator]
                return ValidationResult(is_valid=False, reason="Drag start coordinates out of bounds")
            if not (0 <= action.drag_target_x < self.screen_width and 0 <= action.drag_target_y < self.screen_height):  # type: ignore[operator]
                return ValidationResult(is_valid=False, reason="Drag target coordinates out of bounds")
            self._last_x = action.drag_target_x
            self._last_y = action.drag_target_y
            return ValidationResult(is_valid=True, safe_command=action)

        if action.action_type == ActionType.TYPE_TEXT:
            if action.text_content is None:
                return ValidationResult(is_valid=False, reason="Type text action missing text_content")
            # Enforce safety: block dangerous destructive terminal sequences if unescaped
            forbidden = ("rm -rf /", ":(){ :|:& };:", "dd if=/dev/zero")
            for f in forbidden:
                if f in action.text_content:
                    return ValidationResult(is_valid=False, reason=f"Destructive command detected: {f}")
            return ValidationResult(is_valid=True, safe_command=action)

        if action.action_type in (ActionType.KEY_PRESS, ActionType.HOTKEY):
            if not action.key_sequence:
                return ValidationResult(is_valid=False, reason="Hotkey action missing key_sequence")
            return ValidationResult(is_valid=True, safe_command=action)

        if action.action_type == ActionType.SCROLL:
            if action.scroll_amount is None:
                return ValidationResult(is_valid=False, reason="Scroll action missing scroll_amount")
            return ValidationResult(is_valid=True, safe_command=action)

        if action.action_type == ActionType.WAIT:
            return ValidationResult(is_valid=True, safe_command=action)

        return ValidationResult(is_valid=False, reason=f"Unsupported action type {action.action_type}")


@dataclass
class ExecutionStep:
    """A single step in a hierarchical computer-use plan."""

    step_index: int
    description: str
    command: ActionCommand
    checkpoint_required: bool = False
    expected_state_hash: str | None = None


@dataclass
class ExecutionReport:
    """Outcome of multi-step agent plan execution."""

    success: bool
    total_steps: int
    executed_steps: int
    rollbacks_triggered: int
    error_message: str | None = None
    step_latencies_ms: list[float] = field(default_factory=list)


class AgentExecutionCoordinator:
    """Orchestrates hierarchical agent step execution with rollback checkpoints."""

    def __init__(
        self,
        sandbox: MicroVMSandbox,
        validator: ActionTrajectoryValidator | None = None,
        executor_fn: Callable[[ActionCommand], bool] | None = None,
    ) -> None:
        self.sandbox = sandbox
        self.validator = validator or ActionTrajectoryValidator(
            screen_width=sandbox.config.screen_width,
            screen_height=sandbox.config.screen_height,
        )
        self.executor_fn = executor_fn or (lambda cmd: True)
        self.rollbacks_triggered = 0

    def execute_plan(self, plan: list[ExecutionStep]) -> ExecutionReport:
        """Executes a multi-step plan, enforcing validation and rollback on step failure."""
        executed = 0
        latencies: list[float] = []
        last_checkpoint: str | None = None

        for step in plan:
            t0 = time.time()
            if step.checkpoint_required:
                cid = f"cp-step-{step.step_index}"
                self.sandbox.take_snapshot(cid, {"step_index": step.step_index})
                last_checkpoint = cid

            val = self.validator.validate(step.command)
            if not val.is_valid:
                # Trigger rollback if checkpoint exists
                if last_checkpoint and self.sandbox.restore_snapshot(last_checkpoint):
                    self.rollbacks_triggered += 1
                return ExecutionReport(
                    success=False,
                    total_steps=len(plan),
                    executed_steps=executed,
                    rollbacks_triggered=self.rollbacks_triggered,
                    error_message=f"Validation failed on step {step.step_index}: {val.reason}",
                    step_latencies_ms=latencies,
                )

            # Execute safe command
            success = self.executor_fn(val.safe_command or step.command)
            if not success:
                if last_checkpoint and self.sandbox.restore_snapshot(last_checkpoint):
                    self.rollbacks_triggered += 1
                return ExecutionReport(
                    success=False,
                    total_steps=len(plan),
                    executed_steps=executed,
                    rollbacks_triggered=self.rollbacks_triggered,
                    error_message=f"Executor failed on step {step.step_index}",
                    step_latencies_ms=latencies,
                )

            executed += 1
            latencies.append((time.time() - t0) * 1000.0)

        return ExecutionReport(
            success=True,
            total_steps=len(plan),
            executed_steps=executed,
            rollbacks_triggered=self.rollbacks_triggered,
            step_latencies_ms=latencies,
        )


@dataclass(frozen=True)
class TelemetryEvent:
    """Multimodal telemetry record for an agent execution step."""

    event_id: str
    session_id: str
    step_index: int
    action_type: str
    latency_ms: float
    perceptual_hash: str
    token_cost: int
    status: str
    timestamp: float = field(default_factory=time.time)


class TelemetryRingBuffer:
    """High-throughput asynchronous ring buffer for multimodal agent event streams."""

    def __init__(self, capacity: int = 1000) -> None:
        self.capacity = capacity
        self._buffer: list[TelemetryEvent] = []
        self._lock = threading.Lock()
        self._total_enqueued = 0
        self._total_dropped = 0

    def enqueue(self, event: TelemetryEvent) -> bool:
        """Enqueues a telemetry event with drop-oldest overflow policy."""
        with self._lock:
            self._total_enqueued += 1
            if len(self._buffer) >= self.capacity:
                self._buffer.pop(0)
                self._total_dropped += 1
            self._buffer.append(event)
            return True

    def flush_batch(self, batch_size: int = 50) -> list[TelemetryEvent]:
        """Flushes a batch of buffered events for export to object storage / telemetry."""
        with self._lock:
            count = min(batch_size, len(self._buffer))
            batch = self._buffer[:count]
            self._buffer = self._buffer[count:]
            return batch

    def compute_trajectory_digest(self, session_id: str) -> str:
        """Computes a deterministic SHA-256 digest of session event sequence for evaluation."""
        with self._lock:
            events = [e for e in self._buffer if e.session_id == session_id]
        if not events:
            return hashlib.sha256(b"empty").hexdigest()
        digest = hashlib.sha256()
        for e in events:
            raw = f"{e.step_index}:{e.action_type}:{e.perceptual_hash}:{e.status}".encode()
            digest.update(raw)
        return digest.hexdigest()

    @property
    def current_size(self) -> int:
        with self._lock:
            return len(self._buffer)

    @property
    def total_enqueued(self) -> int:
        return self._total_enqueued

    @property
    def total_dropped(self) -> int:
        return self._total_dropped
