"""Unit tests for Simular Agent Infrastructure & Cloud Orchestrator."""

import concurrent.futures

import pytest

from llm_control_plane.simular_agent_infrastructure_orchestrator import (
    ActionCommand,
    ActionTrajectoryValidator,
    ActionType,
    AgentExecutionCoordinator,
    ExecutionStep,
    MicroVMSandbox,
    MicroVMSandboxPool,
    SandboxConfig,
    SandboxStatus,
    TelemetryEvent,
    TelemetryRingBuffer,
)


def test_sandbox_lifecycle_transitions():
    cfg = SandboxConfig(sandbox_id="vm-test-001", tenant_id="tenant-alpha")
    vm = MicroVMSandbox(config=cfg)

    assert vm.status == SandboxStatus.INITIALIZING

    # Cannot acquire before provisioning
    with pytest.raises(RuntimeError):
        vm.acquire("session-123")

    vm.provision()
    assert vm.status == SandboxStatus.WARM_READY

    vm.acquire("session-123")
    assert vm.status == SandboxStatus.ACQUIRED
    assert vm.active_session_id == "session-123"

    snap_id = vm.take_snapshot("snap-1", {"step": 0})
    assert snap_id == "snap-1"
    assert vm.status == SandboxStatus.SNAPSHOT_SAVED

    # Restore snapshot
    restored = vm.restore_snapshot("snap-1")
    assert restored is True
    assert vm.status == SandboxStatus.RESTORED

    # Non-existent snapshot
    assert vm.restore_snapshot("snap-invalid") is False

    vm.teardown()
    assert vm.status == SandboxStatus.TERMINATED
    assert vm.active_session_id is None


def test_sandbox_pool_management():
    pool = MicroVMSandboxPool(tenant_id="tenant-sg", pool_size=3)
    assert pool.total_count == 3
    assert pool.available_count == 3

    vm1 = pool.acquire_sandbox("sess-1")
    assert vm1.status == SandboxStatus.ACQUIRED
    assert pool.available_count == 2

    pool.acquire_sandbox("sess-2")
    pool.acquire_sandbox("sess-3")
    assert pool.available_count == 0

    # Exhausted pool triggers on-demand creation
    vm4 = pool.acquire_sandbox("sess-4")
    assert pool.total_count == 4
    assert vm4.active_session_id == "sess-4"

    # Release recycling
    pool.release_sandbox(vm1.config.sandbox_id)
    assert pool.available_count == 1
    assert vm1.status == SandboxStatus.WARM_READY


def test_action_trajectory_validator_bounds():
    val = ActionTrajectoryValidator(screen_width=1920, screen_height=1080)

    # Valid click
    res = val.validate(ActionCommand(action_type=ActionType.CLICK, x=100, y=200))
    assert res.is_valid is True
    assert res.reason == "PASS"

    # Out of bounds clicks
    res_oob_x = val.validate(ActionCommand(action_type=ActionType.CLICK, x=2000, y=500))
    assert res_oob_x.is_valid is False
    assert "out of screen bounds" in res_oob_x.reason

    res_neg = val.validate(ActionCommand(action_type=ActionType.CLICK, x=-10, y=500))
    assert res_neg.is_valid is False

    res_missing = val.validate(ActionCommand(action_type=ActionType.CLICK, x=None, y=100))
    assert res_missing.is_valid is False

    # Valid drag and drop
    drag_ok = val.validate(
        ActionCommand(
            action_type=ActionType.DRAG_AND_DROP,
            x=100,
            y=100,
            drag_target_x=500,
            drag_target_y=600,
        )
    )
    assert drag_ok.is_valid is True

    # Drag out of bounds
    drag_bad = val.validate(
        ActionCommand(
            action_type=ActionType.DRAG_AND_DROP,
            x=100,
            y=100,
            drag_target_x=2500,
            drag_target_y=600,
        )
    )
    assert drag_bad.is_valid is False


def test_action_trajectory_validator_safety_and_keys():
    val = ActionTrajectoryValidator()

    # Safe text
    safe_txt = val.validate(ActionCommand(action_type=ActionType.TYPE_TEXT, text_content="echo hello world"))
    assert safe_txt.is_valid is True

    # Destructive text
    bad_txt = val.validate(ActionCommand(action_type=ActionType.TYPE_TEXT, text_content="sudo rm -rf /"))
    assert bad_txt.is_valid is False
    assert "Destructive command detected" in bad_txt.reason

    # Missing text content
    missing_txt = val.validate(ActionCommand(action_type=ActionType.TYPE_TEXT, text_content=None))
    assert missing_txt.is_valid is False

    # Hotkey & Scroll
    hk_ok = val.validate(ActionCommand(action_type=ActionType.HOTKEY, key_sequence="Ctrl+C"))
    assert hk_ok.is_valid is True

    scroll_ok = val.validate(ActionCommand(action_type=ActionType.SCROLL, scroll_amount=-300))
    assert scroll_ok.is_valid is True

    wait_ok = val.validate(ActionCommand(action_type=ActionType.WAIT))
    assert wait_ok.is_valid is True


def test_agent_execution_coordinator_success():
    cfg = SandboxConfig(sandbox_id="vm-coord-1", tenant_id="t1")
    vm = MicroVMSandbox(config=cfg)
    vm.provision()
    vm.acquire("sess-coord")

    executed_cmds = []

    def fake_executor(cmd: ActionCommand) -> bool:
        executed_cmds.append(cmd.action_type)
        return True

    coord = AgentExecutionCoordinator(sandbox=vm, executor_fn=fake_executor)

    plan = [
        ExecutionStep(
            step_index=1,
            description="Click search box",
            command=ActionCommand(action_type=ActionType.CLICK, x=100, y=100),
            checkpoint_required=True,
        ),
        ExecutionStep(
            step_index=2,
            description="Type query",
            command=ActionCommand(action_type=ActionType.TYPE_TEXT, text_content="Simular Agent S"),
        ),
        ExecutionStep(
            step_index=3,
            description="Press enter",
            command=ActionCommand(action_type=ActionType.KEY_PRESS, key_sequence="Enter"),
        ),
    ]

    report = coord.execute_plan(plan)
    assert report.success is True
    assert report.total_steps == 3
    assert report.executed_steps == 3
    assert report.rollbacks_triggered == 0
    assert len(executed_cmds) == 3


def test_agent_execution_coordinator_rollback_on_failure():
    cfg = SandboxConfig(sandbox_id="vm-coord-2", tenant_id="t1")
    vm = MicroVMSandbox(config=cfg)
    vm.provision()
    vm.acquire("sess-coord-2")

    coord = AgentExecutionCoordinator(sandbox=vm)

    plan = [
        ExecutionStep(
            step_index=1,
            description="Step 1 with checkpoint",
            command=ActionCommand(action_type=ActionType.CLICK, x=200, y=200),
            checkpoint_required=True,
        ),
        ExecutionStep(
            step_index=2,
            description="Invalid out of bounds step",
            command=ActionCommand(action_type=ActionType.CLICK, x=9999, y=9999),
        ),
    ]

    report = coord.execute_plan(plan)
    assert report.success is False
    assert report.executed_steps == 1
    assert report.rollbacks_triggered == 1
    assert "out of screen bounds" in (report.error_message or "")
    assert vm.status == SandboxStatus.RESTORED


def test_telemetry_ring_buffer_and_digest():
    buf = TelemetryRingBuffer(capacity=5)

    for i in range(7):
        ev = TelemetryEvent(
            event_id=f"ev-{i}",
            session_id="session-telemetry",
            step_index=i,
            action_type="CLICK",
            latency_ms=12.5,
            perceptual_hash="hash-1234",
            token_cost=50,
            status="SUCCESS",
        )
        buf.enqueue(ev)

    assert buf.total_enqueued == 7
    assert buf.total_dropped == 2
    assert buf.current_size == 5

    digest = buf.compute_trajectory_digest("session-telemetry")
    assert isinstance(digest, str)
    assert len(digest) == 64

    # Batch flush
    batch = buf.flush_batch(batch_size=3)
    assert len(batch) == 3
    assert buf.current_size == 2


def test_telemetry_ring_buffer_concurrency():
    buf = TelemetryRingBuffer(capacity=500)

    def worker(worker_id: int):
        for i in range(50):
            ev = TelemetryEvent(
                event_id=f"w{worker_id}-{i}",
                session_id=f"sess-{worker_id}",
                step_index=i,
                action_type="CLICK",
                latency_ms=5.0,
                perceptual_hash="abc",
                token_cost=10,
                status="OK",
            )
            buf.enqueue(ev)

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(worker, wid) for wid in range(5)]
        for f in futures:
            f.result()

    assert buf.total_enqueued == 250
    assert buf.current_size == 250
