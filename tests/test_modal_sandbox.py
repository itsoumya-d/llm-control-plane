"""Tests for Modal serverless container sandbox, microVM snapshotting, and worker pool."""

import time

import pytest

from llm_control_plane import (
    CgroupV2Limits,
    ModalServerlessSandbox,
    ModalWorkerPool,
    SandboxResourceError,
    SandboxSnapshot,
    SandboxSnapshotError,
    SandboxStatus,
)


def sample_add(a: int, b: int) -> int:
    return a + b


def sample_square(x: int) -> int:
    return x * x


def sample_sleep(seconds: float) -> str:
    time.sleep(seconds)
    return "done"


def sample_fail() -> None:
    raise ValueError("Explicit application failure inside container")


def test_sandbox_cold_and_warm_execution():
    sbx = ModalServerlessSandbox(image_tag="modal-python:3.12")
    assert sbx.status == SandboxStatus.COLD

    res1 = sbx.execute(sample_add, 10, 20)
    assert res1.output == 30
    assert res1.exit_code == 0
    assert res1.was_cold is True
    assert res1.cold_start_ms > 0
    assert sbx.status == SandboxStatus.WARM

    # Second execution is warm
    res2 = sbx.execute(sample_square, 5)
    assert res2.output == 25
    assert res2.was_cold is False
    assert res2.cold_start_ms == 0.0


def test_sandbox_snapshot_creation_and_restoration():
    sbx1 = ModalServerlessSandbox(image_tag="modal-pytorch:2.2")
    sbx1._memory_state["loaded_weights"] = {"layer_1": [0.1, 0.2, 0.3]}

    snap = sbx1.create_snapshot()
    assert isinstance(snap, SandboxSnapshot)
    assert snap.image_tag == "modal-pytorch:2.2"
    assert len(snap.digest) == 16

    # Restore in new cold sandbox
    sbx2 = ModalServerlessSandbox(image_tag="modal-pytorch:2.2")
    assert sbx2.status == SandboxStatus.COLD
    restore_ms = sbx2.restore_from_snapshot(snap)
    assert restore_ms >= 0.0
    assert sbx2.status == SandboxStatus.WARM
    assert sbx2._memory_state["loaded_weights"]["layer_1"] == [0.1, 0.2, 0.3]


def test_sandbox_incompatible_snapshot_error():
    sbx = ModalServerlessSandbox(image_tag="modal-python:3.12")
    snap = SandboxSnapshot(
        snapshot_id="snap_mismatch",
        image_tag="modal-custom-cuda:12.1",
        memory_state={},
        rootfs_layers=[],
    )
    with pytest.raises(SandboxSnapshotError, match="incompatible with sandbox"):
        sbx.restore_from_snapshot(snap)


def test_sandbox_cgroup_memory_limit_error():
    # 10 MB limit
    limits = CgroupV2Limits(memory_max_bytes=10 * 1024 * 1024)
    sbx = ModalServerlessSandbox(limits=limits)

    # Requesting 20 MB -> OOM
    with pytest.raises(SandboxResourceError, match="OOM"):
        sbx.execute(sample_add, 1, 2, estimated_memory_bytes=20 * 1024 * 1024)


def test_sandbox_watchdog_timeout_error():
    # Short timeout 0.1s
    limits = CgroupV2Limits(timeout_seconds=0.1)
    sbx = ModalServerlessSandbox(limits=limits)

    # Function sleeps for 0.5s -> watchdog triggers
    with pytest.raises(SandboxResourceError, match="SandboxTimeout"):
        sbx.execute(sample_sleep, 0.5)


def test_sandbox_ephemeral_volume_mount_and_io():
    sbx = ModalServerlessSandbox()
    sbx.attach_volume("/data/scratch", initial_data={"config.json": '{"mode": "test"}'})

    # Read initial
    val = sbx.read_volume_file("/data/scratch", "config.json")
    assert '{"mode": "test"}' in val

    # Write new file
    sbx.write_volume_file("/data/scratch", "weights.bin", b"\x00\x01\x02\x03")
    read_bin = sbx.read_volume_file("/data/scratch", "weights.bin")
    assert read_bin == b"\x00\x01\x02\x03"

    # Missing file error
    with pytest.raises(FileNotFoundError):
        sbx.read_volume_file("/data/scratch", "missing.bin")

    # Missing mount error
    with pytest.raises(KeyError):
        sbx.write_volume_file("/invalid/mount", "file.txt", "data")


def test_sandbox_error_capture():
    sbx = ModalServerlessSandbox()
    res = sbx.execute(sample_fail)
    assert res.exit_code == 1
    assert res.output is None
    assert "Explicit application failure inside container" in (res.error or "")


def test_sandbox_termination():
    sbx = ModalServerlessSandbox()
    sbx.attach_volume("/tmp", {"test.txt": "hello"})
    sbx.terminate()
    assert sbx.status == SandboxStatus.TERMINATED
    assert len(sbx.volumes) == 0

    with pytest.raises(RuntimeError, match="Cannot execute in terminated sandbox"):
        sbx.execute(sample_add, 1, 2)


def test_worker_pool_prewarming_and_leasing():
    pool = ModalWorkerPool(default_image="modal-python-base:latest", min_warm=2)
    metrics_initial = pool.get_pool_metrics()
    assert metrics_initial["warm_sandboxes"] == 2

    # Acquire warm sandbox
    sbx1, was_cold1 = pool.acquire_sandbox()
    assert was_cold1 is False
    assert sbx1.status == SandboxStatus.WARM

    # Acquire second warm sandbox
    sbx2, was_cold2 = pool.acquire_sandbox()
    assert was_cold2 is False

    # Pool now empty of warm sandboxes; acquiring 3rd should be cold
    sbx3, was_cold3 = pool.acquire_sandbox()
    assert was_cold3 is True

    # Release and recycle sbx1
    pool.release_sandbox(sbx1, recycle=True)
    metrics_after = pool.get_pool_metrics()
    assert metrics_after["warm_sandboxes"] >= 1
    assert metrics_after["warm_hits"] >= 2

    pool.release_sandbox(sbx2, recycle=False)
    pool.release_sandbox(sbx3, recycle=True)


def test_worker_pool_run_function_end_to_end():
    pool = ModalWorkerPool(min_warm=1)
    res = pool.run_function(sample_add, 100, 250)
    assert res.output == 350
    assert res.exit_code == 0
    assert res.was_cold is False

    metrics = pool.get_pool_metrics()
    assert metrics["total_executions"] >= 1
    assert metrics["warm_hit_rate"] > 0.0
