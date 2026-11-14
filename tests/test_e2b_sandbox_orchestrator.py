"""
Tests for E2B MicroVM Sandbox Orchestration Engine.
"""

from __future__ import annotations

import pytest

from llm_control_plane.e2b_sandbox_orchestrator import (
    BinPackingScheduler,
    CapacityExceededError,
    CgroupLimits,
    ClusterNode,
    ForkBombProtectionError,
    MicroVMSandbox,
    OOMKilledError,
    SandboxLiveMigrator,
    SandboxState,
    WarmPoolManager,
)


def test_warm_pool_provisioning_latency():
    """Verify warm pool pre-warming and sub-150ms provisioning."""
    manager = WarmPoolManager(target_pool_size=3)
    created = manager.replenish_pool("python-base", count=3)
    assert created == 3
    assert manager.get_pool_size("python-base") == 3

    sandbox = manager.provision_sandbox(
        tenant_id="tenant-123",
        template_id="python-base",
        vcpu_count=2,
        memory_mb=512,
    )
    assert sandbox.state == SandboxState.RUNNING
    assert sandbox.tenant_id == "tenant-123"
    assert sandbox.boot_latency_ms < 150.0
    assert manager.get_pool_size("python-base") == 2


def test_cold_boot_provisioning():
    """Verify cold boot provisioning succeeds when warm pool is depleted."""
    manager = WarmPoolManager(target_pool_size=0)
    assert manager.get_pool_size("rust-base") == 0

    sandbox = manager.provision_sandbox(
        tenant_id="tenant-456",
        template_id="rust-base",
        vcpu_count=4,
        memory_mb=1024,
    )
    assert sandbox.state == SandboxState.RUNNING
    assert sandbox.template_id == "rust-base"
    assert sandbox.vcpu_count == 4
    assert sandbox.memory_mb == 1024


def test_command_execution_success():
    """Verify command execution inside sandbox records output and timing."""
    sandbox = MicroVMSandbox(
        sandbox_id="sb-test-1",
        tenant_id="tenant-1",
        template_id="base",
        state=SandboxState.RUNNING,
    )

    res = sandbox.execute_command(
        "echo 'hello from e2b sandbox'",
        custom_runner=lambda cmd: (0, "hello from e2b sandbox\n", ""),
    )
    assert res.exit_code == 0
    assert "hello from e2b sandbox" in res.stdout
    assert len(sandbox.command_history) == 1
    assert sandbox.command_history[0].command == "echo 'hello from e2b sandbox'"


def test_cgroup_memory_limit_oom_kill():
    """Verify exceeding hard memory limit terminates sandbox with OOM error."""
    sandbox = MicroVMSandbox(
        sandbox_id="sb-oom",
        tenant_id="tenant-oom",
        template_id="base",
        state=SandboxState.RUNNING,
        cgroup_limits=CgroupLimits(memory_max_bytes=128 * 1024 * 1024),
    )

    # Attempt allocation exceeding 128 MB
    with pytest.raises(OOMKilledError) as exc_info:
        sandbox.execute_command(
            "stress-ng --vm 1 --vm-bytes 200M",
            simulated_memory_delta=200 * 1024 * 1024,
        )

    assert "exceeded memory_max_bytes" in str(exc_info.value)
    assert sandbox.state == SandboxState.TERMINATED


def test_fork_bomb_pid_limit_protection():
    """Verify exceeding max PID limits terminates sandbox with fork bomb error."""
    sandbox = MicroVMSandbox(
        sandbox_id="sb-fork",
        tenant_id="tenant-fork",
        template_id="base",
        state=SandboxState.RUNNING,
        cgroup_limits=CgroupLimits(pids_max=64),
    )

    with pytest.raises(ForkBombProtectionError) as exc_info:
        sandbox.execute_command(
            ":(){ :|:& };:",
            simulated_fork_pids=100,
        )

    assert "exceeded pids_max limit" in str(exc_info.value)
    assert sandbox.state == SandboxState.TERMINATED


def test_bin_packing_scheduler_placement():
    """Verify multi-dimensional scheduler places sandbox onto capable node."""
    node1 = ClusterNode("node-sf-1", "us-west-1a", total_vcpus=8, total_memory_mb=16384, total_disk_gb=500)
    node2 = ClusterNode("node-sf-2", "us-west-1b", total_vcpus=4, total_memory_mb=8192, total_disk_gb=250)

    scheduler = BinPackingScheduler(nodes=[node1, node2])

    sandbox = MicroVMSandbox(
        sandbox_id="sb-large",
        tenant_id="tenant-enterprise",
        template_id="gpu-ml",
        vcpu_count=6,
        memory_mb=12288,
        state=SandboxState.RUNNING,
    )

    selected = scheduler.schedule(sandbox)
    assert selected.node_id == "node-sf-1"
    assert sandbox.current_node_id == "node-sf-1"
    assert node1.allocated_vcpus == 6
    assert node1.allocated_memory_mb == 12288

    # Attempting to schedule a sandbox exceeding all available capacities raises CapacityExceededError
    huge_sb = MicroVMSandbox(
        sandbox_id="sb-huge",
        tenant_id="tenant-enterprise",
        template_id="gpu-ml",
        vcpu_count=16,
        memory_mb=32768,
        state=SandboxState.RUNNING,
    )
    with pytest.raises(CapacityExceededError):
        scheduler.schedule(huge_sb)


def test_tenant_anti_affinity_scheduling():
    """Verify scheduler spreads sandboxes of the same tenant across separate nodes."""
    node1 = ClusterNode("node-1", "zone-a", total_vcpus=16, total_memory_mb=32768, total_disk_gb=1000)
    node2 = ClusterNode("node-2", "zone-b", total_vcpus=16, total_memory_mb=32768, total_disk_gb=1000)

    scheduler = BinPackingScheduler(nodes=[node1, node2])

    sb1 = MicroVMSandbox("sb-1", "tenant-alpha", "base", vcpu_count=2, memory_mb=2048, state=SandboxState.RUNNING)
    sb2 = MicroVMSandbox("sb-2", "tenant-alpha", "base", vcpu_count=2, memory_mb=2048, state=SandboxState.RUNNING)

    n1 = scheduler.schedule(sb1, tenant_anti_affinity=True)
    n2 = scheduler.schedule(sb2, tenant_anti_affinity=True)

    assert n1.node_id != n2.node_id
    assert {n1.node_id, n2.node_id} == {"node-1", "node-2"}


def test_sandbox_live_migration():
    """Verify live migration checkpoints state and relocates sandbox between nodes."""
    node1 = ClusterNode("node-source", "zone-a", total_vcpus=8, total_memory_mb=16384, total_disk_gb=500)
    node2 = ClusterNode("node-target", "zone-b", total_vcpus=8, total_memory_mb=16384, total_disk_gb=500)

    scheduler = BinPackingScheduler(nodes=[node1, node2])
    migrator = SandboxLiveMigrator(scheduler)

    sandbox = MicroVMSandbox(
        "sb-migrating", "tenant-xyz", "base", vcpu_count=2, memory_mb=2048, state=SandboxState.RUNNING
    )
    scheduler.schedule(sandbox)
    assert sandbox.current_node_id == "node-source"

    # Execute a command prior to migration
    sandbox.execute_command("echo 'state-before-migration'")

    snap = migrator.migrate(sandbox, node2)
    assert snap.sandbox_id == "sb-migrating"
    assert sandbox.current_node_id == "node-target"
    assert sandbox.state == SandboxState.RUNNING
    assert "sb-migrating" in node2.hosted_sandboxes
    assert "sb-migrating" not in node1.hosted_sandboxes
    assert len(sandbox.command_history) == 1
