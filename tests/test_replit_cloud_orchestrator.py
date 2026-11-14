"""Unit and integration tests for Replit Cloud Orchestrator."""

from __future__ import annotations

import pytest

from llm_control_plane.replit_cloud_orchestrator import (
    ContainerSandbox,
    ContainerState,
    EnvironmentType,
    ReplitCloudOrchestrator,
    SSLStatus,
)


def test_warm_pool_prewarming() -> None:
    orch = ReplitCloudOrchestrator()
    warmed = orch.prewarm_containers(runtime="nodejs:20", count=3)
    assert warmed == 3
    assert len(orch.warm_pool) == 3
    assert all(s.runtime == "nodejs:20" for s in orch.warm_pool)
    assert all(s.state == ContainerState.RUNNING for s in orch.warm_pool)
    assert all(s.boot_latency_ms > 0.0 for s in orch.warm_pool)


def test_provision_sandbox_from_warm_pool() -> None:
    orch = ReplitCloudOrchestrator()
    orch.prewarm_containers(runtime="python:3.12", count=2)
    assert len(orch.warm_pool) == 2

    sbx = orch.provision_sandbox(
        project_id="prj-agent-alpha",
        environment=EnvironmentType.DEV,
        runtime="python:3.12",
        cpu_cores=4.0,
        memory_limit_mb=4096,
    )

    assert isinstance(sbx, ContainerSandbox)
    assert sbx.project_id == "prj-agent-alpha"
    assert sbx.environment == EnvironmentType.DEV
    assert sbx.runtime == "python:3.12"
    assert sbx.cpu_cores == 4.0
    assert sbx.memory_limit_mb == 4096
    assert sbx.state == ContainerState.RUNNING
    # Warm pool was consumed
    assert len(orch.warm_pool) == 1
    # Recorded in sandboxes map
    assert sbx.sandbox_id in orch.sandboxes


def test_provision_sandbox_cold_start() -> None:
    orch = ReplitCloudOrchestrator()
    # Empty warm pool -> cold provision
    sbx = orch.provision_sandbox(
        project_id="prj-agent-beta",
        environment=EnvironmentType.PROD,
        runtime="go:1.22",
    )
    assert sbx.project_id == "prj-agent-beta"
    assert sbx.runtime == "go:1.22"
    assert sbx.state == ContainerState.RUNNING
    assert sbx.boot_latency_ms >= 100.0


def test_terminate_sandbox_lifecycle() -> None:
    orch = ReplitCloudOrchestrator()
    sbx = orch.provision_sandbox(
        project_id="prj-test",
        environment=EnvironmentType.DEV,
        runtime="nodejs:20",
    )
    assert sbx.state == ContainerState.RUNNING
    ok = orch.terminate_sandbox(sbx.sandbox_id)
    assert ok is True
    assert sbx.state == ContainerState.STOPPED
    assert sbx.active_socket_connections == 0

    # Non-existent sandbox termination returns False
    assert orch.terminate_sandbox("non-existent-id") is False


def test_exec_agent_command_build() -> None:
    orch = ReplitCloudOrchestrator()
    sbx = orch.provision_sandbox(
        project_id="my-nextjs-app",
        environment=EnvironmentType.DEV,
        runtime="nodejs:20",
    )
    res = orch.exec_agent_command(sbx.sandbox_id, "npm run build")
    assert res.exit_code == 0
    assert "Compiled my-nextjs-app successfully" in res.stdout
    assert res.duration_ms > 0.0
    assert len(orch.execution_history) == 1


def test_exec_agent_command_stopped_sandbox_error() -> None:
    orch = ReplitCloudOrchestrator()
    sbx = orch.provision_sandbox(
        project_id="prj-err",
        environment=EnvironmentType.DEV,
        runtime="nodejs:20",
    )
    orch.terminate_sandbox(sbx.sandbox_id)

    with pytest.raises(RuntimeError, match="cannot execute commands"):
        orch.exec_agent_command(sbx.sandbox_id, "npm test")


def test_create_database_branch_isolation() -> None:
    orch = ReplitCloudOrchestrator()
    prod_branch = orch.create_database_branch(
        project_id="prj-store",
        target_env=EnvironmentType.PROD,
    )
    assert prod_branch.environment == EnvironmentType.PROD
    assert "postgresql://replit_user:*****@" in prod_branch.connection_string_masked
    assert prod_branch.is_isolated is True

    # Dev branch branched from prod
    dev_branch = orch.create_database_branch(
        project_id="prj-store",
        target_env=EnvironmentType.DEV,
        source_env=EnvironmentType.PROD,
    )
    assert dev_branch.environment == EnvironmentType.DEV
    assert dev_branch.source_branch_id == prod_branch.branch_id
    assert dev_branch.branch_id != prod_branch.branch_id


def test_verify_and_apply_migration_safe() -> None:
    orch = ReplitCloudOrchestrator()
    branch = orch.create_database_branch("prj-analytics", EnvironmentType.DEV)

    ddl = [
        "CREATE TABLE users (id SERIAL PRIMARY KEY, email TEXT NOT NULL UNIQUE);",
        "CREATE INDEX idx_users_email ON users(email);",
    ]

    # Dry-run test
    dry_res = orch.verify_and_apply_migration(branch.branch_id, "001_create_users", ddl, dry_run=True)
    assert dry_res.success is True
    assert dry_res.dry_run is True
    assert branch.schema_version == 1
    assert len(branch.applied_migrations) == 0

    # Live apply
    live_res = orch.verify_and_apply_migration(branch.branch_id, "001_create_users", ddl, dry_run=False)
    assert live_res.success is True
    assert live_res.dry_run is False
    assert live_res.rollback_checkpoint is not None
    assert branch.schema_version == 2
    assert "001_create_users" in branch.applied_migrations


def test_verify_and_apply_migration_destructive_rejection() -> None:
    orch = ReplitCloudOrchestrator()
    branch = orch.create_database_branch("prj-critical", EnvironmentType.PROD)

    destructive_ddl = ["DROP DATABASE production_db;"]
    res = orch.verify_and_apply_migration(branch.branch_id, "999_drop", destructive_ddl)
    assert res.success is False
    assert "safety gate" in res.error_message.lower()
    assert branch.schema_version == 1


def test_custom_domain_registration_and_ssl() -> None:
    orch = ReplitCloudOrchestrator()
    route = orch.register_custom_domain("myapp.com", "prj-agent", EnvironmentType.PROD)
    assert route.domain == "myapp.com"
    assert route.ssl_status == SSLStatus.PENDING_VALIDATION
    assert route.active_upstream == "prj-agent-prod.replit.app"

    # Issue SSL
    ok = orch.update_ssl_status("myapp.com", SSLStatus.ISSUED, validity_days=90)
    assert ok is True
    assert route.ssl_status == SSLStatus.ISSUED
    assert route.ssl_expires_at is not None


def test_canary_traffic_cutover_and_telemetry() -> None:
    orch = ReplitCloudOrchestrator()
    project_id = "prj-showcase"

    # Prewarm & provision
    orch.prewarm_containers("nodejs:20", count=2)
    sbx = orch.provision_sandbox(project_id, EnvironmentType.PROD, "nodejs:20")
    assert sbx.state == ContainerState.RUNNING

    # DB branch
    orch.create_database_branch(project_id, EnvironmentType.PROD)

    # Domain
    orch.register_custom_domain("showcase.ai", project_id, EnvironmentType.PROD)
    orch.update_ssl_status("showcase.ai", SSLStatus.ISSUED)

    # Canary cutover
    ok, msg = orch.configure_canary_traffic("showcase.ai", 25.0, "upstream-blue")
    assert ok is True
    assert "25.0%" in msg

    # Full cutover
    ok, msg = orch.configure_canary_traffic("showcase.ai", 100.0, "upstream-green")
    assert ok is True
    assert "100% traffic routed" in msg
    assert orch.domain_routes["showcase.ai"].active_upstream == "upstream-green"

    # Telemetry
    telemetry = orch.get_project_cloud_telemetry(project_id)
    assert telemetry["project_id"] == project_id
    assert telemetry["active_sandboxes_count"] == 1
    assert telemetry["database_branches_count"] == 1
    assert telemetry["custom_domains_count"] == 1
    assert telemetry["ssl_healthy_count"] == 1
    assert telemetry["warm_pool_available"] == 1
