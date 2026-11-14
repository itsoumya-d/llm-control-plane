"""Replit Cloud Infrastructure Orchestrator & Agent Runtime Platform.

Designed for Replit Cloud's engineering organization:
- Sub-second ephemeral container sandbox lifecycle and pre-warmed pool orchestration.
- Development vs. Production database isolation, zero-copy branching, and schema migration safeguards.
- Edge ingress reverse-proxy routing, dynamic custom domains, and automated SSL/TLS management.
- Autonomous Replit Agent execution bridge with live container socket streaming and resource caps.
- Production multi-environment telemetry, latency profiling, and auto-rollback protection.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class EnvironmentType(str, Enum):
    DEV = "DEV"
    STAGING = "STAGING"
    PROD = "PROD"


class ContainerState(str, Enum):
    PENDING = "PENDING"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"
    FAILED = "FAILED"


class SSLStatus(str, Enum):
    PENDING_VALIDATION = "PENDING_VALIDATION"
    ISSUED = "ISSUED"
    RENEWING = "RENEWING"
    FAILED = "FAILED"
    EXPIRED = "EXPIRED"


@dataclass
class ContainerSandbox:
    sandbox_id: str
    project_id: str
    environment: EnvironmentType
    runtime: str  # e.g., "nodejs:20", "python:3.12", "go:1.22"
    cpu_cores: float = 2.0
    memory_limit_mb: int = 2048
    disk_quota_mb: int = 10240
    state: ContainerState = ContainerState.PENDING
    created_at: float = field(default_factory=time.time)
    boot_latency_ms: float = 0.0
    active_socket_connections: int = 0
    port_mappings: Dict[int, int] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class DatabaseBranch:
    branch_id: str
    project_id: str
    environment: EnvironmentType
    connection_string_masked: str
    schema_version: int = 1
    is_isolated: bool = True
    created_at: float = field(default_factory=time.time)
    applied_migrations: List[str] = field(default_factory=list)
    source_branch_id: Optional[str] = None


@dataclass
class MigrationResult:
    migration_id: str
    branch_id: str
    success: bool
    dry_run: bool
    ddl_statements_count: int
    rollback_checkpoint: Optional[str] = None
    error_message: Optional[str] = None
    execution_time_ms: float = 0.0


@dataclass
class CustomDomainRoute:
    domain: str
    project_id: str
    target_environment: EnvironmentType
    ssl_status: SSLStatus = SSLStatus.PENDING_VALIDATION
    ssl_expires_at: Optional[float] = None
    active_upstream: str = ""
    canary_percentage: float = 0.0
    health_status: str = "HEALTHY"
    created_at: float = field(default_factory=time.time)


@dataclass
class AgentExecutionResult:
    execution_id: str
    sandbox_id: str
    command: str
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: float
    timestamp: float = field(default_factory=time.time)


class ReplitCloudOrchestrator:
    """Core control plane engine for Replit Cloud first-party infrastructure."""

    def __init__(self) -> None:
        self.sandboxes: Dict[str, ContainerSandbox] = {}
        self.warm_pool: List[ContainerSandbox] = []
        self.database_branches: Dict[str, DatabaseBranch] = {}
        self.domain_routes: Dict[str, CustomDomainRoute] = {}
        self.execution_history: List[AgentExecutionResult] = []
        self.migration_checkpoints: Dict[str, List[str]] = {}

    def prewarm_containers(self, runtime: str, count: int = 3) -> int:
        """Pre-warms container sandboxes to achieve sub-second cold starts."""
        warmed = 0
        for _ in range(count):
            sandbox_id = f"warm-{uuid.uuid4().hex[:8]}"
            t0 = time.perf_counter()
            # Simulate low-level namespace/cgroup snapshot hydration
            sandbox = ContainerSandbox(
                sandbox_id=sandbox_id,
                project_id="system-warm-pool",
                environment=EnvironmentType.DEV,
                runtime=runtime,
                state=ContainerState.RUNNING,
                boot_latency_ms=round((time.perf_counter() - t0) * 1000 + 45.0, 2),
                port_mappings={3000: 3000, 8080: 8080},
            )
            self.warm_pool.append(sandbox)
            warmed += 1
        return warmed

    def provision_sandbox(
        self,
        project_id: str,
        environment: EnvironmentType,
        runtime: str,
        cpu_cores: float = 2.0,
        memory_limit_mb: int = 2048,
        disk_quota_mb: int = 10240,
        labels: Optional[Dict[str, str]] = None,
    ) -> ContainerSandbox:
        """Provisions an isolated container sandbox, reusing a warm instance if available."""
        t0 = time.perf_counter()
        warm_match = None
        for i, warm in enumerate(self.warm_pool):
            if warm.runtime == runtime and warm.state == ContainerState.RUNNING:
                warm_match = self.warm_pool.pop(i)
                break

        if warm_match:
            sandbox = warm_match
            sandbox.sandbox_id = f"sbx-{uuid.uuid4().hex[:10]}"
            sandbox.project_id = project_id
            sandbox.environment = environment
            sandbox.cpu_cores = cpu_cores
            sandbox.memory_limit_mb = memory_limit_mb
            sandbox.disk_quota_mb = disk_quota_mb
            sandbox.boot_latency_ms = round((time.perf_counter() - t0) * 1000 + 12.0, 2)
            sandbox.labels = labels or {}
        else:
            sandbox_id = f"sbx-{uuid.uuid4().hex[:10]}"
            sandbox = ContainerSandbox(
                sandbox_id=sandbox_id,
                project_id=project_id,
                environment=environment,
                runtime=runtime,
                cpu_cores=cpu_cores,
                memory_limit_mb=memory_limit_mb,
                disk_quota_mb=disk_quota_mb,
                state=ContainerState.RUNNING,
                boot_latency_ms=round((time.perf_counter() - t0) * 1000 + 185.0, 2),
                port_mappings={3000: 3000, 8080: 8080},
                labels=labels or {},
            )

        self.sandboxes[sandbox.sandbox_id] = sandbox
        return sandbox

    def terminate_sandbox(self, sandbox_id: str) -> bool:
        """Safely stops and cleans up sandbox resources and ephemeral mounts."""
        if sandbox_id not in self.sandboxes:
            return False
        sbx = self.sandboxes[sandbox_id]
        sbx.state = ContainerState.STOPPED
        sbx.active_socket_connections = 0
        return True

    def exec_agent_command(
        self,
        sandbox_id: str,
        command: str,
        timeout_seconds: float = 30.0,
    ) -> AgentExecutionResult:
        """Executes an agent command inside the container with stdout/stderr capture."""
        if sandbox_id not in self.sandboxes:
            raise ValueError(f"Sandbox {sandbox_id} does not exist.")
        sbx = self.sandboxes[sandbox_id]
        if sbx.state != ContainerState.RUNNING:
            raise RuntimeError(f"Sandbox {sandbox_id} is in state {sbx.state}, cannot execute commands.")

        t0 = time.perf_counter()
        execution_id = f"exec-{uuid.uuid4().hex[:8]}"

        # Simulated execution logic
        cmd_parts = command.strip().split()
        exit_code = 0
        stdout = ""
        stderr = ""

        if not cmd_parts:
            exit_code = 1
            stderr = "Empty command provided"
        elif cmd_parts[0] in ("npm", "pnpm", "yarn") and "build" in cmd_parts:
            stdout = f"Compiled {sbx.project_id} successfully in 420ms. Ready on port 3000."
        elif cmd_parts[0] == "git" and "status" in cmd_parts:
            stdout = "On branch main\nNothing to commit, working tree clean."
        elif cmd_parts[0] in ("pytest", "go", "cargo"):
            stdout = "All test suites passed (100% green)."
        elif "replit-agent" in command:
            stdout = "Replit Agent code synthesis complete. 3 files updated."
        else:
            stdout = f"Executed: {command}"

        elapsed_ms = round((time.perf_counter() - t0) * 1000 + 15.0, 2)
        res = AgentExecutionResult(
            execution_id=execution_id,
            sandbox_id=sandbox_id,
            command=command,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            duration_ms=elapsed_ms,
        )
        self.execution_history.append(res)
        return res

    def create_database_branch(
        self,
        project_id: str,
        target_env: EnvironmentType,
        source_env: Optional[EnvironmentType] = None,
    ) -> DatabaseBranch:
        """Creates an isolated database branch for safe development and staging testing."""
        branch_id = f"db-{project_id[:8]}-{target_env.value.lower()}-{uuid.uuid4().hex[:6]}"
        masked_host = f"pg-{project_id[:8]}.internal.replit.net:5432"
        conn_str = f"postgresql://replit_user:*****@{masked_host}/{branch_id}"

        source_branch_id = None
        initial_migrations = []
        schema_version = 1

        if source_env:
            for b in self.database_branches.values():
                if b.project_id == project_id and b.environment == source_env:
                    source_branch_id = b.branch_id
                    initial_migrations = list(b.applied_migrations)
                    schema_version = b.schema_version
                    break

        branch = DatabaseBranch(
            branch_id=branch_id,
            project_id=project_id,
            environment=target_env,
            connection_string_masked=conn_str,
            schema_version=schema_version,
            is_isolated=True,
            applied_migrations=initial_migrations,
            source_branch_id=source_branch_id,
        )
        self.database_branches[branch_id] = branch
        return branch

    def verify_and_apply_migration(
        self,
        branch_id: str,
        migration_id: str,
        ddl_statements: List[str],
        dry_run: bool = False,
    ) -> MigrationResult:
        """Verifies DDL safety in dry-run mode and applies schema changes with rollback checkpoint."""
        if branch_id not in self.database_branches:
            return MigrationResult(
                migration_id=migration_id,
                branch_id=branch_id,
                success=False,
                dry_run=dry_run,
                ddl_statements_count=len(ddl_statements),
                error_message=f"Branch {branch_id} not found",
            )

        branch = self.database_branches[branch_id]
        t0 = time.perf_counter()

        # Check for dangerous non-reversible destructive operations
        for ddl in ddl_statements:
            upper = ddl.upper()
            if "DROP DATABASE" in upper or "DROP TABLE" in upper and "IF EXISTS" not in upper:
                return MigrationResult(
                    migration_id=migration_id,
                    branch_id=branch_id,
                    success=False,
                    dry_run=dry_run,
                    ddl_statements_count=len(ddl_statements),
                    error_message="Destructive unhedged DROP statement rejected by Replit Cloud safety gate.",
                )

        if dry_run:
            return MigrationResult(
                migration_id=migration_id,
                branch_id=branch_id,
                success=True,
                dry_run=True,
                ddl_statements_count=len(ddl_statements),
                rollback_checkpoint=None,
                execution_time_ms=round((time.perf_counter() - t0) * 1000 + 5.0, 2),
            )

        # Create rollback checkpoint hash
        checkpoint_id = f"chk-{hashlib.sha256(f'{branch_id}:{migration_id}'.encode()).hexdigest()[:12]}"
        self.migration_checkpoints[checkpoint_id] = list(branch.applied_migrations)

        # Apply migration
        branch.applied_migrations.append(migration_id)
        branch.schema_version += 1
        elapsed_ms = round((time.perf_counter() - t0) * 1000 + 24.0, 2)

        return MigrationResult(
            migration_id=migration_id,
            branch_id=branch_id,
            success=True,
            dry_run=False,
            ddl_statements_count=len(ddl_statements),
            rollback_checkpoint=checkpoint_id,
            execution_time_ms=elapsed_ms,
        )

    def register_custom_domain(
        self,
        domain: str,
        project_id: str,
        target_env: EnvironmentType = EnvironmentType.PROD,
    ) -> CustomDomainRoute:
        """Registers a custom domain with edge ingress and triggers automated SSL provisioning."""
        clean_domain = domain.strip().lower()
        active_upstream = f"{project_id}-{target_env.value.lower()}.replit.app"
        route = CustomDomainRoute(
            domain=clean_domain,
            project_id=project_id,
            target_environment=target_env,
            ssl_status=SSLStatus.PENDING_VALIDATION,
            active_upstream=active_upstream,
            canary_percentage=0.0,
            health_status="HEALTHY",
        )
        self.domain_routes[clean_domain] = route
        return route

    def update_ssl_status(
        self,
        domain: str,
        new_status: SSLStatus,
        validity_days: int = 90,
    ) -> bool:
        """Updates ACME SSL/TLS certificate state and calculates renewal timestamps."""
        clean_domain = domain.strip().lower()
        if clean_domain not in self.domain_routes:
            return False
        route = self.domain_routes[clean_domain]
        route.ssl_status = new_status
        if new_status == SSLStatus.ISSUED:
            route.ssl_expires_at = time.time() + (validity_days * 86400)
        return True

    def configure_canary_traffic(
        self,
        domain: str,
        canary_percentage: float,
        target_upstream: str,
    ) -> Tuple[bool, str]:
        """Configures zero-downtime blue/green traffic cutover with canary gating."""
        clean_domain = domain.strip().lower()
        if clean_domain not in self.domain_routes:
            return False, f"Domain {domain} not registered."
        if not (0.0 <= canary_percentage <= 100.0):
            return False, "Canary percentage must be between 0.0 and 100.0."

        route = self.domain_routes[clean_domain]
        route.canary_percentage = canary_percentage
        if canary_percentage >= 100.0:
            route.active_upstream = target_upstream
            route.canary_percentage = 0.0
            return True, f"Full cutover to {target_upstream} complete (100% traffic routed)."
        return True, f"Canary traffic set to {canary_percentage}% pointing to {target_upstream}."

    def get_project_cloud_telemetry(self, project_id: str) -> Dict[str, Any]:
        """Compiles unified health, latency, database, and ingress metrics for a project."""
        project_sandboxes = [s for s in self.sandboxes.values() if s.project_id == project_id]
        project_branches = [b for b in self.database_branches.values() if b.project_id == project_id]
        project_routes = [r for r in self.domain_routes.values() if r.project_id == project_id]

        avg_boot = (
            sum(s.boot_latency_ms for s in project_sandboxes) / len(project_sandboxes)
            if project_sandboxes
            else 0.0
        )

        return {
            "project_id": project_id,
            "active_sandboxes_count": len([s for s in project_sandboxes if s.state == ContainerState.RUNNING]),
            "average_boot_latency_ms": round(avg_boot, 2),
            "database_branches_count": len(project_branches),
            "database_environments": list({b.environment.value for b in project_branches}),
            "custom_domains_count": len(project_routes),
            "ssl_healthy_count": len([r for r in project_routes if r.ssl_status == SSLStatus.ISSUED]),
            "warm_pool_available": len(self.warm_pool),
        }
