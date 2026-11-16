"""
finmid Embedded Financial Platform Orchestrator.

Models cloud infrastructure management, declarative Kubernetes reconcilers,
ephemeral staging preview environments, Terraform IaC drift watchdogs,
and regulated credit gateway observability for finmid's embedded B2B
lending infrastructure on AWS.
"""

from __future__ import annotations

import enum
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class PlatformTier(str, enum.Enum):
    ENTERPRISE_CORE = "ENTERPRISE_CORE"
    GROWTH_MARKETPLACE = "GROWTH_MARKETPLACE"
    DEVELOPER_SANDBOX = "DEVELOPER_SANDBOX"


class EnvironmentState(str, enum.Enum):
    PENDING = "PENDING"
    PROVISIONED = "PROVISIONED"
    DRIFT_DETECTED = "DRIFT_DETECTED"
    DEGRADED = "DEGRADED"
    TEARDOWN = "TEARDOWN"


class ComplianceStandard(str, enum.Enum):
    GDPR = "GDPR"
    ECSP_REGULATION = "ECSP_REGULATION"
    ISO27001 = "ISO27001"
    SOC2_TYPE_II = "SOC2_TYPE_II"


@dataclass
class TenantClusterSpec:
    tenant_id: str
    tenant_name: str
    tier: PlatformTier
    namespace: str
    allocated_cpu_cores: float
    allocated_memory_gib: float
    allowed_compliance_tags: List[ComplianceStandard] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


@dataclass
class EphemeralPreviewEnv:
    pr_number: int
    branch_name: str
    author: str
    namespace: str
    database_snapshot_id: str
    mock_webhook_endpoint: str
    status: EnvironmentState = EnvironmentState.PENDING
    ttl_seconds: int = 7200
    created_at: float = field(default_factory=time.time)

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > self.ttl_seconds


@dataclass
class TerraformResourceState:
    resource_id: str
    resource_type: str
    declared_hash: str
    live_hash: str
    last_reconciled_at: float = field(default_factory=time.time)

    @property
    def has_drift(self) -> bool:
        return self.declared_hash != self.live_hash


@dataclass
class GatewaySLAMetric:
    endpoint: str
    total_requests: int
    p95_latency_ms: float
    p99_latency_ms: float
    error_count: int

    @property
    def error_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.error_count / self.total_requests

    @property
    def is_sla_breached(self) -> bool:
        return self.p99_latency_ms > 150.0 or self.error_rate > 0.01


class EphemeralStagingReconciler:
    """Manages isolated ephemeral preview environments for engineering squads."""

    def __init__(self) -> None:
        self._active_previews: Dict[int, EphemeralPreviewEnv] = {}

    def provision_preview(
        self,
        pr_number: int,
        branch_name: str,
        author: str,
        db_snapshot: str = "snap-base-seed-v1"
    ) -> EphemeralPreviewEnv:
        namespace = f"preview-pr-{pr_number}-{author.lower().replace('_', '-')}"
        webhook_url = f"https://mock-webhooks.finmid-preview.internal/{namespace}"
        
        preview = EphemeralPreviewEnv(
            pr_number=pr_number,
            branch_name=branch_name,
            author=author,
            namespace=namespace,
            database_snapshot_id=db_snapshot,
            mock_webhook_endpoint=webhook_url,
            status=EnvironmentState.PROVISIONED
        )
        self._active_previews[pr_number] = preview
        return preview

    def teardown_preview(self, pr_number: int) -> bool:
        if pr_number in self._active_previews:
            self._active_previews[pr_number].status = EnvironmentState.TEARDOWN
            del self._active_previews[pr_number]
            return True
        return False

    def get_preview(self, pr_number: int) -> Optional[EphemeralPreviewEnv]:
        return self._active_previews.get(pr_number)

    def prune_expired(self) -> int:
        expired_prs = [pr for pr, env in self._active_previews.items() if env.is_expired]
        for pr in expired_prs:
            self.teardown_preview(pr)
        return len(expired_prs)


class TerraformDriftWatchdog:
    """Validates live AWS/Kubernetes resources against declared Terraform state."""

    def __init__(self) -> None:
        self._registry: Dict[str, TerraformResourceState] = {}

    def register_resource(
        self,
        resource_id: str,
        resource_type: str,
        declared_config: Dict[str, Any]
    ) -> TerraformResourceState:
        declared_hash = hashlib.sha256(
            json.dumps(declared_config, sort_keys=True).encode("utf-8")
        ).hexdigest()
        
        state = TerraformResourceState(
            resource_id=resource_id,
            resource_type=resource_type,
            declared_hash=declared_hash,
            live_hash=declared_hash
        )
        self._registry[resource_id] = state
        return state

    def simulate_live_mutation(
        self,
        resource_id: str,
        mutated_config: Dict[str, Any]
    ) -> Optional[TerraformResourceState]:
        if resource_id not in self._registry:
            return None
        new_hash = hashlib.sha256(
            json.dumps(mutated_config, sort_keys=True).encode("utf-8")
        ).hexdigest()
        self._registry[resource_id].live_hash = new_hash
        return self._registry[resource_id]

    def audit_drift(self) -> List[TerraformResourceState]:
        return [res for res in self._registry.values() if res.has_drift]

    def reconcile_drift(self, resource_id: str) -> bool:
        if resource_id in self._registry and self._registry[resource_id].has_drift:
            self._registry[resource_id].live_hash = self._registry[resource_id].declared_hash
            self._registry[resource_id].last_reconciled_at = time.time()
            return True
        return False


class CreditGatewaySLAWatchdog:
    """Monitors API gateway latency and failure metrics for embedded finance partner widgets."""

    def __init__(self) -> None:
        self._metrics: Dict[str, GatewaySLAMetric] = {}

    def record_traffic(
        self,
        endpoint: str,
        total_requests: int,
        p95_latency_ms: float,
        p99_latency_ms: float,
        error_count: int
    ) -> GatewaySLAMetric:
        metric = GatewaySLAMetric(
            endpoint=endpoint,
            total_requests=total_requests,
            p95_latency_ms=p95_latency_ms,
            p99_latency_ms=p99_latency_ms,
            error_count=error_count
        )
        self._metrics[endpoint] = metric
        return metric

    def get_breached_endpoints(self) -> List[GatewaySLAMetric]:
        return [m for m in self._metrics.values() if m.is_sla_breached]


class FinmidPlatformOrchestrator:
    """Central orchestrator coordinating finmid cloud platform, GitOps, and SLAs."""

    def __init__(self) -> None:
        self.tenants: Dict[str, TenantClusterSpec] = {}
        self.preview_reconciler = EphemeralStagingReconciler()
        self.drift_watchdog = TerraformDriftWatchdog()
        self.sla_watchdog = CreditGatewaySLAWatchdog()
        self.audit_log: List[Dict[str, Any]] = []

    def onboard_tenant(
        self,
        tenant_id: str,
        tenant_name: str,
        tier: PlatformTier,
        allowed_standards: Optional[List[ComplianceStandard]] = None
    ) -> TenantClusterSpec:
        standards = allowed_standards or [ComplianceStandard.GDPR, ComplianceStandard.ECSP_REGULATION]
        
        cpu_map = {
            PlatformTier.ENTERPRISE_CORE: 32.0,
            PlatformTier.GROWTH_MARKETPLACE: 16.0,
            PlatformTier.DEVELOPER_SANDBOX: 4.0
        }
        mem_map = {
            PlatformTier.ENTERPRISE_CORE: 128.0,
            PlatformTier.GROWTH_MARKETPLACE: 64.0,
            PlatformTier.DEVELOPER_SANDBOX: 16.0
        }

        spec = TenantClusterSpec(
            tenant_id=tenant_id,
            tenant_name=tenant_name,
            tier=tier,
            namespace=f"finmid-tenant-{tenant_id.lower()}",
            allocated_cpu_cores=cpu_map[tier],
            allocated_memory_gib=mem_map[tier],
            allowed_compliance_tags=standards
        )
        self.tenants[tenant_id] = spec
        self._log_event("TENANT_ONBOARDED", {"tenant_id": tenant_id, "tier": tier.value})
        return spec

    def _log_event(self, action: str, details: Dict[str, Any]) -> None:
        self.audit_log.append({
            "action": action,
            "details": details,
            "timestamp": time.time()
        })

    def run_platform_health_check(self) -> Dict[str, Any]:
        drifted = self.drift_watchdog.audit_drift()
        breached_slas = self.sla_watchdog.get_breached_endpoints()
        return {
            "status": "HEALTHY" if not drifted and not breached_slas else "ATTENTION_REQUIRED",
            "active_tenants": len(self.tenants),
            "active_preview_environments": len(self.preview_reconciler._active_previews),
            "drifted_resources_count": len(drifted),
            "breached_endpoints_count": len(breached_slas)
        }
