"""Flow Traders Infrastructure Platform Orchestrator.

Provides production-grade infrastructure platform engineering components:
- TerraformModuleReconciler: Reusable module verification, policy-as-code evaluation, and drift detection.
- CiCdGuardrailEngine: GitHub Actions CI/CD guardrail rules, security policy gates, and automated rollback control.
- HybridK8sClusterManager: Hybrid Kubernetes cluster supervisor across cloud (GKE/EKS) and on-premise k3s nodes.
- KafkaStreamingWatchdog: Distributed Kafka broker health monitoring, partition skew detection, and consumer lag audits.
- LinuxFleetHostAuditor: Trading server host configuration supervisor, systemd monitoring, and kernel sysctl tuning.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Any


class ClusterType(str, enum.Enum):
    CLOUD_GKE = "gke"
    CLOUD_EKS = "eks"
    ON_PREM_K3S = "k3s"
    BARE_METAL = "bare_metal"


class HostHealthState(str, enum.Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    MAINTENANCE = "maintenance"


@dataclasses.dataclass(frozen=True)
class TerraformModuleSpec:
    name: str
    version: str
    source_path: str
    required_providers: dict[str, str]
    variables: list[str]
    outputs: list[str]
    mandatory_tags: list[str] = dataclasses.field(
        default_factory=lambda: ["Environment", "Owner", "CostCenter", "TradingVenue"]
    )


@dataclasses.dataclass
class PolicyEvaluationResult:
    passed: bool
    violations: list[str]
    warnings: list[str]
    evaluated_rules: int


class TerraformModuleReconciler:
    """Validates reusable Terraform modules, enforces policy-as-code, and detects state drift."""

    def __init__(self, allowed_providers: set[str] | None = None) -> None:
        self.allowed_providers = allowed_providers or {"google", "aws", "kubernetes", "helm", "kafka"}

    def validate_module_spec(self, spec: TerraformModuleSpec) -> PolicyEvaluationResult:
        violations: list[str] = []
        warnings: list[str] = []

        if not spec.name or not spec.version:
            violations.append("Module name and version are strictly required.")

        for provider in spec.required_providers:
            if provider not in self.allowed_providers:
                violations.append(f"Provider '{provider}' is not in approved provider set: {self.allowed_providers}")

        for tag in ["Environment", "Owner"]:
            if tag not in spec.mandatory_tags:
                violations.append(f"Mandatory governance tag '{tag}' missing from module spec.")

        if not spec.outputs:
            warnings.append("Module exports zero outputs; reusable modules should expose resource identifiers.")

        return PolicyEvaluationResult(
            passed=len(violations) == 0,
            violations=violations,
            warnings=warnings,
            evaluated_rules=len(spec.required_providers) + len(spec.mandatory_tags) + 2,
        )

    def detect_resource_drift(
        self,
        declared_state: dict[str, Any],
        actual_state: dict[str, Any],
    ) -> dict[str, Any]:
        drifted_resources: list[dict[str, Any]] = []

        for res_id, declared_attrs in declared_state.items():
            actual_attrs = actual_state.get(res_id)
            if actual_attrs is None:
                drifted_resources.append({
                    "resource_id": res_id,
                    "drift_type": "MISSING_IN_ACTUAL",
                    "details": "Resource declared in code but absent in live environment.",
                })
                continue

            diffs: dict[str, tuple[Any, Any]] = {}
            for attr_key, declared_val in declared_attrs.items():
                actual_val = actual_attrs.get(attr_key)
                if actual_val != declared_val:
                    diffs[attr_key] = (declared_val, actual_val)

            if diffs:
                drifted_resources.append({
                    "resource_id": res_id,
                    "drift_type": "ATTRIBUTE_MISMATCH",
                    "differences": diffs,
                })

        return {
            "has_drift": len(drifted_resources) > 0,
            "drift_count": len(drifted_resources),
            "drifted_resources": drifted_resources,
        }


class CiCdGuardrailEngine:
    """Enforces GitHub Actions CI/CD guardrail rules and safety gates for infrastructure."""

    def __init__(self, require_signed_commits: bool = True, max_blast_radius: int = 15) -> None:
        self.require_signed_commits = require_signed_commits
        self.max_blast_radius = max_blast_radius

    def evaluate_pull_request_gate(
        self,
        pr_context: dict[str, Any],
    ) -> PolicyEvaluationResult:
        violations: list[str] = []
        warnings: list[str] = []

        if pr_context.get("base_branch") == "main":
            if not pr_context.get("has_peer_approval", False):
                violations.append("Production PRs target 'main' must require peer review approval.")
            if self.require_signed_commits and not pr_context.get("commits_signed", False):
                violations.append("All commits must be cryptographically GPG/SSH signed.")

        changed_resources = int(pr_context.get("changed_resources_count", 0))
        if changed_resources > self.max_blast_radius:
            violations.append(
                f"Blast radius ({changed_resources} resources) exceeds threshold of {self.max_blast_radius}."
            )

        if pr_context.get("detected_unpinned_actions", False):
            violations.append("GitHub Actions workflows must use immutable full-commit SHA hashes, not mutable tags.")

        if pr_context.get("secrets_leak_detected", False):
            violations.append("Secrets scanner detected potential raw credentials in proposed diff.")

        return PolicyEvaluationResult(
            passed=len(violations) == 0,
            violations=violations,
            warnings=warnings,
            evaluated_rules=5,
        )

    def evaluate_canary_rollback(
        self,
        error_rate_pct: float,
        p99_latency_ms: float,
        error_threshold_pct: float = 0.5,
        latency_threshold_ms: float = 250.0,
    ) -> dict[str, Any]:
        rollback_needed = (error_rate_pct > error_threshold_pct) or (p99_latency_ms > latency_threshold_ms)
        reasons: list[str] = []
        if error_rate_pct > error_threshold_pct:
            reasons.append(f"Error rate {error_rate_pct:.2f}% exceeds threshold {error_threshold_pct:.2f}%")
        if p99_latency_ms > latency_threshold_ms:
            reasons.append(f"P99 latency {p99_latency_ms:.1f}ms exceeds threshold {latency_threshold_ms:.1f}ms")

        return {
            "rollback_triggered": rollback_needed,
            "reasons": reasons,
            "action": "EXECUTE_AUTOMATED_ROLLBACK" if rollback_needed else "PROCEED_ROLLOUT",
        }


@dataclasses.dataclass
class NodeCondition:
    node_name: str
    cluster_type: ClusterType
    ready: bool
    cni_ok: bool
    memory_pressure: bool
    disk_pressure: bool


class HybridK8sClusterManager:
    """Supervises hybrid Kubernetes clusters across cloud (GKE/EKS) and on-premise k3s nodes."""

    def __init__(self) -> None:
        self.nodes: dict[str, NodeCondition] = {}

    def register_node(self, condition: NodeCondition) -> None:
        self.nodes[condition.node_name] = condition

    def audit_cluster_health(self) -> dict[str, Any]:
        healthy_nodes: list[str] = []
        unhealthy_nodes: list[str] = []

        for node_name, cond in self.nodes.items():
            if cond.ready and cond.cni_ok and not cond.memory_pressure and not cond.disk_pressure:
                healthy_nodes.append(node_name)
            else:
                unhealthy_nodes.append(node_name)

        return {
            "total_nodes": len(self.nodes),
            "healthy_count": len(healthy_nodes),
            "unhealthy_count": len(unhealthy_nodes),
            "healthy_nodes": healthy_nodes,
            "unhealthy_nodes": unhealthy_nodes,
            "is_cluster_healthy": len(unhealthy_nodes) == 0 and len(self.nodes) > 0,
        }

    def prepare_node_drain(self, node_name: str, has_pdb_violations: bool = False) -> dict[str, Any]:
        if node_name not in self.nodes:
            raise KeyError(f"Node '{node_name}' is not registered in cluster manager.")

        if has_pdb_violations:
            return {
                "can_drain": False,
                "reason": "PodDisruptionBudget prevents draining node without risking quorum violation.",
                "action": "REJECT_DRAIN",
            }

        return {
            "can_drain": True,
            "node_name": node_name,
            "action": "CORDON_AND_DRAIN",
            "cluster_type": self.nodes[node_name].cluster_type.value,
        }


@dataclasses.dataclass
class KafkaPartitionMetric:
    topic: str
    partition_id: int
    leader_broker_id: int
    replicas: list[int]
    in_sync_replicas: list[int]
    consumer_lag: int


class KafkaStreamingWatchdog:
    """Monitors Kafka broker health, partition replica consistency, and consumer group lag."""

    def __init__(self, max_allowed_lag: int = 1000) -> None:
        self.max_allowed_lag = max_allowed_lag

    def audit_topic_partitions(self, partitions: list[KafkaPartitionMetric]) -> dict[str, Any]:
        under_replicated: list[dict[str, Any]] = []
        high_lag_partitions: list[dict[str, Any]] = []
        broker_leadership_count: dict[int, int] = {}

        for p in partitions:
            broker_leadership_count[p.leader_broker_id] = broker_leadership_count.get(p.leader_broker_id, 0) + 1

            if len(p.in_sync_replicas) < len(p.replicas):
                under_replicated.append({
                    "topic": p.topic,
                    "partition": p.partition_id,
                    "replicas": p.replicas,
                    "isr": p.in_sync_replicas,
                })

            if p.consumer_lag > self.max_allowed_lag:
                high_lag_partitions.append({
                    "topic": p.topic,
                    "partition": p.partition_id,
                    "lag": p.consumer_lag,
                })

        # Calculate partition distribution skew
        leadership_values = list(broker_leadership_count.values())
        max_leaders = max(leadership_values) if leadership_values else 0
        min_leaders = min(leadership_values) if leadership_values else 0
        skew_ratio = (max_leaders / min_leaders) if min_leaders > 0 else 1.0

        return {
            "total_partitions_audited": len(partitions),
            "under_replicated_count": len(under_replicated),
            "under_replicated_partitions": under_replicated,
            "high_lag_count": len(high_lag_partitions),
            "high_lag_partitions": high_lag_partitions,
            "broker_leadership_distribution": broker_leadership_count,
            "partition_skew_ratio": round(skew_ratio, 2),
            "health_status": "CRITICAL" if under_replicated else ("WARNING" if high_lag_partitions else "HEALTHY"),
        }


@dataclasses.dataclass
class LinuxHostSpec:
    hostname: str
    kernel_version: str
    active_systemd_services: list[str]
    sysctl_params: dict[str, Any]
    disk_usage_pct: float
    load_average: float


class LinuxFleetHostAuditor:
    """Supervises Linux trading hosts, verifies systemd services, and checks kernel tuning."""

    REQUIRED_SYSCTL = {
        "net.core.rmem_max": 16777216,
        "net.core.wmem_max": 16777216,
        "fs.file-max": 2097152,
    }

    def __init__(self, critical_services: list[str] | None = None) -> None:
        self.critical_services = critical_services or ["sshd", "chronyd", "node_exporter"]

    def audit_host(self, host: LinuxHostSpec) -> dict[str, Any]:
        missing_services: list[str] = [
            svc for svc in self.critical_services if svc not in host.active_systemd_services
        ]

        sysctl_mismatches: dict[str, dict[str, Any]] = {}
        for param, expected_val in self.REQUIRED_SYSCTL.items():
            actual_val = host.sysctl_params.get(param)
            if actual_val is None or actual_val < expected_val:
                sysctl_mismatches[param] = {
                    "expected_minimum": expected_val,
                    "actual": actual_val,
                }

        state = HostHealthState.HEALTHY
        if missing_services or sysctl_mismatches or host.disk_usage_pct > 90.0:
            state = HostHealthState.DEGRADED
        if host.disk_usage_pct > 95.0 or host.load_average > 64.0:
            state = HostHealthState.CRITICAL

        return {
            "hostname": host.hostname,
            "health_state": state.value,
            "missing_critical_services": missing_services,
            "sysctl_mismatches": sysctl_mismatches,
            "disk_usage_pct": host.disk_usage_pct,
            "is_ready_for_trading": state == HostHealthState.HEALTHY,
        }
