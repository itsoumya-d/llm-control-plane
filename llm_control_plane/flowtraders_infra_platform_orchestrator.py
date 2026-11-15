"""Flow Traders Infrastructure Platform Orchestrator & Multi-Cloud Fleet Governance Engine.

Designed for mission-critical quantitative trading infrastructure:
- TerraformModuleReconciler: Reusable IaC module templating, OPA policy evaluation, and drift detection.
- CiCdGuardrailEngine: Risk-scored change deployment gates, blast radius assessment, and emergency stop gates.
- HybridK8sClusterManager: Hybrid GKE cloud and on-prem k3s bare-metal node orchestration with PDB-aware cordon/drain.
- KafkaStreamingWatchdog: Market data stream lag monitoring, under-replicated partition detection, and rebalancing.
- LinuxFleetHostAuditor: Bare-metal trading host systemd daemon supervision and low-latency kernel sysctl compliance.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class ClusterType(str, Enum):
    CLOUD_GKE = "CLOUD_GKE"
    CLOUD_EKS = "CLOUD_EKS"
    ON_PREM_K3S = "ON_PREM_K3S"


class ChangeRiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class NodeConditionStatus(str, Enum):
    TRUE = "TRUE"
    FALSE = "FALSE"
    UNKNOWN = "UNKNOWN"


# ============================================================================
# 1. Terraform Module Reconciler & OPA Policy Evaluation
# ============================================================================

@dataclass
class TerraformResource:
    resource_type: str
    resource_name: str
    attributes: Dict[str, Any]
    tags: Dict[str, str] = field(default_factory=dict)

    @property
    def address(self) -> str:
        return f"{self.resource_type}.{self.resource_name}"

    def compute_hash(self) -> str:
        serialized = json.dumps(
            {"type": self.resource_type, "name": self.resource_name, "attrs": self.attributes, "tags": self.tags},
            sort_keys=True,
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


@dataclass
class PolicyEvaluationResult:
    allowed: bool
    violations: List[str]
    warnings: List[str]


@dataclass
class DriftDetectionResult:
    has_drift: bool
    added_attributes: Dict[str, Any]
    modified_attributes: Dict[str, Tuple[Any, Any]]  # (expected, observed)
    removed_attributes: List[str]


class TerraformModuleReconciler:
    """Manages reusable Terraform module templates, OPA policy compliance, and state drift detection."""

    MANDATORY_TAGS = {"team", "environment", "cost-center"}
    PROHIBITED_OPEN_PORTS = {22, 3389, 5432, 9092, 27017}

    def __init__(self, module_name: str, version: str) -> None:
        self.module_name = module_name
        self.version = version
        self.resources: Dict[str, TerraformResource] = {}
        self.recorded_hashes: Dict[str, str] = {}

    def register_resource(self, resource: TerraformResource) -> None:
        self.resources[resource.address] = resource
        self.recorded_hashes[resource.address] = resource.compute_hash()

    def evaluate_policies(self, resource: TerraformResource) -> PolicyEvaluationResult:
        violations: List[str] = []
        warnings: List[str] = []

        # Check mandatory tags
        missing_tags = self.MANDATORY_TAGS - set(resource.tags.keys())
        if missing_tags:
            violations.append(f"Resource {resource.address} missing mandatory tags: {sorted(missing_tags)}")

        # Check security groups / firewall rules for open 0.0.0.0/0 ingress
        if "security_group" in resource.resource_type or "firewall" in resource.resource_type:
            ingress_rules = resource.attributes.get("ingress", [])
            if isinstance(ingress_rules, dict):
                ingress_rules = [ingress_rules]
            for rule in ingress_rules:
                cidr_blocks = rule.get("cidr_blocks", [])
                from_port = rule.get("from_port")
                to_port = rule.get("to_port")
                if "0.0.0.0/0" in cidr_blocks and from_port is not None and to_port is not None:
                    for prohibited in self.PROHIBITED_OPEN_PORTS:
                        if from_port <= prohibited <= to_port:
                            violations.append(
                                f"Resource {resource.address} opens prohibited port {prohibited} to 0.0.0.0/0"
                            )

        # Enforce encryption at rest for storage resources
        if any(token in resource.resource_type for token in ("storage", "bucket", "disk")):
            encrypted = resource.attributes.get("encryption", {}).get("enabled")
            kms_key = (
                resource.attributes.get("kms_key_id")
                or resource.attributes.get("encryption", {}).get("kms_key_id")
            )
            if encrypted is not True and not kms_key:
                violations.append(f"Resource {resource.address} must enable KMS encryption at rest")

        # Database retention warnings
        if "database" in resource.resource_type or "db" in resource.resource_type:
            retention_days = resource.attributes.get("backup_retention_days", 0)
            if retention_days < 7:
                warnings.append(
                    f"Resource {resource.address} backup retention is {retention_days} days (< 7 days recommended)"
                )

        return PolicyEvaluationResult(allowed=len(violations) == 0, violations=violations, warnings=warnings)

    def detect_drift(self, resource_address: str, observed_attributes: Dict[str, Any]) -> DriftDetectionResult:
        if resource_address not in self.resources:
            raise KeyError(f"Unknown resource address: {resource_address}")

        expected = self.resources[resource_address].attributes
        added: Dict[str, Any] = {}
        modified: Dict[str, Tuple[Any, Any]] = {}
        removed: List[str] = []

        # Find modified and removed
        for k, exp_val in expected.items():
            if k not in observed_attributes:
                removed.append(k)
            elif observed_attributes[k] != exp_val:
                modified[k] = (exp_val, observed_attributes[k])

        # Find added
        for k, obs_val in observed_attributes.items():
            if k not in expected:
                added[k] = obs_val

        has_drift = bool(added or modified or removed)
        return DriftDetectionResult(
            has_drift=has_drift,
            added_attributes=added,
            modified_attributes=modified,
            removed_attributes=removed,
        )


# ============================================================================
# 2. CI/CD Guardrail Engine & Blast Radius Assessment
# ============================================================================

@dataclass
class PlannedChange:
    resource_address: str
    resource_type: str
    action: str  # "CREATE", "UPDATE", "DELETE", "NOOP"
    is_sensitive: bool = False


@dataclass
class BlastRadiusAssessment:
    risk_level: ChangeRiskLevel
    risk_score: float  # 0.0 to 100.0
    creates: int
    updates: int
    deletes: int
    sensitive_deletes: int
    details: List[str]


class CiCdGuardrailEngine:
    """Evaluates planned infrastructure changes against blast radius metrics and automated emergency gates."""

    SENSITIVE_TYPES = {
        "google_compute_network",
        "aws_vpc",
        "kubernetes_cluster",
        "kafka_cluster",
        "postgresql_database",
        "cloud_nat_gateway",
    }

    def __init__(self, max_allowed_risk_score: float = 75.0) -> None:
        self.max_allowed_risk_score = max_allowed_risk_score

    def assess_blast_radius(self, changes: List[PlannedChange]) -> BlastRadiusAssessment:
        creates = 0
        updates = 0
        deletes = 0
        sensitive_deletes = 0
        details: List[str] = []

        score = 0.0

        for change in changes:
            is_sensitive = change.resource_type in self.SENSITIVE_TYPES or change.is_sensitive
            if change.action == "CREATE":
                creates += 1
                score += 2.0
            elif change.action == "UPDATE":
                updates += 1
                score += 5.0 if is_sensitive else 2.5
            elif change.action == "DELETE":
                deletes += 1
                if is_sensitive:
                    sensitive_deletes += 1
                    score += 35.0
                    details.append(f"Destruction of sensitive resource: {change.resource_address}")
                else:
                    score += 10.0

        # Cap score at 100.0
        risk_score = min(100.0, score)

        if sensitive_deletes > 0 or risk_score >= 80.0:
            level = ChangeRiskLevel.CRITICAL
        elif risk_score >= 50.0:
            level = ChangeRiskLevel.HIGH
        elif risk_score >= 20.0:
            level = ChangeRiskLevel.MEDIUM
        else:
            level = ChangeRiskLevel.LOW

        return BlastRadiusAssessment(
            risk_level=level,
            risk_score=risk_score,
            creates=creates,
            updates=updates,
            deletes=deletes,
            sensitive_deletes=sensitive_deletes,
            details=details,
        )

    def evaluate_gate(self, assessment: BlastRadiusAssessment, bypass_token: Optional[str] = None) -> Tuple[bool, str]:
        if assessment.sensitive_deletes > 0 and not bypass_token:
            msg = (
                f"Emergency gate blocked deployment: {assessment.sensitive_deletes} sensitive resource(s) "
                "slated for deletion without senior bypass token."
            )
            return (False, msg)
        if assessment.risk_score > self.max_allowed_risk_score and not bypass_token:
            msg = (
                f"Risk score {assessment.risk_score:.1f} exceeds threshold {self.max_allowed_risk_score:.1f}. "
                "Manual approval required."
            )
            return (False, msg)
        return (
            True,
            f"Deployment approved at risk level {assessment.risk_level.value} (score {assessment.risk_score:.1f}).",
        )


# ============================================================================
# 3. Hybrid Kubernetes Cluster Manager (GKE & Bare-Metal k3s)
# ============================================================================

@dataclass
class NodeCondition:
    type: str  # "Ready", "DiskPressure", "MemoryPressure", "NetworkUnavailable"
    status: NodeConditionStatus
    reason: Optional[str] = None


@dataclass
class K8sNode:
    name: str
    cluster_type: ClusterType
    conditions: Dict[str, NodeCondition]
    cordoned: bool = False
    drain_completed: bool = False
    pods: List[str] = field(default_factory=list)


@dataclass
class PodDisruptionBudget:
    app_label: str
    min_available: int
    current_available: int

    def can_evict(self) -> bool:
        return self.current_available > self.min_available


class HybridK8sClusterManager:
    """Supervises hybrid cloud (GKE) and bare-metal (k3s) Kubernetes nodes with safe eviction policies."""

    def __init__(self) -> None:
        self.nodes: Dict[str, K8sNode] = {}
        self.pdbs: Dict[str, PodDisruptionBudget] = {}

    def register_node(self, node: K8sNode) -> None:
        self.nodes[node.name] = node

    def register_pdb(self, pdb: PodDisruptionBudget) -> None:
        self.pdbs[pdb.app_label] = pdb

    def is_node_healthy(self, node_name: str) -> bool:
        node = self.nodes.get(node_name)
        if not node:
            return False

        ready = node.conditions.get("Ready")
        disk = node.conditions.get("DiskPressure")
        mem = node.conditions.get("MemoryPressure")
        net = node.conditions.get("NetworkUnavailable")

        if ready is None or ready.status != NodeConditionStatus.TRUE:
            return False
        if disk and disk.status == NodeConditionStatus.TRUE:
            return False
        if mem and mem.status == NodeConditionStatus.TRUE:
            return False
        if net and net.status == NodeConditionStatus.TRUE:
            return False

        return True

    def cordon_node(self, node_name: str) -> bool:
        node = self.nodes.get(node_name)
        if not node:
            return False
        node.cordoned = True
        return True

    def drain_node(self, node_name: str, app_labels: Dict[str, str]) -> Tuple[bool, List[str]]:
        """Drains a node, ensuring PodDisruptionBudgets are respected."""
        node = self.nodes.get(node_name)
        if not node:
            return False, ["Node not found"]
        if not node.cordoned:
            self.cordon_node(node_name)

        evicted: List[str] = []
        blocked: List[str] = []

        for pod in list(node.pods):
            app = app_labels.get(pod, "default")
            pdb = self.pdbs.get(app)
            if pdb and not pdb.can_evict():
                blocked.append(f"Pod {pod} (app={app}) blocked by PDB: min_available={pdb.min_available}")
            else:
                if pdb:
                    pdb.current_available -= 1
                evicted.append(pod)

        if not blocked:
            node.pods.clear()
            node.drain_completed = True
            return True, evicted
        else:
            # Partially drained or blocked
            for p in evicted:
                node.pods.remove(p)
            return False, blocked


# ============================================================================
# 4. Kafka Streaming Watchdog & Partition Rebalancing
# ============================================================================

@dataclass
class TopicPartition:
    topic: str
    partition: int
    leader_broker_id: int
    replicas: List[int]
    in_sync_replicas: List[int]
    current_offset: int
    log_end_offset: int

    @property
    def lag(self) -> int:
        return max(0, self.log_end_offset - self.current_offset)

    @property
    def is_under_replicated(self) -> bool:
        return len(self.in_sync_replicas) < len(self.replicas)


@dataclass
class BrokerHealthSummary:
    broker_id: int
    assigned_partitions: int
    lead_partitions: int
    healthy: bool


class KafkaStreamingWatchdog:
    """Monitors high-throughput market data Kafka clusters, detecting partition skew and consumer lag."""

    def __init__(self, lag_threshold: int = 5000) -> None:
        self.lag_threshold = lag_threshold
        self.partitions: List[TopicPartition] = []

    def record_partition(self, tp: TopicPartition) -> None:
        self.partitions.append(tp)

    def detect_under_replicated_partitions(self) -> List[TopicPartition]:
        return [p for p in self.partitions if p.is_under_replicated]

    def detect_high_lag_partitions(self) -> List[TopicPartition]:
        return [p for p in self.partitions if p.lag >= self.lag_threshold]

    def compute_broker_distribution(self, broker_ids: List[int]) -> Dict[int, BrokerHealthSummary]:
        res: Dict[int, BrokerHealthSummary] = {
            b: BrokerHealthSummary(broker_id=b, assigned_partitions=0, lead_partitions=0, healthy=True)
            for b in broker_ids
        }
        for tp in self.partitions:
            if tp.leader_broker_id in res:
                res[tp.leader_broker_id].lead_partitions += 1
            for r in tp.replicas:
                if r in res:
                    res[r].assigned_partitions += 1

        # Check for under-replicated brokers
        for tp in self.partitions:
            missing = set(tp.replicas) - set(tp.in_sync_replicas)
            for m in missing:
                if m in res:
                    res[m].healthy = False

        return res

    def generate_rebalance_recommendations(self, broker_ids: List[int]) -> List[Dict[str, Any]]:
        """Generates partition rebalance recommendations to distribute leader burden evenly."""
        distribution = self.compute_broker_distribution(broker_ids)
        total_lead = sum(d.lead_partitions for d in distribution.values())
        if not broker_ids:
            return []
        target_per_broker = total_lead / len(broker_ids)

        recommendations: List[Dict[str, Any]] = []
        # Find overloaded brokers (> 1.5x target)
        for b_id, summary in distribution.items():
            if summary.lead_partitions > target_per_broker * 1.5 and summary.lead_partitions > 2:
                recommendations.append({
                    "action": "SHED_LEADERSHIP",
                    "broker_id": b_id,
                    "current_leads": summary.lead_partitions,
                    "target_leads": int(target_per_broker),
                })
        return recommendations


# ============================================================================
# 5. Linux Fleet Host Auditor & Kernel Tuning
# ============================================================================

@dataclass
class ServiceAuditResult:
    service_name: str
    active: bool
    enabled: bool
    status_text: str


@dataclass
class SysctlSetting:
    parameter: str
    expected_minimum: int
    observed_value: int

    @property
    def is_compliant(self) -> bool:
        return self.observed_value >= self.expected_minimum


class LinuxFleetHostAuditor:
    """Supervises bare-metal trading fleet hosts, validating systemd units and high-frequency network settings."""

    RECOMMENDED_SYSCTL = {
        "net.core.rmem_max": 16777216,
        "net.core.wmem_max": 16777216,
        "net.core.somaxconn": 65535,
        "fs.file-max": 2097152,
    }

    def __init__(self, hostname: str) -> None:
        self.hostname = hostname

    def audit_services(self, service_states: Dict[str, Dict[str, Any]]) -> List[ServiceAuditResult]:
        results: List[ServiceAuditResult] = []
        for name, data in service_states.items():
            active = data.get("active") is True or data.get("SubState") == "running"
            enabled = data.get("enabled") is True or data.get("UnitFileState") == "enabled"
            results.append(ServiceAuditResult(
                service_name=name,
                active=active,
                enabled=enabled,
                status_text="RUNNING" if active else "INACTIVE_OR_FAILED",
            ))
        return results

    def audit_kernel_sysctl(self, current_settings: Dict[str, int]) -> Tuple[bool, List[SysctlSetting]]:
        evaluations: List[SysctlSetting] = []
        all_compliant = True

        for param, exp_min in self.RECOMMENDED_SYSCTL.items():
            observed = current_settings.get(param, 0)
            setting = SysctlSetting(parameter=param, expected_minimum=exp_min, observed_value=observed)
            if not setting.is_compliant:
                all_compliant = False
            evaluations.append(setting)

        return all_compliant, evaluations
