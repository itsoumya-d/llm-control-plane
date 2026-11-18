"""Dynatrace Kafka Fleet Operator & Multi-Cloud Observability Engine.

Production-grade Kubernetes-native operator reconciler, Strimzi CRD lifecycle controller,
Cruise Control partition rebalancer, and Prometheus observability telemetry runtime for
managing planetary-scale event streaming fleets across AWS, Azure, and GCP.
"""

from __future__ import annotations

import copy
import dataclasses
import enum
import hashlib
import json
import logging
import math
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("dynatrace.kafka.fleet")


class CloudProvider(str, enum.Enum):
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"


class ClusterState(str, enum.Enum):
    PROVISIONING = "PROVISIONING"
    READY = "READY"
    RECONCILING = "RECONCILING"
    UPGRADING = "UPGRADING"
    REBALANCING = "REBALANCING"
    DRIFT_DETECTED = "DRIFT_DETECTED"
    DEGRADED = "DEGRADED"


class AclPermission(str, enum.Enum):
    ALLOW = "ALLOW"
    DENY = "DENY"


class AclOperation(str, enum.Enum):
    READ = "READ"
    WRITE = "WRITE"
    CREATE = "CREATE"
    DESCRIBE = "DESCRIBE"
    ALTER = "ALTER"
    ALL = "ALL"


@dataclasses.dataclass(frozen=True)
class KafkaTopicSpec:
    name: str
    partitions: int
    replication_factor: int
    min_insync_replicas: int = 2
    retention_ms: int = 604800000  # 7 days
    cleanup_policy: str = "delete"
    segment_bytes: int = 1073741824  # 1GB

    def validate(self, broker_count: int) -> List[str]:
        errors: List[str] = []
        if self.partitions <= 0:
            errors.append(f"Topic {self.name}: partitions must be > 0 (got {self.partitions})")
        if self.replication_factor <= 0 or self.replication_factor > broker_count:
            errors.append(
                f"Topic {self.name}: replication_factor {self.replication_factor} must be in 1..{broker_count}"
            )
        if self.min_insync_replicas > self.replication_factor:
            errors.append(
                f"Topic {self.name}: min_insync_replicas {self.min_insync_replicas} cannot exceed replication_factor {self.replication_factor}"
            )
        return errors


@dataclasses.dataclass(frozen=True)
class KafkaUserAclSpec:
    principal: str
    resource_type: str  # "topic", "group", "cluster"
    resource_name: str
    operation: AclOperation
    permission: AclPermission = AclPermission.ALLOW
    host: str = "*"


@dataclasses.dataclass
class BrokerResourceSpec:
    cpu_cores: float = 4.0
    memory_gb: float = 16.0
    storage_gb: int = 1000
    storage_type: str = "gp3"  # e.g., gp3, managed-premium, pd-ssd


@dataclasses.dataclass
class KafkaClusterSpec:
    cluster_id: str
    name: str
    namespace: str
    cloud_provider: CloudProvider
    region: str
    kafka_version: str = "3.8.0"
    broker_replicas: int = 3
    resources: BrokerResourceSpec = dataclasses.field(default_factory=BrokerResourceSpec)
    rack_awareness_enabled: bool = True
    cruise_control_enabled: bool = True
    tls_encryption_enabled: bool = True
    topics: Dict[str, KafkaTopicSpec] = dataclasses.field(default_factory=dict)
    acls: List[KafkaUserAclSpec] = dataclasses.field(default_factory=list)

    def spec_hash(self) -> str:
        data = {
            "version": self.kafka_version,
            "replicas": self.broker_replicas,
            "resources": dataclasses.asdict(self.resources),
            "rack_awareness": self.rack_awareness_enabled,
            "topics": {k: dataclasses.asdict(v) for k, v in sorted(self.topics.items())},
            "acls": [dataclasses.asdict(a) for a in self.acls],
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()[:16]

    def to_strimzi_cr(self) -> Dict[str, Any]:
        """Generate official Strimzi Kafka CRD representation."""
        return {
            "apiVersion": "kafka.strimzi.io/v1beta2",
            "kind": "Kafka",
            "metadata": {
                "name": self.name,
                "namespace": self.namespace,
                "labels": {
                    "app.kubernetes.io/managed-by": "dynatrace-kafka-fleet-operator",
                    "cloud.dynatrace.com/provider": self.cloud_provider.value,
                    "cloud.dynatrace.com/region": self.region,
                },
            },
            "spec": {
                "kafka": {
                    "version": self.kafka_version,
                    "replicas": self.broker_replicas,
                    "listeners": [
                        {"name": "plain", "port": 9092, "type": "internal", "tls": False},
                        {"name": "tls", "port": 9093, "type": "internal", "tls": self.tls_encryption_enabled},
                    ],
                    "config": {
                        "offsets.topic.replication.factor": min(3, self.broker_replicas),
                        "transaction.state.log.replication.factor": min(3, self.broker_replicas),
                        "transaction.state.log.min.isr": min(2, self.broker_replicas),
                        "default.replication.factor": min(3, self.broker_replicas),
                        "min.insync.replicas": 2 if self.broker_replicas >= 3 else 1,
                        "auto.create.topics.enable": "false",
                    },
                    "storage": {
                        "type": "persistent-claim",
                        "size": f"{self.resources.storage_gb}Gi",
                        "class": f"{self.cloud_provider.value}-fast",
                        "deleteClaim": False,
                    },
                    "rack": {"topologyKey": "topology.kubernetes.io/zone"} if self.rack_awareness_enabled else None,
                },
                "cruiseControl": {"autoRebalance": self.cruise_control_enabled},
            },
        }


@dataclasses.dataclass
class BrokerRuntimeStatus:
    broker_id: int
    pod_name: str
    rack: str
    kafka_version: str
    in_sync: bool = True
    disk_used_gb: float = 200.0
    disk_capacity_gb: float = 1000.0
    cpu_percent: float = 25.0
    leader_partitions: int = 10
    total_partitions: int = 30


@dataclasses.dataclass
class KafkaClusterStatus:
    cluster_id: str
    current_state: ClusterState
    active_version: str
    active_replicas: int
    brokers: List[BrokerRuntimeStatus] = dataclasses.field(default_factory=list)
    applied_spec_hash: str = ""
    under_replicated_partitions: int = 0
    active_controller_count: int = 1
    max_consumer_lag: int = 0
    conditions: List[Dict[str, Any]] = dataclasses.field(default_factory=list)


# -------------------------------------------------------------------------
# Reconciler & Quality Gate Engine
# -------------------------------------------------------------------------

@dataclasses.dataclass
class QualityGateResult:
    passed: bool
    reasons: List[str] = dataclasses.field(default_factory=list)
    checks_performed: List[str] = dataclasses.field(default_factory=list)


class KafkaFleetReconciler:
    """Reconciles declarative desired spec against live cluster runtime."""

    def __init__(self, max_allowed_lag_for_upgrade: int = 5000):
        self.max_allowed_lag_for_upgrade = max_allowed_lag_for_upgrade

    def evaluate_pre_release_gate(
        self, spec: KafkaClusterSpec, status: KafkaClusterStatus
    ) -> QualityGateResult:
        reasons: List[str] = []
        checks = [
            "zero_under_replicated_partitions",
            "active_controller_elected",
            "consumer_lag_within_sla",
            "all_brokers_in_sync",
            "topic_specs_valid",
        ]

        if status.under_replicated_partitions > 0:
            reasons.append(
                f"Blocked: {status.under_replicated_partitions} under-replicated partitions detected"
            )

        if status.active_controller_count != 1:
            reasons.append(
                f"Blocked: active controller count is {status.active_controller_count} (must be exactly 1)"
            )

        if status.max_consumer_lag > self.max_allowed_lag_for_upgrade:
            reasons.append(
                f"Blocked: max consumer lag {status.max_consumer_lag} exceeds SLA threshold {self.max_allowed_lag_for_upgrade}"
            )

        out_of_sync = [b.broker_id for b in status.brokers if not b.in_sync]
        if out_of_sync:
            reasons.append(f"Blocked: brokers {out_of_sync} are not currently in-sync")

        for topic in spec.topics.values():
            topic_errs = topic.validate(spec.broker_replicas)
            reasons.extend(topic_errs)

        return QualityGateResult(
            passed=len(reasons) == 0,
            reasons=reasons,
            checks_performed=checks,
        )

    def detect_configuration_drift(
        self, desired: KafkaClusterSpec, current: KafkaClusterStatus
    ) -> List[str]:
        drift: List[str] = []
        if desired.kafka_version != current.active_version:
            drift.append(f"Version drift: desired {desired.kafka_version} vs active {current.active_version}")

        if desired.broker_replicas != current.active_replicas:
            drift.append(
                f"Replica drift: desired {desired.broker_replicas} vs active {current.active_replicas}"
            )

        spec_hash = desired.spec_hash()
        if current.applied_spec_hash and spec_hash != current.applied_spec_hash:
            drift.append(f"Spec hash mismatch: desired {spec_hash} != applied {current.applied_spec_hash}")

        return drift

    def reconcile_cluster(
        self, desired: KafkaClusterSpec, current: KafkaClusterStatus
    ) -> Tuple[KafkaClusterStatus, List[str]]:
        """Executes one reconciliation step and returns updated status and execution log."""
        updated = copy.deepcopy(current)
        logs: List[str] = []

        # Step 1: Pre-release validation quality gate
        gate = self.evaluate_pre_release_gate(desired, current)
        if not gate.passed:
            updated.current_state = ClusterState.DEGRADED
            updated.conditions.append({
                "type": "QualityGateFailed",
                "status": "True",
                "reasons": gate.reasons,
            })
            logs.append(f"Reconciliation blocked by quality gate: {gate.reasons}")
            return updated, logs

        # Step 2: Drift detection
        drifts = self.detect_configuration_drift(desired, current)
        if drifts:
            logs.append(f"Detected configuration drift: {drifts}")
            updated.current_state = ClusterState.DRIFT_DETECTED

        # Step 3: Handle rolling version upgrade
        if desired.kafka_version != current.active_version:
            updated.current_state = ClusterState.UPGRADING
            logs.append(
                f"Executing rolling upgrade: {current.active_version} -> {desired.kafka_version}"
            )
            # Sequential rolling upgrade node-by-node
            for broker in updated.brokers:
                logs.append(f"Restarting broker {broker.broker_id} with Kafka {desired.kafka_version}")
                broker.kafka_version = desired.kafka_version
                broker.in_sync = True
            updated.active_version = desired.kafka_version
            logs.append(f"Rolling upgrade completed successfully across {len(updated.brokers)} brokers")

        # Step 4: Handle scaling
        if desired.broker_replicas > current.active_replicas:
            updated.current_state = ClusterState.RECONCILING
            diff = desired.broker_replicas - current.active_replicas
            logs.append(f"Scaling out cluster by +{diff} brokers")
            start_id = max([b.broker_id for b in updated.brokers], default=-1) + 1
            for i in range(diff):
                new_id = start_id + i
                rack_id = f"zone-{(new_id % 3) + 1}"
                updated.brokers.append(BrokerRuntimeStatus(
                    broker_id=new_id,
                    pod_name=f"{desired.name}-kafka-{new_id}",
                    rack=rack_id,
                    kafka_version=desired.kafka_version,
                    in_sync=True,
                    disk_capacity_gb=float(desired.resources.storage_gb),
                ))
            updated.active_replicas = desired.broker_replicas

        # Finalize state
        updated.applied_spec_hash = desired.spec_hash()
        updated.current_state = ClusterState.READY
        logs.append(f"Cluster {desired.name} reconciled to READY state (hash: {updated.applied_spec_hash})")
        return updated, logs


# -------------------------------------------------------------------------
# Cruise Control Partition Rebalancing Engine
# -------------------------------------------------------------------------

@dataclasses.dataclass
class PartitionMoveProposal:
    topic: str
    partition: int
    source_broker: int
    target_broker: int
    size_mb: float
    throttle_bytes_per_sec: int = 52428800  # 50 MB/s throttle


class CruiseControlPartitionRebalancer:
    """Calculates partition reassignment proposals to balance disk and leader load."""

    def __init__(self, disk_imbalance_threshold: float = 0.15, max_moves_per_batch: int = 20):
        self.disk_imbalance_threshold = disk_imbalance_threshold
        self.max_moves_per_batch = max_moves_per_batch

    def compute_imbalance_ratio(self, brokers: List[BrokerRuntimeStatus]) -> float:
        if not brokers or len(brokers) < 2:
            return 0.0
        ratios = [b.disk_used_gb / max(1.0, b.disk_capacity_gb) for b in brokers]
        avg = sum(ratios) / len(ratios)
        if avg == 0.0:
            return 0.0
        variance = sum((r - avg) ** 2 for r in ratios) / len(ratios)
        return math.sqrt(variance) / avg

    def generate_rebalance_proposals(
        self, brokers: List[BrokerRuntimeStatus], topics: Dict[str, KafkaTopicSpec]
    ) -> List[PartitionMoveProposal]:
        if len(brokers) < 2:
            return []

        # Sort brokers by disk utilization ratio
        sorted_brokers = sorted(
            brokers,
            key=lambda b: b.disk_used_gb / max(1.0, b.disk_capacity_gb),
            reverse=True,
        )
        heaviest = sorted_brokers[0]
        lightest = sorted_brokers[-1]

        heavy_ratio = heaviest.disk_used_gb / heaviest.disk_capacity_gb
        light_ratio = lightest.disk_used_gb / lightest.disk_capacity_gb

        # Check if difference exceeds threshold
        if (heavy_ratio - light_ratio) < self.disk_imbalance_threshold:
            return []

        proposals: List[PartitionMoveProposal] = []
        # Move partitions from heaviest to lightest
        topic_name = next(iter(topics.keys()), "telemetry-events")
        excess_gb = (heavy_ratio - light_ratio) * heaviest.disk_capacity_gb / 2.0
        move_size_mb = 500.0  # 500MB per partition
        num_partitions_to_move = min(
            self.max_moves_per_batch,
            max(1, int((excess_gb * 1024) / move_size_mb)),
        )

        for p_id in range(num_partitions_to_move):
            proposals.append(PartitionMoveProposal(
                topic=topic_name,
                partition=p_id,
                source_broker=heaviest.broker_id,
                target_broker=lightest.broker_id,
                size_mb=move_size_mb,
            ))

        return proposals


# -------------------------------------------------------------------------
# Prometheus Observability & Telemetry Collector
# -------------------------------------------------------------------------

class KafkaObservabilityAndLagMonitor:
    """Collects and formats Prometheus metrics for broker health and consumer lag."""

    def __init__(self, cluster_name: str, namespace: str):
        self.cluster_name = cluster_name
        self.namespace = namespace

    def calculate_consumer_lag(
        self, topic_offsets: Dict[int, int], consumer_offsets: Dict[int, int]
    ) -> Dict[int, int]:
        """Calculates lag per partition: topic log-end offset minus consumer committed offset."""
        lag: Dict[int, int] = {}
        for p, log_end in topic_offsets.items():
            committed = consumer_offsets.get(p, 0)
            lag[p] = max(0, log_end - committed)
        return lag

    def export_prometheus_metrics(
        self,
        status: KafkaClusterStatus,
        consumer_lags: Dict[str, Dict[str, int]],  # group -> topic -> total lag
    ) -> str:
        lines = [
            f"# HELP kafka_server_brokermonitoring_underreplicatedpartitions Number of under-replicated partitions.",
            f"# TYPE kafka_server_brokermonitoring_underreplicatedpartitions gauge",
            f'kafka_server_brokermonitoring_underreplicatedpartitions{{cluster="{self.cluster_name}",namespace="{self.namespace}"}} {status.under_replicated_partitions}',
            f"# HELP kafka_controller_activecontrollercount Number of active controllers in the cluster.",
            f"# TYPE kafka_controller_activecontrollercount gauge",
            f'kafka_controller_activecontrollercount{{cluster="{self.cluster_name}",namespace="{self.namespace}"}} {status.active_controller_count}',
            f"# HELP kafka_cluster_active_replicas Number of active broker replicas.",
            f"# TYPE kafka_cluster_active_replicas gauge",
            f'kafka_cluster_active_replicas{{cluster="{self.cluster_name}",namespace="{self.namespace}"}} {status.active_replicas}',
            f"# HELP kafka_consumergroup_lag Number of unprocessed messages per consumer group and topic.",
            f"# TYPE kafka_consumergroup_lag gauge",
        ]

        for group, topics in consumer_lags.items():
            for topic, lag in topics.items():
                lines.append(
                    f'kafka_consumergroup_lag{{cluster="{self.cluster_name}",group="{group}",topic="{topic}"}} {lag}'
                )

        for b in status.brokers:
            lines.append(
                f'kafka_broker_disk_used_ratio{{cluster="{self.cluster_name}",broker_id="{b.broker_id}"}} {(b.disk_used_gb / b.disk_capacity_gb):.4f}'
            )

        return "\n".join(lines) + "\n"


# -------------------------------------------------------------------------
# GitOps & Helm Template Synthesis
# -------------------------------------------------------------------------

class ArgoCDGitOpsValidator:
    """Synthesizes Helm values and validates ArgoCD Application manifests."""

    @staticmethod
    def generate_helm_values(spec: KafkaClusterSpec) -> Dict[str, Any]:
        return {
            "cluster": {
                "name": spec.name,
                "version": spec.kafka_version,
                "provider": spec.cloud_provider.value,
                "region": spec.region,
                "replicas": spec.broker_replicas,
                "storage": {
                    "sizeGb": spec.resources.storage_gb,
                    "type": spec.resources.storage_type,
                },
                "rackAwareness": spec.rack_awareness_enabled,
                "cruiseControl": spec.cruise_control_enabled,
            },
            "topics": [
                {
                    "name": t.name,
                    "partitions": t.partitions,
                    "replicationFactor": t.replication_factor,
                    "minInsyncReplicas": t.min_insync_replicas,
                }
                for t in spec.topics.values()
            ],
        }

    @staticmethod
    def generate_argocd_application(
        spec: KafkaClusterSpec, repo_url: str, path: str, target_revision: str = "HEAD"
    ) -> Dict[str, Any]:
        return {
            "apiVersion": "argoproj.io/v1alpha1",
            "kind": "Application",
            "metadata": {
                "name": f"{spec.name}-gitops",
                "namespace": "argocd",
                "finalizers": ["resources-finalizer.argocd.argoproj.io"],
            },
            "spec": {
                "project": "default",
                "source": {
                    "repoURL": repo_url,
                    "targetRevision": target_revision,
                    "path": path,
                    "helm": {"valueFiles": ["values.yaml"]},
                },
                "destination": {
                    "server": "https://kubernetes.default.svc",
                    "namespace": spec.namespace,
                },
                "syncPolicy": {
                    "automated": {"prune": True, "selfHeal": True},
                    "syncOptions": ["CreateNamespace=true"],
                },
            },
        }


# -------------------------------------------------------------------------
# AIOps Autonomous Incident Triage & Remediation
# -------------------------------------------------------------------------

@dataclasses.dataclass
class IncidentDiagnosis:
    incident_type: str
    severity: str  # P1, P2, P3
    root_cause: str
    suggested_action: str
    automated_remediation_applicable: bool


class AIOpsToilAutomator:
    """Diagnoses common Kafka cluster failure signatures and recommends automated fixes."""

    @staticmethod
    def diagnose_cluster(status: KafkaClusterStatus) -> Optional[IncidentDiagnosis]:
        if status.active_controller_count == 0:
            return IncidentDiagnosis(
                incident_type="CONTROLLER_ELECTION_FAILURE",
                severity="P1",
                root_cause="No active Kafka controller elected; KRaft or ZooKeeper consensus quorum lost.",
                suggested_action="Restart controller pod quorum and verify metadata network connectivity.",
                automated_remediation_applicable=False,
            )

        if status.under_replicated_partitions > 0:
            return IncidentDiagnosis(
                incident_type="UNDER_REPLICATED_PARTITIONS",
                severity="P2",
                root_cause=f"{status.under_replicated_partitions} partitions are out of sync due to broker disk saturation or network partitions.",
                suggested_action="Execute Cruise Control partition rebalance and verify follower replica fetcher threads.",
                automated_remediation_applicable=True,
            )

        # Check for disk saturation on any broker
        for b in status.brokers:
            ratio = b.disk_used_gb / max(1.0, b.disk_capacity_gb)
            if ratio >= 0.90:
                return IncidentDiagnosis(
                    incident_type="BROKER_DISK_EXHAUSTION",
                    severity="P1",
                    root_cause=f"Broker {b.broker_id} has exceeded 90% disk utilization ({b.disk_used_gb}GB / {b.disk_capacity_gb}GB).",
                    suggested_action="Trigger automated PVC storage expansion or offload partitions via Cruise Control.",
                    automated_remediation_applicable=True,
                )

        if status.max_consumer_lag > 50000:
            return IncidentDiagnosis(
                incident_type="CONSUMER_GROUP_STALL",
                severity="P2",
                root_cause=f"Downstream consumer lag spiked to {status.max_consumer_lag} messages, exceeding SLA.",
                suggested_action="Scale out consumer group worker pods or adjust max.poll.records batch size.",
                automated_remediation_applicable=True,
            )

        return None
