"""Zilliz Cloud Platform and Vector Lakebase Orchestrator.

Provides automated multi-tenant Milvus vector database cluster lifecycle management,
elastic vector query and indexing auto-scaling, canary progressive rolling upgrades
with rollback safety, point-in-time state recovery, and SLO-driven telemetry.
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


class ClusterState(str, enum.Enum):
    """Lifecycle states of a Zilliz Cloud managed Milvus cluster."""

    PENDING = "PENDING"
    PROVISIONING = "PROVISIONING"
    RUNNING = "RUNNING"
    UPGRADING = "UPGRADING"
    SCALING = "SCALING"
    RECOVERING = "RECOVERING"
    STOPPED = "STOPPED"
    FAILED = "FAILED"


class CloudProvider(str, enum.Enum):
    """Supported cloud infrastructure providers."""

    AWS = "AWS"
    GCP = "GCP"
    AZURE = "AZURE"


class IndexType(str, enum.Enum):
    """Vector indexing algorithms supported in Vector Lakebase."""

    HNSW = "HNSW"
    IVF_FLAT = "IVF_FLAT"
    DISKANN = "DISKANN"
    AUTOINDEX = "AUTOINDEX"


@dataclass
class TenantQuota:
    """Resource quotas allocated to a Zilliz Cloud tenant."""

    max_clusters: int = 5
    max_cpu_cores: int = 64
    max_memory_gb: int = 256
    max_storage_gb: int = 2048
    max_vectors_millions: float = 100.0


@dataclass
class MilvusClusterSpec:
    """Specification for a managed Milvus vector database cluster."""

    cluster_id: str
    tenant_id: str
    cloud_provider: CloudProvider
    region: str
    milvus_version: str = "v2.4.5"
    query_nodes: int = 2
    index_nodes: int = 1
    data_nodes: int = 2
    cpu_per_node: int = 4
    memory_gb_per_node: int = 16
    storage_gb: int = 100
    vector_dimension: int = 1536
    index_type: IndexType = IndexType.HNSW

    @property
    def total_cpu_cores(self) -> int:
        return (self.query_nodes + self.index_nodes + self.data_nodes) * self.cpu_per_node

    @property
    def total_memory_gb(self) -> int:
        return (self.query_nodes + self.index_nodes + self.data_nodes) * self.memory_gb_per_node


@dataclass
class BackupSnapshot:
    """Point-in-time backup snapshot for a Milvus cluster."""

    snapshot_id: str
    cluster_id: str
    created_at: float
    milvus_version: str
    vector_count_millions: float
    index_type: IndexType
    status: str = "COMPLETED"


@dataclass
class ClusterRuntime:
    """Active runtime state of a Milvus cluster."""

    spec: MilvusClusterSpec
    state: ClusterState = ClusterState.PENDING
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    active_query_nodes: int = 0
    active_index_nodes: int = 0
    active_data_nodes: int = 0
    current_vectors_millions: float = 0.0
    query_qps: float = 0.0
    p99_latency_ms: float = 0.0
    error_rate: float = 0.0
    version_history: List[str] = field(default_factory=list)
    snapshots: Dict[str, BackupSnapshot] = field(default_factory=dict)


@dataclass
class SLOBreachRecord:
    """Record of an SLO breach event."""

    timestamp: float
    metric: str
    observed_value: float
    threshold_value: float
    severity: str


class ZillizCloudPlatformOrchestrator:
    """Orchestrator for Zilliz Cloud multi-tenant Milvus vector database platform.

    Handles cluster provisioning, tenant isolation, auto-scaling, rolling upgrades,
    backups, and SLO-driven telemetry monitoring.
    """

    def __init__(
        self,
        slo_p99_latency_threshold_ms: float = 15.0,
        slo_error_rate_threshold: float = 0.001,
    ) -> None:
        self.slo_p99_latency_threshold_ms = slo_p99_latency_threshold_ms
        self.slo_error_rate_threshold = slo_error_rate_threshold
        self._tenants: Dict[str, TenantQuota] = {}
        self._clusters: Dict[str, ClusterRuntime] = {}
        self._breaches: Dict[str, List[SLOBreachRecord]] = {}

    def register_tenant(self, tenant_id: str, quota: Optional[TenantQuota] = None) -> None:
        """Register a new tenant with resource quotas."""
        self._tenants[tenant_id] = quota or TenantQuota()

    def get_tenant_quota(self, tenant_id: str) -> TenantQuota:
        """Retrieve resource quota for a tenant."""
        if tenant_id not in self._tenants:
            self.register_tenant(tenant_id)
        return self._tenants[tenant_id]

    def _get_tenant_usage(self, tenant_id: str) -> Tuple[int, int, int]:
        """Calculate current cluster count, total CPU, and total memory used by a tenant."""
        clusters = [
            c for c in self._clusters.values() if c.spec.tenant_id == tenant_id and c.state != ClusterState.STOPPED
        ]
        total_cpu = sum(c.spec.total_cpu_cores for c in clusters)
        total_memory = sum(c.spec.total_memory_gb for c in clusters)
        return len(clusters), total_cpu, total_memory

    def provision_cluster(self, spec: MilvusClusterSpec) -> ClusterRuntime:
        """Provision a new managed Milvus vector database cluster with admission checks."""
        if spec.cluster_id in self._clusters:
            raise ValueError(f"Cluster with ID '{spec.cluster_id}' already exists")

        quota = self.get_tenant_quota(spec.tenant_id)
        current_count, current_cpu, current_mem = self._get_tenant_usage(spec.tenant_id)

        if current_count + 1 > quota.max_clusters:
            raise PermissionError(f"Tenant '{spec.tenant_id}' exceeds max clusters quota ({quota.max_clusters})")
        if current_cpu + spec.total_cpu_cores > quota.max_cpu_cores:
            raise PermissionError(f"Tenant '{spec.tenant_id}' exceeds CPU quota ({quota.max_cpu_cores} cores)")
        if current_mem + spec.total_memory_gb > quota.max_memory_gb:
            raise PermissionError(f"Tenant '{spec.tenant_id}' exceeds memory quota ({quota.max_memory_gb} GB)")

        runtime = ClusterRuntime(
            spec=spec,
            state=ClusterState.PROVISIONING,
            active_query_nodes=spec.query_nodes,
            active_index_nodes=spec.index_nodes,
            active_data_nodes=spec.data_nodes,
            version_history=[spec.milvus_version],
        )

        # Transition to running upon successful resource allocation
        runtime.state = ClusterState.RUNNING
        runtime.updated_at = time.time()
        self._clusters[spec.cluster_id] = runtime
        self._breaches[spec.cluster_id] = []
        return runtime

    def scale_cluster(self, cluster_id: str, target_query_nodes: int, target_index_nodes: int) -> ClusterRuntime:
        """Scale cluster query and indexing nodes with tenant quota validation."""
        if cluster_id not in self._clusters:
            raise KeyError(f"Cluster '{cluster_id}' not found")

        runtime = self._clusters[cluster_id]
        if runtime.state not in (ClusterState.RUNNING, ClusterState.SCALING):
            raise RuntimeError(f"Cannot scale cluster in state '{runtime.state}'")

        if target_query_nodes < 1 or target_index_nodes < 1:
            raise ValueError("Query nodes and index nodes must be at least 1")

        old_query = runtime.spec.query_nodes
        old_index = runtime.spec.index_nodes

        runtime.spec.query_nodes = target_query_nodes
        runtime.spec.index_nodes = target_index_nodes

        # Validate against quota
        quota = self.get_tenant_quota(runtime.spec.tenant_id)
        _, current_cpu, current_mem = self._get_tenant_usage(runtime.spec.tenant_id)

        if current_cpu > quota.max_cpu_cores or current_mem > quota.max_memory_gb:
            # Rollback spec change
            runtime.spec.query_nodes = old_query
            runtime.spec.index_nodes = old_index
            raise PermissionError("Scaling target exceeds tenant CPU or memory quota")

        runtime.state = ClusterState.SCALING
        runtime.active_query_nodes = target_query_nodes
        runtime.active_index_nodes = target_index_nodes
        runtime.state = ClusterState.RUNNING
        runtime.updated_at = time.time()
        return runtime

    def evaluate_auto_scaling(self, cluster_id: str, current_qps: float, p99_latency_ms: float) -> Tuple[bool, str]:
        """Evaluate auto-scaling heuristics based on search query QPS and P99 latency."""
        if cluster_id not in self._clusters:
            raise KeyError(f"Cluster '{cluster_id}' not found")

        runtime = self._clusters[cluster_id]
        current_query_nodes = runtime.active_query_nodes

        # Scale UP heuristic: High latency or QPS > 500 per node
        if p99_latency_ms > self.slo_p99_latency_threshold_ms or (current_qps / max(1, current_query_nodes)) > 500.0:
            target_nodes = current_query_nodes + 1
            try:
                self.scale_cluster(cluster_id, target_nodes, runtime.active_index_nodes)
                return True, f"Scaled UP query nodes from {current_query_nodes} to {target_nodes}"
            except PermissionError as e:
                return False, f"Scale UP blocked by quota: {e}"

        # Scale DOWN heuristic: Low latency and low QPS (< 100 per node), minimum 2 nodes
        low_latency = p99_latency_ms < (self.slo_p99_latency_threshold_ms * 0.4)
        low_qps = (current_qps / max(1, current_query_nodes)) < 100.0
        if low_latency and low_qps and current_query_nodes > 2:
            target_nodes = current_query_nodes - 1
            self.scale_cluster(cluster_id, target_nodes, runtime.active_index_nodes)
            return True, f"Scaled DOWN query nodes from {current_query_nodes} to {target_nodes}"

        return False, "Cluster sizing optimal, no auto-scaling triggered"

    def execute_rolling_upgrade(
        self,
        cluster_id: str,
        target_version: str,
        canary_steps: int = 2,
        simulated_error_rate: float = 0.0,
    ) -> Tuple[bool, str]:
        """Execute a canary progressive rolling upgrade with automatic rollback on error."""
        if cluster_id not in self._clusters:
            raise KeyError(f"Cluster '{cluster_id}' not found")

        runtime = self._clusters[cluster_id]
        if runtime.state != ClusterState.RUNNING:
            raise RuntimeError(f"Cannot upgrade cluster in state '{runtime.state}'")

        previous_version = runtime.spec.milvus_version
        runtime.state = ClusterState.UPGRADING
        runtime.updated_at = time.time()

        for step in range(1, canary_steps + 1):
            # If error rate spikes during canary phase, trigger instant rollback
            if simulated_error_rate > self.slo_error_rate_threshold:
                runtime.state = ClusterState.RECOVERING
                runtime.spec.milvus_version = previous_version
                runtime.state = ClusterState.RUNNING
                runtime.updated_at = time.time()
                return False, (
                    f"Canary step {step} failed: error rate {simulated_error_rate} "
                    f"exceeded threshold. Rolled back to {previous_version}."
                )

        # Successful upgrade completion
        runtime.spec.milvus_version = target_version
        runtime.version_history.append(target_version)
        runtime.state = ClusterState.RUNNING
        runtime.updated_at = time.time()
        return True, f"Successfully upgraded cluster to {target_version} across {canary_steps} canary stages."

    def create_backup_snapshot(self, cluster_id: str, snapshot_id: str) -> BackupSnapshot:
        """Create a point-in-time snapshot for vector collections and index state."""
        if cluster_id not in self._clusters:
            raise KeyError(f"Cluster '{cluster_id}' not found")

        runtime = self._clusters[cluster_id]
        snapshot = BackupSnapshot(
            snapshot_id=snapshot_id,
            cluster_id=cluster_id,
            created_at=time.time(),
            milvus_version=runtime.spec.milvus_version,
            vector_count_millions=runtime.current_vectors_millions,
            index_type=runtime.spec.index_type,
        )
        runtime.snapshots[snapshot_id] = snapshot
        return snapshot

    def restore_from_backup(self, cluster_id: str, snapshot_id: str) -> bool:
        """Restore cluster state from a saved backup snapshot."""
        if cluster_id not in self._clusters:
            raise KeyError(f"Cluster '{cluster_id}' not found")

        runtime = self._clusters[cluster_id]
        if snapshot_id not in runtime.snapshots:
            raise KeyError(f"Snapshot '{snapshot_id}' not found in cluster '{cluster_id}'")

        snapshot = runtime.snapshots[snapshot_id]
        runtime.state = ClusterState.RECOVERING
        runtime.spec.milvus_version = snapshot.milvus_version
        runtime.spec.index_type = snapshot.index_type
        runtime.current_vectors_millions = snapshot.vector_count_millions
        runtime.state = ClusterState.RUNNING
        runtime.updated_at = time.time()
        return True

    def record_telemetry(
        self,
        cluster_id: str,
        query_qps: float,
        p99_latency_ms: float,
        error_rate: float,
        current_vectors_millions: Optional[float] = None,
    ) -> None:
        """Record runtime observability telemetry and track SLO breaches."""
        if cluster_id not in self._clusters:
            raise KeyError(f"Cluster '{cluster_id}' not found")

        runtime = self._clusters[cluster_id]
        runtime.query_qps = query_qps
        runtime.p99_latency_ms = p99_latency_ms
        runtime.error_rate = error_rate
        if current_vectors_millions is not None:
            runtime.current_vectors_millions = current_vectors_millions
        runtime.updated_at = time.time()

        now = time.time()
        if p99_latency_ms > self.slo_p99_latency_threshold_ms:
            self._breaches[cluster_id].append(
                SLOBreachRecord(
                    timestamp=now,
                    metric="p99_latency_ms",
                    observed_value=p99_latency_ms,
                    threshold_value=self.slo_p99_latency_threshold_ms,
                    severity="HIGH" if p99_latency_ms > self.slo_p99_latency_threshold_ms * 1.5 else "MEDIUM",
                )
            )

        if error_rate > self.slo_error_rate_threshold:
            self._breaches[cluster_id].append(
                SLOBreachRecord(
                    timestamp=now,
                    metric="error_rate",
                    observed_value=error_rate,
                    threshold_value=self.slo_error_rate_threshold,
                    severity="CRITICAL",
                )
            )

    def get_slo_report(self, cluster_id: str) -> Dict[str, Any]:
        """Generate comprehensive SLO compliance and health status report."""
        if cluster_id not in self._clusters:
            raise KeyError(f"Cluster '{cluster_id}' not found")

        runtime = self._clusters[cluster_id]
        breaches = self._breaches.get(cluster_id, [])

        latency_compliant = runtime.p99_latency_ms <= self.slo_p99_latency_threshold_ms
        error_rate_compliant = runtime.error_rate <= self.slo_error_rate_threshold
        overall_compliant = latency_compliant and error_rate_compliant

        # Score out of 100
        health_score = 100.0
        if not latency_compliant:
            health_score -= 25.0
        if not error_rate_compliant:
            health_score -= 35.0
        health_score -= min(30.0, len(breaches) * 5.0)

        return {
            "cluster_id": cluster_id,
            "tenant_id": runtime.spec.tenant_id,
            "state": runtime.state.value,
            "milvus_version": runtime.spec.milvus_version,
            "cloud_provider": runtime.spec.cloud_provider.value,
            "region": runtime.spec.region,
            "query_nodes": runtime.active_query_nodes,
            "index_nodes": runtime.active_index_nodes,
            "data_nodes": runtime.active_data_nodes,
            "metrics": {
                "query_qps": runtime.query_qps,
                "p99_latency_ms": runtime.p99_latency_ms,
                "error_rate": runtime.error_rate,
                "vector_count_millions": runtime.current_vectors_millions,
            },
            "slo_compliance": {
                "p99_latency_compliant": latency_compliant,
                "error_rate_compliant": error_rate_compliant,
                "overall_compliant": overall_compliant,
                "breach_count": len(breaches),
            },
            "health_score": max(0.0, health_score),
            "recent_breaches": [
                {
                    "metric": b.metric,
                    "observed": b.observed_value,
                    "threshold": b.threshold_value,
                    "severity": b.severity,
                }
                for b in breaches[-5:]
            ],
        }
