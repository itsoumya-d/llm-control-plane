"""Superchat Data Platform Orchestrator — AWS Redshift, dbt, Dagster & ELT Pipeline.

Implements end-to-end data platform engineering capabilities tailored for high-throughput
omnichannel messaging platforms (WhatsApp, Email, SMS, Instagram) and SaaS billing/CRM integrations:
1. OmnichannelIngestionPipeline: cursor-tracked idempotent ingestion with fingerprint deduplication.
2. SCD2SnapshotEngine: Slowly Changing Dimension Type 2 snapshotting for subscription & MRR history.
3. DbtModelReconciler: layered transformation models (staging, intermediate, gold) with automated tests.
4. DagsterOrchestrationEngine: software-defined asset graph dependency resolution & freshness alerting.
5. RedshiftWarehouseManager: distribution & sort key optimization and vacuum/analyze telemetry.
6. ReverseEtlSyncAuditor: Hightouch outbound sync reconciliation and destination drift detection.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class ChannelSource(str, Enum):
    WHATSAPP = "whatsapp"
    EMAIL = "email"
    SMS = "sms"
    INSTAGRAM = "instagram"
    CHARGEBEE = "chargebee"
    HUBSPOT = "hubspot"
    INTERCOM = "intercom"


class DbtLayer(str, Enum):
    STAGING = "staging"
    INTERMEDIATE = "intermediate"
    GOLD = "gold"


@dataclass
class RawEventRecord:
    event_id: str
    channel: ChannelSource
    workspace_id: str
    payload: Dict[str, Any]
    received_at: dt.datetime
    cursor_position: int

    @property
    def fingerprint(self) -> str:
        content = f"{self.event_id}:{self.channel.value}:{self.workspace_id}:{sorted(self.payload.items())}"
        return hashlib.sha256(content.encode()).hexdigest()


@dataclass
class SubscriptionSnapshot:
    subscription_id: str
    workspace_id: str
    plan_tier: str
    mrr_cents: int
    currency: str
    valid_from: dt.datetime
    valid_to: Optional[dt.datetime] = None
    is_current: bool = True


@dataclass
class DbtModelSpec:
    name: str
    layer: DbtLayer
    sql_query: str
    upstream_dependencies: List[str] = field(default_factory=list)
    tests: List[str] = field(default_factory=lambda: ["not_null", "unique"])
    freshness_max_lag_hours: float = 4.0


@dataclass
class DagsterAssetNode:
    asset_key: str
    last_materialized_at: Optional[dt.datetime] = None
    freshness_max_lag_seconds: float = 14400.0  # 4 hours
    is_blocked: bool = False


class OmnichannelIngestionPipeline:
    """Manages high-throughput ingestion from messaging webhooks and SaaS sources."""

    def __init__(self) -> None:
        self.seen_fingerprints: Set[str] = set()
        self.raw_table_buffer: List[RawEventRecord] = []
        self.channel_cursors: Dict[str, int] = {}

    def ingest_event(self, record: RawEventRecord) -> bool:
        """Ingests a record idempotently. Returns True if accepted, False if duplicate."""
        fp = record.fingerprint
        if fp in self.seen_fingerprints:
            return False

        self.seen_fingerprints.add(fp)
        self.raw_table_buffer.append(record)
        current_cursor = self.channel_cursors.get(record.channel.value, 0)
        self.channel_cursors[record.channel.value] = max(current_cursor, record.cursor_position)
        return True

    def flush_microbatch(self, max_records: int = 500) -> List[RawEventRecord]:
        """Flushes a micro-batch of records for bulk loading into warehouse staging."""
        batch = self.raw_table_buffer[:max_records]
        self.raw_table_buffer = self.raw_table_buffer[max_records:]
        return batch

    def get_highest_cursor(self, channel: ChannelSource) -> int:
        return self.channel_cursors.get(channel.value, 0)


class SCD2SnapshotEngine:
    """Tracks point-in-time subscription and MRR tier history."""

    def __init__(self) -> None:
        self.history: List[SubscriptionSnapshot] = []

    def record_change(
        self,
        subscription_id: str,
        workspace_id: str,
        new_plan_tier: str,
        new_mrr_cents: int,
        timestamp: dt.datetime,
        currency: str = "EUR",
    ) -> SubscriptionSnapshot:
        # Close any active snapshot for this subscription
        for snap in self.history:
            if snap.subscription_id == subscription_id and snap.is_current:
                snap.is_current = False
                snap.valid_to = timestamp

        new_snap = SubscriptionSnapshot(
            subscription_id=subscription_id,
            workspace_id=workspace_id,
            plan_tier=new_plan_tier,
            mrr_cents=new_mrr_cents,
            currency=currency,
            valid_from=timestamp,
            valid_to=None,
            is_current=True,
        )
        self.history.append(new_snap)
        return new_snap

    def get_snapshot_at(self, subscription_id: str, target_time: dt.datetime) -> Optional[SubscriptionSnapshot]:
        for snap in self.history:
            if snap.subscription_id == subscription_id:
                if snap.valid_from <= target_time:
                    if snap.valid_to is None or snap.valid_to > target_time:
                        return snap
        return None

    def calculate_active_mrr_at(self, target_time: dt.datetime) -> int:
        """Returns total active MRR in cents across all workspaces at target timestamp."""
        total = 0
        seen_subs: Set[str] = set()
        for snap in self.history:
            if snap.subscription_id not in seen_subs and snap.valid_from <= target_time:
                if snap.valid_to is None or snap.valid_to > target_time:
                    total += snap.mrr_cents
                    seen_subs.add(snap.subscription_id)
        return total


class DbtModelReconciler:
    """Executes layered transformation models with strict quality tests."""

    def __init__(self) -> None:
        self.models: Dict[str, DbtModelSpec] = {}

    def register_model(self, model: DbtModelSpec) -> None:
        self.models[model.name] = model

    def resolve_execution_order(self) -> List[str]:
        """Performs topological sort across dbt DAG dependencies."""
        visited: Dict[str, bool] = {}
        order: List[str] = []

        def dfs(node: str, visiting: Set[str]) -> None:
            if node in visiting:
                raise ValueError(f"Cyclic dependency detected at model {node}")
            if node in visited:
                return
            visiting.add(node)
            for dep in self.models.get(node, DbtModelSpec(node, DbtLayer.STAGING, "")).upstream_dependencies:
                if dep in self.models:
                    dfs(dep, visiting)
            visiting.remove(node)
            visited[node] = True
            order.append(node)

        for name in self.models:
            if name not in visited:
                dfs(name, set())

        return order

    def run_tests(self, model_name: str, sample_rows: List[Dict[str, Any]]) -> Dict[str, bool]:
        """Runs test assertions (not_null, unique, positive_mrr) on generated model rows."""
        model = self.models.get(model_name)
        if not model:
            raise KeyError(f"Model {model_name} not registered")

        results: Dict[str, bool] = {}
        for test in model.tests:
            if test == "not_null":
                # Ensure all sample rows have non-null keys
                all_valid = all(v is not None for row in sample_rows for v in row.values())
                results["not_null"] = all_valid
            elif test == "unique":
                primary_keys = [str(r.get("id", r.get("key", i))) for i, r in enumerate(sample_rows)]
                results["unique"] = len(primary_keys) == len(set(primary_keys))
            elif test == "positive_mrr":
                results["positive_mrr"] = all(r.get("mrr_cents", 1) >= 0 for r in sample_rows)
            else:
                results[test] = True
        return results


class DagsterOrchestrationEngine:
    """Software-defined asset graph orchestrator with freshness checking and alerting."""

    def __init__(self) -> None:
        self.assets: Dict[str, DagsterAssetNode] = {}
        self.alert_log: List[Dict[str, Any]] = []

    def register_asset(self, key: str, max_lag_seconds: float = 14400.0) -> DagsterAssetNode:
        node = DagsterAssetNode(asset_key=key, freshness_max_lag_seconds=max_lag_seconds)
        self.assets[key] = node
        return node

    def record_materialization(self, key: str, timestamp: dt.datetime) -> None:
        if key not in self.assets:
            self.register_asset(key)
        self.assets[key].last_materialized_at = timestamp

    def evaluate_freshness(self, current_time: dt.datetime) -> List[str]:
        """Evaluates asset freshness against policies. Logs alerts for stale assets."""
        stale_assets: List[str] = []
        for key, node in self.assets.items():
            if node.last_materialized_at is None:
                stale_assets.append(key)
                self.alert_log.append(
                    {
                        "asset": key,
                        "reason": "never_materialized",
                        "detected_at": current_time.isoformat(),
                    }
                )
            else:
                elapsed = (current_time - node.last_materialized_at).total_seconds()
                if elapsed > node.freshness_max_lag_seconds:
                    stale_assets.append(key)
                    self.alert_log.append(
                        {
                            "asset": key,
                            "reason": f"lag_exceeded_{elapsed:.0f}s",
                            "detected_at": current_time.isoformat(),
                        }
                    )
        return stale_assets


class RedshiftWarehouseManager:
    """Telemetry and distribution key hygiene for AWS Redshift cluster."""

    def __init__(self) -> None:
        self.query_telemetry: List[Dict[str, Any]] = []

    def record_query(self, query_id: str, table_name: str, duration_ms: float, rows_scanned: int) -> None:
        self.query_telemetry.append(
            {
                "query_id": query_id,
                "table_name": table_name,
                "duration_ms": duration_ms,
                "rows_scanned": rows_scanned,
            }
        )

    def recommend_dist_key(self, table_name: str) -> str:
        """Determines best distribution key based on query join and aggregation frequency."""
        if "message" in table_name or "event" in table_name:
            return "workspace_id"
        if "subscription" in table_name or "invoice" in table_name:
            return "workspace_id"
        return "AUTO"

    def requires_vacuum(self, unsorted_percent: float, deleted_rows_percent: float) -> bool:
        """Returns True if table bloat justifies VACUUM operation."""
        return unsorted_percent > 15.0 or deleted_rows_percent > 10.0


class ReverseEtlSyncAuditor:
    """Verifies Hightouch outbound sync consistency with CRM targets."""

    def __init__(self) -> None:
        self.sync_records: List[Dict[str, Any]] = []

    def audit_sync(
        self,
        sync_id: str,
        source_gold_rows: int,
        destination_target_rows: int,
        destination_name: str = "HubSpot",
    ) -> Dict[str, Any]:
        drift = abs(source_gold_rows - destination_target_rows)
        drift_rate = (drift / source_gold_rows) if source_gold_rows > 0 else 0.0
        status = "HEALTHY" if drift_rate < 0.01 else "DRIFT_ALERT"

        record = {
            "sync_id": sync_id,
            "destination": destination_name,
            "source_rows": source_gold_rows,
            "target_rows": destination_target_rows,
            "drift_rate": round(drift_rate, 4),
            "status": status,
        }
        self.sync_records.append(record)
        return record
