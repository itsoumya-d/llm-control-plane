"""Pinecone Experience Platform Telemetry & Usage-Based Billing Engine.

Designed for Pinecone's Experience Platform engineering team:
- Sub-second Read Unit (RU) & Write Unit (WU) vector query telemetry calculations.
- Idempotent usage metering with deduplication hashing to prevent double-billing.
- Multi-tenant namespace spend attribution and proactive threshold alerting.
- Resilient billing integration gateway with schema drift validation and outbox dispatch.
- FinOps Open Cost and Usage Specification (FOCUS v1.4) enterprise export generator.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


class OperationType(str, Enum):
    QUERY = "QUERY"
    FETCH = "FETCH"
    UPSERT = "UPSERT"
    DELETE = "DELETE"
    LIST = "LIST"


class AlertSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class VectorTelemetryEvent:
    event_id: str
    account_id: str
    project_id: str
    index_name: str
    namespace: str
    operation: OperationType
    vector_dim: int
    vector_count: int
    top_k: int = 10
    metadata_bytes: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    read_units: float = 0.0
    write_units: float = 0.0
    cost_usd: float = 0.0
    dedup_hash: str = ""

    def __post_init__(self) -> None:
        if not self.dedup_hash:
            payload = (
                f"{self.event_id}:{self.account_id}:{self.project_id}:"
                f"{self.index_name}:{self.namespace}:{self.operation.value}:"
                f"{self.timestamp.isoformat()}"
            )
            self.dedup_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()


class VectorTelemetryEngine:
    """Engine calculating RUs, WUs, and storage costs for Pinecone serverless."""

    RU_COST_USD = 0.000008  # ~$8.00 per 1M Read Units
    WU_COST_USD = 0.000016  # ~$16.00 per 1M Write Units
    STORAGE_GB_MONTH_USD = 0.33  # ~$0.33 per GB/month

    def __init__(self) -> None:
        self._events: List[VectorTelemetryEvent] = []
        self._seen_hashes: Set[str] = set()

    def calculate_read_units(
        self,
        operation: OperationType,
        vector_dim: int,
        top_k: int = 10,
        metadata_bytes: int = 0,
    ) -> float:
        """Calculate Read Units (RUs) based on data scanned and retrieved.

        Rule: 1 RU per 2KB scanned/retrieved. Vector float32 = 4 bytes per dim.
        """
        if operation not in (OperationType.QUERY, OperationType.FETCH, OperationType.LIST):
            return 0.0

        vector_bytes = top_k * vector_dim * 4
        total_scanned_bytes = vector_bytes + metadata_bytes
        # Minimum 1 RU per query operation
        units = max(1.0, total_scanned_bytes / 2048.0)
        return round(units, 4)

    def calculate_write_units(
        self,
        operation: OperationType,
        vector_dim: int,
        vector_count: int,
        metadata_bytes: int = 0,
    ) -> float:
        """Calculate Write Units (WUs) for upserts/updates.

        Rule: 1 WU per 2KB written. Minimum 1 WU per write operation.
        """
        if operation != OperationType.UPSERT:
            return 0.0

        written_bytes = (vector_count * vector_dim * 4) + metadata_bytes
        units = max(1.0, written_bytes / 2048.0)
        return round(units, 4)

    def calculate_storage_gb(
        self,
        vector_count: int,
        vector_dim: int,
        metadata_bytes_avg: int = 256,
        replication_factor: int = 1,
    ) -> float:
        """Calculate active storage consumption in Gigabytes."""
        bytes_per_vector = (vector_dim * 4) + metadata_bytes_avg
        total_bytes = vector_count * bytes_per_vector * replication_factor
        return round(total_bytes / (1024.0**3), 6)

    def record_event(
        self,
        event_id: str,
        account_id: str,
        project_id: str,
        index_name: str,
        namespace: str,
        operation: OperationType,
        vector_dim: int,
        vector_count: int,
        top_k: int = 10,
        metadata_bytes: int = 0,
    ) -> Tuple[bool, VectorTelemetryEvent]:
        """Record a telemetry event with idempotency and cost calculation."""
        rus = self.calculate_read_units(operation, vector_dim, top_k, metadata_bytes)
        wus = self.calculate_write_units(
            operation, vector_dim, vector_count, metadata_bytes
        )
        cost = (rus * self.RU_COST_USD) + (wus * self.WU_COST_USD)

        event = VectorTelemetryEvent(
            event_id=event_id,
            account_id=account_id,
            project_id=project_id,
            index_name=index_name,
            namespace=namespace,
            operation=operation,
            vector_dim=vector_dim,
            vector_count=vector_count,
            top_k=top_k,
            metadata_bytes=metadata_bytes,
            read_units=rus,
            write_units=wus,
            cost_usd=round(cost, 6),
        )

        if event.dedup_hash in self._seen_hashes:
            return False, event  # Duplicate event detected and suppressed

        self._seen_hashes.add(event.dedup_hash)
        self._events.append(event)
        return True, event

    def aggregate_by_namespace(
        self, account_id: str, project_id: str
    ) -> Dict[str, Dict[str, Any]]:
        """Aggregate usage telemetry grouped by namespace for billing transparency."""
        summary: Dict[str, Dict[str, Any]] = {}
        for ev in self._events:
            if ev.account_id != account_id or ev.project_id != project_id:
                continue
            ns = ev.namespace or "default"
            if ns not in summary:
                summary[ns] = {
                    "total_events": 0,
                    "read_units": 0.0,
                    "write_units": 0.0,
                    "total_cost_usd": 0.0,
                    "operations": {},
                }
            summary[ns]["total_events"] += 1
            summary[ns]["read_units"] += ev.read_units
            summary[ns]["write_units"] += ev.write_units
            summary[ns]["total_cost_usd"] += ev.cost_usd
            op_name = ev.operation.value
            summary[ns]["operations"][op_name] = (
                summary[ns]["operations"].get(op_name, 0) + 1
            )

        for _ns, data in summary.items():
            data["read_units"] = round(data["read_units"], 4)
            data["write_units"] = round(data["write_units"], 4)
            data["total_cost_usd"] = round(data["total_cost_usd"], 6)

        return summary


@dataclass
class SpendBudgetRule:
    rule_id: str
    account_id: str
    project_id: str
    budget_limit_usd: float
    warning_threshold_pct: float = 80.0
    critical_threshold_pct: float = 100.0
    webhook_url: Optional[str] = None


class SpendAlertManager:
    """Manages proactive spend threshold notifications and budget enforcement."""

    def __init__(self) -> None:
        self._rules: Dict[str, SpendBudgetRule] = {}
        self._alert_history: List[Dict[str, Any]] = []

    def register_rule(self, rule: SpendBudgetRule) -> None:
        self._rules[rule.rule_id] = rule

    def evaluate_spend(
        self, account_id: str, project_id: str, current_spend_usd: float
    ) -> List[Dict[str, Any]]:
        """Evaluate current spend against registered budget rules."""
        triggered_alerts: List[Dict[str, Any]] = []
        for rule in self._rules.values():
            if rule.account_id != account_id or rule.project_id != project_id:
                continue

            pct_used = (current_spend_usd / rule.budget_limit_usd) * 100.0
            severity = None
            if pct_used >= rule.critical_threshold_pct:
                severity = AlertSeverity.CRITICAL
            elif pct_used >= rule.warning_threshold_pct:
                severity = AlertSeverity.WARNING

            if severity:
                alert = {
                    "rule_id": rule.rule_id,
                    "account_id": account_id,
                    "project_id": project_id,
                    "severity": severity.value,
                    "current_spend_usd": round(current_spend_usd, 4),
                    "budget_limit_usd": rule.budget_limit_usd,
                    "percent_consumed": round(pct_used, 2),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "webhook_target": rule.webhook_url,
                }
                self._alert_history.append(alert)
                triggered_alerts.append(alert)

        return triggered_alerts


class BillingIntegrationGateway:
    """Resilient gateway for external billing synchronization (Stripe/Invoicing)."""

    REQUIRED_SCHEMA_KEYS = {
        "account_id",
        "project_id",
        "period_start",
        "period_end",
        "read_units",
        "write_units",
        "total_amount_usd",
    }

    def __init__(self) -> None:
        self._outbox_queue: List[Dict[str, Any]] = []

    def validate_billing_payload(self, payload: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate payload against schema to prevent schema drift."""
        missing = self.REQUIRED_SCHEMA_KEYS - set(payload.keys())
        if missing:
            return False, f"Schema drift detected. Missing required keys: {sorted(missing)}"

        if payload["total_amount_usd"] < 0:
            return False, "Invalid billing payload: total_amount_usd must be >= 0"

        return True, "Payload valid"

    def enqueue_for_sync(self, payload: Dict[str, Any]) -> Tuple[bool, str]:
        """Enqueue validated billing event into transactional outbox."""
        is_valid, msg = self.validate_billing_payload(payload)
        if not is_valid:
            return False, msg

        outbox_entry = {
            "entry_id": hashlib.sha256(
                json.dumps(payload, sort_keys=True).encode("utf-8")
            ).hexdigest()[:16],
            "payload": payload,
            "status": "QUEUED",
            "enqueued_at": datetime.now(timezone.utc).isoformat(),
            "retry_count": 0,
        }
        self._outbox_queue.append(outbox_entry)
        return True, outbox_entry["entry_id"]

    def reconcile_invoice(
        self, telemetry_total_usd: float, invoice_total_usd: float, tolerance: float = 0.01
    ) -> Tuple[bool, float]:
        """Reconcile metered telemetry totals against invoiced line items."""
        diff = abs(telemetry_total_usd - invoice_total_usd)
        is_reconciled = diff <= tolerance
        return is_reconciled, round(diff, 6)


class FinOpsCostExporter:
    """FinOps Open Cost and Usage Specification (FOCUS v1.4) Export Generator."""

    @staticmethod
    def generate_focus_records(
        events: List[VectorTelemetryEvent],
        currency: str = "USD",
    ) -> List[Dict[str, Any]]:
        """Map internal vector telemetry events into standardized FOCUS v1.4 format."""
        records: List[Dict[str, Any]] = []
        for ev in events:
            records.append(
                {
                    "ChargePeriodStart": ev.timestamp.isoformat(),
                    "ChargePeriodEnd": ev.timestamp.isoformat(),
                    "BillingAccountId": ev.account_id,
                    "SubAccountId": ev.project_id,
                    "ServiceName": "Pinecone Serverless Vector Database",
                    "ServiceCategory": "AI / Vector Search",
                    "ResourceName": f"{ev.index_name}/{ev.namespace or 'default'}",
                    "OperationType": ev.operation.value,
                    "ConsumedQuantity": (
                        ev.read_units
                        if ev.operation in (OperationType.QUERY, OperationType.FETCH)
                        else ev.write_units
                    ),
                    "ConsumedUnit": (
                        "ReadUnit"
                        if ev.operation in (OperationType.QUERY, OperationType.FETCH)
                        else "WriteUnit"
                    ),
                    "BilledCost": ev.cost_usd,
                    "BillingCurrency": currency,
                    "ChargeDescription": f"Pinecone serverless {ev.operation.value} vector compute",
                }
            )
        return records

    @staticmethod
    def export_focus_csv(events: List[VectorTelemetryEvent]) -> str:
        """Export FOCUS v1.4 records as an enterprise RFC-4180 CSV string."""
        records = FinOpsCostExporter.generate_focus_records(events)
        if not records:
            return ""

        output = io.StringIO()
        fieldnames = list(records[0].keys())
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(r)
        return output.getvalue()
