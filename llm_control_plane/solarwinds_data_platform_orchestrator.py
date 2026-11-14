"""SolarWinds Data Platform Orchestrator.

Provides enterprise BigQuery platform administration, Dataplex-based column-level
PII masking and fine-grained IAM governance, Fivetran ELT ingestion reliability
monitoring, schema drift quarantine, and query cost/slot optimization.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class DataplexIAMRole(str, Enum):
    """Fine-grained Dataplex access roles for sensitive data."""

    FINE_GRAINED_READER = "roles/datacatalog.categoryFineGrainedReader"
    MASKED_READER = "roles/bigquery.maskedReader"
    NO_ACCESS = "roles/noAccess"


class MaskingRule(str, Enum):
    """Column-level data masking policies enforced via Dataplex policy tags."""

    HASH_SHA256 = "HASH_SHA256"
    PARTIAL_EMAIL = "PARTIAL_EMAIL"
    PARTIAL_PHONE = "PARTIAL_PHONE"
    NULLIFY = "NULLIFY"
    PLAINTEXT = "PLAINTEXT"


class FivetranSyncState(str, Enum):
    """Fivetran connector synchronization lifecycle states."""

    SUCCESS = "SUCCESS"
    WARNING = "WARNING"
    FAILED = "FAILED"
    SYNCING = "SYNCING"
    QUARANTINED = "QUARANTINED"


class WorkloadTier(str, Enum):
    """BigQuery dynamic slot reservation workload tiers."""

    CRITICAL_ELT = "CRITICAL_ELT"
    BI_REPORTING = "BI_REPORTING"
    ADHOC_ANALYTICS = "ADHOC_ANALYTICS"


@dataclass
class PolicyTag:
    """Dataplex policy tag metadata governing a sensitive column."""

    tag_id: str
    display_name: str
    masking_rule: MaskingRule
    taxonomy: str
    description: str = ""


@dataclass
class BigQueryTableMetadata:
    """BigQuery table architecture schema and governance configuration."""

    project_id: str
    dataset_id: str
    table_id: str
    schema: dict[str, str]  # column_name -> data_type
    partition_field: str | None = None
    clustering_fields: list[str] = field(default_factory=list)
    policy_tags: dict[str, str] = field(default_factory=dict)  # column_name -> tag_id

    @property
    def fully_qualified_id(self) -> str:
        """Returns standard BigQuery standard SQL table identifier."""
        return f"{self.project_id}.{self.dataset_id}.{self.table_id}"


@dataclass
class QueryCostEstimate:
    """Calculated BigQuery query execution cost and slot assessment."""

    estimated_bytes_scanned: int
    estimated_cost_usd: float
    is_partition_pruned: bool
    requires_approval: bool
    assigned_tier: WorkloadTier
    allocated_slots: int


@dataclass
class SchemaDriftReport:
    """Schema drift detection summary across Fivetran source and BigQuery target."""

    connector_id: str
    table_name: str
    added_columns: list[str]
    altered_columns: dict[str, tuple[str, str]]  # col -> (old_type, new_type)
    deleted_columns: list[str]
    is_breaking_change: bool
    quarantine_triggered: bool


class DataplexGovernanceEngine:
    """Administers Dataplex policy tags, fine-grained IAM roles, and column masking."""

    def __init__(self, default_taxonomy: str = "enterprise_pii_governance") -> None:
        self.default_taxonomy = default_taxonomy
        self.policy_tags: dict[str, PolicyTag] = {}
        self.user_role_bindings: dict[tuple[str, str], DataplexIAMRole] = {}  # (user_email, tag_id) -> role

    def register_policy_tag(
        self,
        tag_id: str,
        display_name: str,
        masking_rule: MaskingRule,
        description: str = "",
        taxonomy: str | None = None,
    ) -> PolicyTag:
        """Registers a Dataplex policy tag in the governance catalog."""
        tag = PolicyTag(
            tag_id=tag_id,
            display_name=display_name,
            masking_rule=masking_rule,
            taxonomy=taxonomy or self.default_taxonomy,
            description=description,
        )
        self.policy_tags[tag_id] = tag
        return tag

    def bind_user_role(self, user_email: str, tag_id: str, role: DataplexIAMRole) -> None:
        """Grants fine-grained or masked reader IAM permissions to a user or service account."""
        if tag_id not in self.policy_tags:
            raise KeyError(f"Policy tag {tag_id} does not exist in catalog.")
        self.user_role_bindings[(user_email, tag_id)] = role

    def get_effective_role(self, user_email: str, tag_id: str) -> DataplexIAMRole:
        """Resolves effective Dataplex role adhering to least-privilege principles."""
        return self.user_role_bindings.get((user_email, tag_id), DataplexIAMRole.NO_ACCESS)

    def mask_column_value(self, value: Any, rule: MaskingRule) -> Any:
        """Applies column-level masking transformation to sensitive attributes."""
        if value is None:
            return None

        val_str = str(value)

        if rule == MaskingRule.PLAINTEXT:
            return value

        if rule == MaskingRule.NULLIFY:
            return None

        if rule == MaskingRule.HASH_SHA256:
            return hashlib.sha256(val_str.encode("utf-8")).hexdigest()

        if rule == MaskingRule.PARTIAL_EMAIL:
            if "@" in val_str:
                local, domain = val_str.split("@", 1)
                prefix = local[0] if local else ""
                return f"{prefix}***@{domain}"
            return "***masked_email***"

        if rule == MaskingRule.PARTIAL_PHONE:
            digits = "".join(filter(str.isdigit, val_str))
            if len(digits) >= 4:
                return f"***-***-{digits[-4:]}"
            return "***masked_phone***"

        return "***MASKED***"

    def apply_governance_to_record(
        self,
        record: dict[str, Any],
        table_meta: BigQueryTableMetadata,
        caller_email: str,
    ) -> dict[str, Any]:
        """Applies Dataplex column masking to a record based on caller's IAM role."""
        sanitized = dict(record)

        for col, tag_id in table_meta.policy_tags.items():
            if col not in sanitized:
                continue

            tag = self.policy_tags.get(tag_id)
            if not tag:
                continue

            role = self.get_effective_role(caller_email, tag_id)

            if role == DataplexIAMRole.FINE_GRAINED_READER:
                # Allowed to view raw unmasked plaintext
                continue
            elif role == DataplexIAMRole.MASKED_READER:
                # Apply column masking policy rule
                sanitized[col] = self.mask_column_value(sanitized[col], tag.masking_rule)
            else:
                # NO_ACCESS: Column is completely redacted
                sanitized[col] = None

        return sanitized


class BigQueryPlatformOptimizer:
    """Enforces query cost guardrails, partition pruning, and dynamic slot reservations."""

    COST_PER_TB_USD = 6.25  # BigQuery on-demand pricing ($6.25 per TB scanned)
    BYTES_PER_TB = 1024**4

    def __init__(self, high_cost_threshold_usd: float = 25.0) -> None:
        self.high_cost_threshold_usd = high_cost_threshold_usd
        self.slot_pools: dict[WorkloadTier, int] = {
            WorkloadTier.CRITICAL_ELT: 1000,
            WorkloadTier.BI_REPORTING: 500,
            WorkloadTier.ADHOC_ANALYTICS: 200,
        }

    def validate_table_architecture(self, meta: BigQueryTableMetadata) -> list[str]:
        """Audits table schema against enterprise partitioning and clustering best practices."""
        issues: list[str] = []

        if not meta.partition_field:
            issues.append(f"Table {meta.fully_qualified_id} lacks partition_field. Enterprise tables must partition.")

        if len(meta.clustering_fields) > 4:
            issues.append(f"Table {meta.fully_qualified_id} has {len(meta.clustering_fields)} cluster fields (max 4).")

        return issues

    def estimate_query_cost(
        self,
        bytes_scanned: int,
        is_partition_pruned: bool,
        tier: WorkloadTier,
    ) -> QueryCostEstimate:
        """Calculates on-demand query cost in USD and verifies slot budget governance."""
        cost_usd = (bytes_scanned / self.BYTES_PER_TB) * self.COST_PER_TB_USD
        requires_approval = cost_usd >= self.high_cost_threshold_usd or not is_partition_pruned
        allocated_slots = self.slot_pools.get(tier, 100)

        return QueryCostEstimate(
            estimated_bytes_scanned=bytes_scanned,
            estimated_cost_usd=round(cost_usd, 4),
            is_partition_pruned=is_partition_pruned,
            requires_approval=requires_approval,
            assigned_tier=tier,
            allocated_slots=allocated_slots,
        )


class FivetranIngestionReliabilityMonitor:
    """Monitors Fivetran connectors, detects schema drift, and triggers automated quarantine."""

    def __init__(self, max_allowed_lag_minutes: int = 30) -> None:
        self.max_allowed_lag_minutes = max_allowed_lag_minutes
        self.connector_registry: dict[str, dict[str, Any]] = {}
        self.drift_history: list[SchemaDriftReport] = []

    def register_connector(
        self,
        connector_id: str,
        service_type: str,
        target_table: str,
        sync_frequency_minutes: int = 15,
    ) -> None:
        """Registers a Fivetran connector into the health monitoring grid."""
        self.connector_registry[connector_id] = {
            "connector_id": connector_id,
            "service_type": service_type,
            "target_table": target_table,
            "sync_frequency_minutes": sync_frequency_minutes,
            "last_sync_time": datetime.now(timezone.utc),
            "status": FivetranSyncState.SUCCESS,
            "records_synced": 0,
        }

    def evaluate_schema_drift(
        self,
        connector_id: str,
        table_name: str,
        source_schema: dict[str, str],
        existing_schema: dict[str, str],
    ) -> SchemaDriftReport:
        """Detects column additions, type modifications, and breaking alterations."""
        added = [col for col in source_schema if col not in existing_schema]
        deleted = [col for col in existing_schema if col not in source_schema]
        altered: dict[str, tuple[str, str]] = {}

        for col, col_type in source_schema.items():
            if col in existing_schema and existing_schema[col] != col_type:
                altered[col] = (existing_schema[col], col_type)

        # Breaking changes include type conversions or dropped required fields
        is_breaking = len(altered) > 0 or len(deleted) > 0
        quarantine = is_breaking

        report = SchemaDriftReport(
            connector_id=connector_id,
            table_name=table_name,
            added_columns=added,
            altered_columns=altered,
            deleted_columns=deleted,
            is_breaking_change=is_breaking,
            quarantine_triggered=quarantine,
        )

        if quarantine and connector_id in self.connector_registry:
            self.connector_registry[connector_id]["status"] = FivetranSyncState.QUARANTINED

        self.drift_history.append(report)
        return report

    def check_connector_sla(self, connector_id: str, current_time: datetime | None = None) -> bool:
        """Validates whether connector sync latency complies with the maximum allowed SLA."""
        info = self.connector_registry.get(connector_id)
        if not info:
            raise KeyError(f"Connector {connector_id} not registered.")

        now = current_time or datetime.now(timezone.utc)
        elapsed_minutes = (now - info["last_sync_time"]).total_seconds() / 60.0
        return elapsed_minutes <= self.max_allowed_lag_minutes


class SolarWindsDataPlatformOrchestrator:
    """Comprehensive Enterprise Data Platform Orchestrator for SolarWinds.

    Unifies Dataplex security, BigQuery cost/slot optimization, and Fivetran ELT reliability.
    """

    def __init__(self) -> None:
        self.governance = DataplexGovernanceEngine()
        self.optimizer = BigQueryPlatformOptimizer()
        self.ingestion = FivetranIngestionReliabilityMonitor()

    def setup_telemetry_governance(
        self,
        project_id: str = "swi-telemetry-prod",
        dataset_id: str = "observability_analytics",
    ) -> BigQueryTableMetadata:
        """Bootstraps production Dataplex policy tags and governs telemetry tables."""
        tag_customer = self.governance.register_policy_tag(
            tag_id="tag_customer_id",
            display_name="Customer Identifier",
            masking_rule=MaskingRule.HASH_SHA256,
            description="Pseudonymized SHA-256 customer identifier",
        )
        tag_email = self.governance.register_policy_tag(
            tag_id="tag_user_email",
            display_name="User Email Address",
            masking_rule=MaskingRule.PARTIAL_EMAIL,
            description="Masked email retaining first character and domain",
        )

        schema = {
            "event_id": "STRING",
            "timestamp": "TIMESTAMP",
            "customer_id": "STRING",
            "user_email": "STRING",
            "metric_name": "STRING",
            "metric_value": "FLOAT64",
        }

        policy_tags = {
            "customer_id": tag_customer.tag_id,
            "user_email": tag_email.tag_id,
        }

        table_meta = BigQueryTableMetadata(
            project_id=project_id,
            dataset_id=dataset_id,
            table_id="telemetry_metrics_v1",
            schema=schema,
            partition_field="timestamp",
            clustering_fields=["customer_id", "metric_name"],
            policy_tags=policy_tags,
        )

        return table_meta
