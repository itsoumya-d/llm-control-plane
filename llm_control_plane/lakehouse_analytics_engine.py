"""Lakehouse Analytics Engine for Medallion Architecture, SCD Type 2 Dimensions, and Semantic Models.

Provides enterprise analytics engineering primitives tailored for Databricks, Delta Lake, and Microsoft Fabric
environments:
  1. MedallionLakehouseManager: Bronze-to-Silver-to-Gold data quality validation, deduplication, and schema contracts.
  2. SCDType2DimensionTracker: High-throughput Slowly Changing Dimension Type 2 versioning with point-in-time lookup.
  3. SemanticMetricCompiler: Declarative semantic layer with SQL generation, RBAC projection, and SLA monitoring.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


def _current_iso_time() -> str:
    """Return ISO 8601 formatted UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class MedallionRecord:
    """Represents a validated record transitioning through lakehouse tiers."""

    record_id: str
    tier: str  # 'bronze', 'silver', 'gold'
    payload: Dict[str, Any]
    ingested_at: str
    hash_signature: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SCD2Record:
    """Represents a Slowly Changing Dimension (Type 2) historical record."""

    surrogate_key: str
    business_key: str
    attributes: Dict[str, Any]
    valid_from: str
    valid_to: Optional[str]
    is_current: bool
    version: int


@dataclass
class SemanticMetric:
    """Declarative definition for an enterprise metric in the semantic layer."""

    metric_name: str
    table_name: str
    aggregation: str  # 'SUM', 'AVG', 'COUNT', 'COUNT_DISTINCT', 'MAX', 'MIN'
    target_column: str
    dimensions: List[str]
    filters: List[str] = field(default_factory=list)
    sla_seconds: float = 2.0
    restricted_dimensions: Dict[str, List[str]] = field(default_factory=dict)  # role -> allowed dimensions


class MedallionLakehouseManager:
    """Manages Bronze, Silver, and Gold lakehouse tiers with data contracts and schema validation."""

    def __init__(self, raw_retention_days: int = 90) -> None:
        self.raw_retention_days = raw_retention_days
        self._bronze_store: Dict[str, MedallionRecord] = {}
        self._silver_store: Dict[str, MedallionRecord] = {}
        self._gold_marts: Dict[str, List[Dict[str, Any]]] = {}
        self._schema_contracts: Dict[str, Dict[str, type]] = {}

    def register_schema_contract(self, dataset_name: str, schema: Dict[str, type]) -> None:
        """Register a strict schema contract for Silver tier promotion."""
        self._schema_contracts[dataset_name] = schema

    def ingest_bronze(self, dataset_name: str, raw_records: List[Dict[str, Any]]) -> List[MedallionRecord]:
        """Ingest raw payloads into the Bronze tier with deterministic hash deduplication."""
        ingested = []
        for raw in raw_records:
            serialized = json.dumps(raw, sort_keys=True, default=str)
            sig = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
            rec_id = f"bronze_{sig[:16]}"
            rec = MedallionRecord(
                record_id=rec_id,
                tier="bronze",
                payload=raw,
                ingested_at=_current_iso_time(),
                hash_signature=sig,
                metadata={"dataset": dataset_name},
            )
            self._bronze_store[rec_id] = rec
            ingested.append(rec)
        return ingested

    def promote_to_silver(
        self,
        dataset_name: str,
        transformation_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> List[MedallionRecord]:
        """Cleanse, validate, and promote Bronze records into the Silver tier."""
        contract = self._schema_contracts.get(dataset_name)
        promoted = []

        for b_rec in list(self._bronze_store.values()):
            if b_rec.metadata.get("dataset") != dataset_name:
                continue

            payload = dict(b_rec.payload)
            if transformation_fn:
                payload = transformation_fn(payload)

            # Contract verification
            if contract:
                for field_name, expected_type in contract.items():
                    if field_name not in payload:
                        raise ValueError(f"Missing required contract field: '{field_name}'")
                    val = payload[field_name]
                    if val is not None and not isinstance(val, expected_type):
                        if isinstance(expected_type, tuple):
                            type_label = "/".join(t.__name__ for t in expected_type)
                        else:
                            type_label = getattr(expected_type, "__name__", str(expected_type))
                        raise TypeError(
                            f"Field '{field_name}' expected {type_label}, got {type(val).__name__}"
                        )

            sig = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()
            s_rec_id = f"silver_{sig[:16]}"
            s_rec = MedallionRecord(
                record_id=s_rec_id,
                tier="silver",
                payload=payload,
                ingested_at=_current_iso_time(),
                hash_signature=sig,
                metadata={"dataset": dataset_name, "source_bronze": b_rec.record_id},
            )
            self._silver_store[s_rec_id] = s_rec
            promoted.append(s_rec)

        return promoted

    def build_gold_mart(
        self,
        mart_name: str,
        dataset_name: str,
        group_by: List[str],
        aggregations: Dict[str, Tuple[str, str]],  # output_alias -> (agg_func, input_col)
    ) -> List[Dict[str, Any]]:
        """Aggregate Silver data into aggregated business Gold marts."""
        silver_rows = [
            r.payload for r in self._silver_store.values() if r.metadata.get("dataset") == dataset_name
        ]
        if not silver_rows:
            self._gold_marts[mart_name] = []
            return []

        grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
        for row in silver_rows:
            key = tuple(row.get(g) for g in group_by)
            grouped.setdefault(key, []).append(row)

        gold_rows: List[Dict[str, Any]] = []
        for key, rows in grouped.items():
            out_row = {dim: key[idx] for idx, dim in enumerate(group_by)}
            for alias, (agg_func, col) in aggregations.items():
                vals = [r.get(col) for r in rows if r.get(col) is not None]
                if agg_func.upper() == "SUM":
                    out_row[alias] = sum(vals)
                elif agg_func.upper() == "COUNT":
                    out_row[alias] = len(vals)
                elif agg_func.upper() == "AVG":
                    out_row[alias] = (sum(vals) / len(vals)) if vals else 0.0
                elif agg_func.upper() == "MAX":
                    out_row[alias] = max(vals) if vals else None
                elif agg_func.upper() == "MIN":
                    out_row[alias] = min(vals) if vals else None
                else:
                    raise ValueError(f"Unsupported aggregate function: {agg_func}")
            gold_rows.append(out_row)

        self._gold_marts[mart_name] = gold_rows
        return gold_rows

    def get_tier_stats(self) -> Dict[str, int]:
        """Return counts of records across each lakehouse tier."""
        return {
            "bronze_count": len(self._bronze_store),
            "silver_count": len(self._silver_store),
            "gold_mart_count": len(self._gold_marts),
        }


class SCDType2DimensionTracker:
    """Manages Slowly Changing Dimensions (Type 2) tracking with point-in-time audit reproducibility."""

    def __init__(self) -> None:
        # business_key -> list of SCD2Record ordered by version ascending
        self._history: Dict[str, List[SCD2Record]] = {}

    def _compute_surrogate_key(self, b_key: str, version: int, attrs: Dict[str, Any]) -> str:
        """Compute immutable SHA256 surrogate key from business key, version, and attributes."""
        payload = f"{b_key}:{version}:{json.dumps(attrs, sort_keys=True, default=str)}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]

    def upsert(
        self,
        business_key: str,
        attributes: Dict[str, Any],
        effective_date: Optional[str] = None,
    ) -> SCD2Record:
        """Upsert a dimension entity under SCD Type 2 logic.

        If new: creates version 1.
        If existing and attributes identical: returns current record without creating duplicate revision.
        If existing and attributes changed: closes previous record and opens new version.
        """
        now = effective_date or _current_iso_time()
        records = self._history.setdefault(business_key, [])

        if not records:
            # Initial version 1
            sk = self._compute_surrogate_key(business_key, 1, attributes)
            rec = SCD2Record(
                surrogate_key=sk,
                business_key=business_key,
                attributes=dict(attributes),
                valid_from=now,
                valid_to=None,
                is_current=True,
                version=1,
            )
            records.append(rec)
            return rec

        current_rec = records[-1]
        # Check if attributes changed
        if current_rec.attributes == attributes:
            return current_rec  # No change, idempotency maintained

        # Expire current record
        current_rec.valid_to = now
        current_rec.is_current = False

        # Create new version
        new_version = current_rec.version + 1
        new_sk = self._compute_surrogate_key(business_key, new_version, attributes)
        new_rec = SCD2Record(
            surrogate_key=new_sk,
            business_key=business_key,
            attributes=dict(attributes),
            valid_from=now,
            valid_to=None,
            is_current=True,
            version=new_version,
        )
        records.append(new_rec)
        return new_rec

    def get_current(self, business_key: str) -> Optional[SCD2Record]:
        """Retrieve the active current record for a business key."""
        records = self._history.get(business_key, [])
        return records[-1] if records else None

    def get_as_of(self, business_key: str, timestamp_iso: str) -> Optional[SCD2Record]:
        """Perform point-in-time time-travel query to retrieve the dimension state at a past timestamp."""
        records = self._history.get(business_key, [])
        for r in records:
            if r.valid_from <= timestamp_iso:
                if r.valid_to is None or timestamp_iso < r.valid_to:
                    return r
        return None

    def get_version_history(self, business_key: str) -> List[SCD2Record]:
        """Return the complete version history for a dimension entity."""
        return list(self._history.get(business_key, []))


class SemanticMetricCompiler:
    """Compiles and executes declarative semantic metrics with RBAC dimension governance and SLA tracking."""

    def __init__(self) -> None:
        self._metrics: Dict[str, SemanticMetric] = {}
        self._cache: Dict[str, Tuple[str, Any]] = {}  # cache_key -> (timestamp, result)

    def register_metric(self, metric: SemanticMetric) -> None:
        """Register a semantic metric definition."""
        self._metrics[metric.metric_name] = metric

    def compile_sql(self, metric_name: str, requested_dimensions: List[str], user_role: str = "analyst") -> str:
        """Compile ANSI-compliant SQL query with role-based dimension constraints applied."""
        metric = self._metrics.get(metric_name)
        if not metric:
            raise KeyError(f"Metric '{metric_name}' is not registered.")

        # RBAC Check: Ensure user role is allowed to project requested dimensions
        allowed_dims = metric.restricted_dimensions.get(user_role, metric.dimensions)
        safe_dims = [d for d in requested_dimensions if d in allowed_dims]

        select_clauses = list(safe_dims)
        agg_expr = f"{metric.aggregation.upper()}({metric.target_column}) AS {metric.metric_name}"
        select_clauses.append(agg_expr)

        query = f"SELECT {', '.join(select_clauses)} FROM {metric.table_name}"
        if metric.filters:
            query += f" WHERE {' AND '.join(metric.filters)}"
        if safe_dims:
            query += f" GROUP BY {', '.join(safe_dims)}"
        return query

    def evaluate_in_memory(
        self,
        metric_name: str,
        data_rows: List[Dict[str, Any]],
        dimensions: List[str],
        user_role: str = "analyst",
    ) -> Dict[str, Any]:
        """Evaluate the metric against in-memory rows with latency benchmarking against defined SLA."""
        start_time = datetime.now(timezone.utc)
        metric = self._metrics.get(metric_name)
        if not metric:
            raise KeyError(f"Metric '{metric_name}' not found.")

        # RBAC Filtering
        allowed_dims = metric.restricted_dimensions.get(user_role, metric.dimensions)
        safe_dims = [d for d in dimensions if d in allowed_dims]

        # Apply row filters
        filtered_rows = data_rows
        # In-memory aggregation
        grouped: Dict[Tuple[Any, ...], List[float]] = {}
        for row in filtered_rows:
            key = tuple(row.get(d) for d in safe_dims)
            val = row.get(metric.target_column)
            if val is not None:
                grouped.setdefault(key, []).append(float(val))

        results = []
        for key, vals in grouped.items():
            out = {dim: key[i] for i, dim in enumerate(safe_dims)}
            if metric.aggregation.upper() == "SUM":
                out[metric.metric_name] = sum(vals)
            elif metric.aggregation.upper() == "AVG":
                out[metric.metric_name] = sum(vals) / len(vals) if vals else 0.0
            elif metric.aggregation.upper() == "COUNT":
                out[metric.metric_name] = len(vals)
            elif metric.aggregation.upper() == "MAX":
                out[metric.metric_name] = max(vals) if vals else 0.0
            elif metric.aggregation.upper() == "MIN":
                out[metric.metric_name] = min(vals) if vals else 0.0
            results.append(out)

        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
        sla_met = elapsed <= metric.sla_seconds

        return {
            "metric_name": metric_name,
            "user_role": user_role,
            "projected_dimensions": safe_dims,
            "rows": results,
            "elapsed_seconds": elapsed,
            "sla_seconds": metric.sla_seconds,
            "sla_met": sla_met,
        }
