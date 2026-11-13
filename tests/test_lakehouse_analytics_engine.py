"""Tests for Lakehouse Analytics Engine (Medallion, SCD Type 2, and Semantic Models)."""

import pytest

from llm_control_plane.lakehouse_analytics_engine import (
    MedallionLakehouseManager,
    SCDType2DimensionTracker,
    SemanticMetric,
    SemanticMetricCompiler,
)


def test_medallion_bronze_ingestion_and_deduplication() -> None:
    mgr = MedallionLakehouseManager()
    raw = [
        {"customer_id": "C01", "region": "Auckland", "spend": 1200},
        {"customer_id": "C01", "region": "Auckland", "spend": 1200},  # duplicate payload
        {"customer_id": "C02", "region": "Wellington", "spend": 850},
    ]
    recs = mgr.ingest_bronze("customer_spend", raw)
    assert len(recs) == 3
    # Distinct records in internal storage because identical payloads share hash
    stats = mgr.get_tier_stats()
    assert stats["bronze_count"] == 2  # Deduplicated in bronze store by hash


def test_medallion_silver_schema_contract_success() -> None:
    mgr = MedallionLakehouseManager()
    mgr.register_schema_contract("sales", {"sale_id": str, "amount": (int, float), "approved": bool})
    mgr.ingest_bronze("sales", [{"sale_id": "S100", "amount": 450.50, "approved": True}])

    promoted = mgr.promote_to_silver("sales")
    assert len(promoted) == 1
    assert promoted[0].tier == "silver"
    assert promoted[0].payload["sale_id"] == "S100"


def test_medallion_silver_schema_contract_type_error() -> None:
    mgr = MedallionLakehouseManager()
    mgr.register_schema_contract("sales", {"sale_id": str, "amount": (int, float)})
    mgr.ingest_bronze("sales", [{"sale_id": "S101", "amount": "INVALID_STRING"}])

    with pytest.raises(TypeError, match="Field 'amount' expected"):
        mgr.promote_to_silver("sales")


def test_medallion_silver_schema_contract_missing_field() -> None:
    mgr = MedallionLakehouseManager()
    mgr.register_schema_contract("sales", {"sale_id": str, "region": str})
    mgr.ingest_bronze("sales", [{"sale_id": "S102"}])

    with pytest.raises(ValueError, match="Missing required contract field: 'region'"):
        mgr.promote_to_silver("sales")


def test_medallion_gold_mart_aggregation_sum_and_avg() -> None:
    mgr = MedallionLakehouseManager()
    sales = [
        {"store": "Auckland Central", "revenue": 1000.0},
        {"store": "Auckland Central", "revenue": 2000.0},
        {"store": "Christchurch", "revenue": 1500.0},
    ]
    mgr.ingest_bronze("store_sales", sales)
    mgr.promote_to_silver("store_sales")

    gold = mgr.build_gold_mart(
        "mart_store_summary",
        "store_sales",
        group_by=["store"],
        aggregations={
            "total_revenue": ("SUM", "revenue"),
            "avg_revenue": ("AVG", "revenue"),
            "transaction_count": ("COUNT", "revenue"),
        },
    )

    assert len(gold) == 2
    akl = next(g for g in gold if g["store"] == "Auckland Central")
    assert akl["total_revenue"] == 3000.0
    assert akl["avg_revenue"] == 1500.0
    assert akl["transaction_count"] == 2


def test_medallion_tier_stats() -> None:
    mgr = MedallionLakehouseManager()
    mgr.ingest_bronze("events", [{"ev_id": "1"}])
    mgr.promote_to_silver("events")
    mgr.build_gold_mart("mart_ev", "events", group_by=["ev_id"], aggregations={"cnt": ("COUNT", "ev_id")})

    stats = mgr.get_tier_stats()
    assert stats["bronze_count"] == 1
    assert stats["silver_count"] == 1
    assert stats["gold_mart_count"] == 1


def test_scd2_initial_version_creation() -> None:
    tracker = SCDType2DimensionTracker()
    rec = tracker.upsert("CUST_42", {"tier": "standard", "credit_limit": 5000}, effective_date="2026-01-01T00:00:00Z")

    assert rec.business_key == "CUST_42"
    assert rec.version == 1
    assert rec.is_current is True
    assert rec.valid_from == "2026-01-01T00:00:00Z"
    assert rec.valid_to is None
    assert rec.attributes["tier"] == "standard"


def test_scd2_idempotency_when_attributes_unchanged() -> None:
    tracker = SCDType2DimensionTracker()
    tracker.upsert("CUST_42", {"tier": "standard"}, effective_date="2026-01-01T00:00:00Z")
    rec2 = tracker.upsert("CUST_42", {"tier": "standard"}, effective_date="2026-01-02T00:00:00Z")

    assert rec2.version == 1
    assert len(tracker.get_version_history("CUST_42")) == 1


def test_scd2_attribute_update_creates_new_version_and_expires_old() -> None:
    tracker = SCDType2DimensionTracker()
    v1 = tracker.upsert("CUST_42", {"tier": "standard", "credit": 5000}, effective_date="2026-01-01T00:00:00Z")
    v2 = tracker.upsert("CUST_42", {"tier": "premium", "credit": 15000}, effective_date="2026-02-01T00:00:00Z")

    assert v1.is_current is False
    assert v1.valid_to == "2026-02-01T00:00:00Z"

    assert v2.version == 2
    assert v2.is_current is True
    assert v2.valid_from == "2026-02-01T00:00:00Z"
    assert v2.valid_to is None
    assert v2.attributes["tier"] == "premium"


def test_scd2_multiple_updates_history_integrity() -> None:
    tracker = SCDType2DimensionTracker()
    tracker.upsert("ACC_99", {"status": "pending"}, effective_date="2026-01-01T00:00:00Z")
    tracker.upsert("ACC_99", {"status": "active"}, effective_date="2026-01-05T00:00:00Z")
    tracker.upsert("ACC_99", {"status": "suspended"}, effective_date="2026-01-10T00:00:00Z")

    history = tracker.get_version_history("ACC_99")
    assert len(history) == 3
    assert [h.version for h in history] == [1, 2, 3]
    assert [h.is_current for h in history] == [False, False, True]


def test_scd2_point_in_time_time_travel_query() -> None:
    tracker = SCDType2DimensionTracker()
    tracker.upsert("EMP_07", {"dept": "Data Analytics"}, effective_date="2026-01-01T00:00:00Z")
    tracker.upsert("EMP_07", {"dept": "AI Delivery"}, effective_date="2026-06-01T00:00:00Z")

    as_of_march = tracker.get_as_of("EMP_07", "2026-03-15T00:00:00Z")
    assert as_of_march is not None
    assert as_of_march.attributes["dept"] == "Data Analytics"

    as_of_july = tracker.get_as_of("EMP_07", "2026-07-20T00:00:00Z")
    assert as_of_july is not None
    assert as_of_july.attributes["dept"] == "AI Delivery"


def test_semantic_metric_sql_compilation_with_group_by() -> None:
    compiler = SemanticMetricCompiler()
    metric = SemanticMetric(
        metric_name="monthly_active_subscribers",
        table_name="gold.analytics_subscriptions",
        aggregation="COUNT",
        target_column="subscriber_id",
        dimensions=["region", "plan_tier"],
        filters=["is_active = true", "churn_date IS NULL"],
    )
    compiler.register_metric(metric)

    sql = compiler.compile_sql("monthly_active_subscribers", ["region", "plan_tier"])
    assert "SELECT region, plan_tier, COUNT(subscriber_id) AS monthly_active_subscribers" in sql
    assert "FROM gold.analytics_subscriptions" in sql
    assert "WHERE is_active = true AND churn_date IS NULL" in sql
    assert "GROUP BY region, plan_tier" in sql


def test_semantic_metric_rbac_dimension_masking() -> None:
    compiler = SemanticMetricCompiler()
    metric = SemanticMetric(
        metric_name="avg_deal_size",
        table_name="gold.crm_deals",
        aggregation="AVG",
        target_column="deal_value",
        dimensions=["industry", "client_confidential_id"],
        restricted_dimensions={"external_partner": ["industry"]},  # mask client_confidential_id
    )
    compiler.register_metric(metric)

    # Analyst role gets all
    sql_analyst = compiler.compile_sql("avg_deal_size", ["industry", "client_confidential_id"], user_role="analyst")
    assert "client_confidential_id" in sql_analyst

    # External partner has restricted access
    sql_partner = compiler.compile_sql(
        "avg_deal_size", ["industry", "client_confidential_id"], user_role="external_partner"
    )
    assert "client_confidential_id" not in sql_partner
    assert "industry" in sql_partner


def test_semantic_metric_in_memory_evaluation_and_sla() -> None:
    compiler = SemanticMetricCompiler()
    metric = SemanticMetric(
        metric_name="total_orders",
        table_name="gold.sales",
        aggregation="SUM",
        target_column="amount",
        dimensions=["country"],
        sla_seconds=1.0,
    )
    compiler.register_metric(metric)

    data = [
        {"country": "NZ", "amount": 500},
        {"country": "NZ", "amount": 750},
        {"country": "AU", "amount": 300},
    ]

    res = compiler.evaluate_in_memory("total_orders", data, ["country"])
    assert res["metric_name"] == "total_orders"
    assert res["sla_met"] is True
    assert len(res["rows"]) == 2

    nz_row = next(r for r in res["rows"] if r["country"] == "NZ")
    assert nz_row["total_orders"] == 1250.0
