"""Unit tests for SolarWinds Data Platform Orchestrator."""

from datetime import datetime, timedelta, timezone

from llm_control_plane.solarwinds_data_platform_orchestrator import (
    BigQueryTableMetadata,
    DataplexGovernanceEngine,
    DataplexIAMRole,
    FivetranIngestionReliabilityMonitor,
    FivetranSyncState,
    MaskingRule,
    SolarWindsDataPlatformOrchestrator,
    WorkloadTier,
)


def test_dataplex_governance_and_column_masking():
    """Verifies Dataplex policy tag registration, role bindings, and column masking."""
    gov = DataplexGovernanceEngine()

    tag_email = gov.register_policy_tag(
        tag_id="tag_email",
        display_name="Email Address",
        masking_rule=MaskingRule.PARTIAL_EMAIL,
    )
    tag_phone = gov.register_policy_tag(
        tag_id="tag_phone",
        display_name="Phone Number",
        masking_rule=MaskingRule.PARTIAL_PHONE,
    )
    tag_secret = gov.register_policy_tag(
        tag_id="tag_secret",
        display_name="API Secret",
        masking_rule=MaskingRule.NULLIFY,
    )

    table_meta = BigQueryTableMetadata(
        project_id="swi-prod",
        dataset_id="core",
        table_id="users",
        schema={"user_id": "STRING", "email": "STRING", "phone": "STRING", "api_key": "STRING"},
        partition_field="created_at",
        policy_tags={"email": tag_email.tag_id, "phone": tag_phone.tag_id, "api_key": tag_secret.tag_id},
    )

    gov.bind_user_role("analyst@solarwinds.com", tag_email.tag_id, DataplexIAMRole.MASKED_READER)
    gov.bind_user_role("analyst@solarwinds.com", tag_phone.tag_id, DataplexIAMRole.MASKED_READER)
    gov.bind_user_role("secops@solarwinds.com", tag_email.tag_id, DataplexIAMRole.FINE_GRAINED_READER)

    raw_record = {
        "user_id": "usr_9981",
        "email": "soumya@solarwinds.com",
        "phone": "+353871234567",
        "api_key": "sk_live_verysecret123",
    }

    # Analyst (Masked Reader for email/phone, No Access for api_key)
    analyst_view = gov.apply_governance_to_record(raw_record, table_meta, "analyst@solarwinds.com")
    assert analyst_view["user_id"] == "usr_9981"
    assert analyst_view["email"] == "s***@solarwinds.com"
    assert analyst_view["phone"] == "***-***-4567"
    assert analyst_view["api_key"] is None  # No access redacts column

    # SecOps (Fine-Grained Reader for email, No Access for phone/api_key)
    secops_view = gov.apply_governance_to_record(raw_record, table_meta, "secops@solarwinds.com")
    assert secops_view["email"] == "soumya@solarwinds.com"  # Raw plaintext
    assert secops_view["phone"] is None


def test_bigquery_table_architecture_and_cost_estimation():
    """Tests BigQuery partitioning/clustering checks and query cost/slot calculations."""
    orchestrator = SolarWindsDataPlatformOrchestrator()
    optimizer = orchestrator.optimizer

    # Valid table architecture
    valid_meta = BigQueryTableMetadata(
        project_id="swi-prod",
        dataset_id="metrics",
        table_id="device_telemetry",
        schema={"timestamp": "TIMESTAMP", "device_id": "STRING"},
        partition_field="timestamp",
        clustering_fields=["device_id"],
    )
    assert optimizer.validate_table_architecture(valid_meta) == []

    # Invalid table architecture (missing partition, >4 clustering fields)
    invalid_meta = BigQueryTableMetadata(
        project_id="swi-prod",
        dataset_id="metrics",
        table_id="bad_table",
        schema={"id": "STRING"},
        partition_field=None,
        clustering_fields=["f1", "f2", "f3", "f4", "f5"],
    )
    issues = optimizer.validate_table_architecture(invalid_meta)
    assert len(issues) == 2
    assert "lacks partition_field" in issues[0]
    assert "max 4" in issues[1]

    # Query cost calculation: 1 TB query ($6.25 USD)
    one_tb_bytes = 1024**4
    estimate = optimizer.estimate_query_cost(
        bytes_scanned=one_tb_bytes,
        is_partition_pruned=True,
        tier=WorkloadTier.CRITICAL_ELT,
    )
    assert estimate.estimated_cost_usd == 6.25
    assert not estimate.requires_approval
    assert estimate.allocated_slots == 1000

    # Query without partition pruning triggers approval requirement
    unpruned_estimate = optimizer.estimate_query_cost(
        bytes_scanned=1024**3,  # 1 GB
        is_partition_pruned=False,
        tier=WorkloadTier.ADHOC_ANALYTICS,
    )
    assert unpruned_estimate.requires_approval
    assert unpruned_estimate.allocated_slots == 200


def test_fivetran_schema_drift_and_quarantine():
    """Validates Fivetran connector monitoring, schema drift detection, and quarantine triggers."""
    monitor = FivetranIngestionReliabilityMonitor(max_allowed_lag_minutes=20)
    monitor.register_connector(
        connector_id="conn_salesforce_crm",
        service_type="salesforce",
        target_table="raw_salesforce_accounts",
    )

    existing_schema = {"account_id": "STRING", "mrr": "FLOAT64", "created_at": "TIMESTAMP"}

    # Non-breaking drift (new column added)
    non_breaking_source = {
        "account_id": "STRING",
        "mrr": "FLOAT64",
        "created_at": "TIMESTAMP",
        "industry": "STRING",
    }
    report1 = monitor.evaluate_schema_drift(
        "conn_salesforce_crm",
        "raw_salesforce_accounts",
        non_breaking_source,
        existing_schema,
    )
    assert report1.added_columns == ["industry"]
    assert not report1.is_breaking_change
    assert not report1.quarantine_triggered

    # Breaking drift (type alteration: FLOAT64 -> STRING)
    breaking_source = {
        "account_id": "STRING",
        "mrr": "STRING",  # Breaking change
        "created_at": "TIMESTAMP",
    }
    report2 = monitor.evaluate_schema_drift(
        "conn_salesforce_crm",
        "raw_salesforce_accounts",
        breaking_source,
        existing_schema,
    )
    assert report2.is_breaking_change
    assert report2.quarantine_triggered
    assert monitor.connector_registry["conn_salesforce_crm"]["status"] == FivetranSyncState.QUARANTINED


def test_fivetran_connector_sla_monitoring():
    """Validates Fivetran connector sync latency SLA enforcement."""
    monitor = FivetranIngestionReliabilityMonitor(max_allowed_lag_minutes=15)
    monitor.register_connector("conn_stripe_billing", "stripe", "raw_invoices")

    now = datetime.now(timezone.utc)
    # 10 minutes elapsed (compliant with 15 min SLA)
    assert monitor.check_connector_sla("conn_stripe_billing", current_time=now + timedelta(minutes=10))

    # 25 minutes elapsed (breaches 15 min SLA)
    assert not monitor.check_connector_sla("conn_stripe_billing", current_time=now + timedelta(minutes=25))


def test_solarwinds_orchestrator_end_to_end():
    """Verifies SolarWindsDataPlatformOrchestrator setup and governance integration."""
    orchestrator = SolarWindsDataPlatformOrchestrator()
    table_meta = orchestrator.setup_telemetry_governance(
        project_id="solarwinds-cork-prod",
        dataset_id="observability_telemetry",
    )

    assert table_meta.fully_qualified_id == "solarwinds-cork-prod.observability_telemetry.telemetry_metrics_v1"
    assert table_meta.partition_field == "timestamp"
    assert "customer_id" in table_meta.policy_tags
    assert "user_email" in table_meta.policy_tags

    # Bind Masked Reader role to customer_id tag
    tag_id = table_meta.policy_tags["customer_id"]
    orchestrator.governance.bind_user_role("cork_analyst@solarwinds.com", tag_id, DataplexIAMRole.MASKED_READER)

    record = {
        "event_id": "evt_101",
        "timestamp": "2026-11-14T10:00:00Z",
        "customer_id": "cust_ireland_42",
        "user_email": "engineer@solarwinds.ie",
        "metric_name": "cpu_utilization_pct",
        "metric_value": 78.4,
    }

    sanitized = orchestrator.governance.apply_governance_to_record(
        record,
        table_meta,
        "cork_analyst@solarwinds.com",
    )
    # customer_id should be hashed with SHA-256
    import hashlib

    expected_hash = hashlib.sha256("cust_ireland_42".encode("utf-8")).hexdigest()
    assert sanitized["customer_id"] == expected_hash
    # user_email had no role bound, so it is redacted
    assert sanitized["user_email"] is None
    assert sanitized["metric_value"] == 78.4
