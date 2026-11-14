"""Unit and integration tests for Airbyte Agent Infrastructure & MCP Connector Runtime."""

from __future__ import annotations

from llm_control_plane.airbyte_agent_infrastructure import (
    AirbyteAgentInfrastructure,
    AirbyteConnector,
    ConnectorType,
    SemanticModel,
    SyncStatus,
)


def test_default_catalog_registration() -> None:
    infra = AirbyteAgentInfrastructure()
    assert len(infra.connectors) >= 3
    assert "conn-pg-prod" in infra.connectors
    assert "conn-sf-crm" in infra.connectors
    assert "conn-stripe-billing" in infra.connectors
    assert "customer_lifetime_revenue" in infra.semantic_models


def test_mcp_discover_schema() -> None:
    infra = AirbyteAgentInfrastructure()
    res = infra.discover_schema("conn-pg-prod")
    assert res["status"] == 200
    assert res["connector_name"] == "Production PostgreSQL Replica"
    assert "users" in res["streams"]
    assert "orders" in res["streams"]


def test_mcp_discover_schema_not_found() -> None:
    infra = AirbyteAgentInfrastructure()
    res = infra.discover_schema("conn-nonexistent")
    assert res["status"] == 404
    assert "not found" in res["error"]


def test_rate_limiting_throttle_burst() -> None:
    infra = AirbyteAgentInfrastructure()
    # Create connector with small bucket of 2 tokens
    small_conn = AirbyteConnector(
        connector_id="conn-low-limit",
        name="Rate Limited API",
        connector_type=ConnectorType.HUBSPOT,
        streams=["deals"],
        rate_limit_per_minute=2,
        current_tokens=2.0,
    )
    infra.register_connector(small_conn)

    assert infra.check_rate_limit("conn-low-limit", cost=1.0) is True
    assert infra.check_rate_limit("conn-low-limit", cost=1.0) is True
    # 3rd attempt without refill time should be rejected
    assert infra.check_rate_limit("conn-low-limit", cost=1.0) is False


def test_trigger_sync_success() -> None:
    infra = AirbyteAgentInfrastructure()
    job = infra.trigger_sync("conn-pg-prod", "orders", requested_by="agent-reporter")
    assert job.status == SyncStatus.COMPLETED
    assert job.records_synced == 1500
    assert job.bytes_synced > 0
    assert job.duration_ms > 0
    assert len(infra.audit_log) >= 1
    assert infra.audit_log[-1]["action"] == "TRIGGER_SYNC"


def test_trigger_sync_invalid_stream() -> None:
    infra = AirbyteAgentInfrastructure()
    job = infra.trigger_sync("conn-pg-prod", "unmapped_table")
    assert job.status == SyncStatus.FAILED
    assert "not available" in job.error_message


def test_semantic_query_success() -> None:
    infra = AirbyteAgentInfrastructure()
    res = infra.execute_semantic_query(
        model_name="customer_lifetime_revenue",
        dimensions=["customer_id", "region"],
        metrics=["total_spent_usd", "order_count"],
        user_role="agent",
    )
    assert res["status"] == 200
    assert res["row_count"] == 2
    assert res["rows"][0]["customer_id"] == "cust_101"


def test_semantic_query_invalid_projection() -> None:
    infra = AirbyteAgentInfrastructure()
    res = infra.execute_semantic_query(
        model_name="customer_lifetime_revenue",
        dimensions=["unauthorized_ssn_column"],
        metrics=["total_spent_usd"],
        user_role="agent",
    )
    assert res["status"] == 400
    assert "unauthorized_ssn_column" in res["invalid_dimensions"]


def test_semantic_query_role_unauthorized() -> None:
    infra = AirbyteAgentInfrastructure()
    # Model restricted to admin only
    restricted = SemanticModel(
        model_name="executive_payroll_margins",
        source_table="payroll",
        dimensions=["department"],
        metrics=["margin"],
        allowed_roles={"admin"},
    )
    infra.register_semantic_model(restricted)

    res = infra.execute_semantic_query(
        model_name="executive_payroll_margins",
        dimensions=["department"],
        metrics=["margin"],
        user_role="agent",
    )
    assert res["status"] == 403
    assert "denied" in res["error"].lower()


def test_sandbox_create_and_update_record() -> None:
    infra = AirbyteAgentInfrastructure()
    org_id = "org_test_99"

    # 1. Create record
    res_create = infra.sandbox_mutate_record(
        org_id=org_id,
        object_type="hubspot_deal",
        record_id="deal_1001",
        properties={"deal_name": "Series A Renewal", "amount": 75000.0},
    )
    assert res_create["status"] == 200
    assert res_create["action"] == "CREATE"
    assert res_create["properties"]["amount"] == 75000.0

    # 2. Update record
    res_update = infra.sandbox_mutate_record(
        org_id=org_id,
        object_type="hubspot_deal",
        record_id="deal_1001",
        properties={"amount": 90000.0, "stage": "closed_won"},
    )
    assert res_update["status"] == 200
    assert res_update["action"] == "UPDATE"
    assert res_update["properties"]["amount"] == 90000.0
    assert res_update["properties"]["stage"] == "closed_won"


def test_sandbox_fault_injection() -> None:
    infra = AirbyteAgentInfrastructure()
    infra.set_fault_injection("sandbox/salesforce_lead", 503)

    res = infra.sandbox_mutate_record(
        org_id="org_alpha",
        object_type="salesforce_lead",
        record_id="lead_55",
        properties={"email": "lead@corp.com"},
    )
    assert res["status"] == 503
    assert "outage" in res["error"].lower()

    infra.clear_fault_injection()
    res_cleared = infra.sandbox_mutate_record(
        org_id="org_alpha",
        object_type="salesforce_lead",
        record_id="lead_55",
        properties={"email": "lead@corp.com"},
    )
    assert res_cleared["status"] == 200


def test_sandbox_state_reset() -> None:
    infra = AirbyteAgentInfrastructure()
    org_id = "org_ephemeral"
    infra.sandbox_mutate_record(org_id, "contact", "c1", {"name": "Alice"})
    assert "c1" in infra.sandbox_storage[org_id]

    infra.sandbox_reset_state(org_id)
    assert len(infra.sandbox_storage[org_id]) == 0


def test_evaluate_agent_run_and_audit_hash() -> None:
    infra = AirbyteAgentInfrastructure()
    ops = [
        {"type": "discover", "connector_id": "conn-pg-prod"},
        {"type": "sync", "connector_id": "conn-pg-prod", "stream_name": "orders"},
        {
            "type": "semantic_query",
            "model_name": "customer_lifetime_revenue",
            "dimensions": ["customer_id"],
            "metrics": ["total_spent_usd"],
        },
        {
            "type": "sandbox_mutate",
            "org_id": "org_eval_1",
            "object_type": "stripe_invoice",
            "record_id": "inv_882",
            "properties": {"status": "paid"},
        },
    ]

    report = infra.evaluate_agent_run(agent_id="data-platform-agent", operations=ops)
    assert report.tools_invoked == 4
    assert report.total_records == 1500
    assert report.passed_safety_gate is True
    assert report.p50_latency_ms > 0
    assert report.p95_latency_ms >= report.p50_latency_ms
    assert len(report.audit_hash) == 64  # SHA-256 hash length
