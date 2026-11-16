"""Comprehensive unit tests for Arta Finance B2B Wealth Platform Orchestrator."""

import pytest

from llm_control_plane.arta_wealth_platform_orchestrator import (
    AgenticAdvisorGuardrailEngine,
    AssetClass,
    CanonicalInvestmentOrder,
    PartnerGatewayMessage,
    PartnerGatewayTransformer,
    PartnerProtocol,
    PartnerTelemetryMonitor,
    RiskProfile,
    TemporalWealthSagaCoordinator,
)


@pytest.fixture
def transformer():
    return PartnerGatewayTransformer(partner_secret="arta_test_partner_secret_123")


def test_hmac_signature_verification(transformer):
    payload_str = '{"partner_id": "DBS_BANK", "transaction_id": "TX_9901"}'
    sig = transformer.compute_signature(payload_str)
    assert transformer.verify_signature(payload_str, sig) is True
    assert transformer.verify_signature(payload_str, "invalid_sig") is False


def test_gateway_normalize_rest_valid(transformer):
    raw_payload = {
        "partner_id": "DBS_PRIVATE_BANK",
        "transaction_id": "TX_1001",
        "client_id": "SG_HNW_8821",
        "asset_id": "ARTA_GLOBAL_TECH_PE",
        "asset_class": "PRIVATE_EQUITY",
        "amount": 250000.0,
        "currency": "USD",
        "is_accredited_investor": True,
        "risk_score": 75.0,
    }
    msg = PartnerGatewayMessage(
        raw_payload=raw_payload,
        protocol=PartnerProtocol.REST_JSON,
        signature="test_sig",
    )
    order = transformer.normalize(msg)
    assert order.partner_id == "DBS_PRIVATE_BANK"
    assert order.amount == 250000.0
    assert order.currency == "USD"
    assert order.asset_class == AssetClass.PRIVATE_EQUITY
    assert order.is_accredited_investor is True
    assert order.idempotency_key is not None


def test_gateway_non_accredited_investor_private_market_rejected(transformer):
    raw_payload = {
        "partner_id": "STANDARD_CHARTERED",
        "transaction_id": "TX_1002",
        "client_id": "SG_RETAIL_001",
        "asset_id": "ARTA_VENTURE_OPPORTUNITY",
        "asset_class": "VENTURE_CAPITAL",
        "amount": 50000.0,
        "currency": "USD",
        "is_accredited_investor": False,
    }
    msg = PartnerGatewayMessage(
        raw_payload=raw_payload,
        protocol=PartnerProtocol.REST_JSON,
        signature="test_sig",
    )
    with pytest.raises(PermissionError, match="MAS Notice SFA 04-N12"):
        transformer.normalize(msg)


def test_gateway_idempotency_deduplication(transformer):
    raw_payload = {
        "partner_id": "OCBC_BANK",
        "transaction_id": "TX_1003",
        "client_id": "SG_HNW_002",
        "asset_id": "ARTA_PUBLIC_INDEX",
        "asset_class": "PUBLIC_EQUITY",
        "amount": 10000.0,
        "currency": "SGD",
        "is_accredited_investor": True,
    }
    msg = PartnerGatewayMessage(
        raw_payload=raw_payload,
        protocol=PartnerProtocol.REST_JSON,
        signature="test_sig",
    )
    order = transformer.normalize(msg)
    assert order.amount == 10000.0

    # Second call with same parameters must be rejected
    with pytest.raises(ValueError, match="Duplicate transaction detected"):
        transformer.normalize(msg)


def test_temporal_saga_coordinator_success():
    order = CanonicalInvestmentOrder(
        order_id="ARTA-ORD-001",
        partner_id="DBS_BANK",
        client_id="CLIENT_123",
        asset_id="ARTA_CREDIT_TR1",
        asset_class=AssetClass.PRIVATE_CREDIT,
        amount=100000.0,
        currency="USD",
        idempotency_key="IDEM_001",
        risk_score=60.0,
        is_accredited_investor=True,
    )
    audit_trail = []

    def step1_exec(o):
        audit_trail.append("step1_done")
        return True

    def step1_comp(o):
        audit_trail.append("step1_reverted")
        return True

    def step2_exec(o):
        audit_trail.append("step2_done")
        return True

    def step2_comp(o):
        audit_trail.append("step2_reverted")
        return True

    coordinator = TemporalWealthSagaCoordinator(order)
    coordinator.add_step("PartnerHold", step1_exec, step1_comp)
    coordinator.add_step("CustodyAllocation", step2_exec, step2_comp)

    success, logs = coordinator.execute()
    assert success is True
    assert coordinator.is_success is True
    assert coordinator.compensated is False
    assert audit_trail == ["step1_done", "step2_done"]
    assert len(coordinator.completed_steps) == 2


def test_temporal_saga_coordinator_compensation_on_failure():
    order = CanonicalInvestmentOrder(
        order_id="ARTA-ORD-002",
        partner_id="UOB_BANK",
        client_id="CLIENT_456",
        asset_id="ARTA_VENTURE_TR2",
        asset_class=AssetClass.VENTURE_CAPITAL,
        amount=500000.0,
        currency="USD",
        idempotency_key="IDEM_002",
        risk_score=80.0,
        is_accredited_investor=True,
    )
    compensations = []

    def step1_exec(o):
        return True

    def step1_comp(o):
        compensations.append("PartnerHold_Compensated")
        return True

    def step2_exec(o):
        # Step 2 fails (e.g. custodian API timeout)
        return False

    def step2_comp(o):
        compensations.append("Custody_Compensated")
        return True

    coordinator = TemporalWealthSagaCoordinator(order)
    coordinator.add_step("PartnerHold", step1_exec, step1_comp)
    coordinator.add_step("CustodyPlacement", step2_exec, step2_comp)

    success, logs = coordinator.execute()
    assert success is False
    assert coordinator.is_success is False
    assert coordinator.compensated is True
    assert "PartnerHold_Compensated" in compensations


def test_agentic_advisor_guardrail_tool_whitelist():
    guard = AgenticAdvisorGuardrailEngine()
    ok, _ = guard.validate_tool_call("portfolio_rebalancer", {"target_risk": "GROWTH"})
    assert ok is True

    bad_ok, bad_msg = guard.validate_tool_call("arbitrary_fund_drain", {})
    assert bad_ok is False
    assert "Unauthorized tool" in bad_msg


def test_agentic_advisor_guardrail_portfolio_rebalance_validation():
    guard = AgenticAdvisorGuardrailEngine()

    # Conservative profile: cannot exceed 30% public equity or 0% PE
    invalid_allocation = {
        AssetClass.CASH_EQUIVALENT: 0.40,
        AssetClass.PUBLIC_EQUITY: 0.50,  # Max is 0.30
        AssetClass.PRIVATE_CREDIT: 0.10,
    }
    ok, errors = guard.validate_portfolio_rebalance(RiskProfile.CONSERVATIVE, invalid_allocation)
    assert ok is False
    assert any("exceeds maximum allowed" in e for e in errors)

    valid_allocation = {
        AssetClass.CASH_EQUIVALENT: 0.60,
        AssetClass.PUBLIC_EQUITY: 0.30,
        AssetClass.PRIVATE_CREDIT: 0.10,
    }
    valid_ok, valid_errs = guard.validate_portfolio_rebalance(RiskProfile.CONSERVATIVE, valid_allocation)
    assert valid_ok is True
    assert len(valid_errs) == 0


def test_partner_telemetry_monitor():
    monitor = PartnerTelemetryMonitor(partner_id="DBS_BANK", p99_sla_ms=100.0)
    for i in range(100):
        # 95 requests fast, 5 requests slow
        lat = 30.0 if i < 95 else 120.0
        monitor.record_request(lat, is_error=(i == 99))

    summary = monitor.get_summary()
    assert summary["total_requests"] == 100
    assert summary["errors"] == 1
    assert summary["error_rate"] == 0.01
    assert summary["p50_latency_ms"] == 30.0
    assert summary["p99_latency_ms"] == 120.0
    assert summary["sla_breached"] is True
