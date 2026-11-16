"""Unit tests for Oscilar AI Risk Decisioning Platform Orchestrator."""

from typing import List

import pytest

from llm_control_plane.oscilar_risk_decisioning_orchestrator import (
    ClickHouseOLAPExporter,
    DecisionOutcome,
    RiskDecisionEngine,
    RiskRuleEvaluator,
    RiskSeverity,
    SagaStepStatus,
    SlidingWindowVelocityEngine,
    TemporalRiskSagaCoordinator,
    TransactionEvent,
)


@pytest.fixture
def base_event() -> TransactionEvent:
    return TransactionEvent(
        event_id="tx_1001",
        account_id="acc_user_42",
        amount=150.00,
        currency="USD",
        timestamp=1700000000.0,
        ip_address="198.51.100.10",
        device_fingerprint="fp_chrome_mac_01",
        merchant_category="5411",  # Grocery stores
    )


def test_transaction_event_hash(base_event: TransactionEvent) -> None:
    event_hash = base_event.compute_event_hash()
    assert isinstance(event_hash, str)
    assert len(event_hash) == 64
    assert event_hash == base_event.compute_event_hash()


def test_sliding_window_velocity_aggregation() -> None:
    engine = SlidingWindowVelocityEngine(default_window_seconds=60.0)
    t_base = 1700000000.0

    for i in range(4):
        event = TransactionEvent(
            event_id=f"tx_{i}",
            account_id="acc_99",
            amount=50.0 * (i + 1),
            currency="USD",
            timestamp=t_base + (i * 10),
            ip_address=f"10.0.0.{i}",
            device_fingerprint=f"dev_{i % 2}",
            merchant_category="5812",
        )
        engine.record_event(event)

    velocity = engine.get_velocity("acc_99", window_seconds=60.0, current_time=t_base + 35)
    assert velocity.event_count == 4
    assert velocity.total_amount == 50.0 + 100.0 + 150.0 + 200.0  # 500.0
    assert velocity.unique_ips == 4
    assert velocity.unique_devices == 2
    assert velocity.max_single_amount == 200.0


def test_velocity_pruning_outdated_events() -> None:
    engine = SlidingWindowVelocityEngine(default_window_seconds=30.0)
    t_base = 1700000000.0

    # Old event
    old_event = TransactionEvent(
        event_id="tx_old",
        account_id="acc_prune",
        amount=100.0,
        currency="USD",
        timestamp=t_base - 100.0,
        ip_address="1.1.1.1",
        device_fingerprint="dev_old",
        merchant_category="5411",
    )
    engine.record_event(old_event)

    # Fresh event
    fresh_event = TransactionEvent(
        event_id="tx_fresh",
        account_id="acc_prune",
        amount=250.0,
        currency="USD",
        timestamp=t_base,
        ip_address="1.1.1.2",
        device_fingerprint="dev_fresh",
        merchant_category="5411",
    )
    engine.record_event(fresh_event)

    vel = engine.get_velocity("acc_prune", window_seconds=30.0, current_time=t_base)
    assert vel.event_count == 1
    assert vel.total_amount == 250.0


def test_rule_evaluator_triggers() -> None:
    evaluator = RiskRuleEvaluator(
        max_events_per_minute=2,
        max_spend_per_5min=1000.0,
        max_devices_per_window=1,
        high_risk_mccs={"7995"},
    )

    t_now = 1700000000.0
    from llm_control_plane.oscilar_risk_decisioning_orchestrator import VelocityMetrics

    vel_high = VelocityMetrics(
        account_id="acc_flag",
        window_seconds=60.0,
        event_count=5,
        total_amount=5000.0,
        unique_ips=3,
        unique_devices=3,
        max_single_amount=2000.0,
        calculated_at=t_now,
    )

    event_mcc = TransactionEvent(
        event_id="tx_gambling",
        account_id="acc_flag",
        amount=2000.0,
        currency="USD",
        timestamp=t_now,
        ip_address="10.10.10.10",
        device_fingerprint="dev_suspect",
        merchant_category="7995",
    )

    results = evaluator.evaluate_rules(event_mcc, vel_high)
    failed_rules = [r.rule_name for r in results if not r.passed]

    assert "VELOCITY_SPIKE" in failed_rules
    assert "RAPID_SPEND_INRUSH" in failed_rules
    assert "DEVICE_HOPPING_DETECTED" in failed_rules
    assert "HIGH_RISK_MCC" in failed_rules


def test_decision_engine_approve_low_risk(base_event: TransactionEvent) -> None:
    decision_engine = RiskDecisionEngine()
    record = decision_engine.evaluate_transaction(base_event)

    assert record.outcome == DecisionOutcome.APPROVE
    assert record.severity == RiskSeverity.LOW
    assert record.composite_risk_score == 0.0
    assert record.triggered_rules == []
    assert record.latency_ms >= 0.0


def test_decision_engine_decline_high_risk() -> None:
    decision_engine = RiskDecisionEngine(
        rule_evaluator=RiskRuleEvaluator(max_events_per_minute=1, max_spend_per_5min=500.0, high_risk_mccs={"6051"}),
        decline_threshold=70.0,
    )
    t_now = 1700000000.0

    # Flood with prior events
    for i in range(5):
        decision_engine.velocity_engine.record_event(
            TransactionEvent(
                event_id=f"tx_prior_{i}",
                account_id="acc_fraud",
                amount=300.0,
                currency="USD",
                timestamp=t_now + i,
                ip_address=f"10.0.0.{i}",
                device_fingerprint=f"dev_{i}",
                merchant_category="6051",
            )
        )

    eval_event = TransactionEvent(
        event_id="tx_eval",
        account_id="acc_fraud",
        amount=1000.0,
        currency="USD",
        timestamp=t_now + 10,
        ip_address="10.0.0.99",
        device_fingerprint="dev_99",
        merchant_category="6051",
    )

    record = decision_engine.evaluate_transaction(eval_event)
    assert record.outcome == DecisionOutcome.DECLINE
    assert record.severity == RiskSeverity.CRITICAL
    assert record.composite_risk_score >= 70.0


def test_temporal_risk_saga_successful_execution() -> None:
    saga = TemporalRiskSagaCoordinator(saga_id="saga_tx_001")
    audit_trail: List[str] = []

    def step1_velocity_check() -> bool:
        audit_trail.append("velocity_ok")
        return True

    def comp1_velocity_check() -> bool:
        audit_trail.append("undo_velocity")
        return True

    def step2_external_screening() -> bool:
        audit_trail.append("screening_ok")
        return True

    def comp2_external_screening() -> bool:
        audit_trail.append("undo_screening")
        return True

    saga.add_step("s1", "VelocityCheck", step1_velocity_check, comp1_velocity_check)
    saga.add_step("s2", "ExternalSanctionScreening", step2_external_screening, comp2_external_screening)

    success, logs = saga.execute()
    assert success is True
    assert saga.state == SagaStepStatus.COMPLETED
    assert audit_trail == ["velocity_ok", "screening_ok"]


def test_temporal_risk_saga_compensation_on_failure() -> None:
    saga = TemporalRiskSagaCoordinator(saga_id="saga_tx_fail_002")
    audit_trail: List[str] = []

    def step1_hold_reserve() -> bool:
        audit_trail.append("hold_reserved")
        return True

    def comp1_hold_release() -> bool:
        audit_trail.append("hold_released")
        return True

    def step2_failing_third_party() -> bool:
        audit_trail.append("call_failed_provider")
        return False

    def comp2_third_party() -> bool:
        audit_trail.append("comp_third_party")
        return True

    saga.add_step("s1", "ReserveRiskHold", step1_hold_reserve, comp1_hold_release)
    saga.add_step("s2", "ExternalBureauVerification", step2_failing_third_party, comp2_third_party)

    success, logs = saga.execute()
    assert success is False
    assert saga.state == SagaStepStatus.COMPENSATED
    assert "hold_reserved" in audit_trail
    assert "call_failed_provider" in audit_trail
    assert "hold_released" in audit_trail


def test_clickhouse_olap_export_row(base_event: TransactionEvent) -> None:
    engine = RiskDecisionEngine()
    record = engine.evaluate_transaction(base_event)
    row = ClickHouseOLAPExporter.to_clickhouse_row(record)

    assert row["event_id"] == "tx_1001"
    assert row["account_id"] == "acc_user_42"
    assert row["decision"] == "APPROVE"
    assert row["severity"] == "LOW"
    assert row["partition_month"] == 202311  # timestamp 1700000000 is Nov 2023
    assert row["amount"] == 150.00
    assert row["currency"] == "USD"
    assert row["mcc"] == "5411"
