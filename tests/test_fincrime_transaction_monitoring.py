"""Unit tests for FinCrime Transaction Monitoring & AML Compliance Evaluator."""

import datetime as dt
import pytest
from llm_control_plane.fincrime_transaction_monitoring import (
    AlertSeverity,
    FinCrimeComplianceEvaluator,
    RuleType,
    SuspiciousActivityReport,
    Transaction,
    TransactionMonitoringEngine,
)


def test_clean_transaction_passes_without_alerts():
    engine = TransactionMonitoringEngine()
    now = dt.datetime.now(dt.timezone.utc)
    tx = Transaction(
        transaction_id="tx_001",
        account_id="acc_123",
        counterparty_id="cpty_456",
        amount=150.0,
        currency="EUR",
        timestamp=now,
        country_code="IE",
        payment_method="SEPA",
    )

    res = engine.ingest_transaction(tx)
    assert not res.is_flagged
    assert len(res.violations) == 0
    assert res.risk_score == 0.0
    assert res.evaluation_latency_ms >= 0.0


def test_sanctions_screening_triggers_critical_alert():
    engine = TransactionMonitoringEngine()
    engine.register_sanctioned_parties(["SANCTIONED_CORP_01"])

    now = dt.datetime.now(dt.timezone.utc)
    tx = Transaction(
        transaction_id="tx_sanction_01",
        account_id="acc_normal",
        counterparty_id="SANCTIONED_CORP_01",
        amount=2500.0,
        currency="EUR",
        timestamp=now,
        country_code="DE",
    )

    res = engine.ingest_transaction(tx)
    assert res.is_flagged
    assert any(v.rule_type == RuleType.SANCTIONS_SCREENING for v in res.violations)
    assert any(v.severity == AlertSeverity.CRITICAL for v in res.violations)
    assert res.risk_score >= 50.0


def test_high_risk_jurisdiction_triggers_alert():
    engine = TransactionMonitoringEngine()
    now = dt.datetime.now(dt.timezone.utc)
    tx = Transaction(
        transaction_id="tx_fatf_01",
        account_id="acc_risk",
        counterparty_id="cpty_foreign",
        amount=5000.0,
        currency="EUR",
        timestamp=now,
        country_code="IR",  # Iran - FATF black list
    )

    res = engine.ingest_transaction(tx)
    assert res.is_flagged
    assert any(v.rule_type == RuleType.HIGH_RISK_JURISDICTION for v in res.violations)
    assert any(v.severity == AlertSeverity.HIGH for v in res.violations)
    assert res.risk_score >= 30.0


def test_structuring_smurfing_detection():
    engine = TransactionMonitoringEngine(structuring_lower_bound=8500.0)
    now = dt.datetime.now(dt.timezone.utc)

    # First transaction just below €10,000 reporting threshold
    tx1 = Transaction(
        transaction_id="tx_smurf_01",
        account_id="acc_struct",
        counterparty_id="cpty_broker",
        amount=9500.0,
        currency="EUR",
        timestamp=now - dt.timedelta(hours=2),
        country_code="IE",
    )
    res1 = engine.ingest_transaction(tx1)
    # Only 1 transaction so far, structuring requires multiple
    assert not any(v.rule_type == RuleType.STRUCTURING_SMURFING for v in res1.violations)

    # Second transaction within window just below threshold
    tx2 = Transaction(
        transaction_id="tx_smurf_02",
        account_id="acc_struct",
        counterparty_id="cpty_broker",
        amount=9800.0,
        currency="EUR",
        timestamp=now - dt.timedelta(minutes=30),
        country_code="IE",
    )
    res2 = engine.ingest_transaction(tx2)
    assert res2.is_flagged
    assert any(v.rule_type == RuleType.STRUCTURING_SMURFING for v in res2.violations)
    struct_violation = next(v for v in res2.violations if v.rule_type == RuleType.STRUCTURING_SMURFING)
    assert "tx_smurf_01" in struct_violation.matched_transaction_ids
    assert "tx_smurf_02" in struct_violation.matched_transaction_ids


def test_velocity_and_volume_threshold_exceeded():
    engine = TransactionMonitoringEngine(
        velocity_window_hours=24,
        max_daily_amount=20000.0,
        max_daily_transactions=3,
    )
    now = dt.datetime.now(dt.timezone.utc)

    # Add 4 transactions
    for i in range(4):
        tx = Transaction(
            transaction_id=f"tx_vol_{i}",
            account_id="acc_velocity",
            counterparty_id=f"cpty_{i}",
            amount=6000.0,
            currency="EUR",
            timestamp=now - dt.timedelta(hours=i),
            country_code="IE",
        )
        res = engine.ingest_transaction(tx)

    # 4th transaction should trigger velocity violation
    assert res.is_flagged
    assert any(v.rule_type == RuleType.VELOCITY_THRESHOLD for v in res.violations)


def test_sar_generation_and_grounding_verification():
    engine = TransactionMonitoringEngine()
    engine.register_sanctioned_parties(["SANCTIONED_ENTITY_X"])
    evaluator = FinCrimeComplianceEvaluator(engine)

    now = dt.datetime.now(dt.timezone.utc)
    tx = Transaction(
        transaction_id="tx_sar_001",
        account_id="acc_audit_01",
        counterparty_id="SANCTIONED_ENTITY_X",
        amount=125000.0,
        currency="EUR",
        timestamp=now,
        country_code="IE",
    )
    eval_res = engine.ingest_transaction(tx)
    assert eval_res.is_flagged

    sar = evaluator.generate_audit_report("acc_audit_01", eval_res)
    assert sar.report_id.startswith("SAR-")
    assert sar.account_id == "acc_audit_01"
    assert sar.severity == AlertSeverity.CRITICAL
    assert sar.recommended_action == "IMMEDIATE_FREEZE_AND_REGULATORY_DISCLOSURE"
    assert "tx_sar_001" in sar.supporting_transaction_ids
    assert len(sar.audit_hash) == 64  # SHA256 length


def test_sar_grounding_verification_detects_hallucination():
    engine = TransactionMonitoringEngine()
    evaluator = FinCrimeComplianceEvaluator(engine)

    # Fabricate a SAR with a transaction ID not in records
    ungrounded_sar = SuspiciousActivityReport(
        report_id="SAR-fake",
        account_id="acc_fake",
        severity=AlertSeverity.HIGH,
        summary="Fake",
        narrative="Fake",
        recommended_action="REVIEW",
        supporting_transaction_ids=["tx_ghost_999"],
        audit_hash="fakehash",
    )

    real_txs = [
        Transaction(
            transaction_id="tx_real_1",
            account_id="acc_fake",
            counterparty_id="cpty_1",
            amount=100.0,
            currency="EUR",
            timestamp=dt.datetime.now(dt.timezone.utc),
            country_code="IE",
        )
    ]

    with pytest.raises(ValueError, match="Grounding failure"):
        evaluator.verify_grounding(ungrounded_sar, real_txs)
