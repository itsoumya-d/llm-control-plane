"""Unit tests for Sharesies-aligned RegTech Compliance Engine."""

import time

from llm_control_plane.regtech_compliance_engine import (
    AlertSeverity,
    AMLTransactionMonitor,
    CustodyEvent,
    EventType,
    ReconciliationStatus,
    RegTechCustodyReconciler,
    TamperEvidentAuditLedger,
)


def test_idempotent_event_handling():
    reconciler = RegTechCustodyReconciler()
    event = CustodyEvent(
        event_id="evt_001",
        idempotency_key="idemp_key_100",
        account_id="acc_usr_99",
        event_type=EventType.DEPOSIT,
        symbol="NZD",
        amount=500.0,
        market="CASH",
        timestamp=time.time(),
    )
    status1 = reconciler.process_event(event)
    assert status1 == ReconciliationStatus.MATCHED

    # Replay same event with identical idempotency key
    status2 = reconciler.process_event(event)
    assert status2 == ReconciliationStatus.DUPLICATE

    bal = reconciler.get_ledger_balance("NZD")
    assert bal["total_debits"] == 500.0
    assert bal["net_balance"] == 500.0


def test_in_order_authorization_and_settlement():
    reconciler = RegTechCustodyReconciler()
    t = time.time()
    auth = CustodyEvent(
        event_id="auth_101",
        idempotency_key="k_auth_101",
        account_id="acc_user_42",
        event_type=EventType.AUTHORIZATION,
        symbol="AIR.NZ",
        amount=150.0,
        market="NZX",
        timestamp=t,
    )
    status_auth = reconciler.process_event(auth)
    assert status_auth == ReconciliationStatus.MATCHED

    settlement = CustodyEvent(
        event_id="set_201",
        idempotency_key="k_set_201",
        account_id="acc_user_42",
        event_type=EventType.SETTLEMENT,
        symbol="AIR.NZ",
        amount=150.0,
        market="NZX",
        timestamp=t + 10,
        custodian_ref="auth_101",
    )
    status_set = reconciler.process_event(settlement)
    assert status_set == ReconciliationStatus.MATCHED

    bal = reconciler.get_ledger_balance("AIR.NZ")
    assert bal["total_debits"] == 150.0
    assert bal["total_credits"] == 150.0
    assert bal["net_balance"] == 0.0


def test_out_of_order_settlement_parking_and_resolution():
    reconciler = RegTechCustodyReconciler()
    t = time.time()

    # Settlement arrives FIRST (broker clearing file arrives before internal auth record)
    settlement = CustodyEvent(
        event_id="set_301",
        idempotency_key="k_set_301",
        account_id="acc_user_88",
        event_type=EventType.SETTLEMENT,
        symbol="AAPL",
        amount=25.0,
        market="US",
        timestamp=t,
        custodian_ref="auth_301",
    )
    status_parked = reconciler.process_event(settlement)
    assert status_parked == ReconciliationStatus.PARKED_AWAITING_AUTH
    assert reconciler.parked_count == 1

    # Later, internal order authorization arrives
    auth = CustodyEvent(
        event_id="auth_301",
        idempotency_key="k_auth_301",
        account_id="acc_user_88",
        event_type=EventType.AUTHORIZATION,
        symbol="AAPL",
        amount=25.0,
        market="US",
        timestamp=t + 15,
    )
    status_resolved = reconciler.process_event(auth)
    assert status_resolved == ReconciliationStatus.MATCHED
    assert reconciler.parked_count == 0

    bal = reconciler.get_ledger_balance("AAPL")
    assert bal["total_debits"] == 25.0
    assert bal["total_credits"] == 25.0
    assert bal["net_balance"] == 0.0


def test_amount_mismatch_detection():
    reconciler = RegTechCustodyReconciler()
    t = time.time()
    auth = CustodyEvent(
        event_id="auth_401",
        idempotency_key="k_auth_401",
        account_id="acc_user_12",
        event_type=EventType.AUTHORIZATION,
        symbol="NZD",
        amount=1000.0,
        market="CASH",
        timestamp=t,
    )
    reconciler.process_event(auth)

    settlement = CustodyEvent(
        event_id="set_401",
        idempotency_key="k_set_401",
        account_id="acc_user_12",
        event_type=EventType.SETTLEMENT,
        symbol="NZD",
        amount=990.0,  # 10 NZD mismatch
        market="CASH",
        timestamp=t + 5,
        custodian_ref="auth_401",
    )
    status = reconciler.process_event(settlement)
    assert status == ReconciliationStatus.AMOUNT_MISMATCH


def test_aml_large_transaction_threshold():
    monitor = AMLTransactionMonitor(reporting_threshold_nzd=10000.0)
    alerts = monitor.evaluate_transaction("user_101", 12500.0)
    assert len(alerts) >= 1
    assert any(a.rule_name == "LARGE_TRANSACTION_THRESHOLD" and a.severity == AlertSeverity.HIGH for a in alerts)


def test_aml_structuring_detection():
    monitor = AMLTransactionMonitor(reporting_threshold_nzd=10000.0, structuring_window_sec=3600.0)
    now = time.time()
    # Transaction 1: $8,000 (just below threshold)
    alerts1 = monitor.evaluate_transaction("user_struct_1", 8000.0, timestamp=now)
    assert not any(a.rule_name == "POTENTIAL_STRUCTURING_DETECTED" for a in alerts1)

    # Transaction 2: $8,500 within same window (total $16,500)
    alerts2 = monitor.evaluate_transaction("user_struct_1", 8500.0, timestamp=now + 600)
    assert any(a.rule_name == "POTENTIAL_STRUCTURING_DETECTED" for a in alerts2)


def test_aml_sanctions_match():
    monitor = AMLTransactionMonitor()
    alerts = monitor.evaluate_transaction(
        "user_bad",
        250.0,
        recipient_id="SANCTIONED_ENTITY_01",
    )
    assert len(alerts) == 1
    assert alerts[0].rule_name == "SANCTIONS_LIST_MATCH"
    assert alerts[0].severity == AlertSeverity.CRITICAL
    assert alerts[0].risk_score == 1.0


def test_tamper_evident_audit_ledger_integrity():
    ledger = TamperEvidentAuditLedger()
    h1 = ledger.append_audit_entry("CREATE_USER", "admin_01", {"user_id": "usr_77", "tier": "gold"})
    h2 = ledger.append_audit_entry("ASSET_TRANSFER", "system_worker", {"from": "cust_1", "to": "cust_2", "shares": 50})
    h3 = ledger.append_audit_entry("COMPLIANCE_OVERRIDE", "compliance_lead", {"reason": "PEP verification cleared"})

    assert len(ledger._entries) == 3
    assert ledger._entries[0]["prev_hash"] == TamperEvidentAuditLedger.GENESIS_HASH
    assert ledger._entries[1]["prev_hash"] == h1
    assert ledger._entries[2]["prev_hash"] == h2
    assert ledger._current_hash == h3

    is_valid, err = ledger.verify_chain_integrity()
    assert is_valid is True
    assert err is None

    manifest = ledger.export_regulatory_manifest()
    assert manifest["total_entries"] == 3
    assert manifest["chain_valid"] is True
    assert manifest["head_hash"] == h3


def test_tamper_evident_audit_ledger_tampering_detected():
    ledger = TamperEvidentAuditLedger()
    ledger.append_audit_entry("EVENT_A", "worker", {"v": 1})
    ledger.append_audit_entry("EVENT_B", "worker", {"v": 2})
    ledger.append_audit_entry("EVENT_C", "worker", {"v": 3})

    # Simulate unauthorized database modification of historical payload
    ledger._entries[1]["payload"]["v"] = 9999

    is_valid, err = ledger.verify_chain_integrity()
    assert is_valid is False
    assert "Hash corruption detected at index 1" in str(err)
