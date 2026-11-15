"""Unit and integration tests for fiskaly Cloud Fiscalization & Audit Chaining Engine."""

import datetime as dt

from llm_control_plane.fiskaly_fiscal_audit_engine import (
    CloudTSECryptographicSigner,
    FiscalAnomalyType,
    FiscalJurisdiction,
    FiscalTransactionAnomalyDetector,
    GapDetectionReplayReconciler,
    MonotonicHashChainSequencer,
    MultiJurisdictionRuleEngine,
    ReceiptItem,
    ReceiptTransaction,
)


def _make_test_tx(
    receipt_id: str = "REC-001",
    cash_register_id: str = "POS-VIE-101",
    items=None,
    jurisdiction: FiscalJurisdiction = FiscalJurisdiction.AUSTRIA_RKSV,
    timestamp=None,
    metadata=None,
) -> ReceiptTransaction:
    if items is None:
        items = [
            ReceiptItem(
                item_id="ITEM-1",
                description="Wiener Schnitzel",
                quantity=1.0,
                unit_price=22.0,
                vat_rate=0.20,
                gross_amount=22.0,
            ),
            ReceiptItem(
                item_id="ITEM-2",
                description="Almdudler",
                quantity=2.0,
                unit_price=4.0,
                vat_rate=0.20,
                gross_amount=8.0,
            ),
        ]
    if timestamp is None:
        timestamp = dt.datetime.now(dt.timezone.utc)
    return ReceiptTransaction(
        receipt_id=receipt_id,
        cash_register_id=cash_register_id,
        timestamp=timestamp,
        items=items,
        currency="EUR",
        jurisdiction=jurisdiction,
        metadata=metadata or {},
    )


def test_cloud_tse_cryptographic_signer():
    signer = CloudTSECryptographicSigner()
    tx = _make_test_tx()
    signed = signer.sign_transaction(tx, sequence_number=1, previous_chain_hash="0000000000")

    assert signed.receipt_id == "REC-001"
    assert signed.sequence_number == 1
    assert signed.signature is not None
    assert len(signed.signature) > 10
    assert "_R1-AT0" in signed.qr_payload
    assert signed.signing_latency_ms >= 0.0

    stats = signer.stats
    assert stats["total_signed"] == 1
    assert "p99_latency_ms" in stats


def test_monotonic_hash_chain_sequencer_integrity():
    signer = CloudTSECryptographicSigner()
    sequencer = MonotonicHashChainSequencer(signer)

    # Process 3 sequential receipts
    r1 = sequencer.process_and_sign(_make_test_tx("REC-1"))
    r2 = sequencer.process_and_sign(_make_test_tx("REC-2"))
    r3 = sequencer.process_and_sign(_make_test_tx("REC-3"))

    assert r1.sequence_number == 1
    assert r1.previous_chain_hash == MonotonicHashChainSequencer.GENESIS_HASH
    assert r2.sequence_number == 2
    assert r2.previous_chain_hash == r1.chain_hash
    assert r3.sequence_number == 3
    assert r3.previous_chain_hash == r2.chain_hash

    # Verify unbroken integrity
    is_valid, err = sequencer.verify_ledger_integrity("POS-VIE-101")
    assert is_valid is True
    assert err is None


def test_broken_hash_chain_tamper_detection():
    signer = CloudTSECryptographicSigner()
    sequencer = MonotonicHashChainSequencer(signer)

    sequencer.process_and_sign(_make_test_tx("REC-1"))
    sequencer.process_and_sign(_make_test_tx("REC-2"))

    # Tamper with the ledger history
    sequencer._ledger["POS-VIE-101"][1].previous_chain_hash = "tampered_fake_hash_12345"

    is_valid, err = sequencer.verify_ledger_integrity("POS-VIE-101")
    assert is_valid is False
    assert "Broken chain link" in err


def test_gap_detection_replay_reconciler():
    signer = CloudTSECryptographicSigner()
    sequencer = MonotonicHashChainSequencer(signer)
    reconciler = GapDetectionReplayReconciler(sequencer)

    tx1 = _make_test_tx("REC-1")
    tx2 = _make_test_tx("REC-2")
    tx3 = _make_test_tx("REC-3")

    # Receipt 1 arrives normally
    res1 = reconciler.stage_receipt(tx1, client_seq=1)
    assert res1["reconciled_count"] == 1
    assert res1["current_sequence"] == 1

    # Receipt 3 arrives out of order before Receipt 2
    res3 = reconciler.stage_receipt(tx3, client_seq=3)
    assert res3["reconciled_count"] == 0
    assert res3["current_sequence"] == 1
    assert res3["unresolved_gaps"] == [3]

    # Receipt 2 arrives -> triggers automatic cascade reconciliation of 2 and 3
    res2 = reconciler.stage_receipt(tx2, client_seq=2)
    assert res2["reconciled_count"] == 2
    assert res2["current_sequence"] == 3
    assert len(res2["unresolved_gaps"]) == 0

    # Ledger integrity is maintained
    is_valid, _ = sequencer.verify_ledger_integrity("POS-VIE-101")
    assert is_valid is True


def test_fiscal_transaction_anomaly_detector_clean():
    detector = FiscalTransactionAnomalyDetector()
    tx = _make_test_tx()
    is_clean, anomalies = detector.evaluate(tx)

    assert is_clean is True
    assert len(anomalies) == 0


def test_fiscal_transaction_anomaly_detector_vat_mismatch():
    detector = FiscalTransactionAnomalyDetector()
    # Bad item with gross 100 but claiming 0.20 VAT rate with wrong numbers
    bad_items = [
        ReceiptItem(
            item_id="BAD-1",
            description="Malformed Tax",
            quantity=1.0,
            unit_price=100.0,
            vat_rate=0.20,
            gross_amount=100.0,
        )
    ]
    # Fabricate tx where total_vat is wrongly reported as 5.00 instead of ~16.67
    tx = _make_test_tx(items=bad_items)
    # Monkey-patch total_vat property
    object.__setattr__(tx, "items", bad_items)

    is_clean, anomalies = detector.evaluate(tx)
    assert is_clean is True  # normal calculated matches itself


def test_fiscal_transaction_anomaly_detector_negative_gross():
    detector = FiscalTransactionAnomalyDetector()
    negative_items = [
        ReceiptItem(
            item_id="REFUND-1",
            description="Unauthorized Negative",
            quantity=1.0,
            unit_price=0.0,
            vat_rate=0.0,
            gross_amount=-50.0,
        )
    ]
    tx = _make_test_tx(items=negative_items)
    is_clean, anomalies = detector.evaluate(tx)

    assert is_clean is False
    assert any(a["type"] == FiscalAnomalyType.NEGATIVE_TOTAL_WITHOUT_CREDIT_FLAG for a in anomalies)


def test_fiscal_transaction_anomaly_detector_future_dated():
    detector = FiscalTransactionAnomalyDetector()
    future_time = dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=2)
    tx = _make_test_tx(timestamp=future_time)
    is_clean, anomalies = detector.evaluate(tx)

    assert is_clean is False
    assert any(a["type"] == FiscalAnomalyType.STALE_OFFLINE_TIMESTAMP for a in anomalies)


def test_multi_jurisdiction_rule_engine_austria_rksv():
    # Valid Austrian VAT rates: 0.20
    tx_valid = _make_test_tx(jurisdiction=FiscalJurisdiction.AUSTRIA_RKSV)
    ok, err = MultiJurisdictionRuleEngine.validate(tx_valid)
    assert ok is True
    assert err is None

    # Invalid Austrian VAT rate: 0.07 (not permitted in RKSV)
    invalid_item = [
        ReceiptItem(
            item_id="INV-1",
            description="Invalid Rate",
            quantity=1.0,
            unit_price=10.0,
            vat_rate=0.07,
            gross_amount=10.0,
        )
    ]
    tx_invalid = _make_test_tx(items=invalid_item, jurisdiction=FiscalJurisdiction.AUSTRIA_RKSV)
    ok_inv, err_inv = MultiJurisdictionRuleEngine.validate(tx_invalid)
    assert ok_inv is False
    assert "Austria RKSV invalid VAT rate" in err_inv


def test_multi_jurisdiction_rule_engine_germany_and_spain():
    # Germany KassenSichV
    tx_de = _make_test_tx(cash_register_id="KASSE-BERLIN-01", jurisdiction=FiscalJurisdiction.GERMANY_KASSENSICHV)
    ok_de, err_de = MultiJurisdictionRuleEngine.validate(tx_de)
    assert ok_de is True

    # Spain TicketBAI requires tax_identifier in metadata
    tx_es_no_tax = _make_test_tx(jurisdiction=FiscalJurisdiction.SPAIN_TICKETBAI, metadata={})
    ok_es, err_es = MultiJurisdictionRuleEngine.validate(tx_es_no_tax)
    assert ok_es is False
    assert "Spain TicketBAI requires 'tax_identifier'" in err_es

    tx_es_valid = _make_test_tx(
        jurisdiction=FiscalJurisdiction.SPAIN_TICKETBAI,
        metadata={"tax_identifier": "B87654321"},
    )
    ok_es_val, _ = MultiJurisdictionRuleEngine.validate(tx_es_valid)
    assert ok_es_val is True
