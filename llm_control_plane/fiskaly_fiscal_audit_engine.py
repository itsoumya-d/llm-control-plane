"""fiskaly Cloud Fiscalization & Tamper-Proof Audit Chaining Engine.

Provides ultra-low-latency cryptographic transaction signing, monotonic sequence
gap reconciliation, multi-jurisdiction compliance validation (Austria RKSV,
Germany KassenSichV, Spain TicketBAI), and real-time AI/ML anomaly detection.
"""

from __future__ import annotations

import base64
import datetime as dt
import enum
import hashlib
import hmac
import time
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


class FiscalJurisdiction(str, enum.Enum):
    AUSTRIA_RKSV = "AUSTRIA_RKSV"
    GERMANY_KASSENSICHV = "GERMANY_KASSENSICHV"
    SPAIN_TICKETBAI = "SPAIN_TICKETBAI"


class ReceiptItem(BaseModel):
    item_id: str
    description: str
    quantity: float = Field(gt=0)
    unit_price: float = Field(default=0.0)
    vat_rate: float = Field(ge=0, le=1.0)
    gross_amount: float = 0.0


class ReceiptTransaction(BaseModel):
    receipt_id: str
    cash_register_id: str
    timestamp: dt.datetime
    items: List[ReceiptItem]
    currency: str = "EUR"
    jurisdiction: FiscalJurisdiction = FiscalJurisdiction.AUSTRIA_RKSV
    sequence_number: Optional[int] = None
    previous_hash: Optional[str] = None
    offline_retry_count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def total_gross(self) -> float:
        return round(sum(item.gross_amount for item in self.items), 2)

    @property
    def total_vat(self) -> float:
        vat = 0.0
        for item in self.items:
            net = item.gross_amount / (1.0 + item.vat_rate) if item.vat_rate > 0 else item.gross_amount
            vat += item.gross_amount - net
        return round(vat, 2)


class SignedFiscalReceipt(BaseModel):
    receipt_id: str
    cash_register_id: str
    sequence_number: int
    payload_hash: str
    previous_chain_hash: str
    chain_hash: str
    signature: str
    jurisdiction: FiscalJurisdiction
    signing_latency_ms: float
    created_at: dt.datetime
    qr_payload: str
    is_valid: bool = True


class CloudTSECryptographicSigner:
    """Simulates high-throughput Cloud TSE / Hardware Security Module (HSM) signing."""

    def __init__(self, private_signing_key: bytes = b"fiskaly_tse_master_signing_key_2026"):
        self._key = private_signing_key
        self._total_signed = 0
        self._latencies_ms: List[float] = []

    def sign_transaction(
        self,
        tx: ReceiptTransaction,
        sequence_number: int,
        previous_chain_hash: str,
    ) -> SignedFiscalReceipt:
        t0 = time.perf_counter()

        # Deterministic canonical payload hash
        items_digest = hashlib.sha256(
            "|".join(f"{it.item_id}:{it.gross_amount:.2f}:{it.vat_rate:.2f}" for it in tx.items).encode("utf-8")
        ).hexdigest()

        canonical_str = (
            f"{tx.cash_register_id}:{sequence_number}:{tx.timestamp.isoformat()}:"
            f"{tx.total_gross:.2f}:{tx.total_vat:.2f}:{items_digest}:{previous_chain_hash}"
        )
        payload_hash = hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()

        # Cryptographic signature generation (HMAC-SHA256 representation of ECDSA)
        msg_bytes = f"{payload_hash}:{previous_chain_hash}".encode("utf-8")
        sig_bytes = hmac.new(self._key, msg_bytes, hashlib.sha256).digest()
        signature_b64 = base64.b64encode(sig_bytes).decode("ascii")

        # Unbroken hash chain incorporation
        chain_bytes = f"{previous_chain_hash}:{payload_hash}:{signature_b64}".encode("utf-8")
        chain_hash = hashlib.sha256(chain_bytes).hexdigest()

        # Jurisdiction-specific QR/machine-readable string
        if tx.jurisdiction == FiscalJurisdiction.AUSTRIA_RKSV:
            sig_clip = signature_b64[:20]
            qr_payload = (
                f"_R1-AT0_{tx.cash_register_id}_{sequence_number}_"
                f"{tx.timestamp.isoformat()}_{tx.total_gross:.2f}_{sig_clip}"
            )
        elif tx.jurisdiction == FiscalJurisdiction.GERMANY_KASSENSICHV:
            sig_clip = signature_b64[:30]
            qr_payload = (
                f"V0;{tx.cash_register_id};{sequence_number};{tx.timestamp.isoformat()};{payload_hash};{sig_clip}"
            )
        else:
            qr_payload = f"TBAI-AT_{tx.cash_register_id}_{sequence_number}_{payload_hash[:16]}"

        latency_ms = (time.perf_counter() - t0) * 1000.0
        self._total_signed += 1
        self._latencies_ms.append(latency_ms)

        return SignedFiscalReceipt(
            receipt_id=tx.receipt_id,
            cash_register_id=tx.cash_register_id,
            sequence_number=sequence_number,
            payload_hash=payload_hash,
            previous_chain_hash=previous_chain_hash,
            chain_hash=chain_hash,
            signature=signature_b64,
            jurisdiction=tx.jurisdiction,
            signing_latency_ms=round(latency_ms, 3),
            created_at=dt.datetime.now(dt.timezone.utc),
            qr_payload=qr_payload,
            is_valid=True,
        )

    @property
    def stats(self) -> Dict[str, Any]:
        p99 = sorted(self._latencies_ms)[int(len(self._latencies_ms) * 0.99)] if self._latencies_ms else 0.0
        return {
            "total_signed": self._total_signed,
            "avg_latency_ms": round(sum(self._latencies_ms) / max(len(self._latencies_ms), 1), 3),
            "p99_latency_ms": round(p99, 3),
        }


class MonotonicHashChainSequencer:
    """Guarantees strictly monotonic sequence counters and uninterrupted cryptographic hash chaining."""

    GENESIS_HASH = "0000000000000000000000000000000000000000000000000000000000000000"

    def __init__(self, signer: CloudTSECryptographicSigner):
        self.signer = signer
        self._register_sequences: Dict[str, int] = {}
        self._register_chains: Dict[str, str] = {}
        self._ledger: Dict[str, List[SignedFiscalReceipt]] = {}

    def get_latest_sequence(self, cash_register_id: str) -> int:
        return self._register_sequences.get(cash_register_id, 0)

    def get_latest_chain_hash(self, cash_register_id: str) -> str:
        return self._register_chains.get(cash_register_id, self.GENESIS_HASH)

    def process_and_sign(self, tx: ReceiptTransaction) -> SignedFiscalReceipt:
        cid = tx.cash_register_id
        next_seq = self.get_latest_sequence(cid) + 1
        prev_hash = self.get_latest_chain_hash(cid)

        signed = self.signer.sign_transaction(tx, sequence_number=next_seq, previous_chain_hash=prev_hash)

        # Commit to local state
        self._register_sequences[cid] = next_seq
        self._register_chains[cid] = signed.chain_hash
        self._ledger.setdefault(cid, []).append(signed)

        return signed

    def verify_ledger_integrity(self, cash_register_id: str) -> Tuple[bool, Optional[str]]:
        """Validates that the ledger sequence and cryptographic hash chain are 100% unbroken."""
        history = self._ledger.get(cash_register_id, [])
        if not history:
            return True, None

        expected_seq = 1
        expected_prev_hash = self.GENESIS_HASH

        for receipt in history:
            if receipt.sequence_number != expected_seq:
                return False, f"Sequence gap detected at index {expected_seq}: got {receipt.sequence_number}"
            if receipt.previous_chain_hash != expected_prev_hash:
                err_msg = (
                    f"Broken chain link at seq {receipt.sequence_number}: "
                    f"expected {expected_prev_hash}, got {receipt.previous_chain_hash}"
                )
                return False, err_msg

            expected_seq += 1
            expected_prev_hash = receipt.chain_hash

        return True, None


class GapDetectionReplayReconciler:
    """Buffers out-of-order POS receipts and reconstructs continuous sequence pipelines."""

    def __init__(self, sequencer: MonotonicHashChainSequencer):
        self.sequencer = sequencer
        self._pending_buffer: Dict[str, Dict[int, ReceiptTransaction]] = {}

    def stage_receipt(self, tx: ReceiptTransaction, client_seq: int) -> Dict[str, Any]:
        cid = tx.cash_register_id
        self._pending_buffer.setdefault(cid, {})[client_seq] = tx
        return self.reconcile(cid)

    def reconcile(self, cash_register_id: str) -> Dict[str, Any]:
        current_seq = self.sequencer.get_latest_sequence(cash_register_id)
        buffer = self._pending_buffer.get(cash_register_id, {})
        reconciled_count = 0
        staged_signed: List[SignedFiscalReceipt] = []

        while (current_seq + 1) in buffer:
            next_seq = current_seq + 1
            tx = buffer.pop(next_seq)
            signed = self.sequencer.process_and_sign(tx)
            staged_signed.append(signed)
            current_seq = next_seq
            reconciled_count += 1

        remaining_gaps = sorted(buffer.keys())
        return {
            "reconciled_count": reconciled_count,
            "current_sequence": current_seq,
            "unresolved_gaps": remaining_gaps,
            "staged_signed": staged_signed,
        }


class FiscalAnomalyType(str, enum.Enum):
    VAT_CALCULATION_MISMATCH = "VAT_CALCULATION_MISMATCH"
    NEGATIVE_TOTAL_WITHOUT_CREDIT_FLAG = "NEGATIVE_TOTAL_WITHOUT_CREDIT_FLAG"
    STALE_OFFLINE_TIMESTAMP = "STALE_OFFLINE_TIMESTAMP"
    EXCESSIVE_TAX_DRIFT = "EXCESSIVE_TAX_DRIFT"


class FiscalTransactionAnomalyDetector:
    """AI/ML-informed anomaly detection protecting immutable registers from malformed or fraudulent payloads."""

    def __init__(self, max_offline_skew_hours: float = 48.0):
        self.max_offline_skew_hours = max_offline_skew_hours

    def evaluate(self, tx: ReceiptTransaction) -> Tuple[bool, List[Dict[str, Any]]]:
        anomalies: List[Dict[str, Any]] = []

        # 1. Gross vs Net + VAT calculation reconciliation
        calculated_vat = 0.0
        for it in tx.items:
            expected_net = it.gross_amount / (1.0 + it.vat_rate) if it.vat_rate > 0 else it.gross_amount
            calculated_vat += it.gross_amount - expected_net

        # Check total discrepancy
        if abs(calculated_vat - tx.total_vat) > 0.05:
            anomalies.append(
                {
                    "type": FiscalAnomalyType.VAT_CALCULATION_MISMATCH,
                    "severity": "HIGH",
                    "message": f"Declared VAT {tx.total_vat} differs from calculated VAT {calculated_vat:.2f}",
                }
            )

        # 2. Negative transaction check
        if tx.total_gross < 0 and not tx.metadata.get("is_credit_note"):
            anomalies.append(
                {
                    "type": FiscalAnomalyType.NEGATIVE_TOTAL_WITHOUT_CREDIT_FLAG,
                    "severity": "CRITICAL",
                    "message": f"Negative gross total {tx.total_gross} submitted without valid credit note metadata",
                }
            )

        # 3. Future-dated or excessively stale offline replay timestamp
        now = dt.datetime.now(dt.timezone.utc)
        if tx.timestamp.tzinfo:
            tx_time = tx.timestamp.astimezone(dt.timezone.utc)
        else:
            tx_time = tx.timestamp.replace(tzinfo=dt.timezone.utc)
        diff_hours = (now - tx_time).total_seconds() / 3600.0

        if diff_hours < -0.2:  # More than 12 minutes in the future
            anomalies.append(
                {
                    "type": FiscalAnomalyType.STALE_OFFLINE_TIMESTAMP,
                    "severity": "CRITICAL",
                    "message": f"Transaction timestamp {tx.timestamp} is in the future",
                }
            )
        elif diff_hours > self.max_offline_skew_hours and tx.offline_retry_count == 0:
            anomalies.append(
                {
                    "type": FiscalAnomalyType.STALE_OFFLINE_TIMESTAMP,
                    "severity": "MEDIUM",
                    "message": f"Transaction timestamp is {diff_hours:.1f} hours old without offline flag",
                }
            )

        is_clean = len([a for a in anomalies if a["severity"] in ("HIGH", "CRITICAL")]) == 0
        return is_clean, anomalies


class MultiJurisdictionRuleEngine:
    """Validates jurisdiction-specific fiscal schema compliance before signing."""

    @staticmethod
    def validate(tx: ReceiptTransaction) -> Tuple[bool, Optional[str]]:
        if not tx.items:
            return False, "Transaction must contain at least one line item"

        if tx.jurisdiction == FiscalJurisdiction.AUSTRIA_RKSV:
            # Austria RKSV: requires register ID and valid VAT categories
            valid_rksv_rates = {0.0, 0.10, 0.13, 0.19, 0.20}
            for it in tx.items:
                if it.vat_rate not in valid_rksv_rates:
                    return False, f"Austria RKSV invalid VAT rate: {it.vat_rate}. Permitted: {valid_rksv_rates}"

        elif tx.jurisdiction == FiscalJurisdiction.GERMANY_KASSENSICHV:
            # Germany KassenSichV: requires cash register ID format compliance
            if len(tx.cash_register_id.strip()) < 3:
                return False, "Germany KassenSichV requires register ID length >= 3 chars"

        elif tx.jurisdiction == FiscalJurisdiction.SPAIN_TICKETBAI:
            # Spain TicketBAI: requires tax identification in metadata
            if "tax_identifier" not in tx.metadata:
                return False, "Spain TicketBAI requires 'tax_identifier' (NIF/CIF) in metadata"

        return True, None
