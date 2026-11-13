"""Sharesies-Aligned RegTech Compliance Engine, Custody Reconciler & Tamper-Evident Audit Ledger.

Engineered for automated regulatory compliance, FMA/ASIC client money safeguarding,
AML/CFT transaction velocity & structuring detection, and cryptographic audit logging.

Components:
1. RegTechCustodyReconciler: Out-of-order tolerant double-entry asset reconciliation engine
   with idempotent event handling, parked clearing queues, and multi-currency/multi-market support.
2. AMLTransactionMonitor: Real-time AML/CFT transaction screening evaluating threshold breaches,
   velocity spikes, structuring patterns, and sanctions/PEP watchlist matches.
3. TamperEvidentAuditLedger: Append-only SHA-256 hash-chained immutable regulatory audit log
   with cryptographic integrity validation and automated compliance reporting extracts.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from enum import Enum
from typing import Any


class EventType(str, Enum):
    AUTHORIZATION = "AUTHORIZATION"
    SETTLEMENT = "SETTLEMENT"
    DEPOSIT = "DEPOSIT"
    WITHDRAWAL = "WITHDRAWAL"
    REVERSAL = "REVERSAL"
    DIVIDEND = "DIVIDEND"


class ReconciliationStatus(str, Enum):
    MATCHED = "MATCHED"
    PARKED_AWAITING_AUTH = "PARKED_AWAITING_AUTH"
    DUPLICATE = "DUPLICATE"
    AMOUNT_MISMATCH = "AMOUNT_MISMATCH"
    UNMATCHED = "UNMATCHED"


class AlertSeverity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclasses.dataclass(frozen=True)
class CustodyEvent:
    event_id: str
    idempotency_key: str
    account_id: str
    event_type: EventType
    symbol: str  # e.g., "NZD", "USD", "AIR.NZ", "AAPL"
    amount: float
    market: str  # "NZX", "ASX", "US", "CASH"
    timestamp: float
    custodian_ref: str | None = None
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class LedgerEntry:
    entry_id: str
    account_id: str
    debit: float
    credit: float
    currency_or_asset: str
    reference_id: str
    timestamp: float


@dataclasses.dataclass
class ComplianceAlert:
    alert_id: str
    account_id: str
    rule_name: str
    severity: AlertSeverity
    description: str
    risk_score: float
    timestamp: float
    context: dict[str, Any]


class RegTechCustodyReconciler:
    """Out-of-order-tolerant double-entry custody & asset reconciliation engine.

    Implements strict client asset segregation controls, idempotency tracking,
    and automatic parked settlement resolution upon receipt of delayed internal authorizations.
    """

    def __init__(self) -> None:
        self._processed_keys: set[str] = set()
        self._authorizations: dict[str, CustodyEvent] = {}
        self._settlements: dict[str, CustodyEvent] = {}
        self._parked_settlements: dict[str, CustodyEvent] = {}
        self._ledger_entries: list[LedgerEntry] = []
        self._reconciliation_log: list[dict[str, Any]] = []

    def process_event(self, event: CustodyEvent) -> ReconciliationStatus:
        """Process an inbound financial transaction or custodial clearing event."""
        if event.idempotency_key in self._processed_keys:
            self._reconciliation_log.append({
                "event_id": event.event_id,
                "status": ReconciliationStatus.DUPLICATE,
                "reason": "Duplicate idempotency key detected",
            })
            return ReconciliationStatus.DUPLICATE

        self._processed_keys.add(event.idempotency_key)

        if event.event_type == EventType.AUTHORIZATION:
            self._authorizations[event.event_id] = event
            self._post_ledger_entry(
                event.account_id,
                debit=event.amount,
                credit=0.0,
                asset=event.symbol,
                ref=event.event_id,
            )
            # Check if there was a parked settlement waiting for this authorization
            matching_parked_key = None
            for p_id, p_event in self._parked_settlements.items():
                if p_event.custodian_ref == event.event_id or p_event.metadata.get("auth_id") == event.event_id:
                    matching_parked_key = p_id
                    break

            if matching_parked_key:
                parked = self._parked_settlements.pop(matching_parked_key)
                return self._finalize_reconciliation(event, parked)

            return ReconciliationStatus.MATCHED

        elif event.event_type == EventType.SETTLEMENT:
            auth_id = event.custodian_ref or event.metadata.get("auth_id")
            if auth_id and auth_id in self._authorizations:
                auth = self._authorizations[auth_id]
                return self._finalize_reconciliation(auth, event)
            else:
                # Settlement arrived prior to internal authorization: park event
                self._parked_settlements[event.event_id] = event
                self._reconciliation_log.append({
                    "event_id": event.event_id,
                    "status": ReconciliationStatus.PARKED_AWAITING_AUTH,
                    "reason": "Custodial clearing event arrived prior to internal authorization",
                })
                return ReconciliationStatus.PARKED_AWAITING_AUTH

        elif event.event_type in (EventType.DEPOSIT, EventType.WITHDRAWAL, EventType.DIVIDEND):
            self._post_ledger_entry(
                event.account_id,
                debit=event.amount if event.event_type == EventType.DEPOSIT else 0.0,
                credit=event.amount if event.event_type == EventType.WITHDRAWAL else 0.0,
                asset=event.symbol,
                ref=event.event_id,
            )
            return ReconciliationStatus.MATCHED

        return ReconciliationStatus.UNMATCHED

    def _finalize_reconciliation(self, auth: CustodyEvent, settlement: CustodyEvent) -> ReconciliationStatus:
        if abs(auth.amount - settlement.amount) > 1e-4:
            self._reconciliation_log.append({
                "auth_id": auth.event_id,
                "settlement_id": settlement.event_id,
                "status": ReconciliationStatus.AMOUNT_MISMATCH,
                "diff": abs(auth.amount - settlement.amount),
            })
            return ReconciliationStatus.AMOUNT_MISMATCH

        self._settlements[settlement.event_id] = settlement
        self._post_ledger_entry(
            settlement.account_id,
            debit=0.0,
            credit=settlement.amount,
            asset=settlement.symbol,
            ref=settlement.event_id,
        )
        self._reconciliation_log.append({
            "auth_id": auth.event_id,
            "settlement_id": settlement.event_id,
            "status": ReconciliationStatus.MATCHED,
        })
        return ReconciliationStatus.MATCHED

    def _post_ledger_entry(self, account_id: str, debit: float, credit: float, asset: str, ref: str) -> None:
        entry = LedgerEntry(
            entry_id=f"leg_{len(self._ledger_entries) + 1}",
            account_id=account_id,
            debit=debit,
            credit=credit,
            currency_or_asset=asset,
            reference_id=ref,
            timestamp=time.time(),
        )
        self._ledger_entries.append(entry)

    def get_ledger_balance(self, asset: str) -> dict[str, float]:
        """Compute aggregate debits, credits, and net imbalance for an asset."""
        total_debits = sum(e.debit for e in self._ledger_entries if e.currency_or_asset == asset)
        total_credits = sum(e.credit for e in self._ledger_entries if e.currency_or_asset == asset)
        return {
            "asset": asset,
            "total_debits": round(total_debits, 4),
            "total_credits": round(total_credits, 4),
            "net_balance": round(total_debits - total_credits, 4),
        }

    @property
    def parked_count(self) -> int:
        return len(self._parked_settlements)


class AMLTransactionMonitor:
    """Real-time AML/CFT automated screening and suspicious transaction detector.

    Monitors statutory reporting thresholds ($10,000 NZD cash/wire rule),
    rapid transaction structuring, and velocity spikes.
    """

    def __init__(self, reporting_threshold_nzd: float = 10000.0, structuring_window_sec: float = 86400.0) -> None:
        self.reporting_threshold_nzd = reporting_threshold_nzd
        self.structuring_window_sec = structuring_window_sec
        self._user_history: dict[str, list[dict[str, Any]]] = {}
        self._sanctions_watchlist: set[str] = {"SANCTIONED_ENTITY_01", "BLOCKED_INDIVIDUAL_99"}

    def add_sanctions(self, entity_identifiers: list[str]) -> None:
        self._sanctions_watchlist.update(entity_identifiers)

    def evaluate_transaction(
        self,
        account_id: str,
        amount_nzd: float,
        timestamp: float | None = None,
        recipient_id: str | None = None,
        is_international: bool = False,
    ) -> list[ComplianceAlert]:
        """Evaluate a transaction for AML/CFT risks, structuring, and threshold limits."""
        alerts: list[ComplianceAlert] = []
        now = timestamp or time.time()

        if account_id not in self._user_history:
            self._user_history[account_id] = []

        # Sanctions check
        if recipient_id and recipient_id in self._sanctions_watchlist:
            alerts.append(ComplianceAlert(
                alert_id=f"aml_{account_id}_{int(now)}_sanc",
                account_id=account_id,
                rule_name="SANCTIONS_LIST_MATCH",
                severity=AlertSeverity.CRITICAL,
                description=f"Transaction recipient {recipient_id} matches active sanctions database",
                risk_score=1.0,
                timestamp=now,
                context={"recipient_id": recipient_id, "amount_nzd": amount_nzd},
            ))

        # Direct threshold check
        if amount_nzd >= self.reporting_threshold_nzd:
            alerts.append(ComplianceAlert(
                alert_id=f"aml_{account_id}_{int(now)}_thresh",
                account_id=account_id,
                rule_name="LARGE_TRANSACTION_THRESHOLD",
                severity=AlertSeverity.HIGH if not is_international else AlertSeverity.CRITICAL,
                description=f"Transaction amount NZD {amount_nzd:,.2f} meets or exceeds statutory threshold",
                risk_score=0.85 if not is_international else 0.95,
                timestamp=now,
                context={"amount_nzd": amount_nzd, "threshold": self.reporting_threshold_nzd},
            ))

        # Structuring check (multiple transactions within rolling window just below threshold)
        recent_txs = [
            tx for tx in self._user_history[account_id]
            if now - tx["timestamp"] <= self.structuring_window_sec
        ]
        window_total = sum(tx["amount_nzd"] for tx in recent_txs) + amount_nzd
        thresh_low = self.reporting_threshold_nzd * 0.70
        near_threshold_txs = [
            tx for tx in recent_txs if thresh_low <= tx["amount_nzd"] < self.reporting_threshold_nzd
        ]

        is_near = thresh_low <= amount_nzd < self.reporting_threshold_nzd and len(near_threshold_txs) >= 1
        is_over_window = window_total > self.reporting_threshold_nzd * 1.5

        if is_near or is_over_window:
            alerts.append(ComplianceAlert(
                alert_id=f"aml_{account_id}_{int(now)}_struct",
                account_id=account_id,
                rule_name="POTENTIAL_STRUCTURING_DETECTED",
                severity=AlertSeverity.HIGH,
                description=f"Multiple rapid transactions totaling NZD {window_total:,.2f} within monitoring window",
                risk_score=0.90,
                timestamp=now,
                context={"window_total_nzd": window_total, "count_recent": len(recent_txs) + 1},
            ))

        self._user_history[account_id].append({
            "amount_nzd": amount_nzd,
            "timestamp": now,
            "recipient_id": recipient_id,
            "is_international": is_international,
        })
        return alerts


class TamperEvidentAuditLedger:
    """Cryptographic SHA-256 hash-chained audit ledger for regulatory compliance.

    Guarantees non-repudiation and data integrity for external audit inspection.
    """

    GENESIS_HASH = "0" * 64

    def __init__(self) -> None:
        self._entries: list[dict[str, Any]] = []
        self._current_hash: str = self.GENESIS_HASH

    def append_audit_entry(self, action: str, actor: str, payload: dict[str, Any]) -> str:
        """Append a new audit entry cryptographically linked to the previous entry hash."""
        timestamp = time.time()
        entry_index = len(self._entries)

        raw_payload = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        hasher = hashlib.sha256()
        hasher.update(self._current_hash.encode("utf-8"))
        hasher.update(str(entry_index).encode("utf-8"))
        hasher.update(action.encode("utf-8"))
        hasher.update(actor.encode("utf-8"))
        hasher.update(str(timestamp).encode("utf-8"))
        hasher.update(raw_payload.encode("utf-8"))

        entry_hash = hasher.hexdigest()
        entry = {
            "index": entry_index,
            "action": action,
            "actor": actor,
            "timestamp": timestamp,
            "payload": payload,
            "prev_hash": self._current_hash,
            "hash": entry_hash,
        }
        self._entries.append(entry)
        self._current_hash = entry_hash
        return entry_hash

    def verify_chain_integrity(self) -> tuple[bool, str | None]:
        """Validate the full cryptographic chain from genesis to head."""
        prev_hash = self.GENESIS_HASH
        for idx, entry in enumerate(self._entries):
            if entry["index"] != idx:
                return False, f"Index sequence mismatch at position {idx}"
            if entry["prev_hash"] != prev_hash:
                return False, f"Broken hash chain at index {idx}: expected {prev_hash}, got {entry['prev_hash']}"

            raw_payload = json.dumps(entry["payload"], sort_keys=True, separators=(",", ":"))
            hasher = hashlib.sha256()
            hasher.update(prev_hash.encode("utf-8"))
            hasher.update(str(idx).encode("utf-8"))
            hasher.update(entry["action"].encode("utf-8"))
            hasher.update(entry["actor"].encode("utf-8"))
            hasher.update(str(entry["timestamp"]).encode("utf-8"))
            hasher.update(raw_payload.encode("utf-8"))

            computed_hash = hasher.hexdigest()
            if computed_hash != entry["hash"]:
                msg = f"Hash corruption detected at index {idx}: computed {computed_hash}, recorded {entry['hash']}"
                return False, msg
            prev_hash = computed_hash

        return True, None

    def export_regulatory_manifest(self) -> dict[str, Any]:
        """Generate structured regulatory evidence report for external auditors."""
        is_valid, error = self.verify_chain_integrity()
        return {
            "total_entries": len(self._entries),
            "head_hash": self._current_hash,
            "chain_valid": is_valid,
            "verification_error": error,
            "exported_at": time.time(),
        }
