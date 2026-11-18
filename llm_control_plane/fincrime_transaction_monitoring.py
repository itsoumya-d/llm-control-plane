"""FinCrime Transaction Monitoring & AML Compliance Evaluator.

Provides real-time event-driven transaction screening, velocity and structuring detection,
sanctioned jurisdiction filtering, and agentic AI compliance audit generation with deterministic
grounding verification for regulated financial institutions.
"""

from __future__ import annotations

import datetime as dt
import enum
import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set


class AlertSeverity(str, enum.Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RuleType(str, enum.Enum):
    VELOCITY_THRESHOLD = "VELOCITY_THRESHOLD"
    STRUCTURING_SMURFING = "STRUCTURING_SMURFING"
    HIGH_RISK_JURISDICTION = "HIGH_RISK_JURISDICTION"
    RAPID_IN_OUT = "RAPID_IN_OUT"
    SANCTIONS_SCREENING = "SANCTIONS_SCREENING"


@dataclass(frozen=True)
class Transaction:
    transaction_id: str
    account_id: str
    counterparty_id: str
    amount: float
    currency: str
    timestamp: dt.datetime
    country_code: str
    payment_method: str = "SEPA"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "account_id": self.account_id,
            "counterparty_id": self.counterparty_id,
            "amount": self.amount,
            "currency": self.currency,
            "timestamp": self.timestamp.isoformat(),
            "country_code": self.country_code,
            "payment_method": self.payment_method,
            "metadata": self.metadata,
        }


@dataclass
class RuleViolation:
    rule_type: RuleType
    severity: AlertSeverity
    rule_name: str
    details: str
    matched_transaction_ids: List[str]


@dataclass
class EvaluationResult:
    transaction_id: str
    is_flagged: bool
    violations: List[RuleViolation] = field(default_factory=list)
    risk_score: float = 0.0  # Normalized 0.0 to 100.0
    evaluation_latency_ms: float = 0.0


@dataclass
class SuspiciousActivityReport:
    report_id: str
    account_id: str
    severity: AlertSeverity
    summary: str
    narrative: str
    recommended_action: str
    supporting_transaction_ids: List[str]
    audit_hash: str


class TransactionMonitoringEngine:
    """In-memory distributed sliding-window transaction monitoring and screening engine."""

    HIGH_RISK_COUNTRIES = {"IR", "KP", "MM", "SY", "RU"}
    REPORTING_THRESHOLD_EUR = 10000.0

    def __init__(
        self,
        velocity_window_hours: int = 24,
        max_daily_amount: float = 50000.0,
        max_daily_transactions: int = 10,
        structuring_lower_bound: float = 8500.0,
    ) -> None:
        self.velocity_window_hours = velocity_window_hours
        self.max_daily_amount = max_daily_amount
        self.max_daily_transactions = max_daily_transactions
        self.structuring_lower_bound = structuring_lower_bound
        self._history: Dict[str, List[Transaction]] = {}
        self._sanctioned_parties: Set[str] = set()

    def register_sanctioned_parties(self, party_ids: List[str]) -> None:
        for pid in party_ids:
            self._sanctioned_parties.add(pid.strip().upper())

    def ingest_transaction(self, tx: Transaction) -> EvaluationResult:
        t0 = dt.datetime.now(dt.timezone.utc)
        violations: List[RuleViolation] = []

        history = self._history.setdefault(tx.account_id, [])
        history.append(tx)

        # 1. Sanctions Screening
        cpty = tx.counterparty_id.strip().upper()
        acct = tx.account_id.strip().upper()
        if cpty in self._sanctioned_parties or acct in self._sanctioned_parties:
            violations.append(
                RuleViolation(
                    rule_type=RuleType.SANCTIONS_SCREENING,
                    severity=AlertSeverity.CRITICAL,
                    rule_name="Target Counterparty on Active Sanctions Watchlist",
                    details=f"Counterparty {tx.counterparty_id} matches designated entity.",
                    matched_transaction_ids=[tx.transaction_id],
                )
            )

        # 2. High-Risk Jurisdiction Screening
        if tx.country_code.upper() in self.HIGH_RISK_COUNTRIES:
            violations.append(
                RuleViolation(
                    rule_type=RuleType.HIGH_RISK_JURISDICTION,
                    severity=AlertSeverity.HIGH,
                    rule_name="FATF High-Risk Jurisdiction Outflow/Inflow",
                    details=f"Transaction routed through high-risk territory: {tx.country_code}.",
                    matched_transaction_ids=[tx.transaction_id],
                )
            )

        # Filter recent window for temporal velocity & structuring checks
        cutoff = tx.timestamp - dt.timedelta(hours=self.velocity_window_hours)
        recent_txs = [t for t in history if t.timestamp >= cutoff]

        # 3. Structuring / Smurfing Detection
        structuring_candidates = [
            t
            for t in recent_txs
            if self.structuring_lower_bound <= t.amount < self.REPORTING_THRESHOLD_EUR
        ]
        if len(structuring_candidates) >= 2:
            bound = self.structuring_lower_bound
            thresh = self.REPORTING_THRESHOLD_EUR
            win = self.velocity_window_hours
            msg = (
                f"Detected {len(structuring_candidates)} transfers between €{bound:.0f} "
                f"and €{thresh:.0f} within {win}h."
            )
            violations.append(
                RuleViolation(
                    rule_type=RuleType.STRUCTURING_SMURFING,
                    severity=AlertSeverity.HIGH,
                    rule_name="Suspected Transaction Structuring (Smurfing)",
                    details=msg,
                    matched_transaction_ids=[t.transaction_id for t in structuring_candidates],
                )
            )

        # 4. Velocity & Cumulative Volume Thresholds
        recent_volume = sum(t.amount for t in recent_txs)
        if len(recent_txs) > self.max_daily_transactions or recent_volume > self.max_daily_amount:
            sev = AlertSeverity.MEDIUM if recent_volume <= self.max_daily_amount * 1.5 else AlertSeverity.HIGH
            count = len(recent_txs)
            win = self.velocity_window_hours
            detail_msg = f"Account accumulated {count} transactions totaling €{recent_volume:.2f} within {win}h."
            violations.append(
                RuleViolation(
                    rule_type=RuleType.VELOCITY_THRESHOLD,
                    severity=sev,
                    rule_name="Excessive Account Velocity and Volume",
                    details=detail_msg,
                    matched_transaction_ids=[t.transaction_id for t in recent_txs],
                )
            )

        # Calculate composite risk score
        risk_score = 0.0
        for v in violations:
            if v.severity == AlertSeverity.CRITICAL:
                risk_score += 50.0
            elif v.severity == AlertSeverity.HIGH:
                risk_score += 30.0
            elif v.severity == AlertSeverity.MEDIUM:
                risk_score += 15.0
            else:
                risk_score += 5.0

        risk_score = min(100.0, risk_score)
        t1 = dt.datetime.now(dt.timezone.utc)
        elapsed_ms = (t1 - t0).total_seconds() * 1000.0

        return EvaluationResult(
            transaction_id=tx.transaction_id,
            is_flagged=len(violations) > 0,
            violations=violations,
            risk_score=risk_score,
            evaluation_latency_ms=elapsed_ms,
        )

    def get_account_history(self, account_id: str) -> List[Transaction]:
        return list(self._history.get(account_id, []))


class FinCrimeComplianceEvaluator:
    """Agentic AI Compliance Evaluator with Grounding Verification & SAR Generation."""

    def __init__(self, engine: TransactionMonitoringEngine) -> None:
        self.engine = engine

    def generate_audit_report(
        self, account_id: str, evaluation_result: EvaluationResult
    ) -> SuspiciousActivityReport:
        if not evaluation_result.is_flagged:
            raise ValueError("Cannot generate SAR for unflagged evaluation result.")

        tx_history = self.engine.get_account_history(account_id)
        matched_ids = set()
        for v in evaluation_result.violations:
            matched_ids.update(v.matched_transaction_ids)

        relevant_txs = [t for t in tx_history if t.transaction_id in matched_ids]
        total_amount = sum(t.amount for t in relevant_txs)
        currencies = sorted(list({t.currency for t in relevant_txs}))
        curr_str = "/".join(currencies)

        highest_severity = AlertSeverity.LOW
        for v in evaluation_result.violations:
            if v.severity == AlertSeverity.CRITICAL:
                highest_severity = AlertSeverity.CRITICAL
                break
            elif v.severity == AlertSeverity.HIGH:
                highest_severity = AlertSeverity.HIGH
            elif v.severity == AlertSeverity.MEDIUM and highest_severity == AlertSeverity.LOW:
                highest_severity = AlertSeverity.MEDIUM

        narrative_lines = [
            f"Account {account_id} triggered automated financial crime screening alerts.",
            f"Total flagged exposure: {total_amount:.2f} ({curr_str}) across {len(relevant_txs)} transactions.",
            "Violations observed:",
        ]
        for v in evaluation_result.violations:
            narrative_lines.append(f" - [{v.severity.value}] {v.rule_name}: {v.details}")

        narrative = "\n".join(narrative_lines)

        if highest_severity in (AlertSeverity.CRITICAL, AlertSeverity.HIGH):
            recommended_action = "IMMEDIATE_FREEZE_AND_REGULATORY_DISCLOSURE"
        else:
            recommended_action = "ENHANCED_DUE_DILIGENCE_REVIEW"

        payload = f"{account_id}:{highest_severity.value}:{total_amount:.2f}:{sorted(list(matched_ids))}"
        audit_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        report_id = f"SAR-{account_id[:8]}-{audit_hash[:8]}"

        summary_text = (
            f"Automated AML Alert on account {account_id} with risk score "
            f"{evaluation_result.risk_score:.1f}/100"
        )

        report = SuspiciousActivityReport(
            report_id=report_id,
            account_id=account_id,
            severity=highest_severity,
            summary=summary_text,
            narrative=narrative,
            recommended_action=recommended_action,
            supporting_transaction_ids=sorted(list(matched_ids)),
            audit_hash=audit_hash,
        )

        self.verify_grounding(report, relevant_txs)
        return report

    def verify_grounding(
        self, report: SuspiciousActivityReport, raw_transactions: List[Transaction]
    ) -> bool:
        """Verify that all supporting transaction IDs exist in raw transactions."""
        raw_ids = {t.transaction_id for t in raw_transactions}
        for tid in report.supporting_transaction_ids:
            if tid not in raw_ids:
                raise ValueError(f"Grounding failure: Transaction ID {tid} not found in raw records.")
        return True
