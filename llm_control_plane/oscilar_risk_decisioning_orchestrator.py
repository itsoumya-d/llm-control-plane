"""Oscilar AI Risk Decisioning Platform Orchestrator & Stream Coordinator.

Designed for Oscilar's stream-native AI Risk Decisioning architecture (KFC: Kafka, Flink, ClickHouse).
Provides:
1. Real-time Sliding-Window Velocity & Event Aggregation Engine (Flink CEP model).
2. Multi-Tier AI Risk Scoring & Rule Evaluation Engine (fraud, credit, AML).
3. Distributed Temporal Saga Coordinator for Multi-Step Risk Verification & Compensations.
4. ClickHouse OLAP Columnar Export Layer for Historical Feature Storage & Rule Backtesting.
5. Dynamic Feature Flag & Rule Weighting Manager (LaunchDarkly pattern).
"""

from __future__ import annotations

import enum
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


class DecisionOutcome(str, enum.Enum):
    APPROVE = "APPROVE"
    STEP_UP_AUTH = "STEP_UP_AUTH"
    MANUAL_REVIEW = "MANUAL_REVIEW"
    DECLINE = "DECLINE"


class RiskSeverity(str, enum.Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class SagaStepStatus(str, enum.Enum):
    PENDING = "PENDING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    COMPENSATED = "COMPENSATED"


@dataclass
class TransactionEvent:
    event_id: str
    account_id: str
    amount: float
    currency: str
    timestamp: float
    ip_address: str
    device_fingerprint: str
    merchant_category: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def compute_event_hash(self) -> str:
        payload = f"{self.event_id}:{self.account_id}:{self.amount}:{self.currency}:{self.timestamp}:{self.ip_address}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class VelocityMetrics:
    account_id: str
    window_seconds: float
    event_count: int
    total_amount: float
    unique_ips: int
    unique_devices: int
    max_single_amount: float
    calculated_at: float


class SlidingWindowVelocityEngine:
    """Stateful in-memory sliding window aggregator modeling Apache Flink CEP state."""

    def __init__(self, default_window_seconds: float = 300.0) -> None:
        self.default_window_seconds = default_window_seconds
        self._history: Dict[str, List[TransactionEvent]] = {}

    def record_event(self, event: TransactionEvent) -> None:
        if event.account_id not in self._history:
            self._history[event.account_id] = []
        self._history[event.account_id].append(event)
        self.prune(event.account_id, current_time=event.timestamp)

    def prune(self, account_id: str, current_time: Optional[float] = None) -> None:
        if account_id not in self._history:
            return
        now = current_time if current_time is not None else time.time()
        cutoff = now - (self.default_window_seconds * 2.0)
        self._history[account_id] = [e for e in self._history[account_id] if e.timestamp >= cutoff]

    def get_velocity(
        self, account_id: str, window_seconds: Optional[float] = None, current_time: Optional[float] = None
    ) -> VelocityMetrics:
        win = window_seconds if window_seconds is not None else self.default_window_seconds
        now = current_time if current_time is not None else time.time()
        cutoff = now - win

        events = self._history.get(account_id, [])
        active = [e for e in events if cutoff <= e.timestamp <= now]

        count = len(active)
        total_amt = sum(e.amount for e in active)
        ips = len({e.ip_address for e in active})
        devs = len({e.device_fingerprint for e in active})
        max_amt = max((e.amount for e in active), default=0.0)

        return VelocityMetrics(
            account_id=account_id,
            window_seconds=win,
            event_count=count,
            total_amount=round(total_amt, 2),
            unique_ips=ips,
            unique_devices=devs,
            max_single_amount=round(max_amt, 2),
            calculated_at=now,
        )


@dataclass
class RuleEvaluationResult:
    rule_name: str
    passed: bool
    score_delta: float
    severity: RiskSeverity
    reason: str


class RiskRuleEvaluator:
    """Evaluates fraud and compliance rules with configurable weights and thresholds."""

    def __init__(
        self,
        max_events_per_minute: int = 5,
        max_spend_per_5min: float = 10000.0,
        max_devices_per_window: int = 3,
        high_risk_mccs: Optional[Set[str]] = None,
    ) -> None:
        self.max_events_per_minute = max_events_per_minute
        self.max_spend_per_5min = max_spend_per_5min
        self.max_devices_per_window = max_devices_per_window
        self.high_risk_mccs = high_risk_mccs or {"6051", "7995", "4829"}  # Crypto, Gambling, Wire transfers

    def evaluate_rules(self, event: TransactionEvent, velocity: VelocityMetrics) -> List[RuleEvaluationResult]:
        results: List[RuleEvaluationResult] = []

        # Rule 1: High Velocity Transaction Spikes
        if velocity.event_count > self.max_events_per_minute:
            results.append(
                RuleEvaluationResult(
                    rule_name="VELOCITY_SPIKE",
                    passed=False,
                    score_delta=35.0,
                    severity=RiskSeverity.HIGH,
                    reason=(
                        f"Account {event.account_id} executed {velocity.event_count} events in "
                        f"{velocity.window_seconds}s (limit {self.max_events_per_minute})"
                    ),
                )
            )
        else:
            results.append(
                RuleEvaluationResult(
                    rule_name="VELOCITY_SPIKE",
                    passed=True,
                    score_delta=0.0,
                    severity=RiskSeverity.LOW,
                    reason="Transaction frequency within acceptable baseline",
                )
            )

        # Rule 2: Sudden Spend Inrush
        if velocity.total_amount > self.max_spend_per_5min:
            results.append(
                RuleEvaluationResult(
                    rule_name="RAPID_SPEND_INRUSH",
                    passed=False,
                    score_delta=40.0,
                    severity=RiskSeverity.HIGH,
                    reason=(
                        f"Cumulative spend {velocity.total_amount} {event.currency} exceeded "
                        f"window ceiling {self.max_spend_per_5min}"
                    ),
                )
            )
        else:
            results.append(
                RuleEvaluationResult(
                    rule_name="RAPID_SPEND_INRUSH",
                    passed=True,
                    score_delta=0.0,
                    severity=RiskSeverity.LOW,
                    reason="Cumulative window spend within limit",
                )
            )

        # Rule 3: Device / Session Hopping
        if velocity.unique_devices > self.max_devices_per_window:
            results.append(
                RuleEvaluationResult(
                    rule_name="DEVICE_HOPPING_DETECTED",
                    passed=False,
                    score_delta=30.0,
                    severity=RiskSeverity.HIGH,
                    reason=f"Detected {velocity.unique_devices} distinct devices within sliding window",
                )
            )
        else:
            results.append(
                RuleEvaluationResult(
                    rule_name="DEVICE_HOPPING_DETECTED",
                    passed=True,
                    score_delta=0.0,
                    severity=RiskSeverity.LOW,
                    reason="Single or consistent device profile observed",
                )
            )

        # Rule 4: High-Risk Merchant Category Code (MCC)
        if event.merchant_category in self.high_risk_mccs:
            results.append(
                RuleEvaluationResult(
                    rule_name="HIGH_RISK_MCC",
                    passed=False,
                    score_delta=25.0,
                    severity=RiskSeverity.MEDIUM,
                    reason=f"Transaction directed to high-risk merchant category code {event.merchant_category}",
                )
            )
        else:
            results.append(
                RuleEvaluationResult(
                    rule_name="HIGH_RISK_MCC",
                    passed=True,
                    score_delta=0.0,
                    severity=RiskSeverity.LOW,
                    reason="Standard merchant category code",
                )
            )

        return results


@dataclass
class DecisionRecord:
    event_id: str
    account_id: str
    composite_risk_score: float
    outcome: DecisionOutcome
    severity: RiskSeverity
    triggered_rules: List[str]
    latency_ms: float
    evaluated_at: float
    features: Dict[str, Any]


class RiskDecisionEngine:
    """Core decisioning orchestrator integrating velocity, rules, and decision thresholds."""

    def __init__(
        self,
        velocity_engine: Optional[SlidingWindowVelocityEngine] = None,
        rule_evaluator: Optional[RiskRuleEvaluator] = None,
        step_up_threshold: float = 30.0,
        manual_review_threshold: float = 60.0,
        decline_threshold: float = 80.0,
    ) -> None:
        self.velocity_engine = velocity_engine or SlidingWindowVelocityEngine()
        self.rule_evaluator = rule_evaluator or RiskRuleEvaluator()
        self.step_up_threshold = step_up_threshold
        self.manual_review_threshold = manual_review_threshold
        self.decline_threshold = decline_threshold

    def evaluate_transaction(self, event: TransactionEvent) -> DecisionRecord:
        t0 = time.perf_counter()

        # 1. Update & query velocity
        self.velocity_engine.record_event(event)
        velocity = self.velocity_engine.get_velocity(event.account_id, current_time=event.timestamp)

        # 2. Evaluate rule heuristics
        rule_results = self.rule_evaluator.evaluate_rules(event, velocity)

        # 3. Calculate composite score
        total_score = min(100.0, sum(r.score_delta for r in rule_results))
        triggered = [r.rule_name for r in rule_results if not r.passed]

        # 4. Resolve outcome
        if total_score >= self.decline_threshold:
            outcome = DecisionOutcome.DECLINE
            sev = RiskSeverity.CRITICAL
        elif total_score >= self.manual_review_threshold:
            outcome = DecisionOutcome.MANUAL_REVIEW
            sev = RiskSeverity.HIGH
        elif total_score >= self.step_up_threshold:
            outcome = DecisionOutcome.STEP_UP_AUTH
            sev = RiskSeverity.MEDIUM
        else:
            outcome = DecisionOutcome.APPROVE
            sev = RiskSeverity.LOW

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return DecisionRecord(
            event_id=event.event_id,
            account_id=event.account_id,
            composite_risk_score=round(total_score, 1),
            outcome=outcome,
            severity=sev,
            triggered_rules=triggered,
            latency_ms=round(latency_ms, 3),
            evaluated_at=event.timestamp,
            features={
                "event_count_window": velocity.event_count,
                "total_amount_window": velocity.total_amount,
                "unique_ips_window": velocity.unique_ips,
                "unique_devices_window": velocity.unique_devices,
                "currency": event.currency,
                "amount": event.amount,
                "mcc": event.merchant_category,
            },
        )


@dataclass
class SagaStep:
    step_id: str
    name: str
    status: SagaStepStatus
    action: Callable[[], bool]
    compensate: Callable[[], bool]
    error_message: Optional[str] = None


class TemporalRiskSagaCoordinator:
    """Simulates Temporal saga orchestration for multi-step async risk verification workflows."""

    def __init__(self, saga_id: str) -> None:
        self.saga_id = saga_id
        self.steps: List[SagaStep] = []
        self.state: SagaStepStatus = SagaStepStatus.PENDING

    def add_step(self, step_id: str, name: str, action: Callable[[], bool], compensate: Callable[[], bool]) -> None:
        self.steps.append(
            SagaStep(
                step_id=step_id,
                name=name,
                status=SagaStepStatus.PENDING,
                action=action,
                compensate=compensate,
            )
        )

    def execute(self) -> Tuple[bool, List[str]]:
        executed_steps: List[SagaStep] = []
        logs: List[str] = []

        for step in self.steps:
            try:
                success = step.action()
                if success:
                    step.status = SagaStepStatus.COMPLETED
                    executed_steps.append(step)
                    logs.append(f"Step '{step.name}' ({step.step_id}) completed successfully.")
                else:
                    step.status = SagaStepStatus.FAILED
                    step.error_message = "Action returned false"
                    logs.append(f"Step '{step.name}' ({step.step_id}) failed. Commencing compensation saga.")
                    self._compensate(executed_steps, logs)
                    self.state = SagaStepStatus.COMPENSATED
                    return False, logs
            except Exception as ex:
                step.status = SagaStepStatus.FAILED
                step.error_message = str(ex)
                logs.append(f"Step '{step.name}' ({step.step_id}) threw exception: {ex}. Compensating.")
                self._compensate(executed_steps, logs)
                self.state = SagaStepStatus.COMPENSATED
                return False, logs

        self.state = SagaStepStatus.COMPLETED
        return True, logs

    def _compensate(self, executed_steps: List[SagaStep], logs: List[str]) -> None:
        for step in reversed(executed_steps):
            try:
                comp_ok = step.compensate()
                step.status = SagaStepStatus.COMPENSATED if comp_ok else SagaStepStatus.FAILED
                c_status = "SUCCESS" if comp_ok else "FAIL"
                logs.append(f"Compensated step '{step.name}' ({step.step_id}): status={c_status}")
            except Exception as ex:
                step.status = SagaStepStatus.FAILED
                logs.append(f"Failed to compensate step '{step.name}': {ex}")


class ClickHouseOLAPExporter:
    """Formats decision records into ClickHouse columnar format for backtesting & analytics."""

    @staticmethod
    def to_clickhouse_row(record: DecisionRecord) -> Dict[str, Any]:
        eval_dt = time.gmtime(record.evaluated_at)
        partition_month = f"{eval_dt.tm_year}{eval_dt.tm_mon:02d}"

        return {
            "partition_month": int(partition_month),
            "event_id": record.event_id,
            "account_id": record.account_id,
            "risk_score": record.composite_risk_score,
            "decision": record.outcome.value,
            "severity": record.severity.value,
            "triggered_rules": record.triggered_rules,
            "latency_ms": record.latency_ms,
            "event_timestamp": int(record.evaluated_at),
            "feature_event_count": record.features.get("event_count_window", 0),
            "feature_total_amount": record.features.get("total_amount_window", 0.0),
            "feature_unique_ips": record.features.get("unique_ips_window", 0),
            "feature_unique_devices": record.features.get("unique_devices_window", 0),
            "currency": record.features.get("currency", "USD"),
            "amount": record.features.get("amount", 0.0),
            "mcc": record.features.get("mcc", ""),
        }
