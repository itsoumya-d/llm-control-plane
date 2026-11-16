"""Arta Finance B2B Wealth Platform Orchestrator & Partner Gateway Coordinator.

Designed for Arta Finance's B2B Wealth Platform (Singapore MAS CMS Licensed).
Provides:
1. Bidirectional Partner Ingress Gateway (ISO 20022 / REST schema normalization & SHA-256 idempotency).
2. Distributed Multi-Asset Transaction Settlement Coordinator with Temporal Saga compensation logic.
3. Agentic Wealth Advisor Guardrail & Tool-Calling Verification Engine under MAS FEAT principles.
4. Partner Ingress Telemetry & SLA Compliance Monitor (p99 tracking & error budget tracking).
"""

from __future__ import annotations

import enum
import hashlib
import hmac
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


class RiskProfile(str, enum.Enum):
    CONSERVATIVE = "CONSERVATIVE"
    BALANCED = "BALANCED"
    GROWTH = "GROWTH"
    AGGRESSIVE = "AGGRESSIVE"


class AssetClass(str, enum.Enum):
    PUBLIC_EQUITY = "PUBLIC_EQUITY"
    PRIVATE_EQUITY = "PRIVATE_EQUITY"
    VENTURE_CAPITAL = "VENTURE_CAPITAL"
    PRIVATE_CREDIT = "PRIVATE_CREDIT"
    REAL_ESTATE = "REAL_ESTATE"
    STRUCTURED_NOTE = "STRUCTURED_NOTE"
    CASH_EQUIVALENT = "CASH_EQUIVALENT"


class SagaStepStatus(str, enum.Enum):
    PENDING = "PENDING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    COMPENSATED = "COMPENSATED"


class PartnerProtocol(str, enum.Enum):
    ISO_20022 = "ISO_20022"
    REST_JSON = "REST_JSON"
    GRPC_PB = "GRPC_PB"


@dataclass
class CanonicalInvestmentOrder:
    order_id: str
    partner_id: str
    client_id: str
    asset_id: str
    asset_class: AssetClass
    amount: float
    currency: str
    idempotency_key: str
    risk_score: float
    is_accredited_investor: bool
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass
class PartnerGatewayMessage:
    raw_payload: Dict[str, Any]
    protocol: PartnerProtocol
    signature: str
    received_at: float = field(default_factory=time.time)


class PartnerGatewayTransformer:
    """Normalizes heterogeneous partner messages into Arta canonical models with strict security."""

    def __init__(self, partner_secret: str, max_single_order_limit: float = 10_000_000.0) -> None:
        self.partner_secret = partner_secret
        self.max_single_order_limit = max_single_order_limit
        self._processed_idempotency_keys: Dict[str, float] = {}

    def compute_signature(self, payload_str: str) -> str:
        return hmac.new(
            self.partner_secret.encode("utf-8"),
            payload_str.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def verify_signature(self, payload_str: str, signature: str) -> bool:
        expected = self.compute_signature(payload_str)
        return hmac.compare_digest(expected, signature)

    def compute_idempotency_key(self, partner_id: str, ext_tx_id: str, amount: float, currency: str) -> str:
        raw = f"{partner_id}:{ext_tx_id}:{amount:.2f}:{currency}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def normalize(self, message: PartnerGatewayMessage) -> CanonicalInvestmentOrder:
        payload = message.raw_payload
        partner_id = payload.get("partner_id") or payload.get("Fr", {}).get("OrgId", {}).get("Id", "UNKNOWN")
        ext_tx_id = payload.get("transaction_id") or payload.get("TxId") or payload.get("id")
        amount = float(payload.get("amount") or payload.get("Amt", {}).get("Value", 0.0))
        currency = str(payload.get("currency") or payload.get("Amt", {}).get("Ccy", "USD")).upper()
        asset_id = payload.get("asset_id") or payload.get("FinInstrmId", "ARTA_DEFAULT_FUND")

        raw_asset_class = str(payload.get("asset_class", "PUBLIC_EQUITY")).upper()
        try:
            asset_class = AssetClass(raw_asset_class)
        except ValueError:
            asset_class = AssetClass.PUBLIC_EQUITY

        client_id = payload.get("client_id") or payload.get("Dbtr", {}).get("Id", "CLIENT_001")
        risk_score = float(payload.get("risk_score", 50.0))
        is_accredited = bool(payload.get("is_accredited_investor", False))

        if amount <= 0.0:
            raise ValueError(f"Invalid order amount: {amount}. Must be strictly positive.")
        if amount > self.max_single_order_limit:
            raise ValueError(
                f"Order amount {amount} exceeds statutory single-order ceiling {self.max_single_order_limit}"
            )

        if asset_class in (AssetClass.PRIVATE_EQUITY, AssetClass.PRIVATE_CREDIT, AssetClass.VENTURE_CAPITAL):
            if not is_accredited:
                raise PermissionError(
                    "MAS Notice SFA 04-N12: Non-accredited investor restricted from private market tranches."
                )

        idem_key = self.compute_idempotency_key(partner_id, ext_tx_id, amount, currency)
        if idem_key in self._processed_idempotency_keys:
            raise ValueError(f"Duplicate transaction detected for idempotency key: {idem_key}")

        self._processed_idempotency_keys[idem_key] = time.time()

        order_id = f"ARTA-ORD-{int(time.time()*1000)}-{hashlib.md5(idem_key.encode()).hexdigest()[:6]}"
        return CanonicalInvestmentOrder(
            order_id=order_id,
            partner_id=partner_id,
            client_id=client_id,
            asset_id=asset_id,
            asset_class=asset_class,
            amount=amount,
            currency=currency,
            idempotency_key=idem_key,
            risk_score=risk_score,
            is_accredited_investor=is_accredited,
            metadata=payload.get("metadata", {}),
        )


@dataclass
class SagaStep:
    name: str
    execute_fn: Callable[[CanonicalInvestmentOrder], bool]
    compensate_fn: Callable[[CanonicalInvestmentOrder], bool]
    status: SagaStepStatus = SagaStepStatus.PENDING
    error: Optional[str] = None


class TemporalWealthSagaCoordinator:
    """Manages distributed multi-asset transactions with state machine compensation guarantees."""

    def __init__(self, order: CanonicalInvestmentOrder) -> None:
        self.order = order
        self.steps: List[SagaStep] = []
        self.completed_steps: List[SagaStep] = []
        self.is_success: bool = False
        self.compensated: bool = False

    def add_step(
        self,
        name: str,
        execute_fn: Callable[[CanonicalInvestmentOrder], bool],
        compensate_fn: Callable[[CanonicalInvestmentOrder], bool],
    ) -> TemporalWealthSagaCoordinator:
        self.steps.append(SagaStep(name=name, execute_fn=execute_fn, compensate_fn=compensate_fn))
        return self

    def execute(self) -> Tuple[bool, List[str]]:
        logs: List[str] = []
        for step in self.steps:
            try:
                ok = step.execute_fn(self.order)
                if ok:
                    step.status = SagaStepStatus.COMPLETED
                    self.completed_steps.append(step)
                    logs.append(f"Step '{step.name}' executed successfully.")
                else:
                    step.status = SagaStepStatus.FAILED
                    step.error = "Step returned failure"
                    logs.append(f"Step '{step.name}' failed. Initiating compensation saga.")
                    self._compensate(logs)
                    return False, logs
            except Exception as e:
                step.status = SagaStepStatus.FAILED
                step.error = str(e)
                logs.append(f"Step '{step.name}' raised exception: {e}. Initiating compensation saga.")
                self._compensate(logs)
                return False, logs

        self.is_success = True
        return True, logs

    def _compensate(self, logs: List[str]) -> None:
        self.compensated = True
        for step in reversed(self.completed_steps):
            try:
                comp_ok = step.compensate_fn(self.order)
                if comp_ok:
                    step.status = SagaStepStatus.COMPENSATED
                    logs.append(f"Compensation for step '{step.name}' completed.")
                else:
                    logs.append(f"Compensation for step '{step.name}' returned False.")
            except Exception as e:
                logs.append(f"Compensation error on step '{step.name}': {e}")


class AgenticAdvisorGuardrailEngine:
    """Enforces deterministic tool-calling rules and MAS FEAT risk boundaries for AI financial copilots."""

    ASSET_ALLOCATION_LIMITS: Dict[RiskProfile, Dict[AssetClass, float]] = {
        RiskProfile.CONSERVATIVE: {
            AssetClass.CASH_EQUIVALENT: 0.60,
            AssetClass.PUBLIC_EQUITY: 0.30,
            AssetClass.PRIVATE_CREDIT: 0.10,
            AssetClass.PRIVATE_EQUITY: 0.0,
            AssetClass.VENTURE_CAPITAL: 0.0,
            AssetClass.STRUCTURED_NOTE: 0.0,
        },
        RiskProfile.BALANCED: {
            AssetClass.CASH_EQUIVALENT: 0.20,
            AssetClass.PUBLIC_EQUITY: 0.50,
            AssetClass.PRIVATE_CREDIT: 0.20,
            AssetClass.PRIVATE_EQUITY: 0.10,
            AssetClass.VENTURE_CAPITAL: 0.0,
            AssetClass.STRUCTURED_NOTE: 0.0,
        },
        RiskProfile.GROWTH: {
            AssetClass.CASH_EQUIVALENT: 0.10,
            AssetClass.PUBLIC_EQUITY: 0.50,
            AssetClass.PRIVATE_CREDIT: 0.15,
            AssetClass.PRIVATE_EQUITY: 0.15,
            AssetClass.VENTURE_CAPITAL: 0.10,
            AssetClass.STRUCTURED_NOTE: 0.0,
        },
        RiskProfile.AGGRESSIVE: {
            AssetClass.CASH_EQUIVALENT: 0.05,
            AssetClass.PUBLIC_EQUITY: 0.40,
            AssetClass.PRIVATE_CREDIT: 0.15,
            AssetClass.PRIVATE_EQUITY: 0.20,
            AssetClass.VENTURE_CAPITAL: 0.15,
            AssetClass.STRUCTURED_NOTE: 0.05,
        },
    }

    def __init__(self, allowed_tools: Optional[List[str]] = None) -> None:
        self.allowed_tools = set(allowed_tools or [
            "portfolio_rebalancer",
            "risk_tolerance_evaluator",
            "tax_harvesting_scanner",
            "private_market_tranche_allocator",
            "liquidity_forecaster",
        ])

    def validate_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Tuple[bool, str]:
        if tool_name not in self.allowed_tools:
            return False, f"Unauthorized tool: '{tool_name}' not in allowed partner tool whitelist."
        if not isinstance(arguments, dict):
            return False, "Tool arguments must be a dictionary."
        return True, "Tool call validated."

    def validate_portfolio_rebalance(
        self,
        risk_profile: RiskProfile,
        proposed_allocation: Dict[AssetClass, float],
    ) -> Tuple[bool, List[str]]:
        errors: List[str] = []
        total = sum(proposed_allocation.values())
        if abs(total - 1.0) > 0.001:
            errors.append(f"Proposed allocation total {total:.4f} != 1.0 (100%).")

        limits = self.ASSET_ALLOCATION_LIMITS.get(risk_profile, {})
        for asset, weight in proposed_allocation.items():
            max_allowed = limits.get(asset, 0.0)
            if weight > max_allowed + 0.0001:
                errors.append(
                    f"Asset {asset.value} weight {weight:.2%} exceeds maximum allowed {max_allowed:.2%} "
                    f"for {risk_profile.value} profile."
                )

        return len(errors) == 0, errors


class PartnerTelemetryMonitor:
    """Tracks partner integration latency, error budgets, and SLA thresholds."""

    def __init__(self, partner_id: str, p99_sla_ms: float = 150.0) -> None:
        self.partner_id = partner_id
        self.p99_sla_ms = p99_sla_ms
        self.latencies_ms: List[float] = []
        self.errors: int = 0
        self.total_requests: int = 0

    def record_request(self, duration_ms: float, is_error: bool = False) -> None:
        self.latencies_ms.append(duration_ms)
        self.total_requests += 1
        if is_error:
            self.errors += 1

    def compute_percentile(self, percentile: float) -> float:
        if not self.latencies_ms:
            return 0.0
        sorted_lat = sorted(self.latencies_ms)
        idx = int(len(sorted_lat) * (percentile / 100.0))
        return sorted_lat[min(idx, len(sorted_lat) - 1)]

    def get_summary(self) -> Dict[str, Any]:
        p50 = self.compute_percentile(50.0)
        p95 = self.compute_percentile(95.0)
        p99 = self.compute_percentile(99.0)
        err_rate = (self.errors / self.total_requests) if self.total_requests > 0 else 0.0
        return {
            "partner_id": self.partner_id,
            "total_requests": self.total_requests,
            "errors": self.errors,
            "error_rate": err_rate,
            "p50_latency_ms": p50,
            "p95_latency_ms": p95,
            "p99_latency_ms": p99,
            "sla_breached": p99 > self.p99_sla_ms,
        }
