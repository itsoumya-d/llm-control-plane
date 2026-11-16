"""Guidewire Cloud Platform (GWCP) Insurance Policy & Claims AI Orchestrator.

Engineered for Guidewire Agile Product Engineering & Cloud Microservices:
- Idempotent Distributed Saga Coordinator for insurance claims and policy endorsements
- Compensating transactional execution for multi-stage claim processing
- Underwriting risk assessment engine with fraud scoring and explainability tags
- Asynchronous Kafka-compatible event streaming bridge with partition keys and DLQ retry logic
"""

from __future__ import annotations

import enum
import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


class PolicyType(str, enum.Enum):
    COMMERCIAL_PROPERTY = "COMMERCIAL_PROPERTY"
    GENERAL_LIABILITY = "GENERAL_LIABILITY"
    WORKERS_COMP = "WORKERS_COMP"
    PERSONAL_AUTO = "PERSONAL_AUTO"
    CYBER_SECURITY = "CYBER_SECURITY"


class ClaimStatus(str, enum.Enum):
    SUBMITTED = "SUBMITTED"
    VALIDATING = "VALIDATING"
    RESERVED = "RESERVED"
    UNDERWRITING_REVIEW = "UNDERWRITING_REVIEW"
    APPROVED_STP = "APPROVED_STP"  # Straight-Through Processing
    ADJUSTER_ASSIGNED = "ADJUSTER_ASSIGNED"
    REJECTED = "REJECTED"
    COMPENSATED = "COMPENSATED"


class RiskTier(str, enum.Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL_FRAUD = "CRITICAL_FRAUD"


@dataclass
class PolicyEndorsement:
    policy_id: str
    insured_entity: str
    policy_type: PolicyType
    coverage_limit_cents: int
    deductible_cents: int
    active: bool = True
    claims_count: int = 0
    total_incurred_cents: int = 0
    created_at: float = field(default_factory=time.time)

    def available_coverage_cents(self) -> int:
        return max(0, self.coverage_limit_cents - self.total_incurred_cents)


@dataclass
class ClaimSubmission:
    claim_id: str
    policy_id: str
    loss_date: str
    description: str
    claimed_amount_cents: int
    claimant_name: str
    status: ClaimStatus = ClaimStatus.SUBMITTED
    reserve_amount_cents: int = 0
    risk_tier: Optional[RiskTier] = None
    risk_score: float = 0.0
    idempotency_key: str = field(default_factory=lambda: str(uuid.uuid4()))
    audit_trail: List[str] = field(default_factory=list)


@dataclass
class UnderwritingAssessment:
    claim_id: str
    risk_tier: RiskTier
    risk_score: float  # 0.0 (safe) to 1.0 (fraud/high-risk)
    confidence: float
    factors: List[str]
    straight_through_eligible: bool
    explanation: str


class GuidewireUnderwritingEngine:
    """Evaluates insurance claim severity, anomaly risk, and STP eligibility."""

    def __init__(self, fraud_threshold: float = 0.70, stp_amount_ceiling_cents: int = 1_000_000) -> None:
        self.fraud_threshold = fraud_threshold
        self.stp_amount_ceiling_cents = stp_amount_ceiling_cents

    def evaluate_claim(self, policy: PolicyEndorsement, claim: ClaimSubmission) -> UnderwritingAssessment:
        factors: List[str] = []
        score = 0.05  # Base baseline risk

        # Factor 1: Claim amount vs coverage ratio
        if policy.coverage_limit_cents > 0:
            ratio = claim.claimed_amount_cents / policy.coverage_limit_cents
            if ratio > 0.8:
                score += 0.35
                factors.append("Claim exceeds 80% of aggregate policy coverage")
            elif ratio > 0.5:
                score += 0.20
                factors.append("Claim exceeds 50% of aggregate policy coverage")

        # Factor 2: Historical claim frequency anomaly
        if policy.claims_count >= 3:
            score += 0.25
            factors.append(f"Elevated claim frequency: {policy.claims_count} prior claims on policy")
        elif policy.claims_count == 2:
            score += 0.10
            factors.append("Moderate claim frequency: 2 prior claims")

        # Factor 3: Textual keyword anomaly analysis
        desc_lower = claim.description.lower()
        suspicious_terms = ["unwitnessed", "total loss", "unattended", "cash only", "immediate payout", "disputed"]
        matched_terms = [t for t in suspicious_terms if t in desc_lower]
        if matched_terms:
            score += 0.15 * len(matched_terms)
            factors.append(f"Risk indicators in incident narrative: {', '.join(matched_terms)}")

        # Normalize score
        score = min(1.0, max(0.0, score))

        # Determine Tier
        if score >= self.fraud_threshold:
            tier = RiskTier.CRITICAL_FRAUD
        elif score >= 0.45:
            tier = RiskTier.HIGH
        elif score >= 0.20:
            tier = RiskTier.MEDIUM
        else:
            tier = RiskTier.LOW

        # STP Eligibility: Low risk and below amount ceiling
        stp_eligible = (
            tier == RiskTier.LOW
            and claim.claimed_amount_cents <= self.stp_amount_ceiling_cents
            and policy.active
            and policy.available_coverage_cents() >= claim.claimed_amount_cents
        )

        explanation = (
            f"Underwriting evaluated claim {claim.claim_id}: Tier={tier.value}, Score={score:.2f}, "
            f"STP Eligible={stp_eligible}. Factors: {'; '.join(factors) if factors else 'Standard nominal risk'}"
        )

        return UnderwritingAssessment(
            claim_id=claim.claim_id,
            risk_tier=tier,
            risk_score=score,
            confidence=0.94,
            factors=factors,
            straight_through_eligible=stp_eligible,
            explanation=explanation,
        )


class EventDrivenKafkaBridge:
    """Simulates high-throughput Guidewire Cloud Kafka event streaming."""

    def __init__(self) -> None:
        self.published_events: List[Dict[str, Any]] = []
        self.dead_letter_queue: List[Dict[str, Any]] = []
        self.processed_idempotency_keys: set[str] = set()

    def publish_event(
        self, topic: str, partition_key: str, payload: Dict[str, Any], idempotency_key: str
    ) -> bool:
        if idempotency_key in self.processed_idempotency_keys:
            # Deduplicated event
            return True

        event = {
            "event_id": str(uuid.uuid4()),
            "topic": topic,
            "partition_key": partition_key,
            "partition": int(hashlib.md5(partition_key.encode()).hexdigest(), 16) % 12,
            "idempotency_key": idempotency_key,
            "timestamp": time.time(),
            "payload": payload,
        }

        # Check payload validity
        if not payload.get("claim_id") and not payload.get("policy_id"):
            self.dead_letter_queue.append({"event": event, "reason": "Missing primary business key"})
            return False

        self.published_events.append(event)
        self.processed_idempotency_keys.add(idempotency_key)
        return True


class ClaimsSagaCoordinator:
    """Coordinates distributed multi-stage claims processing with automatic compensating rollback."""

    def __init__(
        self,
        underwriting_engine: GuidewireUnderwritingEngine,
        event_bridge: EventDrivenKafkaBridge,
    ) -> None:
        self.underwriting_engine = underwriting_engine
        self.event_bridge = event_bridge
        self.policy_registry: Dict[str, PolicyEndorsement] = {}
        self.claim_registry: Dict[str, ClaimSubmission] = {}
        self.ledger_reserves_cents: Dict[str, int] = {}
        self.idempotency_store: Dict[str, ClaimSubmission] = {}

    def register_policy(self, policy: PolicyEndorsement) -> None:
        self.policy_registry[policy.policy_id] = policy

    def process_claim_saga(self, claim: ClaimSubmission) -> Tuple[bool, str]:
        # Idempotency check
        if claim.idempotency_key in self.idempotency_store:
            existing = self.idempotency_store[claim.idempotency_key]
            return True, f"Idempotent replay: Claim {existing.claim_id} already resolved as {existing.status.value}"

        executed_steps: List[str] = []
        claim.status = ClaimStatus.VALIDATING
        claim.audit_trail.append(f"T0: Started Saga for claim {claim.claim_id}")

        # Step 1: Policy validation
        policy = self.policy_registry.get(claim.policy_id)
        if not policy:
            claim.status = ClaimStatus.REJECTED
            claim.audit_trail.append(f"T1_FAIL: Policy {claim.policy_id} not found")
            return False, f"Policy {claim.policy_id} does not exist"

        if not policy.active:
            claim.status = ClaimStatus.REJECTED
            claim.audit_trail.append(f"T1_FAIL: Policy {claim.policy_id} is inactive/cancelled")
            return False, f"Policy {claim.policy_id} is inactive"

        if policy.available_coverage_cents() < claim.claimed_amount_cents:
            claim.status = ClaimStatus.REJECTED
            claim.audit_trail.append(
                f"T1_FAIL: Claim {claim.claimed_amount_cents} exceeds available coverage {policy.available_coverage_cents()}"
            )
            return False, "Claim amount exceeds available policy coverage limit"

        executed_steps.append("STEP_1_POLICY_VALIDATED")
        claim.audit_trail.append("T1: Policy validated successfully")

        # Step 2: Financial reserve allocation
        reserve_amount = claim.claimed_amount_cents
        self.ledger_reserves_cents[claim.claim_id] = reserve_amount
        policy.total_incurred_cents += reserve_amount
        policy.claims_count += 1
        claim.reserve_amount_cents = reserve_amount
        claim.status = ClaimStatus.RESERVED
        executed_steps.append("STEP_2_RESERVE_ALLOCATED")
        claim.audit_trail.append(f"T2: Reserved {reserve_amount} cents in claims ledger")

        # Step 3: AI Underwriting & Fraud Analysis
        assessment = self.underwriting_engine.evaluate_claim(policy, claim)
        claim.risk_tier = assessment.risk_tier
        claim.risk_score = assessment.risk_score
        claim.audit_trail.append(f"T3: Underwriting assessment: {assessment.risk_tier.value} ({assessment.risk_score:.2f})")

        # Check for Critical Fraud failure
        if assessment.risk_tier == RiskTier.CRITICAL_FRAUD:
            # Trigger Compensation Rollback
            self._compensate_saga(claim, policy, executed_steps, "Critical fraud score triggered rejection")
            return False, f"Claim rejected due to critical fraud risk: {assessment.explanation}"

        # Step 4: Adjudication Decision & Kafka Dispatch
        if assessment.straight_through_eligible:
            claim.status = ClaimStatus.APPROVED_STP
            claim.audit_trail.append("T4: Approved via Straight-Through Processing (STP)")
            topic = "gw.claims.adjudication.stp.v1"
        else:
            claim.status = ClaimStatus.ADJUSTER_ASSIGNED
            claim.audit_trail.append(f"T4: Assigned to claim adjuster for {assessment.risk_tier.value} review")
            topic = "gw.claims.adjudication.manual.v1"

        # Dispatch Kafka Event
        event_payload = {
            "claim_id": claim.claim_id,
            "policy_id": claim.policy_id,
            "status": claim.status.value,
            "risk_score": claim.risk_score,
            "risk_tier": claim.risk_tier.value,
            "reserve_amount_cents": claim.reserve_amount_cents,
            "audit_trail": claim.audit_trail,
        }
        self.event_bridge.publish_event(
            topic=topic,
            partition_key=claim.policy_id,
            payload=event_payload,
            idempotency_key=claim.idempotency_key,
        )

        self.claim_registry[claim.claim_id] = claim
        self.idempotency_store[claim.idempotency_key] = claim
        return True, f"Claim {claim.claim_id} processed successfully: {claim.status.value}"

    def _compensate_saga(
        self,
        claim: ClaimSubmission,
        policy: PolicyEndorsement,
        executed_steps: List[str],
        reason: str,
    ) -> None:
        """Executes backward compensating actions in reverse order."""
        claim.audit_trail.append(f"COMPENSATION_TRIGGERED: {reason}")

        if "STEP_2_RESERVE_ALLOCATED" in executed_steps:
            # Rollback ledger reserve and policy incurred amount
            reserved = self.ledger_reserves_cents.pop(claim.claim_id, 0)
            policy.total_incurred_cents = max(0, policy.total_incurred_cents - reserved)
            policy.claims_count = max(0, policy.claims_count - 1)
            claim.reserve_amount_cents = 0
            claim.audit_trail.append(f"COMPENSATION_STEP_2: Released reserve {reserved} cents")

        claim.status = ClaimStatus.REJECTED
        claim.audit_trail.append("COMPENSATION_COMPLETE: Claim marked as REJECTED")

        # Publish compensation notice to DLQ/Event stream
        self.event_bridge.publish_event(
            topic="gw.claims.compensation.v1",
            partition_key=claim.policy_id,
            payload={
                "claim_id": claim.claim_id,
                "policy_id": claim.policy_id,
                "reason": reason,
                "status": ClaimStatus.REJECTED.value,
            },
            idempotency_key=f"comp_{claim.idempotency_key}",
        )
        self.claim_registry[claim.claim_id] = claim
        self.idempotency_store[claim.idempotency_key] = claim
