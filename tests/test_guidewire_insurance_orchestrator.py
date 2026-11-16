"""Unit and Integration Tests for Guidewire Insurance Orchestrator.

Tests:
- Policy endorsements and coverage limit checks
- AI underwriting risk evaluation and fraud score weighting
- Multi-stage Saga execution (Happy Path STP & Manual Adjuster Assignment)
- Backward compensating transactions on critical fraud detection
- Idempotent execution and duplicate request replays
- Kafka event bus publishing, partition key hashing, and DLQ handling
"""

import pytest

from llm_control_plane.guidewire_insurance_orchestrator import (
    ClaimsSagaCoordinator,
    ClaimStatus,
    ClaimSubmission,
    EventDrivenKafkaBridge,
    GuidewireUnderwritingEngine,
    PolicyEndorsement,
    PolicyType,
    RiskTier,
)


@pytest.fixture
def underwriting_engine():
    return GuidewireUnderwritingEngine(fraud_threshold=0.70, stp_amount_ceiling_cents=500_000)


@pytest.fixture
def event_bridge():
    return EventDrivenKafkaBridge()


@pytest.fixture
def coordinator(underwriting_engine, event_bridge):
    coord = ClaimsSagaCoordinator(underwriting_engine=underwriting_engine, event_bridge=event_bridge)
    # Register standard commercial policy
    policy = PolicyEndorsement(
        policy_id="POL-GW-IRE-1001",
        insured_entity="Dublin Tech Logistics Ltd",
        policy_type=PolicyType.COMMERCIAL_PROPERTY,
        coverage_limit_cents=10_000_000,  # €100,000
        deductible_cents=50_000,          # €500
        active=True,
    )
    coord.register_policy(policy)
    return coord


def test_policy_endorsement_coverage_calculation():
    policy = PolicyEndorsement(
        policy_id="POL-TEST-1",
        insured_entity="Acme Corp",
        policy_type=PolicyType.GENERAL_LIABILITY,
        coverage_limit_cents=5_000_000,
        deductible_cents=10_000,
        total_incurred_cents=1_500_000,
    )
    assert policy.available_coverage_cents() == 3_500_000


def test_underwriting_low_risk_stp_eligibility(underwriting_engine):
    policy = PolicyEndorsement(
        policy_id="POL-STP-1",
        insured_entity="Irish Retail Services",
        policy_type=PolicyType.COMMERCIAL_PROPERTY,
        coverage_limit_cents=20_000_000,
        deductible_cents=20_000,
    )
    claim = ClaimSubmission(
        claim_id="CLM-101",
        policy_id="POL-STP-1",
        loss_date="2026-11-15",
        description="Minor water pipe leak in staff kitchen, verified by plumber invoice.",
        claimed_amount_cents=150_000,  # €1,500
        claimant_name="Sean O'Connor",
    )
    assessment = underwriting_engine.evaluate_claim(policy, claim)
    assert assessment.risk_tier == RiskTier.LOW
    assert assessment.straight_through_eligible is True
    assert assessment.risk_score < 0.20


def test_underwriting_critical_fraud_detection(underwriting_engine):
    policy = PolicyEndorsement(
        policy_id="POL-FRAUD-1",
        insured_entity="Shady Operations Ltd",
        policy_type=PolicyType.COMMERCIAL_PROPERTY,
        coverage_limit_cents=5_000_000,
        deductible_cents=50_000,
        claims_count=4,  # High prior claims
    )
    claim = ClaimSubmission(
        claim_id="CLM-FRAUD-1",
        policy_id="POL-FRAUD-1",
        loss_date="2026-11-15",
        description=(
            "Unwitnessed total loss of high-value inventory, unattended premises, cash only payout requested."
        ),
        claimed_amount_cents=4_500_000,  # 90% of coverage
        claimant_name="Suspicious Actor",
    )
    assessment = underwriting_engine.evaluate_claim(policy, claim)
    assert assessment.risk_tier == RiskTier.CRITICAL_FRAUD
    assert assessment.straight_through_eligible is False
    assert assessment.risk_score >= 0.70
    assert len(assessment.factors) >= 3


def test_claims_saga_happy_path_stp(coordinator, event_bridge):
    claim = ClaimSubmission(
        claim_id="CLM-STP-200",
        policy_id="POL-GW-IRE-1001",
        loss_date="2026-11-14",
        description="Cracked office storefront window due to windstorm.",
        claimed_amount_cents=200_000,  # €2,000 (well within STP ceiling)
        claimant_name="Mary Murphy",
    )
    success, msg = coordinator.process_claim_saga(claim)
    assert success is True
    assert "APPROVED_STP" in msg
    assert claim.status == ClaimStatus.APPROVED_STP
    assert claim.reserve_amount_cents == 200_000
    assert len(event_bridge.published_events) == 1
    assert event_bridge.published_events[0]["topic"] == "gw.claims.adjudication.stp.v1"


def test_claims_saga_adjuster_assigned_high_value(coordinator, event_bridge):
    claim = ClaimSubmission(
        claim_id="CLM-ADJ-300",
        policy_id="POL-GW-IRE-1001",
        loss_date="2026-11-12",
        description="Warehouse roof structural storm damage requiring engineering survey.",
        claimed_amount_cents=3_000_000,  # €30,000 (exceeds STP €5,000 ceiling)
        claimant_name="Liam Kelly",
    )
    success, msg = coordinator.process_claim_saga(claim)
    assert success is True
    assert "ADJUSTER_ASSIGNED" in msg
    assert claim.status == ClaimStatus.ADJUSTER_ASSIGNED
    assert claim.reserve_amount_cents == 3_000_000
    assert len(event_bridge.published_events) == 1
    assert event_bridge.published_events[0]["topic"] == "gw.claims.adjudication.manual.v1"


def test_claims_saga_compensating_rollback_on_fraud(coordinator, event_bridge):
    policy = coordinator.policy_registry["POL-GW-IRE-1001"]
    policy.claims_count = 3  # Increase prior count to elevate fraud score

    initial_incurred = policy.total_incurred_cents
    claim = ClaimSubmission(
        claim_id="CLM-FRAUD-ROLLBACK",
        policy_id="POL-GW-IRE-1001",
        loss_date="2026-11-16",
        description="Unwitnessed total loss unattended equipment disputed immediate payout.",
        claimed_amount_cents=9_000_000,  # 90% of coverage limit
        claimant_name="Fraud Syndicate",
    )
    success, msg = coordinator.process_claim_saga(claim)
    assert success is False
    assert "fraud" in msg.lower()
    assert claim.status == ClaimStatus.REJECTED
    # Verify reserves and policy counts rolled back cleanly
    assert coordinator.ledger_reserves_cents.get(claim.claim_id) is None
    assert policy.total_incurred_cents == initial_incurred
    assert any("COMPENSATION_TRIGGERED" in entry for entry in claim.audit_trail)
    assert any(ev["topic"] == "gw.claims.compensation.v1" for ev in event_bridge.published_events)


def test_claims_saga_rejects_missing_policy(coordinator):
    claim = ClaimSubmission(
        claim_id="CLM-NO-POL",
        policy_id="POL-NONEXISTENT",
        loss_date="2026-11-15",
        description="Test damage",
        claimed_amount_cents=100_000,
        claimant_name="John Doe",
    )
    success, msg = coordinator.process_claim_saga(claim)
    assert success is False
    assert "does not exist" in msg
    assert claim.status == ClaimStatus.REJECTED


def test_claims_saga_rejects_inactive_policy(coordinator):
    inactive_policy = PolicyEndorsement(
        policy_id="POL-CANCELLED",
        insured_entity="Defunct Bar Ltd",
        policy_type=PolicyType.GENERAL_LIABILITY,
        coverage_limit_cents=1_000_000,
        deductible_cents=10_000,
        active=False,
    )
    coordinator.register_policy(inactive_policy)

    claim = ClaimSubmission(
        claim_id="CLM-INACTIVE",
        policy_id="POL-CANCELLED",
        loss_date="2026-11-15",
        description="Slip and fall claim",
        claimed_amount_cents=200_000,
        claimant_name="Jane Doe",
    )
    success, msg = coordinator.process_claim_saga(claim)
    assert success is False
    assert "inactive" in msg


def test_claims_saga_rejects_exceeding_coverage(coordinator):
    claim = ClaimSubmission(
        claim_id="CLM-OVER-LIMIT",
        policy_id="POL-GW-IRE-1001",
        loss_date="2026-11-15",
        description="Massive catastrophe claim",
        claimed_amount_cents=15_000_000,  # €150,000 exceeds €100,000 limit
        claimant_name="Huge Claim",
    )
    success, msg = coordinator.process_claim_saga(claim)
    assert success is False
    assert "exceeds" in msg.lower()


def test_idempotent_saga_replay(coordinator):
    claim = ClaimSubmission(
        claim_id="CLM-IDEMPOTENT-1",
        policy_id="POL-GW-IRE-1001",
        loss_date="2026-11-15",
        description="Lightning damage to server hardware",
        claimed_amount_cents=150_000,
        claimant_name="Pat Higgins",
        idempotency_key="guidewire-idem-unique-key-999",
    )
    # First execution
    success_1, msg_1 = coordinator.process_claim_saga(claim)
    assert success_1 is True

    # Replay with same idempotency key
    success_2, msg_2 = coordinator.process_claim_saga(claim)
    assert success_2 is True
    assert "Idempotent replay" in msg_2


def test_kafka_event_bridge_deduplication_and_dlq(event_bridge):
    # Valid publish
    res_1 = event_bridge.publish_event(
        topic="gw.events.test",
        partition_key="POL-123",
        payload={"claim_id": "CLM-1", "policy_id": "POL-123"},
        idempotency_key="key-abc-1",
    )
    assert res_1 is True
    assert len(event_bridge.published_events) == 1

    # Duplicate publish with same key
    res_2 = event_bridge.publish_event(
        topic="gw.events.test",
        partition_key="POL-123",
        payload={"claim_id": "CLM-1", "policy_id": "POL-123"},
        idempotency_key="key-abc-1",
    )
    assert res_2 is True
    assert len(event_bridge.published_events) == 1  # Not duplicated

    # Invalid payload missing business keys -> DLQ
    res_3 = event_bridge.publish_event(
        topic="gw.events.test",
        partition_key="POL-INVALID",
        payload={"dummy": "corrupt_data"},
        idempotency_key="key-invalid-99",
    )
    assert res_3 is False
    assert len(event_bridge.dead_letter_queue) == 1
