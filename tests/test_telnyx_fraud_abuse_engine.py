"""Tests for Telnyx Inline Fraud & Abuse Decision Engine."""

from __future__ import annotations

import pytest

from llm_control_plane.telnyx_fraud_abuse_engine import (
    AbuseArchetype,
    AdversarialTextNormalizer,
    AITTrafficWatchdog,
    DecisionAction,
    FailMode,
    HumanInTheLoopReplayHarness,
    LLMJudgeVerdict,
    ShadowModeEvaluator,
    TelnyxFraudDecisionEngine,
)


def test_adversarial_normalizer_zero_width_and_homoglyphs() -> None:
    """Verify stripping of zero-width characters, homoglyphs, and leetspeak."""
    normalizer = AdversarialTextNormalizer()

    # Text containing zero-width spaces (\u200B), Cyrillic 'а' (\u0430), and leetspeak '@'
    # "P[zw]а[zw]yp[zw]@l" -> "paypal"
    adversarial_text = "P\u200B\u0430\u200Cyp\u200D@l Security Alert: Verify now at https://bit.ly/secure-login"
    result = normalizer.normalize(adversarial_text)

    assert "paypal" in result.normalized_text
    assert result.zero_width_chars_removed >= 3
    assert result.homoglyphs_replaced >= 1
    assert result.leetspeak_substitutions >= 1
    assert "https://bit.ly/secure-login" in result.extracted_urls
    assert result.evasion_risk_score > 0.05


def test_adversarial_normalizer_empty() -> None:
    """Verify empty input handling."""
    normalizer = AdversarialTextNormalizer()
    res = normalizer.normalize("")
    assert res.normalized_text == ""
    assert res.evasion_risk_score == 0.0
    assert len(res.extracted_urls) == 0


def test_ait_traffic_watchdog_entropy_detection() -> None:
    """Verify entropy and volume based Artificially Inflated Traffic (AIT) detection."""
    watchdog = AITTrafficWatchdog(velocity_threshold=20, entropy_min=1.0)
    entity = "entity_fraud_bot_01"

    # Send 25 messages to identical prefix sequential numbers with single template
    for i in range(25):
        recipient = f"+3538711100{i:02d}"
        is_ait, risk, reason = watchdog.evaluate_ait(
            entity_id=entity,
            recipient=recipient,
            normalized_template="your verification code is 123456",
        )

    assert is_ait is True
    assert risk > 0.6
    assert "AIT detected" in reason

    # Legitimate entity with diverse recipients
    legit_entity = "entity_legit_corp"
    for i in range(15):
        recipient = f"+3538{i:02d}999{i*7:03d}"
        is_ait, _, _ = watchdog.evaluate_ait(
            entity_id=legit_entity,
            recipient=recipient,
            normalized_template=f"unique transactional receipt id {i}",
        )
    assert is_ait is False


def test_inline_decision_engine_brand_impersonation() -> None:
    """Verify detection and quarantine/block for brand impersonation."""
    engine = TelnyxFraudDecisionEngine()

    # Message claiming to be Revolut from suspicious alphanumeric sender 'VerifyNow'
    msg = "R\u0435v\u043Elut Alert: Unauthorized transaction detected. Log in to cancel: http://revolut-cancel.cc"
    decision = engine.evaluate_message(
        message=msg,
        sender_id="VerifyNow",
        recipient="+353871234567",
        entity_id="cust_test_123",
    )

    assert decision.archetype == AbuseArchetype.BRAND_IMPERSONATION
    assert decision.action in (DecisionAction.BLOCK, DecisionAction.QUARANTINE_REVIEW)
    assert decision.confidence >= 0.8
    assert decision.latency_ms < 50.0  # Sub-50ms SLA budget
    assert decision.evasion_score > 0.0
    assert len(engine.audit_ledger) == 1


def test_inline_decision_engine_phishing_triggers() -> None:
    """Verify smishing call-to-action trigger detection."""
    engine = TelnyxFraudDecisionEngine()

    msg = "Immediate action required: Your account suspended due to billing failure. Click here to verify your identity: http://bit.ly/fix-billing"
    decision = engine.evaluate_message(
        message=msg,
        sender_id="Alert",
        recipient="+353879876543",
        entity_id="cust_test_456",
    )

    assert decision.archetype == AbuseArchetype.PHISHING_SMISHING
    assert decision.action == DecisionAction.BLOCK
    assert decision.confidence >= 0.85
    assert any("shortener" in r.lower() or "phishing" in r.lower() for r in decision.reasons)


def test_inline_decision_engine_legitimate_otp() -> None:
    """Verify standard legitimate transactional traffic passes cleanly."""
    engine = TelnyxFraudDecisionEngine()

    msg = "Your Dublin City Library reservation for book 'Designing Data-Intensive Applications' is ready for collection at Ilac Centre."
    decision = engine.evaluate_message(
        message=msg,
        sender_id="DubLibrary",
        recipient="+353875551212",
        entity_id="cust_legit_library",
    )

    assert decision.archetype == AbuseArchetype.LEGITIMATE
    assert decision.action == DecisionAction.ALLOW
    assert decision.confidence >= 0.90


def test_fail_open_and_fail_closed_resilience() -> None:
    """Verify graceful degradation under simulated pipeline errors."""
    def broken_judge(text: str, sender: str) -> LLMJudgeVerdict:
        raise TimeoutError("Downstream inference service timed out")

    engine = TelnyxFraudDecisionEngine(llm_judge_fn=broken_judge)

    # Fail-Open mode (typical for mission-critical transactional paths)
    res_open = engine.evaluate_message(
        message="Your package will arrive at 3 PM",
        sender_id="Courier",
        recipient="+353870001122",
        entity_id="cust_courier",
        fail_mode=FailMode.FAIL_OPEN,
    )
    assert res_open.action == DecisionAction.ALLOW
    assert any("Fallback mode FAIL_OPEN applied" in r for r in res_open.reasons)

    # Fail-Closed mode (for high-risk unverified routes)
    res_closed = engine.evaluate_message(
        message="Your package will arrive at 3 PM",
        sender_id="Courier",
        recipient="+353870001122",
        entity_id="cust_courier",
        fail_mode=FailMode.FAIL_CLOSED,
    )
    assert res_closed.action == DecisionAction.QUARANTINE_REVIEW
    assert any("Fallback mode FAIL_CLOSED applied" in r for r in res_closed.reasons)


def test_shadow_mode_evaluator_metrics() -> None:
    """Verify holdout evaluation precision, recall, F1, and Brier calibration."""
    engine = TelnyxFraudDecisionEngine()
    evaluator = ShadowModeEvaluator(engine)

    corpus = [
        {
            "message": "AIB Alert: Unauthorized debit of €450 detected. Cancel at https://aib-cancel.cc",
            "sender_id": "BankAlert",
            "recipient": "+353871112233",
            "true_archetype": AbuseArchetype.BRAND_IMPERSONATION,
            "is_fraud": True,
        },
        {
            "message": "Account suspended: Verify your identity immediately or card deactivated: http://bit.ly/sec-acc",
            "sender_id": "Security",
            "recipient": "+353872223344",
            "true_archetype": AbuseArchetype.PHISHING_SMISHING,
            "is_fraud": True,
        },
        {
            "message": "Your table for 2 at Chapter One Dublin is confirmed for tonight at 19:30.",
            "sender_id": "ChapterOne",
            "recipient": "+353873334455",
            "true_archetype": AbuseArchetype.LEGITIMATE,
            "is_fraud": False,
        },
        {
            "message": "Dublin Bus real-time: Route 46A arriving in 4 mins at Stop 758.",
            "sender_id": "TFI",
            "recipient": "+353874445566",
            "true_archetype": AbuseArchetype.LEGITIMATE,
            "is_fraud": False,
        },
    ]

    metrics = evaluator.evaluate_corpus(corpus)

    assert metrics.total_samples == 4
    assert metrics.true_positives == 2
    assert metrics.true_negatives == 2
    assert metrics.false_positives == 0
    assert metrics.false_negatives == 0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1_score == 1.0
    assert metrics.brier_calibration_score < 0.1
    assert "BRAND_IMPERSONATION" in metrics.archetype_breakdown
    assert "LEGITIMATE" in metrics.archetype_breakdown


def test_human_in_the_loop_replay_harness() -> None:
    """Verify ingestion of human reviewer dispositions into regression test suites."""
    engine = TelnyxFraudDecisionEngine()
    harness = HumanInTheLoopReplayHarness(engine)

    # 1. Run an evaluation that generates an audit ledger entry
    res = engine.evaluate_message(
        message="Exclusive offer: Claim your free festival ticket now at http://fest-tickets.cc",
        sender_id="PromoDirect",
        recipient="+353879998877",
        entity_id="cust_marketing_hub",
    )

    audit_id = res.audit_id
    assert audit_id is not None

    # 2. Ingest reviewer disposition
    success = harness.ingest_reviewer_disposition(
        audit_id=audit_id,
        disposition="FALSE_POSITIVE",
        corrected_archetype=AbuseArchetype.LEGITIMATE,
        notes="Legitimate opt-in marketing campaign confirmed by customer ops",
    )
    assert success is True
    assert len(harness.replay_test_cases) == 1

    # Ingest unknown audit_id returns False
    assert harness.ingest_reviewer_disposition("unknown_id", "TRUE_POSITIVE", AbuseArchetype.LEGITIMATE) is False
