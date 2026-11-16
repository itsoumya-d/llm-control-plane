"""Tests for Solidroad Simulation & Multi-Turn Evaluation Orchestrator."""

from llm_control_plane.solidroad_simulation_evaluation_orchestrator import (
    CRMIngestionPipeline,
    DialogueTurn,
    MultiTurnEvalHarness,
    PersonaArchetype,
    SessionEvaluationReport,
    SimulationConfig,
    SolidroadSimulationEngine,
    TurnEvaluation,
)


def test_crm_ingestion_sanitizes_credit_cards_and_emails():
    raw_text = "Customer says: My card is 4111-2222-3333-4444 and my email is test.user@example.com."
    sanitized = CRMIngestionPipeline.sanitize_text(raw_text)
    assert "4111-2222-3333-4444" not in sanitized
    assert "test.user@example.com" not in sanitized
    assert "[REDACTED_CARD]" in sanitized
    assert "[REDACTED_EMAIL]" in sanitized


def test_crm_ingestion_sanitizes_irish_ppsn_and_phone():
    raw_text = "Irish PPSN is 1234567T and contact number is +353 1 234 5678."
    sanitized = CRMIngestionPipeline.sanitize_text(raw_text)
    assert "1234567T" not in sanitized
    assert "[REDACTED_PPSN]" in sanitized
    assert "[REDACTED_PHONE]" in sanitized


def test_zendesk_and_intercom_payload_parsing():
    zd_payload = {
        "comments": [
            {"author_type": "customer", "body": "My package didn't arrive."},
            {"author_type": "agent", "body": "I apologize for the delay. Let me check your tracking number."},
        ]
    }
    zd_turns = CRMIngestionPipeline.ingest_zendesk_ticket(zd_payload)
    assert len(zd_turns) == 2
    assert zd_turns[0].speaker == "customer"
    assert zd_turns[1].speaker == "agent"

    ic_payload = {
        "conversation_parts": [
            {"part_type": "comment", "body": "Need help resetting my credentials."},
            {"part_type": "admin_reply", "body": "Happy to help with password reset."},
        ]
    }
    ic_turns = CRMIngestionPipeline.ingest_intercom_conversation(ic_payload)
    assert len(ic_turns) == 2
    assert ic_turns[0].speaker == "customer"
    assert ic_turns[1].speaker == "agent"


def test_simulation_engine_initializes_sentiment():
    config = SimulationConfig(
        scenario_id="SCENARIO_001",
        persona=PersonaArchetype.FRUSTRATED_CUSTOMER,
        target_topic="Billing dispute",
        expected_resolution="Issue refund",
    )
    engine = SolidroadSimulationEngine(config)
    assert engine.current_sentiment == -0.6
    assert not engine.resolved


def test_simulation_engine_sentiment_drift_on_empathy():
    config = SimulationConfig(
        scenario_id="SCENARIO_002",
        persona=PersonaArchetype.FRUSTRATED_CUSTOMER,
        target_topic="Delayed delivery",
        expected_resolution="Expedited replacement",
    )
    engine = SolidroadSimulationEngine(config)
    initial_sentiment = engine.current_sentiment

    # Agent responds with strong empathy
    cust_turn = engine.receive_agent_turn(
        "I completely understand your frustration and apologize for the delay. I will help you resolve this."
    )
    assert engine.current_sentiment > initial_sentiment
    assert cust_turn.speaker == "customer"
    assert len(engine.history) == 2


def test_simulation_engine_resolution_triggers_positive_turn():
    config = SimulationConfig(
        scenario_id="SCENARIO_003",
        persona=PersonaArchetype.CONFUSED_NOVICE,
        target_topic="Subscription cancellation",
        expected_resolution="Cancel confirmation",
    )
    engine = SolidroadSimulationEngine(config)
    cust_turn = engine.receive_agent_turn("I have completed your cancellation and your refund processed in full.")
    assert engine.resolved is True
    assert "Thank you so much!" in cust_turn.utterance


def test_multiturn_eval_harness_turn_empathy_scoring():
    eval_result = MultiTurnEvalHarness.evaluate_turn(
        "I understand and apologize for this inconvenience. I appreciate your patience.",
        turn_id=1,
    )
    assert isinstance(eval_result, TurnEvaluation)
    assert eval_result.empathy_score >= 80.0
    assert eval_result.compliance_passed is True
    assert not eval_result.escalation_needed


def test_multiturn_eval_harness_detects_compliance_violation():
    eval_result = MultiTurnEvalHarness.evaluate_turn(
        "Don't worry, I can bypass security verification for you this one time.",
        turn_id=2,
    )
    assert eval_result.compliance_passed is False
    assert any("bypass security verification" in note for note in eval_result.feedback_notes)


def test_multiturn_eval_harness_full_session_evaluation():
    config = SimulationConfig(
        scenario_id="SCENARIO_004",
        persona=PersonaArchetype.TECHNICAL_POWER_USER,
        target_topic="API rate limiting",
        expected_resolution="Limit increased",
    )
    history = [
        DialogueTurn(
            turn_id=1, speaker="customer", utterance="My webhook requests are hitting 429 errors.", sentiment_score=-0.2
        ),
        DialogueTurn(
            turn_id=2,
            speaker="agent",
            utterance="I understand the urgency and apologize for the API throttling. Let me verify your tier.",
        ),
        DialogueTurn(turn_id=3, speaker="customer", utterance="Thanks, here is my app ID.", sentiment_score=0.1),
        DialogueTurn(
            turn_id=4,
            speaker="agent",
            utterance=(
                "I appreciate your patience while I verified your account. Your quota upgrade is completed and fixed."
            ),
        ),
        DialogueTurn(
            turn_id=5,
            speaker="customer",
            utterance="Awesome, everything is resolved. Appreciate it!",
            sentiment_score=0.7,
        ),
    ]

    report = MultiTurnEvalHarness.evaluate_session("SESS_12345", config, history)
    assert isinstance(report, SessionEvaluationReport)
    assert report.total_turns == 5
    assert report.compliance_pass_rate == 1.0
    assert report.resolution_achieved is True
    assert report.overall_grade in ("EXCELLENT", "PASS")
    assert not report.violations


def test_multiturn_eval_harness_handles_empty_agent_turns():
    config = SimulationConfig(
        scenario_id="SCENARIO_005",
        persona=PersonaArchetype.URGENT_EXECUTIVE,
        target_topic="SLA breach",
        expected_resolution="Executive briefing",
    )
    report = MultiTurnEvalHarness.evaluate_session("SESS_EMPTY", config, [])
    assert report.overall_grade == "INCOMPLETE"
    assert report.total_turns == 0
