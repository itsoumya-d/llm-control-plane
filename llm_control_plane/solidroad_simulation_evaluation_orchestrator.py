"""Solidroad Simulation & Multi-Turn Evaluation Orchestrator.

Engineered for Solidroad's Dublin engineering hub:
- Real-time customer support persona simulation engine
- Multi-turn LLM evaluation harnesses (empathy, policy compliance, resolution efficiency)
- CRM transcript ingestion pipeline with strict PII sanitization (Zendesk, Intercom)
"""

from __future__ import annotations

import enum
import re
import time
from typing import Any, Dict, List

from pydantic import BaseModel, Field


class PersonaArchetype(str, enum.Enum):
    FRUSTRATED_CUSTOMER = "frustrated_customer"
    CONFUSED_NOVICE = "confused_novice"
    TECHNICAL_POWER_USER = "technical_power_user"
    URGENT_EXECUTIVE = "urgent_executive"


class DialogueTurn(BaseModel):
    turn_id: int
    speaker: str = Field(description="'customer' or 'agent'")
    utterance: str
    timestamp: float = Field(default_factory=time.time)
    sentiment_score: float = Field(default=0.0, ge=-1.0, le=1.0)


class SimulationConfig(BaseModel):
    scenario_id: str
    persona: PersonaArchetype
    target_topic: str
    expected_resolution: str
    max_turns: int = 10
    latency_budget_ms: int = 250


class TurnEvaluation(BaseModel):
    turn_id: int
    empathy_score: float = Field(ge=0.0, le=100.0)
    compliance_passed: bool
    accuracy_score: float = Field(ge=0.0, le=100.0)
    escalation_needed: bool = False
    feedback_notes: List[str] = Field(default_factory=list)


class SessionEvaluationReport(BaseModel):
    session_id: str
    scenario_id: str
    persona: PersonaArchetype
    total_turns: int
    avg_empathy: float
    compliance_pass_rate: float
    resolution_achieved: bool
    overall_grade: str
    violations: List[str] = Field(default_factory=list)


class CRMIngestionPipeline:
    """Ingests raw transcripts from Zendesk and Intercom, sanitizing customer PII."""

    # Regex patterns for sanitization
    CREDIT_CARD_RX = re.compile(r"\b(?:\d{4}[ -]?){3}\d{4}\b")
    EMAIL_RX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
    PHONE_RX = re.compile(r"\b(?:\+\d{1,3}[ -]?)?(?:\(?\d{1,4}\)?[ -]?)?\d{3,4}[ -]?\d{3,4}\b")
    IRISH_PPSN_RX = re.compile(r"\b\d{7}[A-Wa-z]{1,2}\b")

    @classmethod
    def sanitize_text(cls, text: str) -> str:
        """Strip all sensitive identifiers prior to ingestion."""
        sanitized = cls.IRISH_PPSN_RX.sub("[REDACTED_PPSN]", text)
        sanitized = cls.CREDIT_CARD_RX.sub("[REDACTED_CARD]", sanitized)
        sanitized = cls.EMAIL_RX.sub("[REDACTED_EMAIL]", sanitized)
        sanitized = cls.PHONE_RX.sub("[REDACTED_PHONE]", sanitized)
        return sanitized

    @classmethod
    def ingest_zendesk_ticket(cls, payload: Dict[str, Any]) -> List[DialogueTurn]:
        """Ingest Zendesk ticket comments into structured DialogueTurn sequence."""
        comments = payload.get("comments", [])
        turns: List[DialogueTurn] = []
        for i, c in enumerate(comments, 1):
            body = cls.sanitize_text(c.get("body", ""))
            author_type = c.get("author_type", "customer")
            speaker = "agent" if author_type in ("agent", "admin") else "customer"
            turns.append(DialogueTurn(turn_id=i, speaker=speaker, utterance=body))
        return turns

    @classmethod
    def ingest_intercom_conversation(cls, payload: Dict[str, Any]) -> List[DialogueTurn]:
        """Ingest Intercom conversation parts into structured DialogueTurn sequence."""
        parts = payload.get("conversation_parts", [])
        turns: List[DialogueTurn] = []
        for i, p in enumerate(parts, 1):
            body = cls.sanitize_text(p.get("body", ""))
            part_type = p.get("part_type", "comment")
            speaker = "agent" if part_type in ("admin_reply", "bot_reply") else "customer"
            turns.append(DialogueTurn(turn_id=i, speaker=speaker, utterance=body))
        return turns


class SolidroadSimulationEngine:
    """Simulates realistic customer responses and maintains interactive state."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.history: List[DialogueTurn] = []
        self.current_sentiment: float = -0.6 if config.persona == PersonaArchetype.FRUSTRATED_CUSTOMER else 0.0
        self.resolved: bool = False

    def receive_agent_turn(self, utterance: str) -> DialogueTurn:
        """Process trainee/agent input and generate simulated customer reaction."""
        turn_num = len(self.history) + 1
        agent_turn = DialogueTurn(turn_id=turn_num, speaker="agent", utterance=utterance)
        self.history.append(agent_turn)

        # Update simulated customer sentiment based on empathy markers
        empathy_cues = ["understand", "apologize", "sorry", "help you", "resolve", "appreciate"]
        empathy_detected = any(cue in utterance.lower() for cue in empathy_cues)
        if empathy_detected:
            self.current_sentiment = min(1.0, self.current_sentiment + 0.3)
        else:
            self.current_sentiment = max(-1.0, self.current_sentiment - 0.1)

        # Check resolution condition
        resolution_cues = ["refund processed", "account credited", "issue resolved", "fixed", "completed"]
        if any(rc in utterance.lower() for rc in resolution_cues):
            self.resolved = True
            self.current_sentiment = max(0.5, self.current_sentiment)

        # Generate customer reaction
        cust_turn_num = len(self.history) + 1
        if self.resolved:
            response = "Thank you so much! That completely resolves my issue. I really appreciate your quick help."
        elif self.current_sentiment < -0.3:
            response = (
                "I've been waiting for hours and this is really frustrating. "
                "Can you guarantee this will get sorted today?"
            )
        elif self.current_sentiment < 0.2:
            response = "Okay, I see what you're saying, but what are the next steps to get this confirmed?"
        else:
            response = "Thanks for looking into this. That sounds much clearer now. What else do you need from me?"

        customer_turn = DialogueTurn(
            turn_id=cust_turn_num,
            speaker="customer",
            utterance=response,
            sentiment_score=round(self.current_sentiment, 2),
        )
        self.history.append(customer_turn)
        return customer_turn


class MultiTurnEvalHarness:
    """Evaluates trainee multi-turn dialogues across empathy, compliance, and accuracy."""

    COMPLIANCE_FORBIDDEN = [
        "give you my personal number",
        "guarantee 100% stock returns",
        "bypass security verification",
        "share your password",
    ]

    @classmethod
    def evaluate_turn(cls, agent_utterance: str, turn_id: int) -> TurnEvaluation:
        """Evaluate an individual trainee response."""
        utt_lower = agent_utterance.lower()
        feedback: List[str] = []

        # 1. Empathy scoring
        empathy_tokens = ["understand", "apologize", "sorry", "appreciate", "patience", "happy to help"]
        matched_empathy = sum(1 for t in empathy_tokens if t in utt_lower)
        empathy_score = min(100.0, 40.0 + (matched_empathy * 20.0))
        if matched_empathy == 0:
            feedback.append("Consider expressing empathy or acknowledging customer concern early.")

        # 2. Compliance check
        compliance_passed = True
        for forbidden in cls.COMPLIANCE_FORBIDDEN:
            if forbidden in utt_lower:
                compliance_passed = False
                feedback.append(f"Compliance violation: '{forbidden}' is strictly prohibited.")

        # 3. Accuracy / professionalism score
        professional_tokens = ["policy", "account", "reference", "verify", "secure", "process", "resolution"]
        matched_prof = sum(1 for t in professional_tokens if t in utt_lower)
        accuracy_score = min(100.0, 50.0 + (matched_prof * 15.0))

        # 4. Escalation check
        escalation_needed = "manager" in utt_lower or "escalate" in utt_lower

        return TurnEvaluation(
            turn_id=turn_id,
            empathy_score=empathy_score,
            compliance_passed=compliance_passed,
            accuracy_score=accuracy_score,
            escalation_needed=escalation_needed,
            feedback_notes=feedback,
        )

    @classmethod
    def evaluate_session(
        cls, session_id: str, config: SimulationConfig, history: List[DialogueTurn]
    ) -> SessionEvaluationReport:
        """Evaluate complete multi-turn simulation session."""
        agent_turns = [t for t in history if t.speaker == "agent"]
        if not agent_turns:
            return SessionEvaluationReport(
                session_id=session_id,
                scenario_id=config.scenario_id,
                persona=config.persona,
                total_turns=0,
                avg_empathy=0.0,
                compliance_pass_rate=1.0,
                resolution_achieved=False,
                overall_grade="INCOMPLETE",
                violations=["No agent responses recorded."],
            )

        evaluations: List[TurnEvaluation] = []
        violations: List[str] = []

        for t in agent_turns:
            turn_eval = cls.evaluate_turn(t.utterance, t.turn_id)
            evaluations.append(turn_eval)
            if not turn_eval.compliance_passed:
                violations.extend(turn_eval.feedback_notes)

        avg_empathy = round(sum(e.empathy_score for e in evaluations) / len(evaluations), 2)
        compliance_pass_rate = round(sum(1.0 for e in evaluations if e.compliance_passed) / len(evaluations), 2)

        # Resolution achieved if last customer turn indicates positive resolution
        resolution_achieved = False
        if history and history[-1].speaker == "customer":
            resolution_achieved = history[-1].sentiment_score >= 0.4

        # Overall grade computation
        if compliance_pass_rate < 1.0:
            overall_grade = "FAIL_COMPLIANCE"
        elif avg_empathy >= 80.0 and resolution_achieved:
            overall_grade = "EXCELLENT"
        elif avg_empathy >= 65.0:
            overall_grade = "PASS"
        else:
            overall_grade = "NEEDS_IMPROVEMENT"

        return SessionEvaluationReport(
            session_id=session_id,
            scenario_id=config.scenario_id,
            persona=config.persona,
            total_turns=len(history),
            avg_empathy=avg_empathy,
            compliance_pass_rate=compliance_pass_rate,
            resolution_achieved=resolution_achieved,
            overall_grade=overall_grade,
            violations=violations,
        )
