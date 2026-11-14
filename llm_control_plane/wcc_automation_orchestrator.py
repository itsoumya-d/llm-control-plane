"""Wellington City Council Intelligent Process Automation & Decision Support Engine.

Orchestrates multi-stage civic workflows, document processing, and AI-assisted
decision support with safety guardrails, human-in-the-loop escalation, and
deterministic audit logging compliant with NZ Local Government guidelines.
"""
from __future__ import annotations

import enum
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("llm_control_plane.wcc_automation")


class WorkflowTaskType(str, enum.Enum):
    DOCUMENT_EXTRACTION = "document_extraction"
    DECISION_SUPPORT = "decision_support"
    PROCESS_ORCHESTRATION = "process_orchestration"
    INTEGRATION_SYNC = "integration_sync"
    HUMAN_ESCALATION = "human_escalation"


class WorkflowStatus(str, enum.Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ESCALATED = "escalated"
    FAILED = "failed"


class MunicipalDocumentCategory(str, enum.Enum):
    RESOURCE_CONSENT = "resource_consent"
    RATES_VALUATION = "rates_valuation"
    INFRASTRUCTURE_SERVICE = "infrastructure_service"
    ENVIRONMENTAL_PERMIT = "environmental_permit"
    COMMUNITY_GRANT = "community_grant"
    UNKNOWN = "unknown"


@dataclass
class WorkflowStep:
    step_id: str
    task_type: WorkflowTaskType
    name: str
    handler: Callable[[Dict[str, Any]], Dict[str, Any]]
    required_inputs: List[str] = field(default_factory=list)
    timeout_seconds: float = 30.0
    retry_limit: int = 3
    is_terminal: bool = False


@dataclass
class ProcessExecutionRecord:
    process_id: str
    started_at: str
    completed_at: Optional[str]
    status: WorkflowStatus
    steps_executed: List[str] = field(default_factory=list)
    state_payload: Dict[str, Any] = field(default_factory=dict)
    audit_hash: str = ""
    escalation_reason: Optional[str] = None


class DecisionSupportGuardrail:
    """Enforces policy boundaries, confidence thresholds, and deterministic escalation."""

    def __init__(self, min_confidence_threshold: float = 0.85):
        self.min_confidence_threshold = min_confidence_threshold

    def evaluate_decision(
        self,
        predicted_category: str,
        confidence: float,
        context: Dict[str, Any],
    ) -> Tuple[bool, Optional[str]]:
        """Determines whether AI decision can proceed automatically or requires human review."""
        if confidence < self.min_confidence_threshold:
            return (
                False,
                f"Confidence score {confidence:.2f} is below statutory threshold {self.min_confidence_threshold:.2f}",
            )

        # High-impact planning decisions require manual sign-off under local government bylaws
        if predicted_category == MunicipalDocumentCategory.RESOURCE_CONSENT.value:
            estimated_value = context.get("estimated_capex_nzd", 0)
            if estimated_value > 250_000:
                return (
                    False,
                    f"Resource consent value NZD ${estimated_value:,.2f} exceeds automated delegation cap ($250,000)",
                )

        return True, None


class CouncilDocumentClassifier:
    """Classifies incoming council submissions and extracts structured metadata."""

    KEYWORD_MAPPINGS = {
        MunicipalDocumentCategory.RESOURCE_CONSENT: [
            "district plan",
            "building consent",
            "resource management act",
            "rma s88",
            "zoning",
            "subdivision",
        ],
        MunicipalDocumentCategory.RATES_VALUATION: [
            "rates assessment",
            "capital value",
            "land value",
            "rating unit",
            "rates rebate",
        ],
        MunicipalDocumentCategory.INFRASTRUCTURE_SERVICE: [
            "water pipe",
            "stormwater",
            "pothole",
            "roading",
            "wellington water",
            "waste collection",
        ],
        MunicipalDocumentCategory.ENVIRONMENTAL_PERMIT: [
            "noise control",
            "trade waste",
            "food hygiene",
            "liquor licence",
        ],
        MunicipalDocumentCategory.COMMUNITY_GRANT: [
            "arts grant",
            "social grant",
            "sports subsidy",
            "community fund",
        ],
    }

    @classmethod
    def classify_text(cls, text: str) -> Tuple[MunicipalDocumentCategory, float]:
        text_lower = text.lower()
        best_cat = MunicipalDocumentCategory.UNKNOWN
        max_matches = 0
        total_matched = 0

        for cat, keywords in cls.KEYWORD_MAPPINGS.items():
            matches = sum(1 for kw in keywords if kw in text_lower)
            if matches > max_matches:
                max_matches = matches
                best_cat = cat
            total_matched += matches

        if max_matches == 0:
            return MunicipalDocumentCategory.UNKNOWN, 0.0

        confidence = min(0.98, 0.60 + (max_matches * 0.12))
        return best_cat, round(confidence, 2)


class WellingtonCouncilAutomationOrchestrator:
    """Orchestrates end-to-end civic automation workflows with verifiable telemetry."""

    def __init__(self, guardrail: Optional[DecisionSupportGuardrail] = None):
        self.guardrail = guardrail or DecisionSupportGuardrail()
        self.steps: Dict[str, WorkflowStep] = {}
        self.execution_history: List[ProcessExecutionRecord] = []

    def register_step(self, step: WorkflowStep) -> None:
        self.steps[step.step_id] = step

    def _compute_audit_hash(self, process_id: str, payload: Dict[str, Any], timestamp: str) -> str:
        serialized = json.dumps({"process_id": process_id, "payload": payload, "timestamp": timestamp}, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def execute_workflow(
        self,
        process_id: str,
        step_ids: List[str],
        initial_payload: Dict[str, Any],
    ) -> ProcessExecutionRecord:
        start_time = datetime.now(timezone.utc).isoformat()
        current_state = dict(initial_payload)
        executed_steps = []

        record = ProcessExecutionRecord(
            process_id=process_id,
            started_at=start_time,
            completed_at=None,
            status=WorkflowStatus.IN_PROGRESS,
            state_payload=current_state,
        )

        for sid in step_ids:
            if sid not in self.steps:
                record.status = WorkflowStatus.FAILED
                record.completed_at = datetime.now(timezone.utc).isoformat()
                record.audit_hash = self._compute_audit_hash(process_id, current_state, start_time)
                return record

            step = self.steps[sid]

            # Validate input prerequisites
            for req in step.required_inputs:
                if req not in current_state:
                    record.status = WorkflowStatus.FAILED
                    record.completed_at = datetime.now(timezone.utc).isoformat()
                    record.audit_hash = self._compute_audit_hash(process_id, current_state, start_time)
                    return record

            # Execute step handler
            try:
                output = step.handler(current_state)
                current_state.update(output)
                executed_steps.append(sid)
            except Exception as exc:
                logger.error("Step %s failed for process %s: %s", sid, process_id, exc)
                record.status = WorkflowStatus.FAILED
                record.completed_at = datetime.now(timezone.utc).isoformat()
                record.audit_hash = self._compute_audit_hash(process_id, current_state, start_time)
                return record

            # Check decision support guardrail if applicable
            if step.task_type == WorkflowTaskType.DECISION_SUPPORT:
                category = current_state.get("predicted_category", "")
                confidence = current_state.get("confidence", 0.0)
                allowed, reason = self.guardrail.evaluate_decision(category, confidence, current_state)
                if not allowed:
                    record.status = WorkflowStatus.ESCALATED
                    record.escalation_reason = reason
                    record.completed_at = datetime.now(timezone.utc).isoformat()
                    record.steps_executed = executed_steps
                    record.state_payload = current_state
                    record.audit_hash = self._compute_audit_hash(process_id, current_state, start_time)
                    self.execution_history.append(record)
                    return record

        record.status = WorkflowStatus.COMPLETED
        record.completed_at = datetime.now(timezone.utc).isoformat()
        record.steps_executed = executed_steps
        record.state_payload = current_state
        record.audit_hash = self._compute_audit_hash(process_id, current_state, start_time)
        self.execution_history.append(record)
        return record
