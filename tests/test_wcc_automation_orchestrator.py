"""Unit tests for Wellington City Council Process Automation & Decision Support Orchestrator."""
from __future__ import annotations

import pytest

from llm_control_plane.wcc_automation_orchestrator import (
    CouncilDocumentClassifier,
    DecisionSupportGuardrail,
    MunicipalDocumentCategory,
    ProcessExecutionRecord,
    WellingtonCouncilAutomationOrchestrator,
    WorkflowStatus,
    WorkflowStep,
    WorkflowTaskType,
)


def test_council_document_classifier_resource_consent():
    text = "Application under Resource Management Act for building consent and district plan subdivision in Khandallah."
    category, confidence = CouncilDocumentClassifier.classify_text(text)
    assert category == MunicipalDocumentCategory.RESOURCE_CONSENT
    assert confidence >= 0.84


def test_council_document_classifier_rates():
    text = "Query regarding 2026 rating unit capital value, annual rates assessment, and rates rebate eligibility."
    category, confidence = CouncilDocumentClassifier.classify_text(text)
    assert category == MunicipalDocumentCategory.RATES_VALUATION
    assert confidence >= 0.84


def test_council_document_classifier_infrastructure():
    text = "Urgent service request: leaking stormwater water pipe on Victoria Street affecting Wellington Water mains."
    category, confidence = CouncilDocumentClassifier.classify_text(text)
    assert category == MunicipalDocumentCategory.INFRASTRUCTURE_SERVICE
    assert confidence >= 0.80


def test_council_document_classifier_unknown():
    text = "Random unrelated text with no civic or local government keywords."
    category, confidence = CouncilDocumentClassifier.classify_text(text)
    assert category == MunicipalDocumentCategory.UNKNOWN
    assert confidence == 0.0


def test_decision_guardrail_confidence_threshold():
    guardrail = DecisionSupportGuardrail(min_confidence_threshold=0.85)

    # Valid confidence
    allowed, reason = guardrail.evaluate_decision("resource_consent", 0.92, {"estimated_capex_nzd": 50_000})
    assert allowed is True
    assert reason is None

    # Low confidence -> escalation
    allowed, reason = guardrail.evaluate_decision("resource_consent", 0.70, {})
    assert allowed is False
    assert "below statutory threshold" in reason


def test_decision_guardrail_delegation_cap():
    guardrail = DecisionSupportGuardrail(min_confidence_threshold=0.80)

    # Over 250,000 capex -> must escalate
    allowed, reason = guardrail.evaluate_decision(
        "resource_consent", 0.95, {"estimated_capex_nzd": 450_000}
    )
    assert allowed is False
    assert "exceeds automated delegation cap" in reason


def test_workflow_orchestration_happy_path():
    orchestrator = WellingtonCouncilAutomationOrchestrator()

    # Step 1: Ingestion & Classification
    def ingest_step(state):
        raw = state["raw_text"]
        cat, conf = CouncilDocumentClassifier.classify_text(raw)
        return {"predicted_category": cat.value, "confidence": conf}

    # Step 2: Decision Routing
    def decision_step(state):
        return {"routed_queue": f"queue_{state['predicted_category']}"}

    # Step 3: SAP / Enterprise Integration Sync
    def integration_step(state):
        return {"external_ticket_id": f"WCC-2026-{state['predicted_category'][:3].upper()}-9912"}

    orchestrator.register_step(
        WorkflowStep(
            step_id="step_ingest",
            task_type=WorkflowTaskType.DOCUMENT_EXTRACTION,
            name="Ingest and Classify",
            handler=ingest_step,
            required_inputs=["raw_text"],
        )
    )
    orchestrator.register_step(
        WorkflowStep(
            step_id="step_decision",
            task_type=WorkflowTaskType.DECISION_SUPPORT,
            name="Policy & Threshold Evaluation",
            handler=decision_step,
            required_inputs=["predicted_category", "confidence"],
        )
    )
    orchestrator.register_step(
        WorkflowStep(
            step_id="step_sync",
            task_type=WorkflowTaskType.INTEGRATION_SYNC,
            name="Enterprise Systems Sync",
            handler=integration_step,
            required_inputs=["routed_queue"],
        )
    )

    initial_payload = {
        "raw_text": "Stormwater water pipe failure reported by resident on Lambton Quay Wellington Water.",
        "estimated_capex_nzd": 12_000,
    }

    record = orchestrator.execute_workflow(
        process_id="proc-wcc-001",
        step_ids=["step_ingest", "step_decision", "step_sync"],
        initial_payload=initial_payload,
    )

    assert record.status == WorkflowStatus.COMPLETED
    assert record.steps_executed == ["step_ingest", "step_decision", "step_sync"]
    assert record.state_payload["predicted_category"] == MunicipalDocumentCategory.INFRASTRUCTURE_SERVICE.value
    assert record.state_payload["routed_queue"] == "queue_infrastructure_service"
    assert "WCC-2026-INF" in record.state_payload["external_ticket_id"]
    assert len(record.audit_hash) == 64  # SHA-256


def test_workflow_orchestration_escalation_path():
    orchestrator = WellingtonCouncilAutomationOrchestrator(
        guardrail=DecisionSupportGuardrail(min_confidence_threshold=0.85)
    )

    def ingest_step(state):
        return {
            "predicted_category": MunicipalDocumentCategory.RESOURCE_CONSENT.value,
            "confidence": 0.95,
        }

    def decision_step(state):
        return {"decision_evaluated": True}

    orchestrator.register_step(
        WorkflowStep(
            step_id="ingest",
            task_type=WorkflowTaskType.DOCUMENT_EXTRACTION,
            name="Ingest",
            handler=ingest_step,
        )
    )
    orchestrator.register_step(
        WorkflowStep(
            step_id="decision",
            task_type=WorkflowTaskType.DECISION_SUPPORT,
            name="Decision",
            handler=decision_step,
            required_inputs=["predicted_category", "confidence"],
        )
    )

    # Capex of 600,000 exceeds delegation cap ($250k) -> triggers escalation
    record = orchestrator.execute_workflow(
        process_id="proc-wcc-002",
        step_ids=["ingest", "decision"],
        initial_payload={"estimated_capex_nzd": 600_000},
    )

    assert record.status == WorkflowStatus.ESCALATED
    assert "exceeds automated delegation cap" in record.escalation_reason
    assert record.steps_executed == ["ingest", "decision"]
    assert len(record.audit_hash) == 64
