"""Unit tests for Revision Village Platform Engineering and AI Coding Agent Orchestrator."""

import pytest
from llm_control_plane.revision_village_platform_orchestrator import (
    AgentGuardrailEvaluation,
    AgentPermissionLevel,
    CodingAgentAction,
    CodingAgentGuardrailGate,
    ErrorAttributionResult,
    FailureCategory,
    ObservabilityAttributionEngine,
    RevisionVillagePlatformEngine,
    TelemetrySpan,
    TerraformEphemeralPreviewManager,
)


def test_guardrail_gate_approves_safe_action():
    gate = CodingAgentGuardrailGate()
    action = CodingAgentAction(
        action_id="act-101",
        agent_id="agent-claude-3-7",
        agent_model="claude-3-7-sonnet",
        action_type="modify_code",
        target_files=["src/components/MathExamRenderer.tsx", "src/hooks/useTimer.ts"],
        diff_payload="+ export function MathExamRenderer() { return <div>IB Exam</div>; }",
        requested_permission=AgentPermissionLevel.FEATURE_BRANCH_WRITE,
        context_tokens=1400
    )

    eval_result = gate.evaluate_action(action)
    assert eval_result.approved is True
    assert eval_result.risk_score < 0.20
    assert len(eval_result.rejection_reasons) == 0


def test_guardrail_gate_blocks_sensitive_file_modification():
    gate = CodingAgentGuardrailGate()
    action = CodingAgentAction(
        action_id="act-102",
        agent_id="agent-devin-v2",
        agent_model="devin-deepseek",
        action_type="modify_code",
        target_files=["src/config/production_database.tf"],
        diff_payload="+ resource 'aws_db_instance' 'prod' { publicly_accessible = true }",
        requested_permission=AgentPermissionLevel.FEATURE_BRANCH_WRITE
    )

    eval_result = gate.evaluate_action(action)
    assert eval_result.approved is False
    assert any("sensitive resource" in r.lower() for r in eval_result.rejection_reasons)
    assert eval_result.risk_score >= 0.60


def test_guardrail_gate_blocks_student_pii_query():
    gate = CodingAgentGuardrailGate()
    action = CodingAgentAction(
        action_id="act-103",
        agent_id="agent-swe-bench",
        agent_model="gpt-4o",
        action_type="modify_code",
        target_files=["src/services/studentSync.ts"],
        diff_payload="const users = await db.query('SELECT * FROM student_records WHERE active = 1');",
        requested_permission=AgentPermissionLevel.FEATURE_BRANCH_WRITE
    )

    eval_result = gate.evaluate_action(action)
    assert eval_result.approved is False
    assert any("ferpa/coppa" in r.lower() for r in eval_result.rejection_reasons)


def test_guardrail_gate_blocks_unauthorized_production_deploy():
    gate = CodingAgentGuardrailGate()
    action = CodingAgentAction(
        action_id="act-104",
        agent_id="agent-autonomous-worker",
        agent_model="claude-3-5-sonnet",
        action_type="deploy_service",
        target_files=["deploy.sh"],
        diff_payload="./deploy-prod.sh",
        requested_permission=AgentPermissionLevel.PRODUCTION_DEPLOY
    )

    eval_result = gate.evaluate_action(action)
    assert eval_result.approved is False
    assert any("ephemeral" in m.lower() for m in eval_result.required_mitigations)


def test_observability_error_attribution_what_whom_why():
    engine = ObservabilityAttributionEngine()
    trace_id = "trace-ib-math-hl-202611"

    # Add healthy spans
    engine.record_span(TelemetrySpan(
        span_id="span-1",
        trace_id=trace_id,
        service_name="api-gateway",
        endpoint="/v1/exam/submission",
        student_cohort="IB_DIPLOMA_MAY2027",
        affected_user_ids=["student-nz-01", "student-nz-02"],
        latency_ms=120.0,
        status_code=200
    ))

    # Add failing span
    engine.record_span(TelemetrySpan(
        span_id="span-2",
        trace_id=trace_id,
        service_name="exam-evaluation-engine",
        endpoint="/internal/evaluate-markscheme",
        student_cohort="IB_DIPLOMA_MAY2027",
        affected_user_ids=["student-nz-01"],
        latency_ms=450.0,
        status_code=500,
        error_message="TypeError: Cannot read property 'step_mark' of undefined",
        commit_sha="a9f4c81"
    ))

    attribution = engine.analyze_trace(trace_id)
    assert attribution is not None
    assert "exam-evaluation-engine" in attribution.what_failed
    assert attribution.failure_category == FailureCategory.TYPE_CHECK_FAILURE
    assert attribution.whom_affected["total_affected_users"] >= 2
    assert "IB_DIPLOMA_MAY2027" in attribution.whom_affected["cohorts"]
    assert "null-check" in attribution.why_failed
    assert attribution.deploying_commit_sha == "a9f4c81"
    assert "TypeScript compiler" in attribution.recommended_remediation


def test_ephemeral_preview_manager_lifecycle():
    manager = TerraformEphemeralPreviewManager()
    env = manager.provision_environment("feat/rv-ai-tutoring", "b7d12f4")

    assert env["status"] == "HEALTHY"
    assert "preview-" in env["environment_id"]
    assert env["endpoint"].startswith("https://preview-")

    status = manager.get_status(env["environment_id"])
    assert status == env

    teardown = manager.teardown_environment(env["environment_id"])
    assert teardown is True
    assert manager.get_status(env["environment_id"]) is None


def test_revision_village_platform_engine_end_to_end():
    engine = RevisionVillagePlatformEngine()

    safe_action = CodingAgentAction(
        action_id="act-201",
        agent_id="agent-platform-assistant",
        agent_model="claude-3-7-sonnet",
        action_type="modify_code",
        target_files=["src/routes/practiceQuestions.ts"],
        diff_payload="+ router.get('/topic/:id', getQuestionsByTopic);",
        requested_permission=AgentPermissionLevel.FEATURE_BRANCH_WRITE
    )

    result = engine.process_agent_submission(
        safe_action,
        branch_name="feat/practice-questions",
        commit_sha="c38b291"
    )

    assert result["success"] is True
    assert result["stage"] == "PREVIEW_PROVISIONED"
    assert result["environment"]["status"] == "HEALTHY"
