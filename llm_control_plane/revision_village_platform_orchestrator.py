"""Revision Village Platform Engineering & AI Coding Agent Guardrail Orchestrator.

Provides automated agent action evaluation, permission boundary verification,
distributed telemetry error attribution (what failed, whom it affected, and why),
and ephemeral environment provisioning for high-velocity EdTech platform delivery.
"""

from __future__ import annotations

import enum
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


class AgentPermissionLevel(str, enum.Enum):
    READ_ONLY = "read_only"
    FEATURE_BRANCH_WRITE = "feature_branch_write"
    PRODUCTION_DEPLOY = "production_deploy"
    RESTRICTED_SECURITY = "restricted_security"


class FailureCategory(str, enum.Enum):
    SYNTAX_ERROR = "syntax_error"
    TYPE_CHECK_FAILURE = "type_check_failure"
    INTEGRATION_TIMEOUT = "integration_timeout"
    SCHEMA_VIOLATION = "schema_violation"
    PERMISSION_DENIED = "permission_denied"
    STUDENT_PII_LEAK = "student_pii_leak"


@dataclass
class CodingAgentAction:
    action_id: str
    agent_id: str
    agent_model: str
    action_type: str  # e.g., "modify_code", "execute_migration", "deploy_service"
    target_files: List[str]
    diff_payload: str
    requested_permission: AgentPermissionLevel
    context_tokens: int = 0
    created_at: float = field(default_factory=time.time)


@dataclass
class AgentGuardrailEvaluation:
    action_id: str
    approved: bool
    risk_score: float  # 0.0 (safe) to 1.0 (critical risk)
    rejection_reasons: List[str] = field(default_factory=list)
    required_mitigations: List[str] = field(default_factory=list)
    evaluated_at: float = field(default_factory=time.time)


@dataclass
class TelemetrySpan:
    span_id: str
    trace_id: str
    service_name: str
    endpoint: str
    student_cohort: str
    affected_user_ids: List[str]
    latency_ms: float
    status_code: int
    error_message: Optional[str] = None
    commit_sha: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class ErrorAttributionResult:
    trace_id: str
    what_failed: str
    failure_category: FailureCategory
    whom_affected: Dict[str, Any]  # cohort, user_count, sample_users
    why_failed: str
    deploying_commit_sha: Optional[str]
    recommended_remediation: str


class CodingAgentGuardrailGate:
    """Enforces safety, permissions, and quality invariants on AI coding agent actions."""

    SENSITIVE_PATTERNS: Set[str] = {
        ".env", "credentials", "secret", "private_key",
        "student_pii", "ferpa_records", "coppa_consent",
        "production_database.tf", "iam_root_policy.tf"
    }

    def __init__(self, max_diff_lines: int = 1000, max_risk_threshold: float = 0.65) -> None:
        self.max_diff_lines = max_diff_lines
        self.max_risk_threshold = max_risk_threshold

    def evaluate_action(self, action: CodingAgentAction) -> AgentGuardrailEvaluation:
        rejection_reasons: List[str] = []
        mitigations: List[str] = []
        risk_score = 0.05

        # Rule 1: Check sensitive file boundaries
        for f in action.target_files:
            lower_f = f.lower()
            if any(p in lower_f for p in self.SENSITIVE_PATTERNS):
                if action.requested_permission != AgentPermissionLevel.RESTRICTED_SECURITY:
                    rejection_reasons.append(f"Unauthorized touch of sensitive resource: {f}")
                    risk_score += 0.60

        # Rule 2: Evaluate diff magnitude
        diff_lines = len(action.diff_payload.strip().splitlines())
        if diff_lines > self.max_diff_lines:
            rejection_reasons.append(f"Diff size ({diff_lines} lines) exceeds autonomous limit ({self.max_diff_lines})")
            risk_score += 0.35
            mitigations.append("Split agent task into modular sub-PRs")
        elif diff_lines > 300:
            risk_score += 0.15

        # Rule 3: Disallow direct production deploys for AI agents without senior signoff
        if action.requested_permission == AgentPermissionLevel.PRODUCTION_DEPLOY:
            rejection_reasons.append("Autonomous agent cannot directly request PRODUCTION_DEPLOY; requires ephemeral staging verification")
            risk_score += 0.40
            mitigations.append("Deploy to ephemeral preview environment first")

        # Rule 4: Check for student PII leaks or un-sanitized queries
        if "select * from student_records" in action.diff_payload.lower():
            rejection_reasons.append("Unscoped student record query detected; potential FERPA/COPPA breach")
            risk_score += 0.70
            mitigations.append("Use scoped projection with student pseudonymization")

        risk_score = min(1.0, max(0.0, risk_score))
        approved = len(rejection_reasons) == 0 and risk_score <= self.max_risk_threshold

        return AgentGuardrailEvaluation(
            action_id=action.action_id,
            approved=approved,
            risk_score=round(risk_score, 3),
            rejection_reasons=rejection_reasons,
            required_mitigations=mitigations
        )


class ObservabilityAttributionEngine:
    """Answers what failed, whom it affected, and why by correlating spans and commit history."""

    def __init__(self) -> None:
        self._spans: List[TelemetrySpan] = []

    def record_span(self, span: TelemetrySpan) -> None:
        self._spans.append(span)

    def analyze_trace(self, trace_id: str) -> Optional[ErrorAttributionResult]:
        trace_spans = [s for s in self._spans if s.trace_id == trace_id]
        if not trace_spans:
            return None

        error_spans = [s for s in trace_spans if s.status_code >= 400 or s.error_message]
        if not error_spans:
            return None

        primary_error = error_spans[0]
        msg = (primary_error.error_message or "").lower()

        # Categorize failure
        if "type error" in msg or "cannot read property" in msg:
            cat = FailureCategory.TYPE_CHECK_FAILURE
            why = "Missing null-check on AST evaluation payload during exam submission."
            remedy = "Enforce strict TypeScript compiler boundary in CI pipeline."
        elif "timeout" in msg or primary_error.latency_ms > 5000:
            cat = FailureCategory.INTEGRATION_TIMEOUT
            why = f"Downstream service exceeded timeout budget ({primary_error.latency_ms}ms)."
            remedy = "Add circuit breaker and exponential backoff retry policy."
        elif "schema" in msg or "unrecognized field" in msg:
            cat = FailureCategory.SCHEMA_VIOLATION
            why = "API payload diverged from shared OpenAPI / JSON-schema contract."
            remedy = "Add contract testing with automated schema drift detection."
        elif "permission" in msg or primary_error.status_code == 403:
            cat = FailureCategory.PERMISSION_DENIED
            why = "Agent or client token lacked appropriate IAM scope."
            remedy = "Verify IAM role mapping in Terraform environment config."
        else:
            cat = FailureCategory.SYNTAX_ERROR
            why = f"Runtime uncaught exception: {primary_error.error_message}"
            remedy = "Inspect stack trace and add defensive unit test case."

        all_affected_users = set()
        cohorts = set()
        for s in trace_spans:
            all_affected_users.update(s.affected_user_ids)
            if s.student_cohort:
                cohorts.add(s.student_cohort)

        return ErrorAttributionResult(
            trace_id=trace_id,
            what_failed=f"{primary_error.service_name}::{primary_error.endpoint} ({primary_error.status_code})",
            failure_category=cat,
            whom_affected={
                "cohorts": sorted(list(cohorts)),
                "total_affected_users": len(all_affected_users),
                "sample_user_ids": sorted(list(all_affected_users))[:5]
            },
            why_failed=why,
            deploying_commit_sha=primary_error.commit_sha,
            recommended_remediation=remedy
        )


class TerraformEphemeralPreviewManager:
    """Provisions and validates ephemeral test environments with isolated AWS state locks."""

    def __init__(self) -> None:
        self._active_environments: Dict[str, Dict[str, Any]] = {}

    def provision_environment(self, branch_name: str, commit_sha: str) -> Dict[str, Any]:
        env_id = f"preview-{hashlib.sha256(branch_name.encode()).hexdigest()[:8]}"
        endpoint = f"https://{env_id}.revisionvillage.internal"
        record = {
            "environment_id": env_id,
            "branch": branch_name,
            "commit_sha": commit_sha,
            "endpoint": endpoint,
            "status": "HEALTHY",
            "created_at": time.time(),
            "fargate_tasks": 2,
            "state_lock": f"s3://rv-terraform-state/env/{env_id}.tfstate"
        }
        self._active_environments[env_id] = record
        return record

    def get_status(self, env_id: str) -> Optional[Dict[str, Any]]:
        return self._active_environments.get(env_id)

    def teardown_environment(self, env_id: str) -> bool:
        if env_id in self._active_environments:
            del self._active_environments[env_id]
            return True
        return False


class RevisionVillagePlatformEngine:
    """Unified coordinator linking AI agent actions, ephemeral testing, and telemetry attribution."""

    def __init__(self) -> None:
        self.guardrail = CodingAgentGuardrailGate()
        self.telemetry = ObservabilityAttributionEngine()
        self.preview_manager = TerraformEphemeralPreviewManager()

    def process_agent_submission(
        self,
        action: CodingAgentAction,
        branch_name: str,
        commit_sha: str
    ) -> Dict[str, Any]:
        eval_result = self.guardrail.evaluate_action(action)
        if not eval_result.approved:
            return {
                "success": False,
                "stage": "GUARDRAIL_REJECTED",
                "evaluation": eval_result,
                "environment": None
            }

        env = self.preview_manager.provision_environment(branch_name, commit_sha)
        return {
            "success": True,
            "stage": "PREVIEW_PROVISIONED",
            "evaluation": eval_result,
            "environment": env
        }
