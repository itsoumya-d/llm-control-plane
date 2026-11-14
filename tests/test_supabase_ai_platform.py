"""
Unit tests for Supabase AI Platform execution layer and governance engine.
"""

import time

import pytest

from llm_control_plane.supabase_ai_platform import (
    AgentRegistration,
    AtomicRollbackEngine,
    DurableStateStore,
    GateStatus,
    GoldenEvaluationHarness,
    GoldenTestCase,
    HumanReviewGate,
    InterruptionBudgetManager,
    RiskTier,
    SupabaseAIPlatform,
)


def test_agent_registration_tool_grants():
    agent = AgentRegistration(
        agent_id="agent-triage",
        name="Issue Triage Agent",
        owner="eng-ops",
        risk_tier=RiskTier.TIER_1_READ_ONLY,
        allowed_tools={"github_read_issue", "postgres_query"},
    )
    assert agent.validate_tool_access("github_read_issue") is True
    assert agent.validate_tool_access("postgres_query") is True
    assert agent.validate_tool_access("slack_post_message") is False

    agent.enabled = False
    assert agent.validate_tool_access("github_read_issue") is False


def test_durable_state_store_checkpoint_and_recovery():
    store = DurableStateStore()
    payload_v1 = {"step": "ingest", "items_processed": 42}
    cp1 = store.save_checkpoint("agent-etl", payload_v1)
    assert cp1.version == 1
    assert store.get_version_count("agent-etl") == 1

    payload_v2 = {"step": "transform", "items_processed": 100}
    store.save_checkpoint("agent-etl", payload_v2)
    assert store.get_version_count("agent-etl") == 2

    # Recover latest state
    recovered = store.recover_latest_state("agent-etl")
    assert recovered == payload_v2

    # Check non-existent agent
    assert store.recover_latest_state("non-existent") is None


def test_human_review_gate_lifecycle():
    gate = HumanReviewGate()
    ticket = gate.create_ticket(
        agent_id="agent-billing",
        action_type="refund_charge",
        action_payload={"amount_cents": 5000, "customer": "cust_123"},
        ttl_seconds=1.0,
    )
    assert ticket.status == GateStatus.PENDING
    assert gate.get_status(ticket.ticket_id) == GateStatus.PENDING

    # Approve
    approved = gate.approve_ticket(ticket.ticket_id, reviewer="lead-sec", reason="Verified ticket")
    assert approved is True
    assert gate.get_status(ticket.ticket_id) == GateStatus.APPROVED

    # Create second ticket and reject
    ticket2 = gate.create_ticket("agent-billing", "delete_account", {}, ttl_seconds=10.0)
    rejected = gate.reject_ticket(ticket2.ticket_id, reviewer="lead-sec", reason="Dangerous")
    assert rejected is True
    assert gate.get_status(ticket2.ticket_id) == GateStatus.REJECTED

    # Create third ticket and test expiration
    ticket3 = gate.create_ticket("agent-billing", "payout", {}, ttl_seconds=0.05)
    time.sleep(0.1)
    assert gate.get_status(ticket3.ticket_id) == GateStatus.EXPIRED
    assert gate.approve_ticket(ticket3.ticket_id, "lead-sec") is False


def test_atomic_rollback_engine():
    rollback = AtomicRollbackEngine()
    state = {"db_row": True, "cache_key": True}

    def revert_db():
        state["db_row"] = False
        return True

    def revert_cache():
        state["cache_key"] = False
        return True

    rollback.record("step-1", "insert_db", revert_db)
    rollback.record("step-2", "set_cache", revert_cache)
    assert rollback.pending_count == 2

    # Trigger rollback
    results = rollback.rollback_all()
    assert results["step-2"] is True
    assert results["step-1"] is True
    assert state["db_row"] is False
    assert state["cache_key"] is False
    assert rollback.pending_count == 0


def test_interruption_budget_and_bundling():
    mgr = InterruptionBudgetManager()
    recipient = "engineer@supabase.io"

    # Hourly budget of 2
    assert mgr.can_notify_immediately(recipient, hourly_budget=2) is True
    mgr.record_notification(recipient)
    assert mgr.can_notify_immediately(recipient, hourly_budget=2) is True
    mgr.record_notification(recipient)

    # Exceeded budget
    assert mgr.can_notify_immediately(recipient, hourly_budget=2) is False

    # Buffer non-urgent messages
    count1 = mgr.buffer_message(recipient, {"type": "pr_review", "id": 101})
    count2 = mgr.buffer_message(recipient, {"type": "lint_warning", "id": 102})
    assert count1 == 1
    assert count2 == 2

    # Flush digest
    digest = mgr.flush_digest(recipient)
    assert len(digest) == 2
    assert digest[0]["id"] == 101
    assert mgr.flush_digest(recipient) == []


def test_golden_evaluation_harness():
    harness = GoldenEvaluationHarness(pass_threshold_pct=80.0)
    harness.add_case(
        GoldenTestCase(
            test_id="case-1",
            prompt="Summarize incident 404",
            required_keywords=["root cause", "resolved"],
            forbidden_keywords=["hallucination", "unknown error"],
            expected_risk_tier=RiskTier.TIER_1_READ_ONLY,
        )
    )
    harness.add_case(
        GoldenTestCase(
            test_id="case-2",
            prompt="Draft customer update",
            required_keywords=["apology", "status"],
            forbidden_keywords=["refund promised"],
            expected_risk_tier=RiskTier.TIER_2_INTERNAL_DRAFT,
        )
    )

    def dummy_runner(prompt: str):
        if "incident" in prompt:
            return {
                "output": "Root cause identified and resolved cleanly.",
                "tier": RiskTier.TIER_1_READ_ONLY,
            }
        return {
            "output": "We sincerely offer our apology. System status restored.",
            "tier": RiskTier.TIER_2_INTERNAL_DRAFT,
        }

    report = harness.evaluate_agent(dummy_runner)
    assert report.total_cases == 2
    assert report.passed_cases == 2
    assert report.score_pct == 100.0
    assert report.passed is True
    assert len(report.failures) == 0


def test_supabase_ai_platform_orchestration():
    platform = SupabaseAIPlatform()

    agent1 = AgentRegistration(
        agent_id="agent-read",
        name="Telemetry Reader",
        owner="analytics",
        risk_tier=RiskTier.TIER_1_READ_ONLY,
        allowed_tools={"query_logs", "aggregate_metrics"},
    )
    agent2 = AgentRegistration(
        agent_id="agent-action",
        name="Production Deployer",
        owner="infra",
        risk_tier=RiskTier.TIER_3_EXTERNAL_ACTION,
        allowed_tools={"promote_canary", "revert_traffic"},
    )
    platform.register_agent(agent1)
    platform.register_agent(agent2)

    # 1. Unregistered tool denied
    with pytest.raises(PermissionError, match="tool grant"):
        platform.dispatch_action("agent-read", "unauthorized_tool", {})

    # 2. Structurally forbidden dangerous operations denied
    with pytest.raises(PermissionError, match="structurally forbidden"):
        platform.dispatch_action("agent-read", "system_record_status_write", {})

    # 3. Safe Tier 1 action executes directly
    res1 = platform.dispatch_action("agent-read", "query_logs", {"filter": "error"})
    assert res1["status"] == "EXECUTED"

    # 4. Consequential Tier 3 action paused pending human approval gate
    res2 = platform.dispatch_action("agent-action", "promote_canary", {"version": "v2.5.0"})
    assert res2["status"] == "AWAITING_HUMAN_APPROVAL"
    ticket_id = res2["ticket_id"]

    # Trying to execute unapproved ticket fails
    with pytest.raises(PermissionError, match="not approved"):
        platform.execute_approved_ticket(ticket_id)

    # Approve ticket and execute
    platform.review_gate.approve_ticket(ticket_id, reviewer="lead_engineer")
    res_exec = platform.execute_approved_ticket(ticket_id)
    assert res_exec["status"] == "EXECUTED"
    assert res_exec["reviewer"] == "lead_engineer"

    # Audit log records all steps
    assert len(platform.execution_audit_log) >= 3
