"""Tests for Inkeep Open-Source Agent Platform SDK & Runtime Orchestrator."""

import pytest

from llm_control_plane.inkeep_agent_platform_sdk import (
    AgentStreamEvent,
    AgentToolDefinition,
    DLPScanner,
    EventType,
    InkeepAgentPlatformSDK,
)


def test_tool_definition_validation_and_execution():
    def mock_search(args):
        return {"query": args["query"], "results": ["Doc 1", "Doc 2"]}

    tool = AgentToolDefinition(
        name="semantic_search",
        description="Search knowledge base for relevant documents",
        parameters_schema={"type": "object", "required": ["query"], "properties": {"query": {"type": "string"}}},
        handler=mock_search,
    )

    assert tool.validate_args({"query": "TypeScript SDK"})
    assert not tool.validate_args({"missing": "param"})

    result = tool.execute({"query": "TypeScript SDK"})
    assert result["query"] == "TypeScript SDK"
    assert len(result["results"]) == 2

    with pytest.raises(ValueError, match="Missing required arguments"):
        tool.execute({"wrong": "arg"})


def test_session_lifecycle_and_branching():
    sdk = InkeepAgentPlatformSDK()
    session = sdk.create_session(session_id="parent-001", metadata={"environment": "production"})

    session.add_message("user", "Hello Inkeep!", tokens=10)
    session.add_message("assistant", "Welcome to Inkeep Agent Platform.", tokens=20)

    assert session.tokens_consumed == 30
    assert len(session.history) == 2
    assert session.is_active

    # Branch session
    child = sdk.branch_session("parent-001", new_session_id="child-001")
    assert child.session_id == "child-001"
    assert child.parent_id == "parent-001"
    assert len(child.history) == 2
    assert child.metadata["environment"] == "production"

    child.add_message("user", "Exploring an alternate branch idea.", tokens=15)
    assert len(child.history) == 3
    assert len(session.history) == 2  # Parent remains untouched


def test_session_token_budget_limit():
    sdk = InkeepAgentPlatformSDK()
    session = sdk.create_session(token_budget_limit=100)

    session.add_message("user", "Short message", tokens=50)
    assert session.is_active

    session.add_message("assistant", "Very long payload", tokens=60)
    assert not session.is_active  # 110 > 100 limit

    with pytest.raises(RuntimeError, match="has exceeded its token limit"):
        sdk.dispatch_prompt(session.session_id, "Another turn")


def test_dlp_scanner_patterns_and_redaction():
    text_with_keys = (
        "Here is my secret sk-ant-api03-abcdef123456789012345 "
        "and SSN 000-12-3456 and token Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.xyz"
    )
    violations = DLPScanner.scan(text_with_keys)
    assert "anthropic_api_key" in violations
    assert "ssn" in violations
    assert "bearer_token" in violations

    redacted = DLPScanner.redact(text_with_keys)
    assert "sk-ant" not in redacted
    assert "000-12-3456" not in redacted
    assert "[REDACTED:ANTHROPIC_API_KEY]" in redacted
    assert "[REDACTED:SSN]" in redacted


def test_dispatch_prompt_with_tools_and_streaming_events():
    sdk = InkeepAgentPlatformSDK()
    session = sdk.create_session(session_id="stream-sess")
    assert session.session_id == "stream-sess"

    # Register tool
    def calc_add(args):
        return args["a"] + args["b"]

    tool = AgentToolDefinition(
        name="add_numbers",
        description="Add two numbers",
        parameters_schema={"type": "object", "required": ["a", "b"]},
        handler=calc_add,
    )
    sdk.register_tool(tool)

    emitted_events = []

    def on_event(event: AgentStreamEvent):
        emitted_events.append(event)

    result = sdk.dispatch_prompt(
        session_id="stream-sess",
        prompt="Please calculate 40 + 2",
        requested_tools=[{"name": "add_numbers", "args": {"a": 40, "b": 2}}],
        event_callback=on_event,
    )

    assert result["session_id"] == "stream-sess"
    assert len(result["tool_results"]) == 1
    assert result["tool_results"][0]["result"] == 42

    event_types = [e.event_type for e in emitted_events]
    assert EventType.TOKEN in event_types
    assert EventType.TOOL_START in event_types
    assert EventType.TOOL_RESULT in event_types
    assert EventType.COMPLETED in event_types


def test_human_in_the_loop_approval_flow():
    sdk = InkeepAgentPlatformSDK()
    session = sdk.create_session(session_id="approval-sess")
    assert session.session_id == "approval-sess"

    def destructive_deploy(args):
        return f"Deployed {args['target']} successfully"

    deploy_tool = AgentToolDefinition(
        name="deploy_to_production",
        description="Deploy software to production environment",
        parameters_schema={"type": "object", "required": ["target"]},
        handler=destructive_deploy,
        requires_human_approval=True,
    )
    sdk.register_tool(deploy_tool)

    events = []
    res = sdk.dispatch_prompt(
        session_id="approval-sess",
        prompt="Deploy candidate to production cluster",
        requested_tools=[{"name": "deploy_to_production", "args": {"target": "k8s-us-east-1"}}],
        event_callback=lambda e: events.append(e),
    )

    tool_res = res["tool_results"][0]
    assert tool_res["status"] == "pending_approval"
    approval_id = tool_res["approval_id"]

    # Check event
    approval_events = [e for e in events if e.event_type == EventType.HUMAN_APPROVAL_REQUIRED]
    assert len(approval_events) == 1
    assert approval_events[0].payload["approval_id"] == approval_id

    # Resolve approval (Approve)
    resolution = sdk.resolve_approval(approval_id, approved=True)
    assert resolution["status"] == "approved_and_executed"
    assert "Deployed k8s-us-east-1 successfully" in resolution["result"]


def test_agent_evaluation_scoring():
    sdk = InkeepAgentPlatformSDK()

    score = sdk.evaluate_execution(
        grounding_score=0.95,
        relevance_score=0.90,
        tool_accuracy_score=0.92,
        safety_score=1.00,
    )

    # 0.35*0.95 + 0.25*0.90 + 0.25*0.92 + 0.15*1.00 = 0.3325 + 0.225 + 0.230 + 0.15 = 0.9375 -> 0.938
    assert score.composite_score == 0.938
    assert score.passes_threshold

    low_score = sdk.evaluate_execution(
        grounding_score=0.40,
        relevance_score=0.50,
        tool_accuracy_score=0.60,
        safety_score=0.80,
    )
    assert not low_score.passes_threshold

    with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
        sdk.evaluate_execution(1.5, 0.9, 0.9, 1.0)
