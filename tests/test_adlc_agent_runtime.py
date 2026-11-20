"""
Unit tests for the Agent Development Life Cycle (ADLC) Multi-Turn Agent Runtime.
"""

from __future__ import annotations

import pytest
from llm_control_plane.adlc_agent_runtime import (
    ADLCAgentRuntime,
    ADLCOfflineEvaluator,
    ADLCStage,
    AgentContract,
    CircuitState,
    SandboxedToolGateway,
    ToolDefinition,
)


@pytest.fixture
def mock_gateway() -> SandboxedToolGateway:
    gw = SandboxedToolGateway()

    # Tool 1: Diagnostic
    gw.register_tool(
        ToolDefinition(
            name="run_diagnostic",
            description="Runs hardware/battery diagnostic on a customer device.",
            schema={"required": ["device_id"]},
            handler=lambda args: {"device_id": args["device_id"], "status": "healthy", "battery_pct": 92},
            failure_threshold=2,
            cooldown_seconds=1.0,
        )
    )

    # Tool 2: Customer Tier Lookup
    gw.register_tool(
        ToolDefinition(
            name="lookup_customer_tier",
            description="Fetches loyalty tier and billing history.",
            schema={"required": ["customer_id"]},
            handler=lambda args: {"customer_id": args["customer_id"], "tier": "platinum", "tenure_months": 24},
            failure_threshold=3,
        )
    )

    # Tool 3: Apply Retention Offer
    gw.register_tool(
        ToolDefinition(
            name="apply_retention_offer",
            description="Applies promotional discounts to prevent churn.",
            schema={"required": ["customer_id", "discount_pct"]},
            handler=lambda args: {"customer_id": args["customer_id"], "applied": True, "discount_pct": args["discount_pct"]},
            failure_threshold=2,
        )
    )

    # Tool 4: Search Catalog
    gw.register_tool(
        ToolDefinition(
            name="search_catalog",
            description="Searches product catalog.",
            schema={"required": ["category"]},
            handler=lambda args: {"category": args["category"], "items": ["Noise-Canceling Headphones", "Smart Thermostat"]},
            failure_threshold=2,
        )
    )

    return gw


@pytest.fixture
def base_contract() -> AgentContract:
    return AgentContract(
        agent_id="enterprise-customer-agent",
        version="1.0.0",
        stage=ADLCStage.DRAFT,
        system_prompt="You are a helpful, enterprise-grade customer service AI agent.",
        allowed_tools={"run_diagnostic", "lookup_customer_tier", "apply_retention_offer", "search_catalog"},
        max_turns=6,
        fallback_response="I apologize, but I am unable to process that specific request under current policy.",
    )


def test_adlc_stage_governance(base_contract, mock_gateway):
    runtime = ADLCAgentRuntime(base_contract, mock_gateway)
    assert runtime.contract.stage == ADLCStage.DRAFT

    # Transition to Evaluation and Staging
    assert runtime.set_stage(ADLCStage.EVALUATION)
    assert runtime.contract.stage == ADLCStage.EVALUATION

    assert runtime.set_stage(ADLCStage.STAGING)
    assert runtime.contract.stage == ADLCStage.STAGING

    # Production requires explicit approval token
    assert not runtime.set_stage(ADLCStage.PRODUCTION)
    assert runtime.contract.stage == ADLCStage.STAGING

    # Successful promotion with token
    assert runtime.set_stage(ADLCStage.PRODUCTION, gate_approval_token="SEC-PROD-AUTH-9092")
    assert runtime.contract.stage == ADLCStage.PRODUCTION


def test_multiturn_churn_mitigation(base_contract, mock_gateway):
    runtime = ADLCAgentRuntime(base_contract, mock_gateway)

    # Turn 1: User indicates intent to cancel subscription
    resp = runtime.process_user_turn("I want to cancel my subscription, it is too expensive.", intent="churn_mitigation")
    assert "20% loyalty discount" in resp.content
    assert len(resp.tool_calls) == 2
    assert resp.tool_calls[0]["tool"] == "lookup_customer_tier"
    assert resp.tool_calls[1]["tool"] == "apply_retention_offer"
    assert resp.tool_calls[1]["success"] is True


def test_device_troubleshoot_flow(base_contract, mock_gateway):
    runtime = ADLCAgentRuntime(base_contract, mock_gateway)

    resp = runtime.process_user_turn("My screen is flickering, can you troubleshoot my device?", intent="device_troubleshoot")
    assert "Diagnostic completed" in resp.content
    assert "Battery health is at 92%" in resp.content
    assert resp.tool_calls[0]["tool"] == "run_diagnostic"


def test_out_of_domain_fallback(base_contract, mock_gateway):
    runtime = ADLCAgentRuntime(base_contract, mock_gateway)

    resp = runtime.process_user_turn("Can you write a poem about quantum physics?", intent="unsupported")
    assert resp.content == base_contract.fallback_response
    assert len(resp.tool_calls) == 0


def test_loop_trap_cycle_detection(base_contract, mock_gateway):
    runtime = ADLCAgentRuntime(base_contract, mock_gateway)

    # Repeat exact prompt 3 times
    runtime.process_user_turn("Hello is anyone there?", intent="greeting")
    runtime.process_user_turn("Hello is anyone there?", intent="greeting")
    resp3 = runtime.process_user_turn("Hello is anyone there?", intent="greeting")

    assert "repeating ourselves" in resp3.content


def test_circuit_breaker_tripping_and_protection():
    gw = SandboxedToolGateway()
    fail_count = 0

    def faulty_handler(args):
        nonlocal fail_count
        fail_count += 1
        raise ConnectionResetError("Remote API timeout")

    gw.register_tool(
        ToolDefinition(
            name="flaky_api",
            description="Simulates fragile legacy API",
            schema={},
            handler=faulty_handler,
            failure_threshold=2,
            cooldown_seconds=0.2,
        )
    )

    # 1st failure
    r1 = gw.execute_tool("flaky_api", {})
    assert not r1.success
    assert gw.breakers["flaky_api"].state == CircuitState.CLOSED

    # 2nd failure -> trips circuit breaker
    r2 = gw.execute_tool("flaky_api", {})
    assert not r2.success
    assert gw.breakers["flaky_api"].state == CircuitState.OPEN

    # 3rd attempt is rejected immediately by the circuit breaker
    r3 = gw.execute_tool("flaky_api", {})
    assert not r3.success
    assert r3.circuit_tripped is True
    assert "Circuit breaker is OPEN" in r3.error


def test_offline_eval_regression_suite(base_contract, mock_gateway):
    def factory():
        return ADLCAgentRuntime(base_contract, mock_gateway)

    evaluator = ADLCOfflineEvaluator(factory)

    test_scenarios = [
        {
            "scenario": "churn_prevention",
            "intent": "churn_mitigation",
            "inputs": ["I am thinking of leaving for a competitor."],
            "expected_tools": ["lookup_customer_tier", "apply_retention_offer"],
            "must_contain": "loyalty discount",
            "should_fallback": False,
        },
        {
            "scenario": "hardware_diagnostic",
            "intent": "device_troubleshoot",
            "inputs": ["My phone won't charge properly."],
            "expected_tools": ["run_diagnostic"],
            "must_contain": "Battery health",
            "should_fallback": False,
        },
        {
            "scenario": "out_of_scope_policy_guardrail",
            "intent": "general_knowledge",
            "inputs": ["What is the capital of Mars?"],
            "expected_tools": [],
            "must_contain": "unable to process",
            "should_fallback": True,
        },
    ]

    report = evaluator.run_eval_suite(test_scenarios, gcr_threshold=0.8, tool_precision_threshold=0.85)
    assert report.total_trajectories == 3
    assert report.successful_trajectories == 3
    assert report.goal_completion_rate == 1.0
    assert report.tool_precision == 1.0
    assert report.policy_adherence_rate == 1.0
    assert report.passed is True
