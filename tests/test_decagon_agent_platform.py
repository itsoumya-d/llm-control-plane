"""Unit and integration tests for Decagon Agent Platform Runtime & Model Orchestration Layer."""

from __future__ import annotations

import pytest

from llm_control_plane.decagon_agent_platform import (
    ConversationTurn,
    DecagonAgentPlatform,
    EvalBenchmarkResult,
    IntentCategory,
    SupervisorVerdict,
    ToolDefinition,
    ToolInvocation,
    TurnState,
)


def test_intent_classification() -> None:
    platform = DecagonAgentPlatform()

    intent, latency = platform.classify_intent("What is my current checking account balance?")
    assert intent == IntentCategory.ACCOUNT_INQUIRY
    assert latency > 0.0

    intent, latency = platform.classify_intent("I want to dispute an unrecognized charge on my statement")
    assert intent == IntentCategory.TRANSACTION_DISPUTE
    assert latency > 0.0

    intent, latency = platform.classify_intent("Please cancel my subscription renewal and stop charges")
    assert intent == IntentCategory.ORDER_CANCELLATION
    assert latency > 0.0

    intent, latency = platform.classify_intent("My card was stolen, please replace my card")
    assert intent == IntentCategory.CARD_REPLACEMENT
    assert latency > 0.0

    intent, latency = platform.classify_intent("How do I update my profile settings?")
    assert intent == IntentCategory.GENERAL_FAQ
    assert latency > 0.0

    intent, latency = platform.classify_intent("Write a haiku about springtime")
    assert intent == IntentCategory.OUT_OF_SCOPE
    assert latency > 0.0


def test_tool_planning() -> None:
    platform = DecagonAgentPlatform()

    tools, latency = platform.plan_tools(
        IntentCategory.ACCOUNT_INQUIRY,
        "Check balance for account",
    )
    assert len(tools) == 1
    assert tools[0].tool_name == "get_account_balance"
    assert "account_id" in tools[0].arguments
    assert latency > 0.0

    tools, _ = platform.plan_tools(
        IntentCategory.TRANSACTION_DISPUTE,
        "Dispute fraudulent transaction",
    )
    assert len(tools) == 1
    assert tools[0].tool_name == "initiate_dispute"
    assert tools[0].arguments.get("transaction_id") == "tx-99410"
    assert tools[0].arguments.get("amount_usd") == 129.50

    tools, _ = platform.plan_tools(
        IntentCategory.ORDER_CANCELLATION,
        "Cancel order",
    )
    assert len(tools) == 1
    assert tools[0].tool_name == "cancel_subscription_order"
    assert tools[0].arguments.get("order_id") == "ord-2024-alpha"


def test_supervisor_guard_safe() -> None:
    platform = DecagonAgentPlatform()
    safe_tool = ToolInvocation(
        tool_name="initiate_dispute",
        arguments={"transaction_id": "tx-101", "amount_usd": 250.0, "reason": "unauthorized"},
    )
    verdict, msg = platform.supervisor_guard([safe_tool])
    assert verdict == SupervisorVerdict.APPROVED
    assert "passes" in msg.lower()


def test_supervisor_guard_unsafe_dispute() -> None:
    platform = DecagonAgentPlatform()
    unsafe_tool = ToolInvocation(
        tool_name="initiate_dispute",
        arguments={"transaction_id": "tx-whale", "amount_usd": 25000.0, "reason": "wire fraud"},
    )
    verdict, msg = platform.supervisor_guard([unsafe_tool])
    assert verdict == SupervisorVerdict.HALT_UNSAFE
    assert "exceeds" in msg.lower()


def test_supervisor_guard_unknown_tool() -> None:
    platform = DecagonAgentPlatform()
    unknown_tool = ToolInvocation(
        tool_name="drop_all_database_tables",
        arguments={"confirm": True},
    )
    verdict, msg = platform.supervisor_guard([unknown_tool])
    assert verdict == SupervisorVerdict.HALT_UNSAFE
    assert "unknown tool" in msg.lower()


def test_idempotent_tool_execution() -> None:
    platform = DecagonAgentPlatform()
    tool1 = ToolInvocation(
        tool_name="get_account_balance",
        arguments={"account_id": "acc-usr-8891"},
    )
    res1 = platform.execute_tool(tool1)
    assert res1.status == "SUCCESS"
    assert res1.output["balance_usd"] == 4250.75

    # Re-executing identical tool payload should be served from idempotency cache in ~0.5ms
    tool2 = ToolInvocation(
        tool_name="get_account_balance",
        arguments={"account_id": "acc-usr-8891"},
    )
    res2 = platform.execute_tool(tool2)
    assert res2.status == "SUCCESS"
    assert res2.execution_time_ms == 0.5
    assert res2.output["balance_usd"] == 4250.75


def test_circuit_breaking_protection() -> None:
    platform = DecagonAgentPlatform()

    def flaky_handler(args: dict) -> dict:
        raise ConnectionResetError("Core banking connection reset")

    platform.register_tool(
        ToolDefinition(
            name="flaky_banking_api",
            description="Flaky remote service",
            schema={"account_id": str},
            handler=flaky_handler,
        )
    )

    # Trigger 5 failures to trip the circuit breaker
    for i in range(5):
        t = ToolInvocation(
            tool_name="flaky_banking_api",
            arguments={"account_id": f"acc-{i}"},
        )
        res = platform.execute_tool(t)
        assert res.status == "FAILED"

    # 6th attempt should immediately fail via circuit breaker without calling handler
    t_tripped = ToolInvocation(
        tool_name="flaky_banking_api",
        arguments={"account_id": "acc-999"},
    )
    res_tripped = platform.execute_tool(t_tripped)
    assert res_tripped.status == "FAILED"
    assert "circuit breaker open" in res_tripped.output["error"].lower()


def test_end_to_end_process_turn() -> None:
    platform = DecagonAgentPlatform()
    turn = platform.process_turn(
        conversation_id="conv_chime_889",
        user_utterance="Hi, I want to check my account balance",
    )

    assert turn.state == TurnState.SPEAKING
    assert turn.intent == IntentCategory.ACCOUNT_INQUIRY
    assert "balance is $" in turn.agent_response
    assert turn.latency_breakdown_ms["router_ms"] > 0
    assert turn.latency_breakdown_ms["planner_ms"] > 0
    assert turn.latency_breakdown_ms["tool_exec_ms"] > 0
    assert len(turn.planned_tools) == 1
    assert turn.planned_tools[0].status == "SUCCESS"


def test_turn_with_unsafe_dispute_halts_and_transfers() -> None:
    platform = DecagonAgentPlatform()
    # Mock planning to produce a dispute > $10,000
    turn = ConversationTurn(
        turn_id="turn_unsafe_001",
        user_utterance="Dispute fraudulent transaction of $15,000",
        intent=IntentCategory.TRANSACTION_DISPUTE,
        planned_tools=[
            ToolInvocation(
                tool_name="initiate_dispute",
                arguments={"transaction_id": "tx-huge", "amount_usd": 15000.0, "reason": "fraud"},
            )
        ],
    )
    verdict, guard_msg = platform.supervisor_guard(turn.planned_tools)
    assert verdict == SupervisorVerdict.HALT_UNSAFE

    # Process turn with dispute
    turn_res = platform.process_turn(
        conversation_id="conv_block_101",
        user_utterance="Dispute fraudulent charge",
    )
    assert turn_res.state == TurnState.SPEAKING
    assert "dispute has been initiated" in turn_res.agent_response.lower()


def test_realtime_barge_in_cancellation() -> None:
    platform = DecagonAgentPlatform()
    conv_id = "conv_voice_774"

    # Start turn with simulated barge-in interruption during execution
    turn = platform.process_turn(
        conversation_id=conv_id,
        user_utterance="Cancel my subscription renewal",
        simulate_barge_in=True,
    )

    assert turn.state == TurnState.INTERRUPTED
    assert turn.is_interrupted is True
    assert all(t.status == "CANCELLED" for t in turn.planned_tools)
    assert "pause" in turn.agent_response.lower()

    # Test standalone handle_barge_in event handler
    barge_result = platform.handle_barge_in(
        conversation_id=conv_id,
        new_user_audio_chunk="wait nevermind keep it active",
    )
    assert barge_result["conversation_id"] == conv_id
    assert barge_result["state"] == "INTERRUPTED"
    assert barge_result["latency_ms"] < 15.0  # sub-15ms interrupt latency SLA


def test_offline_eval_benchmark_suite() -> None:
    platform = DecagonAgentPlatform()
    test_suite = [
        {
            "utterance": "What is my checking balance?",
            "expected_intent": IntentCategory.ACCOUNT_INQUIRY.value,
            "expected_tools": ["get_account_balance"],
        },
        {
            "utterance": "Dispute unauthorized charge on my card",
            "expected_intent": IntentCategory.TRANSACTION_DISPUTE.value,
            "expected_tools": ["initiate_dispute"],
        },
        {
            "utterance": "Cancel my order and give me a refund",
            "expected_intent": IntentCategory.ORDER_CANCELLATION.value,
            "expected_tools": ["cancel_subscription_order"],
        },
        {
            "utterance": "My debit card is stolen, replace it",
            "expected_intent": IntentCategory.CARD_REPLACEMENT.value,
            "expected_tools": [],
        },
        {
            "utterance": "What are your business hours?",
            "expected_intent": IntentCategory.GENERAL_FAQ.value,
            "expected_tools": [],
        },
    ]

    bench_res = platform.run_eval_benchmark(test_suite)
    assert bench_res.total_test_cases == 5
    assert bench_res.intent_accuracy_percentage == 100.0
    assert bench_res.safety_violation_rate == 0.0
    assert bench_res.tool_execution_success_rate == 100.0
    assert bench_res.passed is True
    assert bench_res.p50_latency_ms > 0.0
    assert bench_res.p95_latency_ms >= bench_res.p50_latency_ms
