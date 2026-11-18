import asyncio

from llm_control_plane.freehand_enterprise_agent_runtime import (
    ActionRiskTier,
    AgentReplayEvalHarness,
    EnterprisePolicyEnforcer,
    FreehandEnterpriseAgentRuntime,
    WriteAheadLogStateCheckpoint,
)


def test_wal_integrity_and_cryptographic_chain():
    async def _run():
        wal = WriteAheadLogStateCheckpoint("sess-procure-101")
        await wal.append_event("set_variable", {"key": "vendor_id", "value": "VEND-8842"})
        await wal.append_event("execute_action", {"action": "draft_po", "amount": 12500})
        await wal.append_event("set_variable", {"key": "status", "value": "AWAITING_MATCH"})

        assert wal.event_count == 3
        assert wal.verify_integrity() is True

        # Tamper with an event to ensure detection
        wal._events[1].payload["amount"] = 999999
        assert wal.verify_integrity() is False

    asyncio.run(_run())


def test_wal_idempotency_key_deduplication():
    async def _run():
        wal = WriteAheadLogStateCheckpoint("sess-idemp-202")
        ev1 = await wal.append_event(
            "execute_action", {"action": "create_payment", "amount": 5000}, idempotency_key="idemp-key-inv-9921"
        )
        # Duplicate call with identical key
        ev2 = await wal.append_event(
            "execute_action", {"action": "create_payment", "amount": 5000}, idempotency_key="idemp-key-inv-9921"
        )

        assert ev1.event_id == ev2.event_id
        assert wal.event_count == 1

    asyncio.run(_run())


def test_wal_replay_state_reconstruction():
    async def _run():
        wal = WriteAheadLogStateCheckpoint("sess-replay-303")
        await wal.append_event("set_variable", {"key": "department", "value": "SupplyChain"})
        await wal.append_event("set_variable", {"key": "approval_limit", "value": 25000})
        await wal.append_event("execute_action", {"action": "match_invoice", "match_confidence": 0.98})

        state = wal.replay_state()
        assert state["session_id"] == "sess-replay-303"
        assert state["variables"]["department"] == "SupplyChain"
        assert state["variables"]["approval_limit"] == 25000
        assert len(state["executed_actions"]) == 1
        assert state["executed_actions"][0]["action"] == "match_invoice"

    asyncio.run(_run())


def test_policy_enforcer_risk_tiers_and_escalation():
    policy = EnterprisePolicyEnforcer(max_auto_spend_usd=50000.0)

    # Read action -> Tier 1
    res_read = policy.evaluate_action("query_inventory", {"sku": "CHIP-A100"})
    assert res_read.allowed is True
    assert res_read.risk_tier == ActionRiskTier.TIER_1_AUTO_APPROVED
    assert res_read.requires_escalation is False

    # Moderate spend -> Tier 2
    res_spend = policy.evaluate_action("create_po", {"amount_usd": 15000.0})
    assert res_spend.allowed is True
    assert res_spend.risk_tier == ActionRiskTier.TIER_2_BOUNDED_MUTATION

    # Excessive spend -> Tier 3 Escalation
    res_high = policy.evaluate_action("create_po", {"amount_usd": 750000.0})
    assert res_high.allowed is False
    assert res_high.risk_tier == ActionRiskTier.TIER_3_HUMAN_CONSENSUS
    assert res_high.requires_escalation is True

    # Critical security action -> Blocked Tier 3
    res_bank = policy.evaluate_action("modify_bank_details", {"vendor_id": "V-123"})
    assert res_bank.allowed is False
    assert res_bank.risk_tier == ActionRiskTier.TIER_3_HUMAN_CONSENSUS


def test_policy_enforcer_circuit_breaker():
    policy = EnterprisePolicyEnforcer()
    for _ in range(3):
        policy.record_outcome(success=False)

    res = policy.evaluate_action("query_inventory", {})
    assert res.allowed is False
    assert res.requires_escalation is True
    assert "circuit breaker" in res.reason.lower()

    policy.reset_circuit_breaker()
    res_reset = policy.evaluate_action("query_inventory", {})
    assert res_reset.allowed is True


def test_freehand_runtime_execution_and_telemetry():
    async def _run():
        runtime = FreehandEnterpriseAgentRuntime(max_auto_spend_usd=30000.0)

        # Register mock ERP tool
        def mock_reconcile_invoice(params):
            return {"reconciled": True, "invoice_id": params["invoice_id"], "diff_usd": 0.0}

        runtime.register_tool("reconcile_invoice", mock_reconcile_invoice)

        # Execute allowed action
        res = await runtime.execute_step("reconcile_invoice", {"invoice_id": "INV-772", "amount_usd": 12000.0})
        assert res["success"] is True
        assert res["status"] == "COMPLETED"
        assert res["result"]["reconciled"] is True

        # Execute action exceeding threshold
        res_blocked = await runtime.execute_step("reconcile_invoice", {"invoice_id": "INV-889", "amount_usd": 95000.0})
        assert res_blocked["success"] is False
        assert res_blocked["status"] == "BLOCKED"
        assert res_blocked["requires_escalation"] is True

        # Verify telemetry traces
        assert len(runtime.trace_spans) == 2
        assert runtime.trace_spans[0].status == "SUCCESS"
        assert runtime.trace_spans[1].status == "BLOCKED"
        assert runtime.wal.verify_integrity() is True

    asyncio.run(_run())


def test_agent_replay_eval_harness():
    async def _run():
        golden_scenarios = [
            {"action": "query_inventory", "params": {"sku": "X1"}, "expected_success": True},
            {"action": "reconcile_po", "params": {"amount_usd": 4000.0}, "expected_success": True},
            {"action": "modify_bank_details", "params": {}, "expected_success": False},
            {"action": "reconcile_po", "params": {"amount_usd": 120000.0}, "expected_success": False},
        ]

        harness = AgentReplayEvalHarness(golden_scenarios)

        def runtime_factory():
            rt = FreehandEnterpriseAgentRuntime(max_auto_spend_usd=20000.0)
            rt.register_tool("query_inventory", lambda p: {"stock": 42})
            rt.register_tool("reconcile_po", lambda p: {"status": "ok"})
            return rt

        metrics = await harness.run_benchmark(runtime_factory)
        assert metrics["total_scenarios"] == 4
        assert metrics["successful_runs"] == 4
        assert metrics["completion_rate"] == 1.0
        assert metrics["policy_violations_prevented"] == 2
        assert metrics["benchmark_status"] == "PASS"

    asyncio.run(_run())
