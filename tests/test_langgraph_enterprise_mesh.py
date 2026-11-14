"""Comprehensive test suite for LangGraph Enterprise Mesh and Eval-Driven Development (EDD)."""

from __future__ import annotations

import time

from llm_control_plane.langgraph_enterprise_mesh import (
    AgentRole,
    ChannelType,
    CyclicGraphValidator,
    DurableCheckpointBroker,
    EnterpriseCompanyHarness,
    EvalDrivenBenchmarkHarness,
    HumanInTheLoopInterrupter,
    RunSpanType,
    StateSnapshot,
    TraceMiningEvaluator,
)


def test_channel_reducers_and_updates() -> None:
    broker = DurableCheckpointBroker()
    harness = EnterpriseCompanyHarness(checkpoint_broker=broker)

    harness.register_channel("messages", ChannelType.APPEND_ONLY_LIST, [])
    harness.register_channel("metadata", ChannelType.MERGE_DICT, {})
    harness.register_channel("counter", ChannelType.CHECKPOINTED_STATE, 0, reducer=lambda old, new: old + new)

    init_state = harness.initialize_state()
    assert init_state["messages"] == []
    assert init_state["metadata"] == {}
    assert init_state["counter"] == 0

    updates = {
        "messages": ["User: Hello world"],
        "metadata": {"tenant_id": "cust_apac_101", "region": "ap-southeast-1"},
        "counter": 5,
    }
    state_1 = harness.apply_channel_update(init_state, updates)
    assert state_1["messages"] == ["User: Hello world"]
    assert state_1["metadata"] == {"tenant_id": "cust_apac_101", "region": "ap-southeast-1"}
    assert state_1["counter"] == 5

    # Second update verifying list append, dict merge, and custom counter reducer
    updates_2 = {
        "messages": ["Agent: Ready to assist"],
        "metadata": {"session_active": True},
        "counter": 3,
    }
    state_2 = harness.apply_channel_update(state_1, updates_2)
    assert len(state_2["messages"]) == 2
    assert state_2["messages"][1] == "Agent: Ready to assist"
    assert state_2["metadata"] == {
        "tenant_id": "cust_apac_101",
        "region": "ap-southeast-1",
        "session_active": True,
    }
    assert state_2["counter"] == 8


def test_durable_checkpoint_broker_compaction() -> None:
    broker = DurableCheckpointBroker(max_history_per_thread=5)
    thread_id = "thread_apac_enterprise_1"

    chk_ids = []
    for step in range(1, 10):
        snap = broker.save_checkpoint(
            thread_id=thread_id,
            active_node=f"node_{step}",
            channels={"step": step, "status": "running"},
            step_index=step,
        )
        chk_ids.append(snap.checkpoint_id)

    # Verify compaction capped history at max_history (5)
    history = broker._checkpoints[thread_id]
    assert len(history) == 5
    # First checkpoint preserved
    assert history[0].step_index == 1
    # Latest checkpoint preserved
    assert history[-1].step_index == 9


def test_time_travel_thread_forking() -> None:
    broker = DurableCheckpointBroker()
    source_thread = "thread_prod_agent"
    target_thread = "thread_fork_debug"

    snap1 = broker.save_checkpoint(source_thread, "node_plan", {"plan": "execute SQL"}, step_index=1)
    broker.save_checkpoint(
        source_thread, "node_exec", {"plan": "execute SQL", "sql_result": [1, 2, 3]}, step_index=2
    )

    forked = broker.fork_thread(source_thread, snap1.checkpoint_id, target_thread)
    assert forked.thread_id == target_thread
    assert forked.step_index == 1
    assert forked.channels == {"plan": "execute SQL"}
    assert forked.parent_checkpoint_id == snap1.checkpoint_id

    # Verify source and forked threads are independent
    broker.save_checkpoint(target_thread, "node_alternate", {"plan": "execute vector search"}, step_index=2)
    assert len(broker._checkpoints[source_thread]) == 2
    assert len(broker._checkpoints[target_thread]) == 2


def test_cyclic_graph_validator() -> None:
    validator = CyclicGraphValidator(max_recursion_limit=20)
    nodes = ["supervisor", "researcher", "critic", "end"]
    edges = [("supervisor", "researcher"), ("researcher", "critic")]
    # Cycle between critic and researcher (if revisions needed), terminal to end
    conditional = {"critic": ["researcher", "end"]}
    terminals = ["end"]

    report = validator.validate_topology(nodes, edges, conditional, terminals)
    assert report["valid"] is True
    assert report["total_nodes"] == 4
    assert len(report["unreachable_from_terminal"]) == 0
    assert report["cycle_count"] >= 1
    assert any(set(cycle) == {"researcher", "critic"} for cycle in report["detected_cycles"])

    # Test an invalid graph with a dead-end unlinked node
    nodes_bad = ["supervisor", "orphan_worker", "end"]
    edges_bad = [("supervisor", "end")]
    report_bad = validator.validate_topology(nodes_bad, edges_bad, {}, terminals)
    assert report_bad["valid"] is False
    assert "orphan_worker" in report_bad["unreachable_from_terminal"]


def test_human_in_the_loop_interruption() -> None:
    interrupter = HumanInTheLoopInterrupter()
    # Require human approval if tool involves money transfer or DB drop
    interrupter.register_interrupt(
        "payment_executor",
        condition=lambda state: state.get("requires_compliance", False),
    )

    state_safe = {"requires_compliance": False, "amount": 10}
    assert interrupter.should_interrupt("payment_executor", state_safe) is False

    state_sensitive = {"requires_compliance": True, "amount": 50000}
    assert interrupter.should_interrupt("payment_executor", state_sensitive) is True

    snap = StateSnapshot("th_pay_1", "chk_1", 1, "payment_executor", state_sensitive)
    interrupter.pause_execution("th_pay_1", "payment_executor", "High-value transfer > $10k", snap)
    assert interrupter.is_paused("th_pay_1") is True

    # Resume with approval
    resumed = interrupter.resume_execution(
        "th_pay_1",
        approver="compliance_officer_alice",
        modifications={"compliance_approved": True, "approval_token": "TOK-9921"},
    )
    assert resumed["approver"] == "compliance_officer_alice"
    assert resumed["updated_channels"]["compliance_approved"] is True
    assert interrupter.is_paused("th_pay_1") is False


def test_trace_mining_evaluator_metrics() -> None:
    evaluator = TraceMiningEvaluator()
    t0 = time.time()

    evaluator.record_span(
        "supervisor", RunSpanType.AGENT, {"q": "search"}, {"res": "ok"}, t0, t0 + 0.05
    )
    evaluator.record_span(
        "retriever_tool", RunSpanType.TOOL, {"query": "vector"}, {"docs": [1]}, t0 + 0.05, t0 + 0.08, status="success"
    )
    evaluator.record_span(
        "db_tool",
        RunSpanType.TOOL,
        {"table": "logs"},
        {},
        t0 + 0.08,
        t0 + 0.10,
        status="error",
        error_message="timeout",
    )

    metrics = evaluator.compute_trajectory_metrics()
    assert metrics["total_spans"] == 3
    assert metrics["tool_spans_count"] == 2
    assert metrics["tool_precision"] == 0.5  # 1 success out of 2 tool calls
    assert metrics["error_rate"] == round(1 / 3, 4)
    assert metrics["latency_p50_ms"] >= 0.0

    ds = evaluator.export_langsmith_dataset()
    assert len(ds) == 3
    assert ds[0]["run_type"] == "agent"
    assert ds[1]["run_type"] == "tool"


def test_enterprise_company_harness_execution() -> None:
    broker = DurableCheckpointBroker()
    interrupter = HumanInTheLoopInterrupter()
    evaluator = TraceMiningEvaluator()
    harness = EnterpriseCompanyHarness(broker, interrupter, trace_evaluator=evaluator)

    harness.register_channel("query", ChannelType.LAST_WRITE_WINS, "")
    harness.register_channel("thought_trace", ChannelType.APPEND_ONLY_LIST, [])
    harness.register_channel("artifacts", ChannelType.MERGE_DICT, {})

    def supervisor_handler(state: dict) -> dict:
        return {
            "thought_trace": ["Supervisor: Analyzed query and routing to specialist."],
            "artifacts": {"routed_to": "deployed_specialist"},
        }

    harness.register_agent(
        role=AgentRole.SUPERVISOR,
        name="supervisor",
        handler=supervisor_handler,
        allowed_tools=["route_agent", "log_audit"],
    )

    thread_id = "apac_customer_session_7"
    res = harness.execute_turn(thread_id, "supervisor", step_index=1)
    assert res["status"] == "success"
    assert res["state"]["thought_trace"] == ["Supervisor: Analyzed query and routing to specialist."]
    assert res["state"]["artifacts"]["routed_to"] == "deployed_specialist"

    latest = broker.get_latest_checkpoint(thread_id)
    assert latest is not None
    assert latest.step_index == 1


def test_eval_driven_benchmark_harness() -> None:
    broker = DurableCheckpointBroker()
    harness = EnterpriseCompanyHarness(broker)

    def specialist_handler(state: dict) -> dict:
        return {
            "final_output": "Agent deployed successfully with durable checkpointing and zero memory leak.",
            "history": ["Specialist: Processed enterprise prompt."],
        }

    harness.register_agent(
        role=AgentRole.DEPLOYED_SPECIALIST,
        name="specialist",
        handler=specialist_handler,
        allowed_tools=["deploy_container", "verify_health"],
    )

    benchmark = EvalDrivenBenchmarkHarness(harness)
    dataset = [
        {
            "input": {"prompt": "Deploy agent"},
            "criteria": {"keyword": "durable checkpointing"},
            "start_node": "specialist",
        },
        {
            "input": {"prompt": "Scale cluster"},
            "criteria": {"keyword": "Agent deployed"},
            "start_node": "specialist",
        },
    ]

    def simple_judge(state: dict, criteria: dict) -> float:
        out = state.get("final_output", "")
        kw = criteria.get("keyword", "")
        return 1.0 if kw in out else 0.0

    report = benchmark.run_eval_dataset(dataset, simple_judge)
    assert report["total_eval_samples"] == 2
    assert report["passed_samples"] == 2
    assert report["pass_rate"] == 1.0
    assert report["benchmark_status"] == "PASS"
