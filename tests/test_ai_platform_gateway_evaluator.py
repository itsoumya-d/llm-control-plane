"""
Tests for Adaptive AI Platform Gateway Router, Vector RAG Engine, and Regression Evaluator.
"""

import time

import pytest

from llm_control_plane.ai_platform_gateway_evaluator import (
    AdaptiveLLMGateway,
    CircuitState,
    Document,
    ModelSpec,
    RegressionEvaluator,
    RoutingRequest,
    TestCase,
    VectorKnowledgeEngine,
)


@pytest.fixture
def sample_gateway() -> AdaptiveLLMGateway:
    gw = AdaptiveLLMGateway()
    gw.register_model(
        ModelSpec(
            model_id="azure-gpt4o",
            provider="azure_openai",
            cost_per_1k_input=0.005,
            cost_per_1k_output=0.015,
            avg_latency_ms=450.0,
            max_context_tokens=128000,
            quality_tier=5,
        )
    )
    gw.register_model(
        ModelSpec(
            model_id="azure-gpt4o-mini",
            provider="azure_openai",
            cost_per_1k_input=0.00015,
            cost_per_1k_output=0.0006,
            avg_latency_ms=180.0,
            max_context_tokens=128000,
            quality_tier=3,
        )
    )
    gw.register_model(
        ModelSpec(
            model_id="self-hosted-llama3-70b",
            provider="self_hosted",
            cost_per_1k_input=0.0008,
            cost_per_1k_output=0.0012,
            avg_latency_ms=320.0,
            max_context_tokens=8192,
            quality_tier=4,
        )
    )
    return gw


def test_gateway_route_high_priority_picks_top_tier(sample_gateway: AdaptiveLLMGateway):
    req = RoutingRequest(
        prompt="Synthesize robotics telemetry telemetry error logs across swarm fleet.",
        max_tokens=256,
        priority="high",
    )
    decision = sample_gateway.route_request(req)
    assert decision.chosen_model == "azure-gpt4o"
    assert decision.provider == "azure_openai"
    assert decision.estimated_latency_ms == 450.0
    assert decision.circuit_status == "closed"


def test_gateway_route_low_priority_picks_cost_efficient(sample_gateway: AdaptiveLLMGateway):
    req = RoutingRequest(
        prompt="Format simple key-value tags.",
        max_tokens=100,
        priority="low",
    )
    decision = sample_gateway.route_request(req)
    assert decision.chosen_model == "azure-gpt4o-mini"
    assert decision.estimated_cost < 0.001


def test_gateway_circuit_breaker_failover(sample_gateway: AdaptiveLLMGateway):
    model_id = "azure-gpt4o"
    # Trip circuit breaker
    sample_gateway.record_failure(model_id)
    sample_gateway.record_failure(model_id)
    assert sample_gateway.check_circuit_breaker(model_id) == CircuitState.CLOSED
    sample_gateway.record_failure(model_id)
    assert sample_gateway.check_circuit_breaker(model_id) == CircuitState.OPEN

    # Request high priority - should now avoid azure-gpt4o and route to self-hosted or mini
    req = RoutingRequest(prompt="Urgent fleet mission dispatch", priority="high")
    decision = sample_gateway.route_request(req)
    assert decision.chosen_model != "azure-gpt4o"
    assert decision.chosen_model in ("self-hosted-llama3-70b", "azure-gpt4o-mini")

    # Verify recovery cooldown
    sample_gateway.models[model_id].last_failure_time = time.time() - 40.0
    assert sample_gateway.check_circuit_breaker(model_id) == CircuitState.HALF_OPEN
    sample_gateway.record_success(model_id)
    assert sample_gateway.check_circuit_breaker(model_id) == CircuitState.CLOSED


def test_gateway_execution_logging(sample_gateway: AdaptiveLLMGateway):
    req = RoutingRequest(prompt="Audit warehouse navigation coordinate drift", max_tokens=150)
    res = sample_gateway.execute_request(req)
    assert res.success is True
    assert res.total_tokens > 0
    assert len(sample_gateway.execution_logs) == 1


def test_vector_knowledge_engine_chunking_and_hybrid_search():
    engine = VectorKnowledgeEngine(vector_dim=16)
    doc1 = Document(
        doc_id="robotics_manual_01",
        content=(
            "The autonomous intralogistics system utilizes swarm intelligence "
            "and omnidirectional drive units to navigate complex industrial warehouses. "
            "Each mobile robot dynamically coordinates routes without central dispatch bottleneck."
        ),
        metadata={"category": "robotics", "source": "engineering_whitepaper"},
    )
    chunks = engine.ingest_document(doc1, chunk_size=15, chunk_overlap=5)
    assert len(chunks) >= 2
    assert chunks[0].doc_id == "robotics_manual_01"

    # Search
    results = engine.hybrid_search("swarm intelligence navigation", top_k=2)
    assert len(results) > 0
    assert results[0].score > 0.0
    assert "swarm" in results[0].text.lower()


def test_groundedness_verification():
    engine = VectorKnowledgeEngine()
    doc = Document(
        doc_id="policy_01",
        content="Fleet safety systems enforce optical lidar perimeter stops within 15ms of obstacle detection.",
    )
    engine.ingest_document(doc)
    retrieved = engine.hybrid_search("lidar perimeter obstacle stop", top_k=1)

    grounded_answer = "Optical lidar perimeter stops occur within 15ms of obstacle detection."
    is_grounded, ratio = engine.verify_groundedness(grounded_answer, retrieved)
    assert is_grounded is True
    assert ratio >= 0.65

    ungrounded_answer = "The spacecraft uses rocket thrusters to reach orbit velocity in interplanetary space."
    is_grounded_bad, ratio_bad = engine.verify_groundedness(ungrounded_answer, retrieved)
    assert is_grounded_bad is False
    assert ratio_bad < 0.30


def test_regression_evaluator_golden_dataset(sample_gateway: AdaptiveLLMGateway):
    evaluator = RegressionEvaluator(sample_gateway)
    evaluator.add_test_case(
        TestCase(
            test_id="TC-001",
            prompt="What navigation mechanism does the mobile robot rely on?",
            expected_facts=["swarm intelligence", "omnidirectional"],
            reference_output="Decentralized swarm intelligence and patented omnidirectional drives.",
            max_acceptable_latency_ms=800.0,
        )
    )
    evaluator.add_test_case(
        TestCase(
            test_id="TC-002",
            prompt="Explain the database identity model.",
            expected_facts=["least-privilege", "managed identity"],
            reference_output="Connects to PostgreSQL using managed identity with least-privilege role boundaries.",
            max_acceptable_latency_ms=800.0,
        )
    )

    report = evaluator.run_evaluations()
    assert report.total_tests == 2
    assert report.passed_tests == 2
    assert report.pass_rate == 1.0
    assert report.is_deployable is True
    assert len(report.regressions_detected) == 0
    assert report.p50_latency_ms > 0
