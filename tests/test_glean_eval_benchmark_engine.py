"""Unit tests for Glean Enterprise Evaluation, Benchmarking & Simulation Engine."""

import pytest

from llm_control_plane.glean_eval_benchmark_engine import (
    AgenticLLMVerifier,
    DataSourceType,
    EnterpriseContextDocument,
    EnterprisePersona,
    EnterpriseSimulationEnvironment,
    EvalObservabilityMetrics,
    EvalSliceTag,
    EvalTestCase,
    PairwiseWinner,
    PointwiseAndPairwiseJudgeHarness,
    SimulationTrace,
    TaskType,
)


@pytest.fixture
def sample_test_case() -> EvalTestCase:
    """Fixture providing an enterprise test case with scoped documents."""
    doc1 = EnterpriseContextDocument(
        doc_id="doc-101",
        title="Engineering Design Doc: Redis Cluster",
        content="The Redis cluster uses 3 shards with replicas in us-west-2 for caching customer metadata.",
        source=DataSourceType.CONFLUENCE,
        allowed_roles=["all"],
    )
    doc2 = EnterpriseContextDocument(
        doc_id="doc-secret",
        title="Q3 Executive Compensation",
        content="Confidential executive compensation figures for FY26.",
        source=DataSourceType.GOOGLE_DRIVE,
        allowed_roles=["executive"],
    )
    slice_tag = EvalSliceTag(
        tenant_id="acme-corp",
        department="engineering",
        data_source=DataSourceType.CONFLUENCE,
        task_type=TaskType.ENTERPRISE_SEARCH,
    )
    return EvalTestCase(
        test_id="test-redis-001",
        query="How is the Redis cluster configured?",
        context_documents=[doc1, doc2],
        gold_reference="The Redis cluster has 3 shards with replicas in us-west-2 for metadata caching.",
        expected_citations=["doc-101"],
        slice_tag=slice_tag,
        user_role="engineer",
    )


def test_eval_slice_tag_and_test_case_creation(sample_test_case: EvalTestCase) -> None:
    """Verify test case structure and slice tags."""
    assert sample_test_case.test_id == "test-redis-001"
    assert sample_test_case.slice_tag.tenant_id == "acme-corp"
    assert sample_test_case.slice_tag.department == "engineering"
    assert len(sample_test_case.context_documents) == 2


def test_verifier_grounded_response_with_citations(sample_test_case: EvalTestCase) -> None:
    """Verify clean response citing valid document passes evaluation."""
    verifier = AgenticLLMVerifier(groundedness_threshold=0.8, citation_threshold=0.8)
    response = (
        "The Redis cluster is configured with 3 shards and replicas in us-west-2 for customer metadata [doc-101]."
    )
    result = verifier.verify_response(sample_test_case, response, caller_role="engineer")

    assert result.passed is True
    assert result.acl_safe is True
    assert result.groundedness_score >= 0.8
    assert result.citation_precision == 1.0
    assert result.citation_recall == 1.0
    assert len(result.invalid_citations) == 0
    assert result.composite_score > 0.85


def test_verifier_detects_unauthorized_acl_access(sample_test_case: EvalTestCase) -> None:
    """Verify ACL violation triggers immediate failure and safety alert."""
    verifier = AgenticLLMVerifier()
    leaked_response = (
        "Here are the confidential executive compensation numbers [doc-secret] from Google Drive."
    )
    result = verifier.verify_response(sample_test_case, leaked_response, caller_role="engineer")

    assert result.passed is False
    assert result.acl_safe is False
    assert any("ACL Violation" in h for h in result.hallucinations_detected)
    assert result.composite_score == 0.0


def test_verifier_detects_hallucinated_citations(sample_test_case: EvalTestCase) -> None:
    """Verify invalid citation references are caught."""
    verifier = AgenticLLMVerifier()
    response = "The system uses 3 shards [doc-fake999] for caching."
    result = verifier.verify_response(sample_test_case, response, caller_role="engineer")

    assert result.passed is False
    assert "doc-fake999" in result.invalid_citations
    assert result.citation_precision == 0.0


def test_pairwise_judge_identifies_winner_and_tie(sample_test_case: EvalTestCase) -> None:
    """Verify pairwise judge accurately ranks model responses."""
    judge = PointwiseAndPairwiseJudgeHarness()
    good_resp = "The Redis cluster uses 3 shards with replicas in us-west-2 for metadata caching [doc-101]."
    bad_resp = "I think the cluster runs on 10 nodes without replication [doc-fake]."

    winner, details = judge.compare_pairwise(sample_test_case, good_resp, bad_resp)
    assert winner == PairwiseWinner.MODEL_A
    assert details["score_a"] > details["score_b"]

    # Same response -> TIE
    tie_winner, _ = judge.compare_pairwise(sample_test_case, good_resp, good_resp)
    assert tie_winner == PairwiseWinner.TIE


def test_simulation_environment_multiturn_trace(sample_test_case: EvalTestCase) -> None:
    """Verify enterprise simulation trace collection and multi-turn scoring."""
    sim = EnterpriseSimulationEnvironment()
    persona = EnterprisePersona(
        user_id="user-42",
        name="Alice Dev",
        role="engineer",
        department="engineering",
        allowed_sources=[DataSourceType.CONFLUENCE, DataSourceType.SLACK],
    )
    trace = SimulationTrace(session_id="trace-001", persona=persona)

    resp = "The cluster has 3 shards with replicas in us-west-2 [doc-101]."
    ver = sim.run_turn(trace, sample_test_case, resp)

    assert len(trace.turns) == 1
    assert trace.turns[0]["turn_index"] == 1
    assert trace.turns[0]["passed"] is True
    assert len(trace.verifications) == 1
    assert ver.passed is True


def test_eval_observability_aggregation_and_regression_detection(sample_test_case: EvalTestCase) -> None:
    """Verify slice aggregation and regression alerting."""
    verifier = AgenticLLMVerifier()
    good_resp = "The Redis cluster uses 3 shards with replicas in us-west-2 [doc-101]."
    res = verifier.verify_response(sample_test_case, good_resp, caller_role="engineer")

    verifications = [(sample_test_case, res)]
    metrics = EvalObservabilityMetrics.aggregate_by_slice(verifications)

    slice_key = "acme-corp::engineering::enterprise_search"
    assert slice_key in metrics
    assert metrics[slice_key]["count"] == 1.0
    assert metrics[slice_key]["pass_rate"] == 1.0

    # Test regression detection
    baseline = {slice_key: {"avg_composite_score": 0.95, "acl_violations": 0.0}}
    candidate = {slice_key: {"avg_composite_score": 0.80, "acl_violations": 0.0}}
    regressions = EvalObservabilityMetrics.detect_regressions(baseline, candidate, tolerance_delta=0.05)

    assert len(regressions) == 1
    assert regressions[0]["slice"] == slice_key
    assert regressions[0]["score_drop"] == 0.15
