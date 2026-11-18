"""Tests for Riplo Agentic Workflow Engine."""

from __future__ import annotations

import pytest

from llm_control_plane.riplo_agentic_workflow_engine import (
    AgenticWorkflowDAG,
    CitationType,
    ClaimNode,
    EvidenceChunk,
    EvidenceProvenanceGraph,
    PartnerReadyEvaluationHarness,
    WorkflowStep,
    WorkflowStepStatus,
)


def test_evidence_provenance_graph_basic() -> None:
    graph = EvidenceProvenanceGraph()
    chunk1 = EvidenceChunk(
        chunk_id="chunk-101",
        source_uri="s3://pe-data/deal_room/10k_filing.pdf",
        section_reference="Item 7 - MD&A, p. 42",
        content="ARR expanded by 34% YoY with net revenue retention of 118%.",
        token_count=18,
    )
    graph.register_chunk(chunk1)
    assert graph.get_chunk("chunk-101") is not None
    assert graph.get_chunk("chunk-101").content_hash is not None

    claim = ClaimNode(
        claim_id="claim-1",
        statement="Target company achieved 118% NRR and 34% ARR expansion.",
        citation_type=CitationType.STATISTICAL_METRIC,
        cited_chunk_ids=["chunk-101"],
    )
    graph.add_claim(claim)

    assert graph.verify_claim("claim-1") is True
    stats = graph.verify_all_claims()
    assert stats["total_claims"] == 1
    assert stats["verified_claims"] == 1
    assert stats["coverage_ratio"] == 1.0
    assert stats["is_partner_ready"] is True


def test_evidence_provenance_graph_unverified() -> None:
    graph = EvidenceProvenanceGraph()
    floating_claim = ClaimNode(
        claim_id="claim-2",
        statement="Management projects 45% EBITDA margins by FY2028.",
        citation_type=CitationType.FINANCIAL_TABLE,
        cited_chunk_ids=[],
    )
    graph.add_claim(floating_claim)
    assert graph.verify_claim("claim-2") is False

    stats = graph.verify_all_claims()
    assert stats["coverage_ratio"] == 0.0
    assert stats["is_partner_ready"] is False


def test_dag_cycle_detection() -> None:
    dag = AgenticWorkflowDAG()
    step_a = WorkflowStep(step_id="step-a", name="Market Sizing", task_description="Estimate TAM/SAM")
    step_b = WorkflowStep(
        step_id="step-b", name="Competitive Moat", task_description="Analyze pricing power", dependencies=["step-a"]
    )
    step_c = WorkflowStep(
        step_id="step-c", name="Synthesis", task_description="Final PE thesis", dependencies=["step-b"]
    )
    dag.add_step(step_a)
    dag.add_step(step_b)
    dag.add_step(step_c)

    order = dag.get_execution_order()
    assert order == ["step-a", "step-b", "step-c"]

    # Introduce a cyclical dependency
    step_cycle = WorkflowStep(
        step_id="step-cycle", name="Cyclic Step", task_description="Triggers cycle", dependencies=["step-c"]
    )
    dag.add_step(step_cycle)
    with pytest.raises(ValueError, match="Cycle detected"):
        step_a.dependencies.append("step-cycle")
        dag._validate_acyclic()


def test_dag_execution_and_token_budget() -> None:
    dag = AgenticWorkflowDAG(max_workflow_tokens=5000)
    chunk = EvidenceChunk(
        chunk_id="chk-1",
        source_uri="companies_house://filing_2026.pdf",
        section_reference="Balance Sheet",
        content="Cash and cash equivalents stand at £14.2M.",
        token_count=12,
    )
    dag.provenance_graph.register_chunk(chunk)

    step1 = WorkflowStep(step_id="s1", name="Financial Review", task_description="Inspect liquidity")
    dag.add_step(step1)

    def mock_exec(step: WorkflowStep):
        claim = ClaimNode(
            claim_id="c1",
            statement="The business holds £14.2M in liquid cash.",
            citation_type=CitationType.FINANCIAL_TABLE,
            cited_chunk_ids=["chk-1"],
        )
        return {"liquidity_status": "STRONG"}, 1200, [claim]

    res = dag.execute_step("s1", mock_exec)
    assert res["status"] == WorkflowStepStatus.COMPLETED.value
    assert res["tokens_consumed"] == 1200
    assert dag.provenance_graph.verify_claim("c1") is True


def test_partner_ready_evaluation_harness() -> None:
    graph = EvidenceProvenanceGraph()
    chunk1 = EvidenceChunk(
        chunk_id="c-pe-1",
        source_uri="call_transcript://expert_network_04.txt",
        section_reference="Transcript Line 140",
        content="Gross retention is above 94% across enterprise tier.",
        token_count=15,
    )
    chunk2 = EvidenceChunk(
        chunk_id="c-pe-2",
        source_uri="financial_model://dcf_v3.xlsx",
        section_reference="Tab: Unit Economics",
        content="LTV to CAC ratio is 4.8x on blended basis.",
        token_count=14,
    )
    graph.register_chunk(chunk1)
    graph.register_chunk(chunk2)

    claim1 = ClaimNode(
        claim_id="cl-1",
        statement="Gross enterprise retention sustained above 94%.",
        citation_type=CitationType.STATISTICAL_METRIC,
        cited_chunk_ids=["c-pe-1"],
    )
    claim2 = ClaimNode(
        claim_id="cl-2",
        statement="Blended LTV/CAC ratio is 4.8x.",
        citation_type=CitationType.FINANCIAL_TABLE,
        cited_chunk_ids=["c-pe-2"],
    )
    graph.add_claim(claim1)
    graph.add_claim(claim2)

    score = PartnerReadyEvaluationHarness.evaluate(graph, total_tokens_used=18500, target_token_budget=50000)
    assert score.passed is True
    assert score.composite_score >= 90.0
    assert score.citation_coverage == 1.0
    assert len(score.reasons) == 0
