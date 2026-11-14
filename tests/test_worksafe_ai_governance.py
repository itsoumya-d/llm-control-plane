"""Unit tests for WorkSafe New Zealand AI Business Governance framework."""

import pytest
from llm_control_plane.worksafe_ai_governance import (
    AlgorithmCharterTier,
    GovernanceStatus,
    AIUseCaseProposal,
    WorkSafeAIBusinessGovernanceEvaluator,
)


def test_tier_1_low_risk_evaluation():
    evaluator = WorkSafeAIBusinessGovernanceEvaluator()
    proposal = AIUseCaseProposal(
        use_case_id="WKS-AI-001",
        title="Internal Technical Documentation Summarizer",
        business_unit="Digital Implementation",
        sponsor_name="Delivery Lead",
        intended_purpose="Summarize internal SOPs and guidance notes for engineering staff",
        data_sources=["Internal Wiki", "Public Standards"],
        involves_identifiable_individuals=False,
        involves_maori_data=False,
        influences_statutory_decisions=False,
        human_in_the_loop_mandatory=False,
        model_provider="Anthropic Claude 3.5 Sonnet",
    )

    eval_result = evaluator.evaluate_proposal(proposal)
    assert eval_result["charter_tier"] == AlgorithmCharterTier.TIER_1_LOW.value
    assert eval_result["status"] == GovernanceStatus.APPROVED.value
    assert len(eval_result["blocking_issues"]) == 0
    assert eval_result["te_tiriti_report"]["compliant"] is True


def test_tier_2_moderate_risk_pia_required():
    evaluator = WorkSafeAIBusinessGovernanceEvaluator()
    proposal = AIUseCaseProposal(
        use_case_id="WKS-AI-002",
        title="Inspector Voice-to-Text Incident Note Tagger",
        business_unit="Field Operations",
        sponsor_name="Chief Inspector",
        intended_purpose="Transcribe on-site audio notes from safety inspections",
        data_sources=["Field Audio Recordings"],
        involves_identifiable_individuals=True,
        involves_maori_data=False,
        influences_statutory_decisions=False,
        human_in_the_loop_mandatory=True,
        model_provider="Local Whisper",
        metadata={"privacy_impact_assessment_completed": False},
    )

    eval_result = evaluator.evaluate_proposal(proposal)
    assert eval_result["charter_tier"] == AlgorithmCharterTier.TIER_2_MODERATE.value
    assert eval_result["status"] == GovernanceStatus.REQUIRES_REMEDIATION.value
    assert any("Privacy Impact Assessment" in issue for issue in eval_result["blocking_issues"])

    # Remediate PIA
    proposal.metadata["privacy_impact_assessment_completed"] = True
    eval_result_remediated = evaluator.evaluate_proposal(proposal)
    assert eval_result_remediated["status"] == GovernanceStatus.APPROVED.value


def test_tier_3_high_risk_statutory_decision_support():
    evaluator = WorkSafeAIBusinessGovernanceEvaluator()
    proposal = AIUseCaseProposal(
        use_case_id="WKS-AI-003",
        title="Automated Hazard Penalty Recommendation Engine",
        business_unit="Enforcement & Compliance",
        sponsor_name="General Counsel",
        intended_purpose="Recommend statutory improvement notices and penalty ranges under HSWA 2015",
        data_sources=["Historical Breach Records", "Investigation Findings"],
        involves_identifiable_individuals=True,
        involves_maori_data=False,
        influences_statutory_decisions=True,
        human_in_the_loop_mandatory=False,  # Intentionally missing HITL
        model_provider="Azure OpenAI GPT-4o",
        metadata={
            "privacy_impact_assessment_completed": True,
            "explainability_framework_ready": False,
        },
    )

    eval_result = evaluator.evaluate_proposal(proposal)
    assert eval_result["charter_tier"] == AlgorithmCharterTier.TIER_3_HIGH.value
    assert eval_result["status"] == GovernanceStatus.REQUIRES_REMEDIATION.value
    assert any("human-in-the-loop" in issue for issue in eval_result["blocking_issues"])
    assert any("explainability" in issue for issue in eval_result["blocking_issues"])

    # Remediate by adding mandatory human validation and explainability audit
    proposal.human_in_the_loop_mandatory = True
    proposal.metadata["explainability_framework_ready"] = True

    eval_result_remediated = evaluator.evaluate_proposal(proposal)
    assert eval_result_remediated["status"] == GovernanceStatus.APPROVED.value


def test_te_tiriti_maori_data_sovereignty_safeguards():
    evaluator = WorkSafeAIBusinessGovernanceEvaluator()
    proposal = AIUseCaseProposal(
        use_case_id="WKS-AI-004",
        title="Forestry Worker Health Surveillance & Risk Forecaster",
        business_unit="Insights & Harm Prevention",
        sponsor_name="Head of Insights",
        intended_purpose="Predict respiratory and chronic physical strain risks across regional workforce cohorts",
        data_sources=["Regional Occupational Health Surveys"],
        involves_identifiable_individuals=False,
        involves_maori_data=True,
        influences_statutory_decisions=False,
        human_in_the_loop_mandatory=True,
        model_provider="In-House Scikit-Learn",
        metadata={
            "maori_governance_reviewed": False,
            "community_benefit_explicit": False,
            "demographic_disparity_tested": False,
        },
    )

    report = evaluator.audit_te_tiriti_safeguards(proposal)
    assert report.overall_compliant is False
    assert len(report.notes) == 3

    # Satisfy Māori Data Sovereignty principles
    proposal.metadata["maori_governance_reviewed"] = True
    proposal.metadata["community_benefit_explicit"] = True
    proposal.metadata["demographic_disparity_tested"] = True

    report_compliant = evaluator.audit_te_tiriti_safeguards(proposal)
    assert report_compliant.overall_compliant is True
    assert report_compliant.mana_whakahaere_passed is True
    assert report_compliant.mana_motuhake_passed is True
    assert report_compliant.mana_tiaki_passed is True


def test_governance_lifecycle_and_decision_ledger():
    evaluator = WorkSafeAIBusinessGovernanceEvaluator()
    proposal = AIUseCaseProposal(
        use_case_id="WKS-AI-005",
        title="High-Harm Industry Prioritization Engine",
        business_unit="Implementation Delivery",
        sponsor_name="Delivery Lead",
        intended_purpose="Prioritize target intervention sectors using ACC and incident telemetry",
        data_sources=["ACC Injury Data", "Notifiable Events Database"],
        involves_identifiable_individuals=False,
        involves_maori_data=False,
        influences_statutory_decisions=False,
        human_in_the_loop_mandatory=True,
        model_provider="AWS SageMaker",
    )

    reg_id = evaluator.register_use_case(proposal)
    assert reg_id == "WKS-AI-005"

    with pytest.raises(ValueError, match="already registered"):
        evaluator.register_use_case(proposal)

    decision = evaluator.record_decision(
        use_case_id="WKS-AI-005",
        status=GovernanceStatus.APPROVED,
        approver="Data & AI Governance Board",
        rationale="Compliant with Algorithm Charter Tier 1, clear operational boundaries, robust validation.",
    )

    assert decision.status == GovernanceStatus.APPROVED
    assert decision.tier == AlgorithmCharterTier.TIER_1_LOW
    assert len(decision.hash_signature) == 64

    # Verify enterprise report
    report = evaluator.export_summary_report()
    assert report["total_registered_use_cases"] == 1
    assert report["status_distribution"]["APPROVED"] == 1
    assert report["total_audit_events"] == 2
