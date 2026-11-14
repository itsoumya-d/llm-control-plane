"""Comprehensive test suite for Deloitte Trustworthy AI™ Governance & Model Risk Framework."""

import pytest

from llm_control_plane.deloitte_trustworthy_ai import (
    AssuranceVerdict,
    DeloitteTrustworthyAIEngine,
    ModelEvaluationTarget,
    RiskClassification,
    TrustDimension,
)


@pytest.fixture
def target_financial():
    return ModelEvaluationTarget(
        model_id="fin-gpt-4o-risk",
        model_name="Wealth Advisory Client Copilot",
        version="v2.4.0",
        domain="Financial Advisory",
        intended_use="Advisory portfolio generation for retail investors",
        risk_tier=RiskClassification.HIGH,
        has_human_oversight=True,
        data_retention_days=60,
    )


@pytest.fixture
def target_low_risk():
    return ModelEvaluationTarget(
        model_id="doc-summarizer-internal",
        model_name="Internal Knowledge Summarizer",
        version="v1.0.1",
        domain="Internal Operations",
        intended_use="Summarizing public conference transcripts",
        risk_tier=RiskClassification.LOW,
        has_human_oversight=False,
    )


def test_fairness_evaluation_ideal(target_financial):
    engine = DeloitteTrustworthyAIEngine()
    score = engine.evaluate_fairness(
        target=target_financial,
        disparity_ratio=0.98,
        demographic_parity_diff=0.02,
    )
    assert score.passed is True
    assert score.score >= 95.0
    assert len(score.findings) == 0
    assert len(score.remediations) == 0


def test_fairness_evaluation_breach(target_financial):
    engine = DeloitteTrustworthyAIEngine()
    # Breaching four-fifths rule
    score = engine.evaluate_fairness(
        target=target_financial,
        disparity_ratio=0.65,
        demographic_parity_diff=0.18,
    )
    assert score.passed is False
    assert any("four-fifths" in f for f in score.findings)
    assert any("debiasing" in r for r in score.remediations)


def test_robustness_evaluation(target_financial):
    engine = DeloitteTrustworthyAIEngine()
    score = engine.evaluate_robustness(
        target=target_financial,
        hallucination_rate=0.01,
        adversarial_stability=0.96,
        out_of_distribution_f1=0.94,
    )
    assert score.passed is True
    assert score.score >= 90.0

    # High hallucination failure
    failing_score = engine.evaluate_robustness(
        target=target_financial,
        hallucination_rate=0.12,
        adversarial_stability=0.70,
        out_of_distribution_f1=0.60,
    )
    assert failing_score.passed is False
    assert any("tolerance" in f for f in failing_score.findings)


def test_explainability_evaluation(target_financial):
    engine = DeloitteTrustworthyAIEngine()
    score = engine.evaluate_explainability(
        target=target_financial,
        has_model_card=True,
        feature_importance_coverage=0.95,
        decision_traceability=True,
    )
    assert score.passed is True
    assert score.score == 97.5

    # Missing model card and traceability
    failing_score = engine.evaluate_explainability(
        target=target_financial,
        has_model_card=False,
        feature_importance_coverage=0.40,
        decision_traceability=False,
    )
    assert failing_score.passed is False
    assert any("Model Card" in f for f in failing_score.findings)


def test_safety_security_evaluation(target_financial):
    engine = DeloitteTrustworthyAIEngine()
    score = engine.evaluate_safety_security(
        target=target_financial,
        jailbreak_defense_rate=99.0,
        prompt_injection_blocked=98.5,
        malicious_code_detection_rate=99.5,
    )
    assert score.passed is True
    assert score.score >= 95.0

    # Vulnerable model
    vuln_score = engine.evaluate_safety_security(
        target=target_financial,
        jailbreak_defense_rate=80.0,
        prompt_injection_blocked=75.0,
        malicious_code_detection_rate=80.0,
    )
    assert vuln_score.passed is False
    assert len(vuln_score.findings) >= 2


def test_governance_evaluation(target_financial):
    engine = DeloitteTrustworthyAIEngine()
    score = engine.evaluate_governance(
        target=target_financial,
        designated_owner=True,
        has_incident_response_playbook=True,
        audit_trail_immutable=True,
    )
    assert score.passed is True
    assert score.score == 100.0

    # Unregulated deployment
    unreg_target = ModelEvaluationTarget(
        model_id="shadow-model",
        model_name="Shadow GenAI Bot",
        version="v0.1",
        domain="HR",
        intended_use="Autonomous screening",
        risk_tier=RiskClassification.HIGH,
        has_human_oversight=False,
    )
    score_unreg = engine.evaluate_governance(
        target=unreg_target,
        designated_owner=False,
        has_incident_response_playbook=False,
        audit_trail_immutable=False,
    )
    assert score_unreg.passed is False
    assert score_unreg.score == 0.0
    assert len(score_unreg.findings) == 4


def test_privacy_evaluation(target_financial):
    engine = DeloitteTrustworthyAIEngine()
    score = engine.evaluate_privacy(
        target=target_financial,
        pii_leakage_rate=0.001,
        compliance_with_privacy_act=True,
        anonymization_present=True,
    )
    assert score.passed is True

    breach_score = engine.evaluate_privacy(
        target=target_financial,
        pii_leakage_rate=0.04,
        compliance_with_privacy_act=False,
        anonymization_present=False,
    )
    assert breach_score.passed is False
    assert any("Privacy Act 2020" in f for f in breach_score.findings)


def test_full_audit_lifecycle_passed(target_financial):
    engine = DeloitteTrustworthyAIEngine()
    metrics = {
        "disparity_ratio": 0.96,
        "demographic_parity_diff": 0.03,
        "hallucination_rate": 0.015,
        "adversarial_stability": 0.95,
        "out_of_distribution_f1": 0.92,
        "has_model_card": True,
        "feature_importance_coverage": 0.90,
        "decision_traceability": True,
        "jailbreak_defense_rate": 98.0,
        "prompt_injection_blocked": 97.5,
        "malicious_code_detection_rate": 99.0,
        "designated_owner": True,
        "has_incident_response_playbook": True,
        "audit_trail_immutable": True,
        "pii_leakage_rate": 0.0,
        "compliance_with_privacy_act": True,
        "anonymization_present": True,
    }
    report = engine.run_full_audit(target_financial, metrics)
    assert report.verdict == AssuranceVerdict.PASSED
    assert report.overall_trust_score >= 90.0
    assert report.report_id.startswith("DELOITTE-TAI-")
    assert len(report.audit_hash) == 64
    assert "Deloitte Trustworthy AI Framework" in report.standards_aligned
    assert "ISO/IEC 42001:2023" in report.standards_aligned
    assert len(report.executive_summary) > 20


def test_full_audit_conditional_and_failure(target_financial):
    engine = DeloitteTrustworthyAIEngine()
    # 1 dimension fails (e.g. explainability)
    metrics_cond = {
        "disparity_ratio": 0.96,
        "demographic_parity_diff": 0.03,
        "hallucination_rate": 0.01,
        "adversarial_stability": 0.95,
        "out_of_distribution_f1": 0.92,
        "has_model_card": False,  # Missing
        "feature_importance_coverage": 0.50,
        "decision_traceability": True,
        "jailbreak_defense_rate": 98.0,
        "prompt_injection_blocked": 97.5,
        "malicious_code_detection_rate": 99.0,
        "designated_owner": True,
        "has_incident_response_playbook": True,
        "audit_trail_immutable": True,
        "pii_leakage_rate": 0.0,
        "compliance_with_privacy_act": True,
        "anonymization_present": True,
    }
    report_cond = engine.run_full_audit(target_financial, metrics_cond)
    assert report_cond.verdict in (AssuranceVerdict.CONDITIONAL_PASS, AssuranceVerdict.REMEDIATION_REQUIRED)

    # Catastrophic failure across multiple dimensions
    metrics_fail = {
        "disparity_ratio": 0.40,
        "demographic_parity_diff": 0.45,
        "hallucination_rate": 0.18,
        "adversarial_stability": 0.40,
        "out_of_distribution_f1": 0.30,
        "has_model_card": False,
        "feature_importance_coverage": 0.10,
        "decision_traceability": False,
        "jailbreak_defense_rate": 40.0,
        "prompt_injection_blocked": 30.0,
        "malicious_code_detection_rate": 50.0,
        "designated_owner": False,
        "has_incident_response_playbook": False,
        "audit_trail_immutable": False,
        "pii_leakage_rate": 0.08,
        "compliance_with_privacy_act": False,
        "anonymization_present": False,
    }
    report_fail = engine.run_full_audit(target_financial, metrics_fail)
    assert report_fail.verdict == AssuranceVerdict.FAIL
    assert report_fail.overall_trust_score < 40.0


def test_custom_thresholds_override(target_low_risk):
    engine = DeloitteTrustworthyAIEngine(
        custom_thresholds={TrustDimension.SAFE_SECURE: 99.0}
    )
    score = engine.evaluate_safety_security(
        target=target_low_risk,
        jailbreak_defense_rate=95.0,
        prompt_injection_blocked=95.0,
        malicious_code_detection_rate=95.0,
    )
    # Score is 95.0, custom threshold is 99.0 -> fails despite low risk tier
    assert score.score == 95.0
    assert score.threshold == 99.0
    assert score.passed is False
