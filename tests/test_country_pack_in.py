"""Unit tests for India (IN) Regulatory & AI Governance Country Pack."""

import pytest

from llm_control_plane.packs import evaluate_jurisdiction_policy, load_country_pack
from llm_control_plane.types import ChatMessage, CompletionRequest

URL_DPDP = "https://www.meity.gov.in/content/digital-personal-data-protection-act-2023"
URL_CERT = "https://www.cert-in.org.in/"
URL_RBI = "https://www.rbi.org.in/Scripts/BS_ViewMasDirections.aspx?id=12562"


def _make_request(
    use_case: str = "default",
    metadata: dict = None,
) -> CompletionRequest:
    return CompletionRequest(
        tenant_id="tenant-in-enterprise",
        model="claude-3-5-sonnet",
        messages=[ChatMessage(role="user", content="Process incoming customer data.")],
        use_case=use_case,
        metadata=metadata or {},
        trace_id="trace-in-001",
    )


def test_in_pack_loader_valid():
    """Verify India pack rules load properly and all rules link to official government sources."""
    rules = load_country_pack("in")
    assert len(rules) >= 4
    rule_ids = {r.id for r in rules}
    assert "IN-DPDP-S8" in rule_ids
    assert "IN-DPDP-S9" in rule_ids
    assert "IN-CERT-LOG" in rule_ids
    assert "IN-RBI-GOV" in rule_ids

    for rule in rules:
        assert rule.source_url.startswith("https://")
        assert any(
            domain in rule.source_url
            for domain in ["meity.gov.in", "cert-in.org.in", "rbi.org.in"]
        )


def test_in_dpdp_s8_pii_redaction_enforcement():
    """Test DPDP Act 2023 Section 8: Personal data security safeguards."""
    req = _make_request(metadata={"contains_unredacted_pii": "true"})
    decision = evaluate_jurisdiction_policy(req, country="in")
    assert not decision.allowed
    assert any("India DPDP Act 2023 Section 8" in r for r in decision.reasons)
    assert any(URL_DPDP in r for r in decision.reasons)

    # Clean request passes
    req_clean = _make_request(metadata={"contains_unredacted_pii": "false"})
    decision_clean = evaluate_jurisdiction_policy(req_clean, country="in")
    assert decision_clean.allowed


def test_in_dpdp_s9_child_protection():
    """Test DPDP Act 2023 Section 9: Processing of personal data of children."""
    req = _make_request(metadata={"subject_is_child": "true", "parental_consent_verified": "false"})
    decision = evaluate_jurisdiction_policy(req, country="in")
    assert not decision.allowed
    assert any("India DPDP Act 2023 Section 9" in r for r in decision.reasons)
    assert any(URL_DPDP in r for r in decision.reasons)

    # With verified parental consent, it passes
    req_compliant = _make_request(metadata={"subject_is_child": "true", "parental_consent_verified": "true"})
    decision_compliant = evaluate_jurisdiction_policy(req_compliant, country="in")
    assert decision_compliant.allowed


def test_in_cert_in_audit_log_retention():
    """Test CERT-In Directions: 180-day secure audit logging retention."""
    req = _make_request(metadata={"audit_logging_enabled": "false"})
    decision = evaluate_jurisdiction_policy(req, country="in")
    assert not decision.allowed
    assert any("CERT-In Cyber Security Directions" in r for r in decision.reasons)
    assert any(URL_CERT in r for r in decision.reasons)


def test_in_rbi_governance_model_explainability():
    """Test RBI IT Governance: Autonomous financial decisions require explainability."""
    req = _make_request(
        use_case="financial_autonomous_decision",
        metadata={"model_explainability_verified": "false"},
    )
    decision = evaluate_jurisdiction_policy(req, country="in")
    assert not decision.allowed
    assert any("RBI IT Governance Framework" in r for r in decision.reasons)
    assert any(URL_RBI in r for r in decision.reasons)

    req_compliant = _make_request(
        use_case="financial_autonomous_decision",
        metadata={"model_explainability_verified": "true"},
    )
    decision_compliant = evaluate_jurisdiction_policy(req_compliant, country="in")
    assert decision_compliant.allowed


def test_unknown_country_pack_raises_error():
    """Test that specifying an unknown country pack raises ValueError instead of guessing."""
    with pytest.raises(ValueError, match="unsupported jurisdiction"):
        load_country_pack("unknown_xyz")
