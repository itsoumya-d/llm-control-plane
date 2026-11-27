"""Unit tests for New Zealand (NZ) Regulatory & AI Governance Country Pack."""

import pytest

from llm_control_plane.packs import evaluate_jurisdiction_policy, load_country_pack
from llm_control_plane.types import ChatMessage, CompletionRequest

URL_IPP5 = "https://www.privacy.org.nz/privacy-act-2020/privacy-principles/5-storage-and-security-of-personal-information/"
URL_IPP11 = (
    "https://www.privacy.org.nz/privacy-act-2020/privacy-principles/11-limits-on-disclosure-of-personal-information/"
)
URL_ALGO = (
    "https://www.data.govt.nz/toolkit/data-ethics/government-algorithm-transparency-and-accountability/algorithm-charter/"
)
URL_NZISM = "https://nzism.gcsb.govt.nz/"


def _make_request(
    use_case: str = "default",
    metadata: dict = None,
) -> CompletionRequest:
    return CompletionRequest(
        tenant_id="tenant-nz-enterprise",
        model="claude-3-5-sonnet",
        messages=[ChatMessage(role="user", content="Classify the incoming feedback.")],
        use_case=use_case,
        metadata=metadata or {},
        trace_id="trace-nz-001",
    )


def test_nz_pack_loader_valid():
    """Verify NZ pack rules load properly and all rules link to official government sources."""
    rules = load_country_pack("nz")
    assert len(rules) >= 4
    rule_ids = {r.id for r in rules}
    assert "NZ-PRIV-005" in rule_ids
    assert "NZ-PRIV-011" in rule_ids
    assert "NZ-ALGO-CHARTER" in rule_ids
    assert "NZ-NZISM-CLOUD" in rule_ids

    for rule in rules:
        assert rule.source_url.startswith("https://")
        assert any(
            domain in rule.source_url
            for domain in ["privacy.org.nz", "data.govt.nz", "nzism.gcsb.govt.nz"]
        )


def test_nz_privacy_act_ipp5_redaction_enforcement():
    """Test Privacy Act 2020 IPP 5: Storage and security of personal information."""
    req = _make_request(metadata={"contains_unredacted_pii": "true"})
    decision = evaluate_jurisdiction_policy(req, country="nz")
    assert not decision.allowed
    assert any("NZ Privacy Act 2020 IPP 5" in r for r in decision.reasons)
    assert any(URL_IPP5 in r for r in decision.reasons)


def test_nz_privacy_act_ipp11_cross_border_control():
    """Test Privacy Act 2020 IPP 11: Cross-border disclosure controls."""
    req = _make_request(metadata={"cross_border_unauthorized": "true"})
    decision = evaluate_jurisdiction_policy(req, country="nz")
    assert not decision.allowed
    assert any("NZ Privacy Act 2020 IPP 11" in r for r in decision.reasons)
    assert any(URL_IPP11 in r for r in decision.reasons)


def test_nz_algorithm_charter_human_oversight():
    """Test Algorithm Charter for Aotearoa NZ: Human oversight mandate for automated decisions."""
    req = _make_request(use_case="autonomous_decision", metadata={})
    decision = evaluate_jurisdiction_policy(req, country="nz")
    assert not decision.allowed
    assert any("Algorithm Charter for Aotearoa NZ" in r for r in decision.reasons)
    assert any(URL_ALGO in r for r in decision.reasons)

    # With verified human oversight, it passes
    req_compliant = _make_request(
        use_case="autonomous_decision",
        metadata={"human_oversight_verified": "true"},
    )
    decision_compliant = evaluate_jurisdiction_policy(req_compliant, country="nz")
    assert decision_compliant.allowed


def test_nz_nzism_cloud_classification():
    """Test NZISM Chapter 16: Classified data requires verified sovereign tenant."""
    req = _make_request(metadata={"data_classification": "RESTRICTED"})
    decision = evaluate_jurisdiction_policy(req, country="nz")
    assert not decision.allowed
    assert any("NZISM Cloud AI controls" in r for r in decision.reasons)
    assert any(URL_NZISM in r for r in decision.reasons)

    # With verified sovereign tenant, it passes
    req_compliant = _make_request(
        metadata={"data_classification": "RESTRICTED", "sovereign_tenant_verified": "true"},
    )
    decision_compliant = evaluate_jurisdiction_policy(req_compliant, country="nz")
    assert decision_compliant.allowed


def test_unknown_country_rejection():
    """Test that requesting an unknown jurisdiction raises an explicit ValueError without guessing."""
    req = _make_request()
    with pytest.raises(ValueError, match="unsupported jurisdiction"):
        evaluate_jurisdiction_policy(req, country="unknown_territory")
