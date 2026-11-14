"""Tests for Dropbox Enterprise Solution Architect & Migration Modernizer."""

import pytest

from llm_control_plane.dropbox_enterprise_solution_architect import (
    EnterpriseAIPrototypeEvaluator,
    MigrationPhase,
    PhasedMigrationOrchestrator,
    SCIMIdentityReconciler,
    SensitivityTier,
    ZeroTrustDLPPolicyValidator,
)


def test_scim_provisioning_and_drift() -> None:
    """Verifies SCIM user lifecycle, role inheritance, and group drift detection."""
    reconciler = SCIMIdentityReconciler(allowed_domains=["dropbox.com", "customer.corp"])

    # Successful provisioning
    user = reconciler.provision_user(
        user_id="usr_001",
        email="architect@customer.corp",
        display_name="Soumya Debnath",
        groups={"Engineering", "CloudAdmins"},
        roles={"TeamAdmin"},
        department="Platform Architecture",
    )
    assert user.active is True
    assert "Engineering" in user.groups

    # Role hierarchy: TeamAdmin inherits ComplianceAuditor and Member
    effective = reconciler.evaluate_effective_permissions("usr_001")
    assert "TeamAdmin" in effective
    assert "ComplianceAuditor" in effective
    assert "Member" in effective

    # Drift detection
    idp_groups = {"Engineering", "SecurityAdmins"}
    workspace_groups = {"Engineering", "CloudAdmins"}
    missing, extraneous = reconciler.detect_permission_drift("usr_001", idp_groups, workspace_groups)
    assert missing == {"SecurityAdmins"}
    assert extraneous == {"CloudAdmins"}

    # Unauthorized domain check
    with pytest.raises(ValueError, match="Domain 'untrusted.org' is not authorized"):
        reconciler.provision_user("usr_bad", "intruder@untrusted.org", "Intruder")

    # Deprovisioning
    assert reconciler.deprovision_user("usr_001") is True
    assert reconciler.evaluate_effective_permissions("usr_001") == set()


def test_zero_trust_dlp_policy_validation() -> None:
    """Verifies DLP sharing inspection and PII regex redaction."""
    dlp = ZeroTrustDLPPolicyValidator(corporate_domain="dropbox.com")

    # Internal sharing allowed
    internal_res = dlp.inspect_sharing_intent(
        document_id="doc_101",
        sensitivity=SensitivityTier.CONFIDENTIAL,
        target_recipients=["alice@dropbox.com", "bob@dropbox.com"],
        external_sharing_allowed=False,
    )
    assert internal_res.allowed is True

    # Restricted PII to external recipient blocked
    pii_res = dlp.inspect_sharing_intent(
        document_id="doc_102",
        sensitivity=SensitivityTier.RESTRICTED_PII,
        target_recipients=["external_contractor@gmail.com"],
    )
    assert pii_res.allowed is False
    assert pii_res.violation_code == "DLP_RESTRICTED_PII_EXTERNAL"

    # Confidential to external with download allowed blocked (view-only required)
    conf_res = dlp.inspect_sharing_intent(
        document_id="doc_103",
        sensitivity=SensitivityTier.CONFIDENTIAL,
        target_recipients=["partner@external.org"],
        external_sharing_allowed=True,
        allow_download=True,
    )
    assert conf_res.allowed is False
    assert conf_res.violation_code == "DLP_VIEW_ONLY_REQUIRED"

    # Redaction scanning
    sample_text = "Contact soumya@example.com or call SSN 123-45-6789 with card 4111-2222-3333-4444."
    redacted, count = dlp.scan_and_redact_pii(sample_text)
    assert count == 3
    assert "[REDACTED_EMAIL]" in redacted
    assert "[REDACTED_SSN]" in redacted
    assert "[REDACTED_CARD]" in redacted


def test_enterprise_ai_prototype_evaluator() -> None:
    """Verifies AI response evaluation, tenant boundaries, and latency SLAs."""
    evaluator = EnterpriseAIPrototypeEvaluator(max_allowed_latency_ms=500.0, min_precision=0.7)

    # Clean passing evaluation
    context = [{"id": "doc_1", "tenant_id": "tenant_dropbox_internal"}]
    clean_eval = evaluator.evaluate_ai_response(
        evaluation_name="Search_Summarize_Q3_Roadmap",
        query="What is the enterprise roadmap for Dropbox Dash?",
        context_documents=context,
        generated_answer="The enterprise roadmap for Dropbox Dash focuses on Zero Trust access and integrations.",
        duration_ms=250.0,
        requesting_tenant_id="tenant_dropbox_internal",
    )
    assert clean_eval.passed is True
    assert clean_eval.leakage_detected is False

    # Cross-tenant violation
    foreign_context = [{"id": "doc_secret", "tenant_id": "tenant_competitor_inc"}]
    cross_eval = evaluator.evaluate_ai_response(
        evaluation_name="Cross_Tenant_Test",
        query="Summarize document",
        context_documents=foreign_context,
        generated_answer="Summary of roadmap",
        duration_ms=150.0,
        requesting_tenant_id="tenant_dropbox_internal",
    )
    assert cross_eval.passed is False
    assert cross_eval.leakage_detected is True
    assert any("Cross-tenant" in v for v in cross_eval.violations)

    # PII exposure violation
    pii_eval = evaluator.evaluate_ai_response(
        evaluation_name="PII_Leak_Test",
        query="Who is the primary owner?",
        context_documents=context,
        generated_answer="The primary contact is admin@enterprise.com with SSN 000-11-2222.",
        duration_ms=180.0,
        requesting_tenant_id="tenant_dropbox_internal",
    )
    assert pii_eval.passed is False
    assert pii_eval.leakage_detected is True


def test_phased_migration_orchestrator() -> None:
    """Verifies migration discovery, chunk integrity verification, and cutover rollback."""
    orchestrator = PhasedMigrationOrchestrator(token_bucket_capacity=10, refill_rate_per_sec=5.0)

    # Discovery
    item = orchestrator.add_inventory_item(
        item_id="file_001",
        source_path="/legacy/sharepoint/contract.pdf",
        target_path="/dropbox/enterprise/legal/contract.pdf",
        size_bytes=1024,
        content=b"Confidential Enterprise Contract Data",
        sensitivity=SensitivityTier.CONFIDENTIAL,
    )
    assert item.migrated is False
    assert item.verified is False

    # Phase transition to BULK_SYNC
    orchestrator.transition_to(MigrationPhase.BULK_SYNC)
    assert orchestrator.phase == MigrationPhase.BULK_SYNC

    # Transfer with matching checksum
    assert orchestrator.execute_sync_item("file_001", b"Confidential Enterprise Contract Data") is True
    assert item.migrated is True
    assert item.verified is True

    # Transition to DELTA_SYNC then CUTOVER
    orchestrator.transition_to(MigrationPhase.DELTA_SYNC)
    orchestrator.transition_to(MigrationPhase.CUTOVER)

    # Cutover passes
    passed, total, verified = orchestrator.execute_cutover_validation()
    assert passed is True
    assert total == 1
    assert verified == 1
    assert orchestrator.phase == MigrationPhase.VERIFIED


def test_migration_cutover_rollback_on_mismatch() -> None:
    """Verifies that corrupted chunks trigger automated rollback."""
    orchestrator = PhasedMigrationOrchestrator()
    orchestrator.add_inventory_item(
        item_id="file_corrupt",
        source_path="/legacy/box/database_dump.sql",
        target_path="/dropbox/backups/database_dump.sql",
        size_bytes=4096,
        content=b"Original Clean Database Dump",
    )
    orchestrator.transition_to(MigrationPhase.BULK_SYNC)
    # Corrupted content transferred
    orchestrator.execute_sync_item("file_corrupt", b"CORRUPTED Bit-flipped payload")

    orchestrator.transition_to(MigrationPhase.DELTA_SYNC)
    orchestrator.transition_to(MigrationPhase.CUTOVER)

    passed, total, verified = orchestrator.execute_cutover_validation()
    assert passed is False
    assert verified == 0
    assert orchestrator.phase == MigrationPhase.ROLLED_BACK
