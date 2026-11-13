"""Tests for Doppler secrets management and ephemeral token broker."""

import threading
import time

import pytest

from llm_control_plane import (
    ConnectorRegistry,
    DopplerConfig,
    DopplerHighPrivilegeApprovalRequiredError,
    DopplerLeaseExhaustedError,
    DopplerLeaseExpiredError,
    DopplerLeaseRevokedError,
    DopplerSecretNotFoundError,
    DopplerSecretsBroker,
    DopplerTenantIsolationViolationError,
    ToolCallRequest,
    register_doppler_mcp_tools,
)


def test_doppler_config_defaults() -> None:
    cfg = DopplerConfig()
    assert cfg.project == "ai-platform"
    assert cfg.environment == "production"
    assert cfg.default_lease_ttl_seconds == 300.0
    assert cfg.max_lease_ttl_seconds == 3600.0
    assert cfg.default_max_accesses == 5
    assert cfg.enforce_strict_redaction is True
    assert cfg.audit_hmac_secret == "doppler-audit-hmac-key-2026"


def test_secret_registration_and_lease_resolution() -> None:
    broker = DopplerSecretsBroker()
    broker.set_secret("OPENAI_API_KEY", "mock_key_val_1234567890abcdef")

    lease = broker.request_lease(
        tenant_id="tenant_alpha",
        secret_name="OPENAI_API_KEY",
        tool_name="agent_executor",
        ttl_seconds=60.0,
        max_access_count=3,
    )

    assert lease.lease_id.startswith("lease_")
    assert lease.secret_name == "OPENAI_API_KEY"
    assert lease.tenant_id == "tenant_alpha"
    assert lease.access_count == 0
    assert lease.is_valid() is True

    # Resolve secret
    secret_val = broker.resolve_secret(lease.lease_id, tenant_id="tenant_alpha")
    assert secret_val == "mock_key_val_1234567890abcdef"
    assert lease.access_count == 1

    masked = broker.mask_secret(secret_val)
    assert masked.startswith("mock_k")
    assert masked.endswith("cdef")


def test_secret_not_found_raises() -> None:
    broker = DopplerSecretsBroker()
    with pytest.raises(DopplerSecretNotFoundError) as exc:
        broker.request_lease(
            tenant_id="tenant_beta",
            secret_name="NON_EXISTENT_KEY",
            tool_name="test_tool",
        )
    assert "NON_EXISTENT_KEY" in str(exc.value)


def test_lease_expiry_enforcement() -> None:
    broker = DopplerSecretsBroker()
    broker.set_secret("STRIPE_API_KEY", "mock_dummy_billing_credentials_987654321")

    lease = broker.request_lease(
        tenant_id="tenant_payments",
        secret_name="STRIPE_API_KEY",
        tool_name="billing_agent",
        ttl_seconds=0.05,  # 50ms
    )

    time.sleep(0.06)
    assert lease.is_valid() is False

    with pytest.raises(DopplerLeaseExpiredError):
        broker.resolve_secret(lease.lease_id, tenant_id="tenant_payments")


def test_lease_quota_exhaustion() -> None:
    broker = DopplerSecretsBroker()
    broker.set_secret("DATABASE_URL", "postgresql://user:pass@db.internal:5432/core")

    lease = broker.request_lease(
        tenant_id="tenant_db",
        secret_name="DATABASE_URL",
        tool_name="migration_agent",
        max_access_count=2,
    )

    v1 = broker.resolve_secret(lease.lease_id, tenant_id="tenant_db")
    assert v1.startswith("postgresql://")
    v2 = broker.resolve_secret(lease.lease_id, tenant_id="tenant_db")
    assert v2 == v1

    # Third attempt exceeds quota
    with pytest.raises(DopplerLeaseExhaustedError):
        broker.resolve_secret(lease.lease_id, tenant_id="tenant_db")


def test_lease_revocation() -> None:
    broker = DopplerSecretsBroker()
    broker.set_secret("PROD_TOKEN", "mock_production_scoped_token_value_xyz")

    lease = broker.request_lease(
        tenant_id="tenant_ops",
        secret_name="PROD_TOKEN",
        tool_name="deployer",
    )

    revoked = broker.revoke_lease(lease.lease_id, tenant_id="tenant_ops")
    assert revoked is True
    assert lease.revoked is True
    assert lease.is_valid() is False

    with pytest.raises(DopplerLeaseRevokedError):
        broker.resolve_secret(lease.lease_id, tenant_id="tenant_ops")


def test_high_privilege_approval_guardrail() -> None:
    broker = DopplerSecretsBroker()
    broker.set_secret(
        "ROOT_SIGNING_KEY",
        "mock_ed25519_priv_key_highly_confidential",
        high_privilege=True,
    )

    # Missing approval token
    with pytest.raises(DopplerHighPrivilegeApprovalRequiredError):
        broker.request_lease(
            tenant_id="tenant_crypto",
            secret_name="ROOT_SIGNING_KEY",
            tool_name="key_signer",
        )

    # Invalid approval token
    with pytest.raises(DopplerHighPrivilegeApprovalRequiredError):
        broker.request_lease(
            tenant_id="tenant_crypto",
            secret_name="ROOT_SIGNING_KEY",
            tool_name="key_signer",
            approval_token="invalid_token_xyz",
        )

    # Valid human approval token
    lease = broker.request_lease(
        tenant_id="tenant_crypto",
        secret_name="ROOT_SIGNING_KEY",
        tool_name="key_signer",
        approval_token="appr_sec_ops_approved_20261113",
    )
    assert lease.high_privilege is True
    resolved = broker.resolve_secret(lease.lease_id, tenant_id="tenant_crypto")
    assert resolved == "mock_ed25519_priv_key_highly_confidential"


def test_tenant_isolation_enforcement() -> None:
    broker = DopplerSecretsBroker()
    broker.set_secret(
        "TENANT_A_KEY",
        "secret_val_for_tenant_a_only",
        allowed_tenants={"tenant_a"},
    )

    # Tenant B denied lease
    with pytest.raises(DopplerTenantIsolationViolationError):
        broker.request_lease(
            tenant_id="tenant_b",
            secret_name="TENANT_A_KEY",
            tool_name="agent_b",
        )

    # Tenant A succeeds
    lease_a = broker.request_lease(
        tenant_id="tenant_a",
        secret_name="TENANT_A_KEY",
        tool_name="agent_a",
    )

    # Tenant B cannot resolve Tenant A's lease
    with pytest.raises(DopplerTenantIsolationViolationError):
        broker.resolve_secret(lease_a.lease_id, tenant_id="tenant_b")

    # Tenant B cannot revoke Tenant A's lease
    with pytest.raises(DopplerTenantIsolationViolationError):
        broker.revoke_lease(lease_a.lease_id, tenant_id="tenant_b")


def test_prompt_trace_redaction() -> None:
    broker = DopplerSecretsBroker()
    raw_secret = "mock_doppler_confidential_token_998877"
    broker.set_secret("INTEGRATION_TOKEN", raw_secret)

    trace = f"Agent prompt: Connecting to Doppler with {raw_secret} to fetch records."
    redacted = broker.redact_trace(trace)

    assert raw_secret not in redacted
    assert "[DOPPLER_SECRET_REDACTED:INTEGRATION_TOKEN:" in redacted


def test_cryptographic_audit_trail() -> None:
    broker = DopplerSecretsBroker()
    broker.set_secret("AUDIT_TEST_KEY", "sensitive_audit_secret_data_332211")

    lease = broker.request_lease(
        tenant_id="tenant_compliance",
        secret_name="AUDIT_TEST_KEY",
        tool_name="compliance_agent",
    )
    broker.resolve_secret(lease.lease_id, tenant_id="tenant_compliance")
    broker.revoke_lease(lease.lease_id, tenant_id="tenant_compliance")

    audit_events = broker.get_audit_log(tenant_id="tenant_compliance")
    assert len(audit_events) == 3
    assert [ev.event_type for ev in audit_events] == [
        "lease_granted",
        "secret_accessed",
        "lease_revoked",
    ]

    for ev in audit_events:
        assert broker.verify_audit_event(ev) is True


def test_prune_expired_leases() -> None:
    broker = DopplerSecretsBroker()
    broker.set_secret("PRUNE_KEY", "prune_test_secret_val")

    l1 = broker.request_lease("t1", "PRUNE_KEY", "tool1", ttl_seconds=0.01)
    l2 = broker.request_lease("t1", "PRUNE_KEY", "tool2", ttl_seconds=300.0)

    time.sleep(0.02)
    pruned = broker.prune_expired_leases()
    assert pruned == 1
    assert l1.lease_id not in broker._leases
    assert l2.lease_id in broker._leases


def test_mcp_connector_tools_integration() -> None:
    registry = ConnectorRegistry()
    broker = DopplerSecretsBroker()
    broker.set_secret("CLAUDE_API_KEY", "mock_claude_token_spec_9911")

    register_doppler_mcp_tools(registry, broker)

    # 1. doppler_request_secret_lease
    req_lease = ToolCallRequest(
        tool_name="doppler_request_secret_lease",
        arguments={"secret_name": "CLAUDE_API_KEY", "ttl_seconds": 120.0, "tenant_id": "tenant_mcp"},
        tenant_id="tenant_mcp",
    )
    res_lease = registry.execute(req_lease)
    assert res_lease.status == "executed"
    lease_id = res_lease.output["lease_id"]
    assert lease_id.startswith("lease_")

    # 2. doppler_resolve_secret
    req_resolve = ToolCallRequest(
        tool_name="doppler_resolve_secret",
        arguments={"lease_id": lease_id, "tenant_id": "tenant_mcp"},
        tenant_id="tenant_mcp",
    )
    res_resolve = registry.execute(req_resolve)
    assert res_resolve.status == "executed"
    assert res_resolve.output["secret_value"] == "mock_claude_token_spec_9911"
    assert res_resolve.output["masked_display"].startswith("mock_c")

    # 3. doppler_audit_secrets
    req_audit = ToolCallRequest(
        tool_name="doppler_audit_secrets",
        arguments={"tenant_id": "tenant_mcp"},
        tenant_id="tenant_mcp",
    )
    res_audit = registry.execute(req_audit)
    assert res_audit.status == "executed"
    assert res_audit.output["event_count"] >= 2
    assert res_audit.output["events"][0]["verified"] is True

    # 4. doppler_revoke_secret_lease
    req_revoke = ToolCallRequest(
        tool_name="doppler_revoke_secret_lease",
        arguments={"lease_id": lease_id, "tenant_id": "tenant_mcp"},
        tenant_id="tenant_mcp",
    )
    res_revoke = registry.execute(req_revoke)
    assert res_revoke.status == "executed"
    assert res_revoke.output["revoked"] is True


def test_concurrency_thread_safety() -> None:
    broker = DopplerSecretsBroker()
    broker.set_secret("CONCURRENT_KEY", "multi_threaded_secret_value_xyz")

    lease = broker.request_lease(
        tenant_id="tenant_thread",
        secret_name="CONCURRENT_KEY",
        tool_name="worker_thread",
        max_access_count=100,
    )

    errors = []

    def worker() -> None:
        try:
            val = broker.resolve_secret(lease.lease_id, tenant_id="tenant_thread")
            assert val == "multi_threaded_secret_value_xyz"
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(25)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0
    assert lease.access_count == 25
