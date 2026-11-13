"""Tests for ClickHouse Cloud Platform Auth (ReBAC, SCIM 2.0, JWT, and audit telemetry)."""

import pytest

from llm_control_plane import (
    AccessDecisionTelemetry,
    ClickHouseReBACEngine,
    ClickHouseSCIMManager,
    ClickHouseTokenIssuer,
    Permission,
    PrincipalType,
    ResourceType,
)


def test_rebac_direct_permission_evaluation() -> None:
    """Verify direct ReBAC relationships evaluate correctly."""
    engine = ClickHouseReBACEngine()
    engine.add_relation(
        resource_type=ResourceType.DATABASE,
        resource_id="db_analytics",
        relation="editor",
        subject_type=PrincipalType.USER,
        subject_id="usr_alice",
    )

    decision_read = engine.check_permission("usr_alice", Permission.READ, ResourceType.DATABASE, "db_analytics")
    assert decision_read.allowed is True
    assert "Direct editor grant" in decision_read.decision_reason

    decision_write = engine.check_permission("usr_alice", Permission.WRITE, ResourceType.DATABASE, "db_analytics")
    assert decision_write.allowed is True

    decision_admin = engine.check_permission("usr_alice", Permission.ADMIN, ResourceType.DATABASE, "db_analytics")
    assert decision_admin.allowed is False


def test_rebac_hierarchical_inheritance() -> None:
    """Verify permissions inherit from Organization down to Table."""
    engine = ClickHouseReBACEngine()
    # Org -> Workspace -> Service -> Database -> Table
    engine.set_parent_resource(ResourceType.WORKSPACE, "ws_prod", ResourceType.ORGANIZATION, "org_acme")
    engine.set_parent_resource(ResourceType.SERVICE, "svc_ch_primary", ResourceType.WORKSPACE, "ws_prod")
    engine.set_parent_resource(ResourceType.DATABASE, "db_metrics", ResourceType.SERVICE, "svc_ch_primary")
    engine.set_parent_resource(ResourceType.TABLE, "tbl_events", ResourceType.DATABASE, "db_metrics")

    # Grant admin on organization to user Bob
    engine.add_relation(
        resource_type=ResourceType.ORGANIZATION,
        resource_id="org_acme",
        relation="admin",
        subject_type=PrincipalType.USER,
        subject_id="usr_bob",
    )

    # Bob should inherit permission to execute queries on the child table
    decision = engine.check_permission("usr_bob", Permission.EXECUTE_QUERY, ResourceType.TABLE, "tbl_events")
    assert decision.allowed is True
    assert "Inherited from parent" in decision.decision_reason

    # Charlie has no permissions
    decision_charlie = engine.check_permission(
        "usr_charlie", Permission.EXECUTE_QUERY, ResourceType.TABLE, "tbl_events"
    )
    assert decision_charlie.allowed is False


def test_scim_provisioning_and_immediate_deprovisioning() -> None:
    """Verify enterprise SCIM 2.0 lifecycle and immediate revocation."""
    rebac = ClickHouseReBACEngine()
    scim = ClickHouseSCIMManager(rebac)

    user = scim.provision_user("usr_dave", "ext_12345", "dave@example.com", "dave@example.com")
    assert user.active is True
    assert scim.is_user_active("usr_dave") is True

    rebac.add_relation(
        resource_type=ResourceType.WORKSPACE,
        resource_id="ws_finance",
        relation="viewer",
        subject_type=PrincipalType.USER,
        subject_id="usr_dave",
    )
    assert rebac.check_permission("usr_dave", Permission.READ, ResourceType.WORKSPACE, "ws_finance").allowed is True

    # Immediate deprovisioning
    scim.deprovision_user("usr_dave")
    assert scim.is_user_active("usr_dave") is False
    assert rebac.check_permission("usr_dave", Permission.READ, ResourceType.WORKSPACE, "ws_finance").allowed is False


def test_jwt_token_issuance_and_revocation() -> None:
    """Verify short-lived JWT token lifecycle and blacklist verification."""
    rebac = ClickHouseReBACEngine()
    scim = ClickHouseSCIMManager(rebac)
    scim.provision_user("usr_eve", "ext_67890", "eve@example.com", "eve@example.com")

    issuer = ClickHouseTokenIssuer(scim, secret_key="test-secret-key")
    token = issuer.issue_token("usr_eve", "org_acme", scopes=["read:db", "write:db"], ttl_seconds=300)

    assert token.principal_id == "usr_eve"
    assert not token.is_expired
    assert issuer.validate_token(token) is True

    # Immediate token revocation
    issuer.revoke_token(token.token_id)
    assert issuer.validate_token(token) is False

    # Deprovisioning user invalidates token validation
    token2 = issuer.issue_token("usr_eve", "org_acme", scopes=["read:db"], ttl_seconds=300)
    scim.deprovision_user("usr_eve")
    assert issuer.validate_token(token2) is False

    with pytest.raises(PermissionError):
        issuer.issue_token("usr_eve", "org_acme", scopes=["read:db"])


def test_access_decision_audit_telemetry_hash_chain() -> None:
    """Verify authorization telemetry records tamper-evident SHA-256 hash chains."""
    engine = ClickHouseReBACEngine()
    engine.add_relation(
        resource_type=ResourceType.DATABASE,
        resource_id="db_sec",
        relation="viewer",
        subject_type=PrincipalType.USER,
        subject_id="usr_frank",
    )

    telemetry = AccessDecisionTelemetry()
    dec1 = engine.check_permission("usr_frank", Permission.READ, ResourceType.DATABASE, "db_sec")
    dec2 = engine.check_permission("usr_frank", Permission.WRITE, ResourceType.DATABASE, "db_sec")

    telemetry.record_decision(dec1)
    telemetry.record_decision(dec2)

    assert len(telemetry.events) == 2
    assert telemetry.verify_integrity() is True

    # Tamper test
    telemetry.events[0]["allowed"] = not telemetry.events[0]["allowed"]
    assert telemetry.verify_integrity() is False
