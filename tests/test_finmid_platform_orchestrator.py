"""Unit tests for finmid Embedded Financial Platform Orchestrator."""

from llm_control_plane.finmid_platform_orchestrator import (
    ComplianceStandard,
    EnvironmentState,
    EphemeralStagingReconciler,
    FinmidPlatformOrchestrator,
    PlatformTier,
    TerraformDriftWatchdog,
)


def test_tenant_onboarding_enterprise_tier():
    orchestrator = FinmidPlatformOrchestrator()
    spec = orchestrator.onboard_tenant(
        tenant_id="wolt-eu",
        tenant_name="Wolt Delivery Platform",
        tier=PlatformTier.ENTERPRISE_CORE,
        allowed_standards=[ComplianceStandard.GDPR, ComplianceStandard.ECSP_REGULATION, ComplianceStandard.ISO27001]
    )

    assert spec.tenant_id == "wolt-eu"
    assert spec.tier == PlatformTier.ENTERPRISE_CORE
    assert spec.allocated_cpu_cores == 32.0
    assert spec.allocated_memory_gib == 128.0
    assert spec.namespace == "finmid-tenant-wolt-eu"
    assert ComplianceStandard.ISO27001 in spec.allowed_compliance_tags
    assert len(orchestrator.audit_log) == 1
    assert orchestrator.audit_log[0]["action"] == "TENANT_ONBOARDED"


def test_tenant_onboarding_sandbox_tier():
    orchestrator = FinmidPlatformOrchestrator()
    spec = orchestrator.onboard_tenant(
        tenant_id="b2b-saas-demo",
        tenant_name="Demo SaaS Integration",
        tier=PlatformTier.DEVELOPER_SANDBOX
    )

    assert spec.tier == PlatformTier.DEVELOPER_SANDBOX
    assert spec.allocated_cpu_cores == 4.0
    assert spec.allocated_memory_gib == 16.0


def test_ephemeral_preview_provision_and_teardown():
    reconciler = EphemeralStagingReconciler()
    preview = reconciler.provision_preview(
        pr_number=142,
        branch_name="feat/instant-credit-settlement",
        author="soumya_debnath"
    )

    assert preview.pr_number == 142
    assert preview.status == EnvironmentState.PROVISIONED
    assert "preview-pr-142-soumya-debnath" == preview.namespace
    assert preview.mock_webhook_endpoint.startswith("https://mock-webhooks.finmid-preview.internal/")

    fetched = reconciler.get_preview(142)
    assert fetched is not None
    assert fetched.branch_name == "feat/instant-credit-settlement"

    # Teardown
    ok = reconciler.teardown_preview(142)
    assert ok is True
    assert reconciler.get_preview(142) is None
    assert reconciler.teardown_preview(999) is False


def test_ephemeral_preview_ttl_expiration():
    reconciler = EphemeralStagingReconciler()
    preview = reconciler.provision_preview(
        pr_number=204,
        branch_name="fix/vat-invoice-rounding",
        author="platform_team"
    )

    # Force expiration
    preview.ttl_seconds = 0
    assert preview.is_expired is True

    pruned_count = reconciler.prune_expired()
    assert pruned_count == 1
    assert reconciler.get_preview(204) is None


def test_terraform_drift_watchdog_detection_and_reconciliation():
    watchdog = TerraformDriftWatchdog()
    declared_config = {
        "cidr_block": "10.100.0.0/16",
        "enable_dns_hostnames": True,
        "tags": {"Environment": "Production", "Owner": "Platform"}
    }

    state = watchdog.register_resource(
        resource_id="vpc-finmid-prod-eu-central",
        resource_type="aws_vpc",
        declared_config=declared_config
    )

    assert state.has_drift is False
    assert len(watchdog.audit_drift()) == 0

    # Simulate drift
    mutated_config = dict(declared_config)
    mutated_config["enable_dns_hostnames"] = False
    watchdog.simulate_live_mutation("vpc-finmid-prod-eu-central", mutated_config)

    assert state.has_drift is True
    drifted = watchdog.audit_drift()
    assert len(drifted) == 1
    assert drifted[0].resource_id == "vpc-finmid-prod-eu-central"

    # Reconcile drift
    reconciled = watchdog.reconcile_drift("vpc-finmid-prod-eu-central")
    assert reconciled is True
    assert state.has_drift is False
    assert len(watchdog.audit_drift()) == 0


def test_credit_gateway_sla_monitoring_healthy():
    orchestrator = FinmidPlatformOrchestrator()
    metric = orchestrator.sla_watchdog.record_traffic(
        endpoint="/api/v1/credit/eligibility-check",
        total_requests=10000,
        p95_latency_ms=28.5,
        p99_latency_ms=64.2,
        error_count=5
    )

    assert metric.error_rate == 0.0005
    assert metric.is_sla_breached is False
    assert len(orchestrator.sla_watchdog.get_breached_endpoints()) == 0


def test_credit_gateway_sla_monitoring_breached_latency():
    orchestrator = FinmidPlatformOrchestrator()
    metric = orchestrator.sla_watchdog.record_traffic(
        endpoint="/api/v1/checkout/instant-payout",
        total_requests=5000,
        p95_latency_ms=120.0,
        p99_latency_ms=185.0,  # exceeds 150ms limit
        error_count=10
    )

    assert metric.is_sla_breached is True
    breached = orchestrator.sla_watchdog.get_breached_endpoints()
    assert len(breached) == 1
    assert breached[0].endpoint == "/api/v1/checkout/instant-payout"


def test_credit_gateway_sla_monitoring_breached_error_rate():
    orchestrator = FinmidPlatformOrchestrator()
    metric = orchestrator.sla_watchdog.record_traffic(
        endpoint="/api/v1/kyc/identity-verify",
        total_requests=1000,
        p95_latency_ms=45.0,
        p99_latency_ms=90.0,
        error_count=25  # 2.5% error rate, exceeds 1% limit
    )

    assert metric.error_rate == 0.025
    assert metric.is_sla_breached is True


def test_platform_health_check_reporting():
    orchestrator = FinmidPlatformOrchestrator()
    orchestrator.onboard_tenant("tenant-a", "Tenant A", PlatformTier.GROWTH_MARKETPLACE)
    orchestrator.preview_reconciler.provision_preview(10, "feat/new-api", "dev")

    # Initially healthy
    health = orchestrator.run_platform_health_check()
    assert health["status"] == "HEALTHY"
    assert health["active_tenants"] == 1
    assert health["active_preview_environments"] == 1
    assert health["drifted_resources_count"] == 0

    # Introduce drift
    orchestrator.drift_watchdog.register_resource("s3-bucket", "aws_s3_bucket", {"versioning": True})
    orchestrator.drift_watchdog.simulate_live_mutation("s3-bucket", {"versioning": False})

    health_degraded = orchestrator.run_platform_health_check()
    assert health_degraded["status"] == "ATTENTION_REQUIRED"
    assert health_degraded["drifted_resources_count"] == 1
