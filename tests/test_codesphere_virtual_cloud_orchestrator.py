"""Unit & Integration Test Suite for Codesphere Virtual Cloud Orchestrator."""

import asyncio
import hashlib
import hmac
import json

import pytest

from llm_control_plane.codesphere_virtual_cloud_orchestrator import (
    CodesphereVirtualCloudOrchestrator,
    ComputeNode,
    ManagedServiceBroker,
    ManagedServiceType,
    NodeProvider,
    ServiceStatus,
    SovereignIAMSecretsEngine,
    WorkloadSpec,
)


@pytest.fixture
def master_kek():
    return "0123456789abcdef0123456789abcdef-master-kek-32bytes"


@pytest.fixture
def orchestrator(master_kek):
    orch = CodesphereVirtualCloudOrchestrator(master_kek=master_kek)
    orch.register_compute_node(
        ComputeNode(
            node_id="baremetal-de-01",
            provider=NodeProvider.BARE_METAL,
            region="europe-west3-karlsruhe",
            total_cpu_millicores=16000,
            total_memory_mb=64000,
            available_cpu_millicores=16000,
            available_memory_mb=64000,
            is_air_gapped=False,
        )
    )
    orch.register_compute_node(
        ComputeNode(
            node_id="sovereign-airgap-01",
            provider=NodeProvider.ON_PREM_SOVEREIGN,
            region="europe-west3-karlsruhe",
            total_cpu_millicores=8000,
            total_memory_mb=32000,
            available_cpu_millicores=8000,
            available_memory_mb=32000,
            is_air_gapped=True,
        )
    )
    return orch


def test_compute_node_allocation():
    node = ComputeNode(
        node_id="test-node-1",
        provider=NodeProvider.HETZNER_CLOUD,
        region="eu-central-1",
        total_cpu_millicores=4000,
        total_memory_mb=8000,
        available_cpu_millicores=4000,
        available_memory_mb=8000,
    )
    assert node.can_accommodate(1000, 2000)
    node.allocate("w-1", 1000, 2000)
    assert node.available_cpu_millicores == 3000
    assert node.available_memory_mb == 6000
    assert "w-1" in node.active_workload_ids

    # Release
    node.release("w-1", 1000, 2000)
    assert node.available_cpu_millicores == 4000
    assert node.available_memory_mb == 8000
    assert "w-1" not in node.active_workload_ids


def test_iam_rbac_and_audit_integrity(master_kek):
    iam = SovereignIAMSecretsEngine(master_kek)
    iam.assign_role_permissions("tenant-alpha", {"deployment:create", "deployment:read"})

    assert iam.authorize("tenant-alpha", "deployment:create")
    assert not iam.authorize("tenant-alpha", "deployment:delete")
    assert not iam.authorize("tenant-beta", "deployment:create")
    assert iam.verify_audit_chain_integrity()


def test_envelope_encryption_and_tenant_isolation(master_kek):
    iam = SovereignIAMSecretsEngine(master_kek)
    secret_payload = "DATABASE_URL=postgres://sovereign:secret@db.internal:5432/codesphere"

    envelope = iam.encrypt_secret("tenant-alpha", secret_payload)
    assert envelope["tenant_id"] == "tenant-alpha"
    assert "ciphertext_b64" in envelope

    # Decrypt valid
    decrypted = iam.decrypt_secret(envelope, "tenant-alpha")
    assert decrypted == secret_payload

    # Cross-tenant violation
    with pytest.raises(PermissionError):
        iam.decrypt_secret(envelope, "tenant-hacker")

    # Tamper detection
    tampered_envelope = dict(envelope)
    tampered_envelope["auth_tag"] = "invalid_tampered_tag"
    with pytest.raises(ValueError, match="Integrity check failed"):
        iam.decrypt_secret(tampered_envelope, "tenant-alpha")


def test_workload_mtls_cert_issuance(master_kek):
    iam = SovereignIAMSecretsEngine(master_kek)
    cert = iam.issue_workload_mtls_cert("tenant-de", "workload-api-1")
    assert cert["workload_id"] == "workload-api-1"
    assert cert["subject_cn"] == "workload-api-1.tenant-tenant-de.codesphere.internal"
    assert len(cert["signature"]) == 64


def test_managed_service_broker_lifecycle():
    async def _test():
        broker = ManagedServiceBroker()
        svc = await broker.provision_service("tenant-gov", ManagedServiceType.POSTGRESQL)
        assert svc.status == ServiceStatus.READY
        assert "postgresql://" in svc.connection_uri
        assert broker.get_service(svc.service_id) is not None

        # Deprovision
        success = await broker.deprovision_service(svc.service_id)
        assert success is True
        assert broker.get_service(svc.service_id) is None

    asyncio.run(_test())


def test_orchestrator_scheduling_and_airgap(orchestrator):
    orch = orchestrator
    orch.iam.assign_role_permissions("tenant-gov", {"deployment:create"})

    # Schedule standard workload
    spec = WorkloadSpec(
        workload_id="app-frontend",
        tenant_id="tenant-gov",
        image_ref="codesphere/frontend:v1",
        cpu_millicores=2000,
        memory_mb=4000,
        require_airgap=False,
    )
    node = orch.schedule_workload(spec)
    assert node.node_id == "baremetal-de-01"
    assert "app-frontend" in orch.active_workloads

    # Schedule air-gapped sovereign workload
    sovereign_spec = WorkloadSpec(
        workload_id="secure-vault",
        tenant_id="tenant-gov",
        image_ref="codesphere/vault:v1",
        cpu_millicores=2000,
        memory_mb=4000,
        require_airgap=True,
    )
    sov_node = orch.schedule_workload(sovereign_spec)
    assert sov_node.node_id == "sovereign-airgap-01"
    assert sov_node.is_air_gapped is True


def test_warm_pool_prewarming_fast_path(orchestrator):
    orch = orchestrator
    orch.iam.assign_role_permissions("tenant-dev", {"deployment:create"})

    prewarmed = orch.prewarm_workspaces(count=2)
    assert prewarmed == 2
    assert len(orch.warm_pool) == 2

    # Fast-path allocation for small developer workspace
    micro_spec = WorkloadSpec(
        workload_id="dev-ws-42",
        tenant_id="tenant-dev",
        image_ref="codesphere/workspace-node20:latest",
        cpu_millicores=500,
        memory_mb=256,
    )
    node = orch.schedule_workload(micro_spec)
    assert len(orch.warm_pool) == 1
    assert "dev-ws-42" in node.active_workload_ids


def test_rolling_update_and_rollback(orchestrator):
    orch = orchestrator
    orch.iam.assign_role_permissions("tenant-corp", {"deployment:create"})

    spec_v1 = WorkloadSpec(
        workload_id="api-gateway",
        tenant_id="tenant-corp",
        image_ref="codesphere/gateway:1.0.0",
        cpu_millicores=1000,
        memory_mb=2048,
    )
    orch.schedule_workload(spec_v1)

    spec_v2 = WorkloadSpec(
        workload_id="api-gateway",
        tenant_id="tenant-corp",
        image_ref="codesphere/gateway:2.0.0",
        cpu_millicores=1200,
        memory_mb=3072,
    )
    updated = orch.update_workload_rolling(spec_v2)
    assert updated is True
    assert orch.active_workloads["api-gateway"][0].version == 2

    # Rollback to v1
    rolled_back = orch.rollback_workload("api-gateway")
    assert rolled_back.version == 1
    assert rolled_back.image_ref == "codesphere/gateway:1.0.0"

    # Terminate
    orch.terminate_workload("api-gateway")
    assert "api-gateway" not in orch.active_workloads


def test_ci_webhook_event_verification(orchestrator):
    orch = orchestrator
    secret = "codesphere-webhook-secret-token"
    payload = json.dumps({"ref": "refs/heads/production", "after": "d4a3b2c1"}).encode("utf-8")
    valid_sig = "sha256=" + hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()

    result = orch.process_ci_webhook_event(secret, payload, valid_sig)
    assert result["status"] == "ACCEPTED"
    assert result["commit_sha"] == "d4a3b2c1"

    # Invalid signature
    with pytest.raises(PermissionError):
        orch.process_ci_webhook_event(secret, payload, "sha256=bad_signature")


def test_p99_latency_telemetry(orchestrator):
    orch = orchestrator
    for i in range(100):
        orch.record_telemetry("query_latency_ms", float(i), {"endpoint": "/deploy"})

    p99 = orch.get_latency_p99("query_latency_ms")
    assert p99 == 99.0
