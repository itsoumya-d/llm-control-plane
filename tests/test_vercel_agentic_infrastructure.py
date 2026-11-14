"""Unit and integration tests for Vercel Agentic Infrastructure Platform."""

import hashlib
import json
import time

from llm_control_plane.vercel_agentic_infrastructure import (
    AgenticBuildOrchestrator,
    BuildState,
    RemoteArtifactCache,
    TokenBucketRateLimiter,
)


def _compute_sig(payload: dict, secret: str = "vercel_agentic_webhook_secret") -> str:
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw + secret.encode("utf-8")).hexdigest()


def test_token_bucket_rate_limiter():
    limiter = TokenBucketRateLimiter(capacity=3, refill_rate_per_sec=10.0)
    assert limiter.allow("tenant-1") is True
    assert limiter.allow("tenant-1") is True
    assert limiter.allow("tenant-1") is True
    # Exceeded
    assert limiter.allow("tenant-1") is False

    # Other tenant is unaffected (isolated)
    assert limiter.allow("tenant-2") is True

    # After brief sleep, token refills
    time.sleep(0.15)
    assert limiter.allow("tenant-1") is True


def test_remote_artifact_cache():
    cache = RemoteArtifactCache()
    key1 = cache.compute_cache_key("nextjs", "src_hash_123", {"NODE_ENV": "production"})
    key2 = cache.compute_cache_key("nextjs", "src_hash_123", {"NODE_ENV": "production"})
    assert key1 == key2

    # Miss initially
    assert cache.get(key1) is None

    # Store artifact
    data = b"export default function Page() { return <h1>Hello Vercel</h1>; }"
    art = cache.put(key1, data, {"version": "v1"})
    assert art.size_bytes == len(data)
    assert art.cache_key == key1

    # Hit
    retrieved = cache.get(key1)
    assert retrieved is not None
    assert retrieved.content_hash == hashlib.sha256(data).hexdigest()

    metrics = cache.metrics
    assert metrics["hits"] == 1
    assert metrics["misses"] == 1
    assert metrics["hit_ratio"] == 0.5


def test_webhook_ingest_and_signature():
    orchestrator = AgenticBuildOrchestrator()
    payload = {
        "tenant_id": "tenant-v0",
        "project_id": "proj-agent-ai",
        "branch": "feat/autonomous-button",
        "commit_sha": "a1b2c3d4",
        "trigger": "AGENT_V0",
    }
    sig = _compute_sig(payload)

    # Valid webhook
    res = orchestrator.handle_webhook_event("deployment.created", payload, sig)
    assert res["status"] == "ACCEPTED"
    assert res["state"] == BuildState.QUEUED.value
    dpl_id = res["deployment_id"]
    assert dpl_id in orchestrator.deployments

    # Invalid signature
    bad_res = orchestrator.handle_webhook_event("deployment.created", payload, "invalid_sig")
    assert bad_res["status"] == "REJECTED"


def test_webhook_rate_throttling():
    limiter = TokenBucketRateLimiter(capacity=1, refill_rate_per_sec=0.1)
    orchestrator = AgenticBuildOrchestrator(rate_limiter=limiter)

    payload = {"tenant_id": "spammer-agent", "project_id": "spam"}
    sig = _compute_sig(payload)

    res1 = orchestrator.handle_webhook_event("deployment.created", payload, sig)
    assert res1["status"] == "ACCEPTED"

    res2 = orchestrator.handle_webhook_event("deployment.created", payload, sig)
    assert res2["status"] == "THROTTLED"


def test_build_execution_lifecycle_and_caching():
    orchestrator = AgenticBuildOrchestrator()
    payload = {
        "tenant_id": "org-cursor",
        "project_id": "dashboard-app",
        "branch": "main",
        "commit_sha": "commit_1111",
        "trigger": "AGENT_API",
    }
    sig = _compute_sig(payload)
    accepted = orchestrator.handle_webhook_event("deployment.created", payload, sig)
    dpl_id = accepted["deployment_id"]

    # 1. First build (cold cache -> cache miss)
    record1 = orchestrator.execute_build(dpl_id, "nextjs", "source_tree_v1")
    assert record1.state == BuildState.READY
    assert record1.cache_hit is False
    assert "dashboard-app" in record1.url
    assert record1.duration_ms > 0

    # 2. Trigger another build on the same source (warm cache -> cache hit)
    payload2 = {
        "tenant_id": "org-cursor",
        "project_id": "dashboard-app",
        "branch": "main",
        "commit_sha": "commit_2222",
        "trigger": "AGENT_API",
    }
    sig2 = _compute_sig(payload2)
    acc2 = orchestrator.handle_webhook_event("deployment.created", payload2, sig2)
    dpl_id2 = acc2["deployment_id"]

    record2 = orchestrator.execute_build(dpl_id2, "nextjs", "source_tree_v1")
    assert record2.state == BuildState.READY
    assert record2.cache_hit is True


def test_build_failure_and_diagnostics():
    orchestrator = AgenticBuildOrchestrator()
    payload = {
        "tenant_id": "org-v0",
        "project_id": "broken-app",
        "commit_sha": "fail123",
        "trigger": "AGENT_V0",
    }
    sig = _compute_sig(payload)
    acc = orchestrator.handle_webhook_event("deployment.created", payload, sig)
    dpl_id = acc["deployment_id"]

    error_msg = "Error: Cannot find module '@radix-ui/react-slot'\nType error: TS2322"
    record = orchestrator.execute_build(dpl_id, "nextjs", "broken_src", mock_failure=error_msg)
    assert record.state == BuildState.FAILED
    assert record.error_summary == error_msg

    diagnostics = orchestrator.parse_error_diagnostics(error_msg)
    assert diagnostics["has_errors"] is True
    assert len(diagnostics["remediation_hints"]) == 2
    assert "Missing npm dependency" in diagnostics["remediation_hints"][0]
    assert "TypeScript contract mismatch" in diagnostics["remediation_hints"][1]
