import time

from llm_control_plane.qdrant_vector_control_plane_gateway import (
    OAuth2TokenScope,
    QdrantMCPAgentBridge,
    QdrantVectorControlPlaneGateway,
    ScopedOAuth2TokenIntrospector,
    TokenBucketRateLimiter,
    VectorClusterObservabilityCollector,
    VectorClusterSpec,
)


def test_scoped_oauth2_token_issuance_and_introspection():
    introspector = ScopedOAuth2TokenIntrospector("secret-signing-key-qdrant-2026")
    scopes = {OAuth2TokenScope.READ_COLLECTION, OAuth2TokenScope.MCP_INVOKE}
    allowed_colls = {"support-kb-prod", "customer-embeddings"}

    token = introspector.issue_token(
        tenant_id="tenant-acme-corp",
        principal_id="agent-claude-support",
        scopes=scopes,
        allowed_collections=allowed_colls,
        ttl_seconds=3600.0,
        is_agent=True,
    )

    claims = introspector.introspect(token)
    assert claims is not None
    assert claims.tenant_id == "tenant-acme-corp"
    assert claims.principal_id == "agent-claude-support"
    assert claims.is_agent is True
    assert claims.has_scope(OAuth2TokenScope.MCP_INVOKE) is True
    assert claims.has_scope(OAuth2TokenScope.ADMIN_CLUSTER) is False
    assert claims.can_access_collection("support-kb-prod") is True
    assert claims.can_access_collection("finance-private") is False


def test_token_cache_and_expiration():
    introspector = ScopedOAuth2TokenIntrospector("secret-key", cache_ttl_seconds=0.05)
    token = introspector.issue_token(
        tenant_id="tenant-1",
        principal_id="user-1",
        scopes={OAuth2TokenScope.READ_COLLECTION},
        allowed_collections={"default"},
        ttl_seconds=0.1,
    )

    # Fast cache hit
    assert introspector.introspect(token) is not None

    # Wait for TTL expiration
    time.sleep(0.12)
    assert introspector.introspect(token) is None


def test_token_bucket_rate_limiter():
    limiter = TokenBucketRateLimiter(capacity=3, refill_rate_per_sec=1.0)
    key = "agent-burst-tester"

    assert limiter.allow(key, tokens=1) is True
    assert limiter.allow(key, tokens=1) is True
    assert limiter.allow(key, tokens=1) is True
    # Capacity exhausted
    assert limiter.allow(key, tokens=1) is False


def test_mcp_agent_bridge_tool_invocation_and_auth():
    introspector = ScopedOAuth2TokenIntrospector("mcp-secret")
    bridge = QdrantMCPAgentBridge(introspector)

    def echo_handler(args):
        return f"Echo: {args.get('text')}"

    bridge.register_mcp_tool("echo_tool", echo_handler)

    valid_token = introspector.issue_token(
        tenant_id="t1",
        principal_id="agent-1",
        scopes={OAuth2TokenScope.MCP_INVOKE},
        allowed_collections={"*"},
    )

    unauthorized_token = introspector.issue_token(
        tenant_id="t1",
        principal_id="agent-2",
        scopes={OAuth2TokenScope.READ_COLLECTION},
        allowed_collections={"*"},
    )

    # Valid invocation
    res = bridge.invoke_tool(valid_token, "echo_tool", {"text": "hello Qdrant"})
    assert res["success"] is True
    assert res["result"] == "Echo: hello Qdrant"
    assert len(bridge.trace_spans) == 1
    assert bridge.trace_spans[0].status == "SUCCESS"

    # Unauthorized scope
    res_err = bridge.invoke_tool(unauthorized_token, "echo_tool", {"text": "denied"})
    assert res_err["success"] is False
    assert res_err["code"] == "FORBIDDEN"


def test_mcp_agent_bridge_collection_access_control():
    introspector = ScopedOAuth2TokenIntrospector("mcp-secret-2")
    bridge = QdrantMCPAgentBridge(introspector)

    bridge.register_mcp_tool("vector_search", lambda args: {"hits": [1, 2, 3]})

    token = introspector.issue_token(
        tenant_id="t1",
        principal_id="agent-scoped",
        scopes={OAuth2TokenScope.MCP_INVOKE},
        allowed_collections={"allowed_collection"},
    )

    # Allowed collection access
    res_ok = bridge.invoke_tool(
        token,
        "vector_search",
        {"collection_name": "allowed_collection", "vector": [0.1, 0.2]},
    )
    assert res_ok["success"] is True

    # Forbidden collection access
    res_forbidden = bridge.invoke_tool(
        token,
        "vector_search",
        {"collection_name": "restricted_collection", "vector": [0.1, 0.2]},
    )
    assert res_forbidden["success"] is False
    assert res_forbidden["code"] == "FORBIDDEN"


def test_control_plane_gateway_cluster_provisioning_and_reconcile():
    introspector = ScopedOAuth2TokenIntrospector("cp-gateway-secret")
    gateway = QdrantVectorControlPlaneGateway(introspector)

    admin_token = introspector.issue_token(
        tenant_id="tenant-cloud-ops",
        principal_id="sre-platform-lead",
        scopes={OAuth2TokenScope.ADMIN_CLUSTER},
        allowed_collections={"*"},
    )

    spec = VectorClusterSpec(
        cluster_id="cluster-fra-qdrant-01",
        tenant_id="tenant-cloud-ops",
        region="eu-central-1",
        node_count=3,
        vector_dimension=1536,
        distance_metric="Cosine",
    )

    res = gateway.provision_cluster(admin_token, spec)
    assert res["success"] is True
    assert res["cluster"].status == "ACTIVE"

    # Verify retrieval
    retrieved = gateway.get_cluster(admin_token, "cluster-fra-qdrant-01")
    assert retrieved is not None
    assert retrieved.node_count == 3

    # Verify reconciliation loop
    active_count = gateway.reconcile_clusters()
    assert active_count == 1


def test_vector_cluster_observability_latency_percentiles():
    collector = VectorClusterObservabilityCollector()

    # Empty stats
    assert collector.get_latency_percentiles() == {"p50": 0.0, "p95": 0.0, "p99": 0.0}

    # Add 100 latency observations: 1ms to 100ms
    for lat in range(1, 101):
        collector.record_query_latency(float(lat))

    p = collector.get_latency_percentiles()
    assert p["p50"] == 51.0
    assert p["p95"] == 96.0
    assert p["p99"] == 100.0

    collector.update_cluster_metrics(vectors=1_000_000, segments=8, memory_bytes=2_147_483_648)
    assert collector._metrics["total_vectors"] == 1_000_000
