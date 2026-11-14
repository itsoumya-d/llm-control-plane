"""Tests for Speakeasy Enterprise AI Control Plane SRE Engine."""

import time

from llm_control_plane.speakeasy_control_plane_sre import (
    AgentSessionSecurityPolicy,
    CanaryStage,
    CircuitBreaker,
    CircuitBreakerState,
    ControlPlaneCanaryRollout,
    MCPServerDefinition,
    SpeakeasyAIControlPlaneProxy,
    SREObservabilityMetrics,
    TokenBucketRateLimiter,
    ToolExecutionRequest,
)


def test_mcp_server_definition_and_tool_discovery():
    server = MCPServerDefinition(
        server_id="mcp-github",
        name="GitHub Enterprise MCP",
        protocol_version="2024-11-05",
        endpoint_url="https://mcp.internal.corp/github",
        supported_tools=["create_issue", "merge_pr", "search_code"],
        is_healthy=True,
    )
    assert server.has_tool("merge_pr")
    assert not server.has_tool("drop_table")
    assert server.protocol_version == "2024-11-05"


def test_session_security_policy_allowed_tools():
    policy = AgentSessionSecurityPolicy(
        session_id="sess-001",
        tenant_id="tenant-corp",
        allowed_tools=["read_file", "search_code"],
        max_token_budget=10000,
    )
    assert policy.is_tool_allowed("read_file")
    assert not policy.is_tool_allowed("execute_shell")

    wildcard_policy = AgentSessionSecurityPolicy(
        session_id="sess-wildcard",
        tenant_id="tenant-corp",
        allowed_tools=["*"],
        max_token_budget=10000,
    )
    assert wildcard_policy.is_tool_allowed("any_custom_tool")


def test_session_security_policy_dlp_detection():
    policy = AgentSessionSecurityPolicy(
        session_id="sess-002",
        tenant_id="tenant-corp",
        allowed_tools=["*"],
        max_token_budget=5000,
    )
    safe_payload = {"query": "SELECT count(*) FROM users", "limit": 10}
    assert len(policy.inspect_dlp_violations(safe_payload)) == 0

    leaked_api_key = {"headers": {"Authorization": "api_key=sk-proj-1234567890abcdef"}}
    violations = policy.inspect_dlp_violations(leaked_api_key)
    assert len(violations) > 0

    leaked_ssn = {"user_ssn": "123-45-6789"}
    assert len(policy.inspect_dlp_violations(leaked_ssn)) > 0


def test_token_budget_enforcement():
    policy = AgentSessionSecurityPolicy(
        session_id="sess-003",
        tenant_id="tenant-corp",
        allowed_tools=["*"],
        max_token_budget=1000,
        current_token_spend=850,
    )
    assert policy.has_sufficient_budget(100)
    assert not policy.has_sufficient_budget(200)


def test_token_bucket_rate_limiter():
    limiter = TokenBucketRateLimiter(capacity=3, refill_rate_per_sec=1.0)
    assert limiter.consume(1)
    assert limiter.consume(1)
    assert limiter.consume(1)
    assert not limiter.consume(1)


def test_circuit_breaker_trips_to_open_on_failures():
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout_sec=0.1)
    assert cb.state == CircuitBreakerState.CLOSED
    assert cb.can_execute()

    cb.record_failure()
    cb.record_failure()
    assert cb.state == CircuitBreakerState.CLOSED

    cb.record_failure()
    assert cb.state == CircuitBreakerState.OPEN
    assert not cb.can_execute()


def test_circuit_breaker_recovery_half_open_to_closed():
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout_sec=0.05)
    cb.record_failure()
    cb.record_failure()
    assert cb.state == CircuitBreakerState.OPEN

    time.sleep(0.06)
    assert cb.can_execute()
    assert cb.state == CircuitBreakerState.HALF_OPEN

    cb.record_success()
    cb.record_success()
    assert cb.state == CircuitBreakerState.CLOSED


def test_proxy_successful_tool_execution():
    proxy = SpeakeasyAIControlPlaneProxy()
    server = MCPServerDefinition(
        server_id="mcp-linear",
        name="Linear MCP",
        protocol_version="2024-11-05",
        endpoint_url="https://mcp.internal/linear",
        supported_tools=["create_ticket", "get_ticket"],
    )
    proxy.register_server(server)

    policy = AgentSessionSecurityPolicy(
        session_id="sess-linear-1",
        tenant_id="tenant-1",
        allowed_tools=["create_ticket", "get_ticket"],
        max_token_budget=10000,
    )
    proxy.register_session(policy)

    req = ToolExecutionRequest(
        request_id="req-101",
        session_id="sess-linear-1",
        tool_name="create_ticket",
        arguments={"title": "Fix memory leak in runner"},
        estimated_tokens=150,
    )
    res = proxy.execute_tool(req)
    assert res.status == "SUCCESS"
    assert res.output["status"] == "executed"
    assert res.tokens_consumed == 150
    assert policy.current_token_spend == 150
    assert proxy.metrics.successful_requests == 1


def test_proxy_blocks_unauthorized_tool_and_dlp():
    proxy = SpeakeasyAIControlPlaneProxy()
    server = MCPServerDefinition(
        server_id="mcp-db",
        name="DB MCP",
        protocol_version="2024-11-05",
        endpoint_url="https://mcp.internal/db",
        supported_tools=["query_db", "delete_all"],
    )
    proxy.register_server(server)

    policy = AgentSessionSecurityPolicy(
        session_id="sess-db-1",
        tenant_id="tenant-1",
        allowed_tools=["query_db"],
        max_token_budget=5000,
    )
    proxy.register_session(policy)

    # 1. Blocked unauthorized tool
    req_unauth = ToolExecutionRequest(
        request_id="req-201",
        session_id="sess-db-1",
        tool_name="delete_all",
        arguments={},
    )
    res1 = proxy.execute_tool(req_unauth)
    assert res1.status == "BLOCKED_SECURITY"
    assert "forbidden" in res1.error_message

    # 2. Blocked DLP sensitive pattern
    req_dlp = ToolExecutionRequest(
        request_id="req-202",
        session_id="sess-db-1",
        tool_name="query_db",
        arguments={"secret": "api_key=sk-live-99999"},
    )
    res2 = proxy.execute_tool(req_dlp)
    assert res2.status == "BLOCKED_SECURITY"
    assert "DLP violation" in res2.error_message


def test_sre_metrics_percentiles_and_error_budget():
    metrics = SREObservabilityMetrics(target_slo=0.99)
    for lat in [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]:
        metrics.record_event("SUCCESS", lat, 50)

    pcts = metrics.compute_percentiles()
    assert pcts["p50"] == 60.0
    assert pcts["p90"] == 100.0
    assert pcts["p99"] == 100.0
    assert metrics.calculate_error_rate() == 0.0

    metrics.record_event("FAILED", 500.0, 0)
    assert metrics.calculate_error_rate() > 0.0
    assert metrics.calculate_error_budget_burn() > 0.0


def test_canary_rollout_advance_and_rollback():
    rollout = ControlPlaneCanaryRollout(
        rollout_id="rollout-v2-1",
        target_version="v2.1.0",
        error_rate_threshold=0.05,
        p99_latency_threshold_ms=1000.0,
    )
    assert rollout.current_stage == CanaryStage.STAGE_10

    # Healthy metrics -> Advance
    healthy_metrics = SREObservabilityMetrics()
    for _ in range(100):
        healthy_metrics.record_event("SUCCESS", 120.0)

    assert rollout.evaluate_and_advance(healthy_metrics)
    assert rollout.current_stage == CanaryStage.STAGE_25

    assert rollout.evaluate_and_advance(healthy_metrics)
    assert rollout.current_stage == CanaryStage.STAGE_50

    assert rollout.evaluate_and_advance(healthy_metrics)
    assert rollout.current_stage == CanaryStage.STAGE_100
    assert rollout.is_completed

    # Degraded rollout -> Automatic rollback
    degraded_rollout = ControlPlaneCanaryRollout(
        rollout_id="rollout-v2-2",
        target_version="v2.2.0",
        error_rate_threshold=0.02,
    )
    unhealthy_metrics = SREObservabilityMetrics()
    for _ in range(80):
        unhealthy_metrics.record_event("SUCCESS", 100.0)
    for _ in range(20):
        unhealthy_metrics.record_event("FAILED", 500.0)

    assert not degraded_rollout.evaluate_and_advance(unhealthy_metrics)
    assert degraded_rollout.current_stage == CanaryStage.ROLLED_BACK
    assert "Error rate" in degraded_rollout.rollback_reason
