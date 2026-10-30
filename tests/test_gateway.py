import pytest

from llm_control_plane.gateway import GatewayConfig, LLMGateway
from llm_control_plane.policy import PolicyEngine, PolicyViolation
from llm_control_plane.providers import FakeProvider, provider_map
from llm_control_plane.rate_limit import TenantRateLimiter
from llm_control_plane.routing import RoutingTable
from llm_control_plane.types import ChatMessage, CompletionRequest, ProviderError, RateLimitExceeded


class Clock:
    def __init__(self, now=0.0):
        self.now = now

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def make_gateway(primary, fallback=None, config=None, clock=None, limiter=None, policy=None):
    providers = {"fake-primary": primary}
    targets = [{"provider": "fake-primary", "model": "fake-large"}]
    if fallback is not None:
        providers["fake-fallback"] = fallback
        targets.append({"provider": "fake-fallback", "model": "fake-small"})
    routing = RoutingTable.from_dict({"smart": targets})
    return LLMGateway(
        provider_map(*providers.values()),
        routing,
        config=config or GatewayConfig(max_attempts_per_target=1),
        clock=clock or (lambda: 0.0),
        limiter=limiter,
        policy=policy,
    )


def request(content="hello world", trace_id=None, tenant="tenant-a"):
    return CompletionRequest(
        tenant_id=tenant,
        model="smart",
        messages=[ChatMessage(role="user", content=content)],
        trace_id=trace_id,
    )


def test_primary_success_and_cost_attribution():
    provider = FakeProvider("fake-primary", responses=["hello"], latency_ms=12.5)
    gateway = make_gateway(provider)
    result = gateway.complete(request())
    assert result.text == "hello"
    assert result.provider == "fake-primary"
    assert result.attempts == 1
    assert result.usage.prompt_tokens == 2
    assert result.usage.completion_tokens == 1
    assert result.cost_usd == pytest.approx(2 / 1000 * 0.0005 + 1 / 1000 * 0.0015)
    assert result.latency_ms >= 12.5
    assert gateway.audit.verify() is True


def test_retry_then_success():
    provider = FakeProvider(
        "fake-primary",
        failures=[ProviderError("fake-primary", "rate limited", status_code=429, retryable=True), None],
    )
    delays = []
    gateway = make_gateway(
        provider,
        config=GatewayConfig(max_attempts_per_target=2, base_backoff_seconds=0.01),
    )
    gateway.sleeper = delays.append
    result = gateway.complete(request())
    assert result.provider == "fake-primary"
    assert result.attempts == 2
    assert delays == [0.01]


def test_failover_to_secondary():
    primary = FakeProvider(
        "fake-primary",
        failures=[ProviderError("fake-primary", "timeout", retryable=True)],
    )
    fallback = FakeProvider("fake-fallback", responses=["fallback answer"])
    gateway = make_gateway(primary, fallback)
    result = gateway.complete(request())
    assert result.provider == "fake-fallback"
    assert result.text == "fallback answer"
    assert len(primary.calls) == 1
    assert len(fallback.calls) == 1


def test_non_retryable_error_skips_retries_but_can_fail_over():
    primary = FakeProvider(
        "fake-primary",
        failures=[ProviderError("fake-primary", "bad request", status_code=400, retryable=False)],
    )
    fallback = FakeProvider("fake-fallback", responses=["ok"])
    gateway = make_gateway(primary, fallback, config=GatewayConfig(max_attempts_per_target=3))
    result = gateway.complete(request())
    assert result.provider == "fake-fallback"
    assert len(primary.calls) == 1


def test_all_providers_fail():
    primary = FakeProvider(
        "fake-primary",
        failures=[ProviderError("fake-primary", "down", retryable=False)],
    )
    fallback = FakeProvider(
        "fake-fallback",
        failures=[ProviderError("fake-fallback", "also down", retryable=False)],
    )
    gateway = make_gateway(primary, fallback)
    with pytest.raises(ProviderError) as exc:
        gateway.complete(request())
    assert "all provider targets failed" in str(exc.value)
    assert gateway.audit.verify() is True
    assert gateway.audit.events()[-1].event_type == "request.failed"


def test_idempotent_replay_uses_cached_result():
    provider = FakeProvider("fake-primary", responses=["first"])
    gateway = make_gateway(provider)
    first = gateway.complete(request(trace_id="trace-1"))
    second = gateway.complete(request(trace_id="trace-1"))
    assert second is first
    assert len(provider.calls) == 1
    replays = gateway.metrics.counter_value(
        "gateway_idempotent_replays_total", {"tenant": "tenant-a", "model": "smart"}
    )
    assert replays == 1.0


def test_rate_limit_is_enforced_per_tenant():
    clock = Clock()
    limiter = TenantRateLimiter(capacity=1, refill_per_second=0.1, clock=clock)
    gateway = make_gateway(FakeProvider("fake-primary"), clock=clock, limiter=limiter)
    gateway.complete(request(tenant="a"))
    with pytest.raises(RateLimitExceeded):
        gateway.complete(request(tenant="a"))
    gateway.complete(request(tenant="b"))
    clock.advance(10)
    gateway.complete(request(tenant="a"))


def test_pii_is_redacted_before_provider_and_audit():
    provider = FakeProvider("fake-primary", responses=["ok"])
    gateway = make_gateway(provider)
    result = gateway.complete(request("email me at person@example.com"))
    assert "[REDACTED_EMAIL]" in provider.calls[0].messages[0].content
    assert result.redactions == {"email": 1}
    assert "person@example.com" not in gateway.audit.to_jsonl()


def test_unknown_alias_fails():
    gateway = make_gateway(FakeProvider("fake-primary"))
    bad = CompletionRequest(
        tenant_id="tenant-a",
        model="missing",
        messages=[ChatMessage(role="user", content="hi")],
    )
    with pytest.raises(ProviderError):
        gateway.complete(bad)
    assert gateway.audit.events()[-1].event_type == "request.failed"


def test_missing_provider_fails_over_or_raises():
    routing = RoutingTable.from_dict({"smart": [{"provider": "missing", "model": "x"}]})
    gateway = LLMGateway({}, routing, config=GatewayConfig(max_attempts_per_target=1))
    with pytest.raises(ProviderError):
        gateway.complete(request())
    assert any(event.event_type == "provider.missing" for event in gateway.audit.events())


def test_health_and_metrics_output():
    provider = FakeProvider("fake-primary")
    gateway = make_gateway(provider)
    gateway.complete(request())
    assert gateway.health() == {"fake-primary": True}
    metrics = gateway.metrics.prometheus_text()
    assert "gateway_requests_total" in metrics
    assert "provider_success_total" in metrics
    provider.healthy = False
    assert gateway.health() == {"fake-primary": False}


def test_policy_denial_happens_before_provider_call():
    provider = FakeProvider("fake-primary", responses=["should not run"])
    gateway = make_gateway(
        provider,
        policy=PolicyEngine(allowed_models=frozenset({"approved"})),
    )
    with pytest.raises(PolicyViolation):
        gateway.complete(request())
    assert provider.calls == []
    assert gateway.audit.verify() is True
    assert gateway.audit.events()[-1].event_type == "request.policy_denied"
    denied = gateway.metrics.counter_value(
        "gateway_policy_denied_total",
        {"tenant": "tenant-a", "model": "smart", "policy": "default"},
    )
    assert denied == 1.0
