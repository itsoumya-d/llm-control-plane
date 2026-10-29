from fastapi.testclient import TestClient

from llm_control_plane.api import create_app
from llm_control_plane.demo import build_demo_gateway
from llm_control_plane.gateway import GatewayConfig, LLMGateway
from llm_control_plane.providers import FakeProvider, provider_map
from llm_control_plane.rate_limit import TenantRateLimiter
from llm_control_plane.routing import RoutingTable
from llm_control_plane.types import ProviderError


def client():
    return TestClient(create_app(build_demo_gateway()))


def test_health():
    response = client().get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["audit_chain_valid"] is True


def test_chat_completion():
    response = client().post(
        "/v1/chat/completions",
        json={
            "tenant_id": "t1",
            "model": "smart",
            "messages": [{"role": "user", "content": "What is 2+2?"}],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["text"] == "4"
    assert body["provider"] == "fake-primary"
    assert body["trace_id"]
    assert body["cost_usd"] > 0


def test_chat_redacts_pii():
    response = client().post(
        "/v1/chat/completions",
        json={
            "tenant_id": "t1",
            "model": "smart",
            "messages": [{"role": "user", "content": "Email person@example.com about 2+2"}],
        },
    )
    assert response.status_code == 200
    assert response.json()["redactions"] == {"email": 1}


def test_eval_endpoint():
    response = client().post(
        "/v1/evals/runs",
        json={
            "tenant_id": "t1",
            "model_alias": "smart",
            "dataset": {
                "name": "smoke",
                "version": "v1",
                "cases": [
                    {"id": "capital", "input": "What is the capital of France?", "expected": "Paris"},
                    {"id": "math", "input": "What is 2+2?", "expected": "4"},
                ],
            },
            "scorers": ["contains"],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["run"]["metrics"]["pass_rate"] == 1.0
    assert "Eval run" in body["markdown"]


def test_eval_endpoint_rejects_unknown_scorer():
    response = client().post(
        "/v1/evals/runs",
        json={
            "tenant_id": "t1",
            "dataset": {"name": "d", "cases": [{"id": "a", "input": "x", "expected": "y"}]},
            "scorers": ["nope"],
        },
    )
    assert response.status_code == 400


def test_metrics_endpoint():
    api = client()
    api.post(
        "/v1/chat/completions",
        json={"tenant_id": "t1", "messages": [{"role": "user", "content": "2+2"}]},
    )
    response = api.get("/metrics")
    assert response.status_code == 200
    assert "gateway_requests_total" in response.text


def test_rate_limit_response():
    provider = FakeProvider("fake-primary")
    routing = RoutingTable.from_dict({"smart": [{"provider": "fake-primary", "model": "fake-large"}]})
    gateway = LLMGateway(
        provider_map(provider),
        routing,
        config=GatewayConfig(max_attempts_per_target=1),
        limiter=TenantRateLimiter(capacity=0.0001, refill_per_second=0.0001),
    )
    response = TestClient(create_app(gateway)).post(
        "/v1/chat/completions",
        json={"tenant_id": "t1", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 429
    assert "retry_after_seconds" in response.json()


def test_provider_error_response():
    provider = FakeProvider(
        "fake-primary",
        failures=[ProviderError("fake-primary", "down", retryable=False)],
    )
    routing = RoutingTable.from_dict({"smart": [{"provider": "fake-primary", "model": "fake-large"}]})
    gateway = LLMGateway(provider_map(provider), routing, config=GatewayConfig(max_attempts_per_target=1))
    response = TestClient(create_app(gateway)).post(
        "/v1/chat/completions",
        json={"tenant_id": "t1", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 502
