import pytest

from llm_control_plane.policy import PolicyEngine, PolicyViolation
from llm_control_plane.types import ChatMessage, CompletionRequest


def request(**overrides):
    values = {
        "tenant_id": "tenant-a",
        "model": "smart",
        "messages": [ChatMessage(role="user", content="hello")],
        "max_tokens": 512,
        "use_case": "support",
    }
    values.update(overrides)
    return CompletionRequest(**values)


def test_policy_allows_request_within_limits():
    engine = PolicyEngine(
        allowed_models=frozenset({"smart"}),
        allowed_use_cases=frozenset({"support"}),
        max_tokens=1024,
        require_trace_id=True,
    )
    decision = engine.evaluate(request(trace_id="trace-1"))
    assert decision.allowed is True
    assert decision.reasons == ()
    assert engine.enforce(request(trace_id="trace-1")).allowed is True


def test_policy_reports_each_violation():
    engine = PolicyEngine(
        allowed_models=frozenset({"approved"}),
        allowed_use_cases=frozenset({"support"}),
        max_tokens=100,
        require_trace_id=True,
        blocked_metadata_keys=frozenset({"raw_prompt"}),
    )
    decision = engine.evaluate(
        request(model="unapproved", use_case="training", max_tokens=200, metadata={"raw_prompt": "x"})
    )
    assert decision.allowed is False
    assert any("model" in reason for reason in decision.reasons)
    assert any("use_case" in reason for reason in decision.reasons)
    assert any("max_tokens" in reason for reason in decision.reasons)
    assert any("trace_id" in reason for reason in decision.reasons)
    assert any("raw_prompt" in reason for reason in decision.reasons)


def test_policy_requires_tenant_and_positive_tokens():
    engine = PolicyEngine()
    decision = engine.evaluate(request(tenant_id="  ", max_tokens=0))
    assert decision.allowed is False
    assert any("tenant_id" in reason for reason in decision.reasons)
    assert any("positive" in reason for reason in decision.reasons)


def test_policy_enforce_raises_with_decision():
    engine = PolicyEngine(allowed_models=frozenset({"approved"}))
    with pytest.raises(PolicyViolation) as exc:
        engine.enforce(request())
    assert exc.value.decision.allowed is False
    assert "not allowed" in str(exc.value)
