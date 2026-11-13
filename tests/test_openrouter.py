"""Unit tests for OpenRouter provider adapter and telemetry tracking."""

import io
import json
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from llm_control_plane import (
    ChatMessage,
    CompletionRequest,
    ModelLatencyStats,
    OpenRouterProvider,
    OpenRouterTelemetryTracker,
    ProviderError,
)


def test_model_latency_stats_calculations():
    stats = ModelLatencyStats(model="anthropic/claude-3.5-sonnet")
    assert stats.error_rate == 0.0
    assert stats.avg_latency_ms == 0.0
    assert stats.p95_latency_ms == 0.0
    assert stats.avg_ttft_ms == 0.0

    stats.total_requests = 10
    stats.successful_requests = 8
    stats.failed_requests = 2
    stats.latencies_ms = [100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 200.0]
    stats.ttfts_ms = [40.0, 50.0]

    assert stats.error_rate == 0.2
    assert stats.avg_latency_ms == 146.0
    assert stats.p95_latency_ms == 200.0
    assert stats.avg_ttft_ms == 45.0


def test_telemetry_tracker_success_and_window():
    tracker = OpenRouterTelemetryTracker(window_size=3)
    tracker.record_success("model-a", latency_ms=50.0, ttft_ms=20.0)
    tracker.record_success("model-a", latency_ms=60.0, ttft_ms=25.0)
    tracker.record_success("model-a", latency_ms=70.0, ttft_ms=30.0)
    tracker.record_success("model-a", latency_ms=80.0, ttft_ms=35.0)

    stats = tracker.get_or_create("model-a")
    assert stats.total_requests == 4
    assert stats.successful_requests == 4
    # Window should keep only 3 items
    assert len(stats.latencies_ms) == 3
    assert stats.latencies_ms == [60.0, 70.0, 80.0]
    assert len(stats.ttfts_ms) == 3
    assert stats.ttfts_ms == [25.0, 30.0, 35.0]


def test_telemetry_tracker_failure_and_health():
    tracker = OpenRouterTelemetryTracker()
    model = "openai/gpt-4o"

    # Untried model is healthy by default
    assert tracker.is_healthy(model)

    tracker.record_success(model, 100.0)
    tracker.record_failure(model, 150.0)
    tracker.record_failure(model, 200.0)
    tracker.record_failure(model, 250.0)

    stats = tracker.get_or_create(model)
    assert stats.total_requests == 4
    assert stats.failed_requests == 3
    assert stats.error_rate == 0.75

    # Should be unhealthy due to error rate > 0.25
    assert not tracker.is_healthy(model, max_error_rate=0.25)
    # Should be unhealthy if p95 latency exceeds threshold
    assert not tracker.is_healthy(model, max_error_rate=0.99, max_p95_latency_ms=100.0)


def test_telemetry_tracker_rank_models():
    tracker = OpenRouterTelemetryTracker()
    # Fast healthy model
    tracker.record_success("model-fast", 50.0)
    tracker.record_success("model-fast", 60.0)
    tracker.record_success("model-fast", 55.0)

    # Slow healthy model
    tracker.record_success("model-slow", 300.0)
    tracker.record_success("model-slow", 320.0)
    tracker.record_success("model-slow", 310.0)

    # Broken model
    tracker.record_failure("model-broken", 100.0)
    tracker.record_failure("model-broken", 100.0)
    tracker.record_failure("model-broken", 100.0)

    candidates = ["model-broken", "model-slow", "model-fast", "model-untried"]
    ranked = tracker.rank_models(candidates)

    # Healthy and fast should come first, untried in middle, broken at the end
    assert ranked[0] in ("model-fast", "model-untried")
    assert ranked[-1] == "model-broken"


def test_openrouter_payload_and_headers():
    provider = OpenRouterProvider(
        api_key="sk-or-test-key",
        default_model="anthropic/claude-3.5-sonnet",
        fallback_models=["openai/gpt-4o", "deepseek/deepseek-chat"],
        site_url="https://myapp.com",
        app_name="MyApp",
        provider_preferences={"order": ["Anthropic", "OpenAI"]},
    )

    req = CompletionRequest(
        tenant_id="tenant-1",
        model="anthropic/claude-3.5-sonnet",
        messages=[ChatMessage(role="user", content="hello openrouter")],
        temperature=0.2,
        max_tokens=100,
    )

    payload = provider.build_payload(req)
    assert payload["model"] == "anthropic/claude-3.5-sonnet"
    assert "models" in payload
    assert payload["route"] == "fallback"
    assert "anthropic/claude-3.5-sonnet" in payload["models"]
    assert "openai/gpt-4o" in payload["models"]
    assert payload["provider"] == {"order": ["Anthropic", "OpenAI"]}
    assert payload["temperature"] == 0.2
    assert payload["max_tokens"] == 100

    headers = provider.build_headers()
    assert headers["Authorization"] == "Bearer sk-or-test-key"
    assert headers["HTTP-Referer"] == "https://myapp.com"
    assert headers["X-Title"] == "MyApp"


def test_openrouter_complete_success():
    tracker = OpenRouterTelemetryTracker()
    provider = OpenRouterProvider(
        api_key="sk-test",
        default_model="anthropic/claude-3.5-sonnet",
        telemetry=tracker,
    )

    mock_response_body = {
        "id": "gen-12345",
        "model": "anthropic/claude-3.5-sonnet:beta",
        "choices": [
            {
                "message": {"role": "assistant", "content": "Hello from OpenRouter!"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 6},
    }

    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(mock_response_body).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        req = CompletionRequest(
            tenant_id="tenant-1",
            model="anthropic/claude-3.5-sonnet",
            messages=[ChatMessage(role="user", content="hi")],
        )
        res = provider.complete(req)

        assert res.text == "Hello from OpenRouter!"
        assert res.model == "anthropic/claude-3.5-sonnet:beta"
        assert res.provider == "openrouter"
        assert res.usage.prompt_tokens == 12
        assert res.usage.completion_tokens == 6
        assert res.raw["id"] == "gen-12345"

        stats = tracker.get_or_create("anthropic/claude-3.5-sonnet:beta")
        assert stats.successful_requests == 1
        assert provider.health() is True


def test_openrouter_complete_http_error_retryable():
    tracker = OpenRouterTelemetryTracker()
    provider = OpenRouterProvider(api_key="sk-test", telemetry=tracker)

    err = urllib.error.HTTPError(
        url="https://openrouter.ai/api/v1/chat/completions",
        code=429,
        msg="Too Many Requests",
        hdrs={},
        fp=io.BytesIO(b'{"error": "rate limited"}'),
    )

    with patch("urllib.request.urlopen", side_effect=err):
        req = CompletionRequest(
            tenant_id="tenant-1",
            model="anthropic/claude-3.5-sonnet",
            messages=[ChatMessage(role="user", content="hi")],
        )
        with pytest.raises(ProviderError) as exc_info:
            provider.complete(req)

        assert exc_info.value.retryable is True
        assert exc_info.value.status_code == 429
        assert "OpenRouter HTTP 429" in str(exc_info.value)

        stats = tracker.get_or_create(provider.default_model)
        assert stats.failed_requests == 1


def test_openrouter_complete_network_error():
    tracker = OpenRouterTelemetryTracker()
    provider = OpenRouterProvider(api_key="sk-test", telemetry=tracker)

    err = urllib.error.URLError(reason="Connection refused")

    with patch("urllib.request.urlopen", side_effect=err):
        req = CompletionRequest(
            tenant_id="tenant-1",
            model="anthropic/claude-3.5-sonnet",
            messages=[ChatMessage(role="user", content="hi")],
        )
        with pytest.raises(ProviderError) as exc_info:
            provider.complete(req)

        assert exc_info.value.retryable is True
        assert "Network error" in str(exc_info.value)

        stats = tracker.get_or_create(provider.default_model)
        assert stats.failed_requests == 1
