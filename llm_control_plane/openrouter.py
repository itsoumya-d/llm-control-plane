"""OpenRouter AI routing provider, fallback hierarchies, and latency telemetry."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .types import CompletionRequest, ProviderError, ProviderResponse, Usage


@dataclass
class ModelLatencyStats:
    """Aggregated latency and reliability telemetry for an OpenRouter model route."""

    model: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    latencies_ms: List[float] = field(default_factory=list)
    ttfts_ms: List[float] = field(default_factory=list)

    @property
    def error_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests

    @property
    def avg_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        return sum(self.latencies_ms) / len(self.latencies_ms)

    @property
    def p95_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        sorted_vals = sorted(self.latencies_ms)
        idx = int(len(sorted_vals) * 0.95)
        return sorted_vals[min(idx, len(sorted_vals) - 1)]

    @property
    def avg_ttft_ms(self) -> float:
        if not self.ttfts_ms:
            return 0.0
        return sum(self.ttfts_ms) / len(self.ttfts_ms)


class OpenRouterTelemetryTracker:
    """In-memory telemetry tracker for OpenRouter model routing performance."""

    def __init__(self, window_size: int = 100) -> None:
        self.window_size = window_size
        self._stats: Dict[str, ModelLatencyStats] = {}

    def get_or_create(self, model: str) -> ModelLatencyStats:
        if model not in self._stats:
            self._stats[model] = ModelLatencyStats(model=model)
        return self._stats[model]

    def record_success(
        self,
        model: str,
        latency_ms: float,
        ttft_ms: Optional[float] = None,
    ) -> None:
        stats = self.get_or_create(model)
        stats.total_requests += 1
        stats.successful_requests += 1
        stats.latencies_ms.append(latency_ms)
        if len(stats.latencies_ms) > self.window_size:
            stats.latencies_ms.pop(0)
        if ttft_ms is not None:
            stats.ttfts_ms.append(ttft_ms)
            if len(stats.ttfts_ms) > self.window_size:
                stats.ttfts_ms.pop(0)

    def record_failure(self, model: str, latency_ms: float = 0.0) -> None:
        stats = self.get_or_create(model)
        stats.total_requests += 1
        stats.failed_requests += 1
        if latency_ms > 0:
            stats.latencies_ms.append(latency_ms)
            if len(stats.latencies_ms) > self.window_size:
                stats.latencies_ms.pop(0)

    def is_healthy(
        self,
        model: str,
        max_error_rate: float = 0.25,
        max_p95_latency_ms: float = 10000.0,
    ) -> bool:
        stats = self._stats.get(model)
        if stats is None or stats.total_requests < 3:
            return True
        if stats.error_rate > max_error_rate:
            return False
        if stats.p95_latency_ms > max_p95_latency_ms:
            return False
        return True

    def rank_models(self, candidate_models: List[str]) -> List[str]:
        """Rank candidate fallback models by health and p95 latency."""
        def sort_key(m: str) -> tuple[int, float]:
            stats = self._stats.get(m)
            if stats is None or stats.total_requests == 0:
                return (0, 0.0)  # untried models get neutral priority
            healthy = self.is_healthy(m)
            status_rank = 0 if healthy else 1
            return (status_rank, stats.avg_latency_ms)

        return sorted(candidate_models, key=sort_key)


class OpenRouterProvider:
    """OpenRouter universal model router provider adapter.

    Connects to OpenRouter's v1 chat-completions API with support for:
    - Ordered fallback model hierarchies (`models: [...]`)
    - Upstream provider preferences (ordering, ignoring untrusted providers)
    - Custom headers (HTTP-Referer, X-Title)
    - Latency and TTFT telemetry tracking
    """

    name = "openrouter"

    def __init__(
        self,
        api_key: str,
        default_model: str = "anthropic/claude-3.5-sonnet",
        fallback_models: Optional[List[str]] = None,
        site_url: Optional[str] = None,
        app_name: Optional[str] = None,
        provider_preferences: Optional[Dict[str, Any]] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout_seconds: float = 30.0,
        telemetry: Optional[OpenRouterTelemetryTracker] = None,
    ) -> None:
        self.api_key = api_key
        self.default_model = default_model
        self.fallback_models = list(fallback_models or [])
        self.site_url = site_url
        self.app_name = app_name
        self.provider_preferences = dict(provider_preferences or {})
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.telemetry = telemetry or OpenRouterTelemetryTracker()

    def build_payload(self, request: CompletionRequest) -> Dict[str, Any]:
        primary_model = request.model or self.default_model
        payload: Dict[str, Any] = {
            "model": primary_model,
            "messages": [
                {"role": m.role, "content": m.content}
                for m in request.messages
            ],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }

        # Build fallback model hierarchy if configured
        if self.fallback_models:
            all_models = [primary_model]
            for m in self.fallback_models:
                if m not in all_models:
                    all_models.append(m)
            # Rank models dynamically using telemetry
            ranked = self.telemetry.rank_models(all_models)
            payload["models"] = ranked
            payload["route"] = "fallback"

        if self.provider_preferences:
            payload["provider"] = self.provider_preferences

        return payload

    def build_headers(self) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.app_name:
            headers["X-Title"] = self.app_name
        return headers

    def complete(self, request: CompletionRequest) -> ProviderResponse:
        primary_model = request.model or self.default_model
        payload = self.build_payload(request)
        headers = self.build_headers()

        http_request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        started = time.perf_counter()
        try:
            with urllib.request.urlopen(http_request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000
            self.telemetry.record_failure(primary_model, elapsed_ms)
            is_retryable = exc.code in (429, 502, 503, 504)
            raise ProviderError(
                self.name,
                f"OpenRouter HTTP {exc.code}: {exc.reason}",
                status_code=exc.code,
                retryable=is_retryable,
            ) from exc
        except urllib.error.URLError as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000
            self.telemetry.record_failure(primary_model, elapsed_ms)
            raise ProviderError(self.name, f"Network error: {exc.reason}", retryable=True) from exc

        elapsed_ms = (time.perf_counter() - started) * 1000
        served_model = body.get("model", primary_model)
        self.telemetry.record_success(served_model, elapsed_ms)

        choices = body.get("choices") or []
        text = choices[0].get("message", {}).get("content", "") if choices else ""
        usage = body.get("usage") or {}

        return ProviderResponse(
            text=text,
            model=served_model,
            provider=self.name,
            usage=Usage(
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                completion_tokens=int(usage.get("completion_tokens", 0)),
            ),
            latency_ms=elapsed_ms,
            raw=body,
        )

    def health(self) -> bool:
        return self.telemetry.is_healthy(self.default_model)
