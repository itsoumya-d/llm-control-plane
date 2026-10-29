"""Provider-agnostic LLM gateway with failover, cost attribution, and audit trails."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional

from .audit import AuditLog, canonical_json, sha256_hex
from .metrics import MetricsRegistry
from .pricing import Pricebook, default_pricebook
from .rate_limit import TenantRateLimiter
from .redaction import Redactor
from .routing import RoutingTable
from .types import (
    ChatMessage,
    CompletionRequest,
    GatewayResult,
    ProviderError,
    RateLimitExceeded,
)


@dataclass(frozen=True)
class GatewayConfig:
    max_attempts_per_target: int = 2
    base_backoff_seconds: float = 0.05
    max_backoff_seconds: float = 0.5
    rate_limit_capacity: float = 60.0
    rate_limit_refill_per_second: float = 1.0
    redact_pii: bool = True


class LLMGateway:
    def __init__(
        self,
        providers: Dict[str, object],
        routing: RoutingTable,
        pricebook: Optional[Pricebook] = None,
        audit: Optional[AuditLog] = None,
        limiter: Optional[TenantRateLimiter] = None,
        metrics: Optional[MetricsRegistry] = None,
        config: Optional[GatewayConfig] = None,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
        trace_id_factory: Optional[Callable[[], str]] = None,
    ) -> None:
        self.providers = dict(providers)
        self.routing = routing
        self.pricebook = pricebook or default_pricebook()
        self.audit = audit or AuditLog()
        self.metrics = metrics or MetricsRegistry()
        self.config = config or GatewayConfig()
        self.limiter = limiter or TenantRateLimiter(
            capacity=self.config.rate_limit_capacity,
            refill_per_second=self.config.rate_limit_refill_per_second,
        )
        self.clock = clock
        self.sleeper = sleeper
        self.redactor = Redactor()
        self._trace_id_factory = trace_id_factory or (lambda: uuid.uuid4().hex)
        self._results: Dict[tuple, GatewayResult] = {}
        self._lock = threading.Lock()

    def _redact(self, messages: List[ChatMessage]) -> tuple:
        if not self.config.redact_pii:
            return list(messages), {}
        redacted_texts, counts = self.redactor.redact_many([message.content for message in messages])
        redacted = [
            ChatMessage(role=message.role, content=text)
            for message, text in zip(messages, redacted_texts)
        ]
        return redacted, counts

    def _cached(self, tenant_id: str, trace_id: Optional[str]) -> Optional[GatewayResult]:
        if not trace_id:
            return None
        with self._lock:
            return self._results.get((tenant_id, trace_id))

    def _store(self, tenant_id: str, trace_id: str, result: GatewayResult) -> None:
        with self._lock:
            self._results[(tenant_id, trace_id)] = result

    def complete(self, request: CompletionRequest) -> GatewayResult:
        trace_id = request.trace_id or self._trace_id_factory()
        cached = self._cached(request.tenant_id, request.trace_id)
        if cached is not None:
            self.metrics.inc(
                "gateway_idempotent_replays_total",
                {"tenant": request.tenant_id, "model": request.model},
            )
            return cached

        if not self.limiter.try_acquire(request.tenant_id):
            retry_after = self.limiter.retry_after(request.tenant_id)
            self.metrics.inc(
                "gateway_rate_limited_total",
                {"tenant": request.tenant_id, "model": request.model},
            )
            self.audit.append(
                "request.rate_limited",
                {
                    "trace_id": trace_id,
                    "tenant_id": request.tenant_id,
                    "model": request.model,
                    "retry_after_seconds": retry_after,
                },
            )
            raise RateLimitExceeded(request.tenant_id, retry_after)

        redacted_messages, redactions = self._redact(list(request.messages))
        sanitized = replace(request, messages=redacted_messages, trace_id=trace_id)
        prompt_hash = sha256_hex(
            canonical_json(
                {
                    "messages": [
                        {"role": message.role, "content": message.content}
                        for message in redacted_messages
                    ]
                }
            )
        )

        self.metrics.inc(
            "gateway_requests_total",
            {"tenant": request.tenant_id, "model": request.model, "use_case": request.use_case},
        )
        for kind, count in redactions.items():
            self.metrics.inc("gateway_redactions_total", {"kind": kind}, float(count))

        self.audit.append(
            "request.accepted",
            {
                "trace_id": trace_id,
                "tenant_id": request.tenant_id,
                "model": request.model,
                "use_case": request.use_case,
                "message_count": len(redacted_messages),
                "prompt_sha256": prompt_hash,
                "redactions": redactions,
            },
        )

        try:
            targets = self.routing.resolve(request.model)
        except KeyError as exc:
            self.audit.append(
                "request.failed",
                {"trace_id": trace_id, "reason": str(exc), "attempts": 0},
            )
            self.metrics.inc(
                "gateway_failures_total",
                {"tenant": request.tenant_id, "model": request.model, "reason": "unknown_alias"},
            )
            raise ProviderError("gateway", str(exc), retryable=False) from exc

        errors: List[str] = []
        attempts = 0
        for target in targets:
            provider = self.providers.get(target.provider)
            if provider is None:
                errors.append(f"{target.provider}/{target.model}: provider not registered")
                self.audit.append(
                    "provider.missing",
                    {"trace_id": trace_id, "provider": target.provider, "model": target.model},
                )
                continue

            for attempt in range(self.config.max_attempts_per_target):
                attempts += 1
                started = self.clock()
                self.audit.append(
                    "provider.attempt",
                    {
                        "trace_id": trace_id,
                        "provider": target.provider,
                        "model": target.model,
                        "attempt": attempt + 1,
                    },
                )
                self.metrics.inc(
                    "provider_attempts_total",
                    {"provider": target.provider, "model": target.model},
                )
                try:
                    response = provider.complete(sanitized)  # type: ignore[attr-defined]
                except ProviderError as exc:
                    measured_ms = max(0.0, (self.clock() - started) * 1000.0)
                    errors.append(f"{target.provider}/{target.model}: {exc.message}")
                    self.metrics.inc(
                        "provider_failures_total",
                        {
                            "provider": target.provider,
                            "model": target.model,
                            "retryable": str(exc.retryable).lower(),
                        },
                    )
                    self.audit.append(
                        "provider.failure",
                        {
                            "trace_id": trace_id,
                            "provider": target.provider,
                            "model": target.model,
                            "attempt": attempt + 1,
                            "retryable": exc.retryable,
                            "status_code": exc.status_code,
                            "error": exc.message,
                            "latency_ms": measured_ms,
                        },
                    )
                    if not exc.retryable:
                        break
                    if attempt + 1 < self.config.max_attempts_per_target:
                        delay = min(
                            self.config.max_backoff_seconds,
                            self.config.base_backoff_seconds * (2 ** attempt),
                        )
                        self.sleeper(delay)
                    continue
                except Exception as exc:  # pragma: no cover - defensive boundary
                    errors.append(f"{target.provider}/{target.model}: {exc!r}")
                    self.audit.append(
                        "provider.failure",
                        {
                            "trace_id": trace_id,
                            "provider": target.provider,
                            "model": target.model,
                            "attempt": attempt + 1,
                            "retryable": False,
                            "error": repr(exc),
                        },
                    )
                    break

                measured_ms = max(0.0, (self.clock() - started) * 1000.0)
                latency_ms = max(measured_ms, float(response.latency_ms))
                cost = self.pricebook.cost(target.provider, target.model, response.usage)
                self.metrics.inc(
                    "provider_success_total",
                    {"provider": target.provider, "model": target.model},
                )
                self.metrics.observe(
                    "provider_latency_ms",
                    latency_ms,
                    {"provider": target.provider, "model": target.model},
                )
                self.metrics.inc(
                    "provider_tokens_total",
                    {"provider": target.provider, "model": target.model, "kind": "prompt"},
                    float(response.usage.prompt_tokens),
                )
                self.metrics.inc(
                    "provider_tokens_total",
                    {"provider": target.provider, "model": target.model, "kind": "completion"},
                    float(response.usage.completion_tokens),
                )
                self.metrics.inc(
                    "provider_cost_usd_total",
                    {"provider": target.provider, "model": target.model},
                    cost,
                )
                self.metrics.inc(
                    "tenant_cost_usd_total",
                    {"tenant": request.tenant_id, "use_case": request.use_case},
                    cost,
                )
                success_event = self.audit.append(
                    "provider.success",
                    {
                        "trace_id": trace_id,
                        "provider": target.provider,
                        "model": target.model,
                        "attempt": attempt + 1,
                        "usage": {
                            "prompt_tokens": response.usage.prompt_tokens,
                            "completion_tokens": response.usage.completion_tokens,
                        },
                        "cost_usd": cost,
                        "latency_ms": latency_ms,
                    },
                )
                result = GatewayResult(
                    text=response.text,
                    provider=target.provider,
                    model=response.model,
                    usage=response.usage,
                    cost_usd=cost,
                    latency_ms=latency_ms,
                    trace_id=trace_id,
                    attempts=attempts,
                    redactions=redactions,
                    audit_seq=success_event.seq,
                )
                self.audit.append(
                    "request.completed",
                    {
                        "trace_id": trace_id,
                        "tenant_id": request.tenant_id,
                        "provider": result.provider,
                        "model": result.model,
                        "cost_usd": result.cost_usd,
                        "latency_ms": result.latency_ms,
                        "attempts": result.attempts,
                    },
                )
                self._store(request.tenant_id, trace_id, result)
                return result

        self.audit.append(
            "request.failed",
            {"trace_id": trace_id, "attempts": attempts, "errors": errors},
        )
        self.metrics.inc(
            "gateway_failures_total",
            {"tenant": request.tenant_id, "model": request.model, "reason": "providers_exhausted"},
        )
        raise ProviderError(
            "gateway",
            "all provider targets failed: " + "; ".join(errors),
            retryable=False,
        )

    def health(self) -> Dict[str, bool]:
        return {
            name: bool(provider.health())  # type: ignore[attr-defined]
            for name, provider in self.providers.items()
        }
