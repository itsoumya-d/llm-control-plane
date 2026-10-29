"""Shared request, response, and error types for the control plane."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class ProviderError(Exception):
    """A provider failure with enough metadata for routing decisions."""

    def __init__(
        self,
        provider: str,
        message: str,
        status_code: Optional[int] = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.message = message
        self.status_code = status_code
        self.retryable = retryable


class RateLimitExceeded(Exception):
    """Raised when a tenant exhausts its token bucket."""

    def __init__(self, tenant_id: str, retry_after_seconds: float) -> None:
        super().__init__(
            f"tenant {tenant_id!r} is rate limited; retry in {retry_after_seconds:.3f}s"
        )
        self.tenant_id = tenant_id
        self.retry_after_seconds = retry_after_seconds


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


@dataclass(frozen=True)
class CompletionRequest:
    tenant_id: str
    model: str
    messages: List[ChatMessage]
    temperature: float = 0.0
    max_tokens: int = 512
    use_case: str = "default"
    metadata: Dict[str, str] = field(default_factory=dict)
    trace_id: Optional[str] = None


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True)
class ProviderResponse:
    text: str
    model: str
    provider: str
    usage: Usage
    latency_ms: float = 0.0
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RouteTarget:
    provider: str
    model: str


@dataclass(frozen=True)
class GatewayResult:
    text: str
    provider: str
    model: str
    usage: Usage
    cost_usd: float
    latency_ms: float
    trace_id: str
    attempts: int
    redactions: Dict[str, int]
    audit_seq: int
