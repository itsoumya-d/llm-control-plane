"""Pre-flight policy checks for model, tenant, token, and use-case constraints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Optional, Tuple

from .types import CompletionRequest


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    policy_id: str
    reasons: Tuple[str, ...] = ()

    @property
    def reason(self) -> str:
        return "; ".join(self.reasons)


class PolicyViolation(Exception):
    """Raised when a request is denied before it reaches a provider."""

    def __init__(self, decision: PolicyDecision) -> None:
        super().__init__(f"request denied by {decision.policy_id}: {decision.reason}")
        self.decision = decision


@dataclass(frozen=True)
class PolicyEngine:
    policy_id: str = "default"
    allowed_models: Optional[FrozenSet[str]] = None
    allowed_use_cases: Optional[FrozenSet[str]] = None
    max_tokens: int = 4096
    require_tenant: bool = True
    require_trace_id: bool = False
    blocked_metadata_keys: FrozenSet[str] = frozenset()

    def evaluate(self, request: CompletionRequest) -> PolicyDecision:
        reasons = []
        if self.require_tenant and not request.tenant_id.strip():
            reasons.append("tenant_id is required")
        if self.allowed_models is not None and request.model not in self.allowed_models:
            reasons.append(f"model {request.model!r} is not allowed")
        if self.allowed_use_cases is not None and request.use_case not in self.allowed_use_cases:
            reasons.append(f"use_case {request.use_case!r} is not allowed")
        if request.max_tokens < 1:
            reasons.append("max_tokens must be positive")
        if request.max_tokens > self.max_tokens:
            reasons.append(f"max_tokens {request.max_tokens} exceeds policy cap {self.max_tokens}")
        if self.require_trace_id and not request.trace_id:
            reasons.append("trace_id is required")
        for key in request.metadata:
            if key in self.blocked_metadata_keys:
                reasons.append(f"metadata key {key!r} is blocked")
        return PolicyDecision(
            allowed=not reasons,
            policy_id=self.policy_id,
            reasons=tuple(reasons),
        )

    def enforce(self, request: CompletionRequest) -> PolicyDecision:
        decision = self.evaluate(request)
        if not decision.allowed:
            raise PolicyViolation(decision)
        return decision
