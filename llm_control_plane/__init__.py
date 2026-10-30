"""LLM Control Plane: gateway, evaluations, cost attribution, and audit trails."""

from .audit import AuditEvent, AuditLog
from .evals import (
    ContainsScorer,
    Dataset,
    EvalCase,
    EvalResult,
    EvalRun,
    EvalRunner,
    ExactMatchScorer,
    ExperimentComparison,
    JsonSchemaScorer,
    LatencyBudgetScorer,
    NumericToleranceScorer,
    Score,
    compare_runs,
    render_markdown,
)
from .gateway import GatewayConfig, LLMGateway
from .metrics import MetricsRegistry
from .policy import PolicyDecision, PolicyEngine, PolicyViolation
from .pricing import Price, Pricebook, default_pricebook
from .providers import EchoProvider, FakeProvider, RuleProvider, provider_map
from .rate_limit import TenantRateLimiter, TokenBucket
from .redaction import RedactionResult, Redactor
from .routing import Route, RoutingTable
from .types import (
    ChatMessage,
    CompletionRequest,
    GatewayResult,
    ProviderError,
    ProviderResponse,
    RateLimitExceeded,
    RouteTarget,
    Usage,
)

__version__ = "1.0.0"

__all__ = [
    "AuditEvent",
    "AuditLog",
    "ChatMessage",
    "CompletionRequest",
    "ContainsScorer",
    "Dataset",
    "EchoProvider",
    "EvalCase",
    "EvalResult",
    "EvalRun",
    "EvalRunner",
    "ExactMatchScorer",
    "ExperimentComparison",
    "FakeProvider",
    "GatewayConfig",
    "GatewayResult",
    "JsonSchemaScorer",
    "LatencyBudgetScorer",
    "LLMGateway",
    "MetricsRegistry",
    "NumericToleranceScorer",
    "PolicyDecision",
    "PolicyEngine",
    "PolicyViolation",
    "Price",
    "Pricebook",
    "ProviderError",
    "ProviderResponse",
    "RateLimitExceeded",
    "RedactionResult",
    "Redactor",
    "Route",
    "RouteTarget",
    "RoutingTable",
    "RuleProvider",
    "Score",
    "TenantRateLimiter",
    "TokenBucket",
    "Usage",
    "compare_runs",
    "default_pricebook",
    "provider_map",
    "render_markdown",
]
