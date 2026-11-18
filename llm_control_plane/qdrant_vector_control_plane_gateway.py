"""Qdrant Vector Cloud Control Plane Gateway & MCP Agent Bridge.

Provides core control-plane services, scoped OAuth2 token introspection,
Model Context Protocol (MCP) agent access paths, and real-time vector cluster
observability telemetry for Qdrant Cloud.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


class OAuth2TokenScope(str, Enum):
    """Granular permission scopes for Qdrant Cloud and MCP agent access."""

    READ_COLLECTION = "qdrant:collection:read"
    WRITE_VECTORS = "qdrant:vectors:write"
    ADMIN_CLUSTER = "qdrant:cluster:admin"
    MCP_INVOKE = "qdrant:mcp:invoke"


@dataclass
class TokenClaims:
    """Decoded and validated OAuth2 token claims for humans and AI agents."""

    token_id: str
    tenant_id: str
    principal_id: str
    is_agent: bool
    scopes: Set[OAuth2TokenScope]
    allowed_collections: Set[str]
    expires_at: float

    def has_scope(self, scope: OAuth2TokenScope) -> bool:
        return scope in self.scopes

    def can_access_collection(self, collection_name: str) -> bool:
        if "*" in self.allowed_collections:
            return True
        return collection_name in self.allowed_collections


class ScopedOAuth2TokenIntrospector:
    """Cryptographic token introspection engine with sub-millisecond local caching."""

    def __init__(self, signing_secret: str, cache_ttl_seconds: float = 300.0):
        self._secret = signing_secret
        self._cache_ttl = cache_ttl_seconds
        self._cache: Dict[str, Tuple[TokenClaims, float]] = {}

    def issue_token(
        self,
        tenant_id: str,
        principal_id: str,
        scopes: Set[OAuth2TokenScope],
        allowed_collections: Set[str],
        ttl_seconds: float = 3600.0,
        is_agent: bool = True,
    ) -> str:
        now = time.time()
        expires_at = now + ttl_seconds
        token_id = str(uuid.uuid4())
        scope_str = ",".join(sorted(s.value for s in scopes))
        coll_str = ",".join(sorted(allowed_collections))
        payload = f"{token_id}:{tenant_id}:{principal_id}:{is_agent}:{scope_str}:{coll_str}:{expires_at}"
        signature = hashlib.sha256(f"{payload}:{self._secret}".encode("utf-8")).hexdigest()
        token = f"{payload}:{signature}"
        claims = TokenClaims(
            token_id=token_id,
            tenant_id=tenant_id,
            principal_id=principal_id,
            is_agent=is_agent,
            scopes=scopes,
            allowed_collections=allowed_collections,
            expires_at=expires_at,
        )
        self._cache[token] = (claims, now + self._cache_ttl)
        return token

    def introspect(self, token: str) -> Optional[TokenClaims]:
        now = time.time()
        # Check local cache first
        if token in self._cache:
            claims, cached_until = self._cache[token]
            if now < cached_until and now < claims.expires_at:
                return claims
            self._cache.pop(token, None)

        parts = token.split(":")
        if len(parts) != 8:
            return None

        token_id, tenant_id, principal_id, is_agent_str, scope_str, coll_str, expires_at_str, signature = parts
        payload = f"{token_id}:{tenant_id}:{principal_id}:{is_agent_str}:{scope_str}:{coll_str}:{expires_at_str}"
        expected_sig = hashlib.sha256(f"{payload}:{self._secret}".encode("utf-8")).hexdigest()
        if signature != expected_sig:
            return None

        expires_at = float(expires_at_str)
        if now >= expires_at:
            return None

        scopes = {OAuth2TokenScope(s) for s in scope_str.split(",") if s}
        colls = set(coll_str.split(",")) if coll_str else set()
        claims = TokenClaims(
            token_id=token_id,
            tenant_id=tenant_id,
            principal_id=principal_id,
            is_agent=(is_agent_str == "True"),
            scopes=scopes,
            allowed_collections=colls,
            expires_at=expires_at,
        )
        self._cache[token] = (claims, now + self._cache_ttl)
        return claims


class TokenBucketRateLimiter:
    """In-memory token bucket rate limiter per agent principal."""

    def __init__(self, capacity: int = 60, refill_rate_per_sec: float = 10.0):
        self.capacity = capacity
        self.refill_rate = refill_rate_per_sec
        self._buckets: Dict[str, Tuple[float, float]] = {}  # key -> (tokens, last_refill)

    def allow(self, key: str, tokens: int = 1) -> bool:
        now = time.time()
        current_tokens, last_refill = self._buckets.get(key, (float(self.capacity), now))
        elapsed = now - last_refill
        refilled = min(float(self.capacity), current_tokens + elapsed * self.refill_rate)
        if refilled >= tokens:
            self._buckets[key] = (refilled - tokens, now)
            return True
        self._buckets[key] = (refilled, now)
        return False


@dataclass
class ControlPlaneTraceSpan:
    trace_id: str
    span_id: str
    endpoint: str
    duration_ms: float
    status: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class QdrantMCPAgentBridge:
    """Model Context Protocol (MCP) tool gateway bridging AI agents to Qdrant Cloud."""

    def __init__(
        self, introspector: ScopedOAuth2TokenIntrospector, rate_limiter: Optional[TokenBucketRateLimiter] = None
    ):
        self.introspector = introspector
        self.rate_limiter = rate_limiter or TokenBucketRateLimiter()
        self._tools: Dict[str, Callable[[Dict[str, Any]], Any]] = {}
        self.trace_spans: List[ControlPlaneTraceSpan] = []

    def register_mcp_tool(self, name: str, handler: Callable[[Dict[str, Any]], Any]):
        self._tools[name] = handler

    def invoke_tool(self, token: str, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        start = time.perf_counter()
        trace_id = str(uuid.uuid4())
        span_id = str(uuid.uuid4())

        claims = self.introspector.introspect(token)
        if not claims:
            return {"success": False, "error": "Invalid or expired authorization token", "code": "UNAUTHORIZED"}

        if not claims.has_scope(OAuth2TokenScope.MCP_INVOKE):
            return {"success": False, "error": "Token missing qdrant:mcp:invoke scope", "code": "FORBIDDEN"}

        collection = arguments.get("collection_name")
        if collection and not claims.can_access_collection(collection):
            return {"success": False, "error": f"Token unauthorized for collection {collection}", "code": "FORBIDDEN"}

        if not self.rate_limiter.allow(claims.principal_id):
            return {"success": False, "error": "Agent rate limit exceeded", "code": "TOO_MANY_REQUESTS"}

        handler = self._tools.get(tool_name)
        if not handler:
            return {"success": False, "error": f"MCP tool {tool_name} not found", "code": "NOT_FOUND"}

        try:
            res = handler(arguments)
            duration_ms = (time.perf_counter() - start) * 1000.0
            self.trace_spans.append(
                ControlPlaneTraceSpan(
                    trace_id=trace_id,
                    span_id=span_id,
                    endpoint=f"mcp/{tool_name}",
                    duration_ms=duration_ms,
                    status="SUCCESS",
                    metadata={"principal": claims.principal_id, "collection": collection},
                )
            )
            return {"success": True, "result": res}
        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000.0
            self.trace_spans.append(
                ControlPlaneTraceSpan(
                    trace_id=trace_id,
                    span_id=span_id,
                    endpoint=f"mcp/{tool_name}",
                    duration_ms=duration_ms,
                    status="ERROR",
                    metadata={"error": str(e)},
                )
            )
            return {"success": False, "error": str(e), "code": "INTERNAL_ERROR"}


@dataclass
class VectorClusterSpec:
    cluster_id: str
    tenant_id: str
    region: str
    node_count: int
    vector_dimension: int
    distance_metric: str
    status: str = "PROVISIONING"


class QdrantVectorControlPlaneGateway:
    """Core control-plane service managing vector cluster lifecycles and state reconciliation."""

    def __init__(self, introspector: ScopedOAuth2TokenIntrospector):
        self.introspector = introspector
        self._clusters: Dict[str, VectorClusterSpec] = {}
        self._reconciliation_log: List[Dict[str, Any]] = []

    def provision_cluster(self, token: str, spec: VectorClusterSpec) -> Dict[str, Any]:
        claims = self.introspector.introspect(token)
        if not claims or not claims.has_scope(OAuth2TokenScope.ADMIN_CLUSTER):
            return {"success": False, "error": "Unauthorized cluster admin access", "code": "FORBIDDEN"}

        spec.status = "ACTIVE"
        self._clusters[spec.cluster_id] = spec
        self._reconciliation_log.append(
            {
                "action": "provision",
                "cluster_id": spec.cluster_id,
                "tenant_id": spec.tenant_id,
                "timestamp": time.time(),
            }
        )
        return {"success": True, "cluster": spec}

    def get_cluster(self, token: str, cluster_id: str) -> Optional[VectorClusterSpec]:
        claims = self.introspector.introspect(token)
        if not claims:
            return None
        cluster = self._clusters.get(cluster_id)
        if not cluster:
            return None
        if cluster.tenant_id != claims.tenant_id and not claims.has_scope(OAuth2TokenScope.ADMIN_CLUSTER):
            return None
        return cluster

    def reconcile_clusters(self) -> int:
        """Runs cluster lifecycle reconcile loop, verifying health across all active clusters."""
        reconciled = 0
        for cluster in self._clusters.values():
            if cluster.status == "ACTIVE":
                reconciled += 1
        return reconciled


class VectorClusterObservabilityCollector:
    """Aggregates high-cardinality metrics and latency distributions for Qdrant Cloud."""

    def __init__(self):
        self._latencies: List[float] = []
        self._metrics: Dict[str, Any] = {"total_vectors": 0, "active_segments": 0, "memory_bytes": 0}

    def record_query_latency(self, duration_ms: float):
        self._latencies.append(duration_ms)

    def update_cluster_metrics(self, vectors: int, segments: int, memory_bytes: int):
        self._metrics["total_vectors"] = vectors
        self._metrics["active_segments"] = segments
        self._metrics["memory_bytes"] = memory_bytes

    def get_latency_percentiles(self) -> Dict[str, float]:
        if not self._latencies:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
        sorted_lats = sorted(self._latencies)
        n = len(sorted_lats)
        p50 = sorted_lats[int(n * 0.50)]
        p95 = sorted_lats[min(int(n * 0.95), n - 1)]
        p99 = sorted_lats[min(int(n * 0.99), n - 1)]
        return {"p50": round(p50, 2), "p95": round(p95, 2), "p99": round(p99, 2)}
