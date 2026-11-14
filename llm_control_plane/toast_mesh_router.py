"""Toast Service Mesh Router & Dynamic Traffic Control Plane.

Implements an xDS-inspired dynamic service mesh routing engine for high-scale,
distributed microservice architectures (such as Toast POS, order processing,
kitchen display systems, and payment services).

Key Capabilities:
- Dynamic cluster discovery and endpoint registry with locality awareness.
- Outlier detection and passive health checking (consecutive 5xx ejection).
- Weighted canary traffic splitting and header-based route matching.
- Multi-state circuit breaking (CLOSED, OPEN, HALF_OPEN) with half-open probing.
- Adaptive load shedding across request priority tiers (CRITICAL, STANDARD,
  BEST_EFFORT) to protect vital POS transaction flows during meal rushes.
- Latency percentiles tracking (p50, p95, p99) and telemetry aggregation.
"""

from __future__ import annotations

import enum
import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


class EndpointHealth(str, enum.Enum):
    """Health status of a service endpoint."""

    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNHEALTHY = "UNHEALTHY"
    EJECTED = "EJECTED"


class PriorityTier(str, enum.Enum):
    """Request priority tier for adaptive load shedding."""

    CRITICAL = "CRITICAL"  # POS order creation, card authorization
    STANDARD = "STANDARD"  # Menu lookup, table status updates
    BEST_EFFORT = "BEST_EFFORT"  # Background telemetry, non-vital analytics


class CircuitState(str, enum.Enum):
    """Three-state circuit breaker representation."""

    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


@dataclass
class ServiceEndpoint:
    """Individual service instance in a cluster."""

    endpoint_id: str
    host: str
    port: int
    zone: str = "eu-west-1a"
    weight: int = 100
    health: EndpointHealth = EndpointHealth.HEALTHY
    active_requests: int = 0
    total_requests: int = 0
    failed_requests: int = 0
    consecutive_failures: int = 0
    latencies_ms: List[float] = field(default_factory=list)
    ejected_until: float = 0.0

    def record_success(self, latency_ms: float) -> None:
        """Record successful invocation."""
        self.active_requests = max(0, self.active_requests - 1)
        self.total_requests += 1
        self.consecutive_failures = 0
        self.latencies_ms.append(latency_ms)
        if len(self.latencies_ms) > 1000:
            self.latencies_ms.pop(0)

    def record_failure(self, latency_ms: float, error_status: int = 500) -> None:
        """Record failed invocation and check consecutive failure threshold."""
        self.active_requests = max(0, self.active_requests - 1)
        self.total_requests += 1
        self.failed_requests += 1
        self.consecutive_failures += 1
        self.latencies_ms.append(latency_ms)
        if len(self.latencies_ms) > 1000:
            self.latencies_ms.pop(0)

    def is_available(self, current_time: float) -> bool:
        """Return True if endpoint is healthy or past ejection duration."""
        if self.health == EndpointHealth.EJECTED:
            if current_time >= self.ejected_until:
                self.health = EndpointHealth.HEALTHY
                self.consecutive_failures = 0
                return True
            return False
        return self.health in (EndpointHealth.HEALTHY, EndpointHealth.DEGRADED)


@dataclass
class OutlierDetectionConfig:
    """Configuration for outlier detection and passive ejection."""

    consecutive_5xx_limit: int = 5
    ejection_duration_seconds: float = 30.0
    max_ejection_percent: float = 50.0  # At least 50% endpoints must remain


class CircuitBreaker:
    """Three-state circuit breaker with rate-based tripping and recovery."""

    def __init__(
        self,
        failure_rate_threshold: float = 0.5,
        min_requests_threshold: int = 10,
        recovery_timeout_seconds: float = 15.0,
        half_open_success_threshold: int = 3,
    ) -> None:
        self.failure_rate_threshold = failure_rate_threshold
        self.min_requests_threshold = min_requests_threshold
        self.recovery_timeout_seconds = recovery_timeout_seconds
        self.half_open_success_threshold = half_open_success_threshold

        self.state = CircuitState.CLOSED
        self.opened_at: float = 0.0
        self.half_open_successes: int = 0
        self.total_calls: int = 0
        self.failed_calls: int = 0

    def allow_request(self, current_time: float) -> bool:
        """Determine whether an incoming request is allowed through."""
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            if current_time - self.opened_at >= self.recovery_timeout_seconds:
                self.state = CircuitState.HALF_OPEN
                self.half_open_successes = 0
                return True
            return False
        if self.state == CircuitState.HALF_OPEN:
            # Allow limited probe requests
            return True
        return False

    def record_result(self, success: bool, current_time: float) -> None:
        """Update circuit breaker state based on invocation outcome."""
        self.total_calls += 1
        if not success:
            self.failed_calls += 1

        if self.state == CircuitState.HALF_OPEN:
            if success:
                self.half_open_successes += 1
                if self.half_open_successes >= self.half_open_success_threshold:
                    self.state = CircuitState.CLOSED
                    self.total_calls = 0
                    self.failed_calls = 0
            else:
                self.state = CircuitState.OPEN
                self.opened_at = current_time
        elif self.state == CircuitState.CLOSED:
            if self.total_calls >= self.min_requests_threshold:
                rate = self.failed_calls / self.total_calls
                if rate >= self.failure_rate_threshold:
                    self.state = CircuitState.OPEN
                    self.opened_at = current_time


@dataclass
class RouteDestination:
    """Route target destination cluster with weight."""

    cluster_name: str
    weight: int = 100


@dataclass
class RouteRule:
    """HTTP/gRPC route rule with matching headers, prefixes, and canary splits."""

    route_id: str
    path_prefix: str
    destinations: List[RouteDestination]
    headers_match: Dict[str, str] = field(default_factory=dict)
    timeout_ms: float = 2500.0
    retry_count: int = 2

    def matches(self, path: str, headers: Dict[str, str]) -> bool:
        """Check if request matches path prefix and required headers."""
        if not path.startswith(self.path_prefix):
            return False
        for k, v in self.headers_match.items():
            if headers.get(k) != v:
                return False
        return True

    def select_destination(self) -> str:
        """Perform weighted random selection among target destinations."""
        total_weight = sum(d.weight for d in self.destinations)
        if total_weight <= 0:
            return self.destinations[0].cluster_name
        pick = random.randint(1, total_weight)
        current = 0
        for d in self.destinations:
            current += d.weight
            if pick <= current:
                return d.cluster_name
        return self.destinations[-1].cluster_name


class ServiceCluster:
    """Logical cluster of endpoints for a microservice."""

    def __init__(
        self,
        name: str,
        outlier_config: Optional[OutlierDetectionConfig] = None,
        circuit_breaker: Optional[CircuitBreaker] = None,
    ) -> None:
        self.name = name
        self.endpoints: Dict[str, ServiceEndpoint] = {}
        self.outlier_config = outlier_config or OutlierDetectionConfig()
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        self.round_robin_index = 0

    def add_endpoint(self, endpoint: ServiceEndpoint) -> None:
        """Register or update an endpoint in the cluster."""
        self.endpoints[endpoint.endpoint_id] = endpoint

    def remove_endpoint(self, endpoint_id: str) -> None:
        """Deregister an endpoint from the cluster."""
        self.endpoints.pop(endpoint_id, None)

    def get_healthy_endpoints(self, current_time: float) -> List[ServiceEndpoint]:
        """Return all currently available endpoints."""
        healthy = []
        for ep in self.endpoints.values():
            if ep.is_available(current_time):
                healthy.append(ep)
        return healthy

    def select_endpoint(
        self,
        current_time: float,
        preferred_zone: Optional[str] = None,
        strategy: str = "least_request",
    ) -> Optional[ServiceEndpoint]:
        """Select an endpoint using specified load balancing strategy."""
        healthy = self.get_healthy_endpoints(current_time)
        if not healthy:
            return None

        # Filter by zone if locality routing preferred and matching endpoints exist
        if preferred_zone:
            zone_eps = [ep for ep in healthy if ep.zone == preferred_zone]
            if zone_eps:
                healthy = zone_eps

        if strategy == "least_request":
            return min(healthy, key=lambda ep: ep.active_requests)
        if strategy == "round_robin":
            self.round_robin_index = (self.round_robin_index + 1) % len(healthy)
            return healthy[self.round_robin_index]

        # Default weighted random
        total_w = sum(ep.weight for ep in healthy)
        pick = random.randint(1, total_w)
        curr = 0
        for ep in healthy:
            curr += ep.weight
            if pick <= curr:
                return ep
        return healthy[0]

    def check_outliers(self, current_time: float) -> List[str]:
        """Evaluate endpoints for consecutive 5xx failures and apply ejection."""
        ejected_ids = []
        total = len(self.endpoints)
        if total == 0:
            return ejected_ids

        max_ejections = math.floor(total * (self.outlier_config.max_ejection_percent / 100.0))
        currently_ejected = sum(
            1 for ep in self.endpoints.values() if ep.health == EndpointHealth.EJECTED
        )

        for ep in self.endpoints.values():
            if ep.health != EndpointHealth.EJECTED:
                if ep.consecutive_failures >= self.outlier_config.consecutive_5xx_limit:
                    if currently_ejected < max_ejections:
                        ep.health = EndpointHealth.EJECTED
                        ep.ejected_until = current_time + self.outlier_config.ejection_duration_seconds
                        currently_ejected += 1
                        ejected_ids.append(ep.endpoint_id)
        return ejected_ids


class ToastMeshRouter:
    """Dynamic service mesh router and traffic control plane."""

    def __init__(
        self,
        concurrency_ceiling: int = 1000,
        load_shed_threshold_percent: float = 85.0,
    ) -> None:
        self.clusters: Dict[str, ServiceCluster] = {}
        self.routes: List[RouteRule] = []
        self.concurrency_ceiling = concurrency_ceiling
        self.load_shed_threshold_percent = load_shed_threshold_percent
        self.current_concurrency = 0
        self.total_routed_requests = 0
        self.total_shed_requests = 0

    def register_cluster(self, cluster: ServiceCluster) -> None:
        """Add or update a service cluster."""
        self.clusters[cluster.name] = cluster

    def add_route(self, route: RouteRule) -> None:
        """Add a route rule to the routing table."""
        self.routes.append(route)

    def route_request(
        self,
        path: str,
        headers: Dict[str, str],
        priority: PriorityTier = PriorityTier.STANDARD,
        preferred_zone: Optional[str] = None,
        current_time: Optional[float] = None,
    ) -> Tuple[bool, Optional[str], Optional[ServiceEndpoint], str]:
        """Route an incoming request through matching rules, load shedding, and circuit breaking.

        Returns:
            (allowed, target_cluster_name, target_endpoint, reason_or_status)
        """
        now = current_time if current_time is not None else time.time()

        # 1. Adaptive Load Shedding Check
        load_ratio = (self.current_concurrency / max(1, self.concurrency_ceiling)) * 100.0
        if load_ratio >= self.load_shed_threshold_percent:
            if priority == PriorityTier.BEST_EFFORT:
                self.total_shed_requests += 1
                return False, None, None, "LOAD_SHED_BEST_EFFORT"
            if load_ratio >= 98.0 and priority == PriorityTier.STANDARD:
                self.total_shed_requests += 1
                return False, None, None, "LOAD_SHED_STANDARD"

        # 2. Match Route Rule
        matched_rule: Optional[RouteRule] = None
        for rule in self.routes:
            if rule.matches(path, headers):
                matched_rule = rule
                break

        if not matched_rule:
            return False, None, None, "NO_MATCHING_ROUTE"

        # 3. Select Target Cluster (Canary / Weighted)
        target_cluster_name = matched_rule.select_destination()
        cluster = self.clusters.get(target_cluster_name)
        if not cluster:
            return False, None, None, f"CLUSTER_NOT_FOUND_{target_cluster_name}"

        # 4. Check Circuit Breaker
        if not cluster.circuit_breaker.allow_request(now):
            return False, target_cluster_name, None, "CIRCUIT_BREAKER_OPEN"

        # 5. Check and Run Outlier Detection
        cluster.check_outliers(now)

        # 6. Select Healthy Endpoint
        endpoint = cluster.select_endpoint(now, preferred_zone=preferred_zone)
        if not endpoint:
            return False, target_cluster_name, None, "ALL_ENDPOINTS_UNAVAILABLE"

        # 7. Increment In-Flight Concurrency
        endpoint.active_requests += 1
        self.current_concurrency += 1
        self.total_routed_requests += 1

        return True, target_cluster_name, endpoint, "SUCCESS"

    def complete_request(
        self,
        cluster_name: str,
        endpoint_id: str,
        success: bool,
        latency_ms: float,
        error_status: int = 500,
        current_time: Optional[float] = None,
    ) -> None:
        """Mark request completion, update latencies, and feed circuit breaker."""
        now = current_time if current_time is not None else time.time()
        self.current_concurrency = max(0, self.current_concurrency - 1)

        cluster = self.clusters.get(cluster_name)
        if not cluster:
            return

        cluster.circuit_breaker.record_result(success, now)

        endpoint = cluster.endpoints.get(endpoint_id)
        if endpoint:
            if success:
                endpoint.record_success(latency_ms)
            else:
                endpoint.record_failure(latency_ms, error_status=error_status)

    def calculate_percentiles(self, cluster_name: str) -> Dict[str, float]:
        """Compute p50, p95, and p99 latencies for a cluster."""
        cluster = self.clusters.get(cluster_name)
        if not cluster:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}

        all_latencies: List[float] = []
        for ep in cluster.endpoints.values():
            all_latencies.extend(ep.latencies_ms)

        if not all_latencies:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}

        sorted_latencies = sorted(all_latencies)
        n = len(sorted_latencies)

        def get_p(pct: float) -> float:
            idx = min(n - 1, max(0, int(math.ceil(pct / 100.0 * n)) - 1))
            return sorted_latencies[idx]

        return {
            "p50": round(get_p(50.0), 2),
            "p95": round(get_p(95.0), 2),
            "p99": round(get_p(99.0), 2),
        }
