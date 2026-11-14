"""Unit tests for Toast Service Mesh Router & Dynamic Traffic Control Plane."""

import time

from llm_control_plane.toast_mesh_router import (
    CircuitBreaker,
    CircuitState,
    EndpointHealth,
    OutlierDetectionConfig,
    PriorityTier,
    RouteDestination,
    RouteRule,
    ServiceCluster,
    ServiceEndpoint,
    ToastMeshRouter,
)


def test_endpoint_lifecycle_and_metrics():
    ep = ServiceEndpoint(endpoint_id="ep-1", host="10.0.1.10", port=8080, zone="eu-west-1a")
    assert ep.is_available(time.time())

    # Record success
    ep.active_requests = 1
    ep.record_success(25.4)
    assert ep.active_requests == 0
    assert ep.total_requests == 1
    assert ep.failed_requests == 0
    assert ep.consecutive_failures == 0
    assert len(ep.latencies_ms) == 1

    # Record failure
    ep.active_requests = 1
    ep.record_failure(120.0, error_status=503)
    assert ep.failed_requests == 1
    assert ep.consecutive_failures == 1


def test_circuit_breaker_transitions():
    cb = CircuitBreaker(
        failure_rate_threshold=0.5,
        min_requests_threshold=4,
        recovery_timeout_seconds=10.0,
        half_open_success_threshold=2,
    )
    t0 = 1000.0
    assert cb.state == CircuitState.CLOSED
    assert cb.allow_request(t0)

    # 2 successes, 2 failures -> 50% failure rate -> should trip to OPEN
    cb.record_result(True, t0)
    cb.record_result(True, t0)
    cb.record_result(False, t0)
    cb.record_result(False, t0)
    assert cb.state == CircuitState.OPEN

    # Requests rejected while open
    assert not cb.allow_request(t0 + 2.0)

    # Recovery timeout elapsed -> should enter HALF_OPEN
    assert cb.allow_request(t0 + 11.0)
    assert cb.state == CircuitState.HALF_OPEN

    # Half open successes
    cb.record_result(True, t0 + 11.0)
    assert cb.state == CircuitState.HALF_OPEN
    cb.record_result(True, t0 + 11.0)
    assert cb.state == CircuitState.CLOSED


def test_circuit_breaker_half_open_failure_reopen():
    cb = CircuitBreaker(
        failure_rate_threshold=0.5,
        min_requests_threshold=2,
        recovery_timeout_seconds=5.0,
        half_open_success_threshold=2,
    )
    t0 = 100.0
    cb.record_result(False, t0)
    cb.record_result(False, t0)
    assert cb.state == CircuitState.OPEN

    # Transition to HALF_OPEN after recovery timeout
    assert cb.allow_request(t0 + 6.0)
    assert cb.state == CircuitState.HALF_OPEN

    # Failure in HALF_OPEN immediately trips back to OPEN
    cb.record_result(False, t0 + 6.0)
    assert cb.state == CircuitState.OPEN
    assert not cb.allow_request(t0 + 7.0)


def test_outlier_detection_ejection():
    outlier_cfg = OutlierDetectionConfig(
        consecutive_5xx_limit=3,
        ejection_duration_seconds=20.0,
        max_ejection_percent=50.0,
    )
    cluster = ServiceCluster(name="order-service", outlier_config=outlier_cfg)
    ep1 = ServiceEndpoint(endpoint_id="ep-1", host="10.0.1.1", port=8080)
    ep2 = ServiceEndpoint(endpoint_id="ep-2", host="10.0.1.2", port=8080)
    cluster.add_endpoint(ep1)
    cluster.add_endpoint(ep2)

    t0 = 500.0
    # Simulate 3 consecutive failures on ep1
    ep1.record_failure(50.0)
    ep1.record_failure(50.0)
    ep1.record_failure(50.0)

    ejected = cluster.check_outliers(t0)
    assert "ep-1" in ejected
    assert ep1.health == EndpointHealth.EJECTED
    assert not ep1.is_available(t0 + 5.0)

    # After ejection duration expires, endpoint recovers
    assert ep1.is_available(t0 + 25.0)
    assert ep1.health == EndpointHealth.HEALTHY


def test_locality_aware_endpoint_selection():
    cluster = ServiceCluster(name="kds-service")
    ep_a = ServiceEndpoint(endpoint_id="kds-a", host="10.0.1.1", port=8080, zone="eu-west-1a")
    ep_b = ServiceEndpoint(endpoint_id="kds-b", host="10.0.2.1", port=8080, zone="eu-west-1b")
    cluster.add_endpoint(ep_a)
    cluster.add_endpoint(ep_b)

    t0 = 1000.0
    selected = cluster.select_endpoint(t0, preferred_zone="eu-west-1b")
    assert selected is not None
    assert selected.endpoint_id == "kds-b"


def test_route_rule_matching_and_canary_selection():
    rule = RouteRule(
        route_id="orders-route",
        path_prefix="/v1/orders",
        destinations=[
            RouteDestination(cluster_name="orders-v1", weight=80),
            RouteDestination(cluster_name="orders-v2-canary", weight=20),
        ],
        headers_match={"X-Toast-Channel": "POS"},
    )

    # Path matches and header matches
    assert rule.matches("/v1/orders/create", {"X-Toast-Channel": "POS"})
    # Path mismatches
    assert not rule.matches("/v1/menus/list", {"X-Toast-Channel": "POS"})
    # Header mismatches
    assert not rule.matches("/v1/orders/create", {"X-Toast-Channel": "WEB"})

    dest = rule.select_destination()
    assert dest in ("orders-v1", "orders-v2-canary")


def test_adaptive_load_shedding():
    router = ToastMeshRouter(concurrency_ceiling=100, load_shed_threshold_percent=80.0)
    cluster = ServiceCluster(name="order-service")
    ep = ServiceEndpoint(endpoint_id="ep-1", host="10.0.1.1", port=8080)
    cluster.add_endpoint(ep)
    router.register_cluster(cluster)

    rule = RouteRule(
        route_id="orders-rule",
        path_prefix="/v1/orders",
        destinations=[RouteDestination(cluster_name="order-service", weight=100)],
    )
    router.add_route(rule)

    t0 = 100.0
    # Normal load: 20 active requests
    router.current_concurrency = 20
    ok, target, endpoint, reason = router.route_request(
        "/v1/orders/new", {}, priority=PriorityTier.BEST_EFFORT, current_time=t0
    )
    assert ok
    assert reason == "SUCCESS"

    # Peak rush load: 88 active requests (88% of ceiling >= 80% threshold)
    router.current_concurrency = 88
    # BEST_EFFORT should be shed immediately
    ok_shed, _, _, reason_shed = router.route_request(
        "/v1/orders/new", {}, priority=PriorityTier.BEST_EFFORT, current_time=t0
    )
    assert not ok_shed
    assert reason_shed == "LOAD_SHED_BEST_EFFORT"

    # CRITICAL POS order must still be admitted
    ok_crit, target_crit, _, reason_crit = router.route_request(
        "/v1/orders/new", {}, priority=PriorityTier.CRITICAL, current_time=t0
    )
    assert ok_crit
    assert reason_crit == "SUCCESS"


def test_end_to_end_routing_and_percentiles():
    router = ToastMeshRouter()
    cluster = ServiceCluster(name="payment-gateway")
    ep = ServiceEndpoint(endpoint_id="pay-1", host="10.0.2.1", port=9000)
    cluster.add_endpoint(ep)
    router.register_cluster(cluster)

    rule = RouteRule(
        route_id="pay-rule",
        path_prefix="/v1/payments",
        destinations=[RouteDestination(cluster_name="payment-gateway", weight=100)],
    )
    router.add_route(rule)

    t0 = 1000.0
    ok, cname, endpoint, reason = router.route_request(
        "/v1/payments/authorize", {}, priority=PriorityTier.CRITICAL, current_time=t0
    )
    assert ok
    assert cname == "payment-gateway"
    assert endpoint is not None
    assert endpoint.endpoint_id == "pay-1"
    assert endpoint.active_requests == 1

    # Complete request
    router.complete_request("payment-gateway", "pay-1", success=True, latency_ms=18.5, current_time=t0 + 0.02)
    assert endpoint.active_requests == 0
    assert endpoint.total_requests == 1

    # Add more latencies and verify percentiles
    for lat in [10.0, 15.0, 20.0, 30.0, 40.0, 150.0]:
        router.complete_request("payment-gateway", "pay-1", success=True, latency_ms=lat, current_time=t0)

    pcts = router.calculate_percentiles("payment-gateway")
    assert pcts["p50"] > 0.0
    assert pcts["p95"] > pcts["p50"]
    assert pcts["p99"] >= pcts["p95"]
