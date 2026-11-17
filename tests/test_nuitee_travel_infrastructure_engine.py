"""Tests for Nuitée Travel Infrastructure Engine & Asynchronous Settlement Primitives."""

import time
import pytest

from llm_control_plane.nuitee_travel_infrastructure_engine import (
    BookingRecord,
    BookingRequest,
    BookingStatus,
    CircuitState,
    IdempotentBookingSettlementWorker,
    MultiVerticalTravelEngine,
    SupplierCircuitBreaker,
    TokenBucketRateLimiter,
    TravelInventoryItem,
    VerticalType,
)


def test_multi_vertical_provider_registration_and_search() -> None:
    engine = MultiVerticalTravelEngine()

    def hotel_provider():
        return [
            TravelInventoryItem(
                item_id="hotel-dublin-01",
                vertical=VerticalType.HOTELS,
                supplier_id="hotelbeds",
                name="The Marker Hotel Dublin",
                currency="EUR",
                base_price=220.0,
                taxes=28.6,
                total_price=248.6,
                metadata={"city": "Dublin", "stars": 5},
            )
        ]

    def experience_provider():
        return [
            TravelInventoryItem(
                item_id="exp-wicklow-01",
                vertical=VerticalType.EXPERIENCES,
                supplier_id="getyourguide",
                name="Glendalough & Wicklow Mountains Day Tour",
                currency="EUR",
                base_price=45.0,
                taxes=5.85,
                total_price=50.85,
                metadata={"duration_hours": 8},
            )
        ]

    engine.register_provider(VerticalType.HOTELS, "hotelbeds", hotel_provider)
    engine.register_provider(VerticalType.EXPERIENCES, "getyourguide", experience_provider)

    hotels = engine.search_vertical(VerticalType.HOTELS)
    assert len(hotels) == 1
    assert hotels[0].name == "The Marker Hotel Dublin"
    assert hotels[0].total_price == 248.6

    experiences = engine.search_vertical(VerticalType.EXPERIENCES)
    assert len(experiences) == 1
    assert experiences[0].name == "Glendalough & Wicklow Mountains Day Tour"

    flights = engine.search_vertical(VerticalType.FLIGHTS)
    assert len(flights) == 0


def test_supplier_circuit_breaker_trip_and_cooldown() -> None:
    breaker = SupplierCircuitBreaker(supplier_id="amadeus", failure_threshold=2, cooldown_seconds=10.0)
    assert breaker.state == CircuitState.CLOSED
    assert breaker.can_execute(current_time=100.0)

    # First failure
    breaker.record_failure(current_time=101.0)
    assert breaker.state == CircuitState.CLOSED
    assert breaker.consecutive_failures == 1

    # Second failure trips breaker
    breaker.record_failure(current_time=102.0)
    assert breaker.state == CircuitState.OPEN
    assert breaker.consecutive_failures == 2
    assert not breaker.can_execute(current_time=105.0)

    # After cooldown period, transitions to HALF_OPEN
    assert breaker.can_execute(current_time=113.0)
    assert breaker.state == CircuitState.HALF_OPEN

    # Success in half-open resets to CLOSED
    breaker.record_success()
    assert breaker.state == CircuitState.CLOSED
    assert breaker.consecutive_failures == 0


def test_token_bucket_rate_limiter() -> None:
    limiter = TokenBucketRateLimiter(capacity=2.0, refill_rate_per_sec=1.0)
    t0 = 1000.0
    limiter.last_refill_time = t0
    limiter.tokens = 2.0

    # Consume available tokens
    assert limiter.consume(amount=1.0, current_time=t0)
    assert limiter.consume(amount=1.0, current_time=t0)
    assert not limiter.consume(amount=1.0, current_time=t0)

    # After 1.5 seconds, 1 token available
    assert limiter.consume(amount=1.0, current_time=t0 + 1.5)
    assert not limiter.consume(amount=1.0, current_time=t0 + 1.5)


def test_idempotent_booking_submission_deduplication() -> None:
    engine = MultiVerticalTravelEngine()
    worker = IdempotentBookingSettlementWorker(engine)

    req1 = BookingRequest(
        booking_id="bk_001",
        idempotency_key="idemp_key_unique_abc123",
        vertical=VerticalType.HOTELS,
        customer_id="cust_999",
        partner_id="revolut",
        supplier_id="hotelbeds",
        item_id="hotel-dublin-01",
        amount=248.6,
        currency="EUR",
    )

    record1 = worker.submit_booking_request(req1)
    assert record1.status == BookingStatus.PENDING_HOLD
    assert record1.booking_id == "bk_001"

    # Duplicate request with identical idempotency key
    req2 = BookingRequest(
        booking_id="bk_002_different_id",
        idempotency_key="idemp_key_unique_abc123",
        vertical=VerticalType.HOTELS,
        customer_id="cust_999",
        partner_id="revolut",
        supplier_id="hotelbeds",
        item_id="hotel-dublin-01",
        amount=248.6,
        currency="EUR",
    )

    record2 = worker.submit_booking_request(req2)
    assert record2.booking_id == "bk_001"
    assert any("idempotent_duplicate_request_acknowledged" in msg for msg in record2.audit_trail)


def test_asynchronous_settlement_and_partner_webhook_emission() -> None:
    engine = MultiVerticalTravelEngine()
    engine.register_provider(VerticalType.HOTELS, "hotelbeds", lambda: [])
    worker = IdempotentBookingSettlementWorker(engine)

    req = BookingRequest(
        booking_id="bk_100",
        idempotency_key="idemp_100",
        vertical=VerticalType.HOTELS,
        customer_id="cust_456",
        partner_id="hopper",
        supplier_id="hotelbeds",
        item_id="hotel-dublin-01",
        amount=350.0,
        currency="EUR",
    )

    worker.submit_booking_request(req)
    processed = worker.process_next_job(supplier_ref="NUI-HOTEL-DUBLIN-9988")

    assert processed is not None
    assert processed.status == BookingStatus.CONFIRMED
    assert processed.supplier_confirmation_code == "NUI-HOTEL-DUBLIN-9988"

    webhooks = worker.get_emitted_webhooks()
    assert len(webhooks) == 1
    assert webhooks[0]["partner_id"] == "hopper"
    assert webhooks[0]["booking_id"] == "bk_100"
    assert webhooks[0]["confirmation_code"] == "NUI-HOTEL-DUBLIN-9988"


def test_supplier_failure_routes_to_dead_letter_queue() -> None:
    engine = MultiVerticalTravelEngine()
    engine.register_provider(VerticalType.FLIGHTS, "amadeus", lambda: [])
    worker = IdempotentBookingSettlementWorker(engine)

    req = BookingRequest(
        booking_id="bk_200",
        idempotency_key="idemp_200",
        vertical=VerticalType.FLIGHTS,
        customer_id="cust_789",
        partner_id="uber",
        supplier_id="amadeus",
        item_id="flight-dub-lhr-01",
        amount=180.0,
        currency="EUR",
    )

    worker.submit_booking_request(req)
    processed = worker.process_next_job(simulate_supplier_failure=True)

    assert processed is not None
    assert processed.status == BookingStatus.FAILED
    assert processed.failure_reason == "upstream_supplier_rejected_booking"

    dlq = worker.get_dead_letters()
    assert len(dlq) == 1
    assert dlq[0]["booking_id"] == "bk_200"
    assert dlq[0]["reason"] == "upstream_supplier_rejected_booking"


def test_open_circuit_breaker_fails_fast_to_dlq() -> None:
    engine = MultiVerticalTravelEngine()
    engine.register_provider(VerticalType.CAR_RENTAL, "hertz", lambda: [], failure_threshold=1)
    worker = IdempotentBookingSettlementWorker(engine)

    # Trip breaker for hertz
    engine._circuit_breakers["hertz"].record_failure()
    assert engine.get_supplier_circuit_state("hertz") == CircuitState.OPEN

    req = BookingRequest(
        booking_id="bk_300",
        idempotency_key="idemp_300",
        vertical=VerticalType.CAR_RENTAL,
        customer_id="cust_111",
        partner_id="google",
        supplier_id="hertz",
        item_id="car-dub-ev-01",
        amount=120.0,
        currency="EUR",
    )

    worker.submit_booking_request(req)
    processed = worker.process_next_job()

    assert processed is not None
    assert processed.status == BookingStatus.FAILED
    assert "circuit_open" in processed.failure_reason

    dlq = worker.get_dead_letters()
    assert len(dlq) == 1
    assert dlq[0]["booking_id"] == "bk_300"
