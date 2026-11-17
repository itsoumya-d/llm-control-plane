"""Nuitée Travel Infrastructure Engine — Multi-Vertical Booking & Supplier Resilience Orchestrator.

Implements high-throughput B2B travel distribution primitives for Nuitée's API-first platform:
- Multi-vertical provider abstraction (Hotels, Flights, Experiences, Car Rentals)
- Distributed supplier circuit breaking with automatic failure isolation & half-open recovery
- Token-bucket rate limiting protecting third-party upstream supplier quotas
- Idempotent booking state machine with atomic deduplication and distributed worker queue processing
- Dead-letter queue (DLQ) triage and asynchronous partner webhook event dispatch (Revolut, Hopper, Google, Uber)
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set


class VerticalType(str, enum.Enum):
    HOTELS = "hotels"
    FLIGHTS = "flights"
    EXPERIENCES = "experiences"
    CAR_RENTAL = "car_rental"
    PAYMENT = "payment"


class BookingStatus(str, enum.Enum):
    PENDING_HOLD = "pending_hold"
    HELD = "held"
    CONFIRMED = "confirmed"
    TICKETED = "ticketed"
    CANCELLED = "cancelled"
    FAILED = "failed"
    REFUNDED = "refunded"


class CircuitState(str, enum.Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class TravelInventoryItem:
    item_id: str
    vertical: VerticalType
    supplier_id: str
    name: str
    currency: str
    base_price: float
    taxes: float
    total_price: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    available_until: float = field(default_factory=lambda: time.time() + 900)


@dataclass
class BookingRequest:
    booking_id: str
    idempotency_key: str
    vertical: VerticalType
    customer_id: str
    partner_id: str
    supplier_id: str
    item_id: str
    amount: float
    currency: str
    guest_info: Dict[str, Any] = field(default_factory=dict)
    payment_token: Optional[str] = None


@dataclass
class BookingRecord:
    booking_id: str
    idempotency_key: str
    vertical: VerticalType
    customer_id: str
    partner_id: str
    supplier_id: str
    item_id: str
    amount: float
    currency: str
    status: BookingStatus
    supplier_confirmation_code: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    failure_reason: Optional[str] = None
    audit_trail: List[str] = field(default_factory=list)


@dataclass
class SupplierCircuitBreaker:
    supplier_id: str
    failure_threshold: int = 3
    cooldown_seconds: float = 5.0
    consecutive_failures: int = 0
    state: CircuitState = CircuitState.CLOSED
    last_failure_time: float = 0.0

    def can_execute(self, current_time: Optional[float] = None) -> bool:
        now = current_time if current_time is not None else time.time()
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            if now - self.last_failure_time >= self.cooldown_seconds:
                self.state = CircuitState.HALF_OPEN
                return True
            return False
        if self.state == CircuitState.HALF_OPEN:
            return True
        return False

    def record_success(self) -> None:
        self.consecutive_failures = 0
        self.state = CircuitState.CLOSED

    def record_failure(self, current_time: Optional[float] = None) -> None:
        now = current_time if current_time is not None else time.time()
        self.consecutive_failures += 1
        self.last_failure_time = now
        if self.consecutive_failures >= self.failure_threshold:
            self.state = CircuitState.OPEN


@dataclass
class TokenBucketRateLimiter:
    capacity: float
    refill_rate_per_sec: float
    tokens: float = field(init=False)
    last_refill_time: float = field(init=False)

    def __post_init__(self) -> None:
        self.tokens = self.capacity
        self.last_refill_time = time.time()

    def consume(self, amount: float = 1.0, current_time: Optional[float] = None) -> bool:
        now = current_time if current_time is not None else time.time()
        elapsed = max(0.0, now - self.last_refill_time)
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate_per_sec)
        self.last_refill_time = now

        if self.tokens >= amount:
            self.tokens -= amount
            return True
        return False


class MultiVerticalTravelEngine:
    """Orchestrates multi-vertical travel inventory search, pricing, and supplier resilience."""

    def __init__(self) -> None:
        self._providers: Dict[VerticalType, Dict[str, Callable[[], List[TravelInventoryItem]]]] = {
            v: {} for v in VerticalType
        }
        self._circuit_breakers: Dict[str, SupplierCircuitBreaker] = {}
        self._rate_limiters: Dict[str, TokenBucketRateLimiter] = {}
        self._inventory_cache: Dict[str, TravelInventoryItem] = {}

    def register_provider(
        self,
        vertical: VerticalType,
        supplier_id: str,
        adapter_fn: Callable[[], List[TravelInventoryItem]],
        qps_limit: float = 50.0,
        failure_threshold: int = 3,
        cooldown_sec: float = 5.0,
    ) -> None:
        self._providers[vertical][supplier_id] = adapter_fn
        self._circuit_breakers[supplier_id] = SupplierCircuitBreaker(
            supplier_id=supplier_id,
            failure_threshold=failure_threshold,
            cooldown_seconds=cooldown_sec,
        )
        self._rate_limiters[supplier_id] = TokenBucketRateLimiter(
            capacity=qps_limit,
            refill_rate_per_sec=qps_limit,
        )

    def search_vertical(
        self,
        vertical: VerticalType,
        current_time: Optional[float] = None,
    ) -> List[TravelInventoryItem]:
        results: List[TravelInventoryItem] = []
        now = current_time if current_time is not None else time.time()

        for supplier_id, adapter in self._providers[vertical].items():
            breaker = self._circuit_breakers[supplier_id]
            limiter = self._rate_limiters[supplier_id]

            if not breaker.can_execute(current_time=now):
                continue
            if not limiter.consume(amount=1.0, current_time=now):
                continue

            try:
                items = adapter()
                breaker.record_success()
                for item in items:
                    self._inventory_cache[item.item_id] = item
                    results.append(item)
            except Exception:
                breaker.record_failure(current_time=now)

        return results

    def get_inventory_item(self, item_id: str) -> Optional[TravelInventoryItem]:
        return self._inventory_cache.get(item_id)

    def get_supplier_circuit_state(self, supplier_id: str) -> Optional[CircuitState]:
        breaker = self._circuit_breakers.get(supplier_id)
        return breaker.state if breaker else None


class IdempotentBookingSettlementWorker:
    """Processes asynchronous booking holds, confirmations, dead-letters, and partner webhook events."""

    def __init__(self, engine: MultiVerticalTravelEngine) -> None:
        self.engine = engine
        self._bookings: Dict[str, BookingRecord] = {}
        self._idempotency_index: Dict[str, str] = {}  # idempotency_key -> booking_id
        self._queue: List[BookingRequest] = []
        self._dead_letter_queue: List[Dict[str, Any]] = []
        self._webhook_events: List[Dict[str, Any]] = []

    def submit_booking_request(self, req: BookingRequest) -> BookingRecord:
        # Idempotency check: if already submitted under this key, return the existing record immediately
        if req.idempotency_key in self._idempotency_index:
            existing_id = self._idempotency_index[req.idempotency_key]
            record = self._bookings[existing_id]
            record.audit_trail.append(f"idempotent_duplicate_request_acknowledged_at_{time.time()}")
            return record

        record = BookingRecord(
            booking_id=req.booking_id,
            idempotency_key=req.idempotency_key,
            vertical=req.vertical,
            customer_id=req.customer_id,
            partner_id=req.partner_id,
            supplier_id=req.supplier_id,
            item_id=req.item_id,
            amount=req.amount,
            currency=req.currency,
            status=BookingStatus.PENDING_HOLD,
            audit_trail=[f"created_at_{time.time()}"],
        )
        self._bookings[req.booking_id] = record
        self._idempotency_index[req.idempotency_key] = req.booking_id
        self._queue.append(req)
        return record

    def process_next_job(
        self,
        simulate_supplier_failure: bool = False,
        supplier_ref: Optional[str] = None,
    ) -> Optional[BookingRecord]:
        if not self._queue:
            return None

        req = self._queue.pop(0)
        record = self._bookings[req.booking_id]

        # Verify supplier circuit status before processing settlement
        breaker_state = self.engine.get_supplier_circuit_state(req.supplier_id)
        if breaker_state == CircuitState.OPEN:
            record.status = BookingStatus.FAILED
            record.failure_reason = f"circuit_open_for_supplier_{req.supplier_id}"
            record.updated_at = time.time()
            record.audit_trail.append(f"failed_due_to_open_circuit_{req.supplier_id}")
            self._dead_letter_queue.append({
                "booking_id": req.booking_id,
                "reason": record.failure_reason,
                "timestamp": time.time(),
            })
            return record

        if simulate_supplier_failure:
            record.status = BookingStatus.FAILED
            record.failure_reason = "upstream_supplier_rejected_booking"
            record.updated_at = time.time()
            record.audit_trail.append("supplier_rejection_logged")
            self._dead_letter_queue.append({
                "booking_id": req.booking_id,
                "reason": record.failure_reason,
                "timestamp": time.time(),
            })
            return record

        # Transition to HELD then CONFIRMED
        record.status = BookingStatus.CONFIRMED
        record.supplier_confirmation_code = supplier_ref or f"CONF-{req.vertical.value[:3].upper()}-{req.booking_id[:8]}"
        record.updated_at = time.time()
        record.audit_trail.append(f"confirmed_with_{record.supplier_confirmation_code}")

        # Emit partner webhook event
        self._webhook_events.append({
            "event": "booking.confirmed",
            "partner_id": req.partner_id,
            "booking_id": req.booking_id,
            "vertical": req.vertical.value,
            "confirmation_code": record.supplier_confirmation_code,
            "timestamp": time.time(),
        })

        return record

    def get_booking(self, booking_id: str) -> Optional[BookingRecord]:
        return self._bookings.get(booking_id)

    def get_dead_letters(self) -> List[Dict[str, Any]]:
        return list(self._dead_letter_queue)

    def get_emitted_webhooks(self) -> List[Dict[str, Any]]:
        return list(self._webhook_events)
