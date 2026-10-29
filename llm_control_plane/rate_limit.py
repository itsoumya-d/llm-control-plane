"""Per-tenant token buckets with an injectable clock."""

from __future__ import annotations

import threading
import time
from typing import Callable, Dict


class TokenBucket:
    def __init__(
        self,
        capacity: float,
        refill_per_second: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if refill_per_second <= 0:
            raise ValueError("refill_per_second must be positive")
        self.capacity = float(capacity)
        self.refill_per_second = float(refill_per_second)
        self.clock = clock
        self._tokens = float(capacity)
        self._updated_at = clock()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        now = self.clock()
        elapsed = max(0.0, now - self._updated_at)
        self._tokens = min(self.capacity, self._tokens + elapsed * self.refill_per_second)
        self._updated_at = now

    def try_acquire(self, tokens: float = 1.0) -> bool:
        if tokens <= 0:
            raise ValueError("tokens must be positive")
        with self._lock:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    def retry_after(self, tokens: float = 1.0) -> float:
        with self._lock:
            self._refill()
            deficit = max(0.0, tokens - self._tokens)
            return deficit / self.refill_per_second

    def available(self) -> float:
        with self._lock:
            self._refill()
            return self._tokens


class TenantRateLimiter:
    def __init__(
        self,
        capacity: float = 60.0,
        refill_per_second: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.capacity = capacity
        self.refill_per_second = refill_per_second
        self.clock = clock
        self._buckets: Dict[str, TokenBucket] = {}
        self._lock = threading.Lock()

    def bucket_for(self, tenant_id: str) -> TokenBucket:
        with self._lock:
            bucket = self._buckets.get(tenant_id)
            if bucket is None:
                bucket = TokenBucket(self.capacity, self.refill_per_second, self.clock)
                self._buckets[tenant_id] = bucket
            return bucket

    def try_acquire(self, tenant_id: str, tokens: float = 1.0) -> bool:
        return self.bucket_for(tenant_id).try_acquire(tokens)

    def retry_after(self, tenant_id: str, tokens: float = 1.0) -> float:
        return self.bucket_for(tenant_id).retry_after(tokens)
