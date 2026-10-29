import pytest

from llm_control_plane.rate_limit import TenantRateLimiter, TokenBucket


class Clock:
    def __init__(self, now=0.0):
        self.now = now

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def test_token_bucket_consumes_and_refills():
    clock = Clock()
    bucket = TokenBucket(capacity=2, refill_per_second=1, clock=clock)
    assert bucket.try_acquire() is True
    assert bucket.try_acquire() is True
    assert bucket.try_acquire() is False
    clock.advance(1)
    assert bucket.try_acquire() is True


def test_retry_after_is_computed():
    clock = Clock()
    bucket = TokenBucket(capacity=1, refill_per_second=0.5, clock=clock)
    assert bucket.try_acquire() is True
    assert bucket.retry_after() == pytest.approx(2.0)


def test_tenants_are_isolated():
    clock = Clock()
    limiter = TenantRateLimiter(capacity=1, refill_per_second=1, clock=clock)
    assert limiter.try_acquire("a") is True
    assert limiter.try_acquire("a") is False
    assert limiter.try_acquire("b") is True


def test_invalid_configuration_rejected():
    with pytest.raises(ValueError):
        TokenBucket(capacity=0, refill_per_second=1)
    with pytest.raises(ValueError):
        TokenBucket(capacity=1, refill_per_second=0)
