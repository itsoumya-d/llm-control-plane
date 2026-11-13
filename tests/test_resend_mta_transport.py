"""Unit tests for Resend Mail Transfer Agent (MTA) Transport Infrastructure."""

from llm_control_plane.resend_mta_transport import (
    DeliveryCategory,
    DeliveryOutcomeClassifier,
    IPPoolManager,
    IPPoolType,
    IPWarmupSchedule,
    MultiTierLeakyBucketThrottle,
    ResendMTATransportEngine,
)


def test_outcome_classifier_delivered():
    """Verify 250 response is classified as DELIVERED."""
    classifier = DeliveryOutcomeClassifier()
    outcome = classifier.classify(250, "250 2.0.0 OK dkim=pass header.i=@resend.com")
    assert outcome.category == DeliveryCategory.DELIVERED
    assert outcome.smtp_code == 250
    assert outcome.enhanced_code == "2.0.0"
    assert outcome.is_retryable is False
    assert outcome.recommended_backoff_sec == 0.0


def test_outcome_classifier_rate_limited():
    """Verify mailbox provider rate limit messages are classified as RATE_LIMITED."""
    classifier = DeliveryOutcomeClassifier()
    outcome = classifier.classify(421, "421 4.7.28 Our system has detected an unusual rate of traffic; throttled")
    assert outcome.category == DeliveryCategory.RATE_LIMITED
    assert outcome.is_retryable is True
    assert outcome.recommended_backoff_sec > 0.0


def test_outcome_classifier_transient_deferral():
    """Verify 4XX temporary failures are classified as TRANSIENT_DEFERRAL."""
    classifier = DeliveryOutcomeClassifier()
    outcome = classifier.classify(451, "451 4.3.0 Mail server busy, try again later")
    assert outcome.category == DeliveryCategory.TRANSIENT_DEFERRAL
    assert outcome.is_retryable is True
    assert outcome.recommended_backoff_sec > 0.0


def test_outcome_classifier_reputation_blocked():
    """Verify spam/blocklist rejections are classified as REPUTATION_BLOCKED."""
    classifier = DeliveryOutcomeClassifier()
    outcome = classifier.classify(554, "554 5.7.1 Service unavailable; Client host blocked using Spamhaus")
    assert outcome.category == DeliveryCategory.REPUTATION_BLOCKED
    assert outcome.is_retryable is False
    assert outcome.recommended_backoff_sec == 0.0


def test_outcome_classifier_hard_bounce():
    """Verify invalid user addresses are classified as HARD_BOUNCE."""
    classifier = DeliveryOutcomeClassifier()
    outcome = classifier.classify(550, "550 5.1.1 The email account that you tried to reach does not exist")
    assert outcome.category == DeliveryCategory.HARD_BOUNCE
    assert outcome.is_retryable is False
    assert outcome.recommended_backoff_sec == 0.0


def test_outcome_classifier_protocol_error():
    """Verify unrecognized 500 status codes are classified as PROTOCOL_ERROR."""
    classifier = DeliveryOutcomeClassifier()
    outcome = classifier.classify(500, "500 Syntax error, command unrecognized")
    assert outcome.category == DeliveryCategory.PROTOCOL_ERROR
    assert outcome.is_retryable is False


def test_ip_warmup_schedule():
    """Verify 14-day warmup capacity limits and day advancement."""
    schedule = IPWarmupSchedule(ip_address="198.51.100.25")
    assert schedule.current_day == 1
    assert schedule.daily_capacity == 50
    assert schedule.can_send(50) is True
    assert schedule.can_send(51) is False

    schedule.record_send(50)
    assert schedule.remaining_today == 0
    assert schedule.can_send(1) is False

    schedule.advance_day()
    assert schedule.current_day == 2
    assert schedule.daily_capacity == 100
    assert schedule.sent_today == 0
    assert schedule.can_send(100) is True


def test_ip_pool_manager_dedicated_and_spillover():
    """Verify dedicated IP capacity enforcement and spillover to shared pool."""
    manager = IPPoolManager()
    manager.register_dedicated_ip(customer_id="cust_acme", ip_address="198.51.100.99")

    # Day 1 capacity is 50
    ip, pool_type = manager.select_outbound_ip("cust_acme")
    assert ip == "198.51.100.99"
    assert pool_type == IPPoolType.DEDICATED

    # Exhaust day 1 capacity
    schedule = manager.dedicated_pools["cust_acme"]
    schedule.record_send(49)

    # Next send should spill over to shared pool
    ip2, pool_type2 = manager.select_outbound_ip("cust_acme")
    assert ip2 in manager.shared_ips
    assert pool_type2 == IPPoolType.SHARED


def test_ip_pool_manager_feedback_and_cordon():
    """Verify reputation blocks reduce health score and trigger cordoning."""
    manager = IPPoolManager()
    manager.register_dedicated_ip(customer_id="cust_bad", ip_address="198.51.100.77")
    classifier = DeliveryOutcomeClassifier()

    block_outcome = classifier.classify(554, "554 5.7.1 Blocked by Spamhaus ZEN")
    # Record multiple blocks
    manager.record_feedback("198.51.100.77", block_outcome)
    manager.record_feedback("198.51.100.77", block_outcome)
    manager.record_feedback("198.51.100.77", block_outcome)

    schedule = manager.dedicated_pools["cust_bad"]
    assert schedule.health_score <= 25.0
    assert schedule.is_cordoned is True

    # When cordoned, sends immediately spill over to shared pool
    ip, pool_type = manager.select_outbound_ip("cust_bad")
    assert ip in manager.shared_ips
    assert pool_type == IPPoolType.SHARED


def test_multitier_leaky_bucket_throttle():
    """Verify recipient domain slot acquisition and concurrency capping."""
    throttle = MultiTierLeakyBucketThrottle()
    throttle.domain_caps["customdomain.com"] = 2

    assert throttle.acquire_domain_slot("alice@customdomain.com") is True
    assert throttle.acquire_domain_slot("bob@customdomain.com") is True
    # Third concurrent connection should be blocked
    assert throttle.acquire_domain_slot("charlie@customdomain.com") is False

    throttle.release_domain_slot("alice@customdomain.com")
    assert throttle.acquire_domain_slot("charlie@customdomain.com") is True


def test_mta_engine_dispatch_success():
    """Verify end-to-end email dispatch and audit record generation."""
    engine = ResendMTATransportEngine()
    tx, outcome = engine.dispatch_email(
        customer_id="cust_enterprise_01",
        recipient="user@example.com",
    )

    assert tx.status_code == 250
    assert tx.category == DeliveryCategory.DELIVERED.value
    assert len(tx.audit_hash) == 64
    assert outcome.category == DeliveryCategory.DELIVERED


def test_mta_engine_dispatch_throttled():
    """Verify domain concurrency throttling produces client-side rate limit outcome."""
    engine = ResendMTATransportEngine()
    engine.throttle.domain_caps["throttled.org"] = 1

    # Occupy slot
    engine.throttle.acquire_domain_slot("hold@throttled.org")

    tx, outcome = engine.dispatch_email(
        customer_id="cust_starter_02",
        recipient="test@throttled.org",
    )

    assert tx.status_code == 421
    assert tx.category == DeliveryCategory.RATE_LIMITED.value
    assert outcome.is_retryable is True
    assert "concurrency" in outcome.reason.lower()
