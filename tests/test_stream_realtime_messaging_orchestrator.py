"""Unit tests for Stream Real-Time Messaging & Activity Feeds Storage Orchestrator."""

import datetime as dt

from llm_control_plane.stream_realtime_messaging_orchestrator import (
    DeliveryState,
    PostgresEventPartitionManager,
    RealTimeEventFanoutEngine,
    RocksDBLSMCompactionCoordinator,
    StreamMessage,
    StreamPlatformTelemetryAndSLOTracker,
)


def test_realtime_event_fanout_and_rate_limiting():
    engine = RealTimeEventFanoutEngine(tenant_rate_limit_per_sec=2)

    # Subscriptions
    engine.subscribe("channel_chat_1", "user_alice")
    engine.subscribe("channel_chat_1", "user_bob", tags={"priority"})
    assert engine.get_subscriber_count("channel_chat_1") == 2

    # Publish message with matching tag
    msg1 = StreamMessage(
        message_id="msg_101",
        channel_id="channel_chat_1",
        tenant_id="tenant_acme",
        sender_id="user_carol",
        payload={"text": "hello stream", "tags": ["priority"]},
        timestamp=dt.datetime.now(dt.timezone.utc),
    )
    state, delivered = engine.publish(msg1)
    assert state == DeliveryState.DELIVERED
    assert delivered == 2

    # Publish second message
    msg2 = StreamMessage(
        message_id="msg_102",
        channel_id="channel_chat_1",
        tenant_id="tenant_acme",
        sender_id="user_carol",
        payload={"text": "second message"},
        timestamp=dt.datetime.now(dt.timezone.utc),
    )
    state2, delivered2 = engine.publish(msg2)
    assert state2 == DeliveryState.DELIVERED
    assert delivered2 == 1  # only alice has no tag filter, bob filtered out

    # Third message immediately should trigger rate limit
    msg3 = StreamMessage(
        message_id="msg_103",
        channel_id="channel_chat_1",
        tenant_id="tenant_acme",
        sender_id="user_carol",
        payload={"text": "spam message"},
        timestamp=dt.datetime.now(dt.timezone.utc),
    )
    state3, delivered3 = engine.publish(msg3)
    assert state3 == DeliveryState.RATE_LIMITED
    assert delivered3 == 0

    # Unsubscribe
    assert engine.unsubscribe("channel_chat_1", "user_alice") is True
    assert engine.get_subscriber_count("channel_chat_1") == 1


def test_postgres_event_partition_manager():
    mgr = PostgresEventPartitionManager(base_table="stream_events")
    start = dt.datetime(2026, 11, 14, 0, 0, tzinfo=dt.timezone.utc)
    end = dt.datetime(2026, 11, 15, 0, 0, tzinfo=dt.timezone.utc)

    sql = mgr.generate_create_partition_sql(start, end)
    assert "CREATE TABLE IF NOT EXISTS stream_events_2026_11_14" in sql
    assert len(mgr.list_active_partitions()) == 1

    # Resolve target partition
    evt_time = dt.datetime(2026, 11, 14, 12, 30, tzinfo=dt.timezone.utc)
    target = mgr.resolve_target_partition(evt_time)
    assert target == "stream_events_2026_11_14"

    # Prune partition
    cutoff = dt.datetime(2026, 11, 16, 0, 0, tzinfo=dt.timezone.utc)
    dropped = mgr.prune_expired_partitions(cutoff)
    assert "stream_events_2026_11_14" in dropped
    assert len(mgr.list_active_partitions()) == 0


def test_rocksdb_lsm_compaction_coordinator():
    coord = RocksDBLSMCompactionCoordinator(max_write_buffer_bytes=1000, stall_threshold_ratio=0.8)

    # Small write - no stall
    stalled = coord.record_write("feeds_cf", 500)
    assert stalled is False
    assert coord.get_buffer_utilization("feeds_cf") == 0.5

    # Large write - triggers stall threshold (850 / 1000 >= 0.8)
    stalled = coord.record_write("feeds_cf", 350)
    assert stalled is True
    assert coord.get_buffer_utilization("feeds_cf") == 0.85

    # Schedule compaction
    job = coord.schedule_compaction("feeds_cf", level_from=0, level_to=1)
    assert job.input_bytes == 850
    assert coord.get_buffer_utilization("feeds_cf") == 0.0

    # Write amplification
    w_amp = coord.calculate_write_amplification(total_bytes_written=4500, user_payload_bytes=1000)
    assert w_amp == 4.5


def test_stream_platform_telemetry_and_slo():
    tracker = StreamPlatformTelemetryAndSLOTracker(target_p99_ms=20.0, target_availability=0.99)

    # Record 100 events
    for i in range(99):
        tracker.record_event(latency_ms=10.0 + (i * 0.05), success=True, cost_usd=0.00001)
    # 1 slow event
    tracker.record_event(latency_ms=18.0, success=True, cost_usd=0.00001)

    eval_result = tracker.evaluate_slo_compliance()
    assert eval_result["p99_pass"] is True
    assert eval_result["availability_pass"] is True
    assert eval_result["overall_compliant"] is True
    assert eval_result["cost_per_million_usd"] > 0.0
