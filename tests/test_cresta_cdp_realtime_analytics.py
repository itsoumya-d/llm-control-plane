"""Tests for Cresta CDP Real-Time Analytics & Conversation Intelligence Engine."""

import time

import pytest

from llm_control_plane.cresta_cdp_realtime_analytics import (
    ClickHouseColumnarBuffer,
    CoachingActionType,
    ConversationTurnEvent,
    CrestaCDPRealtimeEngine,
    ElasticsearchConversationIndexer,
    TurnRole,
)


def test_conversation_turn_event_validation():
    # Valid event
    evt = ConversationTurnEvent(
        conversation_id="conv-1",
        turn_id="turn-1",
        tenant_id="tenant-acme",
        agent_id="agent-007",
        customer_id="cust-123",
        timestamp=time.time(),
        role=TurnRole.CUSTOMER,
        text="Hello, I need help with my account billing.",
        sentiment_score=0.1,
        intent="billing_inquiry",
        latency_ms=12.5,
    )
    assert evt.conversation_id == "conv-1"
    assert evt.sentiment_score == 0.1

    # Invalid sentiment score (< -1.0)
    with pytest.raises(ValueError, match="sentiment_score must be between -1.0 and 1.0"):
        ConversationTurnEvent(
            conversation_id="conv-2",
            turn_id="turn-2",
            tenant_id="tenant-acme",
            agent_id="agent-007",
            customer_id="cust-123",
            timestamp=time.time(),
            role=TurnRole.CUSTOMER,
            text="Invalid",
            sentiment_score=-1.5,
            intent="error",
        )

    # Invalid latency (< 0)
    with pytest.raises(ValueError, match="latency_ms cannot be negative"):
        ConversationTurnEvent(
            conversation_id="conv-3",
            turn_id="turn-3",
            tenant_id="tenant-acme",
            agent_id="agent-007",
            customer_id="cust-123",
            timestamp=time.time(),
            role=TurnRole.AGENT,
            text="Invalid",
            sentiment_score=0.0,
            intent="help",
            latency_ms=-5.0,
        )


def test_clickhouse_columnar_buffer_batching_and_flush():
    buffer = ClickHouseColumnarBuffer(batch_size=3)
    now = 1731650000.0  # fixed timestamp for deterministic partition

    evt1 = ConversationTurnEvent(
        conversation_id="conv-1",
        turn_id="t-1",
        tenant_id="tenant-1",
        agent_id="ag-1",
        customer_id="c-1",
        timestamp=now,
        role=TurnRole.CUSTOMER,
        text="First turn",
        sentiment_score=0.2,
        intent="inquiry",
    )
    evt2 = ConversationTurnEvent(
        conversation_id="conv-1",
        turn_id="t-2",
        tenant_id="tenant-1",
        agent_id="ag-1",
        customer_id="c-1",
        timestamp=now + 1,
        role=TurnRole.AGENT,
        text="Second turn",
        sentiment_score=0.4,
        intent="response",
    )
    evt3 = ConversationTurnEvent(
        conversation_id="conv-1",
        turn_id="t-3",
        tenant_id="tenant-1",
        agent_id="ag-1",
        customer_id="c-1",
        timestamp=now + 2,
        role=TurnRole.CUSTOMER,
        text="Third turn",
        sentiment_score=0.8,
        intent="gratitude",
    )

    assert buffer.append(evt1) is False
    assert buffer.append(evt2) is False
    # 3rd event reaches batch_size=3, triggers flush
    assert buffer.append(evt3) is True
    assert buffer._total_flushed_rows == 3


def test_clickhouse_query_agent_performance():
    buffer = ClickHouseColumnarBuffer(batch_size=10)
    now = 1731650000.0

    # Add 4 turns for ag-1 under tenant-1
    for i, (sent, lat, intent) in enumerate([
        (0.2, 10.0, "faq"),
        (0.6, 20.0, "faq"),
        (-0.2, 30.0, "billing"),
        (0.8, 40.0, "technical"),
    ]):
        buffer.append(
            ConversationTurnEvent(
                conversation_id=f"conv-{i}",
                turn_id=f"t-{i}",
                tenant_id="tenant-1",
                agent_id="ag-1",
                customer_id="c-1",
                timestamp=now + (i * 10),
                role=TurnRole.AGENT,
                text=f"Agent response {i}",
                sentiment_score=sent,
                intent=intent,
                latency_ms=lat,
            )
        )

    # Add 1 turn for different agent and different tenant
    buffer.append(
        ConversationTurnEvent(
            conversation_id="conv-other",
            turn_id="t-other",
            tenant_id="tenant-2",
            agent_id="ag-2",
            customer_id="c-2",
            timestamp=now,
            role=TurnRole.AGENT,
            text="Other response",
            sentiment_score=-0.8,
            intent="complaint",
            latency_ms=100.0,
        )
    )

    stats = buffer.query_agent_performance(tenant_id="tenant-1", agent_id="ag-1")
    assert stats["total_turns"] == 4
    assert stats["avg_sentiment"] == 0.35  # (0.2 + 0.6 - 0.2 + 0.8) / 4 = 1.4 / 4 = 0.35
    assert stats["avg_latency_ms"] == 25.0
    assert stats["p95_latency_ms"] == 40.0
    assert stats["intent_distribution"] == {"faq": 2, "billing": 1, "technical": 1}

    # Query non-existent agent returns zeroes
    empty_stats = buffer.query_agent_performance(tenant_id="tenant-1", agent_id="ag-ghost")
    assert empty_stats["total_turns"] == 0
    assert empty_stats["avg_sentiment"] == 0.0


def test_clickhouse_query_trend_and_partition_pruning():
    buffer = ClickHouseColumnarBuffer(batch_size=10)
    # Day 1: 1731628800 (2024-11-15T00:00:00Z)
    day1_ts = 1731628800.0
    # Day 2: 1731715200 (2024-11-16T00:00:00Z)
    day2_ts = 1731715200.0

    buffer.append(
        ConversationTurnEvent(
            conversation_id="c1",
            turn_id="t1",
            tenant_id="tenant-a",
            agent_id="ag-1",
            customer_id="cust-1",
            timestamp=day1_ts + 100,
            role=TurnRole.CUSTOMER,
            text="Turn day 1",
            sentiment_score=0.4,
            intent="inquiry",
        )
    )
    buffer.append(
        ConversationTurnEvent(
            conversation_id="c2",
            turn_id="t2",
            tenant_id="tenant-a",
            agent_id="ag-1",
            customer_id="cust-2",
            timestamp=day2_ts + 200,
            role=TurnRole.CUSTOMER,
            text="Turn day 2",
            sentiment_score=0.8,
            intent="praise",
        )
    )

    trend = buffer.query_tenant_sentiment_trend(
        tenant_id="tenant-a", start_ts=day1_ts, end_ts=day2_ts + 3600
    )
    assert len(trend) == 2
    assert trend[0]["turn_count"] == 1
    assert trend[0]["avg_sentiment"] == 0.4
    assert trend[1]["turn_count"] == 1
    assert trend[1]["avg_sentiment"] == 0.8

    # Prune partitions older than Day 2
    pruned = buffer.prune_partitions_older_than(tenant_id="tenant-a", cutoff_ts=day2_ts)
    assert pruned == 1  # Only Day 1 partition pruned
    remaining = buffer.query_tenant_sentiment_trend(
        tenant_id="tenant-a", start_ts=day1_ts, end_ts=day2_ts + 3600
    )
    assert len(remaining) == 1
    assert remaining[0]["date"] == "2024-11-16"


def test_elasticsearch_indexer_tenant_isolation_and_search():
    indexer = ElasticsearchConversationIndexer()

    # Index document in tenant 1
    indexer.index_turn(
        ConversationTurnEvent(
            conversation_id="conv-1",
            turn_id="turn-t1",
            tenant_id="tenant-alpha",
            agent_id="ag-1",
            customer_id="cust-1",
            timestamp=1000.0,
            role=TurnRole.CUSTOMER,
            text="I want to cancel my enterprise flight reservation",
            sentiment_score=-0.7,
            intent="cancel_reservation",
        )
    )

    # Index document in tenant 2
    indexer.index_turn(
        ConversationTurnEvent(
            conversation_id="conv-2",
            turn_id="turn-t2",
            tenant_id="tenant-beta",
            agent_id="ag-2",
            customer_id="cust-2",
            timestamp=1010.0,
            role=TurnRole.CUSTOMER,
            text="Can you upgrade my hotel room suite?",
            sentiment_score=0.5,
            intent="room_upgrade",
        )
    )

    # Tenant alpha search
    alpha_res = indexer.search(tenant_id="tenant-alpha", query="cancel reservation")
    assert len(alpha_res) == 1
    assert alpha_res[0]["turn_id"] == "turn-t1"
    assert alpha_res[0]["score"] >= 2

    # Cross-tenant leak check: Tenant beta searches for "cancel reservation"
    beta_res = indexer.search(tenant_id="tenant-beta", query="cancel reservation")
    assert len(beta_res) == 0

    # Faceted intent aggregation
    alpha_facets = indexer.faceted_aggregate_intents("tenant-alpha")
    assert alpha_facets == {"cancel_reservation": 1}


def test_elasticsearch_faceted_sentiment_filters():
    indexer = ElasticsearchConversationIndexer()

    for i, (text, sent, intent) in enumerate([
        ("Order arrived broken", -0.8, "shipping_complaint"),
        ("Delivery was slightly late", -0.2, "shipping_complaint"),
        ("Package received in good shape", 0.6, "shipping_success"),
    ]):
        indexer.index_turn(
            ConversationTurnEvent(
                conversation_id="c",
                turn_id=f"t-{i}",
                tenant_id="tenant-shop",
                agent_id="ag",
                customer_id=f"c-{i}",
                timestamp=100.0 + i,
                role=TurnRole.CUSTOMER,
                text=text,
                sentiment_score=sent,
                intent=intent,
            )
        )

    # Filter with min/max sentiment bounds
    negative_turns = indexer.search(
        tenant_id="tenant-shop",
        query="shipping",
        intent="shipping_complaint",
        max_sentiment=-0.5,
    )
    assert len(negative_turns) == 1
    assert negative_turns[0]["turn_id"] == "t-0"


def test_cresta_cdp_realtime_engine_end_to_end_and_coaching():
    engine = CrestaCDPRealtimeEngine(buffer_batch_size=5, latency_slo_target_ms=50.0)

    # 1. Customer severe frustration -> Escalation trigger
    _, rec_esc = engine.ingest_turn(
        ConversationTurnEvent(
            conversation_id="conv-100",
            turn_id="turn-100",
            tenant_id="tenant-marriott",
            agent_id="agent-99",
            customer_id="cust-vip",
            timestamp=time.time(),
            role=TurnRole.CUSTOMER,
            text="This is terrible! My reservation was cancelled without notice!",
            sentiment_score=-0.75,
            intent="complaint",
            latency_ms=15.0,
        )
    )
    assert rec_esc is not None
    assert rec_esc.action_type == CoachingActionType.ESCALATION
    assert "senior supervisor" in rec_esc.guidance_text.lower()

    # 2. Churn / Account cancellation trigger
    _, rec_churn = engine.ingest_turn(
        ConversationTurnEvent(
            conversation_id="conv-101",
            turn_id="turn-101",
            tenant_id="tenant-marriott",
            agent_id="agent-99",
            customer_id="cust-vip",
            timestamp=time.time(),
            role=TurnRole.CUSTOMER,
            text="Please proceed with close_account right away.",
            sentiment_score=-0.1,
            intent="close_account",
            latency_ms=10.0,
        )
    )
    assert rec_churn is not None
    assert rec_churn.action_type == CoachingActionType.SCRIPT_GUIDANCE
    assert "loyalty retention" in rec_churn.guidance_text.lower()

    # 3. Agent compliance warning trigger
    _, rec_comp = engine.ingest_turn(
        ConversationTurnEvent(
            conversation_id="conv-102",
            turn_id="turn-102",
            tenant_id="tenant-marriott",
            agent_id="agent-99",
            customer_id="cust-vip",
            timestamp=time.time(),
            role=TurnRole.AGENT,
            text="Please read out your credit card number and CVV code.",
            sentiment_score=0.0,
            intent="payment",
            latency_ms=18.0,
        )
    )
    assert rec_comp is not None
    assert rec_comp.action_type == CoachingActionType.COMPLIANCE_WARNING
    assert "payment pause recording" in rec_comp.guidance_text.lower()

    # 4. Mild negative suggestion
    _, rec_sugg = engine.ingest_turn(
        ConversationTurnEvent(
            conversation_id="conv-103",
            turn_id="turn-103",
            tenant_id="tenant-marriott",
            agent_id="agent-99",
            customer_id="cust-vip",
            timestamp=time.time(),
            role=TurnRole.CUSTOMER,
            text="I am not sure if this will work properly.",
            sentiment_score=-0.3,
            intent="hesitation",
            latency_ms=12.0,
        )
    )
    assert rec_sugg is not None
    assert rec_sugg.action_type == CoachingActionType.SUGGESTION

    # Telemetry and Health check
    telemetry = engine.get_telemetry_metrics()
    assert telemetry["total_ingested_turns"] == 4
    assert telemetry["p50_ingestion_latency_ms"] >= 0.0
    assert telemetry["slo_compliance_pct"] == 100.0

    health = engine.health_check()
    assert health["status"] == "healthy"
    assert health["elasticsearch_tenants_active"] == 1
