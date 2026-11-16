import asyncio
import pytest

from llm_control_plane.signoz_agent_observability_engine import (
    ClickHouseQuerySanitizer,
    FinishReason,
    NozSessionCoordinator,
    OpenTelemetryGenAISpanCollector,
    OTelSpan,
    SigNozMCPServer,
)


def test_otel_span_cost_estimation():
    span = OTelSpan(
        trace_id="trace-001",
        span_id="span-001",
        parent_span_id=None,
        name="chat_completion",
        service_name="llm-service",
        timestamp_nano=1700000000000000000,
        duration_nano=150000000,
        system="openai",
        model="gpt-4o",
        prompt_tokens=1000,
        completion_tokens=500,
        total_tokens=1500,
        finish_reason=FinishReason.STOP,
    )
    # 1000 * 0.000005 + 500 * 0.000015 = 0.005 + 0.0075 = 0.0125
    cost = span.estimated_cost_usd()
    assert pytest.approx(cost, 0.0001) == 0.0125


def test_clickhouse_query_sanitizer():
    # Valid query gets automatic LIMIT
    q1 = "SELECT service_name, count() FROM signoz_traces GROUP BY service_name"
    sanitized1 = ClickHouseQuerySanitizer.sanitize(q1, max_limit=100)
    assert "LIMIT 100" in sanitized1

    # Explicit limit is preserved
    q2 = "SELECT * FROM signoz_logs WHERE severity = 'ERROR' LIMIT 50"
    sanitized2 = ClickHouseQuerySanitizer.sanitize(q2)
    assert sanitized2 == q2

    # Forbidden DDL/DML throws ValueError
    with pytest.raises(ValueError, match="Forbidden ClickHouse operation detected"):
        ClickHouseQuerySanitizer.sanitize("DROP TABLE signoz_traces")

    with pytest.raises(ValueError, match="Forbidden ClickHouse operation detected"):
        ClickHouseQuerySanitizer.sanitize("ALTER TABLE signoz_metrics DELETE WHERE 1=1")

    with pytest.raises(ValueError, match="Forbidden ClickHouse operation detected"):
        ClickHouseQuerySanitizer.sanitize("INSERT INTO signoz_logs VALUES ('test')")

    with pytest.raises(ValueError, match="Only read-only analytics queries"):
        ClickHouseQuerySanitizer.sanitize("PRAGMA foo_bar = 1")


def test_signoz_mcp_server_tools():
    server = SigNozMCPServer()
    tools = server.list_tools()
    assert len(tools) >= 4
    tool_names = {t["name"] for t in tools}
    assert "signoz_get_trace" in tool_names
    assert "signoz_query_metrics" in tool_names
    assert "signoz_search_logs" in tool_names
    assert "signoz_correlate_anomalies" in tool_names

    # Test tool invocation
    trace_res = server.call_tool("signoz_get_trace", {"trace_id": "tr-xyz-99"})
    assert trace_res["status"] == "success"
    assert trace_res["result"]["trace_id"] == "tr-xyz-99"
    assert len(trace_res["result"]["spans"]) == 4

    anomaly_res = server.call_tool("signoz_correlate_anomalies", {"service": "checkout-svc"})
    assert anomaly_res["status"] == "success"
    assert len(anomaly_res["result"]["correlated_causes"]) >= 2

    missing_res = server.call_tool("unknown_tool", {})
    assert missing_res["status"] == "error"


def test_noz_session_coordinator_streaming():
    async def _run():
        server = SigNozMCPServer()
        coordinator = NozSessionCoordinator(session_id="test-session-1", mcp_server=server, max_history_turns=4)

        events = []
        async for chunk in coordinator.stream_response("Why is checkout-svc latency spiking?"):
            events.append(chunk)

        assert len(events) > 5
        assert any("event: done" in e for e in events)
        assert len(coordinator.messages) == 2  # user + assistant

    asyncio.run(_run())


def test_noz_session_cancellation():
    async def _run():
        server = SigNozMCPServer()
        coordinator = NozSessionCoordinator(session_id="test-cancel-session", mcp_server=server)

        events = []
        stream = coordinator.stream_response("Investigate database query timeout")
        coordinator.cancel()
        async for chunk in stream:
            events.append(chunk)

        assert any("event: cancelled" in e for e in events)

    asyncio.run(_run())


def test_opentelemetry_collector_batch_flushing():
    collector = OpenTelemetryGenAISpanCollector(batch_size=2, max_queue_capacity=5)

    for i in range(4):
        span = OTelSpan(
            trace_id=f"tr-{i}",
            span_id=f"sp-{i}",
            parent_span_id=None,
            name="completion",
            service_name="agent-worker",
            timestamp_nano=1700000000000000000 + i,
            duration_nano=100000000,
            system="anthropic",
            model="claude-3-5-sonnet",
            prompt_tokens=200,
            completion_tokens=100,
            total_tokens=300,
            finish_reason=FinishReason.STOP,
        )
        assert collector.record_span(span) is True

    # Check batch flush
    res1 = collector.flush_batch()
    assert res1["flushed_count"] == 2
    assert res1["remaining_buffered"] == 2
    assert len(res1["columnar_payload"]["trace_ids"]) == 2
    assert res1["columnar_payload"]["trace_ids"] == ["tr-0", "tr-1"]

    res2 = collector.flush_batch()
    assert res2["flushed_count"] == 2
    assert res2["remaining_buffered"] == 0
    assert res2["columnar_payload"]["trace_ids"] == ["tr-2", "tr-3"]


def test_collector_capacity_backpressure():
    collector = OpenTelemetryGenAISpanCollector(batch_size=10, max_queue_capacity=2)
    for i in range(2):
        s = OTelSpan(
            trace_id=f"tr-{i}",
            span_id=f"sp-{i}",
            parent_span_id=None,
            name="completion",
            service_name="agent-worker",
            timestamp_nano=1700000000000000000,
            duration_nano=100000000,
            system="gemini",
            model="gemini-1.5-pro",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            finish_reason=FinishReason.STOP,
        )
        assert collector.record_span(s) is True

    # 3rd span should be rejected due to capacity limit
    overflow_span = OTelSpan(
        trace_id="tr-overflow",
        span_id="sp-overflow",
        parent_span_id=None,
        name="completion",
        service_name="agent-worker",
        timestamp_nano=1700000000000000000,
        duration_nano=100000000,
        system="gemini",
        model="gemini-1.5-pro",
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        finish_reason=FinishReason.STOP,
    )
    assert collector.record_span(overflow_span) is False
