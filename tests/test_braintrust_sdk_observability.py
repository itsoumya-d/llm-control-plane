"""
Unit tests for Braintrust SDK observability and evaluation engine.
"""

import time

import pytest

from llm_control_plane.braintrust_sdk_observability import (
    AsyncTelemetryBuffer,
    BraintrustTracer,
    EvaluationHarness,
    ExactMatchScorer,
    KeywordPresenceScorer,
    OfflineDiskSpooler,
    SpanContext,
    SpanType,
    traced,
)


def test_span_context_metrics_and_duration():
    start = time.time() - 0.05
    end = time.time()
    span = SpanContext(
        trace_id="tr-test-1",
        span_id="sp-1",
        parent_span_id=None,
        name="test_span",
        span_type=SpanType.LLM,
        start_time=start,
        end_time=end,
        metrics={"tokens": 150, "cost_usd": 0.003},
    )
    d = span.to_dict()
    assert d["trace_id"] == "tr-test-1"
    assert d["span_type"] == "llm"
    assert span.duration_ms is not None
    assert span.duration_ms >= 40.0
    assert d["metrics"]["tokens"] == 150


def test_tracer_nested_span_hierarchy():
    tracer = BraintrustTracer()

    with tracer.span("parent_flow", span_type=SpanType.TASK) as parent:
        assert parent.name == "parent_flow"
        assert parent.parent_span_id is None

        with tracer.span("child_llm_call", span_type=SpanType.LLM) as child:
            assert child.parent_span_id == parent.span_id
            assert child.trace_id == parent.trace_id

            with tracer.span("leaf_tool_call", span_type=SpanType.TOOL) as leaf:
                assert leaf.parent_span_id == child.span_id
                assert leaf.trace_id == parent.trace_id

    assert len(tracer._completed_spans) == 3


def test_traced_decorator_execution():
    tracer = BraintrustTracer()

    @traced(name="custom_math_task", span_type=SpanType.TASK, tracer_provider=lambda: tracer)
    def compute_sum(a: int, b: int) -> int:
        return a + b

    result = compute_sum(10, 25)
    assert result == 35
    assert len(tracer._completed_spans) == 1
    span = tracer._completed_spans[0]
    assert span.name == "custom_math_task"
    assert span.output_data == 35


def test_span_error_handling_capture():
    tracer = BraintrustTracer()

    with pytest.raises(ValueError, match="Simulated engine failure"):
        with tracer.span("failing_operation", span_type=SpanType.TASK):
            raise ValueError("Simulated engine failure")

    assert len(tracer._completed_spans) == 1
    span = tracer._completed_spans[0]
    assert span.error is not None
    assert "Simulated engine failure" in span.error


def test_async_telemetry_buffer_flushing():
    flushed_batches = []

    def mock_flush(batch):
        flushed_batches.append(batch)
        return True

    buffer = AsyncTelemetryBuffer(
        flush_handler=mock_flush,
        batch_size=2,
        flush_interval_seconds=0.05,
    )

    buffer.enqueue({"event": 1})
    buffer.enqueue({"event": 2})
    buffer.enqueue({"event": 3})

    count = buffer.flush()
    assert count >= 2
    assert buffer.total_flushed >= 2

    buffer.close()


def test_offline_disk_spooler_resilience():
    spooler = OfflineDiskSpooler()
    spooler.spool({"trace": "item_1"})
    spooler.spool({"trace": "item_2"})
    assert spooler.pending_count == 2

    drained = spooler.drain()
    assert len(drained) == 2
    assert spooler.pending_count == 0


def test_async_buffer_fallback_on_network_failure():
    should_fail = True

    def flakey_flush(batch):
        if should_fail:
            return False
        return True

    buffer = AsyncTelemetryBuffer(
        flush_handler=flakey_flush,
        batch_size=5,
        flush_interval_seconds=1.0,
    )

    buffer.enqueue({"trace": "retry_me_1"})
    buffer.enqueue({"trace": "retry_me_2"})

    # First flush fails and spools to disk
    flushed = buffer.flush()
    assert flushed == 0
    assert buffer.spooler.pending_count == 2

    # Network recovers
    should_fail = False
    flushed_recovery = buffer.flush()
    assert flushed_recovery == 2
    assert buffer.spooler.pending_count == 0

    buffer.close()


def test_evaluation_scorers_and_harness():
    em_scorer = ExactMatchScorer(name="exact_match", threshold=1.0)
    kw_scorer = KeywordPresenceScorer(
        name="keyword_presence",
        required_keywords=["postgres"],
        threshold=1.0,
    )

    harness = EvaluationHarness(scorers=[em_scorer, kw_scorer])
    dataset = [
        {"input": "What database was chosen?", "expected": "Postgres"},
    ]

    def mock_agent(prompt: str) -> str:
        return "Postgres"

    report = harness.run_eval(dataset, mock_agent)
    assert report["total_samples"] == 1
    assert report["passed_samples"] == 1
    assert report["pass_rate_pct"] == 100.0
    assert report["results"][0]["passed"] is True
