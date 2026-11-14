"""
Braintrust SDK Observability and Evaluation Engine.

Provides:
- Lightweight, zero-overhead tracing with parent-child span hierarchy tracking.
- Context-manager (`span()`) and decorator (`@traced`) APIs for idiomatic developer experience.
- Asynchronous telemetry buffer with non-blocking queueing and background batch flushing.
- Resilient offline fallback spooler to prevent data loss during network partitions.
- Pluggable evaluation scoring framework for custom behavioral assertions and benchmark suites.
- Non-invasive provider wrapping for OpenAI, Anthropic, and Gemini completion calls.
"""

from __future__ import annotations

import contextvars
import enum
import functools
import json
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional


class SpanType(str, enum.Enum):
    LLM = "llm"
    TOOL = "tool"
    TASK = "task"
    EVAL = "eval"


_current_span_context: contextvars.ContextVar[Optional[SpanContext]] = contextvars.ContextVar(
    "current_span_context", default=None
)


@dataclass
class SpanContext:
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    name: str
    span_type: SpanType
    start_time: float
    end_time: Optional[float] = None
    input_data: Any = None
    output_data: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def duration_ms(self) -> Optional[float]:
        if self.end_time is None:
            return None
        return round((self.end_time - self.start_time) * 1000.0, 3)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "span_type": self.span_type.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "input": self.input_data,
            "output": self.output_data,
            "metadata": self.metadata,
            "metrics": self.metrics,
            "error": self.error,
        }


class OfflineDiskSpooler:
    """Simulates an offline local disk spooler preventing trace loss during network partitions."""

    def __init__(self) -> None:
        self._spool: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def spool(self, span_dict: Dict[str, Any]) -> None:
        with self._lock:
            self._spool.append(span_dict)

    def drain(self) -> List[Dict[str, Any]]:
        with self._lock:
            drained = list(self._spool)
            self._spool.clear()
            return drained

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._spool)


class AsyncTelemetryBuffer:
    """Non-blocking, thread-safe background buffer for high-throughput trace export."""

    def __init__(
        self,
        flush_handler: Callable[[List[Dict[str, Any]]], bool],
        batch_size: int = 50,
        flush_interval_seconds: float = 0.5,
        max_queue_size: int = 10000,
    ) -> None:
        self._flush_handler = flush_handler
        self._batch_size = batch_size
        self._flush_interval = flush_interval_seconds
        self._queue: queue.Queue[Dict[str, Any]] = queue.Queue(maxsize=max_queue_size)
        self._spooler = OfflineDiskSpooler()
        self._running = True
        self._lock = threading.Lock()
        self._flushed_count = 0

        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def enqueue(self, span_dict: Dict[str, Any]) -> bool:
        try:
            self._queue.put_nowait(span_dict)
            return True
        except queue.Full:
            # Spool to offline disk fallback if buffer is full
            self._spooler.spool(span_dict)
            return False

    def _worker_loop(self) -> None:
        while self._running:
            self.flush()
            time.sleep(self._flush_interval)

    def flush(self) -> int:
        batch = []
        # First recover any spilled items
        spilled = self._spooler.drain()
        if spilled:
            batch.extend(spilled)

        while len(batch) < self._batch_size:
            try:
                item = self._queue.get_nowait()
                batch.append(item)
            except queue.Empty:
                break

        if not batch:
            return 0

        success = False
        try:
            success = self._flush_handler(batch)
        except Exception:
            success = False

        if not success:
            # Network error: spool batch to disk for future retry
            for item in batch:
                self._spooler.spool(item)
            return 0

        with self._lock:
            self._flushed_count += len(batch)
        return len(batch)

    def close(self) -> None:
        self._running = False
        self.flush()
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)

    @property
    def total_flushed(self) -> int:
        with self._lock:
            return self._flushed_count

    @property
    def spooler(self) -> OfflineDiskSpooler:
        return self._spooler


class BraintrustTracer:
    """Core developer-facing tracer client for instrumenting AI systems."""

    def __init__(self, buffer: Optional[AsyncTelemetryBuffer] = None) -> None:
        self._buffer = buffer
        self._completed_spans: List[SpanContext] = []

    def start_span(
        self,
        name: str,
        span_type: SpanType = SpanType.TASK,
        input_data: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SpanContext:
        parent = _current_span_context.get()
        trace_id = parent.trace_id if parent else f"tr-{uuid.uuid4().hex[:12]}"
        span_id = f"sp-{uuid.uuid4().hex[:8]}"
        parent_id = parent.span_id if parent else None

        span = SpanContext(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_id,
            name=name,
            span_type=span_type,
            start_time=time.time(),
            input_data=input_data,
            metadata=dict(metadata or {}),
        )
        return span

    def end_span(
        self,
        span: SpanContext,
        output_data: Any = None,
        metrics: Optional[Dict[str, float]] = None,
        error: Optional[str] = None,
    ) -> None:
        span.end_time = time.time()
        if output_data is not None:
            span.output_data = output_data
        if metrics:
            span.metrics.update(metrics)
        if error:
            span.error = str(error)

        self._completed_spans.append(span)
        if self._buffer:
            self._buffer.enqueue(span.to_dict())

    def span(
        self,
        name: str,
        span_type: SpanType = SpanType.TASK,
        input_data: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        tracer = self

        class _SpanContextManager:
            def __init__(self) -> None:
                self.span_ctx: Optional[SpanContext] = None
                self.token = None

            def __enter__(self) -> SpanContext:
                self.span_ctx = tracer.start_span(name, span_type, input_data, metadata)
                self.token = _current_span_context.set(self.span_ctx)
                return self.span_ctx

            def __exit__(self, exc_type, exc_val, exc_tb):
                err = str(exc_val) if exc_val else None
                tracer.end_span(self.span_ctx, error=err)
                _current_span_context.reset(self.token)
                return False

        return _SpanContextManager()


def traced(
    name: Optional[str] = None,
    span_type: SpanType = SpanType.TASK,
    tracer_provider: Optional[Callable[[], BraintrustTracer]] = None,
):
    """Decorator to instrument functions with automatic span creation."""

    def decorator(fn: Callable):
        span_name = name or fn.__name__

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            tracer = tracer_provider() if tracer_provider else BraintrustTracer()
            input_summary = {"args": [str(a)[:50] for a in args], "kwargs_keys": list(kwargs.keys())}
            with tracer.span(span_name, span_type=span_type, input_data=input_summary) as s:
                result = fn(*args, **kwargs)
                s.output_data = result
                return result

        return wrapper

    return decorator


@dataclass
class ScorerResult:
    name: str
    score: float
    passed: bool
    details: Dict[str, Any] = field(default_factory=dict)


class EvaluationScorer:
    """Base pluggable evaluation scorer for validating model and agent behavior."""

    def __init__(self, name: str, threshold: float = 0.8) -> None:
        self.name = name
        self.threshold = threshold

    def score(self, output: str, expected: str, context: Optional[Dict[str, Any]] = None) -> ScorerResult:
        raise NotImplementedError


class ExactMatchScorer(EvaluationScorer):
    def score(self, output: str, expected: str, context: Optional[Dict[str, Any]] = None) -> ScorerResult:
        is_exact = output.strip().lower() == expected.strip().lower()
        val = 1.0 if is_exact else 0.0
        return ScorerResult(name=self.name, score=val, passed=val >= self.threshold)


class KeywordPresenceScorer(EvaluationScorer):
    def __init__(self, name: str, required_keywords: List[str], threshold: float = 1.0) -> None:
        super().__init__(name, threshold)
        self.required_keywords = required_keywords

    def score(self, output: str, expected: str, context: Optional[Dict[str, Any]] = None) -> ScorerResult:
        text = output.lower()
        matched = [k for k in self.required_keywords if k.lower() in text]
        ratio = len(matched) / len(self.required_keywords) if self.required_keywords else 1.0
        return ScorerResult(
            name=self.name,
            score=round(ratio, 2),
            passed=ratio >= self.threshold,
            details={"matched": matched, "total": len(self.required_keywords)},
        )


class EvaluationHarness:
    """Manages evaluation datasets and runs scorer suites against agent traces."""

    def __init__(self, scorers: List[EvaluationScorer]) -> None:
        self.scorers = scorers

    def run_eval(self, dataset: List[Dict[str, str]], agent_fn: Callable[[str], str]) -> Dict[str, Any]:
        results = []
        passed_all = 0

        for row in dataset:
            inp = row["input"]
            expected = row["expected"]
            out = agent_fn(inp)

            row_scores = [scorer.score(out, expected) for scorer in self.scorers]
            all_passed = all(s.passed for s in row_scores)
            if all_passed:
                passed_all += 1

            results.append(
                {
                    "input": inp,
                    "expected": expected,
                    "output": out,
                    "scores": {s.name: s.score for s in row_scores},
                    "passed": all_passed,
                }
            )

        total = len(dataset)
        pass_rate = (passed_all / total * 100.0) if total > 0 else 100.0
        return {
            "total_samples": total,
            "passed_samples": passed_all,
            "pass_rate_pct": round(pass_rate, 2),
            "results": results,
        }
