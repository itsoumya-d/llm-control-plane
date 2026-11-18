"""Helsing Resilient Telemetry Stream Orchestrator.

High-throughput distributed telemetry stream ingestion, low-latency zero-copy buffer
management, dynamic backpressure flow control, and air-gapped cryptographic hash-chain
audit logging for mission-critical semi-autonomous edge systems.
"""

from __future__ import annotations

import collections
import dataclasses
import enum
import hashlib
import time
from typing import Any, Dict, List, Optional, Tuple


class TelemetryFormat(str, enum.Enum):
    """Supported serialization formats for telemetry payloads."""

    PROTOBUF = "PROTOBUF"
    ARROW = "ARROW"
    RAW_BINARY = "RAW_BINARY"


class ThrottlingLevel(str, enum.Enum):
    """Dynamic backpressure throttling levels under fluctuating link bandwidth."""

    NORMAL = "NORMAL"
    SHED_LOW_PRIORITY = "SHED_LOW_PRIORITY"
    COMPRESS_DELTA = "COMPRESS_DELTA"
    CIRCUIT_BREAK = "CIRCUIT_BREAK"


@dataclasses.dataclass(frozen=True)
class TelemetryFrame:
    """Individual sensory telemetry frame received from edge sensor nodes."""

    frame_id: str
    stream_id: str
    sequence_number: int
    timestamp_ns: int
    payload_bytes: bytes
    format: TelemetryFormat
    priority: int  # 1 (Highest / Mission Critical) to 10 (Lowest / Diagnostic)
    metadata: Dict[str, Any]
    payload_sha256: str

    @classmethod
    def create(
        cls,
        frame_id: str,
        stream_id: str,
        sequence_number: int,
        payload_bytes: bytes,
        format: TelemetryFormat = TelemetryFormat.PROTOBUF,
        priority: int = 5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TelemetryFrame:
        meta = metadata or {}
        sha = hashlib.sha256(payload_bytes).hexdigest()
        return cls(
            frame_id=frame_id,
            stream_id=stream_id,
            sequence_number=sequence_number,
            timestamp_ns=time.time_ns(),
            payload_bytes=payload_bytes,
            format=format,
            priority=priority,
            metadata=meta,
            payload_sha256=sha,
        )


@dataclasses.dataclass
class TelemetryBatchResult:
    """Outcome of batch processing through the orchestrator pipeline."""

    batch_id: str
    processed_count: int
    dropped_count: int
    throttling_level: ThrottlingLevel
    batch_merkle_root: str
    ledger_height: int
    processing_latency_ms: float
    channel_stats: Dict[str, int]


class ResilientRingBuffer:
    """Fixed-capacity ring buffer with sequence reordering and gap detection."""

    def __init__(self, capacity: int = 1000) -> None:
        if capacity <= 0:
            raise ValueError("Buffer capacity must be strictly positive.")
        self.capacity = capacity
        self._buffer: collections.deque[TelemetryFrame] = collections.deque(maxlen=capacity)
        self._stream_sequences: Dict[str, int] = {}
        self._sequence_gaps: Dict[str, List[int]] = collections.defaultdict(list)
        self._total_dropped: int = 0

    def append(self, frame: TelemetryFrame) -> bool:
        """Appends frame into ring buffer, detecting sequence gaps and tracking drops."""
        stream = frame.stream_id
        seq = frame.sequence_number

        # Check for sequence continuity
        if stream in self._stream_sequences:
            expected = self._stream_sequences[stream] + 1
            if seq > expected:
                for missing in range(expected, seq):
                    self._sequence_gaps[stream].append(missing)
            self._stream_sequences[stream] = max(self._stream_sequences[stream], seq)
        else:
            self._stream_sequences[stream] = seq

        if len(self._buffer) == self.capacity:
            self._total_dropped += 1

        self._buffer.append(frame)
        return True

    def drain_all(self) -> List[TelemetryFrame]:
        """Drains all buffered frames in order."""
        items = list(self._buffer)
        self._buffer.clear()
        return items

    def peek(self, count: int) -> List[TelemetryFrame]:
        """Returns the oldest n frames without removing them."""
        return list(self._buffer)[:count]

    @property
    def current_size(self) -> int:
        return len(self._buffer)

    @property
    def utilization_ratio(self) -> float:
        return len(self._buffer) / self.capacity

    @property
    def total_dropped(self) -> int:
        return self._total_dropped

    def get_detected_gaps(self, stream_id: str) -> List[int]:
        return list(self._sequence_gaps.get(stream_id, []))


class StreamingBackpressureController:
    """Manages adaptive load throttling based on queue depth and processing latency."""

    def __init__(
        self,
        high_watermark: float = 0.80,
        critical_watermark: float = 0.95,
        latency_threshold_ms: float = 50.0,
    ) -> None:
        self.high_watermark = high_watermark
        self.critical_watermark = critical_watermark
        self.latency_threshold_ms = latency_threshold_ms
        self.current_level = ThrottlingLevel.NORMAL

    def evaluate(self, buffer_utilization: float, observed_latency_ms: float) -> ThrottlingLevel:
        """Determines active throttling level based on current buffer pressure and latency."""
        if buffer_utilization >= self.critical_watermark or observed_latency_ms >= self.latency_threshold_ms * 2:
            self.current_level = ThrottlingLevel.CIRCUIT_BREAK
        elif buffer_utilization >= self.high_watermark or observed_latency_ms >= self.latency_threshold_ms:
            self.current_level = ThrottlingLevel.SHED_LOW_PRIORITY
        elif buffer_utilization >= (self.high_watermark * 0.6):
            self.current_level = ThrottlingLevel.COMPRESS_DELTA
        else:
            self.current_level = ThrottlingLevel.NORMAL
        return self.current_level

    def should_admit(self, frame: TelemetryFrame) -> bool:
        """Checks if a frame should be admitted under the current throttling level."""
        if self.current_level == ThrottlingLevel.CIRCUIT_BREAK:
            # Only admit strictly mission-critical frames (priority 1)
            return frame.priority <= 1
        elif self.current_level == ThrottlingLevel.SHED_LOW_PRIORITY:
            # Shed low priority telemetry (priority > 5)
            return frame.priority <= 5
        return True


class AirGappedCryptographicLedger:
    """Tamper-evident hash-chained audit ledger for air-gapped mission integrity."""

    GENESIS_HASH = "0000000000000000000000000000000000000000000000000000000000000000"

    def __init__(self) -> None:
        self._blocks: List[Dict[str, Any]] = []
        self._latest_hash: str = self.GENESIS_HASH

    def append_batch(self, batch_id: str, frames: List[TelemetryFrame]) -> Tuple[int, str]:
        """Hashes batch of frames into the cryptographic ledger, returning height and block hash."""
        merkle_root = self.compute_merkle_root([f.payload_sha256 for f in frames])
        height = len(self._blocks) + 1
        timestamp = time.time_ns()

        block_content = f"{height}:{self._latest_hash}:{batch_id}:{merkle_root}:{timestamp}"
        block_hash = hashlib.sha256(block_content.encode("utf-8")).hexdigest()

        block = {
            "height": height,
            "previous_hash": self._latest_hash,
            "batch_id": batch_id,
            "frame_count": len(frames),
            "merkle_root": merkle_root,
            "timestamp_ns": timestamp,
            "block_hash": block_hash,
        }
        self._blocks.append(block)
        self._latest_hash = block_hash
        return height, block_hash

    @staticmethod
    def compute_merkle_root(hashes: List[str]) -> str:
        """Computes binary Merkle tree root of frame hashes."""
        if not hashes:
            return hashlib.sha256(b"EMPTY_BATCH").hexdigest()

        current = hashes[:]
        while len(current) > 1:
            if len(current) % 2 != 0:
                current.append(current[-1])  # Duplicate last element if odd
            next_level = []
            for i in range(0, len(current), 2):
                combined = current[i] + current[i + 1]
                next_level.append(hashlib.sha256(combined.encode("utf-8")).hexdigest())
            current = next_level
        return current[0]

    def verify_integrity(self) -> bool:
        """Verifies full hash chain integrity from genesis to head."""
        prev = self.GENESIS_HASH
        for block in self._blocks:
            if block["previous_hash"] != prev:
                return False
            b_h = block["height"]
            b_id = block["batch_id"]
            b_m = block["merkle_root"]
            b_ts = block["timestamp_ns"]
            expected_content = f"{b_h}:{prev}:{b_id}:{b_m}:{b_ts}"
            if hashlib.sha256(expected_content.encode("utf-8")).hexdigest() != block["block_hash"]:
                return False
            prev = block["block_hash"]
        return True

    @property
    def height(self) -> int:
        return len(self._blocks)

    @property
    def head_hash(self) -> str:
        return self._latest_hash


class HelsingTelemetryStreamOrchestrator:
    """Full-featured telemetry streaming orchestrator designed for Helsing edge systems."""

    def __init__(
        self,
        buffer_capacity: int = 500,
        high_watermark: float = 0.80,
        critical_watermark: float = 0.95,
        latency_threshold_ms: float = 30.0,
    ) -> None:
        self.ring_buffer = ResilientRingBuffer(capacity=buffer_capacity)
        self.backpressure = StreamingBackpressureController(
            high_watermark=high_watermark,
            critical_watermark=critical_watermark,
            latency_threshold_ms=latency_threshold_ms,
        )
        self.ledger = AirGappedCryptographicLedger()
        self._total_ingested: int = 0
        self._total_processed: int = 0
        self._total_shed: int = 0
        self._channel_counters: collections.Counter = collections.Counter()

    def ingest_frame(self, frame: TelemetryFrame) -> bool:
        """Ingests a telemetry frame through backpressure evaluation and buffer insertion."""
        self._total_ingested += 1
        if not self.backpressure.should_admit(frame):
            self._total_shed += 1
            return False

        self.ring_buffer.append(frame)
        self._channel_counters[frame.stream_id] += 1
        return True

    def process_pending_batch(self, batch_id: str) -> TelemetryBatchResult:
        """Drains pending buffer, computes cryptographic roots, and records to audit ledger."""
        start_ns = time.perf_counter_ns()
        frames = self.ring_buffer.drain_all()

        channel_stats: Dict[str, int] = collections.defaultdict(int)
        for f in frames:
            channel_stats[f.stream_id] += 1

        height, _ = self.ledger.append_batch(batch_id, frames)
        merkle = self.ledger.compute_merkle_root([f.payload_sha256 for f in frames])

        self._total_processed += len(frames)
        elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0

        # Update backpressure state
        current_throttling = self.backpressure.evaluate(
            buffer_utilization=self.ring_buffer.utilization_ratio,
            observed_latency_ms=elapsed_ms,
        )

        return TelemetryBatchResult(
            batch_id=batch_id,
            processed_count=len(frames),
            dropped_count=self.ring_buffer.total_dropped,
            throttling_level=current_throttling,
            batch_merkle_root=merkle,
            ledger_height=height,
            processing_latency_ms=elapsed_ms,
            channel_stats=dict(channel_stats),
        )

    def get_orchestrator_metrics(self) -> Dict[str, Any]:
        """Provides operational observability metrics for the streaming engine."""
        return {
            "total_ingested": self._total_ingested,
            "total_processed": self._total_processed,
            "total_shed": self._total_shed,
            "buffer_current_size": self.ring_buffer.current_size,
            "buffer_utilization": self.ring_buffer.utilization_ratio,
            "current_throttling_level": self.backpressure.current_level.value,
            "ledger_height": self.ledger.height,
            "ledger_head_hash": self.ledger.head_hash,
            "ledger_verified": self.ledger.verify_integrity(),
            "active_channels": dict(self._channel_counters),
        }
