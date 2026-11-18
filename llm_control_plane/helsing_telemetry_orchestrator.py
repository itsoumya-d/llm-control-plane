"""
Helsing Resilient Telemetry Streaming Orchestrator.

Models high-throughput sensory ingestion, resilient out-of-order sliding
window ring buffers, adaptive radio backpressure flow control, and air-gapped
cryptographic hash-chain audit logging for Helsing's real-time defence AI
and situational awareness infrastructure.
"""

from __future__ import annotations

import enum
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


class SensorType(str, enum.Enum):
    RF_RADAR = "RF_RADAR"
    EO_IR_OPTICAL = "EO_IR_OPTICAL"
    ACOUSTIC_ARRAY = "ACOUSTIC_ARRAY"
    NAVIGATION_IMU = "NAVIGATION_IMU"


class TelemetryPriority(int, enum.Enum):
    CRITICAL_FLASH = 1
    TACTICAL_HIGH = 2
    STANDARD_ROUTING = 3
    BACKGROUND_TELEMETRY = 4


class StreamHealth(str, enum.Enum):
    OPTIMAL_BROADBAND = "OPTIMAL_BROADBAND"
    DEGRADED_BANDWIDTH = "DEGRADED_BANDWIDTH"
    INTERMITTENT_RADIO = "INTERMITTENT_RADIO"
    AIR_GAPPED_ISOLATION = "AIR_GAPPED_ISOLATION"


class BufferOverflowError(Exception):
    """Raised when the ring buffer exceeds capacity and cannot shed lower priority frames."""

    pass


@dataclass
class TelemetryFrame:
    frame_id: str
    sensor_type: SensorType
    channel_id: str
    sequence_num: int
    timestamp_ms: int
    priority: TelemetryPriority
    payload: bytes
    metadata: Dict[str, Any] = field(default_factory=dict)
    checksum: str = field(default="")
    signature: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.checksum:
            h = hashlib.sha256()
            h.update(self.frame_id.encode("utf-8"))
            h.update(str(self.sequence_num).encode("utf-8"))
            h.update(self.payload)
            self.checksum = h.hexdigest()

    def verify_checksum(self) -> bool:
        h = hashlib.sha256()
        h.update(self.frame_id.encode("utf-8"))
        h.update(str(self.sequence_num).encode("utf-8"))
        h.update(self.payload)
        return h.hexdigest() == self.checksum


class ResilientRingBuffer:
    """
    Sliding-window ring buffer handling out-of-order arrival, sequence gap detection,
    and priority-aware eviction under low-bandwidth tactical constraints.
    """

    def __init__(self, capacity: int = 512) -> None:
        self.capacity = capacity
        self._buffer: Dict[int, TelemetryFrame] = {}
        self._seen_ids: set[str] = set()
        self._evicted_count: int = 0
        self._highest_seq: int = -1

    def ingest(self, frame: TelemetryFrame) -> bool:
        if not frame.verify_checksum():
            raise ValueError(f"Checksum mismatch for frame {frame.frame_id}")

        if frame.frame_id in self._seen_ids:
            return False  # Deduplicated

        if len(self._buffer) >= self.capacity:
            # Evict lowest-priority older frame if incoming frame has higher priority
            lowest_key: Optional[int] = None
            lowest_priority: int = -1
            for seq, existing in self._buffer.items():
                if existing.priority.value > lowest_priority:
                    lowest_priority = existing.priority.value
                    lowest_key = seq

            if lowest_key is not None and lowest_priority > frame.priority.value:
                evicted = self._buffer.pop(lowest_key)
                self._seen_ids.remove(evicted.frame_id)
                self._evicted_count += 1
            else:
                raise BufferOverflowError(f"Buffer at capacity {self.capacity} with no lower priority frame to shed.")

        self._buffer[frame.sequence_num] = frame
        self._seen_ids.add(frame.frame_id)
        if frame.sequence_num > self._highest_seq:
            self._highest_seq = frame.sequence_num
        return True

    def get_ordered_window(self) -> List[TelemetryFrame]:
        return [self._buffer[k] for k in sorted(self._buffer.keys())]

    def detect_missing_sequences(self) -> List[int]:
        if not self._buffer:
            return []
        sorted_seqs = sorted(self._buffer.keys())
        min_seq = sorted_seqs[0]
        max_seq = sorted_seqs[-1]
        expected = set(range(min_seq, max_seq + 1))
        present = set(sorted_seqs)
        return sorted(list(expected - present))

    @property
    def size(self) -> int:
        return len(self._buffer)

    @property
    def evicted_count(self) -> int:
        return self._evicted_count


@dataclass
class AuditBlock:
    index: int
    timestamp_ns: int
    event_type: str
    payload_hash: str
    prev_hash: str
    block_hash: str


class AirGappedAuditChain:
    """
    Cryptographic SHA-256 hash-chain audit ledger designed for air-gapped tactical nodes.
    Ensures deterministic tamper detection without reliance on external certificate authorities.
    """

    GENESIS_PREV_HASH = "0" * 64

    def __init__(self, node_id: str) -> None:
        self.node_id = node_id
        self.chain: List[AuditBlock] = []

    def append_event(self, event_type: str, payload: Dict[str, Any]) -> AuditBlock:
        prev_hash = self.chain[-1].block_hash if self.chain else self.GENESIS_PREV_HASH
        idx = len(self.chain)
        now_ns = time.time_ns()

        payload_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")
        payload_hash = hashlib.sha256(payload_bytes).hexdigest()

        h = hashlib.sha256()
        h.update(str(idx).encode("utf-8"))
        h.update(str(now_ns).encode("utf-8"))
        h.update(event_type.encode("utf-8"))
        h.update(payload_hash.encode("utf-8"))
        h.update(prev_hash.encode("utf-8"))
        block_hash = h.hexdigest()

        block = AuditBlock(
            index=idx,
            timestamp_ns=now_ns,
            event_type=event_type,
            payload_hash=payload_hash,
            prev_hash=prev_hash,
            block_hash=block_hash,
        )
        self.chain.append(block)
        return block

    def verify_integrity(self) -> Tuple[bool, Optional[str]]:
        for i, block in enumerate(self.chain):
            expected_prev = self.chain[i - 1].block_hash if i > 0 else self.GENESIS_PREV_HASH
            if block.prev_hash != expected_prev:
                return False, f"Broken chain link at block {i}: prev_hash mismatch"

            h = hashlib.sha256()
            h.update(str(block.index).encode("utf-8"))
            h.update(str(block.timestamp_ns).encode("utf-8"))
            h.update(block.event_type.encode("utf-8"))
            h.update(block.payload_hash.encode("utf-8"))
            h.update(block.prev_hash.encode("utf-8"))
            if h.hexdigest() != block.block_hash:
                return False, f"Tampered block hash at index {i}"
        return True, None


class HelsingTelemetryStreamer:
    """
    High-performance telemetry streamer bridging real-time multi-sensor frames
    with low-bandwidth tactical networks, ML inference pipelines, and air-gapped audit trails.
    """

    def __init__(self, node_id: str = "helsing-munich-alpha-01", buffer_capacity: int = 512) -> None:
        self.node_id = node_id
        self.health = StreamHealth.OPTIMAL_BROADBAND
        self.buffer = ResilientRingBuffer(capacity=buffer_capacity)
        self.audit_chain = AirGappedAuditChain(node_id=node_id)
        self._channel_registry: Dict[str, SensorType] = {}
        self._processed_frames: int = 0
        self._dropped_frames: int = 0

    def register_channel(self, channel_id: str, sensor_type: SensorType) -> None:
        self._channel_registry[channel_id] = sensor_type
        self.audit_chain.append_event(
            "CHANNEL_REGISTERED",
            {"channel_id": channel_id, "sensor_type": sensor_type.value},
        )

    def set_stream_health(self, health: StreamHealth) -> None:
        old_health = self.health
        self.health = health
        self.audit_chain.append_event(
            "HEALTH_TRANSITION",
            {"from": old_health.value, "to": health.value},
        )

    def ingest_frame(self, frame: TelemetryFrame) -> Dict[str, Any]:
        # Filter or shed background frames under degraded radio links
        if self.health == StreamHealth.INTERMITTENT_RADIO and frame.priority == TelemetryPriority.BACKGROUND_TELEMETRY:
            self._dropped_frames += 1
            return {"status": "SHED_BACKGROUND", "frame_id": frame.frame_id}

        try:
            accepted = self.buffer.ingest(frame)
            if accepted:
                self._processed_frames += 1
                return {"status": "ACCEPTED", "sequence_num": frame.sequence_num}
            return {"status": "DUPLICATE", "frame_id": frame.frame_id}
        except BufferOverflowError:
            self._dropped_frames += 1
            return {"status": "OVERFLOW_DROPPED", "frame_id": frame.frame_id}

    def export_batch_for_ml(self, max_batch: int = 64) -> List[Dict[str, Any]]:
        """
        Extracts ordered frames formatted for real-time PyTorch/ML scoring pipelines.
        """
        ordered = self.buffer.get_ordered_window()
        batch = ordered[:max_batch]
        records: List[Dict[str, Any]] = []
        for f in batch:
            records.append(
                {
                    "frame_id": f.frame_id,
                    "sensor_type": f.sensor_type.value,
                    "channel_id": f.channel_id,
                    "sequence_num": f.sequence_num,
                    "timestamp_ms": f.timestamp_ms,
                    "priority": f.priority.value,
                    "payload_len": len(f.payload),
                    "metadata": f.metadata,
                    "checksum": f.checksum,
                }
            )
        return records

    def get_diagnostics(self) -> Dict[str, Any]:
        missing = self.buffer.detect_missing_sequences()
        valid, reason = self.audit_chain.verify_integrity()
        return {
            "node_id": self.node_id,
            "health": self.health.value,
            "buffer_size": self.buffer.size,
            "buffer_evictions": self.buffer.evicted_count,
            "processed_frames": self._processed_frames,
            "dropped_frames": self._dropped_frames,
            "missing_sequences": missing,
            "audit_chain_length": len(self.audit_chain.chain),
            "audit_chain_valid": valid,
            "audit_chain_error": reason,
        }
