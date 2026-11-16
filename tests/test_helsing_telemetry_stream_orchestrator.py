"""Tests for Helsing Resilient Telemetry Stream Orchestrator."""

import hashlib
import pytest
from llm_control_plane.helsing_telemetry_stream_orchestrator import (
    AirGappedCryptographicLedger,
    HelsingTelemetryStreamOrchestrator,
    ResilientRingBuffer,
    StreamingBackpressureController,
    TelemetryFormat,
    TelemetryFrame,
    ThrottlingLevel,
)


def test_telemetry_frame_creation_and_digest():
    payload = b"\x08\x96\x01\x12\x04test"
    frame = TelemetryFrame.create(
        frame_id="frame-001",
        stream_id="rf-sensor-north",
        sequence_number=1,
        payload_bytes=payload,
        format=TelemetryFormat.PROTOBUF,
        priority=1,
        metadata={"snr_db": 18.5, "frequency_mhz": 1420.4},
    )
    assert frame.frame_id == "frame-001"
    assert frame.stream_id == "rf-sensor-north"
    assert frame.sequence_number == 1
    assert frame.payload_sha256 == hashlib.sha256(payload).hexdigest()
    assert frame.priority == 1
    assert frame.metadata["snr_db"] == 18.5


def test_ring_buffer_capacity_and_overflow_drop():
    buf = ResilientRingBuffer(capacity=5)
    assert buf.capacity == 5
    assert buf.current_size == 0
    assert buf.total_dropped == 0

    for i in range(10):
        frame = TelemetryFrame.create(
            frame_id=f"f-{i}",
            stream_id="eo-ir-cam",
            sequence_number=i + 1,
            payload_bytes=f"sample-{i}".encode("utf-8"),
        )
        buf.append(frame)

    assert buf.current_size == 5
    assert buf.total_dropped == 5
    assert buf.utilization_ratio == 1.0

    drained = buf.drain_all()
    assert len(drained) == 5
    assert buf.current_size == 0
    assert drained[0].frame_id == "f-5"
    assert drained[-1].frame_id == "f-9"


def test_sequence_gap_detection():
    buf = ResilientRingBuffer(capacity=50)

    # Ingest sequences 1, 2, then skip to 6
    buf.append(TelemetryFrame.create("f-1", "acoustic-arr", 1, b"d1"))
    buf.append(TelemetryFrame.create("f-2", "acoustic-arr", 2, b"d2"))
    buf.append(TelemetryFrame.create("f-6", "acoustic-arr", 6, b"d6"))

    gaps = buf.get_detected_gaps("acoustic-arr")
    assert gaps == [3, 4, 5]


def test_backpressure_controller_throttling_and_admission():
    bp = StreamingBackpressureController(
        high_watermark=0.80,
        critical_watermark=0.95,
        latency_threshold_ms=25.0,
    )

    # Normal state
    assert bp.evaluate(buffer_utilization=0.40, observed_latency_ms=10.0) == ThrottlingLevel.NORMAL
    critical_frame = TelemetryFrame.create("f-crit", "stream", 1, b"c", priority=1)
    low_frame = TelemetryFrame.create("f-low", "stream", 2, b"l", priority=8)
    assert bp.should_admit(critical_frame) is True
    assert bp.should_admit(low_frame) is True

    # Shed low priority state
    assert bp.evaluate(buffer_utilization=0.85, observed_latency_ms=15.0) == ThrottlingLevel.SHED_LOW_PRIORITY
    assert bp.should_admit(critical_frame) is True
    assert bp.should_admit(low_frame) is False

    # Circuit break state
    assert bp.evaluate(buffer_utilization=0.98, observed_latency_ms=15.0) == ThrottlingLevel.CIRCUIT_BREAK
    p3_frame = TelemetryFrame.create("f-med", "stream", 3, b"m", priority=3)
    assert bp.should_admit(critical_frame) is True
    assert bp.should_admit(p3_frame) is False


def test_air_gapped_cryptographic_ledger_integrity_and_tampering():
    ledger = AirGappedCryptographicLedger()
    assert ledger.height == 0
    assert ledger.head_hash == AirGappedCryptographicLedger.GENESIS_HASH

    f1 = TelemetryFrame.create("f-1", "rf-1", 1, b"payload1")
    f2 = TelemetryFrame.create("f-2", "rf-1", 2, b"payload2")
    f3 = TelemetryFrame.create("f-3", "rf-1", 3, b"payload3")

    h1, b_hash1 = ledger.append_batch("batch-001", [f1, f2])
    assert h1 == 1
    assert ledger.head_hash == b_hash1
    assert ledger.verify_integrity() is True

    h2, b_hash2 = ledger.append_batch("batch-002", [f3])
    assert h2 == 2
    assert ledger.head_hash == b_hash2
    assert ledger.verify_integrity() is True

    # Tamper with block content
    ledger._blocks[0]["batch_id"] = "tampered-batch"
    assert ledger.verify_integrity() is False


def test_merkle_root_computation():
    ledger = AirGappedCryptographicLedger()
    empty_root = ledger.compute_merkle_root([])
    assert empty_root == hashlib.sha256(b"EMPTY_BATCH").hexdigest()

    h1 = hashlib.sha256(b"item1").hexdigest()
    assert ledger.compute_merkle_root([h1]) == h1

    h2 = hashlib.sha256(b"item2").hexdigest()
    expected_root = hashlib.sha256((h1 + h2).encode("utf-8")).hexdigest()
    assert ledger.compute_merkle_root([h1, h2]) == expected_root


def test_helsing_orchestrator_end_to_end_pipeline():
    orchestrator = HelsingTelemetryStreamOrchestrator(
        buffer_capacity=100,
        high_watermark=0.75,
        critical_watermark=0.90,
    )

    # Ingest 15 frames across multiple tactical sensors
    for i in range(10):
        admitted = orchestrator.ingest_frame(
            TelemetryFrame.create(
                frame_id=f"rf-{i}",
                stream_id="rf-array-alpha",
                sequence_number=i + 1,
                payload_bytes=f"telemetry-rf-{i}".encode("utf-8"),
                priority=2,
            )
        )
        assert admitted is True

    for i in range(5):
        admitted = orchestrator.ingest_frame(
            TelemetryFrame.create(
                frame_id=f"cam-{i}",
                stream_id="eo-ir-cam-bravo",
                sequence_number=i + 1,
                payload_bytes=f"telemetry-cam-{i}".encode("utf-8"),
                priority=4,
            )
        )
        assert admitted is True

    # Process batch
    result = orchestrator.process_pending_batch(batch_id="sort-mission-01")
    assert result.batch_id == "sort-mission-01"
    assert result.processed_count == 15
    assert result.ledger_height == 1
    assert result.channel_stats["rf-array-alpha"] == 10
    assert result.channel_stats["eo-ir-cam-bravo"] == 5

    metrics = orchestrator.get_orchestrator_metrics()
    assert metrics["total_ingested"] == 15
    assert metrics["total_processed"] == 15
    assert metrics["buffer_current_size"] == 0
    assert metrics["ledger_verified"] is True
    assert metrics["active_channels"]["rf-array-alpha"] == 10
    assert metrics["active_channels"]["eo-ir-cam-bravo"] == 5
