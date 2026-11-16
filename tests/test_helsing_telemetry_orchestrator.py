"""
Unit tests for Helsing Resilient Telemetry Streaming Orchestrator.
"""

from __future__ import annotations

import pytest

from llm_control_plane.helsing_telemetry_orchestrator import (
    AirGappedAuditChain,
    BufferOverflowError,
    HelsingTelemetryStreamer,
    ResilientRingBuffer,
    SensorType,
    StreamHealth,
    TelemetryFrame,
    TelemetryPriority,
)


def _make_frame(
    seq: int,
    sensor: SensorType = SensorType.RF_RADAR,
    priority: TelemetryPriority = TelemetryPriority.TACTICAL_HIGH,
    payload_str: str = "rf-data",
) -> TelemetryFrame:
    return TelemetryFrame(
        frame_id=f"frame-{seq}-{sensor.value}",
        sensor_type=sensor,
        channel_id="ch-tactical-01",
        sequence_num=seq,
        timestamp_ms=1700000000000 + seq * 10,
        priority=priority,
        payload=payload_str.encode("utf-8"),
        metadata={"bearing": 182.4, "elevation": 14.2},
    )


def test_telemetry_frame_creation_and_checksum_verification() -> None:
    frame = _make_frame(1)
    assert frame.checksum != ""
    assert frame.verify_checksum() is True

    # Mutate payload and verify checksum failure
    frame.payload = b"tampered-rf-data"
    assert frame.verify_checksum() is False


def test_resilient_ring_buffer_ordered_ingest() -> None:
    buffer = ResilientRingBuffer(capacity=10)
    for i in range(5):
        accepted = buffer.ingest(_make_frame(i))
        assert accepted is True

    # Test deduplication
    duplicate = buffer.ingest(_make_frame(2))
    assert duplicate is False

    window = buffer.get_ordered_window()
    assert len(window) == 5
    assert [f.sequence_num for f in window] == [0, 1, 2, 3, 4]


def test_resilient_ring_buffer_out_of_order_arrival() -> None:
    buffer = ResilientRingBuffer(capacity=10)
    sequences = [5, 2, 1, 4, 3]
    for s in sequences:
        buffer.ingest(_make_frame(s))

    window = buffer.get_ordered_window()
    assert [f.sequence_num for f in window] == [1, 2, 3, 4, 5]


def test_resilient_ring_buffer_gap_detection() -> None:
    buffer = ResilientRingBuffer(capacity=10)
    for s in [10, 11, 14, 15]:
        buffer.ingest(_make_frame(s))

    missing = buffer.detect_missing_sequences()
    assert missing == [12, 13]


def test_resilient_ring_buffer_priority_eviction_under_capacity() -> None:
    buffer = ResilientRingBuffer(capacity=3)
    buffer.ingest(_make_frame(1, priority=TelemetryPriority.BACKGROUND_TELEMETRY))
    buffer.ingest(_make_frame(2, priority=TelemetryPriority.STANDARD_ROUTING))
    buffer.ingest(_make_frame(3, priority=TelemetryPriority.BACKGROUND_TELEMETRY))

    assert buffer.size == 3
    assert buffer.evicted_count == 0

    # Ingest critical frame, should evict a BACKGROUND_TELEMETRY frame
    critical = _make_frame(4, priority=TelemetryPriority.CRITICAL_FLASH)
    buffer.ingest(critical)

    assert buffer.size == 3
    assert buffer.evicted_count == 1
    seqs = [f.sequence_num for f in buffer.get_ordered_window()]
    assert 4 in seqs


def test_resilient_ring_buffer_overflow_rejection() -> None:
    buffer = ResilientRingBuffer(capacity=2)
    buffer.ingest(_make_frame(1, priority=TelemetryPriority.CRITICAL_FLASH))
    buffer.ingest(_make_frame(2, priority=TelemetryPriority.CRITICAL_FLASH))

    low = _make_frame(3, priority=TelemetryPriority.BACKGROUND_TELEMETRY)
    with pytest.raises(BufferOverflowError):
        buffer.ingest(low)


def test_air_gapped_audit_chain_integrity() -> None:
    chain = AirGappedAuditChain(node_id="test-node-01")
    chain.append_event("BOOT", {"fw": "2.4.0"})
    chain.append_event("SENSOR_ARMED", {"sensor": "RF_RADAR"})
    chain.append_event("TARGET_ACQUIRED", {"track_id": "TRK-9021"})

    valid, err = chain.verify_integrity()
    assert valid is True
    assert err is None
    assert len(chain.chain) == 3


def test_air_gapped_audit_chain_tamper_detection() -> None:
    chain = AirGappedAuditChain(node_id="test-node-01")
    chain.append_event("EVENT_1", {"k": "v1"})
    chain.append_event("EVENT_2", {"k": "v2"})
    chain.append_event("EVENT_3", {"k": "v3"})

    # Tamper with event 2
    chain.chain[1].payload_hash = "tampered" * 8
    valid, err = chain.verify_integrity()
    assert valid is False
    assert "Tampered block hash" in (err or "")


def test_helsing_streamer_health_transition_and_background_shedding() -> None:
    streamer = HelsingTelemetryStreamer(node_id="helsing-munich-alpha")
    streamer.register_channel("ch-opt-01", SensorType.EO_IR_OPTICAL)

    # In optimal broadband, background frame accepted
    bg1 = _make_frame(1, priority=TelemetryPriority.BACKGROUND_TELEMETRY)
    res1 = streamer.ingest_frame(bg1)
    assert res1["status"] == "ACCEPTED"

    # Transition to intermittent radio
    streamer.set_stream_health(StreamHealth.INTERMITTENT_RADIO)
    bg2 = _make_frame(2, priority=TelemetryPriority.BACKGROUND_TELEMETRY)
    res2 = streamer.ingest_frame(bg2)
    assert res2["status"] == "SHED_BACKGROUND"

    # Tactical/critical frames still accepted
    crit = _make_frame(3, priority=TelemetryPriority.CRITICAL_FLASH)
    res3 = streamer.ingest_frame(crit)
    assert res3["status"] == "ACCEPTED"


def test_helsing_streamer_export_batch_for_ml_and_diagnostics() -> None:
    streamer = HelsingTelemetryStreamer(node_id="helsing-munich-alpha")
    streamer.register_channel("ch-rf-01", SensorType.RF_RADAR)
    streamer.register_channel("ch-acoustic-01", SensorType.ACOUSTIC_ARRAY)

    for i in range(1, 6):
        sensor = SensorType.RF_RADAR if i % 2 == 0 else SensorType.ACOUSTIC_ARRAY
        streamer.ingest_frame(_make_frame(i, sensor=sensor))

    batch = streamer.export_batch_for_ml(max_batch=4)
    assert len(batch) == 4
    assert batch[0]["sequence_num"] == 1
    assert "checksum" in batch[0]
    assert "payload_len" in batch[0]

    diag = streamer.get_diagnostics()
    assert diag["node_id"] == "helsing-munich-alpha"
    assert diag["buffer_size"] == 5
    assert diag["audit_chain_valid"] is True
    assert diag["audit_chain_length"] >= 2
