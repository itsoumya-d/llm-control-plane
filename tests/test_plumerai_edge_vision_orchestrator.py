"""Tests for Plumerai Edge Vision & Multimodal VLM Orchestrator."""

import pytest

from llm_control_plane.plumerai_edge_vision_orchestrator import (
    BoundingBox,
    EdgeDeviceProfile,
    PlumeraiEdgeVisionOrchestrator,
    QuantizationPrecision,
    VisionEventTrigger,
    VLMSceneQuery,
)


@pytest.fixture
def orchestrator() -> PlumeraiEdgeVisionOrchestrator:
    return PlumeraiEdgeVisionOrchestrator(memory_budget_bytes=1_048_576)


@pytest.fixture
def sample_device() -> EdgeDeviceProfile:
    return EdgeDeviceProfile(
        device_id="cam_entryway_01",
        hardware_target="arm_cortex_m55",
        max_ram_bytes=1_048_576,
        clock_mhz=250,
        supports_vector_extensions=True,
    )


def test_device_registration(orchestrator: PlumeraiEdgeVisionOrchestrator, sample_device: EdgeDeviceProfile) -> None:
    assert orchestrator.register_device(sample_device) is True
    dev = orchestrator.get_device("cam_entryway_01")
    assert dev is not None
    assert dev.hardware_target == "arm_cortex_m55"
    assert dev.max_ram_bytes == 1_048_576


def test_quantization_calibration_int8(orchestrator: PlumeraiEdgeVisionOrchestrator) -> None:
    # 500,000 weights: in INT8 (1 byte each) = 500,000 bytes (< 1MB)
    weights = [0.12 * (i % 17 - 8) for i in range(500_000)]
    res = orchestrator.calibrate_quantization(
        model_name="plumerai_person_detector_v4",
        baseline_weights=weights,
        target_precision=QuantizationPrecision.INT8,
        max_allowed_drop_pct=5.0,
    )
    assert res.precision == QuantizationPrecision.INT8
    assert res.peak_memory_bytes == 500_000
    assert res.meets_memory_budget is True
    assert res.meets_accuracy_budget is True
    assert res.quantized_map > 0.90


def test_quantization_memory_budget_exceeded(orchestrator: PlumeraiEdgeVisionOrchestrator) -> None:
    # 1,500,000 weights in INT8 = 1.5MB (> 1MB budget)
    weights = [0.05 * (i % 10) for i in range(1_500_000)]
    res = orchestrator.calibrate_quantization(
        model_name="plumerai_heavy_vlm_v1",
        baseline_weights=weights,
        target_precision=QuantizationPrecision.INT8,
    )
    assert res.peak_memory_bytes == 1_500_000
    assert res.meets_memory_budget is False


def test_quantization_empty_weights_raises(orchestrator: PlumeraiEdgeVisionOrchestrator) -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        orchestrator.calibrate_quantization("test", [], QuantizationPrecision.INT8)


def test_edge_event_ingestion(orchestrator: PlumeraiEdgeVisionOrchestrator, sample_device: EdgeDeviceProfile) -> None:
    orchestrator.register_device(sample_device)

    ev = VisionEventTrigger(
        event_id="ev_001",
        device_id="cam_entryway_01",
        timestamp_ms=1763260800000,
        event_type="people_detection",
        confidence=0.94,
        bounding_boxes=[BoundingBox(x_min=0.1, y_min=0.2, x_max=0.5, y_max=0.8, label="person", confidence=0.94)],
        feature_embedding=[0.15, 0.45, -0.22, 0.81],
    )
    assert orchestrator.ingest_edge_event(ev) is True
    assert "ev_001" in orchestrator.event_store


def test_edge_event_low_confidence_dropped(
    orchestrator: PlumeraiEdgeVisionOrchestrator,
    sample_device: EdgeDeviceProfile,
) -> None:
    orchestrator.register_device(sample_device)

    ev_low = VisionEventTrigger(
        event_id="ev_low_01",
        device_id="cam_entryway_01",
        timestamp_ms=1763260801000,
        event_type="motion_alarm",
        confidence=0.25,
    )
    assert orchestrator.ingest_edge_event(ev_low) is False
    assert "ev_low_01" not in orchestrator.event_store


def test_edge_event_unregistered_device_raises(orchestrator: PlumeraiEdgeVisionOrchestrator) -> None:
    ev = VisionEventTrigger(
        event_id="ev_unregistered",
        device_id="unknown_cam",
        timestamp_ms=1763260802000,
        event_type="people_detection",
        confidence=0.88,
    )
    with pytest.raises(KeyError, match="not registered"):
        orchestrator.ingest_edge_event(ev)


def test_cloud_vlm_query(orchestrator: PlumeraiEdgeVisionOrchestrator, sample_device: EdgeDeviceProfile) -> None:
    orchestrator.register_device(sample_device)
    ev = VisionEventTrigger(
        event_id="ev_front_door",
        device_id="cam_entryway_01",
        timestamp_ms=1763260810000,
        event_type="face_recognition",
        confidence=0.97,
        bounding_boxes=[
            BoundingBox(x_min=0.2, y_min=0.1, x_max=0.4, y_max=0.4, label="familiar_face", confidence=0.97),
            BoundingBox(x_min=0.1, y_min=0.1, x_max=0.6, y_max=0.9, label="delivery_courier", confidence=0.91),
        ],
    )
    orchestrator.ingest_edge_event(ev)

    query = VLMSceneQuery(
        query_id="query_caption_01",
        camera_id="cam_entryway_01",
        prompt="Describe scene activity and any familiar faces",
        event_ids=["ev_front_door"],
    )
    resp = orchestrator.query_cloud_vlm(query)
    assert resp.query_id == "query_caption_01"
    assert "delivery_courier" in resp.detected_entities
    assert "familiar_face" in resp.detected_entities
    assert resp.confidence > 0.90
    assert resp.tokens_used > 0


def test_embedding_cosine_similarity(
    orchestrator: PlumeraiEdgeVisionOrchestrator,
    sample_device: EdgeDeviceProfile,
) -> None:
    orchestrator.register_device(sample_device)
    ev1 = VisionEventTrigger(
        event_id="ev_face_a",
        device_id="cam_entryway_01",
        timestamp_ms=1763260820000,
        event_type="face_recognition",
        confidence=0.96,
        feature_embedding=[1.0, 0.0, 0.0],
    )
    ev2 = VisionEventTrigger(
        event_id="ev_face_b",
        device_id="cam_entryway_01",
        timestamp_ms=1763260825000,
        event_type="face_recognition",
        confidence=0.95,
        feature_embedding=[1.0, 0.0, 0.0],
    )
    orchestrator.ingest_edge_event(ev1)
    orchestrator.ingest_edge_event(ev2)

    sim = orchestrator.compute_embedding_cosine_similarity("ev_face_a", "ev_face_b")
    assert pytest.approx(sim, 0.001) == 1.0


def test_embedding_cosine_similarity_mismatch_raises(
    orchestrator: PlumeraiEdgeVisionOrchestrator,
    sample_device: EdgeDeviceProfile,
) -> None:
    orchestrator.register_device(sample_device)
    ev1 = VisionEventTrigger(
        event_id="ev_dim3",
        device_id="cam_entryway_01",
        timestamp_ms=1763260830000,
        event_type="face_recognition",
        confidence=0.90,
        feature_embedding=[1.0, 0.0, 0.0],
    )
    ev2 = VisionEventTrigger(
        event_id="ev_dim2",
        device_id="cam_entryway_01",
        timestamp_ms=1763260835000,
        event_type="face_recognition",
        confidence=0.90,
        feature_embedding=[1.0, 0.0],
    )
    orchestrator.ingest_edge_event(ev1)
    orchestrator.ingest_edge_event(ev2)

    with pytest.raises(ValueError, match="equal dimensionality"):
        orchestrator.compute_embedding_cosine_similarity("ev_dim3", "ev_dim2")


def test_telemetry_summary(orchestrator: PlumeraiEdgeVisionOrchestrator, sample_device: EdgeDeviceProfile) -> None:
    orchestrator.register_device(sample_device)
    ev = VisionEventTrigger(
        event_id="ev_telemetry",
        device_id="cam_entryway_01",
        timestamp_ms=1763260840000,
        event_type="people_detection",
        confidence=0.88,
        feature_embedding=[0.1, 0.2, 0.3],
    )
    orchestrator.ingest_edge_event(ev)
    summary = orchestrator.get_telemetry_summary()
    assert summary["total_registered_devices"] == 1
    assert summary["total_edge_events"] == 1
    assert summary["cached_embeddings"] == 1
    assert summary["telemetry_records"] >= 1
