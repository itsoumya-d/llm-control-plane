import pytest

from llm_control_plane.roboflow_edge_cv_active_learning_gateway import (
    ActiveLearningFilter,
    BoundingBox,
    Detection,
    EdgeBatchSyncGateway,
    EdgeDeploymentWatchdog,
    EdgeFramePrediction,
    EdgeSampleRingBuffer,
)


def test_bounding_box_geometry_and_iou():
    box1 = BoundingBox(x_min=10.0, y_min=10.0, x_max=50.0, y_max=50.0)
    assert box1.area() == 1600.0

    # Identical box -> IoU = 1.0
    assert box1.iou(box1) == 1.0

    # Non-overlapping box -> IoU = 0.0
    box2 = BoundingBox(x_min=60.0, y_min=60.0, x_max=100.0, y_max=100.0)
    assert box1.iou(box2) == 0.0

    # Partial overlap
    box3 = BoundingBox(x_min=30.0, y_min=10.0, x_max=70.0, y_max=50.0)
    # intersection: [30, 10] to [50, 50] -> 20 * 40 = 800
    # union: 1600 + 1600 - 800 = 2400 -> 800 / 2400 = 1/3
    assert pytest.approx(box1.iou(box3), rel=1e-3) == 0.3333


def test_active_learning_confidence_filter():
    al_filter = ActiveLearningFilter(min_confidence=0.50, max_confidence=0.82)

    # High confidence detection -> not selected
    high_conf = EdgeFramePrediction(
        frame_id="frame_001",
        timestamp=1000.0,
        camera_id="cam_line_1",
        detections=[
            Detection(label="defect_scratch", confidence=0.96, box=BoundingBox(0, 0, 10, 10))
        ],
        latency_ms=18.5,
    )
    selected, reasons = al_filter.evaluate_sample(high_conf)
    assert not selected
    assert len(reasons) == 0

    # Uncertain detection -> selected for active learning
    uncertain = EdgeFramePrediction(
        frame_id="frame_002",
        timestamp=1001.0,
        camera_id="cam_line_1",
        detections=[
            Detection(label="defect_crack", confidence=0.68, box=BoundingBox(0, 0, 10, 10))
        ],
        latency_ms=19.2,
    )
    selected, reasons = al_filter.evaluate_sample(uncertain)
    assert selected
    assert any("uncertainty_range" in r for r in reasons)


def test_active_learning_iou_conflict_filter():
    al_filter = ActiveLearningFilter(iou_conflict_threshold=0.50)

    # Overlapping conflicting classifications
    b1 = BoundingBox(10, 10, 50, 50)
    b2 = BoundingBox(12, 12, 48, 48)  # High overlap

    conflict_prediction = EdgeFramePrediction(
        frame_id="frame_003",
        timestamp=1002.0,
        camera_id="cam_dock_4",
        detections=[
            Detection(label="pallet", confidence=0.88, box=b1),
            Detection(label="cardboard_box", confidence=0.85, box=b2),
        ],
        latency_ms=22.0,
    )
    selected, reasons = al_filter.evaluate_sample(conflict_prediction)
    assert selected
    assert any("iou_conflict" in r for r in reasons)


def test_edge_sample_ring_buffer_eviction():
    buffer = EdgeSampleRingBuffer(max_capacity=3)
    p = EdgeFramePrediction(
        frame_id="f1", timestamp=1.0, camera_id="c1", detections=[], latency_ms=15.0
    )

    buffer.push(p, ["reason1"])
    buffer.push(p, ["reason2"])
    buffer.push(p, ["reason3"])
    assert buffer.size == 3
    assert buffer.evictions == 0

    # 4th item triggers FIFO eviction of oldest
    buffer.push(p, ["reason4"])
    assert buffer.size == 3
    assert buffer.evictions == 1

    drained = buffer.drain_batch(batch_size=2)
    assert len(drained) == 2
    assert buffer.size == 1


def test_edge_batch_sync_gateway():
    gateway = EdgeBatchSyncGateway()

    # Empty batch
    assert gateway.sync_batch([])["status"] == "EMPTY"

    samples = [{"frame_id": "f_100", "hash": "abc"}]

    # Simulated network failure
    fail_res = gateway.sync_batch(samples, mock_network_failure=True)
    assert fail_res["status"] == "NETWORK_ERROR"
    assert fail_res["consecutive_failures"] == 1
    assert fail_res["backoff_seconds"] == 1.5

    # Successful recovery sync
    success_res = gateway.sync_batch(samples, mock_network_failure=False)
    assert success_res["status"] == "SUCCESS"
    assert success_res["synced_count"] == 1
    assert gateway.consecutive_failures == 0
    assert gateway.total_synced_samples == 1


def test_edge_deployment_watchdog():
    watchdog = EdgeDeploymentWatchdog(max_allowed_latency_ms=30.0)

    # Empty initial state
    assert watchdog.health_summary()["status"] == "IDLE"

    # Add healthy latency readings (< 30ms)
    for lat in [18.0, 20.0, 22.0, 25.0, 28.0]:
        watchdog.record_inference(lat)

    summary = watchdog.health_summary()
    assert summary["healthy"] is True
    assert summary["status"] == "HEALTHY"
    assert summary["p95_latency_ms"] < 30.0

    # Add latency spikes (> 30ms) to trigger degradation
    for lat in [45.0, 60.0, 55.0, 70.0]:
        watchdog.record_inference(lat)

    degraded_summary = watchdog.health_summary()
    assert degraded_summary["healthy"] is False
    assert degraded_summary["status"] == "DEGRADED"
