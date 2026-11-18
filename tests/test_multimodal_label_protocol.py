"""Unit tests for Multimodal Label Format Protocol, Dataset Indexer, and Backpressure Engine."""

from __future__ import annotations

import time
import pytest

from llm_control_plane.multimodal_label_protocol import (
    AnnotationWALReconciler,
    DynamicBackpressureTokenBucket,
    LabelFormatProtocolEngine,
    LabelItem,
    MultimodalDatasetIndexer,
)


def test_label_item_hash_determinism():
    item1 = LabelItem(
        class_name="pedestrian",
        confidence=0.95,
        geometry_type="bbox",
        geometry={"x": 10.0, "y": 20.0, "w": 30.0, "h": 40.0},
    )
    item2 = LabelItem(
        class_name="pedestrian",
        confidence=0.88,
        geometry_type="bbox",
        geometry={"x": 10.0, "y": 20.0, "w": 30.0, "h": 40.0},
    )
    # Different confidence does not change spatial geometry hash
    assert item1.compute_hash() == item2.compute_hash()

    item3 = LabelItem(
        class_name="vehicle",
        confidence=0.95,
        geometry_type="bbox",
        geometry={"x": 10.0, "y": 20.0, "w": 30.0, "h": 40.0},
    )
    assert item1.compute_hash() != item3.compute_hash()


def test_label_format_protocol_serialization_and_validation():
    labels = [
        LabelItem(
            class_name="traffic_light",
            confidence=0.99,
            geometry_type="bbox",
            geometry={"x": 100, "y": 200, "w": 50, "h": 150},
            attributes={"color": "green"},
        ),
        LabelItem(
            class_name="lane_boundary",
            confidence=0.92,
            geometry_type="polyline",
            geometry={"points": [[0, 500], [200, 500], [400, 520]]},
        ),
    ]

    payload = LabelFormatProtocolEngine.serialize_ontology_labels(
        dataset_id="ds_av_highway_01",
        frame_id="frame_00042",
        labels=labels,
    )

    assert payload["protocol_version"] == "2.0.0"
    assert payload["dataset_id"] == "ds_av_highway_01"
    assert payload["frame_id"] == "frame_00042"
    assert payload["item_count"] == 2
    assert len(payload["labels"]) == 2

    valid, err = LabelFormatProtocolEngine.validate_label_schema(payload)
    assert valid is True
    assert err is None


def test_label_deduplication():
    item_dup1 = LabelItem(
        class_name="obstacle",
        confidence=0.9,
        geometry_type="cuboid_3d",
        geometry={"x": 1, "y": 2, "z": 0, "dx": 2, "dy": 4, "dz": 1.5},
    )
    item_dup2 = LabelItem(
        class_name="obstacle",
        confidence=0.95,
        geometry_type="cuboid_3d",
        geometry={"x": 1, "y": 2, "z": 0, "dx": 2, "dy": 4, "dz": 1.5},
    )
    item_unique = LabelItem(
        class_name="curb",
        confidence=0.85,
        geometry_type="polygon",
        geometry={"vertices": [[0, 0], [1, 0], [1, 1]]},
    )

    deduped = LabelFormatProtocolEngine.deduplicate_labels([item_dup1, item_dup2, item_unique])
    assert len(deduped) == 2
    assert deduped[0] == item_dup1
    assert deduped[1] == item_unique


def test_multimodal_dataset_indexer_hybrid_query():
    indexer = MultimodalDatasetIndexer(vector_dim=4)

    # Frame 0: heavy vehicle presence
    indexer.index_frame(
        dataset_id="ds_drive",
        frame_index=0,
        timestamp_ms=0.0,
        embedding=[1.0, 0.0, 0.0, 0.0],
        labels=[
            LabelItem(
                class_name="vehicle",
                confidence=0.95,
                geometry_type="bbox",
                geometry={"x": 10, "y": 10, "w": 50, "h": 50},
            )
        ],
    )

    # Frame 1: pedestrian presence
    indexer.index_frame(
        dataset_id="ds_drive",
        frame_index=1,
        timestamp_ms=100.0,
        embedding=[0.0, 1.0, 0.0, 0.0],
        labels=[
            LabelItem(
                class_name="pedestrian",
                confidence=0.91,
                geometry_type="bbox",
                geometry={"x": 100, "y": 200, "w": 20, "h": 60},
            )
        ],
    )

    # Query with vehicle embedding [1.0, 0.0, 0.0, 0.0] and vehicle filter
    hits = indexer.query_hybrid(
        dataset_id="ds_drive",
        query_embedding=[1.0, 0.0, 0.0, 0.0],
        class_filter="vehicle",
        top_k=5,
    )
    assert len(hits) == 1
    assert hits[0]["frame_index"] == 0
    assert hits[0]["score"] == 1.0

    # Query for pedestrian
    ped_hits = indexer.query_hybrid(
        dataset_id="ds_drive",
        query_embedding=[0.0, 1.0, 0.0, 0.0],
        class_filter="pedestrian",
        top_k=5,
    )
    assert len(ped_hits) == 1
    assert ped_hits[0]["frame_index"] == 1


def test_dynamic_backpressure_token_bucket():
    bucket = DynamicBackpressureTokenBucket(capacity=5.0, refill_rate=2.0, target_p99_ms=100.0)

    # Acquire all tokens
    for _ in range(5):
        ok, wait = bucket.acquire(1.0)
        assert ok is True
        assert wait == 0.0

    # Bucket exhausted
    ok, wait = bucket.acquire(1.0)
    assert ok is False
    assert wait > 0.0

    # Feed high latencies to trigger adaptive backpressure throttling
    for _ in range(25):
        bucket.record_latency(250.0)
    assert bucket.throttle_factor < 1.0


def test_annotation_wal_reconciler():
    wal = AnnotationWALReconciler()

    seq1 = wal.append_entry("insert_label", {"id": "lbl_1", "class": "car"})
    seq2 = wal.append_entry("update_label", {"id": "lbl_1", "confidence": 0.98})
    assert seq1 == 1
    assert seq2 == 2

    cp = wal.checkpoint()
    assert cp == 2

    seq3 = wal.append_entry("delete_label", {"id": "lbl_1"})
    assert seq3 == 3

    unapplied = wal.replay(from_seq=cp)
    assert len(unapplied) == 1
    assert unapplied[0]["type"] == "delete_label"
    assert unapplied[0]["seq"] == 3
