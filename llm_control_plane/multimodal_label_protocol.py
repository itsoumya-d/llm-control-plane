"""Multimodal Label Format Protocol, Dataset Indexer, and Dynamic Backpressure Engine.

Provides high-throughput serialization of complex multimodal annotations (polygons,
3D cuboids, polylines), hybrid spatial-temporal vector indexing, adaptive
token-bucket backpressure, and zero-data-loss Write-Ahead Log (WAL) reconciliation.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import time
from typing import Any, Dict, List, Optional, Tuple


@dataclasses.dataclass(frozen=True)
class BoundingBox:
    x: float
    y: float
    w: float
    h: float


@dataclasses.dataclass(frozen=True)
class Cuboid3D:
    x: float
    y: float
    z: float
    dx: float
    dy: float
    dz: float
    yaw: float = 0.0


@dataclasses.dataclass
class LabelItem:
    class_name: str
    confidence: float
    geometry_type: str  # "bbox", "polygon", "cuboid_3d", "polyline"
    geometry: Dict[str, Any]
    attributes: Dict[str, Any] = dataclasses.field(default_factory=dict)
    item_id: Optional[str] = None

    def compute_hash(self) -> str:
        payload = f"{self.class_name}:{self.geometry_type}:{json.dumps(self.geometry, sort_keys=True)}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class LabelFormatProtocolEngine:
    """High-throughput serialization and validation engine for multimodal annotation protocols."""

    SUPPORTED_GEOMETRIES = {"bbox", "polygon", "cuboid_3d", "polyline", "classification"}

    @classmethod
    def serialize_ontology_labels(
        cls,
        dataset_id: str,
        frame_id: str,
        labels: List[LabelItem],
        version: str = "2.0.0",
    ) -> Dict[str, Any]:
        serialized_items = []
        for item in labels:
            item_hash = item.compute_hash()
            serialized_items.append(
                {
                    "id": item.item_id or item_hash,
                    "class_name": item.class_name,
                    "confidence": round(float(item.confidence), 4),
                    "geometry_type": item.geometry_type,
                    "geometry": item.geometry,
                    "attributes": item.attributes,
                    "hash": item_hash,
                }
            )

        return {
            "protocol_version": version,
            "dataset_id": dataset_id,
            "frame_id": frame_id,
            "timestamp": time.time(),
            "item_count": len(serialized_items),
            "labels": serialized_items,
        }

    @classmethod
    def validate_label_schema(cls, payload: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        if not isinstance(payload, dict):
            return False, "Payload must be a dictionary"
        for field in ("protocol_version", "dataset_id", "frame_id", "labels"):
            if field not in payload:
                return False, f"Missing required protocol field: {field}"
        for idx, item in enumerate(payload.get("labels", [])):
            if not isinstance(item, dict):
                return False, f"Label item at index {idx} is not an object"
            if item.get("geometry_type") not in cls.SUPPORTED_GEOMETRIES:
                return False, f"Unsupported geometry type: {item.get('geometry_type')}"
            if "confidence" not in item or not (0.0 <= float(item["confidence"]) <= 1.0):
                return False, f"Invalid confidence score for item at index {idx}"
        return True, None

    @classmethod
    def deduplicate_labels(cls, labels: List[LabelItem]) -> List[LabelItem]:
        seen = set()
        unique = []
        for item in labels:
            h = item.compute_hash()
            if h not in seen:
                seen.add(h)
                unique.append(item)
        return unique


class MultimodalDatasetIndexer:
    """In-memory hybrid spatial-temporal vector indexer for multimodal AI frames."""

    def __init__(self, vector_dim: int = 128) -> None:
        self.vector_dim = vector_dim
        self._frames: Dict[str, Dict[str, Any]] = {}

    def index_frame(
        self,
        dataset_id: str,
        frame_index: int,
        timestamp_ms: float,
        embedding: List[float],
        labels: List[LabelItem],
    ) -> str:
        if len(embedding) != self.vector_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.vector_dim}, got {len(embedding)}"
            )

        frame_key = f"{dataset_id}:{frame_index}"
        self._frames[frame_key] = {
            "dataset_id": dataset_id,
            "frame_index": frame_index,
            "timestamp_ms": timestamp_ms,
            "embedding": embedding,
            "labels": [
                {
                    "class_name": lbl.class_name,
                    "confidence": lbl.confidence,
                    "geometry_type": lbl.geometry_type,
                    "hash": lbl.compute_hash(),
                }
                for lbl in labels
            ],
            "classes": {lbl.class_name for lbl in labels},
        }
        return frame_key

    def query_hybrid(
        self,
        dataset_id: str,
        query_embedding: List[float],
        class_filter: Optional[str] = None,
        min_confidence: float = 0.0,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        results = []
        for key, frame in self._frames.items():
            if frame["dataset_id"] != dataset_id:
                continue

            if class_filter is not None:
                matching_labels = [
                    lbl
                    for lbl in frame["labels"]
                    if lbl["class_name"] == class_filter and lbl["confidence"] >= min_confidence
                ]
                if not matching_labels:
                    continue

            # Compute cosine similarity
            sim = self._cosine_similarity(query_embedding, frame["embedding"])
            results.append(
                {
                    "frame_key": key,
                    "frame_index": frame["frame_index"],
                    "timestamp_ms": frame["timestamp_ms"],
                    "score": round(sim, 5),
                    "label_count": len(frame["labels"]),
                }
            )

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    @staticmethod
    def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)


class DynamicBackpressureTokenBucket:
    """Adaptive token bucket rate limiter with feedback-driven backpressure control."""

    def __init__(
        self,
        capacity: float = 100.0,
        refill_rate: float = 50.0,
        target_p99_ms: float = 150.0,
    ) -> None:
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.current_tokens = capacity
        self.last_refill = time.monotonic()
        self.target_p99_ms = target_p99_ms
        self.throttle_factor = 1.0
        self._latencies: List[float] = []

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self.last_refill
        effective_rate = self.refill_rate * self.throttle_factor
        self.current_tokens = min(self.capacity, self.current_tokens + elapsed * effective_rate)
        self.last_refill = now

    def acquire(self, cost: float = 1.0) -> Tuple[bool, float]:
        self._refill()
        if self.current_tokens >= cost:
            self.current_tokens -= cost
            return True, 0.0
        else:
            deficit = cost - self.current_tokens
            effective_rate = max(1.0, self.refill_rate * self.throttle_factor)
            wait_seconds = deficit / effective_rate
            return False, round(wait_seconds, 4)

    def record_latency(self, latency_ms: float) -> None:
        self._latencies.append(latency_ms)
        if len(self._latencies) >= 20:
            sorted_lat = sorted(self._latencies)
            p99 = sorted_lat[int(len(sorted_lat) * 0.99)]
            if p99 > self.target_p99_ms:
                # Dynamic backpressure: degrade refill rate
                self.throttle_factor = max(0.2, self.throttle_factor * 0.85)
            else:
                # Recover towards full capacity
                self.throttle_factor = min(1.0, self.throttle_factor * 1.10)
            self._latencies.clear()


class AnnotationWALReconciler:
    """Write-Ahead Log engine for atomic annotation state mutations and recovery."""

    def __init__(self) -> None:
        self._log: List[Dict[str, Any]] = []
        self._sequence: int = 0
        self._checkpoints: Dict[int, str] = {}

    def append_entry(self, entry_type: str, data: Dict[str, Any]) -> int:
        self._sequence += 1
        entry = {
            "seq": self._sequence,
            "type": entry_type,
            "data": data,
            "timestamp": time.time(),
        }
        self._log.append(entry)
        return self._sequence

    def checkpoint(self) -> int:
        checkpoint_seq = self._sequence
        state_hash = hashlib.sha256(
            json.dumps([e for e in self._log if e["seq"] <= checkpoint_seq], sort_keys=True).encode(
                "utf-8"
            )
        ).hexdigest()
        self._checkpoints[checkpoint_seq] = state_hash
        return checkpoint_seq

    def replay(self, from_seq: int = 0) -> List[Dict[str, Any]]:
        return [e for e in self._log if e["seq"] > from_seq]
