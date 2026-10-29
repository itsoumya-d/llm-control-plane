"""Append-only, hash-chained audit log for AI requests and evaluations."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

GENESIS_HASH = "0" * 64


def canonical_json(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass
class AuditEvent:
    seq: int
    timestamp: float
    event_type: str
    payload: Dict[str, Any]
    prev_hash: str
    hash: str

    def body(self) -> Dict[str, Any]:
        return {
            "seq": self.seq,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "payload": self.payload,
            "prev_hash": self.prev_hash,
        }

    def compute_hash(self) -> str:
        return sha256_hex(canonical_json(self.body()))

    def to_dict(self) -> Dict[str, Any]:
        return {**self.body(), "hash": self.hash}


class AuditLog:
    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self.clock = clock
        self._events: List[AuditEvent] = []
        self._lock = threading.Lock()

    def append(self, event_type: str, payload: Dict[str, Any]) -> AuditEvent:
        with self._lock:
            prev_hash = self._events[-1].hash if self._events else GENESIS_HASH
            event = AuditEvent(
                seq=len(self._events) + 1,
                timestamp=self.clock(),
                event_type=event_type,
                payload=payload,
                prev_hash=prev_hash,
                hash="",
            )
            event.hash = event.compute_hash()
            self._events.append(event)
            return event

    def events(self) -> List[AuditEvent]:
        with self._lock:
            return list(self._events)

    def __len__(self) -> int:
        return len(self._events)

    def verify(self) -> bool:
        with self._lock:
            previous = GENESIS_HASH
            for index, event in enumerate(self._events, start=1):
                if event.seq != index:
                    return False
                if event.prev_hash != previous:
                    return False
                if event.hash != event.compute_hash():
                    return False
                previous = event.hash
            return True

    def to_jsonl(self) -> str:
        return "\n".join(canonical_json(event.to_dict()) for event in self.events())

    def find(self, trace_id: str) -> List[AuditEvent]:
        return [
            event
            for event in self.events()
            if event.payload.get("trace_id") == trace_id
        ]

    def head_hash(self) -> str:
        events = self.events()
        return events[-1].hash if events else GENESIS_HASH

    @classmethod
    def from_events(cls, events: List[Dict[str, Any]]) -> "AuditLog":
        log = cls()
        previous = GENESIS_HASH
        for raw in events:
            event = AuditEvent(
                seq=int(raw["seq"]),
                timestamp=float(raw["timestamp"]),
                event_type=str(raw["event_type"]),
                payload=dict(raw["payload"]),
                prev_hash=str(raw["prev_hash"]),
                hash=str(raw["hash"]),
            )
            if event.prev_hash != previous:
                raise ValueError("audit chain is broken")
            if event.hash != event.compute_hash():
                raise ValueError(f"audit event {event.seq} hash mismatch")
            log._events.append(event)
            previous = event.hash
        return log
