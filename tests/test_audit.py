import json

import pytest

from llm_control_plane.audit import GENESIS_HASH, AuditLog


def test_append_and_verify_chain():
    log = AuditLog(clock=lambda: 100.0)
    first = log.append("request.accepted", {"trace_id": "t1"})
    second = log.append("request.completed", {"trace_id": "t1"})
    assert first.seq == 1
    assert first.prev_hash == GENESIS_HASH
    assert second.prev_hash == first.hash
    assert log.verify() is True
    assert log.head_hash() == second.hash


def test_tampering_is_detected():
    log = AuditLog(clock=lambda: 100.0)
    log.append("request.accepted", {"trace_id": "t1"})
    log.append("request.completed", {"trace_id": "t1"})
    log.events()[0].payload["trace_id"] = "tampered"
    assert log.verify() is False


def test_find_returns_trace_events():
    log = AuditLog(clock=lambda: 100.0)
    log.append("request.accepted", {"trace_id": "t1"})
    log.append("request.accepted", {"trace_id": "t2"})
    log.append("request.completed", {"trace_id": "t1"})
    assert len(log.find("t1")) == 2
    assert len(log.find("missing")) == 0


def test_jsonl_round_trip():
    log = AuditLog(clock=lambda: 100.0)
    log.append("request.accepted", {"trace_id": "t1"})
    log.append("request.completed", {"trace_id": "t1"})
    encoded = log.to_jsonl()
    raw = [json.loads(line) for line in encoded.splitlines()]
    restored = AuditLog.from_events(raw)
    assert restored.verify() is True
    assert restored.head_hash() == log.head_hash()


def test_from_events_rejects_broken_chain():
    log = AuditLog(clock=lambda: 100.0)
    log.append("request.accepted", {"trace_id": "t1"})
    raw = json.loads(log.to_jsonl())
    raw["prev_hash"] = "f" * 64
    with pytest.raises(ValueError):
        AuditLog.from_events([raw])
