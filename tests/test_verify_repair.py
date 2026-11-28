from __future__ import annotations

import json

from llm_control_plane.verify_repair import (
    load_rubric,
    repair_gateway_output,
    verify_gateway_output,
)


def _valid_fixture() -> dict:
    return {
        "text": "Hello world from model.",
        "provider": "anthropic",
        "model": "claude-3-haiku",
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
        "cost_usd": 0.00015,
        "latency_ms": 120.5,
        "trace_id": "tr-valid-uuid-1234",
        "audit_seq": 42,
        "redactions": {},
    }


def test_verify_passes_valid_output():
    data = _valid_fixture()
    score = verify_gateway_output(data)
    assert score.passed is True
    assert score.score == 100
    assert len(score.checks) == 5
    for c in score.checks:
        assert c.passed is True


def test_repair_fixes_mechanical_faults(tmp_path):
    broken = {
        "text": "Valid text output.",
        "provider": "openai",
        "model": "gpt-4o-mini",
        "trace_id": "  tr-spaces-trimmed  ",
        "usage": {
            "prompt_tokens": -5,
            "completion_tokens": -2,
            "total_tokens": 999,
        },
        "cost_usd": -0.05,
        "latency_ms": -10.0,
        # missing audit_seq
    }
    initial = verify_gateway_output(broken)
    assert initial.passed is False
    assert initial.score < 80

    ledger_file = tmp_path / "ledger.jsonl"
    result = repair_gateway_output(broken, ledger_path=str(ledger_file), max_rounds=3)

    assert result.passed is True
    assert result.final_score >= 80
    assert result.repaired_output["trace_id"] == "tr-spaces-trimmed"
    assert result.repaired_output["cost_usd"] == 0.0
    assert result.repaired_output["latency_ms"] == 0.0
    assert result.repaired_output["audit_seq"] == 0
    assert result.repaired_output["usage"]["prompt_tokens"] == 0
    assert result.repaired_output["usage"]["completion_tokens"] == 0
    assert result.repaired_output["usage"]["total_tokens"] == 0
    assert len(result.unresolved_issues) == 0


def test_repair_reports_non_mechanical_faults():
    # Empty text and empty provider cannot be mechanically repaired
    non_mechanical_broken = {
        "text": "",
        "provider": "",
        "model": "gpt-4o",
        "trace_id": "valid-trace",
        "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        "cost_usd": 0.01,
        "latency_ms": 50.0,
        "audit_seq": 1,
    }

    result = repair_gateway_output(non_mechanical_broken, max_rounds=3)
    assert result.passed is False
    assert result.repaired_output["text"] == ""  # Never guessed or invented
    assert result.repaired_output["provider"] == ""
    assert any("[HUMAN_ACTION_REQUIRED]" in issue for issue in result.unresolved_issues)


def test_max_rounds_cap(tmp_path):
    broken = {
        "text": "",
        "provider": "",
    }
    ledger_file = tmp_path / "ledger_cap.jsonl"
    result = repair_gateway_output(broken, ledger_path=str(ledger_file), max_rounds=3)
    assert result.rounds == 3
    assert result.passed is False


def test_rubric_file_integrity():
    rubric = load_rubric()
    assert rubric["threshold"] == 80
    checks = rubric["checks"]
    assert len(checks) == 5
    total_weight = sum(c["weight"] for c in checks)
    assert total_weight == 100


def test_ledger_jsonl_output(tmp_path):
    broken = {
        "text": "Repaired sample",
        "provider": "anthropic",
        "model": "claude-3-5-sonnet",
        "trace_id": "tr-ledger-test",
        "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        "cost_usd": -0.01,
        "latency_ms": -5.0,
    }
    ledger_file = tmp_path / "audit_ledger.jsonl"
    result = repair_gateway_output(broken, ledger_path=str(ledger_file), max_rounds=3)
    assert result.passed is True

    lines = ledger_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 1
    for line in lines:
        entry = json.loads(line)
        assert "timestamp" in entry
        assert "round" in entry
        assert "initial_score" in entry
        assert "final_score" in entry
        assert "repaired_fields" in entry
        assert "unresolved" in entry
