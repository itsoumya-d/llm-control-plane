"""End-to-end integration tests for the agent skill and CLI automation."""

import json
import re
import tempfile
from pathlib import Path

from llm_control_plane.cli import main


def test_skill_md_contract():
    root = Path(__file__).resolve().parent.parent
    skill_file = root / "skills" / "llm-control-plane" / "SKILL.md"
    assert skill_file.exists(), "skills/llm-control-plane/SKILL.md must exist"

    text = skill_file.read_text(encoding="utf-8")
    assert "Use this skill when" in text
    steps = re.findall(r"(?m)^\s*\d+[.)]\s", text)
    assert len(steps) >= 5, f"Expected at least 5 numbered steps in SKILL.md, found {len(steps)}"


def test_cli_health_end_to_end(capsys):
    ret = main(["health"])
    assert ret == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["status"] == "healthy"
    assert data["audit_chain_valid"] is True
    assert "fake-primary" in data["providers"]


def test_cli_route_end_to_end(capsys):
    ret = main(["route", "--prompt", "What is 2+2?", "--model-alias", "smart"])
    assert ret == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["text"] == "4"
    assert data["provider"] in {"fake-primary", "fake-fallback"}
    assert "trace_id" in data


def test_cli_eval_end_to_end(capsys):
    ret = main(["eval"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "| Case |" in captured.out
    assert "control-plane-smoke" in captured.out


def test_cli_verify_end_to_end(capsys):
    clean_payload = {
        "text": "Valid completion content",
        "provider": "fake-primary",
        "model": "fake-large",
        "trace_id": "tr-test-clean",
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        "cost_usd": 0.001,
        "latency_ms": 120.0,
        "audit_seq": 1,
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = Path(tmpdir) / "clean.json"
        input_file.write_text(json.dumps(clean_payload), encoding="utf-8")

        ret = main(["verify", "--input", str(input_file)])
        assert ret == 0
        captured = capsys.readouterr()
        res = json.loads(captured.out)
        assert res["score"] == 100
        assert res["passed"] is True


def test_cli_repair_end_to_end(capsys):
    broken_payload = {
        "text": "Generated response",
        "provider": "fake-primary",
        "model": "fake-large",
        "trace_id": "  tr-untrimmed-id  ",
        "usage": {"prompt_tokens": -5, "completion_tokens": -2, "total_tokens": 50},
        "cost_usd": -0.05,
        "latency_ms": -10.0,
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = Path(tmpdir) / "broken.json"
        output_file = Path(tmpdir) / "repaired.json"
        ledger_file = Path(tmpdir) / "ledger.jsonl"
        input_file.write_text(json.dumps(broken_payload), encoding="utf-8")

        # Dry run without --apply: output file should not be created
        ret_dry = main([
            "repair",
            "--input", str(input_file),
            "--output", str(output_file),
            "--ledger", str(ledger_file),
        ])
        assert ret_dry == 0
        assert not output_file.exists()
        capsys.readouterr()

        # Apply with explicit --apply confirmation
        ret_apply = main([
            "repair",
            "--input", str(input_file),
            "--output", str(output_file),
            "--ledger", str(ledger_file),
            "--apply",
        ])
        assert ret_apply == 0
        assert output_file.exists()
        assert ledger_file.exists()

        captured = capsys.readouterr()
        res = json.loads(captured.out)
        assert res["applied"] is True
        assert res["passed"] is True
        assert res["final_score"] >= 80
        assert any("trace_id" in f for f in res["repaired_fields"])

        # Check repaired file contents
        repaired_data = json.loads(output_file.read_text(encoding="utf-8"))
        assert repaired_data["trace_id"] == "tr-untrimmed-id"
        assert repaired_data["cost_usd"] == 0.0
        assert repaired_data["latency_ms"] == 0.0
