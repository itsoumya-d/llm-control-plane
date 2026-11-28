"""Tests for daily source watching, drift detection, and runtime freshness checks."""

from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

from llm_control_plane.watch import check_source_freshness
from scripts.watch import clean_markup, compute_hash, run_watch


def test_clean_markup_strips_html_and_scripts() -> None:
    html = """
    <html>
        <head><title>Test</title><style>body { color: red; }</style></head>
        <body>
            <script>console.log("ignore me");</script>
            <h1>Claude 3.5 Sonnet</h1>
            <p>High intelligence model &nbsp; with 200k context.</p>
        </body>
    </html>
    """
    cleaned = clean_markup(html)
    assert "console.log" not in cleaned
    assert "color: red" not in cleaned
    assert "Claude 3.5 Sonnet" in cleaned
    assert "High intelligence model with 200k context." in cleaned


def test_watch_unchanged_produces_no_diff(tmp_path: Path) -> None:
    test_content = "Claude 3.5 Sonnet specification v1.0"
    content_hash = compute_hash(test_content)

    lock_file = tmp_path / "sources.lock.json"
    lock_data = {
        "sources": [
            {
                "id": "anthropic-models",
                "topic": "providers",
                "source_url": "https://docs.anthropic.com/models",
                "sha256": content_hash,
                "baseline_content": test_content,
            }
        ]
    }
    lock_file.write_text(json.dumps(lock_data), encoding="utf-8")

    watch_dir = tmp_path / "watch"

    def mock_fetcher(url: str) -> tuple[bool, str, str]:
        return True, test_content, ""

    result = run_watch(lock_file, watch_dir, fetcher=mock_fetcher)
    assert result["status"] == "ok"
    assert len(result["changed"]) == 0
    assert len(result["errors"]) == 0

    # Ensure no dated diff files were written in topic folder
    topic_dir = watch_dir / "providers"
    assert not topic_dir.exists() or len(list(topic_dir.glob("*.json"))) == 0

    # Verify latest.json records status as unchanged
    latest_file = watch_dir / "latest.json"
    assert latest_file.exists()
    latest_data = json.loads(latest_file.read_text(encoding="utf-8"))
    assert latest_data["sources"]["anthropic-models"]["status"] == "unchanged"


def test_watch_changed_emits_dated_diff(tmp_path: Path) -> None:
    original_content = "Claude 3.5 Sonnet specification v1.0"
    updated_content = "Claude 3.5 Sonnet specification v1.1 - Added 128k output tokens"

    old_hash = compute_hash(original_content)
    new_hash = compute_hash(updated_content)

    lock_file = tmp_path / "sources.lock.json"
    lock_data = {
        "sources": [
            {
                "id": "anthropic-models",
                "topic": "providers",
                "source_url": "https://docs.anthropic.com/models",
                "sha256": old_hash,
                "baseline_content": original_content,
            }
        ]
    }
    lock_file.write_text(json.dumps(lock_data), encoding="utf-8")

    watch_dir = tmp_path / "watch"
    fixed_now = dt.datetime(2026, 11, 28, 6, 0, 0, tzinfo=dt.timezone.utc)

    def mock_fetcher(url: str) -> tuple[bool, str, str]:
        return True, updated_content, ""

    result = run_watch(lock_file, watch_dir, fetcher=mock_fetcher, now=fixed_now)
    assert result["status"] == "ok"
    assert len(result["changed"]) == 1

    diff_file = watch_dir / "providers" / "2026-11-28.json"
    assert diff_file.exists()

    diff_data = json.loads(diff_file.read_text(encoding="utf-8"))
    assert diff_data["source_id"] == "anthropic-models"
    assert diff_data["old_hash"] == old_hash
    assert diff_data["new_hash"] == new_hash
    assert len(diff_data["diff_lines"]) > 0

    latest_file = watch_dir / "latest.json"
    latest_data = json.loads(latest_file.read_text(encoding="utf-8"))
    assert latest_data["sources"]["anthropic-models"]["status"] == "changed"
    assert latest_data["sources"]["anthropic-models"]["new_hash"] == new_hash


def test_watch_resilient_to_network_failure(tmp_path: Path) -> None:
    lock_file = tmp_path / "sources.lock.json"
    lock_data = {
        "sources": [
            {
                "id": "anthropic-models",
                "topic": "providers",
                "source_url": "https://docs.anthropic.com/models",
                "sha256": "abcdef123456",
            }
        ]
    }
    lock_file.write_text(json.dumps(lock_data), encoding="utf-8")

    watch_dir = tmp_path / "watch"

    def mock_fetcher_failure(url: str) -> tuple[bool, str, str]:
        return False, "", "HTTP 503 Service Unavailable"

    result = run_watch(lock_file, watch_dir, fetcher=mock_fetcher_failure)
    assert result["status"] == "ok"
    assert len(result["changed"]) == 0
    assert len(result["errors"]) == 1
    assert result["errors"][0]["id"] == "anthropic-models"

    latest_file = watch_dir / "latest.json"
    assert latest_file.exists()
    latest_data = json.loads(latest_file.read_text(encoding="utf-8"))
    assert latest_data["sources"]["anthropic-models"]["status"] == "unreachable"


def test_check_source_freshness_clean(tmp_path: Path) -> None:
    watch_dir = tmp_path / "watch"
    watch_dir.mkdir(parents=True, exist_ok=True)
    latest_file = watch_dir / "latest.json"

    latest_data = {
        "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "sources": {
            "anthropic-models": {
                "status": "unchanged",
                "hash": "abcdef123456",
            }
        },
    }
    latest_file.write_text(json.dumps(latest_data), encoding="utf-8")

    warnings = check_source_freshness(watch_dir=watch_dir, max_drift_days=7)
    assert len(warnings) == 0


def test_check_source_freshness_stale_warning(tmp_path: Path) -> None:
    watch_dir = tmp_path / "watch"
    watch_dir.mkdir(parents=True, exist_ok=True)
    latest_file = watch_dir / "latest.json"

    stale_date = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=10)).isoformat()
    latest_data = {
        "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "sources": {
            "anthropic-models": {
                "status": "changed",
                "last_changed": stale_date,
            }
        },
    }
    latest_file.write_text(json.dumps(latest_data), encoding="utf-8")

    warnings = check_source_freshness(watch_dir=watch_dir, max_drift_days=7)
    assert len(warnings) == 1
    assert warnings[0].source_id == "anthropic-models"
    assert "exceeding threshold of 7 days" in warnings[0].message
    assert isinstance(warnings[0].to_dict(), dict)


def test_watch_cli_execution(tmp_path: Path) -> None:
    lock_file = tmp_path / "sources.lock.json"
    lock_data = {"sources": []}
    lock_file.write_text(json.dumps(lock_data), encoding="utf-8")
    watch_dir = tmp_path / "watch"

    cmd = [
        sys.executable,
        "scripts/watch.py",
        "--lock",
        str(lock_file),
        "--watch-dir",
        str(watch_dir),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 0
    assert "Watch completed" in proc.stdout
