#!/usr/bin/env python3
"""Daily watch of upstream official sources for changes and drift."""

from __future__ import annotations

import argparse
import datetime as dt
import difflib
import hashlib
import json
import pathlib
import re
import sys
import urllib.error
import urllib.request
from typing import Any, Callable


def clean_markup(raw_content: str) -> str:
    """Strips scripts, styles, HTML tags and collapses whitespace to isolate text."""
    # Remove script and style blocks
    text = re.sub(r"(?is)<script.*?</script>", " ", raw_content)
    text = re.sub(r"(?is)<style.*?</style>", " ", text)
    # Remove all HTML tags
    text = re.sub(r"(?s)<.*?>", " ", text)
    # Decode basic HTML entities if present
    text = text.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    # Normalize lines and collapse repeated whitespaces
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def compute_hash(text: str) -> str:
    """Computes SHA-256 hash of cleaned text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fetch_url(url: str, timeout: int = 15) -> tuple[bool, str, str]:
    """Fetches URL and returns (success, clean_text, error_message). Never raises."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; LLMControlPlaneWatcher/1.0)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            cleaned = clean_markup(raw)
            return True, cleaned, ""
    except Exception as e:
        return False, "", str(e)


def run_watch(
    lock_path: pathlib.Path,
    watch_dir: pathlib.Path,
    fetcher: Callable[[str], tuple[bool, str, str]] | None = None,
    now: dt.datetime | None = None,
) -> dict[str, Any]:
    """Runs watch logic against sources.lock.json and records diffs. Never exits non-zero on network errors."""
    if fetcher is None:
        fetcher = fetch_url
    if now is None:
        now = dt.datetime.now(dt.timezone.utc)

    today_str = now.strftime("%Y-%m-%d")
    iso_now = now.isoformat()

    if not lock_path.exists():
        return {"status": "error", "message": f"Lock file not found: {lock_path}", "changed": [], "errors": []}

    try:
        lock_data = json.loads(lock_path.read_text(encoding="utf-8"))
    except Exception as e:
        return {"status": "error", "message": f"Invalid JSON in {lock_path}: {e}", "changed": [], "errors": []}

    sources = lock_data.get("sources", [])
    watch_dir.mkdir(parents=True, exist_ok=True)
    latest_path = watch_dir / "latest.json"

    latest_data = {"updated_at": iso_now, "sources": {}}
    if latest_path.exists():
        try:
            latest_data = json.loads(latest_path.read_text(encoding="utf-8"))
            latest_data["updated_at"] = iso_now
        except Exception:
            pass

    changed_list = []
    error_list = []

    for src in sources:
        s_id = src.get("id") or "unknown"
        topic = src.get("topic") or "general"
        url = src.get("source_url") or ""
        expected_hash = src.get("sha256") or ""

        if not url:
            continue

        ok, content, err = fetcher(url)
        if not ok:
            error_list.append({"id": s_id, "url": url, "error": err})
            latest_data.setdefault("sources", {})[s_id] = {
                "status": "unreachable",
                "error": err,
                "last_checked": iso_now,
                "last_hash": expected_hash,
            }
            continue

        new_hash = compute_hash(content)
        if new_hash != expected_hash:
            # Change detected!
            topic_dir = watch_dir / topic
            topic_dir.mkdir(parents=True, exist_ok=True)
            diff_file = topic_dir / f"{today_str}.json"

            # Compute unified diff lines
            expected_content = src.get("baseline_content", "")
            diff_lines = list(
                difflib.unified_diff(
                    expected_content.splitlines(),
                    content.splitlines(),
                    fromfile="baseline",
                    tofile="current",
                    lineterm="",
                )
            )[:100]

            diff_payload = {
                "source_id": s_id,
                "topic": topic,
                "url": url,
                "old_hash": expected_hash,
                "new_hash": new_hash,
                "detected_at": iso_now,
                "diff_lines": diff_lines,
            }
            diff_file.write_text(json.dumps(diff_payload, indent=2), encoding="utf-8")
            changed_list.append({"id": s_id, "url": url, "diff_file": str(diff_file.relative_to(watch_dir.parent))})

            latest_data.setdefault("sources", {})[s_id] = {
                "status": "changed",
                "old_hash": expected_hash,
                "new_hash": new_hash,
                "last_changed": iso_now,
                "last_checked": iso_now,
            }
        else:
            latest_data.setdefault("sources", {})[s_id] = {
                "status": "unchanged",
                "hash": new_hash,
                "last_checked": iso_now,
            }

    latest_path.write_text(json.dumps(latest_data, indent=2), encoding="utf-8")
    return {
        "status": "ok",
        "changed": changed_list,
        "errors": error_list,
        "sources_checked": len(sources),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Daily source watcher for llm-control-plane")
    parser.add_argument("--lock", default="sources.lock.json", help="Path to sources.lock.json")
    parser.add_argument("--watch-dir", default="watch", help="Directory to store watch output")
    args = parser.parse_args()

    lock_file = pathlib.Path(args.lock).resolve()
    watch_dir = pathlib.Path(args.watch_dir).resolve()

    res = run_watch(lock_file, watch_dir)
    ch_len = len(res.get("changed", []))
    err_len = len(res.get("errors", []))
    total = res.get("sources_checked", 0)
    print(f"Watch completed: {total} checked, {ch_len} changed, {err_len} unreachable.")
    for ch in res.get("changed", []):
        print(f"  [CHANGED] {ch['id']}: {ch['url']} -> {ch['diff_file']}")
    for er in res.get("errors", []):
        print(f"  [UNREACHABLE] {er['id']}: {er['url']} ({er['error']})")

    # The script exits 0 even if pages are down, ensuring CI never turns red on network failure
    return 0


if __name__ == "__main__":
    sys.exit(main())
