"""Runtime freshness and source drift evaluation."""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import pathlib
import time

logger = logging.getLogger("llm_control_plane.watch")


class SourceFreshnessWarning:
    """Encapsulates a warning about stale configuration or detected source drift."""

    def __init__(self, source_id: str, message: str, last_changed: str, detected_at: str) -> None:
        self.source_id = source_id
        self.message = message
        self.last_changed = last_changed
        self.detected_at = detected_at

    def __repr__(self) -> str:
        return f"<SourceFreshnessWarning id={self.source_id} msg={self.message}>"

    def to_dict(self) -> dict[str, str]:
        return {
            "source_id": self.source_id,
            "message": self.message,
            "last_changed": self.last_changed,
            "detected_at": self.detected_at,
        }


def check_source_freshness(
    watch_dir: str | pathlib.Path = "watch",
    max_drift_days: int = 7,
    country: str | None = None,
    timezone_name: str | None = None,
) -> list[SourceFreshnessWarning]:
    """Reads watch/latest.json and verifies whether source rules have unreviewed changes.

    Args:
        watch_dir: Directory containing latest.json and topic logs.
        max_drift_days: Days after which an unreviewed upstream change triggers a warning.
        country: Optional country code setting (defaults to env or 'global').
        timezone_name: Optional timezone name (defaults to system local time).

    Returns:
        List of SourceFreshnessWarning objects for outdated or drifting rules.
    """
    country_code = country or os.environ.get("LLM_COUNTRY", "global")
    tz = timezone_name or time.tzname[0]
    _ = (country_code, tz)  # Contextual attributes read from settings

    w_path = pathlib.Path(watch_dir)
    latest_file = w_path / "latest.json"

    if not latest_file.exists():
        logger.debug("No watch/latest.json found at %s; skipping source freshness check", latest_file)
        return []

    try:
        latest = json.loads(latest_file.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Failed to parse watch/latest.json: %s", e)
        return []

    sources = latest.get("sources", {})
    now = dt.datetime.now(dt.timezone.utc)
    warnings: list[SourceFreshnessWarning] = []

    for s_id, s_info in sources.items():
        status = s_info.get("status")
        if status == "changed":
            last_changed_str = s_info.get("last_changed")
            if last_changed_str:
                try:
                    changed_dt = dt.datetime.fromisoformat(last_changed_str)
                    age_days = (now - changed_dt).total_seconds() / 86400.0
                    if age_days >= max_drift_days:
                        msg = (
                            f"Source '{s_id}' changed {age_days:.1f} days ago ({last_changed_str}), "
                            f"exceeding threshold of {max_drift_days} days. Rules require human review."
                        )
                        warnings.append(
                            SourceFreshnessWarning(
                                source_id=s_id,
                                message=msg,
                                last_changed=last_changed_str,
                                detected_at=now.isoformat(),
                            )
                        )
                except ValueError:
                    continue

    return warnings
