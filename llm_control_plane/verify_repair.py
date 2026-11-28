"""Verify-and-repair loop for LLM Control Plane gateway output.

Reviews gateway outputs against a data-defined rubric (rubric.json),
mechanically repairs structural or out-of-bounds fields up to 3 rounds,
reports non-mechanical issues for human review, and logs all attempts to a jsonl ledger.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class CheckResult:
    name: str
    weight: int
    passed: bool
    details: str
    mechanical: bool


@dataclass(frozen=True)
class VerificationScore:
    score: int
    threshold: int
    passed: bool
    checks: List[CheckResult]

    @property
    def check_map(self) -> Dict[str, bool]:
        return {c.name: c.passed for c in self.checks}


@dataclass
class RepairResult:
    repaired_output: Dict[str, Any]
    final_score: int
    passed: bool
    rounds: int
    repaired_fields: List[str]
    unresolved_issues: List[str]
    history: List[Dict[str, Any]] = field(default_factory=list)


def default_rubric_path() -> Path:
    return Path(__file__).parent / "rubric.json"


def load_rubric(rubric_path: Optional[str] = None) -> Dict[str, Any]:
    path = Path(rubric_path) if rubric_path else default_rubric_path()
    with open(path, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)
    return data


def verify_gateway_output(data: Dict[str, Any], rubric_path: Optional[str] = None) -> VerificationScore:
    """Scores gateway output dict from 0 to 100 against rubric checks."""
    rubric = load_rubric(rubric_path)
    threshold = int(rubric.get("threshold", 80))
    checks_config = rubric.get("checks", [])

    results: List[CheckResult] = []
    total_score = 0

    for c in checks_config:
        name = c["name"]
        weight = int(c["weight"])
        is_mech = bool(c.get("mechanical", True))

        if name == "required_envelope":
            required_keys = {"text", "provider", "model", "usage", "cost_usd", "latency_ms", "trace_id", "audit_seq"}
            missing = required_keys - set(data.keys())
            if not missing:
                results.append(CheckResult(name, weight, True, "All required envelope fields present", is_mech))
                total_score += weight
            else:
                results.append(CheckResult(name, weight, False, f"Missing fields: {sorted(missing)}", is_mech))

        elif name == "trace_id_format":
            tid = data.get("trace_id")
            if isinstance(tid, str) and tid.strip() and tid == tid.strip():
                results.append(CheckResult(name, weight, True, "Valid formatted trace_id", is_mech))
                total_score += weight
            else:
                results.append(CheckResult(name, weight, False, f"Invalid or unformatted trace_id: {tid!r}", is_mech))

        elif name == "usage_metrics_non_negative":
            usage = data.get("usage")
            if isinstance(usage, dict):
                p = usage.get("prompt_tokens")
                c_tok = usage.get("completion_tokens")
                tot = usage.get("total_tokens")
                valid_num = (
                    isinstance(p, (int, float))
                    and p >= 0
                    and isinstance(c_tok, (int, float))
                    and c_tok >= 0
                    and (tot is None or tot == p + c_tok)
                )
                if valid_num:
                    results.append(
                        CheckResult(name, weight, True, "Usage tokens non-negative and consistent", is_mech)
                    )
                    total_score += weight
                else:
                    results.append(CheckResult(name, weight, False, f"Inconsistent usage tokens: {usage}", is_mech))
            else:
                results.append(CheckResult(name, weight, False, f"Usage is not a dict: {usage!r}", is_mech))

        elif name == "cost_and_latency_bounds":
            cost = data.get("cost_usd")
            latency = data.get("latency_ms")
            cost_ok = isinstance(cost, (int, float)) and cost >= 0.0
            lat_ok = isinstance(latency, (int, float)) and latency >= 0.0
            if cost_ok and lat_ok:
                results.append(
                    CheckResult(name, weight, True, "cost_usd and latency_ms within non-negative bounds", is_mech)
                )
                total_score += weight
            else:
                results.append(
                    CheckResult(name, weight, False, f"Invalid cost ({cost}) or latency ({latency})", is_mech)
                )

        elif name == "content_presence":
            txt = data.get("text")
            prov = data.get("provider")
            txt_ok = isinstance(txt, str) and len(txt.strip()) > 0
            prov_ok = isinstance(prov, str) and len(prov.strip()) > 0
            if txt_ok and prov_ok:
                results.append(CheckResult(name, weight, True, "Completion text and provider present", is_mech))
                total_score += weight
            else:
                results.append(
                    CheckResult(
                        name,
                        weight,
                        False,
                        f"Missing or empty text ({txt!r}) or provider ({prov!r})",
                        is_mech,
                    )
                )

        else:
            results.append(CheckResult(name, weight, False, f"Unknown check: {name}", is_mech))

    all_non_mech_passed = all(c.passed for c in results if not c.mechanical)
    is_passed = (total_score >= threshold) and all_non_mech_passed

    return VerificationScore(
        score=total_score,
        threshold=threshold,
        passed=is_passed,
        checks=results,
    )


def repair_gateway_output(
    data: Dict[str, Any],
    rubric_path: Optional[str] = None,
    ledger_path: Optional[str] = None,
    max_rounds: int = 3,
) -> RepairResult:
    """Iteratively repairs mechanical errors in gateway output up to max_rounds, logging to ledger."""
    current = copy.deepcopy(data)
    repaired_fields_set: set[str] = set()
    unresolved_issues: List[str] = []
    history: List[Dict[str, Any]] = []

    for round_idx in range(1, max_rounds + 1):
        score_before = verify_gateway_output(current, rubric_path)
        round_repairs: List[str] = []

        if score_before.passed:
            history.append({
                "round": round_idx,
                "score_before": score_before.score,
                "score_after": score_before.score,
                "repairs": [],
                "passed": True,
            })
            if ledger_path:
                _write_ledger_entry(ledger_path, round_idx, score_before.score, score_before.score, [], [])
            return RepairResult(
                repaired_output=current,
                final_score=score_before.score,
                passed=True,
                rounds=round_idx,
                repaired_fields=sorted(repaired_fields_set),
                unresolved_issues=unresolved_issues,
                history=history,
            )

        # Inspect failed checks
        for c in score_before.checks:
            if c.passed:
                continue

            if not c.mechanical:
                flag = f"[HUMAN_ACTION_REQUIRED] {c.name}: {c.details}"
                if flag not in unresolved_issues:
                    unresolved_issues.append(flag)
                continue

            # Mechanical repairs
            if c.name == "required_envelope":
                if "audit_seq" not in current:
                    current["audit_seq"] = 0
                    round_repairs.append("audit_seq")
                if "redactions" not in current:
                    current["redactions"] = {}
                    round_repairs.append("redactions")
                if "cost_usd" not in current:
                    current["cost_usd"] = 0.0
                    round_repairs.append("cost_usd")
                if "latency_ms" not in current:
                    current["latency_ms"] = 0.0
                    round_repairs.append("latency_ms")
                if "model" not in current:
                    current["model"] = "unknown-model"
                    round_repairs.append("model")
                if "trace_id" not in current:
                    current["trace_id"] = "00000000-0000-0000-0000-000000000000"
                    round_repairs.append("trace_id")
                if "usage" not in current:
                    current["usage"] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                    round_repairs.append("usage")

            elif c.name == "trace_id_format":
                tid = current.get("trace_id")
                if isinstance(tid, str):
                    current["trace_id"] = tid.strip()
                    round_repairs.append("trace_id:strip")
                else:
                    current["trace_id"] = "00000000-0000-0000-0000-000000000000"
                    round_repairs.append("trace_id:default")

            elif c.name == "usage_metrics_non_negative":
                usage = current.get("usage")
                if not isinstance(usage, dict):
                    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                    current["usage"] = usage
                    round_repairs.append("usage:init_dict")
                p = max(0, int(usage.get("prompt_tokens", 0)))
                comp = max(0, int(usage.get("completion_tokens", 0)))
                usage["prompt_tokens"] = p
                usage["completion_tokens"] = comp
                usage["total_tokens"] = p + comp
                round_repairs.append("usage:clamped_and_summed")

            elif c.name == "cost_and_latency_bounds":
                cost = current.get("cost_usd", 0.0)
                lat = current.get("latency_ms", 0.0)
                if not isinstance(cost, (int, float)) or cost < 0.0:
                    current["cost_usd"] = 0.0
                    round_repairs.append("cost_usd:clamp_zero")
                if not isinstance(lat, (int, float)) or lat < 0.0:
                    current["latency_ms"] = 0.0
                    round_repairs.append("latency_ms:clamp_zero")

        repaired_fields_set.update(round_repairs)
        score_after = verify_gateway_output(current, rubric_path)

        history.append({
            "round": round_idx,
            "score_before": score_before.score,
            "score_after": score_after.score,
            "repairs": round_repairs,
            "passed": score_after.passed,
        })

        if ledger_path:
            _write_ledger_entry(
                ledger_path,
                round_idx,
                score_before.score,
                score_after.score,
                round_repairs,
                unresolved_issues,
            )

        if score_after.passed:
            return RepairResult(
                repaired_output=current,
                final_score=score_after.score,
                passed=True,
                rounds=round_idx,
                repaired_fields=sorted(repaired_fields_set),
                unresolved_issues=unresolved_issues,
                history=history,
            )

    final_score = verify_gateway_output(current, rubric_path)
    return RepairResult(
        repaired_output=current,
        final_score=final_score.score,
        passed=final_score.passed,
        rounds=max_rounds,
        repaired_fields=sorted(repaired_fields_set),
        unresolved_issues=unresolved_issues,
        history=history,
    )


def _write_ledger_entry(
    ledger_path: str,
    round_idx: int,
    initial_score: int,
    final_score: int,
    repairs: List[str],
    unresolved: List[str],
) -> None:
    path = Path(ledger_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "round": round_idx,
        "initial_score": initial_score,
        "final_score": final_score,
        "repaired_fields": repairs,
        "unresolved": unresolved,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")
