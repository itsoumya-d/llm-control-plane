"""Versioned datasets, deterministic scorers, and experiment comparison."""

from __future__ import annotations

import json
import math
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence

from .audit import canonical_json, sha256_hex
from .gateway import LLMGateway
from .types import ChatMessage, CompletionRequest


@dataclass(frozen=True)
class EvalCase:
    id: str
    input: str
    expected: Any
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "input": self.input,
            "expected": self.expected,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class Dataset:
    name: str
    version: str
    cases: List[EvalCase]

    def __post_init__(self) -> None:
        if not self.cases:
            raise ValueError("dataset needs at least one case")
        ids = [case.id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("dataset case ids must be unique")

    def content_hash(self) -> str:
        payload = {"name": self.name, "version": self.version, "cases": [case.to_dict() for case in self.cases]}
        return sha256_hex(canonical_json(payload))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "content_hash": self.content_hash(),
            "cases": [case.to_dict() for case in self.cases],
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Dataset":
        return cls(
            name=str(raw["name"]),
            version=str(raw["version"]),
            cases=[
                EvalCase(
                    id=str(case["id"]),
                    input=str(case["input"]),
                    expected=case.get("expected"),
                    metadata=dict(case.get("metadata") or {}),
                )
                for case in raw["cases"]
            ],
        )


@dataclass(frozen=True)
class Score:
    name: str
    passed: bool
    value: float
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "value": self.value, "detail": self.detail}


class Scorer(Protocol):
    name: str

    def score(self, case: EvalCase, output: str, latency_ms: float) -> Score:
        ...


class ExactMatchScorer:
    name = "exact_match"

    def __init__(self, normalize: bool = True) -> None:
        self.normalize = normalize

    def _normalize(self, value: str) -> str:
        if not self.normalize:
            return value
        return re.sub(r"\s+", " ", value).strip().lower()

    def score(self, case: EvalCase, output: str, latency_ms: float) -> Score:
        expected = case.expected if isinstance(case.expected, str) else str(case.expected)
        passed = self._normalize(output) == self._normalize(expected)
        return Score(self.name, passed, 1.0 if passed else 0.0, f"expected={expected!r}")


class ContainsScorer:
    name = "contains"

    def __init__(self, case_sensitive: bool = False) -> None:
        self.case_sensitive = case_sensitive

    def score(self, case: EvalCase, output: str, latency_ms: float) -> Score:
        expected = case.expected if isinstance(case.expected, str) else str(case.expected)
        haystack = output if self.case_sensitive else output.lower()
        needle = expected if self.case_sensitive else expected.lower()
        passed = needle in haystack
        return Score(self.name, passed, 1.0 if passed else 0.0, f"needle={expected!r}")


class NumericToleranceScorer:
    name = "numeric_tolerance"

    def __init__(self, tolerance: float = 0.01) -> None:
        self.tolerance = tolerance

    def score(self, case: EvalCase, output: str, latency_ms: float) -> Score:
        expected = float(case.expected)
        match = re.search(r"-?\d+(?:\.\d+)?", output)
        if not match:
            return Score(self.name, False, 0.0, "no numeric value found")
        actual = float(match.group(0))
        delta = abs(actual - expected)
        return Score(self.name, delta <= self.tolerance, 1.0 if delta <= self.tolerance else 0.0, f"delta={delta:g}")


class LatencyBudgetScorer:
    name = "latency_budget"

    def __init__(self, max_ms: float) -> None:
        self.max_ms = max_ms

    def score(self, case: EvalCase, output: str, latency_ms: float) -> Score:
        passed = latency_ms <= self.max_ms
        return Score(self.name, passed, 1.0 if passed else 0.0, f"latency_ms={latency_ms:g}")


def _matches_type(value: Any, expected_type: str) -> bool:
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "null":
        return value is None
    return True


def validate_json_schema(value: Any, schema: Dict[str, Any], path: str = "$") -> List[str]:
    errors: List[str] = []
    expected_type = schema.get("type")
    if expected_type and not _matches_type(value, expected_type):
        return [f"{path}: expected {expected_type}"]
    if "required" in schema and isinstance(value, dict):
        for key in schema["required"]:
            if key not in value:
                errors.append(f"{path}: missing required key {key!r}")
    if "properties" in schema and isinstance(value, dict):
        for key, child_schema in schema["properties"].items():
            if key in value:
                errors.extend(validate_json_schema(value[key], child_schema, f"{path}.{key}"))
    if "items" in schema and isinstance(value, list):
        for index, item in enumerate(value):
            errors.extend(validate_json_schema(item, schema["items"], f"{path}[{index}]"))
    return errors


class JsonSchemaScorer:
    name = "json_schema"

    def score(self, case: EvalCase, output: str, latency_ms: float) -> Score:
        schema = case.expected if isinstance(case.expected, dict) else {}
        try:
            parsed = json.loads(output)
        except json.JSONDecodeError as exc:
            return Score(self.name, False, 0.0, f"invalid json: {exc.msg}")
        errors = validate_json_schema(parsed, schema)
        return Score(self.name, not errors, 1.0 if not errors else 0.0, "; ".join(errors))


@dataclass
class EvalResult:
    case_id: str
    input: str
    expected: Any
    output: str
    latency_ms: float
    trace_id: str
    cost_usd: float
    provider: str
    model: str
    scores: List[Score]
    error: Optional[str] = None

    @property
    def passed(self) -> bool:
        return self.error is None and bool(self.scores) and all(score.passed for score in self.scores)

    @property
    def score_value(self) -> float:
        if not self.scores:
            return 0.0
        return sum(score.value for score in self.scores) / len(self.scores)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "input": self.input,
            "expected": self.expected,
            "output": self.output,
            "latency_ms": self.latency_ms,
            "trace_id": self.trace_id,
            "cost_usd": self.cost_usd,
            "provider": self.provider,
            "model": self.model,
            "scores": [score.to_dict() for score in self.scores],
            "passed": self.passed,
            "score_value": self.score_value,
            "error": self.error,
        }


@dataclass
class EvalRun:
    run_id: str
    dataset_name: str
    dataset_version: str
    dataset_hash: str
    model_alias: str
    tenant_id: str
    use_case: str
    started_at: float
    finished_at: float
    results: List[EvalResult]
    audit_head: str

    def metrics(self) -> Dict[str, float]:
        total = len(self.results)
        passed = sum(1 for result in self.results if result.passed)
        latencies = [result.latency_ms for result in self.results]
        ordered = sorted(latencies)
        p95 = ordered[max(0, min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1))] if ordered else 0.0
        return {
            "total": float(total),
            "passed": float(passed),
            "failed": float(total - passed),
            "pass_rate": passed / total if total else 0.0,
            "mean_score": sum(result.score_value for result in self.results) / total if total else 0.0,
            "p95_latency_ms": p95,
            "total_cost_usd": sum(result.cost_usd for result in self.results),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "dataset_hash": self.dataset_hash,
            "model_alias": self.model_alias,
            "tenant_id": self.tenant_id,
            "use_case": self.use_case,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "audit_head": self.audit_head,
            "metrics": self.metrics(),
            "results": [result.to_dict() for result in self.results],
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "EvalRun":
        return cls(
            run_id=str(raw["run_id"]),
            dataset_name=str(raw["dataset_name"]),
            dataset_version=str(raw["dataset_version"]),
            dataset_hash=str(raw["dataset_hash"]),
            model_alias=str(raw["model_alias"]),
            tenant_id=str(raw["tenant_id"]),
            use_case=str(raw["use_case"]),
            started_at=float(raw["started_at"]),
            finished_at=float(raw["finished_at"]),
            audit_head=str(raw["audit_head"]),
            results=[
                EvalResult(
                    case_id=str(item["case_id"]),
                    input=str(item["input"]),
                    expected=item.get("expected"),
                    output=str(item["output"]),
                    latency_ms=float(item["latency_ms"]),
                    trace_id=str(item["trace_id"]),
                    cost_usd=float(item["cost_usd"]),
                    provider=str(item["provider"]),
                    model=str(item["model"]),
                    scores=[
                        Score(
                            name=str(score["name"]),
                            passed=bool(score["passed"]),
                            value=float(score["value"]),
                            detail=str(score.get("detail", "")),
                        )
                        for score in item.get("scores", [])
                    ],
                    error=item.get("error"),
                )
                for item in raw["results"]
            ],
        )


class EvalRunner:
    def __init__(
        self,
        gateway: LLMGateway,
        scorers: Sequence[Scorer],
        clock: Callable[[], float] = time.time,
        run_id_factory: Optional[Callable[[], str]] = None,
    ) -> None:
        if not scorers:
            raise ValueError("at least one scorer is required")
        self.gateway = gateway
        self.scorers = list(scorers)
        self.clock = clock
        self._run_id_factory = run_id_factory or (lambda: uuid.uuid4().hex)

    def run(
        self,
        dataset: Dataset,
        model_alias: str,
        tenant_id: str,
        use_case: str = "eval",
    ) -> EvalRun:
        started_at = self.clock()
        results: List[EvalResult] = []
        for case in dataset.cases:
            request = CompletionRequest(
                tenant_id=tenant_id,
                model=model_alias,
                messages=[ChatMessage(role="user", content=case.input)],
                use_case=use_case,
            )
            try:
                gateway_result = self.gateway.complete(request)
                scores = [
                    scorer.score(case, gateway_result.text, gateway_result.latency_ms)
                    for scorer in self.scorers
                ]
                results.append(
                    EvalResult(
                        case_id=case.id,
                        input=case.input,
                        expected=case.expected,
                        output=gateway_result.text,
                        latency_ms=gateway_result.latency_ms,
                        trace_id=gateway_result.trace_id,
                        cost_usd=gateway_result.cost_usd,
                        provider=gateway_result.provider,
                        model=gateway_result.model,
                        scores=scores,
                    )
                )
            except Exception as exc:
                results.append(
                    EvalResult(
                        case_id=case.id,
                        input=case.input,
                        expected=case.expected,
                        output="",
                        latency_ms=0.0,
                        trace_id="",
                        cost_usd=0.0,
                        provider="",
                        model="",
                        scores=[],
                        error=str(exc),
                    )
                )
        finished_at = self.clock()
        return EvalRun(
            run_id=self._run_id_factory(),
            dataset_name=dataset.name,
            dataset_version=dataset.version,
            dataset_hash=dataset.content_hash(),
            model_alias=model_alias,
            tenant_id=tenant_id,
            use_case=use_case,
            started_at=started_at,
            finished_at=finished_at,
            results=results,
            audit_head=self.gateway.audit.head_hash(),
        )


@dataclass
class ExperimentComparison:
    baseline_run_id: str
    candidate_run_id: str
    improved: List[str]
    regressed: List[str]
    tied: List[str]
    only_in_baseline: List[str]
    only_in_candidate: List[str]
    pass_rate_delta: float
    mean_score_delta: float
    cost_delta_usd: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline_run_id": self.baseline_run_id,
            "candidate_run_id": self.candidate_run_id,
            "improved": self.improved,
            "regressed": self.regressed,
            "tied": self.tied,
            "only_in_baseline": self.only_in_baseline,
            "only_in_candidate": self.only_in_candidate,
            "pass_rate_delta": self.pass_rate_delta,
            "mean_score_delta": self.mean_score_delta,
            "cost_delta_usd": self.cost_delta_usd,
        }


def compare_runs(baseline: EvalRun, candidate: EvalRun) -> ExperimentComparison:
    baseline_by_id = {result.case_id: result for result in baseline.results}
    candidate_by_id = {result.case_id: result for result in candidate.results}
    improved: List[str] = []
    regressed: List[str] = []
    tied: List[str] = []
    for case_id in sorted(set(baseline_by_id) & set(candidate_by_id)):
        before = baseline_by_id[case_id].score_value
        after = candidate_by_id[case_id].score_value
        if after > before:
            improved.append(case_id)
        elif after < before:
            regressed.append(case_id)
        else:
            tied.append(case_id)
    baseline_metrics = baseline.metrics()
    candidate_metrics = candidate.metrics()
    return ExperimentComparison(
        baseline_run_id=baseline.run_id,
        candidate_run_id=candidate.run_id,
        improved=improved,
        regressed=regressed,
        tied=tied,
        only_in_baseline=sorted(set(baseline_by_id) - set(candidate_by_id)),
        only_in_candidate=sorted(set(candidate_by_id) - set(baseline_by_id)),
        pass_rate_delta=candidate_metrics["pass_rate"] - baseline_metrics["pass_rate"],
        mean_score_delta=candidate_metrics["mean_score"] - baseline_metrics["mean_score"],
        cost_delta_usd=candidate_metrics["total_cost_usd"] - baseline_metrics["total_cost_usd"],
    )


def render_markdown(run: EvalRun) -> str:
    metrics = run.metrics()
    lines = [
        f"# Eval run {run.run_id}",
        "",
        f"- Dataset: `{run.dataset_name}` v{run.dataset_version} (`{run.dataset_hash[:12]}`)",
        f"- Model alias: `{run.model_alias}`",
        f"- Pass rate: **{metrics['pass_rate']:.1%}** ({int(metrics['passed'])}/{int(metrics['total'])})",
        f"- Mean score: {metrics['mean_score']:.3f}",
        f"- p95 latency: {metrics['p95_latency_ms']:.2f} ms",
        f"- Total cost: ${metrics['total_cost_usd']:.6f}",
        f"- Audit head: `{run.audit_head[:16]}`",
        "",
        "| Case | Passed | Score | Latency (ms) | Trace |",
        "|---|---:|---:|---:|---|",
    ]
    for result in run.results:
        lines.append(
            f"| `{result.case_id}` | {'yes' if result.passed else 'no'} | "
            f"{result.score_value:.2f} | {result.latency_ms:.2f} | `{result.trace_id[:12]}` |"
        )
    return "\n".join(lines) + "\n"
