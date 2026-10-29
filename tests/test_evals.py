import pytest

from llm_control_plane.evals import (
    ContainsScorer,
    Dataset,
    EvalCase,
    EvalResult,
    EvalRun,
    EvalRunner,
    ExactMatchScorer,
    JsonSchemaScorer,
    LatencyBudgetScorer,
    NumericToleranceScorer,
    Score,
    compare_runs,
    render_markdown,
    validate_json_schema,
)
from llm_control_plane.gateway import GatewayConfig, LLMGateway
from llm_control_plane.providers import FakeProvider, provider_map
from llm_control_plane.routing import RoutingTable


def make_gateway(provider):
    routing = RoutingTable.from_dict({"smart": [{"provider": provider.name, "model": "fake-large"}]})
    return LLMGateway(provider_map(provider), routing, config=GatewayConfig(max_attempts_per_target=1))


def test_dataset_hash_is_stable_and_version_sensitive():
    dataset = Dataset(name="d", version="v1", cases=[EvalCase(id="a", input="x", expected="y")])
    assert dataset.content_hash() == dataset.content_hash()
    changed = Dataset(name="d", version="v2", cases=[EvalCase(id="a", input="x", expected="y")])
    assert changed.content_hash() != dataset.content_hash()


def test_dataset_rejects_duplicate_ids():
    with pytest.raises(ValueError):
        Dataset(
            name="d",
            version="v1",
            cases=[
                EvalCase(id="a", input="x", expected="y"),
                EvalCase(id="a", input="z", expected="w"),
            ],
        )


def test_exact_match_normalizes_whitespace_and_case():
    scorer = ExactMatchScorer()
    case = EvalCase(id="a", input="x", expected="Paris")
    assert scorer.score(case, "  paris  ", 1.0).passed is True
    assert scorer.score(case, "London", 1.0).passed is False


def test_contains_scorer():
    case = EvalCase(id="a", input="x", expected="evidence")
    assert ContainsScorer().score(case, "The EVIDENCE is attached", 1.0).passed is True
    assert ContainsScorer().score(case, "nothing", 1.0).passed is False


def test_numeric_tolerance():
    case = EvalCase(id="a", input="x", expected=42.0)
    assert NumericToleranceScorer(tolerance=0.1).score(case, "42.05", 1.0).passed is True
    assert NumericToleranceScorer(tolerance=0.1).score(case, "43", 1.0).passed is False
    assert NumericToleranceScorer().score(case, "no number", 1.0).passed is False


def test_latency_budget():
    case = EvalCase(id="a", input="x", expected="y")
    assert LatencyBudgetScorer(max_ms=100).score(case, "y", 99).passed is True
    assert LatencyBudgetScorer(max_ms=100).score(case, "y", 101).passed is False


def test_json_schema_scorer():
    schema = {
        "type": "object",
        "required": ["status"],
        "properties": {"status": {"type": "string"}},
    }
    case = EvalCase(id="a", input="x", expected=schema)
    assert JsonSchemaScorer().score(case, '{"status":"ok"}', 1.0).passed is True
    assert JsonSchemaScorer().score(case, '{"status":1}', 1.0).passed is False
    assert JsonSchemaScorer().score(case, "not json", 1.0).passed is False


def test_validate_json_schema_nested():
    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {"type": "object", "required": ["id"], "properties": {"id": {"type": "integer"}}},
            }
        },
    }
    errors = validate_json_schema({"items": [{"id": "x"}]}, schema)
    assert errors
    assert validate_json_schema({"items": [{"id": 1}]}, schema) == []


def test_eval_runner_records_trace_and_metrics():
    provider = FakeProvider("fake-primary", responses=["Paris", "4"])
    gateway = make_gateway(provider)
    dataset = Dataset(
        name="d",
        version="v1",
        cases=[
            EvalCase(id="capital", input="capital?", expected="Paris"),
            EvalCase(id="math", input="2+2?", expected="4"),
        ],
    )
    run = EvalRunner(gateway, [ContainsScorer()]).run(dataset, "smart", "tenant-a")
    metrics = run.metrics()
    assert metrics["total"] == 2
    assert metrics["passed"] == 2
    assert metrics["pass_rate"] == 1.0
    assert all(result.trace_id for result in run.results)
    assert run.audit_head == gateway.audit.head_hash()


def test_eval_runner_captures_provider_error():
    provider = FakeProvider("fake-primary", failures=[RuntimeError("boom")])
    gateway = make_gateway(provider)
    dataset = Dataset(name="d", version="v1", cases=[EvalCase(id="a", input="x", expected="y")])
    run = EvalRunner(gateway, [ContainsScorer()]).run(dataset, "smart", "tenant-a")
    assert run.results[0].error is not None
    assert run.metrics()["passed"] == 0


def test_compare_runs_detects_improvement_and_regression():
    def result(case_id, value):
        return EvalResult(
            case_id=case_id,
            input="x",
            expected="y",
            output="y",
            latency_ms=1.0,
            trace_id=f"trace-{case_id}",
            cost_usd=0.0,
            provider="p",
            model="m",
            scores=[Score("contains", value >= 1.0, value)],
        )

    baseline = EvalRun(
        "b", "d", "v1", "hash", "smart", "t", "eval", 0.0, 1.0,
        [result("a", 0.0), result("b", 1.0), result("c", 1.0)], "head",
    )
    candidate = EvalRun(
        "c", "d", "v1", "hash", "smart", "t", "eval", 0.0, 1.0,
        [result("a", 1.0), result("b", 0.0), result("c", 1.0)], "head",
    )
    comparison = compare_runs(baseline, candidate)
    assert comparison.improved == ["a"]
    assert comparison.regressed == ["b"]
    assert comparison.tied == ["c"]
    assert comparison.pass_rate_delta == 0.0


def test_eval_run_round_trip():
    provider = FakeProvider("fake-primary", responses=["Paris"])
    gateway = make_gateway(provider)
    dataset = Dataset(name="d", version="v1", cases=[EvalCase(id="a", input="x", expected="Paris")])
    run = EvalRunner(gateway, [ContainsScorer()]).run(dataset, "smart", "tenant-a")
    restored = EvalRun.from_dict(run.to_dict())
    assert restored.to_dict() == run.to_dict()
    assert "Eval run" in render_markdown(run)
