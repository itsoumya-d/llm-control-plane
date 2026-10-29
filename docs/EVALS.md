# Evaluations

## Dataset versioning

A dataset is a named, versioned list of cases. `Dataset.content_hash()` canonicalises the name, version, case inputs, expected values, and metadata. The hash is stored on every `EvalRun`, so a result can be reproduced from the exact dataset content even if the version label is reused.

Duplicate case IDs are rejected at construction time.

## Case shape

```python
EvalCase(
    id="refund-policy",
    input="Can I get a refund after 30 days?",
    expected="refund",
    metadata={"team": "support", "risk": "medium"},
)
```

`expected` is intentionally untyped: a string for exact/contains, a float for numeric tolerance, or a JSON-schema-like dict for structural checks.

## Scorers

| Scorer | Expected type | Pass condition |
|---|---|---|
| `ExactMatchScorer` | `str` | Normalised output equals normalised expected |
| `ContainsScorer` | `str` | Expected substring is present |
| `JsonSchemaScorer` | `dict` | Parsed JSON satisfies type/required/properties/items |
| `NumericToleranceScorer` | `float` | Parsed number is within tolerance |
| `LatencyBudgetScorer` | any | Observed latency is at or below the budget |

A sample passes only when every configured scorer passes. `EvalResult.score_value` is the mean of individual scorer values (0.0 or 1.0 for the built-in scorers).

## Trace linkage

Every `EvalResult` carries:

- the gateway `trace_id`
- provider and model that served the call
- latency and cost
- the individual scorer results
- the run's `audit_head`

This lets an evaluator move from a failed case to the exact gateway audit events that produced it.

## Comparing runs

```python
from llm_control_plane import compare_runs

comparison = compare_runs(baseline_run, candidate_run)
print(comparison.improved)
print(comparison.regressed)
print(comparison.pass_rate_delta)
```

Cases are matched by `case_id`. The comparison reports improved, regressed, tied, and cases present in only one run.

## Failure handling

If a gateway call fails after all provider targets are exhausted, the eval result is recorded with `error` set, empty output, and `passed=False`. The run still completes, so one provider outage does not hide the rest of the dataset.

## Persistence

`EvalRun.to_dict()` and `EvalRun.from_dict()` round-trip the full run, including metrics, audit head, and per-case scores. Store the JSON in Postgres or object storage and index by:

- `dataset_hash`
- `model_alias`
- `started_at`
- `audit_head`

## CI usage

A minimal regression gate:

```bash
python -m llm_control_plane.demo
pytest -q
```

For a real project, run a fixed dataset against a candidate model alias, compare it to the baseline run, and fail the build when `comparison.regressed` is non-empty or `pass_rate_delta` is below the agreed threshold.
