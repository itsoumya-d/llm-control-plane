"""A deterministic end-to-end demo with two providers and a small eval set."""

from __future__ import annotations

from .evals import ContainsScorer, Dataset, EvalCase, EvalRunner, render_markdown
from .gateway import LLMGateway
from .providers import RuleProvider, provider_map
from .routing import RoutingTable

RULES = {
    "2+2": "4",
    "capital of france": "Paris",
    "status ok": '{"status":"ok"}',
    "number 42": "42",
}


def build_demo_gateway() -> LLMGateway:
    primary = RuleProvider("fake-primary", RULES, default="primary-unknown")
    fallback = RuleProvider("fake-fallback", RULES, default="fallback-unknown")
    routing = RoutingTable.from_dict(
        {
            "smart": [
                {"provider": "fake-primary", "model": "fake-large"},
                {"provider": "fake-fallback", "model": "fake-small"},
            ]
        }
    )
    return LLMGateway(provider_map(primary, fallback), routing)


def sample_dataset() -> Dataset:
    return Dataset(
        name="control-plane-smoke",
        version="2026-10-30",
        cases=[
            EvalCase(id="math", input="What is 2+2?", expected="4"),
            EvalCase(id="capital", input="What is the capital of France?", expected="Paris"),
            EvalCase(id="json", input="Return status ok as JSON", expected='{"status":"ok"}'),
            EvalCase(id="number", input="Return the number 42", expected="42"),
        ],
    )


def main() -> int:
    gateway = build_demo_gateway()
    runner = EvalRunner(gateway, [ContainsScorer()])
    run = runner.run(sample_dataset(), model_alias="smart", tenant_id="demo", use_case="smoke")
    print(render_markdown(run))
    print(f"audit_chain_valid={gateway.audit.verify()}")
    print(f"audit_events={len(gateway.audit)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
