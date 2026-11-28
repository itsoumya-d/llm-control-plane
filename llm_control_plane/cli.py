"""CLI entrypoint for LLM Control Plane operations and agent skill automation.

Exposes subcommands for health auditing, model routing simulation, evaluation benchmarking,
rubric scoring verification, and mechanical verify-and-repair remediation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .demo import build_demo_gateway, sample_dataset
from .evals import ContainsScorer, EvalRunner, render_markdown
from .verify_repair import repair_gateway_output, verify_gateway_output


def cmd_health() -> int:
    """Verifies gateway provider configurations and cryptographic audit chain."""
    gateway = build_demo_gateway()
    audit_valid = gateway.audit.verify()
    providers = sorted(list(gateway.providers.keys()))

    payload = {
        "status": "healthy" if audit_valid else "degraded",
        "providers": providers,
        "audit_chain_valid": audit_valid,
        "audit_events_count": len(gateway.audit.events()),
    }
    print(json.dumps(payload, indent=2))
    return 0 if audit_valid else 1


def cmd_route(prompt: str, model_alias: str = "smart") -> int:
    """Simulates prompt dispatch through configured fallback routing table."""
    gateway = build_demo_gateway()
    try:
        from .types import ChatMessage, CompletionRequest

        req = CompletionRequest(
            tenant_id="agent-cli",
            model=model_alias,
            use_case="cli",
            messages=[ChatMessage(role="user", content=prompt)],
        )
        res = gateway.complete(req)
        out = {
            "text": res.text,
            "provider": res.provider,
            "model": res.model,
            "trace_id": res.trace_id,
            "usage": {
                "prompt_tokens": res.usage.prompt_tokens,
                "completion_tokens": res.usage.completion_tokens,
                "total_tokens": res.usage.total_tokens,
            },
            "cost_usd": res.cost_usd,
            "latency_ms": res.latency_ms,
            "audit_seq": res.audit_seq,
        }
        print(json.dumps(out, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, indent=2), file=sys.stderr)
        return 1


def cmd_eval() -> int:
    """Executes deterministic evaluation benchmark and prints markdown scorecard."""
    gateway = build_demo_gateway()
    runner = EvalRunner(gateway, [ContainsScorer()])
    run = runner.run(sample_dataset(), model_alias="smart", tenant_id="agent-eval", use_case="benchmark")
    print(render_markdown(run))
    return 0


def cmd_verify(input_path: str) -> int:
    """Scores gateway output JSON against rubric.json (0-100 threshold 80)."""
    p = Path(input_path)
    if not p.exists():
        print(json.dumps({"error": f"Input file not found: {input_path}"}), file=sys.stderr)
        return 1
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    score = verify_gateway_output(data)
    out = {
        "score": score.score,
        "threshold": score.threshold,
        "passed": score.passed,
        "checks": [
            {
                "name": c.name,
                "weight": c.weight,
                "passed": c.passed,
                "details": c.details,
                "mechanical": c.mechanical,
            }
            for c in score.checks
        ],
    }
    print(json.dumps(out, indent=2))
    return 0 if score.passed else 2


def cmd_repair(
    input_path: str,
    output_path: str | None = None,
    ledger_path: str | None = None,
    apply: bool = False,
) -> int:
    """Executes verify-and-repair loop up to 3 rounds.

    Requires --apply flag to write modified output to disk.
    """
    p = Path(input_path)
    if not p.exists():
        print(json.dumps({"error": f"Input file not found: {input_path}"}), file=sys.stderr)
        return 1
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not apply and output_path:
        print(
            json.dumps({
                "status": "dry_run",
                "message": "File modifications require explicit --apply approval flag. Dry run executed.",
            }),
            file=sys.stderr,
        )

    res = repair_gateway_output(data, ledger_path=ledger_path if apply else None, max_rounds=3)

    if apply and output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(res.repaired_output, f, indent=2)

    summary = {
        "applied": bool(apply and output_path),
        "passed": res.passed,
        "final_score": res.final_score,
        "rounds": res.rounds,
        "repaired_fields": res.repaired_fields,
        "unresolved_issues": res.unresolved_issues,
    }
    print(json.dumps(summary, indent=2))
    return 0 if res.passed else 2


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LLM Control Plane Agent Automation CLI")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # health
    subparsers.add_parser("health", help="Verify gateway health and audit chain integrity")

    # route
    route_parser = subparsers.add_parser("route", help="Simulate gateway completion dispatch")
    route_parser.add_argument("--prompt", required=True, help="Input prompt text")
    route_parser.add_argument("--model-alias", default="smart", help="Routing alias")

    # eval
    subparsers.add_parser("eval", help="Execute deterministic benchmark evaluation suite")

    # verify
    verify_parser = subparsers.add_parser("verify", help="Score gateway envelope against rubric")
    verify_parser.add_argument("--input", required=True, help="Path to input JSON envelope")

    # repair
    repair_parser = subparsers.add_parser("repair", help="Run mechanical verify-and-repair loop")
    repair_parser.add_argument("--input", required=True, help="Path to raw/defective input JSON")
    repair_parser.add_argument("--output", help="Path to write repaired JSON envelope")
    repair_parser.add_argument("--ledger", help="Path to JSONL audit ledger")
    repair_parser.add_argument("--apply", action="store_true", help="Explicit confirmation to write files")

    args = parser.parse_args(argv)

    if args.subcommand == "health":
        return cmd_health()
    if args.subcommand == "route":
        return cmd_route(args.prompt, args.model_alias)
    if args.subcommand == "eval":
        return cmd_eval()
    if args.subcommand == "verify":
        return cmd_verify(args.input)
    if args.subcommand == "repair":
        return cmd_repair(args.input, args.output, args.ledger, args.apply)

    return 0


if __name__ == "__main__":
    sys.exit(main())
