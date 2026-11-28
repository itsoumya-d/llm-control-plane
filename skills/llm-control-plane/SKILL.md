# LLM Control Plane Agent Skill

Use this skill when an autonomous coding agent (such as Claude Code, Codex, or Cursor) needs to inspect gateway health, test model fallback routing, run regression benchmark evaluations, score gateway envelopes against quality rubrics, or mechanically repair malformed response envelopes. It provides verified CLI operations with built-in audit traceability, deterministic execution without external network calls, and human-in-the-loop safety gates requiring explicit confirmation before mutating any local files.

## Procedure

1. Verify Gateway Health and Cryptographic Audit Chain
   Run the system health diagnostic to verify that provider configurations and the tamper-evident audit ledger are valid:
   ```bash
   python -m llm_control_plane.cli health
   ```
   A good result outputs a JSON object with `"status": "healthy"`, `"audit_chain_valid": true`, and an exit code of `0`.

2. Test Fallback Model Routing Simulation
   Simulate a client completion request across configured primary and fallback model providers:
   ```bash
   python -m llm_control_plane.cli route --prompt "What is 2+2?" --model-alias smart
   ```
   A good result returns a formatted completion envelope containing `"text"`, `"provider"`, `"model"`, `"trace_id"`, and non-negative usage and cost metrics.

3. Run Deterministic Benchmark Evaluation Suite
   Execute the offline evaluation harness across standard regression cases to measure precision and response drift:
   ```bash
   python -m llm_control_plane.cli eval
   ```
   A good result prints a markdown scorecard table showing test case results, scores, latency, and a verified audit chain signature.

4. Score Output Envelope Against Declarative Quality Rubric
   Verify whether a generated response payload satisfies the 5-point weighted rubric (threshold 80/100):
   ```bash
   python -m llm_control_plane.cli verify --input <envelope_file.json>
   ```
   A good result prints a JSON breakdown with `"score"`, `"threshold"`, `"passed": true`, and details for each check.

5. Execute Automated Verify-and-Repair Remediation Loop
   Remediate mechanical formatting, default missing fields, or range violations up to 3 rounds. To write changes to disk, pass the `--apply` approval flag:
   ```bash
   python -m llm_control_plane.cli repair --input <broken.json> --output <repaired.json> --ledger <ledger.jsonl> --apply
   ```
   A good result outputs `"applied": true`, `"passed": true`, `"final_score" >= 80`, lists `repaired_fields`, and appends an immutable JSON record to the audit ledger.
