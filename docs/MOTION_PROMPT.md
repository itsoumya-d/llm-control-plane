# Motion prompt — LLM Control Plane

Use this prompt for a 12-second, square (1080×1080) build-in-public motion piece with dotLottie, Remotion, or a HeyGen avatar overlay.

## Scene 1 — the problem (0.0–2.5s)

Dark graphite background, faint grid. A request packet travels from a label `Product service` toward a glowing `LLM Gateway` node. Three provider nodes sit behind it: `Provider A`, `Provider B`, `Provider C`. One provider flickers red. Counter label: `1 provider = 1 point of failure`.

## Scene 2 — the control plane (2.5–6.0s)

The gateway expands into five labelled rails: `Route`, `Retry`, `Rate limit`, `Redact`, `Attribute cost`. The packet hits a 429 on Provider A, forks to Provider B, and succeeds. A token bucket drains and refills. A cost meter ticks upward. PII tokens turn into grey `[REDACTED]` blocks before reaching the provider.

## Scene 3 — the audit chain (6.0–9.0s)

Audit events stack vertically. Each event links to the previous one with a glowing SHA-256 hash. One event turns red and shakes; the chain breaks; then it snaps back to green. Label: `tamper-evident, not just logged`.

## Scene 4 — the eval run (9.0–11.0s)

A dataset card fans into cases. A run card appears with `pass rate 100%`, `p95 0.50 ms`, `cost $0.000015`, and a trace ID. Lines connect the run card back to the audit head hash.

## Scene 5 — end card (11.0–12.0s)

Centered text: `llm-control-plane` / `routing · evals · audit` / `github.com/itsoumya-d/llm-control-plane`. Subtle particle settle. No logo, no stock footage, no fake metrics.

## Motion rules

- 60 fps, ease-in-out, 180–240 ms transitions.
- Every number on screen must match the repository README.
- No more than 12 words per card.
- Minimum on-screen text size: 24 px at 1080×1080.
- Respect Reduce Motion: provide a static end card.

## Caption

> Most AI teams add a second model provider and discover nobody owns routing, retries, cost attribution, evaluation, or the audit trail. I built llm-control-plane as a reference implementation: ordered failover, per-tenant rate limits, PII redaction, a SHA-256 hash-chained audit log, versioned evals and trace linkage. 45 tests, 93% coverage, green CI.
>
> Repo: https://github.com/itsoumya-d/llm-control-plane
