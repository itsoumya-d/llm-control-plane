from llm_control_plane.bench import bench_audit, bench_evals, bench_gateway


def test_benchmarks_run():
    gateway = bench_gateway(iterations=20)
    assert gateway["iterations"] == 20.0
    assert gateway["p95_ms"] >= 0.0
    assert gateway["audit_events"] > 0.0

    audit = bench_audit(iterations=10)
    assert audit["valid"] == 1.0
    assert len(audit["head_hash"]) == 64

    evals = bench_evals(iterations=5, runs=1)
    assert evals["cases"] == 5.0
    assert evals["cases_per_second"] > 0.0
