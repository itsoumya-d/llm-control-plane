import pytest

from llm_control_plane.engineering_productivity_engine import (
    DeploymentEvent,
    DeploymentStatus,
    DORAPerformanceTier,
    EngineeringProductivityAnalyzer,
    PullRequestCycle,
)


def test_engineering_productivity_dora_calculation_normal():
    """1. Normal case: Computes valid DORA metrics and quality friction score."""
    analyzer = EngineeringProductivityAnalyzer(observation_window_days=10.0)

    # 10 deployments: 9 success, 1 failure (CFR = 10%)
    for i in range(9):
        analyzer.record_deployment(
            DeploymentEvent(
                f"dep-{i}",
                "auth-service",
                lead_time_minutes=30.0,
                status=DeploymentStatus.SUCCESS,
            )
        )
    analyzer.record_deployment(
        DeploymentEvent(
            "dep-fail",
            "auth-service",
            lead_time_minutes=60.0,
            status=DeploymentStatus.FAILURE,
            restoration_time_minutes=45.0,
        )
    )

    report = analyzer.compute_dora_metrics("auth-service")
    assert report.service_name == "auth-service"
    assert report.deployment_frequency_per_day == 1.0
    assert report.change_failure_rate_percent == 10.0
    assert report.mean_time_to_restore_minutes == 45.0
    assert report.classification in (DORAPerformanceTier.ELITE, DORAPerformanceTier.HIGH)

    analyzer.record_pr_cycle(
        PullRequestCycle(
            "pr-1",
            "auth-service",
            review_latency_hours=4.0,
            ci_duration_minutes=8.0,
            test_coverage_percent=88.5,
        )
    )
    health = analyzer.evaluate_quality_health("auth-service")
    assert health.average_review_latency_hours == 4.0
    assert health.average_test_coverage_percent == 88.5
    assert health.friction_score < 0.20


def test_engineering_productivity_empty_input_validation():
    """2. Empty input: Rejects empty IDs, service names, and non-positive observation window."""
    with pytest.raises(ValueError, match="observation_window_days must be positive"):
        EngineeringProductivityAnalyzer(observation_window_days=0)

    analyzer = EngineeringProductivityAnalyzer()
    with pytest.raises(ValueError, match="deployment_id cannot be empty"):
        analyzer.record_deployment(DeploymentEvent("", "svc", 10.0, DeploymentStatus.SUCCESS))

    with pytest.raises(ValueError, match="service_name cannot be empty"):
        analyzer.record_deployment(DeploymentEvent("d1", "   ", 10.0, DeploymentStatus.SUCCESS))

    with pytest.raises(ValueError, match="pr_id cannot be empty"):
        analyzer.record_pr_cycle(PullRequestCycle("", "svc", 2.0, 5.0, 80.0))

    with pytest.raises(ValueError, match="service_name cannot be empty"):
        analyzer.compute_dora_metrics("")


def test_engineering_productivity_malformed_input():
    """3. Malformed input: Rejects negative durations, out-of-range coverage, and negative counts."""
    analyzer = EngineeringProductivityAnalyzer()

    with pytest.raises(ValueError, match="lead_time_minutes must be non-negative"):
        analyzer.record_deployment(DeploymentEvent("d1", "svc", -5.0, DeploymentStatus.SUCCESS))

    with pytest.raises(ValueError, match="restoration_time_minutes cannot be negative"):
        analyzer.record_deployment(
            DeploymentEvent(
                "d2",
                "svc",
                10.0,
                DeploymentStatus.FAILURE,
                restoration_time_minutes=-1.0,
            )
        )

    with pytest.raises(ValueError, match="test_coverage_percent must be between 0 and 100"):
        analyzer.record_pr_cycle(PullRequestCycle("p1", "svc", 2.0, 5.0, 105.0))

    with pytest.raises(ValueError, match="flaky_test_count cannot be negative"):
        analyzer.record_pr_cycle(PullRequestCycle("p2", "svc", 2.0, 5.0, 80.0, flaky_test_count=-2))


def test_engineering_productivity_boundary_values():
    """4. Boundary values: 0% and 100% coverage, 0 failures, and exact threshold frequencies."""
    analyzer = EngineeringProductivityAnalyzer(observation_window_days=30.0)

    # 100% coverage and 0% coverage boundaries
    analyzer.record_pr_cycle(PullRequestCycle("p-high", "core-svc", 1.0, 2.0, 100.0, flaky_test_count=0))
    analyzer.record_pr_cycle(PullRequestCycle("p-zero", "core-svc", 1.0, 2.0, 0.0, flaky_test_count=0))

    health = analyzer.evaluate_quality_health("core-svc")
    assert health.average_test_coverage_percent == 50.0

    # 0 failures = 0% CFR, 0 MTTR
    analyzer.record_deployment(
        DeploymentEvent("d-clean", "core-svc", lead_time_minutes=15.0, status=DeploymentStatus.SUCCESS)
    )
    report = analyzer.compute_dora_metrics("core-svc")
    assert report.change_failure_rate_percent == 0.0
    assert report.mean_time_to_restore_minutes == 0.0


def test_engineering_productivity_dependency_failure():
    """5. Dependency failure / unmonitored service: Gracefully reports baseline LOW metrics."""
    analyzer = EngineeringProductivityAnalyzer()
    report = analyzer.compute_dora_metrics("unregistered-service")
    assert report.deployment_frequency_per_day == 0.0
    assert report.classification == DORAPerformanceTier.LOW

    health = analyzer.evaluate_quality_health("unregistered-service")
    assert health.friction_score == 0.0
    assert "No recorded pull request telemetry" in health.bottlenecks


def test_engineering_productivity_metrics_invariance():
    """6. Property that must always hold (invariance):

    - CFR is strictly bounded within [0.0, 100.0].
    - Friction score is strictly bounded within [0.0, 1.0].
    """
    analyzer = EngineeringProductivityAnalyzer(observation_window_days=7.0)

    # Generate extreme friction PRs
    for i in range(5):
        analyzer.record_pr_cycle(
            PullRequestCycle(
                f"pr-extreme-{i}",
                "stressed-svc",
                review_latency_hours=100.0,
                ci_duration_minutes=90.0,
                test_coverage_percent=10.0,
                flaky_test_count=10,
            )
        )
    health = analyzer.evaluate_quality_health("stressed-svc")
    assert 0.0 <= health.friction_score <= 1.0
    assert health.friction_score == 1.0  # Clipped cleanly at 1.0 ceiling
    assert len(health.bottlenecks) >= 3

    # All failures = 100% CFR
    for i in range(3):
        analyzer.record_deployment(
            DeploymentEvent(
                f"d-fail-{i}",
                "stressed-svc",
                lead_time_minutes=120.0,
                status=DeploymentStatus.FAILURE,
                restoration_time_minutes=60.0,
            )
        )
    dora = analyzer.compute_dora_metrics("stressed-svc")
    assert 0.0 <= dora.change_failure_rate_percent <= 100.0
    assert dora.change_failure_rate_percent == 100.0
