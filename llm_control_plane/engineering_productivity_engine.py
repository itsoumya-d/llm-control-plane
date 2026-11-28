"""Engineering Productivity & DORA Metrics Analytics Engine.

Monitors software development lifecycle efficiency, DORA performance indicators,
test coverage tracking, and delivery bottleneck identification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class DeploymentStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"


class DORAPerformanceTier(str, Enum):
    ELITE = "ELITE"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass
class DeploymentEvent:
    deployment_id: str
    service_name: str
    lead_time_minutes: float
    status: DeploymentStatus
    restoration_time_minutes: Optional[float] = None


@dataclass
class PullRequestCycle:
    pr_id: str
    service_name: str
    review_latency_hours: float
    ci_duration_minutes: float
    test_coverage_percent: float
    flaky_test_count: int = 0


@dataclass
class DORAMetricsReport:
    service_name: str
    deployment_frequency_per_day: float
    mean_lead_time_hours: float
    change_failure_rate_percent: float
    mean_time_to_restore_minutes: float
    classification: DORAPerformanceTier


@dataclass
class QualityHealthSummary:
    service_name: str
    average_review_latency_hours: float
    mean_ci_duration_minutes: float
    average_test_coverage_percent: float
    total_flaky_test_incidents: int
    friction_score: float  # 0.0 (frictionless) to 1.0 (severe friction)
    bottlenecks: List[str] = field(default_factory=list)


class EngineeringProductivityAnalyzer:
    """Analyzes DORA metrics, test coverage, and SDLC friction signals across services."""

    def __init__(self, observation_window_days: float = 30.0):
        if observation_window_days <= 0:
            raise ValueError(f"observation_window_days must be positive, got {observation_window_days}")
        self.window_days = observation_window_days
        self.deployments: List[DeploymentEvent] = []
        self.pr_cycles: List[PullRequestCycle] = []

    def record_deployment(self, event: DeploymentEvent) -> None:
        if not event.deployment_id or not event.deployment_id.strip():
            raise ValueError("deployment_id cannot be empty")
        if not event.service_name or not event.service_name.strip():
            raise ValueError("service_name cannot be empty")
        if event.lead_time_minutes < 0:
            raise ValueError(f"lead_time_minutes must be non-negative, got {event.lead_time_minutes}")
        if (
            event.status == DeploymentStatus.FAILURE
            and event.restoration_time_minutes is not None
            and event.restoration_time_minutes < 0
        ):
            raise ValueError(f"restoration_time_minutes cannot be negative, got {event.restoration_time_minutes}")
        self.deployments.append(event)

    def record_pr_cycle(self, cycle: PullRequestCycle) -> None:
        if not cycle.pr_id or not cycle.pr_id.strip():
            raise ValueError("pr_id cannot be empty")
        if not cycle.service_name or not cycle.service_name.strip():
            raise ValueError("service_name cannot be empty")
        if cycle.review_latency_hours < 0 or cycle.ci_duration_minutes < 0:
            raise ValueError("review_latency_hours and ci_duration_minutes must be non-negative")
        if not (0.0 <= cycle.test_coverage_percent <= 100.0):
            raise ValueError(f"test_coverage_percent must be between 0 and 100, got {cycle.test_coverage_percent}")
        if cycle.flaky_test_count < 0:
            raise ValueError(f"flaky_test_count cannot be negative, got {cycle.flaky_test_count}")
        self.pr_cycles.append(cycle)

    def compute_dora_metrics(self, service_name: str) -> DORAMetricsReport:
        if not service_name or not service_name.strip():
            raise ValueError("service_name cannot be empty")
        svc_deploys = [d for d in self.deployments if d.service_name == service_name]
        if not svc_deploys:
            return DORAMetricsReport(
                service_name=service_name,
                deployment_frequency_per_day=0.0,
                mean_lead_time_hours=0.0,
                change_failure_rate_percent=0.0,
                mean_time_to_restore_minutes=0.0,
                classification=DORAPerformanceTier.LOW,
            )

        total_count = len(svc_deploys)
        fail_count = sum(1 for d in svc_deploys if d.status == DeploymentStatus.FAILURE)
        cfr = (fail_count / total_count) * 100.0

        freq_per_day = total_count / self.window_days
        mean_lead_hours = sum(d.lead_time_minutes / 60.0 for d in svc_deploys) / total_count

        recovery_times = [
            d.restoration_time_minutes
            for d in svc_deploys
            if d.status == DeploymentStatus.FAILURE and d.restoration_time_minutes is not None
        ]
        mttr = (sum(recovery_times) / len(recovery_times)) if recovery_times else 0.0

        # Classify based on DORA benchmarks
        if freq_per_day >= 1.0 and mean_lead_hours <= 24.0 and cfr <= 5.0 and mttr <= 60.0:
            cls = DORAPerformanceTier.ELITE
        elif freq_per_day >= 0.2 and mean_lead_hours <= 168.0 and cfr <= 15.0:
            cls = DORAPerformanceTier.HIGH
        elif freq_per_day >= 0.03 and cfr <= 30.0:
            cls = DORAPerformanceTier.MEDIUM
        else:
            cls = DORAPerformanceTier.LOW

        return DORAMetricsReport(
            service_name=service_name,
            deployment_frequency_per_day=round(freq_per_day, 4),
            mean_lead_time_hours=round(mean_lead_hours, 2),
            change_failure_rate_percent=round(cfr, 2),
            mean_time_to_restore_minutes=round(mttr, 2),
            classification=cls,
        )

    def evaluate_quality_health(self, service_name: str) -> QualityHealthSummary:
        if not service_name or not service_name.strip():
            raise ValueError("service_name cannot be empty")
        svc_prs = [p for p in self.pr_cycles if p.service_name == service_name]
        if not svc_prs:
            return QualityHealthSummary(
                service_name=service_name,
                average_review_latency_hours=0.0,
                mean_ci_duration_minutes=0.0,
                average_test_coverage_percent=0.0,
                total_flaky_test_incidents=0,
                friction_score=0.0,
                bottlenecks=["No recorded pull request telemetry"],
            )

        n = len(svc_prs)
        avg_review = sum(p.review_latency_hours for p in svc_prs) / n
        avg_ci = sum(p.ci_duration_minutes for p in svc_prs) / n
        avg_cov = sum(p.test_coverage_percent for p in svc_prs) / n
        flaky_total = sum(p.flaky_test_count for p in svc_prs)

        bottlenecks = []
        friction_points = 0.0

        if avg_review > 24.0:
            bottlenecks.append(f"High review latency: {avg_review:.1f}h exceeds 24h threshold")
            friction_points += 0.35
        elif avg_review > 8.0:
            friction_points += 0.15

        if avg_ci > 20.0:
            bottlenecks.append(f"Slow CI pipeline: {avg_ci:.1f}m exceeds 20m budget")
            friction_points += 0.30
        elif avg_ci > 10.0:
            friction_points += 0.10

        if avg_cov < 75.0:
            bottlenecks.append(f"Sub-target test coverage: {avg_cov:.1f}% below 75% bar")
            friction_points += 0.20

        if flaky_total > 0:
            bottlenecks.append(f"Flaky test detected: {flaky_total} test failures caused by non-deterministic suites")
            friction_points += min(0.35, flaky_total * 0.10)

        friction_score = min(1.0, round(friction_points, 2))

        return QualityHealthSummary(
            service_name=service_name,
            average_review_latency_hours=round(avg_review, 2),
            mean_ci_duration_minutes=round(avg_ci, 2),
            average_test_coverage_percent=round(avg_cov, 2),
            total_flaky_test_incidents=flaky_total,
            friction_score=friction_score,
            bottlenecks=bottlenecks,
        )
