"""Unit tests for Archive Health & Package Forward-Verification Engine."""

import pytest

from llm_control_plane.archive_health import (
    ArchiveHealthPolicy,
    ArchiveStatus,
    PackageArtifact,
    evaluate_archive_health,
)


def test_archive_health_normal_valid_progression() -> None:
    """Normal case: multiple packages advancing versions monotonically with valid dependencies."""
    artifacts = [
        PackageArtifact(name="libcore", version="1.0.0", sha256="abc1"),
        PackageArtifact(name="libcore", version="1.1.0", sha256="abc2"),
        PackageArtifact(name="app-server", version="2.0.0", dependencies=["libcore>=1.0.0"], sha256="def1"),
    ]
    report = evaluate_archive_health(artifacts)

    assert report.status == ArchiveStatus.HEALTHY
    assert report.overall_score == 100.0
    assert report.total_artifacts == 3
    assert report.healthy_count == 3
    assert report.degraded_count == 0
    assert len(report.broken_dependencies) == 0
    assert len(report.version_regressions) == 0
    assert len(report.manifest_digest) == 64


def test_archive_health_empty_manifest_rejection() -> None:
    """Empty input: evaluating an empty artifact sequence raises ValueError."""
    with pytest.raises(ValueError, match="archive cannot be empty"):
        evaluate_archive_health([])


def test_archive_health_malformed_spec_rejection() -> None:
    """Malformed input: invalid semantic version strings are rejected."""
    malformed_artifacts = [
        PackageArtifact(name="pkg-a", version="not.a.version"),
    ]
    with pytest.raises(ValueError, match="malformed artifact spec: invalid semantic version"):
        evaluate_archive_health(malformed_artifacts)

    empty_name_artifacts = [
        PackageArtifact(name="   ", version="1.0.0"),
    ]
    with pytest.raises(ValueError, match="malformed artifact spec: package name cannot be empty"):
        evaluate_archive_health(empty_name_artifacts)


def test_archive_health_version_boundary_equivalence() -> None:
    """Boundary value: sequential patch and minor increments without regression."""
    artifacts = [
        PackageArtifact(name="boundary-pkg", version="0.0.1"),
        PackageArtifact(name="boundary-pkg", version="0.0.2"),
        PackageArtifact(name="boundary-pkg", version="0.1.0"),
        PackageArtifact(name="boundary-pkg", version="1.0.0"),
        PackageArtifact(name="boundary-pkg", version="1.0.0-rc.1"),
    ]
    # In semver, 1.0.0-rc.1 is strictly less than 1.0.0, so sequence [1.0.0, 1.0.0-rc.1] is a regression
    report = evaluate_archive_health(artifacts)
    assert report.status == ArchiveStatus.CRITICAL
    assert any("boundary-pkg regressed from 1.0.0 to 1.0.0-rc.1" in r for r in report.version_regressions)


def test_archive_health_missing_dependency_failure() -> None:
    """Failing dependency: package requires an external or missing package in archive."""
    artifacts = [
        PackageArtifact(name="api-gateway", version="1.0.0", dependencies=["nonexistent-lib>=2.0"]),
    ]
    report = evaluate_archive_health(artifacts)

    assert report.status == ArchiveStatus.DEGRADED
    assert report.overall_score == 80.0
    assert report.degraded_count == 1
    assert len(report.broken_dependencies) == 1
    assert "api-gateway@1.0.0 requires missing dependency 'nonexistent-lib>=2.0'" in report.broken_dependencies[0]


def test_archive_health_regression_rollback_failure() -> None:
    """Version regression failure: package version drops from 2.0.0 to 1.9.0."""
    artifacts = [
        PackageArtifact(name="database-client", version="2.0.0"),
        PackageArtifact(name="database-client", version="1.9.0"),
    ]
    report = evaluate_archive_health(artifacts)

    assert report.status == ArchiveStatus.CRITICAL
    assert len(report.version_regressions) == 1
    assert "database-client regressed from 2.0.0 to 1.9.0" in report.version_regressions[0]
    assert report.overall_score < 80.0


def test_archive_health_score_invariants() -> None:
    """Invariant property: score is strictly bounded in [0.0, 100.0] and audit trail is deterministic."""
    policy = ArchiveHealthPolicy(min_reproducibility_score=85.0)
    # Severe failure case: multiple broken deps and regressions
    artifacts = [
        PackageArtifact(name="p1", version="2.0.0"),
        PackageArtifact(name="p1", version="1.0.0", dependencies=["d1", "d2", "d3", "d4", "d5", "d6"]),
    ]
    report = evaluate_archive_health(artifacts, policy=policy)

    assert 0.0 <= report.overall_score <= 100.0
    assert len(report.audit_trail) >= 3
    assert all(isinstance(entry, str) and len(entry) > 0 for entry in report.audit_trail)
    assert len(report.manifest_digest) == 64
