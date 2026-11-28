"""Archive Health and Package Artifact Forward-Verification Engine."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple


class ArchiveStatus(str, Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True)
class PackageArtifact:
    name: str
    version: str
    dependencies: Sequence[str] = field(default_factory=tuple)
    sha256: str = ""
    status: str = "active"


@dataclass(frozen=True)
class ArchiveHealthPolicy:
    require_monotonic_progression: bool = True
    allow_deprecated_deps: bool = False
    min_reproducibility_score: float = 80.0


@dataclass
class ArchiveHealthReport:
    status: ArchiveStatus
    overall_score: float
    total_artifacts: int
    healthy_count: int
    degraded_count: int
    broken_dependencies: List[str]
    version_regressions: List[str]
    audit_trail: List[str]
    manifest_digest: str


_SEMVER_REGEX = re.compile(
    r"^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)"
    r"(?:-(?P<prerelease>[0-9A-Za-z.-]+))?"
    r"(?:\+(?P<buildmetadata>[0-9A-Za-z.-]+))?$"
)


def parse_semver(version_str: str) -> Tuple[int, int, int, str]:
    """Parse a semantic version string into (major, minor, patch, prerelease)."""
    match = _SEMVER_REGEX.match(version_str.strip())
    if not match:
        raise ValueError(f"malformed artifact spec: invalid semantic version '{version_str}'")
    major = int(match.group("major"))
    minor = int(match.group("minor"))
    patch = int(match.group("patch"))
    prerelease = match.group("prerelease") or ""
    return (major, minor, patch, prerelease)


def compare_semver(v1: str, v2: str) -> int:
    """Compare two semver strings. Returns -1 if v1 < v2, 0 if v1 == v2, 1 if v1 > v2."""
    p1 = parse_semver(v1)
    p2 = parse_semver(v2)
    # Compare numeric tuples first
    if p1[:3] < p2[:3]:
        return -1
    if p1[:3] > p2[:3]:
        return 1
    # Prerelease logic: empty prerelease is normal release (greater than any prerelease)
    if not p1[3] and p2[3]:
        return 1
    if p1[3] and not p2[3]:
        return -1
    if p1[3] < p2[3]:
        return -1
    if p1[3] > p2[3]:
        return 1
    return 0


def evaluate_archive_health(
    artifacts: Sequence[PackageArtifact],
    policy: Optional[ArchiveHealthPolicy] = None,
) -> ArchiveHealthReport:
    """Evaluate archive package artifacts for forward progression, dependency closure, and release health."""
    if not artifacts:
        raise ValueError("archive cannot be empty")

    policy = policy or ArchiveHealthPolicy()
    audit_trail: List[str] = []
    now_iso = datetime.now(timezone.utc).isoformat()
    audit_trail.append(f"{now_iso} - Archive health evaluation initiated for {len(artifacts)} artifacts")

    # Map available packages
    available_packages: Dict[str, List[PackageArtifact]] = {}
    hasher = hashlib.sha256()

    for art in artifacts:
        if not art.name or not art.name.strip():
            raise ValueError("malformed artifact spec: package name cannot be empty")
        # Validate semver format
        parse_semver(art.version)

        available_packages.setdefault(art.name, []).append(art)
        hasher.update(f"{art.name}:{art.version}:{art.sha256}".encode())

    manifest_digest = hasher.hexdigest()
    audit_trail.append(f"{now_iso} - Manifest integrity digest: {manifest_digest}")

    broken_dependencies: List[str] = []
    version_regressions: List[str] = []

    # Check forward progression (monotonic versioning) per package
    for pkg_name, versions in available_packages.items():
        if len(versions) > 1 and policy.require_monotonic_progression:
            for i in range(len(versions) - 1):
                prev_ver = versions[i].version
                next_ver = versions[i + 1].version
                if compare_semver(next_ver, prev_ver) < 0:
                    regression_msg = f"{pkg_name} regressed from {prev_ver} to {next_ver}"
                    version_regressions.append(regression_msg)
                    audit_trail.append(f"{now_iso} - ERROR: Version rollback detected: {regression_msg}")

    # Check dependency closure
    for art in artifacts:
        for dep in art.dependencies:
            dep_clean = re.split(r"[<>=!~ ]+", dep.strip())[0].strip()
            if not dep_clean:
                continue
            if dep_clean not in available_packages:
                broken_msg = f"{art.name}@{art.version} requires missing dependency '{dep}'"
                broken_dependencies.append(broken_msg)
                audit_trail.append(f"{now_iso} - WARN: Unresolved dependency: {broken_msg}")

    # Calculate overall health score
    score = 100.0
    score -= len(broken_dependencies) * 20.0
    score -= len(version_regressions) * 35.0
    score = max(0.0, min(100.0, score))

    if version_regressions or score < 50.0:
        status = ArchiveStatus.CRITICAL
    elif broken_dependencies or score < policy.min_reproducibility_score:
        status = ArchiveStatus.DEGRADED
    else:
        status = ArchiveStatus.HEALTHY

    degraded_count = len(broken_dependencies) + len(version_regressions)
    healthy_count = max(0, len(artifacts) - degraded_count)

    audit_trail.append(
        f"{now_iso} - Evaluation completed: status={status.value}, score={score:.1f}, "
        f"healthy={healthy_count}, degraded={degraded_count}"
    )

    return ArchiveHealthReport(
        status=status,
        overall_score=score,
        total_artifacts=len(artifacts),
        healthy_count=healthy_count,
        degraded_count=degraded_count,
        broken_dependencies=broken_dependencies,
        version_regressions=version_regressions,
        audit_trail=audit_trail,
        manifest_digest=manifest_digest,
    )
