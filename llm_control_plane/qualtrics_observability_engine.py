"""Qualtrics Observability Infrastructure & Customer Experience Correlation Engine.

Enterprise telemetry stream ingestion, infrastructure-to-customer impact correlation,
automated MTTR reduction root cause analysis, and declarative EKS observability automation.
Tailored for Qualtrics Systems Engineer I (Observability Infrastructure) in Dublin, Ireland.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class TelemetryType(str, Enum):
    """Type of ingested operational telemetry."""

    METRIC = "METRIC"
    LOG = "LOG"
    TRACE = "TRACE"


class SeverityLevel(str, Enum):
    """Alert severity level for telemetry events and anomalies."""

    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class TelemetryEvent:
    """Individual operational telemetry event from EKS hosts or microservices."""

    event_id: str
    telemetry_type: TelemetryType
    service_name: str
    cluster_id: str
    timestamp: float
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    log_message: Optional[str] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    latency_ms: Optional[float] = None
    http_status: Optional[int] = None
    tags: Dict[str, str] = field(default_factory=dict)

    def compute_hash(self) -> str:
        """Compute deterministic payload hash for deduplication."""
        raw = (
            f"{self.telemetry_type}:{self.service_name}:{self.cluster_id}:"
            f"{self.metric_name}:{self.trace_id}:{self.span_id}:{self.timestamp}"
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


@dataclass
class CustomerImpactSignal:
    """High-level customer experience impact indicator."""

    metric_name: str
    service_name: str
    timestamp: float
    p99_latency_ms: float
    error_rate: float
    impact_score: float
    is_breached: bool
    breach_reason: Optional[str] = None


@dataclass
class IncidentBrief:
    """Synthesized MTTR reduction brief for on-call engineers."""

    incident_id: str
    service_name: str
    cluster_id: str
    detected_at: float
    culprit_component: str
    root_cause_summary: str
    correlation_score: float
    suggested_runbook: str
    affected_endpoints: List[str] = field(default_factory=list)


class TelemetryIngestionStream:
    """High-throughput operational telemetry parser and ingestion buffer."""

    def __init__(self, max_buffer_size: int = 10000, drop_on_backpressure: bool = False):
        self.max_buffer_size = max_buffer_size
        self.drop_on_backpressure = drop_on_backpressure
        self._buffer: List[TelemetryEvent] = []
        self._seen_hashes: set[str] = set()
        self.total_received = 0
        self.total_dropped = 0
        self.total_deduplicated = 0

    def ingest(self, event: TelemetryEvent) -> bool:
        """Ingest event with deduplication and backpressure protection."""
        self.total_received += 1
        event_hash = event.compute_hash()
        if event_hash in self._seen_hashes:
            self.total_deduplicated += 1
            return False

        if len(self._buffer) >= self.max_buffer_size:
            if self.drop_on_backpressure:
                self.total_dropped += 1
                return False
            # Evict oldest entry
            oldest = self._buffer.pop(0)
            self._seen_hashes.discard(oldest.compute_hash())

        self._buffer.append(event)
        self._seen_hashes.add(event_hash)
        return True

    def flush_batch(self, batch_size: int = 100) -> List[TelemetryEvent]:
        """Flush up to batch_size events for processing."""
        batch = self._buffer[:batch_size]
        self._buffer = self._buffer[batch_size:]
        for ev in batch:
            self._seen_hashes.discard(ev.compute_hash())
        return batch

    @property
    def buffer_depth(self) -> int:
        """Current events in buffer."""
        return len(self._buffer)


class CustomerExperienceCorrelator:
    """Correlates low-level infrastructure telemetry with customer impact metrics."""

    def __init__(self, latency_sla_ms: float = 250.0, error_rate_sla: float = 0.01):
        self.latency_sla_ms = latency_sla_ms
        self.error_rate_sla = error_rate_sla

    def evaluate_customer_impact(
        self,
        service_name: str,
        events: List[TelemetryEvent],
        window_seconds: float = 60.0,
    ) -> CustomerImpactSignal:
        """Calculate p99 latency, error rate, and customer impact breach."""
        now = time.time()
        relevant = [
            e for e in events
            if e.service_name == service_name and (now - e.timestamp) <= window_seconds
        ]
        if not relevant:
            return CustomerImpactSignal(
                metric_name="customer_experience_index",
                service_name=service_name,
                timestamp=now,
                p99_latency_ms=0.0,
                error_rate=0.0,
                impact_score=0.0,
                is_breached=False,
            )

        latencies = [e.latency_ms for e in relevant if e.latency_ms is not None]
        errors = [
            e for e in relevant
            if (e.http_status and e.http_status >= 500)
            or (e.log_message and "ERROR" in e.log_message)
        ]

        # Calculate p99 latency
        if latencies:
            latencies.sort()
            idx = int(math.ceil(0.99 * len(latencies))) - 1
            p99 = latencies[max(0, min(idx, len(latencies) - 1))]
        else:
            p99 = 0.0

        error_rate = len(errors) / len(relevant) if relevant else 0.0

        # Weighted impact score: 60% error rate, 40% latency inflation above SLA
        latency_factor = min(3.0, p99 / self.latency_sla_ms) if self.latency_sla_ms > 0 else 1.0
        error_factor = min(10.0, error_rate / self.error_rate_sla) if self.error_rate_sla > 0 else 1.0
        impact_score = round(0.4 * latency_factor + 0.6 * error_factor, 2)

        is_breached = (p99 > self.latency_sla_ms) or (error_rate > self.error_rate_sla)
        reason = None
        if is_breached:
            reasons = []
            if p99 > self.latency_sla_ms:
                reasons.append(f"p99 latency {p99:.1f}ms exceeds {self.latency_sla_ms:.1f}ms SLA")
            if error_rate > self.error_rate_sla:
                reasons.append(f"error rate {error_rate:.2%} exceeds {self.error_rate_sla:.2%} SLA")
            reason = "; ".join(reasons)

        return CustomerImpactSignal(
            metric_name="customer_experience_index",
            service_name=service_name,
            timestamp=now,
            p99_latency_ms=p99,
            error_rate=error_rate,
            impact_score=impact_score,
            is_breached=is_breached,
            breach_reason=reason,
        )

    def correlate_infrastructure_metric(
        self,
        infra_metric_values: List[float],
        customer_latencies: List[float],
    ) -> float:
        """Compute Pearson correlation coefficient between infra metric and customer latency."""
        n = min(len(infra_metric_values), len(customer_latencies))
        if n < 3:
            return 0.0

        x = infra_metric_values[:n]
        y = customer_latencies[:n]
        mean_x = sum(x) / n
        mean_y = sum(y) / n

        var_x = sum((val - mean_x) ** 2 for val in x)
        var_y = sum((val - mean_y) ** 2 for val in y)

        if var_x == 0.0 or var_y == 0.0:
            return 0.0

        covar = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        r = covar / math.sqrt(var_x * var_y)
        return round(max(-1.0, min(1.0, r)), 4)


class MTTRReductionRootCauseAnalyzer:
    """Automates root cause localization to reduce Mean Time to Resolution."""

    def __init__(self, runbook_registry: Optional[Dict[str, str]] = None):
        self.runbook_registry = runbook_registry or {
            "cpu_throttle": "https://wiki.internal.qualtrics.com/runbooks/eks-pod-cpu-scaling",
            "memory_pressure": "https://wiki.internal.qualtrics.com/runbooks/eks-oom-tuning",
            "database_connection_pool": "https://wiki.internal.qualtrics.com/runbooks/aurora-pool-exhaustion",
            "downstream_timeout": "https://wiki.internal.qualtrics.com/runbooks/service-mesh-timeout",
            "default": "https://wiki.internal.qualtrics.com/runbooks/general-incident-triage",
        }

    def analyze_incident(
        self,
        service_name: str,
        cluster_id: str,
        impact_signal: CustomerImpactSignal,
        metric_correlations: Dict[str, float],
    ) -> IncidentBrief:
        """Generate structured incident brief with identified root cause and runbook."""
        now = time.time()
        incident_id = f"INC-{int(now)}-{service_name[:6].upper()}"

        if not metric_correlations:
            return IncidentBrief(
                incident_id=incident_id,
                service_name=service_name,
                cluster_id=cluster_id,
                detected_at=now,
                culprit_component="unknown",
                root_cause_summary="No correlated infrastructure signals identified.",
                correlation_score=0.0,
                suggested_runbook=self.runbook_registry["default"],
            )

        # Find metric with highest absolute correlation
        culprit, score = max(metric_correlations.items(), key=lambda item: abs(item[1]))

        if "cpu" in culprit.lower():
            runbook_key = "cpu_throttle"
            summary = f"High CPU throttling ({culprit}) correlated with customer latency spikes."
        elif "mem" in culprit.lower() or "oom" in culprit.lower():
            runbook_key = "memory_pressure"
            summary = f"Memory pressure ({culprit}) causing container degradation and GC stalls."
        elif "db" in culprit.lower() or "conn" in culprit.lower() or "pool" in culprit.lower():
            runbook_key = "database_connection_pool"
            summary = f"Database connection pool saturation ({culprit}) delaying response dispatch."
        else:
            runbook_key = "downstream_timeout"
            summary = f"Infrastructure metric anomaly in {culprit} strongly linked to customer impact."

        return IncidentBrief(
            incident_id=incident_id,
            service_name=service_name,
            cluster_id=cluster_id,
            detected_at=now,
            culprit_component=culprit,
            root_cause_summary=summary,
            correlation_score=score,
            suggested_runbook=self.runbook_registry.get(runbook_key, self.runbook_registry["default"]),
            affected_endpoints=[f"/{service_name}/api/v1/query", f"/{service_name}/api/v1/submit"],
        )


class EksObservabilityAutomator:
    """Declarative EKS Prometheus ServiceMonitor and OpenTelemetry agent manager."""

    def __init__(self, cluster_id: str, default_scrape_interval: str = "15s"):
        self.cluster_id = cluster_id
        self.default_scrape_interval = default_scrape_interval
        self._managed_monitors: Dict[str, Dict[str, Any]] = {}

    def generate_service_monitor_manifest(
        self,
        service_name: str,
        namespace: str,
        port_name: str = "http-metrics",
        metrics_path: str = "/metrics",
    ) -> Dict[str, Any]:
        """Generate Kubernetes Prometheus Operator ServiceMonitor CRD."""
        manifest = {
            "apiVersion": "monitoring.coreos.com/v1",
            "kind": "ServiceMonitor",
            "metadata": {
                "name": f"{service_name}-monitor",
                "namespace": namespace,
                "labels": {
                    "app.kubernetes.io/name": service_name,
                    "qualtrics.com/cluster": self.cluster_id,
                    "release": "prometheus-operator",
                },
            },
            "spec": {
                "selector": {
                    "matchLabels": {
                        "app.kubernetes.io/name": service_name,
                    }
                },
                "endpoints": [
                    {
                        "port": port_name,
                        "path": metrics_path,
                        "interval": self.default_scrape_interval,
                        "scrapeTimeout": "10s",
                    }
                ],
            },
        }
        self._managed_monitors[service_name] = manifest
        return manifest

    def reconcile_monitor_drift(
        self,
        service_name: str,
        live_manifest: Dict[str, Any],
    ) -> Tuple[bool, List[str]]:
        """Check for configuration drift between live manifest and desired standard."""
        desired = self._managed_monitors.get(service_name)
        if not desired:
            return False, [f"Service {service_name} has no declared desired monitor."]

        drift_issues = []
        desired_spec = desired.get("spec", {})
        live_spec = live_manifest.get("spec", {})

        desired_endpoints = desired_spec.get("endpoints", [{}])[0]
        live_endpoints = live_spec.get("endpoints", [{}])[0]

        if desired_endpoints.get("interval") != live_endpoints.get("interval"):
            drift_issues.append(
                f"Scrape interval drift: desired {desired_endpoints.get('interval')} "
                f"vs live {live_endpoints.get('interval')}"
            )
        if desired_endpoints.get("path") != live_endpoints.get("path"):
            drift_issues.append(
                f"Metrics path drift: desired {desired_endpoints.get('path')} "
                f"vs live {live_endpoints.get('path')}"
            )

        return len(drift_issues) == 0, drift_issues
