"""Sonatus Edge AI Orchestrator.

Provides production-grade edge intelligence for software-defined vehicles (SDVs),
including in-vehicle resource budget management, INT8/INT4 quantization selection,
high-frequency CAN bus signal streaming anomaly detection, unstructured kernel/OS
log clustering, and hybrid edge-to-cloud closed-loop telemetry triage.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class EdgeQuantizationTier(str, Enum):
    """Supported model quantization tiers for in-vehicle embedded execution."""

    FP32 = "FP32"
    INT8 = "INT8"
    INT4 = "INT4"


class ThermalState(str, Enum):
    """In-vehicle ECU / NPU thermal operating state."""

    NORMAL = "NORMAL"
    WARMED = "WARMED"
    THROTTLED = "THROTTLED"


class IncidentSeverity(str, Enum):
    """Telemetry incident severity classification."""

    NORMAL = "NORMAL"
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class EdgeModelSpec:
    """Specification of an edge AI model deployed on vehicle ECUs."""

    model_id: str
    base_ram_bytes: int  # Uncompressed FP32 baseline weight footprint
    base_latency_ms: float
    current_tier: EdgeQuantizationTier = EdgeQuantizationTier.FP32

    @property
    def current_ram_bytes(self) -> int:
        if self.current_tier == EdgeQuantizationTier.INT4:
            return int(self.base_ram_bytes * 0.125)
        elif self.current_tier == EdgeQuantizationTier.INT8:
            return int(self.base_ram_bytes * 0.25)
        return self.base_ram_bytes

    @property
    def current_latency_ms(self) -> float:
        if self.current_tier == EdgeQuantizationTier.INT4:
            return self.base_latency_ms * 0.45
        elif self.current_tier == EdgeQuantizationTier.INT8:
            return self.base_latency_ms * 0.60
        return self.base_latency_ms


class EdgeResourceBudgetManager:
    """Enforces strict memory budgets (<32MB RAM) and latency SLAs for vehicle ECUs."""

    def __init__(
        self,
        max_ram_bytes: int = 32 * 1024 * 1024,  # 32 MB default
        target_latency_ms: float = 15.0,
    ) -> None:
        self.max_ram_bytes = max_ram_bytes
        self.target_latency_ms = target_latency_ms
        self.models: dict[str, EdgeModelSpec] = {}

    def register_model(self, model: EdgeModelSpec) -> None:
        self.models[model.model_id] = model

    def total_ram_used(self) -> int:
        return sum(m.current_ram_bytes for m in self.models.values())

    def optimize_budgets(self, thermal: ThermalState = ThermalState.NORMAL) -> dict[str, Any]:
        """Dynamically applies INT8 or INT4 quantization to satisfy memory and thermal limits."""
        total_base = sum(m.base_ram_bytes for m in self.models.values())
        adaptations: dict[str, Any] = {}

        for m_id, model in self.models.items():
            if thermal == ThermalState.THROTTLED:
                model.current_tier = EdgeQuantizationTier.INT4
            elif thermal == ThermalState.WARMED:
                model.current_tier = EdgeQuantizationTier.INT8
            else:
                # Normal state: if base footprint exceeds budget, quantize to INT8
                if total_base > self.max_ram_bytes:
                    model.current_tier = EdgeQuantizationTier.INT8
                else:
                    model.current_tier = EdgeQuantizationTier.FP32

            adaptations[m_id] = {
                "tier": model.current_tier.value,
                "ram_bytes": model.current_ram_bytes,
                "latency_ms": round(model.current_latency_ms, 2),
            }

        # If still over budget, aggressively downgrade all to INT4
        if self.total_ram_used() > self.max_ram_bytes:
            for m_id, model in self.models.items():
                model.current_tier = EdgeQuantizationTier.INT4
                adaptations[m_id]["tier"] = EdgeQuantizationTier.INT4.value
                adaptations[m_id]["ram_bytes"] = model.current_ram_bytes
                adaptations[m_id]["latency_ms"] = round(model.current_latency_ms, 2)

        return {
            "total_ram_bytes": self.total_ram_used(),
            "max_ram_bytes": self.max_ram_bytes,
            "within_budget": self.total_ram_used() <= self.max_ram_bytes,
            "thermal_state": thermal.value,
            "models": adaptations,
        }


@dataclass
class CANSignalReading:
    """Individual vehicle CAN bus or Automotive Ethernet signal telemetry sample."""

    signal_name: str
    value: float
    timestamp_epoch_ms: int = field(
        default_factory=lambda: int(datetime.now(timezone.utc).timestamp() * 1000)
    )


class CANBusSignalAnomalyDetector:
    """Real-time streaming anomaly detection for vehicle telemetry over rolling windows."""

    def __init__(self, window_size: int = 50, z_threshold: float = 3.0) -> None:
        self.window_size = window_size
        self.z_threshold = z_threshold
        self._buffers: dict[str, list[float]] = {}

    def ingest_reading(self, reading: CANSignalReading) -> dict[str, Any]:
        """Ingests a telemetry sample, updates rolling stats, and detects anomalies."""
        name = reading.signal_name
        if name not in self._buffers:
            self._buffers[name] = []

        buf = self._buffers[name]
        buf.append(reading.value)
        if len(buf) > self.window_size:
            buf.pop(0)

        if len(buf) < 10:
            return {
                "signal_name": name,
                "current_value": reading.value,
                "is_anomaly": False,
                "z_score": 0.0,
                "confidence": 0.5,
                "status": "WARMING_UP",
            }

        mean = sum(buf) / len(buf)
        variance = sum((x - mean) ** 2 for x in buf) / len(buf)
        std_dev = math.sqrt(variance) if variance > 1e-9 else 1e-4

        z_score = abs(reading.value - mean) / std_dev
        is_anomaly = z_score >= self.z_threshold

        return {
            "signal_name": name,
            "current_value": reading.value,
            "rolling_mean": round(mean, 4),
            "rolling_std": round(std_dev, 4),
            "z_score": round(z_score, 2),
            "is_anomaly": is_anomaly,
            "confidence": min(1.0, round(len(buf) / self.window_size, 2)),
            "status": "ANOMALY" if is_anomaly else "NORMAL",
        }

    def evaluate_multi_signal_correlation(
        self, readings: list[CANSignalReading]
    ) -> dict[str, Any]:
        """Correlates multiple signal channels to detect systemic hardware/firmware faults."""
        results = [self.ingest_reading(r) for r in readings]
        anomalies = [r for r in results if r["is_anomaly"]]

        severity = IncidentSeverity.NORMAL
        fault_type = "NONE"
        action = "CONTINUE_MONITORING"

        # Rule-based automotive fault pattern correlation
        signal_names = {a["signal_name"] for a in anomalies}
        if "battery_cell_voltage_v" in signal_names and "motor_inverter_temp_c" in signal_names:
            severity = IncidentSeverity.CRITICAL
            fault_type = "POWERTRAIN_THERMAL_ELECTRICAL_FAULT"
            action = "DERATE_MOTOR_AND_ISOLATE_PACK"
        elif "steering_angle_deg" in signal_names and "brake_line_pressure_bar" in signal_names:
            severity = IncidentSeverity.CRITICAL
            fault_type = "CHASSIS_DYNAMIC_STABILITY_FAILURE"
            action = "ACTIVATE_ELECTRONIC_STABILITY_PROGRAM"
        elif len(anomalies) >= 2:
            severity = IncidentSeverity.WARNING
            fault_type = "MULTIPLE_SUBSYSTEM_TRANSIENT_ANOMALY"
            action = "LOG_HIGH_RES_PRECURSORS"
        elif len(anomalies) == 1:
            severity = IncidentSeverity.INFO
            fault_type = f"SINGLE_SIGNAL_BURST_{anomalies[0]['signal_name'].upper()}"
            action = "BUFFER_TELEMETRY"

        return {
            "severity": severity.value,
            "fault_type": fault_type,
            "recommended_action": action,
            "total_signals_evaluated": len(readings),
            "anomalous_signals_count": len(anomalies),
            "anomalies": anomalies,
        }


@dataclass
class KernelLogEntry:
    """Individual vehicle OS kernel (dmesg) or service trace log line."""

    message: str
    subsystem: str = "kernel"
    timestamp_epoch_ms: int = field(
        default_factory=lambda: int(datetime.now(timezone.utc).timestamp() * 1000)
    )


class KernelLogClusterEngine:
    """Tokenizes and clusters unstructured vehicle OS logs to detect software regressions."""

    def __init__(self) -> None:
        self.templates: dict[str, int] = {}
        # Regex masks for volatile in-vehicle trace tokens
        self._addr_regex = re.compile(r"0x[0-9a-fA-F]+")
        self._num_regex = re.compile(r"\b\d+\b")
        self._uuid_regex = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
        self._ip_regex = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")

    def extract_template(self, message: str) -> str:
        """Abstracts volatile numbers, addresses, and IDs into invariant log templates."""
        cleaned = self._uuid_regex.sub("<UUID>", message)
        cleaned = self._addr_regex.sub("<ADDR>", cleaned)
        cleaned = self._ip_regex.sub("<IP>", cleaned)
        cleaned = self._num_regex.sub("<NUM>", cleaned)
        return " ".join(cleaned.split())

    def ingest_log(self, log_entry: KernelLogEntry) -> dict[str, Any]:
        """Ingests log line, updates template frequency, and detects abnormal burst entropy."""
        template = self.extract_template(log_entry.message)
        self.templates[template] = self.templates.get(template, 0) + 1

        total_logs = sum(self.templates.values())
        freq = self.templates[template]

        # Calculate Shannon entropy across template distribution
        entropy = 0.0
        for count in self.templates.values():
            p = count / total_logs
            if p > 0:
                entropy -= p * math.log2(p)

        is_novel = freq == 1
        return {
            "template": template,
            "frequency": freq,
            "total_templates_tracked": len(self.templates),
            "fleet_entropy": round(entropy, 4),
            "is_novel_pattern": is_novel,
        }

    def cluster_templates(self) -> list[dict[str, Any]]:
        """Groups similar templates into semantic error clusters using token overlap."""
        clusters: list[dict[str, Any]] = []
        for tmpl, count in self.templates.items():
            tokens = set(tmpl.lower().split())
            matched = False
            for c in clusters:
                c_tokens = c["tokens"]
                intersection = len(tokens.intersection(c_tokens))
                union = len(tokens.union(c_tokens))
                jaccard = intersection / union if union > 0 else 0.0
                if jaccard >= 0.45:
                    c["templates"].append(tmpl)
                    c["total_occurrences"] += count
                    c["tokens"].update(tokens)
                    matched = True
                    break

            if not matched:
                clusters.append({
                    "cluster_id": f"cluster_{len(clusters) + 1}",
                    "templates": [tmpl],
                    "total_occurrences": count,
                    "tokens": tokens,
                })

        # Remove internal set before returning
        return [
            {
                "cluster_id": c["cluster_id"],
                "templates_count": len(c["templates"]),
                "total_occurrences": c["total_occurrences"],
                "sample_template": c["templates"][0],
            }
            for c in clusters
        ]


class HybridEdgeCloudTelemetryTriager:
    """Orchestrates closed-loop telemetry triage: compresses normal data and escalates anomalies."""

    def __init__(
        self,
        can_detector: CANBusSignalAnomalyDetector,
        log_engine: KernelLogClusterEngine,
        budget_mgr: EdgeResourceBudgetManager,
        compression_ratio: float = 28.0,  # Fastlane TSDB 28x compression
    ) -> None:
        self.can_detector = can_detector
        self.log_engine = log_engine
        self.budget_mgr = budget_mgr
        self.compression_ratio = compression_ratio

    def triage_cycle(
        self,
        can_readings: list[CANSignalReading],
        log_entries: list[KernelLogEntry],
        thermal: ThermalState = ThermalState.NORMAL,
    ) -> dict[str, Any]:
        """Executes one closed-loop cycle: evaluates telemetry, budgets, and transmission payload."""
        # 1. Budget optimization
        budget_status = self.budget_mgr.optimize_budgets(thermal)

        # 2. CAN telemetry multi-signal evaluation
        can_eval = self.can_detector.evaluate_multi_signal_correlation(can_readings)

        # 3. Kernel log ingestion
        for entry in log_entries:
            self.log_engine.ingest_log(entry)
        error_keywords = {"error", "warn", "fail", "fault", "panic", "drop", "abort", "critical", "trip"}
        novel_error_logs = [
            e for e in log_entries
            if any(k in e.message.lower() for k in error_keywords)
        ]

        # 4. Closed-loop decision
        is_critical = can_eval["severity"] == IncidentSeverity.CRITICAL.value
        is_warning = can_eval["severity"] == IncidentSeverity.WARNING.value or len(novel_error_logs) > 0

        raw_can_bytes = len(can_readings) * 32  # Simulated 32 bytes per raw frame
        raw_log_bytes = sum(len(e.message) for e in log_entries)
        total_raw_bytes = raw_can_bytes + raw_log_bytes

        if is_critical or is_warning:
            # High-resolution uncompressed incident capture
            transmission_mode = "IMMEDIATE_INCIDENT_ESCALATION"
            transmitted_bytes = total_raw_bytes
            savings_pct = 0.0
        else:
            # Normal state: apply Fastlane TSDB edge compression
            transmission_mode = "COMPRESSED_TSDB_BATCH"
            transmitted_bytes = int(total_raw_bytes / self.compression_ratio)
            savings_pct = round((1.0 - (transmitted_bytes / max(total_raw_bytes, 1))) * 100, 1)

        if is_critical:
            overall_sev = IncidentSeverity.CRITICAL.value
        elif is_warning:
            overall_sev = IncidentSeverity.WARNING.value
        elif can_eval["severity"] != IncidentSeverity.NORMAL.value:
            overall_sev = can_eval["severity"]
        else:
            overall_sev = IncidentSeverity.NORMAL.value

        return {
            "cycle_timestamp": datetime.now(timezone.utc).isoformat(),
            "overall_severity": overall_sev,
            "transmission_mode": transmission_mode,
            "raw_telemetry_bytes": total_raw_bytes,
            "transmitted_bytes": transmitted_bytes,
            "bandwidth_reduction_pct": savings_pct,
            "can_analysis": {
                "fault_type": can_eval["fault_type"],
                "recommended_action": can_eval["recommended_action"],
                "anomalies_detected": can_eval["anomalous_signals_count"],
            },
            "log_analysis": {
                "total_logs_processed": len(log_entries),
                "novel_patterns_detected": len(novel_error_logs),
                "active_clusters": len(self.log_engine.cluster_templates()),
            },
            "edge_budgets": {
                "total_ram_used_bytes": budget_status["total_ram_bytes"],
                "within_budget": budget_status["within_budget"],
                "thermal_state": budget_status["thermal_state"],
            },
        }
