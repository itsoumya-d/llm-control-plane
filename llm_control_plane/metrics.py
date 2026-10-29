"""Minimal thread-safe metrics registry with Prometheus text output."""

from __future__ import annotations

import math
import threading
from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional, Tuple

LabelSet = Tuple[Tuple[str, str], ...]


def _labels(labels: Dict[str, str]) -> LabelSet:
    return tuple(sorted(labels.items()))


def _format_labels(labels: LabelSet) -> str:
    if not labels:
        return ""
    body = ",".join(f'{key}="{value}"' for key, value in labels)
    return "{" + body + "}"


def _percentile(values: List[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


class MetricsRegistry:
    def __init__(self) -> None:
        self._counters: DefaultDict[Tuple[str, LabelSet], float] = defaultdict(float)
        self._observations: DefaultDict[Tuple[str, LabelSet], List[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def inc(self, name: str, labels: Optional[Dict[str, str]] = None, value: float = 1.0) -> None:
        with self._lock:
            self._counters[(name, _labels(labels or {}))] += value

    def observe(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        with self._lock:
            self._observations[(name, _labels(labels or {}))].append(float(value))

    def counter_value(self, name: str, labels: Optional[Dict[str, str]] = None) -> float:
        with self._lock:
            return self._counters[(name, _labels(labels or {}))]

    def observations(self, name: str, labels: Optional[Dict[str, str]] = None) -> List[float]:
        with self._lock:
            return list(self._observations[(name, _labels(labels or {}))])

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        with self._lock:
            return {
                name: {str(labels): value for (metric, labels), value in self._counters.items() if metric == name}
                for name in {metric for metric, _ in self._counters}
            }

    def prometheus_text(self) -> str:
        lines: List[str] = []
        with self._lock:
            counters = sorted(self._counters.items())
            observations = sorted(self._observations.items())
        for (name, labels), value in counters:
            lines.append(f"{name}{_format_labels(labels)} {value:g}")
        for (name, labels), values in observations:
            if not values:
                continue
            lines.append(f"{name}_count{_format_labels(labels)} {len(values)}")
            lines.append(f"{name}_sum{_format_labels(labels)} {sum(values):g}")
            lines.append(f"{name}_p50{_format_labels(labels)} {_percentile(values, 0.50):g}")
            lines.append(f"{name}_p95{_format_labels(labels)} {_percentile(values, 0.95):g}")
        return "\n".join(lines) + ("\n" if lines else "")

    def merge(self, other: "MetricsRegistry") -> None:
        with other._lock:
            counters = list(other._counters.items())
            observations = list(other._observations.items())
        with self._lock:
            for key, value in counters:
                self._counters[key] += value
            for key, values in observations:
                self._observations[key].extend(values)
