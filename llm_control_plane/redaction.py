"""Deterministic PII redaction before provider calls and audit persistence."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Pattern, Tuple


@dataclass(frozen=True)
class RedactionResult:
    text: str
    counts: Dict[str, int]

    @property
    def total(self) -> int:
        return sum(self.counts.values())


class Redactor:
    def __init__(self, patterns: Optional[Dict[str, Pattern[str]]] = None) -> None:
        self._patterns = patterns or {
            "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
            "credit_card": re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
            "iban": re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"),
            "ipv4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
            "phone": re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{7,}\d)(?!\d)"),
        }

    def redact(self, text: str) -> RedactionResult:
        counts: Dict[str, int] = {}
        redacted = text
        for name, pattern in self._patterns.items():
            redacted, count = pattern.subn(f"[REDACTED_{name.upper()}]", redacted)
            if count:
                counts[name] = count
        return RedactionResult(text=redacted, counts=counts)

    def redact_many(self, texts: List[str]) -> Tuple[List[str], Dict[str, int]]:
        totals: Dict[str, int] = {}
        output: List[str] = []
        for text in texts:
            result = self.redact(text)
            output.append(result.text)
            for key, value in result.counts.items():
                totals[key] = totals.get(key, 0) + value
        return output, totals
