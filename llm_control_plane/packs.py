"""Jurisdiction and country pack loader and policy evaluators for LLM Control Plane."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .policy import PolicyDecision
from .types import CompletionRequest

DEFAULT_PACKS_DIR = Path(__file__).resolve().parent.parent / "packs"


@dataclass(frozen=True)
class JurisdictionRule:
    id: str
    name: str
    jurisdiction: str
    standard: str
    what_it_decides: str
    values: Dict[str, Any]
    source_url: str
    checked_on: str


def load_country_pack(country: str, packs_dir: Optional[Path] = None) -> List[JurisdictionRule]:
    """Load and validate rules from packs/<country>/rules.json."""
    base = packs_dir or DEFAULT_PACKS_DIR
    pack_path = base / country.lower() / "rules.json"
    if not pack_path.exists():
        raise ValueError(f"unsupported jurisdiction or missing pack: {country!r} (expected {pack_path})")

    try:
        raw = json.loads(pack_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"failed to parse rules for {country!r}: {exc}") from exc

    if not isinstance(raw, list) or not raw:
        raise ValueError(f"pack rules for {country!r} must be a non-empty list")

    rules: List[JurisdictionRule] = []
    for item in raw:
        rule = JurisdictionRule(
            id=str(item.get("id", "")),
            name=str(item.get("name", "")),
            jurisdiction=str(item.get("jurisdiction", "")),
            standard=str(item.get("standard", "")),
            what_it_decides=str(item.get("what_it_decides", "")),
            values=dict(item.get("values") or {}),
            source_url=str(item.get("source_url", "")),
            checked_on=str(item.get("checked_on", "")),
        )
        if not rule.id or not rule.source_url:
            raise ValueError(f"rule missing id or source_url: {item}")
        rules.append(rule)
    return rules


def evaluate_jurisdiction_policy(
    request: CompletionRequest,
    country: str = "nz",
    packs_dir: Optional[Path] = None,
) -> PolicyDecision:
    """Evaluate a request against the regulatory rules of a specific country pack."""
    rules = load_country_pack(country, packs_dir=packs_dir)
    reasons: List[str] = []

    for rule in rules:
        if rule.id == "NZ-PRIV-005":
            # Principle 5: Storage and security of personal information
            if request.metadata.get("contains_unredacted_pii", "").lower() in ("true", "1", "yes"):
                reasons.append(
                    f"request violates NZ Privacy Act 2020 IPP 5 ({rule.source_url}): "
                    f"unredacted personal information detected in prompt"
                )
        elif rule.id == "NZ-PRIV-011":
            # Principle 11: Cross-border disclosure controls
            if request.metadata.get("cross_border_unauthorized", "").lower() in ("true", "1", "yes"):
                reasons.append(
                    f"request violates NZ Privacy Act 2020 IPP 11 ({rule.source_url}): "
                    f"unauthorized offshore model disclosure without comparable safeguards"
                )
        elif rule.id == "NZ-ALGO-CHARTER":
            # Algorithm Charter: Human oversight mandate for automated decisions
            if request.use_case == "autonomous_decision":
                if request.metadata.get("human_oversight_verified", "").lower() not in ("true", "1", "yes"):
                    reasons.append(
                        f"request violates Algorithm Charter for Aotearoa NZ ({rule.source_url}): "
                        f"high-impact automated decision requires human oversight verification"
                    )
        elif rule.id == "NZ-NZISM-CLOUD":
            # NZISM: Data classification and cryptographic audit
            classification = request.metadata.get("data_classification", "").upper()
            if classification in ("RESTRICTED", "SECRET", "CONFIDENTIAL"):
                if request.metadata.get("sovereign_tenant_verified", "").lower() not in ("true", "1", "yes"):
                    reasons.append(
                        f"request violates NZISM Cloud AI controls ({rule.source_url}): "
                        f"classified data tier '{classification}' requires verified sovereign tenant"
                    )

    return PolicyDecision(
        allowed=not reasons,
        policy_id=f"country-pack-{country.lower()}",
        reasons=tuple(reasons),
    )
