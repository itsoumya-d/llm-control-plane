"""WorkSafe New Zealand (Mahi Haumaru Aotearoa) AI Business Governance & Delivery Framework.

Operationalizes the New Zealand Government Algorithm Charter, Health and Safety at
Work Act 2015 (HSWA) compliance, Māori Data Sovereignty (Te Tiriti o Waitangi) safeguards,
and enterprise Data & Insights delivery lifecycle oversight.
"""

from __future__ import annotations

import enum
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List


class AlgorithmCharterTier(str, enum.Enum):
    """NZ Government Algorithm Charter risk tiers."""
    TIER_1_LOW = "TIER_1_LOW"          # Internal productivity, summary drafting, non-decision support
    TIER_2_MODERATE = "TIER_2_MODERATE"  # Advisory decision support, requires mandatory human validation
    TIER_3_HIGH = "TIER_3_HIGH"          # Automated compliance triage, statutory penalty advice, high public impact


class GovernanceStatus(str, enum.Enum):
    """Governance lifecycle stages for AI use cases."""
    SUBMITTED = "SUBMITTED"
    UNDER_ASSESSMENT = "UNDER_ASSESSMENT"
    APPROVED = "APPROVED"
    REQUIRES_REMEDIATION = "REQUIRES_REMEDIATION"
    REJECTED = "REJECTED"


class RiskCategory(str, enum.Enum):
    """Statutory and operational risk vectors."""
    PRIVACY = "PRIVACY"
    MAORI_DATA_SOVEREIGNTY = "MAORI_DATA_SOVEREIGNTY"
    STATUTORY_COMPLIANCE = "STATUTORY_COMPLIANCE"
    ALGORITHMIC_BIAS = "ALGORITHMIC_BIAS"
    CYBER_SECURITY = "CYBER_SECURITY"
    EXPLAINABILITY = "EXPLAINABILITY"


@dataclass
class TeTiritiSafeguardReport:
    """Māori Data Sovereignty evaluation (Te Mana Raraunga principles)."""
    mana_whakahaere_passed: bool  # Governance & community oversight
    mana_motuhake_passed: bool    # Purpose aligns with collective well-being
    mana_tiaki_passed: bool        # Cultural safety and protection against biased harm categorization
    overall_compliant: bool
    notes: List[str] = field(default_factory=list)


@dataclass
class AIUseCaseProposal:
    """Formal intake specification for an AI/LLM use case at WorkSafe NZ."""
    use_case_id: str
    title: str
    business_unit: str
    sponsor_name: str
    intended_purpose: str
    data_sources: List[str]
    involves_identifiable_individuals: bool
    involves_maori_data: bool
    influences_statutory_decisions: bool
    human_in_the_loop_mandatory: bool
    model_provider: str
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GovernanceDecision:
    """Audit log entry for a formal governance board decision."""
    decision_id: str
    use_case_id: str
    status: GovernanceStatus
    tier: AlgorithmCharterTier
    approver: str
    rationale: str
    timestamp: float
    hash_signature: str


class WorkSafeAIBusinessGovernanceEvaluator:
    """Enterprise AI Business Governance engine for WorkSafe New Zealand."""

    def __init__(self, public_sector_mandate: str = "HSWA_2015"):
        self.mandate = public_sector_mandate
        self.registry: Dict[str, AIUseCaseProposal] = {}
        self.decisions: Dict[str, List[GovernanceDecision]] = {}
        self.audit_log: List[Dict[str, Any]] = []

    def evaluate_risk_tier(self, proposal: AIUseCaseProposal) -> AlgorithmCharterTier:
        """Determines the Algorithm Charter risk tier based on operational impact."""
        if proposal.influences_statutory_decisions:
            return AlgorithmCharterTier.TIER_3_HIGH
        if proposal.involves_identifiable_individuals or proposal.involves_maori_data:
            return AlgorithmCharterTier.TIER_2_MODERATE
        return AlgorithmCharterTier.TIER_1_LOW

    def audit_te_tiriti_safeguards(self, proposal: AIUseCaseProposal) -> TeTiritiSafeguardReport:
        """Validates alignment with Māori Data Sovereignty principles."""
        notes = []
        if not proposal.involves_maori_data:
            return TeTiritiSafeguardReport(
                mana_whakahaere_passed=True,
                mana_motuhake_passed=True,
                mana_tiaki_passed=True,
                overall_compliant=True,
                notes=["No specific Māori data sets or cohort identifiers flagged in proposal."],
            )

        # Involves Māori data: enforce rigorous safeguards
        has_cultural_governance = proposal.metadata.get("maori_governance_reviewed", False)
        non_extractive = proposal.metadata.get("community_benefit_explicit", False)
        bias_mitigation_active = proposal.metadata.get("demographic_disparity_tested", False)

        if not has_cultural_governance:
            notes.append("Requires consultation and sign-off with Māori advisory or data stewardship representative.")
        if not non_extractive:
            notes.append("Must explicitly articulate how the outcomes support equitable safety for Māori workers.")
        if not bias_mitigation_active:
            notes.append("Algorithmic bias and disparity tests must be executed prior to deployment.")

        compliant = has_cultural_governance and non_extractive and bias_mitigation_active
        return TeTiritiSafeguardReport(
            mana_whakahaere_passed=has_cultural_governance,
            mana_motuhake_passed=non_extractive,
            mana_tiaki_passed=bias_mitigation_active,
            overall_compliant=compliant,
            notes=notes,
        )

    def evaluate_proposal(self, proposal: AIUseCaseProposal) -> Dict[str, Any]:
        """Performs a comprehensive automated governance pre-assessment."""
        tier = self.evaluate_risk_tier(proposal)
        te_tiriti = self.audit_te_tiriti_safeguards(proposal)

        blocking_issues: List[str] = []

        # Tier 3 (High Risk) checks
        if tier == AlgorithmCharterTier.TIER_3_HIGH:
            if not proposal.human_in_the_loop_mandatory:
                blocking_issues.append("Tier 3 AI use cases require strictly enforced human-in-the-loop validation.")
            if not proposal.metadata.get("explainability_framework_ready", False):
                blocking_issues.append("Statutory decision support requires an auditable explainability mechanism.")

        # Privacy checks
        if proposal.involves_identifiable_individuals:
            if not proposal.metadata.get("privacy_impact_assessment_completed", False):
                blocking_issues.append("Privacy Impact Assessment (PIA) must be completed before approval.")

        # Māori Data Sovereignty checks
        if not te_tiriti.overall_compliant:
            blocking_issues.extend(te_tiriti.notes)

        status = GovernanceStatus.APPROVED if not blocking_issues else GovernanceStatus.REQUIRES_REMEDIATION

        return {
            "use_case_id": proposal.use_case_id,
            "charter_tier": tier.value,
            "status": status.value,
            "blocking_issues": blocking_issues,
            "te_tiriti_report": {
                "compliant": te_tiriti.overall_compliant,
                "notes": te_tiriti.notes,
            },
            "evaluated_at": time.time(),
        }

    def register_use_case(self, proposal: AIUseCaseProposal) -> str:
        """Registers a new use case in the centralized WorkSafe governance catalog."""
        if proposal.use_case_id in self.registry:
            raise ValueError(f"Use case ID {proposal.use_case_id} already registered.")
        self.registry[proposal.use_case_id] = proposal
        self.decisions[proposal.use_case_id] = []

        self._log_audit("REGISTER", proposal.use_case_id, {"title": proposal.title, "sponsor": proposal.sponsor_name})
        return proposal.use_case_id

    def record_decision(
        self,
        use_case_id: str,
        status: GovernanceStatus,
        approver: str,
        rationale: str,
    ) -> GovernanceDecision:
        """Records an immutable governance determination."""
        if use_case_id not in self.registry:
            raise KeyError(f"Unknown use case ID {use_case_id}")

        proposal = self.registry[use_case_id]
        tier = self.evaluate_risk_tier(proposal)
        t = time.time()

        content = f"{use_case_id}:{status.value}:{approver}:{t}:{rationale}"
        sig = hashlib.sha256(content.encode()).hexdigest()

        dec = GovernanceDecision(
            decision_id=f"dec_{int(t*1000)}",
            use_case_id=use_case_id,
            status=status,
            tier=tier,
            approver=approver,
            rationale=rationale,
            timestamp=t,
            hash_signature=sig,
        )

        self.decisions[use_case_id].append(dec)
        self._log_audit("DECISION", use_case_id, {"status": status.value, "approver": approver, "sig": sig})
        return dec

    def _log_audit(self, action: str, entity_id: str, details: Dict[str, Any]) -> None:
        entry = {
            "action": action,
            "entity_id": entity_id,
            "timestamp": time.time(),
            "details": details,
        }
        self.audit_log.append(entry)

    def export_summary_report(self) -> Dict[str, Any]:
        """Generates enterprise summary for executive delivery leadership."""
        tier_counts = {tier.value: 0 for tier in AlgorithmCharterTier}
        status_counts = {st.value: 0 for st in GovernanceStatus}

        for ucid, prop in self.registry.items():
            tier = self.evaluate_risk_tier(prop)
            tier_counts[tier.value] += 1

            decs = self.decisions.get(ucid, [])
            current_status = decs[-1].status.value if decs else GovernanceStatus.SUBMITTED.value
            status_counts[current_status] += 1

        return {
            "total_registered_use_cases": len(self.registry),
            "tier_distribution": tier_counts,
            "status_distribution": status_counts,
            "total_audit_events": len(self.audit_log),
        }
