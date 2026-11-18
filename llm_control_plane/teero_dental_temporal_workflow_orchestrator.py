"""Teero Dental Marketplace & Revenue Cycle Management (RCM) Temporal Workflow Orchestrator.

Provides event-driven durable workflow execution, automated dental hygienist staffing shift
matching, credential verification, and automated Revenue Cycle Management (RCM) dental claim
auditing with CDT procedure code adjudication for Teero's dental operating system.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


class HygienistSpecialty(str, Enum):
    """Clinical specialties and certifications for dental hygienists."""

    GENERAL_HYGIENE = "general_hygiene"
    PERIODONTAL = "periodontal"
    PEDIATRIC = "pediatric"
    LOCAL_ANESTHESIA = "local_anesthesia"
    ORTHODONTICS = "orthodontics"
    NITROUS_OXIDE = "nitrous_oxide"


class ShiftBookingStatus(str, Enum):
    """Lifecycle states of a marketplace dental shift."""

    REQUESTED = "REQUESTED"
    MATCHING = "MATCHING"
    OFFERED = "OFFERED"
    CONFIRMED = "CONFIRMED"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    DISPUTED = "DISPUTED"


class DentalClaimStatus(str, Enum):
    """Lifecycle states of an insurance RCM claim."""

    DRAFT = "DRAFT"
    VALIDATED = "VALIDATED"
    SUBMITTED = "SUBMITTED"
    ADJUDICATED = "ADJUDICATED"
    PAID = "PAID"
    DENIED = "DENIED"
    APPEALED = "APPEALED"


@dataclass(frozen=True)
class CDTProcedure:
    """Current Dental Terminology (CDT) procedure code definition."""

    code: str
    category: str
    description: str
    standard_fee_usd: float
    requires_tooth_number: bool = False
    requires_quadrant: bool = False


# Canonical CDT reference catalog
CDT_CATALOG: Dict[str, CDTProcedure] = {
    "D0120": CDTProcedure("D0120", "Diagnostic", "Periodic oral evaluation - established patient", 65.0),
    "D0150": CDTProcedure("D0150", "Diagnostic", "Comprehensive oral evaluation - new patient", 110.0),
    "D0210": CDTProcedure("D0210", "Diagnostic", "Intraoral - comprehensive series of radiographic images", 155.0),
    "D0274": CDTProcedure("D0274", "Diagnostic", "Bitewings - four radiographic images", 75.0),
    "D1110": CDTProcedure("D1110", "Preventive", "Prophylaxis - adult", 105.0),
    "D1120": CDTProcedure("D1120", "Preventive", "Prophylaxis - child", 80.0),
    "D1206": CDTProcedure("D1206", "Preventive", "Topical application of fluoride varnish", 45.0),
    "D4341": CDTProcedure(
        "D4341", "Periodontics", "Scaling and root planing - 4+ teeth/quadrant", 285.0, requires_quadrant=True
    ),
    "D4342": CDTProcedure(
        "D4342", "Periodontics", "Scaling and root planing - 1-3 teeth/quadrant", 185.0, requires_quadrant=True
    ),
    "D2391": CDTProcedure(
        "D2391", "Restorative", "Resin composite - 1 surface, posterior", 195.0, requires_tooth_number=True
    ),
    "D2740": CDTProcedure(
        "D2740", "Restorative", "Crown - porcelain/ceramic substrate", 1250.0, requires_tooth_number=True
    ),
}


@dataclass
class DentalHygienist:
    """Registered dental hygienist profile in the Teero staffing marketplace."""

    hygienist_id: str
    name: str
    license_state: str
    active_license: bool
    license_number: str
    specialties: Set[HygienistSpecialty]
    hourly_rate_usd: float
    travel_radius_km: float
    latitude: float
    longitude: float
    rating: float = 5.0
    completed_shifts: int = 0
    available_now: bool = True

    def meets_specialties(self, required: Set[HygienistSpecialty]) -> bool:
        """Verify whether hygienist holds all clinical specialties required for shift."""
        return required.issubset(self.specialties)


@dataclass
class DentalShiftRequest:
    """Staffing shift requisition from a dental practice."""

    shift_id: str
    practice_id: str
    practice_name: str
    required_specialties: Set[HygienistSpecialty]
    duration_hours: float
    max_hourly_rate_usd: float
    practice_lat: float
    practice_lon: float
    status: ShiftBookingStatus = ShiftBookingStatus.REQUESTED
    assigned_hygienist_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)


@dataclass
class DentalClaimItem:
    """Individual procedural line item on a dental claim."""

    cdt_code: str
    fee_usd: float
    tooth_number: Optional[int] = None
    quadrant: Optional[str] = None


@dataclass
class DentalInsuranceClaim:
    """Dental revenue cycle management claim for adjudication."""

    claim_id: str
    practice_id: str
    patient_id: str
    payer_id: str
    items: List[DentalClaimItem]
    status: DentalClaimStatus = DentalClaimStatus.DRAFT
    total_billed_usd: float = 0.0
    allowed_amount_usd: float = 0.0
    paid_amount_usd: float = 0.0
    patient_responsibility_usd: float = 0.0
    adjudication_notes: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def calculate_billed_total(self) -> float:
        self.total_billed_usd = round(sum(it.fee_usd for it in self.items), 2)
        return self.total_billed_usd


class TeeroShiftMatchingEngine:
    """Intelligent dispatch and ranking engine for dental staffing shifts."""

    @staticmethod
    def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Compute approximate distance in kilometers between two geo-coordinates."""
        # Simple Euclidean approximation suitable for localized metro dispatch
        dx = (lon2 - lon1) * 85.0
        dy = (lat2 - lat1) * 111.0
        return round((dx * dx + dy * dy) ** 0.5, 2)

    def rank_candidates(
        self,
        shift: DentalShiftRequest,
        candidates: List[DentalHygienist],
    ) -> List[Tuple[DentalHygienist, float]]:
        """Rank eligible hygienists using composite affinity score."""
        scored: List[Tuple[DentalHygienist, float]] = []

        for h in candidates:
            if not h.active_license or not h.available_now:
                continue
            if not h.meets_specialties(shift.required_specialties):
                continue
            if h.hourly_rate_usd > shift.max_hourly_rate_usd:
                continue

            dist = self.haversine_distance_km(shift.practice_lat, shift.practice_lon, h.latitude, h.longitude)
            if dist > h.travel_radius_km:
                continue

            # Composite match score:
            # 40% rating, 30% rate affordability, 30% geographic proximity
            proximity_score = max(0.0, 100.0 - (dist / max(1.0, h.travel_radius_km)) * 100.0)
            rate_score = (1.0 - (h.hourly_rate_usd / max(1.0, shift.max_hourly_rate_usd))) * 100.0
            rating_score = (h.rating / 5.0) * 100.0

            total_score = round((rating_score * 0.40) + (rate_score * 0.30) + (proximity_score * 0.30), 2)
            scored.append((h, total_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored


class TeeroRcmClaimAuditor:
    """Automated RCM claim pre-billing audit and rules engine."""

    def audit_claim(self, claim: DentalInsuranceClaim) -> Tuple[bool, List[str]]:
        """Verify compliance with CDT rules and coding standards before payer transmission."""
        errors: List[str] = []
        if not claim.items:
            errors.append("Claim must contain at least one procedural line item.")

        claim.calculate_billed_total()

        for idx, item in enumerate(claim.items):
            code_def = CDT_CATALOG.get(item.cdt_code)
            if not code_def:
                errors.append(f"Line {idx + 1}: Unrecognized CDT code '{item.cdt_code}'.")
                continue

            if code_def.requires_tooth_number:
                if item.tooth_number is None or not (1 <= item.tooth_number <= 32):
                    errors.append(f"Line {idx + 1} ({item.cdt_code}): Missing or invalid tooth number (must be 1-32).")

            if code_def.requires_quadrant:
                if not item.quadrant or item.quadrant.upper() not in {"UR", "UL", "LL", "LR"}:
                    errors.append(f"Line {idx + 1} ({item.cdt_code}): Missing or invalid quadrant (UR, UL, LL, LR).")

            if item.fee_usd <= 0:
                errors.append(f"Line {idx + 1} ({item.cdt_code}): Fee must be greater than zero.")

        valid = len(errors) == 0
        if valid:
            claim.status = DentalClaimStatus.VALIDATED
        else:
            claim.status = DentalClaimStatus.DRAFT
            claim.adjudication_notes.extend(errors)

        return valid, errors

    def adjudicate_claim(
        self,
        claim: DentalInsuranceClaim,
        coverage_rate: float = 0.80,
        annual_deductible_usd: float = 50.0,
    ) -> DentalInsuranceClaim:
        """Simulate real-time insurance adjudication and patient responsibility calculation."""
        is_valid, _ = self.audit_claim(claim)
        if not is_valid:
            claim.status = DentalClaimStatus.DENIED
            return claim

        allowed = 0.0
        for item in claim.items:
            code_def = CDT_CATALOG.get(item.cdt_code)
            # Allowed amount caps at standard fee
            standard = code_def.standard_fee_usd if code_def else item.fee_usd
            allowed += min(item.fee_usd, standard)

        claim.allowed_amount_usd = round(allowed, 2)
        after_deductible = max(0.0, claim.allowed_amount_usd - annual_deductible_usd)
        claim.paid_amount_usd = round(after_deductible * coverage_rate, 2)
        claim.patient_responsibility_usd = round(claim.total_billed_usd - claim.paid_amount_usd, 2)
        claim.status = DentalClaimStatus.ADJUDICATED
        claim.adjudication_notes.append(
            f"Adjudicated successfully: Billed ${claim.total_billed_usd:.2f}, "
            f"Allowed ${claim.allowed_amount_usd:.2f}, Plan Paid ${claim.paid_amount_usd:.2f}, "
            f"Patient Owes ${claim.patient_responsibility_usd:.2f}."
        )
        return claim


class TeeroTemporalWorkflowEngine:
    """Temporal durable workflow state machine for shift matching & RCM."""

    def __init__(self) -> None:
        self.matching_engine = TeeroShiftMatchingEngine()
        self.rcm_auditor = TeeroRcmClaimAuditor()
        self.workflows: Dict[str, Dict[str, Any]] = {}
        self.activity_log: List[Dict[str, Any]] = []

    def log_activity(self, workflow_id: str, activity_name: str, status: str, payload: Dict[str, Any]) -> None:
        self.activity_log.append(
            {
                "timestamp": time.time(),
                "workflow_id": workflow_id,
                "activity": activity_name,
                "status": status,
                "payload": payload,
            }
        )

    def run_shift_matching_workflow(
        self,
        shift: DentalShiftRequest,
        candidate_pool: List[DentalHygienist],
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """Execute durable shift dispatch workflow with automated matching and retries."""
        workflow_id = f"wf_shift_{shift.shift_id}_{uuid.uuid4().hex[:6]}"
        self.workflows[workflow_id] = {
            "workflow_id": workflow_id,
            "shift_id": shift.shift_id,
            "status": "RUNNING",
            "state": ShiftBookingStatus.MATCHING,
        }

        self.log_activity(workflow_id, "FilterEligibleHygienists", "STARTED", {"shift_id": shift.shift_id})
        ranked = self.matching_engine.rank_candidates(shift, candidate_pool)

        if not ranked:
            self.log_activity(workflow_id, "FilterEligibleHygienists", "FAILED", {"reason": "No matched candidates"})
            shift.status = ShiftBookingStatus.REQUESTED
            self.workflows[workflow_id]["status"] = "FAILED"
            self.workflows[workflow_id]["error"] = "No available hygienists meeting criteria"
            return self.workflows[workflow_id]

        best_match, score = ranked[0]
        self.log_activity(
            workflow_id,
            "SelectTopCandidate",
            "COMPLETED",
            {"hygienist_id": best_match.hygienist_id, "match_score": score},
        )

        # Confirm booking
        shift.status = ShiftBookingStatus.CONFIRMED
        shift.assigned_hygienist_id = best_match.hygienist_id
        best_match.available_now = False
        best_match.completed_shifts += 1

        self.log_activity(
            workflow_id,
            "ConfirmShiftDispatch",
            "COMPLETED",
            {"assigned_id": best_match.hygienist_id, "practice_id": shift.practice_id},
        )

        self.workflows[workflow_id]["status"] = "COMPLETED"
        self.workflows[workflow_id]["state"] = ShiftBookingStatus.CONFIRMED
        self.workflows[workflow_id]["assigned_hygienist"] = best_match.name
        self.workflows[workflow_id]["match_score"] = score
        return self.workflows[workflow_id]

    def run_rcm_claim_workflow(
        self,
        claim: DentalInsuranceClaim,
        coverage_rate: float = 0.80,
    ) -> Dict[str, Any]:
        """Execute end-to-end RCM claim validation and adjudication workflow."""
        workflow_id = f"wf_rcm_{claim.claim_id}_{uuid.uuid4().hex[:6]}"
        self.workflows[workflow_id] = {
            "workflow_id": workflow_id,
            "claim_id": claim.claim_id,
            "status": "RUNNING",
        }

        self.log_activity(workflow_id, "AuditClaimLines", "STARTED", {"claim_id": claim.claim_id})
        valid, errors = self.rcm_auditor.audit_claim(claim)

        if not valid:
            self.log_activity(workflow_id, "AuditClaimLines", "FAILED", {"errors": errors})
            self.workflows[workflow_id]["status"] = "FAILED"
            self.workflows[workflow_id]["validation_errors"] = errors
            return self.workflows[workflow_id]

        self.log_activity(workflow_id, "AuditClaimLines", "COMPLETED", {"billed_total": claim.total_billed_usd})

        self.log_activity(workflow_id, "AdjudicateClaimWithPayer", "STARTED", {"payer_id": claim.payer_id})
        adjudicated = self.rcm_auditor.adjudicate_claim(claim, coverage_rate=coverage_rate)

        self.log_activity(
            workflow_id,
            "AdjudicateClaimWithPayer",
            "COMPLETED",
            {
                "paid_amount": adjudicated.paid_amount_usd,
                "patient_owed": adjudicated.patient_responsibility_usd,
            },
        )

        self.workflows[workflow_id]["status"] = "COMPLETED"
        self.workflows[workflow_id]["claim_status"] = adjudicated.status.value
        self.workflows[workflow_id]["paid_amount_usd"] = adjudicated.paid_amount_usd
        self.workflows[workflow_id]["patient_responsibility_usd"] = adjudicated.patient_responsibility_usd
        return self.workflows[workflow_id]
