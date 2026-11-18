"""Unit and integration tests for Teero Dental Temporal Workflow Orchestrator."""

import pytest

from llm_control_plane.teero_dental_temporal_workflow_orchestrator import (
    CDT_CATALOG,
    DentalClaimItem,
    DentalClaimStatus,
    DentalHygienist,
    DentalInsuranceClaim,
    DentalShiftRequest,
    HygienistSpecialty,
    ShiftBookingStatus,
    TeeroRcmClaimAuditor,
    TeeroShiftMatchingEngine,
    TeeroTemporalWorkflowEngine,
)


@pytest.fixture
def sample_hygienists():
    h1 = DentalHygienist(
        hygienist_id="hyg_01",
        name="Sarah Jenkins, RDH",
        license_state="TX",
        active_license=True,
        license_number="RDH-94821",
        specialties={HygienistSpecialty.GENERAL_HYGIENE, HygienistSpecialty.LOCAL_ANESTHESIA},
        hourly_rate_usd=55.0,
        travel_radius_km=30.0,
        latitude=30.2672,
        longitude=-97.7431,
        rating=4.9,
        completed_shifts=42,
    )
    h2 = DentalHygienist(
        hygienist_id="hyg_02",
        name="Alex Rivera, RDH",
        license_state="TX",
        active_license=True,
        license_number="RDH-88123",
        specialties={HygienistSpecialty.GENERAL_HYGIENE, HygienistSpecialty.PERIODONTAL},
        hourly_rate_usd=62.0,
        travel_radius_km=25.0,
        latitude=30.2800,
        longitude=-97.7500,
        rating=4.8,
        completed_shifts=35,
    )
    h3 = DentalHygienist(
        hygienist_id="hyg_03",
        name="Inactive Practitioner",
        license_state="TX",
        active_license=False,
        license_number="RDH-11002",
        specialties={HygienistSpecialty.GENERAL_HYGIENE},
        hourly_rate_usd=50.0,
        travel_radius_km=50.0,
        latitude=30.2672,
        longitude=-97.7431,
        rating=4.5,
    )
    return [h1, h2, h3]


@pytest.fixture
def sample_shift():
    return DentalShiftRequest(
        shift_id="shift_101",
        practice_id="prac_austin_central",
        practice_name="Austin Modern Dental Care",
        required_specialties={HygienistSpecialty.GENERAL_HYGIENE, HygienistSpecialty.LOCAL_ANESTHESIA},
        duration_hours=8.0,
        max_hourly_rate_usd=60.0,
        practice_lat=30.2670,
        practice_lon=-97.7430,
    )


def test_cdt_catalog_integrity():
    assert "D0120" in CDT_CATALOG
    assert "D1110" in CDT_CATALOG
    assert CDT_CATALOG["D4341"].requires_quadrant is True
    assert CDT_CATALOG["D2391"].requires_tooth_number is True


def test_shift_matching_engine_filters_and_ranks(sample_shift, sample_hygienists):
    engine = TeeroShiftMatchingEngine()
    ranked = engine.rank_candidates(sample_shift, sample_hygienists)

    assert len(ranked) == 1
    best_candidate, score = ranked[0]
    assert best_candidate.hygienist_id == "hyg_01"
    assert best_candidate.name == "Sarah Jenkins, RDH"
    assert score > 70.0


def test_shift_matching_engine_no_match(sample_shift, sample_hygienists):
    engine = TeeroShiftMatchingEngine()
    # Require specialty no one has
    sample_shift.required_specialties.add(HygienistSpecialty.ORTHODONTICS)
    ranked = engine.rank_candidates(sample_shift, sample_hygienists)
    assert len(ranked) == 0


def test_rcm_claim_auditor_valid_claim():
    auditor = TeeroRcmClaimAuditor()
    claim = DentalInsuranceClaim(
        claim_id="clm_001",
        practice_id="prac_01",
        patient_id="pat_404",
        payer_id="payer_delta_dental",
        items=[
            DentalClaimItem(cdt_code="D0120", fee_usd=65.0),
            DentalClaimItem(cdt_code="D1110", fee_usd=105.0),
            DentalClaimItem(cdt_code="D2391", fee_usd=195.0, tooth_number=14),
            DentalClaimItem(cdt_code="D4341", fee_usd=285.0, quadrant="UR"),
        ],
    )
    valid, errors = auditor.audit_claim(claim)
    assert valid is True
    assert len(errors) == 0
    assert claim.status == DentalClaimStatus.VALIDATED
    assert claim.total_billed_usd == 650.0


def test_rcm_claim_auditor_invalid_claim():
    auditor = TeeroRcmClaimAuditor()
    claim = DentalInsuranceClaim(
        claim_id="clm_002",
        practice_id="prac_01",
        patient_id="pat_404",
        payer_id="payer_metlife",
        items=[
            DentalClaimItem(cdt_code="D2391", fee_usd=195.0),  # missing tooth number
            DentalClaimItem(cdt_code="D4341", fee_usd=285.0),  # missing quadrant
            DentalClaimItem(cdt_code="D9999", fee_usd=50.0),  # invalid CDT code
            DentalClaimItem(cdt_code="D1110", fee_usd=-10.0),  # negative fee
        ],
    )
    valid, errors = auditor.audit_claim(claim)
    assert valid is False
    assert len(errors) == 4
    assert claim.status == DentalClaimStatus.DRAFT


def test_rcm_claim_adjudication():
    auditor = TeeroRcmClaimAuditor()
    claim = DentalInsuranceClaim(
        claim_id="clm_003",
        practice_id="prac_01",
        patient_id="pat_55",
        payer_id="payer_guardian",
        items=[
            DentalClaimItem(cdt_code="D0120", fee_usd=65.0),
            DentalClaimItem(cdt_code="D1110", fee_usd=105.0),
        ],
    )
    adjudicated = auditor.adjudicate_claim(claim, coverage_rate=0.80, annual_deductible_usd=50.0)
    assert adjudicated.status == DentalClaimStatus.ADJUDICATED
    assert adjudicated.allowed_amount_usd == 170.0
    # after $50 deductible = $120 * 0.80 = $96 paid
    assert adjudicated.paid_amount_usd == 96.0
    assert adjudicated.patient_responsibility_usd == 74.0
    assert len(adjudicated.adjudication_notes) > 0


def test_temporal_workflow_shift_dispatch_success(sample_shift, sample_hygienists):
    workflow_engine = TeeroTemporalWorkflowEngine()
    result = workflow_engine.run_shift_matching_workflow(sample_shift, sample_hygienists)

    assert result["status"] == "COMPLETED"
    assert result["state"] == ShiftBookingStatus.CONFIRMED
    assert result["assigned_hygienist"] == "Sarah Jenkins, RDH"
    assert sample_shift.status == ShiftBookingStatus.CONFIRMED
    assert sample_shift.assigned_hygienist_id == "hyg_01"
    assert len(workflow_engine.activity_log) >= 2


def test_temporal_workflow_shift_dispatch_no_candidates(sample_shift, sample_hygienists):
    workflow_engine = TeeroTemporalWorkflowEngine()
    sample_shift.max_hourly_rate_usd = 40.0  # lower than all rates
    result = workflow_engine.run_shift_matching_workflow(sample_shift, sample_hygienists)

    assert result["status"] == "FAILED"
    assert "No available hygienists" in result["error"]


def test_temporal_workflow_rcm_claim_success():
    workflow_engine = TeeroTemporalWorkflowEngine()
    claim = DentalInsuranceClaim(
        claim_id="clm_004",
        practice_id="prac_02",
        patient_id="pat_99",
        payer_id="payer_cigna",
        items=[
            DentalClaimItem(cdt_code="D0120", fee_usd=65.0),
            DentalClaimItem(cdt_code="D1110", fee_usd=105.0),
        ],
    )
    result = workflow_engine.run_rcm_claim_workflow(claim, coverage_rate=0.80)
    assert result["status"] == "COMPLETED"
    assert result["claim_status"] == DentalClaimStatus.ADJUDICATED.value
    assert result["paid_amount_usd"] == 96.0
    assert result["patient_responsibility_usd"] == 74.0


def test_temporal_workflow_rcm_claim_validation_failure():
    workflow_engine = TeeroTemporalWorkflowEngine()
    claim = DentalInsuranceClaim(
        claim_id="clm_005",
        practice_id="prac_02",
        patient_id="pat_99",
        payer_id="payer_cigna",
        items=[],
    )
    result = workflow_engine.run_rcm_claim_workflow(claim)
    assert result["status"] == "FAILED"
    assert "validation_errors" in result
