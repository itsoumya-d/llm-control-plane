import asyncio

import pytest

from llm_control_plane.workforce_agent_browser_orchestrator import (
    StepStatus,
    VerificationStep,
    WorkflowState,
    WorkforceBrowserOrchestrator,
)


@pytest.fixture
def orchestrator():
    return WorkforceBrowserOrchestrator(tenant_id="tenant-enterprise-99")


def test_create_isolated_session(orchestrator):
    config = orchestrator.create_isolated_session(session_id="sess-worker-101", proxy="http://proxy.internal:8080")
    assert config.tenant_id == "tenant-enterprise-99"
    assert config.session_id == "sess-worker-101"
    assert config.proxy_url == "http://proxy.internal:8080"
    assert config.headless is True
    assert config.viewport_width == 1280


def test_dom_hash_computation(orchestrator):
    dom_sample = "<html><body><h1>Workforce Identity Verification</h1></body></html>"
    h1 = orchestrator.compute_dom_hash(dom_sample)
    h2 = orchestrator.compute_dom_hash(dom_sample)
    h3 = orchestrator.compute_dom_hash(dom_sample + "<div>extra</div>")
    assert len(h1) == 64
    assert h1 == h2
    assert h1 != h3


def test_step_execution_success(orchestrator):
    async def _test():
        step = VerificationStep(
            step_id="step-1", name="Navigate", action="goto", target_selector="https://portal.example.com"
        )

        def dummy_handler(s):
            return {"page_title": "Onboarding Portal", "loaded": True}

        res = await orchestrator.execute_step_with_retry(step, dummy_handler)
        assert res.status == StepStatus.COMPLETED
        assert res.output_data["page_title"] == "Onboarding Portal"
        assert res.retry_count == 0
        assert res.duration_ms > 0

    asyncio.run(_test())


def test_step_execution_retry_success(orchestrator):
    async def _test():
        step = VerificationStep(
            step_id="step-2", name="UploadDocument", action="upload", max_retries=2, backoff_factor=1.0
        )

        attempts = 0

        def flaky_handler(s):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise ConnectionResetError("Temporary network reset")
            return {"uploaded": True, "file_id": "doc-992"}

        res = await orchestrator.execute_step_with_retry(step, flaky_handler)
        assert res.status == StepStatus.COMPLETED
        assert res.retry_count == 1
        assert res.output_data["uploaded"] is True

    asyncio.run(_test())


def test_step_execution_max_retries_failure(orchestrator):
    async def _test():
        step = VerificationStep(
            step_id="step-3", name="VerifyTaxID", action="extract", max_retries=2, backoff_factor=1.0
        )

        def broken_handler(s):
            raise TimeoutError("Portal selector timeout")

        res = await orchestrator.execute_step_with_retry(step, broken_handler)
        assert res.status == StepStatus.FAILED
        assert res.retry_count == 3
        assert "Portal selector timeout" in (res.error_message or "")

    asyncio.run(_test())


def test_checkpoint_save_and_restore(orchestrator):
    wf_id = "wf-onboarding-441"
    dom = "<div id='status'>Verified</div>"
    cp = orchestrator.save_checkpoint(
        workflow_id=wf_id,
        step_index=3,
        state=WorkflowState.IN_PROGRESS,
        payload={"worker_id": "W-1234", "doc_type": "national_id"},
        dom_content=dom,
    )
    assert cp.workflow_id == wf_id
    assert cp.current_step_index == 3
    assert cp.dom_snapshot_sha256 is not None

    restored = orchestrator.restore_from_checkpoint(wf_id)
    assert restored is not None
    assert restored.checkpoint_id == cp.checkpoint_id
    assert restored.state_payload["worker_id"] == "W-1234"


def test_document_compliance_verification_valid(orchestrator):
    extracted = {
        "full_name": "Jane Doe",
        "doc_number": "AB123456C",
        "expiry_date": "2030-01-01",
        "country": "IN",
    }
    required = ["full_name", "doc_number", "expiry_date", "country"]
    res = orchestrator.verify_document_compliance("W-991", "passport", extracted, required)
    assert res.verified is True
    assert res.confidence_score >= 0.9
    assert len(res.validation_flags) == 0
    assert len(res.audit_hash) == 64


def test_document_compliance_verification_missing_fields(orchestrator):
    extracted = {
        "full_name": "Jane Doe",
    }
    required = ["full_name", "doc_number", "expiry_date", "country"]
    res = orchestrator.verify_document_compliance("W-991", "passport", extracted, required)
    assert res.verified is False
    assert res.confidence_score < 0.85
    assert len(res.validation_flags) == 1
    assert "Missing mandatory fields" in res.validation_flags[0]
