"""Tests for FalMLInfraOrchestrator."""

import hashlib

import pytest

from llm_control_plane.fal_ml_infra_orchestrator import (
    CIFailureDiagnosis,
    FalMLInfraOrchestrator,
    ModelArtifactSpec,
    PerformanceBenchmarkResult,
    PricingTierSpec,
)


@pytest.fixture
def sample_weights() -> bytes:
    return b"dummy_weights_tensor_buffer_content_for_testing"


@pytest.fixture
def sample_weights_sha256(sample_weights: bytes) -> str:
    return hashlib.sha256(sample_weights).hexdigest()


@pytest.fixture
def valid_spec(sample_weights_sha256: str) -> ModelArtifactSpec:
    return ModelArtifactSpec(
        model_id="fal-ai/fast-sdxl",
        version="v1.4.0",
        artifact_format="safetensors",
        weights_sha256=sample_weights_sha256,
        input_schema={"prompt": "string", "steps": "integer"},
        output_schema={"images": "array", "seed": "integer"},
        target_gpu_architectures=["H100", "A100_80GB"],
        base_vram_requirement_gb=16.0,
    )


@pytest.fixture
def orchestrator() -> FalMLInfraOrchestrator:
    return FalMLInfraOrchestrator(max_allowed_regression_pct=10.0)


def test_model_artifact_spec_validation(sample_weights_sha256: str):
    with pytest.raises(ValueError, match="Unsupported artifact format"):
        ModelArtifactSpec(
            model_id="test",
            version="1.0",
            artifact_format="invalid_fmt",
            weights_sha256=sample_weights_sha256,
            input_schema={},
            output_schema={},
            target_gpu_architectures=["H100"],
            base_vram_requirement_gb=10.0,
        )

    with pytest.raises(ValueError, match="Invalid SHA256 checksum format"):
        ModelArtifactSpec(
            model_id="test",
            version="1.0",
            artifact_format="safetensors",
            weights_sha256="not_a_valid_hex",
            input_schema={},
            output_schema={},
            target_gpu_architectures=["H100"],
            base_vram_requirement_gb=10.0,
        )

    with pytest.raises(ValueError, match="base_vram_requirement_gb must be positive"):
        ModelArtifactSpec(
            model_id="test",
            version="1.0",
            artifact_format="safetensors",
            weights_sha256=sample_weights_sha256,
            input_schema={},
            output_schema={},
            target_gpu_architectures=["H100"],
            base_vram_requirement_gb=-5.0,
        )


def test_validate_model_artifact_success(
    orchestrator: FalMLInfraOrchestrator,
    valid_spec: ModelArtifactSpec,
    sample_weights: bytes,
):
    inputs = {"prompt": "cinematic portrait of an astronaut", "steps": 30}
    outputs = {"images": ["img_001.png"], "seed": 42}
    ref = {"images": ["img_ref.png"], "seed": 42}

    res = orchestrator.validate_model_artifact(
        spec=valid_spec,
        weights_bytes=sample_weights,
        sample_inputs=inputs,
        sample_outputs=outputs,
        reference_outputs=ref,
    )

    assert res.valid is True
    assert res.checksum_verified is True
    assert res.schema_compatibility is True
    assert res.quality_score >= 0.90
    assert len(res.rejection_reasons) == 0


def test_validate_model_artifact_failures(
    orchestrator: FalMLInfraOrchestrator,
    valid_spec: ModelArtifactSpec,
):
    corrupted_weights = b"corrupted"
    inputs = {"prompt": 12345}  # should be string, missing steps
    outputs = {"seed": 42}  # missing images
    ref = {"images": ["img.png"], "seed": 42}

    res = orchestrator.validate_model_artifact(
        spec=valid_spec,
        weights_bytes=corrupted_weights,
        sample_inputs=inputs,
        sample_outputs=outputs,
        reference_outputs=ref,
    )

    assert res.valid is False
    assert res.checksum_verified is False
    assert res.schema_compatibility is False
    assert any("Checksum mismatch" in r for r in res.rejection_reasons)
    assert any("Missing required input parameter" in r for r in res.rejection_reasons)
    assert any("Parameter 'prompt' must be string" in r for r in res.rejection_reasons)
    assert any("Missing expected output field" in r for r in res.rejection_reasons)


def test_validate_model_artifact_quality_drop(
    orchestrator: FalMLInfraOrchestrator,
    valid_spec: ModelArtifactSpec,
    sample_weights: bytes,
):
    inputs = {"prompt": "test prompt", "steps": 20}
    outputs = {"images": [], "seed": 100}
    ref = {"images": ["a", "b"], "seed": 9999}

    res = orchestrator.validate_model_artifact(
        spec=valid_spec,
        weights_bytes=sample_weights,
        sample_inputs=inputs,
        sample_outputs=outputs,
        reference_outputs=ref,
    )

    assert res.checksum_verified is True
    assert res.schema_compatibility is True
    assert res.quality_score < 0.90
    assert res.valid is False
    assert any("Quality score" in r for r in res.rejection_reasons)


def test_performance_benchmark_pass(
    orchestrator: FalMLInfraOrchestrator,
    valid_spec: ModelArtifactSpec,
):
    latencies = [45.0, 48.0, 50.0, 52.0, 55.0, 58.0, 60.0, 62.0, 65.0, 70.0]
    vram = [22.0, 24.5, 25.0, 26.0, 25.5]

    res = orchestrator.run_performance_benchmark(
        spec=valid_spec,
        latency_samples_ms=latencies,
        vram_samples_gb=vram,
        baseline_p90_ms=64.0,
        max_allowed_p90_ms=100.0,
        concurrency_level=8,
    )

    assert res.passed_slo is True
    assert res.latency_p50_ms == 58.0
    assert res.latency_p90_ms == 70.0
    assert res.peak_vram_used_gb == 26.0
    assert res.throughput_rps > 0
    assert 0.0 <= res.oom_risk_score <= 1.0


def test_performance_benchmark_regression_failure(
    orchestrator: FalMLInfraOrchestrator,
    valid_spec: ModelArtifactSpec,
):
    latencies = [120.0, 130.0, 140.0, 150.0, 160.0]
    vram = [75.0, 78.0]

    res = orchestrator.run_performance_benchmark(
        spec=valid_spec,
        latency_samples_ms=latencies,
        vram_samples_gb=vram,
        baseline_p90_ms=100.0,
        max_allowed_p90_ms=120.0,
    )

    assert res.passed_slo is False
    assert res.latency_regression_pct == 60.0  # (160 - 100) / 100 * 100


def test_performance_benchmark_empty_validation(
    orchestrator: FalMLInfraOrchestrator,
    valid_spec: ModelArtifactSpec,
):
    with pytest.raises(ValueError, match="latency_samples_ms cannot be empty"):
        orchestrator.run_performance_benchmark(
            valid_spec, [], [10.0], 50.0, 100.0
        )

    with pytest.raises(ValueError, match="vram_samples_gb cannot be empty"):
        orchestrator.run_performance_benchmark(
            valid_spec, [50.0], [], 50.0, 100.0
        )


def test_pricing_tier_effective_margin():
    pricing = PricingTierSpec(
        gpu_tier="H100",
        compute_cost_per_second_usd=0.0035,
        target_margin_pct=60.0,
        price_per_megapixel_usd=0.005,
        price_per_request_usd=0.012,
    )

    # 1.0 second duration: cost = 0.0035, price = 0.012 -> margin = (0.012 - 0.0035) / 0.012 = ~70.8%
    margin = pricing.effective_margin(avg_duration_sec=1.0)
    assert round(margin, 3) == 0.708

    # Zero price edge case
    free_tier = PricingTierSpec(
        gpu_tier="L40S",
        compute_cost_per_second_usd=0.001,
        target_margin_pct=50.0,
        price_per_megapixel_usd=0.0,
        price_per_request_usd=0.0,
    )
    assert free_tier.effective_margin(1.0) == 0.0


def test_verify_pricing_and_deployment(
    orchestrator: FalMLInfraOrchestrator,
    valid_spec: ModelArtifactSpec,
):
    pricing = PricingTierSpec(
        gpu_tier="H100",
        compute_cost_per_second_usd=0.003,
        target_margin_pct=50.0,
        price_per_megapixel_usd=0.005,
        price_per_request_usd=0.010,
    )

    good_benchmark = PerformanceBenchmarkResult(
        concurrency_level=8,
        latency_p50_ms=40.0,
        latency_p90_ms=55.0,
        latency_p99_ms=65.0,
        throughput_rps=120.0,
        peak_vram_used_gb=24.0,
        oom_risk_score=0.15,
        latency_regression_pct=2.0,
        passed_slo=True,
    )

    ok, msg = orchestrator.verify_pricing_and_deployment(
        valid_spec, good_benchmark, pricing, avg_duration_sec=0.5
    )
    assert ok is True
    assert "Pre-flight approved" in msg

    # Failed SLO benchmark
    bad_benchmark = PerformanceBenchmarkResult(
        concurrency_level=8,
        latency_p50_ms=120.0,
        latency_p90_ms=180.0,
        latency_p99_ms=220.0,
        throughput_rps=40.0,
        peak_vram_used_gb=79.0,
        oom_risk_score=0.98,
        latency_regression_pct=45.0,
        passed_slo=False,
    )
    ok_bad, msg_bad = orchestrator.verify_pricing_and_deployment(
        valid_spec, bad_benchmark, pricing
    )
    assert ok_bad is False
    assert "benchmark failed SLO" in msg_bad

    # High compute duration leading to negative / low margin
    ok_margin, msg_margin = orchestrator.verify_pricing_and_deployment(
        valid_spec, good_benchmark, pricing, avg_duration_sec=3.0
    )
    assert ok_margin is False
    assert "below target margin" in msg_margin


def test_diagnose_ci_pipeline_failure(orchestrator: FalMLInfraOrchestrator):
    diag_oom = orchestrator.diagnose_ci_pipeline_failure(
        "Traceback: torch.cuda.OutOfMemoryError: Tried to allocate 4.20 GiB on GPU 0"
    )
    assert diag_oom.failure_type == "CUDA_OOM"
    assert diag_oom.auto_fixable is True
    assert "enable_activation_checkpointing" in diag_oom.remediation_patch

    diag_hash = orchestrator.diagnose_ci_pipeline_failure(
        "Fatal: WeightsHashError detected in pipeline"
    )
    assert diag_hash.failure_type == "WEIGHT_HASH_MISMATCH"
    assert diag_hash.auto_fixable is True

    diag_lat = orchestrator.diagnose_ci_pipeline_failure(
        "Failed: P90 latency exceeded threshold during load test"
    )
    assert diag_lat.failure_type == "LATENCY_REGRESSION"
    assert diag_lat.auto_fixable is True

    diag_unk = orchestrator.diagnose_ci_pipeline_failure("Random unexpected error in runner")
    assert diag_unk.failure_type == "UNKNOWN_BUILD_FAILURE"
    assert diag_unk.auto_fixable is False


def test_execute_agentic_ci_remediation(orchestrator: FalMLInfraOrchestrator):
    diag = CIFailureDiagnosis(
        failure_type="CUDA_OOM",
        culprit_stage="test_model_inference",
        log_excerpt="OOM error",
        proposed_remediation="Enable activation checkpointing",
        auto_fixable=True,
        remediation_patch={"max_batch_size": 4, "torch_dtype": "bfloat16"},
    )

    base_config = {"model": "fast-sdxl", "max_batch_size": 16}
    updated = orchestrator.execute_agentic_ci_remediation(diag, base_config)

    assert updated["max_batch_size"] == 4
    assert updated["torch_dtype"] == "bfloat16"
    assert updated["_remediation_applied"] == "CUDA_OOM"
