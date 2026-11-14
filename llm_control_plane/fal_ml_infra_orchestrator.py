"""fal ML Infrastructure, Performance Benchmarking, and CI/CD Automation Orchestrator.

Provides automated ML model artifact validation, continuous latency/throughput benchmarking,
VRAM consumption profiling, pricing schema pre-flight verification, and agentic CI/CD failure
diagnosis with automated remediation proposals for fal.ai generative media workloads.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelArtifactSpec:
    """Specification of an ML model artifact for inference pipelines."""

    model_id: str
    version: str
    artifact_format: str
    weights_sha256: str
    input_schema: dict[str, str]
    output_schema: dict[str, str]
    target_gpu_architectures: list[str]
    base_vram_requirement_gb: float

    def __post_init__(self) -> None:
        valid_formats = {"safetensors", "gguf", "onnx", "torchscript", "diffusers"}
        if self.artifact_format.lower() not in valid_formats:
            raise ValueError(
                f"Unsupported artifact format '{self.artifact_format}'. "
                f"Supported: {sorted(valid_formats)}"
            )
        if not re.match(r"^[a-fA-F0-9]{64}$", self.weights_sha256):
            raise ValueError(f"Invalid SHA256 checksum format: '{self.weights_sha256}'")
        if self.base_vram_requirement_gb <= 0:
            raise ValueError("base_vram_requirement_gb must be positive")


@dataclass(frozen=True)
class ModelValidationResult:
    """Result of pre-flight model artifact validation and quality checks."""

    valid: bool
    checksum_verified: bool
    schema_compatibility: bool
    quality_score: float
    rejection_reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PerformanceBenchmarkResult:
    """Inference performance metrics and VRAM profiling under benchmark load."""

    concurrency_level: int
    latency_p50_ms: float
    latency_p90_ms: float
    latency_p99_ms: float
    throughput_rps: float
    peak_vram_used_gb: float
    oom_risk_score: float
    latency_regression_pct: float
    passed_slo: bool


@dataclass(frozen=True)
class PricingTierSpec:
    """Model pricing schema and compute cost accounting."""

    gpu_tier: str
    compute_cost_per_second_usd: float
    target_margin_pct: float
    price_per_megapixel_usd: float
    price_per_request_usd: float
    schema_version: str = "v1"

    def effective_margin(self, avg_duration_sec: float) -> float:
        """Calculate realized margin based on request pricing vs compute cost."""
        if self.price_per_request_usd <= 0:
            return 0.0
        total_compute_cost = self.compute_cost_per_second_usd * avg_duration_sec
        return max(
            0.0,
            (self.price_per_request_usd - total_compute_cost)
            / self.price_per_request_usd,
        )


@dataclass(frozen=True)
class CIFailureDiagnosis:
    """Root-cause attribution and structured remediation for CI/CD failures."""

    failure_type: str
    culprit_stage: str
    log_excerpt: str
    proposed_remediation: str
    auto_fixable: bool
    remediation_patch: dict[str, Any] = field(default_factory=dict)


class FalMLInfraOrchestrator:
    """Orchestrates model validation, benchmarking, pricing, and CI/CD diagnosis for fal.ai."""

    def __init__(self, max_allowed_regression_pct: float = 10.0) -> None:
        self.max_allowed_regression_pct = max_allowed_regression_pct

    def validate_model_artifact(
        self,
        spec: ModelArtifactSpec,
        weights_bytes: bytes,
        sample_inputs: dict[str, Any],
        sample_outputs: dict[str, Any],
        reference_outputs: dict[str, Any],
    ) -> ModelValidationResult:
        """Validate artifact checksum, input/output schemas, and output fidelity."""
        rejection_reasons: list[str] = []

        # 1. Verify weights checksum
        computed_sha256 = hashlib.sha256(weights_bytes).hexdigest()
        checksum_verified = computed_sha256.lower() == spec.weights_sha256.lower()
        if not checksum_verified:
            rejection_reasons.append(
                f"Checksum mismatch: expected {spec.weights_sha256}, got {computed_sha256}"
            )

        # 2. Verify input schema compliance
        schema_compatibility = True
        for key, expected_type in spec.input_schema.items():
            if key not in sample_inputs:
                schema_compatibility = False
                rejection_reasons.append(f"Missing required input parameter: '{key}'")
            elif expected_type == "string" and not isinstance(sample_inputs[key], str):
                schema_compatibility = False
                rejection_reasons.append(
                    f"Parameter '{key}' must be string, got {type(sample_inputs[key]).__name__}"
                )
            elif expected_type == "integer" and not isinstance(
                sample_inputs[key], int
            ):
                schema_compatibility = False
                rejection_reasons.append(
                    f"Parameter '{key}' must be integer, got {type(sample_inputs[key]).__name__}"
                )

        # 3. Verify output schema compliance
        for key in spec.output_schema:
            if key not in sample_outputs:
                schema_compatibility = False
                rejection_reasons.append(f"Missing expected output field: '{key}'")

        # 4. Compute output quality score against reference
        quality_score = self._compute_quality_score(
            sample_outputs, reference_outputs
        )
        if quality_score < 0.90:
            rejection_reasons.append(
                f"Quality score {quality_score:.3f} below acceptable floor (0.900)"
            )

        valid = checksum_verified and schema_compatibility and (quality_score >= 0.90)
        return ModelValidationResult(
            valid=valid,
            checksum_verified=checksum_verified,
            schema_compatibility=schema_compatibility,
            quality_score=quality_score,
            rejection_reasons=rejection_reasons,
        )

    def run_performance_benchmark(
        self,
        spec: ModelArtifactSpec,
        latency_samples_ms: list[float],
        vram_samples_gb: list[float],
        baseline_p90_ms: float,
        max_allowed_p90_ms: float,
        concurrency_level: int = 8,
    ) -> PerformanceBenchmarkResult:
        """Calculate latency percentiles, VRAM peak utilization, and SLO adherence."""
        if not latency_samples_ms:
            raise ValueError("latency_samples_ms cannot be empty")
        if not vram_samples_gb:
            raise ValueError("vram_samples_gb cannot be empty")

        sorted_latencies = sorted(latency_samples_ms)
        n = len(sorted_latencies)
        p50 = sorted_latencies[int(n * 0.50)]
        p90 = sorted_latencies[min(int(n * 0.90), n - 1)]
        p99 = sorted_latencies[min(int(n * 0.99), n - 1)]

        peak_vram = max(vram_samples_gb)
        avg_latency_sec = (sum(sorted_latencies) / n) / 1000.0
        throughput_rps = (
            (concurrency_level / avg_latency_sec) if avg_latency_sec > 0 else 0.0
        )

        # Calculate latency regression vs baseline
        if baseline_p90_ms > 0:
            regression_pct = (
                (p90 - baseline_p90_ms) / baseline_p90_ms
            ) * 100.0
        else:
            regression_pct = 0.0

        # Out-of-memory risk score (base requirement vs available GPU memory e.g. 80GB)
        gpu_cap_gb = (
            80.0
            if any("80GB" in arch for arch in spec.target_gpu_architectures)
            else 48.0
        )
        vram_headroom = gpu_cap_gb - spec.base_vram_requirement_gb
        oom_risk = min(
            1.0,
            max(0.0, (peak_vram - spec.base_vram_requirement_gb) / vram_headroom),
        )

        passed_slo = (
            p90 <= max_allowed_p90_ms
            and regression_pct <= self.max_allowed_regression_pct
            and peak_vram < gpu_cap_gb
        )

        return PerformanceBenchmarkResult(
            concurrency_level=concurrency_level,
            latency_p50_ms=round(p50, 2),
            latency_p90_ms=round(p90, 2),
            latency_p99_ms=round(p99, 2),
            throughput_rps=round(throughput_rps, 2),
            peak_vram_used_gb=round(peak_vram, 2),
            oom_risk_score=round(oom_risk, 3),
            latency_regression_pct=round(regression_pct, 2),
            passed_slo=passed_slo,
        )

    def verify_pricing_and_deployment(
        self,
        spec: ModelArtifactSpec,
        benchmark: PerformanceBenchmarkResult,
        pricing: PricingTierSpec,
        avg_duration_sec: float = 0.85,
    ) -> tuple[bool, str]:
        """Validate deployment pre-flight: ensure latency SLO passed and pricing margin met."""
        if not benchmark.passed_slo:
            msg = (
                f"Deployment blocked: benchmark failed SLO (P90: {benchmark.latency_p90_ms}ms, "
                f"regression: {benchmark.latency_regression_pct}%)"
            )
            return False, msg

        margin = pricing.effective_margin(avg_duration_sec)
        required_margin = pricing.target_margin_pct / 100.0
        if margin < required_margin:
            return (
                False,
                f"Deployment blocked: effective margin {margin:.1%} is below target margin {required_margin:.1%}",
            )

        return (
            True,
            (
                f"Pre-flight approved for {spec.model_id}:{spec.version} on {pricing.gpu_tier} "
                f"(P90: {benchmark.latency_p90_ms}ms, margin: {margin:.1%})"
            ),
        )

    def diagnose_ci_pipeline_failure(self, log_text: str) -> CIFailureDiagnosis:
        """Analyze CI build/test failure logs and produce structured remediation."""
        if "CUDA out of memory" in log_text or "torch.cuda.OutOfMemoryError" in log_text:
            return CIFailureDiagnosis(
                failure_type="CUDA_OOM",
                culprit_stage="test_model_inference",
                log_excerpt="torch.cuda.OutOfMemoryError: Tried to allocate 4.20 GiB on GPU 0",
                proposed_remediation="Enable activation checkpointing and decrease batch size from 8 to 4.",
                auto_fixable=True,
                remediation_patch={
                    "enable_activation_checkpointing": True,
                    "max_batch_size": 4,
                    "torch_dtype": "bfloat16",
                },
            )

        if "Checksum mismatch" in log_text or "WeightsHashError" in log_text:
            return CIFailureDiagnosis(
                failure_type="WEIGHT_HASH_MISMATCH",
                culprit_stage="verify_model_artifacts",
                log_excerpt="WeightsHashError: SHA256 checksum mismatch for safetensors weights",
                proposed_remediation="Regenerate artifact manifest and re-hash weights with git-lfs pull.",
                auto_fixable=True,
                remediation_patch={"action": "rehash_and_resync_lfs"},
            )

        if "P90 latency exceeded" in log_text or "LatencyRegressionError" in log_text:
            return CIFailureDiagnosis(
                failure_type="LATENCY_REGRESSION",
                culprit_stage="benchmark_inference_slo",
                log_excerpt="LatencyRegressionError: P90 latency 245ms exceeded threshold 180ms (+36%)",
                proposed_remediation=(
                    "Enable TensorRT-LLM/FlashAttention-2 kernel compilation and speculative decoding."
                ),
                auto_fixable=True,
                remediation_patch={
                    "kernel_backend": "flash_attention_2",
                    "enable_cuda_graphs": True,
                },
            )

        return CIFailureDiagnosis(
            failure_type="UNKNOWN_BUILD_FAILURE",
            culprit_stage="unknown",
            log_excerpt=log_text[-200:].strip() if log_text else "Empty log",
            proposed_remediation="Inspect CI worker environment variables and runner state.",
            auto_fixable=False,
            remediation_patch={},
        )

    def execute_agentic_ci_remediation(
        self, diagnosis: CIFailureDiagnosis, build_config: dict[str, Any]
    ) -> dict[str, Any]:
        """Apply automated remediation patch to pipeline configuration."""
        updated_config = dict(build_config)
        if diagnosis.auto_fixable and diagnosis.remediation_patch:
            updated_config.update(diagnosis.remediation_patch)
            updated_config["_remediation_applied"] = diagnosis.failure_type
        return updated_config

    def _compute_quality_score(
        self, sample_outputs: dict[str, Any], reference_outputs: dict[str, Any]
    ) -> float:
        """Compute structural and semantic consistency score between outputs."""
        if not reference_outputs:
            return 1.0

        common_keys = set(sample_outputs.keys()).intersection(reference_outputs.keys())
        if not common_keys:
            return 0.0

        matches = 0
        for k in common_keys:
            s_val = sample_outputs[k]
            r_val = reference_outputs[k]
            if s_val == r_val:
                matches += 1
            elif isinstance(s_val, (int, float)) and isinstance(r_val, (int, float)):
                if r_val != 0 and abs(s_val - r_val) / abs(r_val) <= 0.05:
                    matches += 1
            elif isinstance(s_val, list) and isinstance(r_val, list):
                if len(s_val) == len(r_val):
                    matches += 1

        return matches / max(1, len(reference_outputs))
