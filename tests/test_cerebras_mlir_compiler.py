"""Unit tests for Cerebras Wafer-Scale Engine (WSE) MLIR compiler & bring-up harness."""

import pytest

from llm_control_plane.cerebras_mlir_compiler import (
    CerebrasModelBringupHarness,
    GQALoweringPass,
    IRGraph,
    IROp,
    IRTensor,
    KVCacheLayoutOptimizer,
    MemoryHierarchy,
    MLIRDialect,
    NumericalCorrectnessVerifier,
    QuantizationLoweringPass,
    SwiGLUFusionPass,
    TensorDType,
    TensorLayout,
    WaferClusterProfiler,
)


def test_ir_tensor_properties():
    t_fp32 = IRTensor(name="act_fp32", shape=(2, 1024, 4096), dtype=TensorDType.FP32)
    assert t_fp32.element_count == 2 * 1024 * 4096
    assert t_fp32.byte_size == 2 * 1024 * 4096 * 4

    t_fp8 = IRTensor(name="act_fp8", shape=(2, 1024, 4096), dtype=TensorDType.FP8_E4M3)
    assert t_fp8.byte_size == 2 * 1024 * 4096 * 1

    t_int4 = IRTensor(name="w_int4", shape=(4096, 4096), dtype=TensorDType.INT4)
    assert t_int4.byte_size == 4096 * 4096 // 2
    assert t_int4.residence == MemoryHierarchy.ON_CHIP_SRAM


def test_mlir_dialect_registration_and_validation():
    dialect = MLIRDialect("cerebras_test", "Test Dialect")
    dialect.register_op("tiled_gemm", input_count=2)

    op_valid = IROp(
        op_id="%1",
        dialect="cerebras_test",
        op_name="tiled_gemm",
        inputs=[
            IRTensor(name="a", shape=(128, 128), dtype=TensorDType.BF16),
            IRTensor(name="b", shape=(128, 128), dtype=TensorDType.BF16),
        ],
        outputs=[IRTensor(name="c", shape=(128, 128), dtype=TensorDType.BF16)],
    )
    assert dialect.validate_op(op_valid) is True

    op_invalid_inputs = IROp(
        op_id="%2",
        dialect="cerebras_test",
        op_name="tiled_gemm",
        inputs=[IRTensor(name="a", shape=(128, 128), dtype=TensorDType.BF16)],
        outputs=[],
    )
    assert dialect.validate_op(op_invalid_inputs) is False

    op_wrong_dialect = IROp(
        op_id="%3",
        dialect="torch",
        op_name="tiled_gemm",
        inputs=op_valid.inputs,
        outputs=op_valid.outputs,
    )
    assert dialect.validate_op(op_wrong_dialect) is False


def test_swiglu_fusion_pass():
    graph = IRGraph("test_mlp")
    x = IRTensor(name="x", shape=(1, 512, 1024), dtype=TensorDType.BF16)
    gate = IRTensor(name="gate", shape=(1, 512, 2048), dtype=TensorDType.BF16)
    silu = IRTensor(name="silu", shape=(1, 512, 2048), dtype=TensorDType.BF16)
    up = IRTensor(name="up", shape=(1, 512, 2048), dtype=TensorDType.BF16)
    mul = IRTensor(name="mul", shape=(1, 512, 2048), dtype=TensorDType.BF16)

    graph.add_op("torch", "linear", [x], [gate], {"tag": "gate_proj", "in_features": 1024, "out_features": 2048})
    graph.add_op("torch", "silu", [gate], [silu])
    graph.add_op("torch", "linear", [x], [up], {"tag": "up_proj", "in_features": 1024, "out_features": 2048})
    graph.add_op("torch", "mul", [silu, up], [mul])

    assert len(graph.operations) == 4

    fuser = SwiGLUFusionPass()
    fused_graph, count = fuser.apply(graph)

    assert count == 1
    assert len(fused_graph.operations) == 1
    fused_op = fused_graph.operations[0]
    assert fused_op.op_name == "fused_swiglu"
    assert fused_op.dialect == "cerebras_wse"
    assert fused_op.attributes["fused"] is True


def test_gqa_lowering_pass():
    graph = IRGraph("test_attention")
    q = IRTensor(name="q", shape=(1, 2048, 32 * 128), dtype=TensorDType.BF16)
    k = IRTensor(name="k", shape=(1, 2048, 8 * 128), dtype=TensorDType.BF16)
    v = IRTensor(name="v", shape=(1, 2048, 8 * 128), dtype=TensorDType.BF16)
    out = IRTensor(name="attn_out", shape=(1, 2048, 32 * 128), dtype=TensorDType.BF16)

    graph.add_op("torch", "grouped_query_attention", [q, k, v], [out], {"num_q_heads": 32, "num_kv_heads": 8})

    gqa_pass = GQALoweringPass(num_query_heads=32, num_kv_heads=8, head_dim=128)
    lowered_graph, count = gqa_pass.apply(graph)

    assert count == 1
    assert len(lowered_graph.operations) == 2
    bcast_op, sdpa_op = lowered_graph.operations[0], lowered_graph.operations[1]
    assert bcast_op.op_name == "broadcast_kv_heads"
    assert bcast_op.attributes["group_size"] == 4
    assert sdpa_op.op_name == "tiled_sdpa_kernel"
    assert sdpa_op.attributes["head_dim"] == 128


def test_kv_cache_layout_optimizer():
    optimizer = KVCacheLayoutOptimizer(sram_budget_mb=40000.0)

    # Standard Llama 3 8B: 32 layers, 8 KV heads, 128 head dim
    plan = optimizer.optimize_cache_plan(
        batch_size=16,
        max_seq_len=4096,
        num_layers=32,
        num_kv_heads=8,
        head_dim=128,
        dtype=TensorDType.BF16,
    )

    assert plan["batch_size"] == 16
    assert plan["max_seq_len"] == 4096
    assert plan["fits_in_sram"] is True
    assert plan["sram_residence_pct"] > 0.0
    assert plan["recommended_layout"] == TensorLayout.TILED_WAFER_2D
    assert plan["total_cache_mb"] > 0.0


def test_quantization_lowering_pass():
    quant = QuantizationLoweringPass(target_dtype=TensorDType.FP8_E4M3)

    scale = quant.compute_scale_factor(max_abs_val=10.0)
    assert scale > 0.0

    quant_val = quant.quantize_value(val=3.14159, scale=scale)
    assert abs(quant_val - 3.14159) < 0.1

    graph = IRGraph("test_linear")
    x = IRTensor(name="in", shape=(1, 128, 512), dtype=TensorDType.BF16)
    out = IRTensor(name="out", shape=(1, 128, 512), dtype=TensorDType.BF16)
    graph.add_op("torch", "linear", [x], [out])

    lowered_graph, count = quant.apply(graph)
    assert count == 1
    assert lowered_graph.operations[0].inputs[0].dtype == TensorDType.FP8_E4M3


def test_numerical_correctness_verifier_pass():
    verifier = NumericalCorrectnessVerifier()
    ref = [1.0, 2.0, 3.0, 4.0, 5.0]
    lowered = [1.0001, 1.9999, 3.0002, 3.9998, 5.0001]

    result = verifier.verify_activation_drift(ref, lowered, max_allowed_drift=1e-3)
    assert result["passed"] is True
    assert result["status"] == "PASS"
    assert result["nan_count"] == 0
    assert result["max_abs_error"] <= 1e-3


def test_numerical_correctness_verifier_fail():
    verifier = NumericalCorrectnessVerifier()
    ref = [1.0, 2.0, 3.0]
    lowered_drift = [1.0, 2.5, 3.0]  # large error

    result = verifier.verify_activation_drift(ref, lowered_drift, max_allowed_drift=1e-3)
    assert result["passed"] is False
    assert result["status"] == "FAIL_PRECISION_REGRESSION"

    # Test NaN detection
    lowered_nan = [1.0, float("nan"), 3.0]
    result_nan = verifier.verify_activation_drift(ref, lowered_nan, max_allowed_drift=1e-3)
    assert result_nan["passed"] is False
    assert result_nan["nan_count"] == 1


def test_wafer_cluster_profiler():
    profiler = WaferClusterProfiler()
    graph = IRGraph("test_block")
    t_in = IRTensor(name="in", shape=(1, 2048, 4096), dtype=TensorDType.BF16)
    t_out = IRTensor(name="out", shape=(1, 2048, 4096), dtype=TensorDType.BF16)
    graph.add_op("cerebras_wse", "fused_swiglu", [t_in], [t_out], {"fused": True})

    metrics = profiler.profile_execution(graph, batch_size=1, seq_len=2048)
    assert metrics["total_operations"] == 1
    assert metrics["fused_operations"] == 1
    assert metrics["estimated_tokens_per_sec"] > 100.0
    assert metrics["core_utilization_pct"] >= 65.0
    assert "CS-3" in metrics["hardware_target"]


def test_end_to_end_model_bringup_harness():
    harness = CerebrasModelBringupHarness(model_name="llama-3-8b")
    block_graph = harness.build_transformer_block_ir(
        batch_size=1,
        seq_len=2048,
        hidden_dim=4096,
        intermediate_dim=14336,
        num_query_heads=32,
        num_kv_heads=8,
    )

    # Initial graph operations: GQA (1) + Gate linear (1) + Silu (1) + Up linear (1) + Mul (1) + Down linear (1) = 6 ops
    assert len(block_graph.operations) == 6

    compiled = harness.compile_and_optimize(block_graph, enable_fp8=True)

    assert compiled["model_name"] == "llama-3-8b"
    assert compiled["fused_swiglu_ops"] == 1
    assert compiled["gqa_lowered_ops"] == 1
    assert compiled["quantized_tensors"] > 0
    assert compiled["profile"]["estimated_tokens_per_sec"] > 500.0
    assert compiled["profile"]["core_utilization_pct"] >= 65.0


def test_verifier_shape_mismatch():
    verifier = NumericalCorrectnessVerifier()
    with pytest.raises(ValueError, match="Shape mismatch"):
        verifier.verify_activation_drift([1.0, 2.0], [1.0])
