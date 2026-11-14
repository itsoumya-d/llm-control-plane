"""Cerebras Wafer-Scale Engine (WSE) MLIR Compiler & Model Bring-Up Pipeline.

Provides MLIR-style intermediate representation (IR) graph construction,
dialects, operator rewrite patterns (SwiGLU, LayerNorm, GQA), KV-cache SRAM layout
optimization, FP8 quantization passes, and numerical correctness validation.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


class TensorDType(str, enum.Enum):
    """Supported tensor data types across Cerebras WSE compiler passes."""

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"
    INT8 = "int8"
    INT4 = "int4"


class TensorLayout(str, enum.Enum):
    """Hardware layout representations for memory alignment and tiling."""

    ROW_MAJOR = "row_major"
    COL_MAJOR = "col_major"
    TILED_WAFER_2D = "tiled_wafer_2d"
    FUSED_ATTENTION_PACKED = "fused_attention_packed"


class MemoryHierarchy(str, enum.Enum):
    """Memory residence tiers across wafer-scale architecture."""

    ON_CHIP_SRAM = "on_chip_sram"
    EXTERNAL_HBM = "external_hbm"
    FABRIC_TRANSIT = "fabric_transit"


@dataclass(frozen=True)
class IRTensor:
    """Representation of an SSA tensor operand within the MLIR graph."""

    name: str
    shape: Tuple[int, ...]
    dtype: TensorDType
    layout: TensorLayout = TensorLayout.ROW_MAJOR
    residence: MemoryHierarchy = MemoryHierarchy.ON_CHIP_SRAM

    @property
    def element_count(self) -> int:
        count = 1
        for dim in self.shape:
            count *= dim
        return count

    @property
    def byte_size(self) -> int:
        bytes_per_elem = {
            TensorDType.FP32: 4,
            TensorDType.FP16: 2,
            TensorDType.BF16: 2,
            TensorDType.FP8_E4M3: 1,
            TensorDType.FP8_E5M2: 1,
            TensorDType.INT8: 1,
            TensorDType.INT4: 0.5,
        }
        return int(math.ceil(self.element_count * bytes_per_elem[self.dtype]))


@dataclass
class IROp:
    """Representation of an MLIR operation node in the computational graph."""

    op_id: str
    dialect: str
    op_name: str
    inputs: List[IRTensor]
    outputs: List[IRTensor]
    attributes: Dict[str, Any] = field(default_factory=dict)

    @property
    def qualified_name(self) -> str:
        return f"{self.dialect}.{self.op_name}"


class MLIRDialect:
    """Definition of an MLIR dialect with registered operations and validation rules."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.registered_ops: Dict[str, Dict[str, Any]] = {}

    def register_op(self, op_name: str, input_count: Optional[int] = None, schema: Optional[Dict[str, Any]] = None):
        self.registered_ops[op_name] = {
            "input_count": input_count,
            "schema": schema or {},
        }

    def validate_op(self, op: IROp) -> bool:
        if op.dialect != self.name:
            return False
        if op.op_name not in self.registered_ops:
            return False
        expected_inputs = self.registered_ops[op.op_name]["input_count"]
        if expected_inputs is not None and len(op.inputs) != expected_inputs:
            return False
        return True


class IRGraph:
    """Computational graph representing a model module in the Cerebras compiler."""

    def __init__(self, name: str):
        self.name = name
        self.operations: List[IROp] = []
        self._op_counter = 0

    def add_op(
        self,
        dialect: str,
        op_name: str,
        inputs: Sequence[IRTensor],
        outputs: Sequence[IRTensor],
        attributes: Optional[Dict[str, Any]] = None,
    ) -> IROp:
        self._op_counter += 1
        op_id = f"%op_{self._op_counter}_{dialect}_{op_name}"
        op = IROp(
            op_id=op_id,
            dialect=dialect,
            op_name=op_name,
            inputs=list(inputs),
            outputs=list(outputs),
            attributes=attributes or {},
        )
        self.operations.append(op)
        return op

    def find_ops(self, dialect: Optional[str] = None, op_name: Optional[str] = None) -> List[IROp]:
        results = []
        for op in self.operations:
            if dialect and op.dialect != dialect:
                continue
            if op_name and op.op_name != op_name:
                continue
            results.append(op)
        return results

    def clone(self) -> IRGraph:
        new_graph = IRGraph(self.name)
        new_graph._op_counter = self._op_counter
        new_graph.operations = [
            IROp(
                op_id=op.op_id,
                dialect=op.dialect,
                op_name=op.op_name,
                inputs=list(op.inputs),
                outputs=list(op.outputs),
                attributes=dict(op.attributes),
            )
            for op in self.operations
        ]
        return new_graph


class SwiGLUFusionPass:
    """Fuses silu(gate_proj(x)) * up_proj(x) into a single fused tile operator."""

    def __init__(self, target_dialect: str = "cerebras_wse"):
        self.target_dialect = target_dialect

    def apply(self, graph: IRGraph) -> Tuple[IRGraph, int]:
        new_graph = IRGraph(f"{graph.name}_fused_swiglu")
        fused_count = 0
        i = 0
        ops = graph.operations

        while i < len(ops):
            # Check for pattern: linear (gate) -> silu -> linear (up) -> mul
            matched = False
            if i + 3 < len(ops):
                op0, op1, op2, op3 = ops[i], ops[i + 1], ops[i + 2], ops[i + 3]
                if (
                    op0.op_name == "linear"
                    and op0.attributes.get("tag") == "gate_proj"
                    and op1.op_name == "silu"
                    and op2.op_name == "linear"
                    and op2.attributes.get("tag") == "up_proj"
                    and op3.op_name == "mul"
                ):
                    # Fuse into cerebras_wse.fused_swiglu
                    fused_output = op3.outputs[0]
                    fused_op = IROp(
                        op_id=f"%fused_swiglu_{fused_count + 1}",
                        dialect=self.target_dialect,
                        op_name="fused_swiglu",
                        inputs=[op0.inputs[0]],  # common input x
                        outputs=[fused_output],
                        attributes={
                            "hidden_dim": op0.attributes.get("out_features"),
                            "intermediate_dim": op0.attributes.get("in_features"),
                            "fused": True,
                        },
                    )
                    new_graph.operations.append(fused_op)
                    fused_count += 1
                    i += 4
                    matched = True

            if not matched:
                new_graph.operations.append(ops[i])
                i += 1

        return new_graph, fused_count


class GQALoweringPass:
    """Lowers Grouped-Query Attention (GQA) to parallel key-value head broadcasts."""

    def __init__(self, num_query_heads: int, num_kv_heads: int, head_dim: int):
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.group_size = num_query_heads // num_kv_heads

    def apply(self, graph: IRGraph) -> Tuple[IRGraph, int]:
        new_graph = IRGraph(f"{graph.name}_lowered_gqa")
        lowered_count = 0

        for op in graph.operations:
            if op.op_name == "grouped_query_attention":
                # Lower to 1) KV-broadcast over group_size, 2) Flash-attention tiled kernel
                q_tensor, k_tensor, v_tensor = op.inputs[0], op.inputs[1], op.inputs[2]
                out_tensor = op.outputs[0]

                # Broadcast op
                bcast_k = IRTensor(
                    name=f"{k_tensor.name}_bcast",
                    shape=q_tensor.shape,
                    dtype=k_tensor.dtype,
                    layout=TensorLayout.TILED_WAFER_2D,
                )
                bcast_v = IRTensor(
                    name=f"{v_tensor.name}_bcast",
                    shape=q_tensor.shape,
                    dtype=v_tensor.dtype,
                    layout=TensorLayout.TILED_WAFER_2D,
                )

                bcast_op = IROp(
                    op_id=f"%bcast_kv_{lowered_count}",
                    dialect="cerebras_wse",
                    op_name="broadcast_kv_heads",
                    inputs=[k_tensor, v_tensor],
                    outputs=[bcast_k, bcast_v],
                    attributes={"group_size": self.group_size, "num_kv_heads": self.num_kv_heads},
                )
                kernel_op = IROp(
                    op_id=f"%sdpa_wse_{lowered_count}",
                    dialect="cerebras_wse",
                    op_name="tiled_sdpa_kernel",
                    inputs=[q_tensor, bcast_k, bcast_v],
                    outputs=[out_tensor],
                    attributes={"head_dim": self.head_dim, "num_q_heads": self.num_query_heads},
                )
                new_graph.operations.append(bcast_op)
                new_graph.operations.append(kernel_op)
                lowered_count += 1
            else:
                new_graph.operations.append(op)

        return new_graph, lowered_count


class KVCacheLayoutOptimizer:
    """Plans on-chip SRAM vs external memory allocation for continuous batching."""

    def __init__(self, sram_budget_mb: float = 40000.0):
        self.sram_budget_bytes = int(sram_budget_mb * 1024 * 1024)

    def optimize_cache_plan(
        self,
        batch_size: int,
        max_seq_len: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: TensorDType = TensorDType.BF16,
    ) -> Dict[str, Any]:
        bytes_per_elem = 2 if dtype in (TensorDType.FP16, TensorDType.BF16) else 1
        # K and V tensors per token
        bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * bytes_per_elem
        total_cache_bytes = batch_size * max_seq_len * bytes_per_token

        fits_in_sram = total_cache_bytes <= self.sram_budget_bytes
        sram_residence_pct = min(100.0, (self.sram_budget_bytes / max(1, total_cache_bytes)) * 100.0)

        # Tile partitioning across wafer mesh
        wafer_cores = 900000
        tokens_per_core = max(1, (batch_size * max_seq_len) // wafer_cores)

        return {
            "batch_size": batch_size,
            "max_seq_len": max_seq_len,
            "num_layers": num_layers,
            "bytes_per_token": bytes_per_token,
            "total_cache_bytes": total_cache_bytes,
            "total_cache_mb": round(total_cache_bytes / (1024 * 1024), 2),
            "fits_in_sram": fits_in_sram,
            "sram_residence_pct": round(sram_residence_pct, 2),
            "tokens_per_core": tokens_per_core,
            "recommended_layout": TensorLayout.TILED_WAFER_2D if fits_in_sram else TensorLayout.ROW_MAJOR,
        }


class QuantizationLoweringPass:
    """Lowers FP32/BF16 weights and activations to FP8 with scale factor calibration."""

    def __init__(self, target_dtype: TensorDType = TensorDType.FP8_E4M3):
        self.target_dtype = target_dtype

    def compute_scale_factor(self, max_abs_val: float) -> float:
        # Max representable in E4M3 is 448.0
        max_fp8 = 448.0 if self.target_dtype == TensorDType.FP8_E4M3 else 57344.0
        if max_abs_val <= 1e-8:
            return 1.0
        return max_fp8 / max_abs_val

    def quantize_value(self, val: float, scale: float) -> float:
        scaled = val * scale
        max_fp8 = 448.0 if self.target_dtype == TensorDType.FP8_E4M3 else 57344.0
        clamped = max(-max_fp8, min(max_fp8, scaled))
        # simulate truncation to 8-bit precision
        return round(clamped) / scale

    def apply(self, graph: IRGraph) -> Tuple[IRGraph, int]:
        new_graph = IRGraph(f"{graph.name}_quantized_fp8")
        quant_count = 0

        for op in graph.operations:
            new_inputs = []
            for inp in op.inputs:
                if inp.dtype in (TensorDType.FP32, TensorDType.BF16):
                    quant_tensor = IRTensor(
                        name=f"{inp.name}_fp8",
                        shape=inp.shape,
                        dtype=self.target_dtype,
                        layout=inp.layout,
                        residence=inp.residence,
                    )
                    new_inputs.append(quant_tensor)
                    quant_count += 1
                else:
                    new_inputs.append(inp)

            new_op = IROp(
                op_id=op.op_id,
                dialect=op.dialect,
                op_name=op.op_name,
                inputs=new_inputs,
                outputs=op.outputs,
                attributes=dict(op.attributes, quantized=True, target_dtype=self.target_dtype.value),
            )
            new_graph.operations.append(new_op)

        return new_graph, quant_count


class NumericalCorrectnessVerifier:
    """Validates lowered MLIR output against reference PyTorch eager execution."""

    def verify_activation_drift(
        self,
        reference_vals: Sequence[float],
        lowered_vals: Sequence[float],
        max_allowed_drift: float = 1e-3,
    ) -> Dict[str, Any]:
        if len(reference_vals) != len(lowered_vals):
            raise ValueError(f"Shape mismatch: {len(reference_vals)} != {len(lowered_vals)}")

        max_err = 0.0
        sq_err_sum = 0.0
        nan_count = 0
        n = len(reference_vals)

        for ref_val, lowered_val in zip(reference_vals, lowered_vals):
            if math.isnan(lowered_val) or math.isinf(lowered_val):
                nan_count += 1
                continue
            err = abs(ref_val - lowered_val)
            if err > max_err:
                max_err = err
            sq_err_sum += err * err

        rmse = math.sqrt(sq_err_sum / max(1, n))
        passed = (max_err <= max_allowed_drift) and (nan_count == 0)

        return {
            "passed": passed,
            "max_abs_error": round(max_err, 6),
            "rmse": round(rmse, 6),
            "nan_count": nan_count,
            "tolerance": max_allowed_drift,
            "status": "PASS" if passed else "FAIL_PRECISION_REGRESSION",
        }


class WaferClusterProfiler:
    """Profiles execution time, compute tile utilization %, and fabric message latency."""

    def profile_execution(
        self,
        graph: IRGraph,
        batch_size: int,
        seq_len: int,
        clock_ghz: float = 1.1,
    ) -> Dict[str, Any]:
        total_flops = 0
        total_bytes = 0
        fused_op_count = 0

        for op in graph.operations:
            if op.op_name in ("linear", "fused_swiglu", "tiled_sdpa_kernel"):
                # Approximate matrix multiplication FLOPs
                in_elements = op.inputs[0].element_count if op.inputs else 0
                out_elements = op.outputs[0].element_count if op.outputs else 0
                total_flops += 2 * in_elements * (out_elements // max(1, batch_size * seq_len))
            for inp in op.inputs:
                total_bytes += inp.byte_size
            if op.attributes.get("fused", False):
                fused_op_count += 1

        # Cerebras WSE-3 peak performance ~ 125 PFLOPS FP16/BF16
        peak_tflops = 125000.0
        actual_tflops = total_flops / 1e12
        exec_latency_ms = max(0.01, (actual_tflops / peak_tflops) * 1000.0 * 1.25)
        tokens_per_sec = (batch_size * seq_len) / (exec_latency_ms / 1000.0)

        # Core utilization model
        core_utilization_pct = min(98.5, max(65.0, (fused_op_count / max(1, len(graph.operations))) * 100.0 + 35.0))

        return {
            "graph_name": graph.name,
            "total_operations": len(graph.operations),
            "fused_operations": fused_op_count,
            "total_gflops": round(total_flops / 1e9, 2),
            "memory_traffic_mb": round(total_bytes / (1024 * 1024), 2),
            "estimated_latency_ms": round(exec_latency_ms, 3),
            "estimated_tokens_per_sec": round(tokens_per_sec, 1),
            "core_utilization_pct": round(core_utilization_pct, 1),
            "hardware_target": "Cerebras CS-3 (Wafer-Scale Engine 3)",
        }


class CerebrasModelBringupHarness:
    """End-to-end driver: builds model IR, applies lowering passes, and validates output."""

    def __init__(self, model_name: str = "llama-3-8b"):
        self.model_name = model_name
        self.wse_dialect = MLIRDialect("cerebras_wse", "Cerebras Wafer-Scale Engine Dialect")
        self.wse_dialect.register_op("fused_swiglu", input_count=1)
        self.wse_dialect.register_op("tiled_sdpa_kernel", input_count=3)
        self.wse_dialect.register_op("broadcast_kv_heads", input_count=2)

    def build_transformer_block_ir(
        self,
        batch_size: int = 1,
        seq_len: int = 2048,
        hidden_dim: int = 4096,
        intermediate_dim: int = 14336,
        num_query_heads: int = 32,
        num_kv_heads: int = 8,
    ) -> IRGraph:
        graph = IRGraph(f"{self.model_name}_block_0")

        # Input token embedding
        _x = IRTensor(name="input_emb", shape=(batch_size, seq_len, hidden_dim), dtype=TensorDType.BF16)

        # Attention block (GQA)
        q = IRTensor(name="q_proj", shape=(batch_size, seq_len, hidden_dim), dtype=TensorDType.BF16)
        k = IRTensor(
            name="k_proj",
            shape=(batch_size, seq_len, (num_kv_heads * (hidden_dim // num_query_heads))),
            dtype=TensorDType.BF16,
        )
        v = IRTensor(
            name="v_proj",
            shape=(batch_size, seq_len, (num_kv_heads * (hidden_dim // num_query_heads))),
            dtype=TensorDType.BF16,
        )
        attn_out = IRTensor(name="attn_out", shape=(batch_size, seq_len, hidden_dim), dtype=TensorDType.BF16)

        graph.add_op(
            "torch",
            "grouped_query_attention",
            inputs=[q, k, v],
            outputs=[attn_out],
            attributes={"num_q_heads": num_query_heads, "num_kv_heads": num_kv_heads},
        )

        # MLP block (SwiGLU)
        gate_out = IRTensor(name="gate_out", shape=(batch_size, seq_len, intermediate_dim), dtype=TensorDType.BF16)
        graph.add_op(
            "torch",
            "linear",
            inputs=[attn_out],
            outputs=[gate_out],
            attributes={"tag": "gate_proj", "in_features": hidden_dim, "out_features": intermediate_dim},
        )

        silu_out = IRTensor(name="silu_out", shape=(batch_size, seq_len, intermediate_dim), dtype=TensorDType.BF16)
        graph.add_op("torch", "silu", inputs=[gate_out], outputs=[silu_out])

        up_out = IRTensor(name="up_out", shape=(batch_size, seq_len, intermediate_dim), dtype=TensorDType.BF16)
        graph.add_op(
            "torch",
            "linear",
            inputs=[attn_out],
            outputs=[up_out],
            attributes={"tag": "up_proj", "in_features": hidden_dim, "out_features": intermediate_dim},
        )

        mul_out = IRTensor(name="mul_out", shape=(batch_size, seq_len, intermediate_dim), dtype=TensorDType.BF16)
        graph.add_op("torch", "mul", inputs=[silu_out, up_out], outputs=[mul_out])

        down_out = IRTensor(name="block_output", shape=(batch_size, seq_len, hidden_dim), dtype=TensorDType.BF16)
        graph.add_op(
            "torch",
            "linear",
            inputs=[mul_out],
            outputs=[down_out],
            attributes={"tag": "down_proj", "in_features": intermediate_dim, "out_features": hidden_dim},
        )

        return graph

    def compile_and_optimize(
        self,
        graph: IRGraph,
        num_query_heads: int = 32,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        enable_fp8: bool = True,
    ) -> Dict[str, Any]:
        # 1. Fuse SwiGLU operators
        fuser = SwiGLUFusionPass()
        fused_graph, fused_count = fuser.apply(graph)

        # 2. Lower GQA
        gqa_lowerer = GQALoweringPass(num_query_heads, num_kv_heads, head_dim)
        gqa_graph, gqa_count = gqa_lowerer.apply(fused_graph)

        # 3. FP8 quantization pass
        final_graph = gqa_graph
        quant_count = 0
        if enable_fp8:
            quant_pass = QuantizationLoweringPass(TensorDType.FP8_E4M3)
            final_graph, quant_count = quant_pass.apply(gqa_graph)

        # 4. Profile the lowered graph
        profiler = WaferClusterProfiler()
        profile_results = profiler.profile_execution(final_graph, batch_size=1, seq_len=2048)

        return {
            "model_name": self.model_name,
            "original_ops": len(graph.operations),
            "final_ops": len(final_graph.operations),
            "fused_swiglu_ops": fused_count,
            "gqa_lowered_ops": gqa_count,
            "quantized_tensors": quant_count,
            "profile": profile_results,
            "lowered_graph": final_graph,
        }
