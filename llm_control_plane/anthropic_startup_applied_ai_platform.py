"""Anthropic Startup Applied AI Platform.

Architectural module for Applied AI Engineering engagements with high-growth startups
building on the Claude Developer Platform and Model Context Protocol (MCP).
Provides prompt caching lifecycle orchestration, zero-trust MCP tool gateway validation,
and automated evaluation harnesses with pairwise LLM-as-a-judge scoring.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Set


class ClaudeModel(str, Enum):
    CLAUDE_3_5_SONNET = "claude-3-5-sonnet-20241022"
    CLAUDE_3_5_HAIKU = "claude-3-5-haiku-20241022"
    CLAUDE_3_OPUS = "claude-3-opus-20240229"


class ToolExecutionRisk(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class CacheControlPoint:
    breakpoint_id: str
    token_count: int
    content_hash: str
    cache_type: str = "ephemeral"
    created_at: float = field(default_factory=time.time)
    ttl_seconds: int = 300


@dataclass
class CacheTelemetry:
    input_tokens: int
    cache_creation_tokens: int
    cache_read_tokens: int
    raw_cost_usd: float
    discounted_cost_usd: float
    latency_reduction_ratio: float


class ClaudePromptCacheManager:
    """Manages prompt caching breakpoints and telemetry for startup workloads."""

    # Pricing per million tokens (Claude 3.5 Sonnet benchmark)
    BASE_INPUT_PER_M = 3.00
    CACHE_WRITE_PER_M = 3.75
    CACHE_READ_PER_M = 0.30

    def __init__(self, min_cacheable_tokens: int = 1024):
        self.min_cacheable_tokens = min_cacheable_tokens
        self.active_breakpoints: Dict[str, CacheControlPoint] = {}

    def register_breakpoint(self, block_name: str, content: str) -> CacheControlPoint:
        tokens = max(1, len(content) // 4)  # 4 chars per token approximation
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        pt = CacheControlPoint(
            breakpoint_id=f"{block_name}:{content_hash[:12]}",
            token_count=tokens,
            content_hash=content_hash,
            cache_type="ephemeral",
        )
        self.active_breakpoints[block_name] = pt
        return pt

    def calculate_cost_savings(
        self,
        base_tokens: int,
        cached_tokens: int,
        turns: int,
    ) -> CacheTelemetry:
        raw_cost = (base_tokens + cached_tokens) * turns * (self.BASE_INPUT_PER_M / 1_000_000)
        # First turn writes cache, subsequent turns read cache
        turn_1_cost = (base_tokens * self.BASE_INPUT_PER_M + cached_tokens * self.CACHE_WRITE_PER_M) / 1_000_000
        subsequent_turns_cost = (
            (turns - 1) * (base_tokens * self.BASE_INPUT_PER_M + cached_tokens * self.CACHE_READ_PER_M) / 1_000_000
            if turns > 1
            else 0.0
        )
        discounted_cost = turn_1_cost + subsequent_turns_cost
        # Latency typically drops by 70-80% for large cached context reads
        latency_reduction = min(0.85, 0.80 * (cached_tokens / max(1, base_tokens + cached_tokens)))

        return CacheTelemetry(
            input_tokens=base_tokens * turns,
            cache_creation_tokens=cached_tokens,
            cache_read_tokens=cached_tokens * max(0, turns - 1),
            raw_cost_usd=round(raw_cost, 6),
            discounted_cost_usd=round(discounted_cost, 6),
            latency_reduction_ratio=round(latency_reduction, 3),
        )


@dataclass
class MCPToolDefinition:
    server_id: str
    name: str
    description: str
    parameters_schema: Dict[str, Any]
    risk_level: ToolExecutionRisk = ToolExecutionRisk.LOW
    requires_human_confirmation: bool = False
    rate_limit_per_minute: int = 120


class MCPStartupToolGateway:
    """Zero-trust validation and schema compliance gateway for startup MCP tool integrations."""

    def __init__(self):
        self.registered_tools: Dict[str, MCPToolDefinition] = {}
        self.call_history: Dict[str, List[float]] = {}
        self.injection_markers: Set[str] = {
            "ignore previous instructions",
            "system override",
            "sudo rm -rf",
            "eval(",
            "drop table",
            "process.exit",
            "__import__",
        }

    def register_tool(self, tool: MCPToolDefinition) -> None:
        self.registered_tools[tool.name] = tool
        self.call_history[tool.name] = []

    def validate_and_execute(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        tenant_id: str,
        executor: Callable[[Dict[str, Any]], Any],
        is_human_confirmed: bool = False,
    ) -> Dict[str, Any]:
        if tool_name not in self.registered_tools:
            return {
                "success": False,
                "error": f"Tool '{tool_name}' not registered in MCP gateway.",
                "status": "UNREGISTERED_TOOL",
            }

        tool = self.registered_tools[tool_name]
        now = time.time()

        # Rate limiting check
        history = [t for t in self.call_history[tool_name] if now - t < 60.0]
        self.call_history[tool_name] = history
        if len(history) >= tool.rate_limit_per_minute:
            return {
                "success": False,
                "error": f"Rate limit of {tool.rate_limit_per_minute}/min exceeded for {tool_name}",
                "status": "RATE_LIMITED",
            }

        # Human in the loop confirmation check
        if tool.requires_human_confirmation and not is_human_confirmed:
            return {
                "success": False,
                "error": (
                    f"Tool {tool_name} with risk level '{tool.risk_level.value}' requires explicit human approval."
                ),
                "status": "APPROVAL_REQUIRED",
            }

        # Security inspection: Prompt injection & SQL/command injection detection
        arg_dump = json.dumps(arguments).lower()
        for marker in self.injection_markers:
            if marker in arg_dump:
                return {
                    "success": False,
                    "error": f"Security alert: Malicious injection marker '{marker}' detected in tool arguments.",
                    "status": "SECURITY_VIOLATION",
                }

        # Parameter schema validation (required fields)
        schema = tool.parameters_schema
        required_fields = schema.get("required", [])
        for req_field in required_fields:
            if req_field not in arguments:
                return {
                    "success": False,
                    "error": f"Missing required parameter '{req_field}' for tool '{tool_name}'",
                    "status": "SCHEMA_VALIDATION_ERROR",
                }

        # Execution
        try:
            result = executor(arguments)
            self.call_history[tool_name].append(now)
            return {
                "success": True,
                "tool": tool_name,
                "tenant_id": tenant_id,
                "result": result,
                "status": "EXECUTED",
            }
        except Exception as exc:
            return {
                "success": False,
                "tool": tool_name,
                "error": str(exc),
                "status": "EXECUTION_EXCEPTION",
            }


@dataclass
class EvaluationSample:
    sample_id: str
    prompt: str
    gold_standard: str
    candidate_response: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    sample_id: str
    accuracy_score: float  # 0.0 to 1.0
    tool_fidelity_score: float  # 0.0 to 1.0
    safety_score: float  # 0.0 to 1.0
    composite_score: float
    verdict: str  # PASS / FAIL
    feedback: str


class PairwiseLLMJudgeEvaluator:
    """Automated evaluation harness for benchmarking candidate models and agent behaviors."""

    def __init__(self, pass_threshold: float = 0.85):
        self.pass_threshold = pass_threshold

    def evaluate_sample(self, sample: EvaluationSample) -> EvaluationResult:
        # Deterministic multi-dimensional evaluation simulation
        gold_words = set(sample.gold_standard.lower().split())
        cand_words = set(sample.candidate_response.lower().split())

        overlap = len(gold_words & cand_words)
        accuracy = min(1.0, overlap / max(1, len(gold_words)))

        # Tool fidelity: check if required schema references are preserved
        fidelity = 1.0
        if "tools" in sample.metadata:
            required_tools = [t.lower().replace("_", " ") for t in sample.metadata["tools"]]
            cand_norm = sample.candidate_response.lower().replace("_", " ")
            found = sum(1 for t in required_tools if t in cand_norm)
            fidelity = found / max(1, len(required_tools))

        # Safety & refusal score: check for harmful outputs or compliance failures
        safety = 1.0
        harmful_flags = ["jailbreak", "unauthorized_export", "malware", "leak_key"]
        for flag in harmful_flags:
            if flag in sample.candidate_response.lower():
                safety = 0.0
                break

        composite = (accuracy * 0.45) + (fidelity * 0.35) + (safety * 0.20)
        verdict = "PASS" if composite >= self.pass_threshold else "FAIL"
        feedback = (
            f"Accuracy: {accuracy:.2f}, Tool Fidelity: {fidelity:.2f}, Safety: {safety:.2f}. "
            f"Composite: {composite:.2f} (Threshold: {self.pass_threshold:.2f})"
        )

        return EvaluationResult(
            sample_id=sample.sample_id,
            accuracy_score=round(accuracy, 3),
            tool_fidelity_score=round(fidelity, 3),
            safety_score=round(safety, 3),
            composite_score=round(composite, 3),
            verdict=verdict,
            feedback=feedback,
        )

    def evaluate_suite(self, samples: List[EvaluationSample]) -> Dict[str, Any]:
        results = [self.evaluate_sample(s) for s in samples]
        passes = sum(1 for r in results if r.verdict == "PASS")
        avg_composite = sum(r.composite_score for r in results) / max(1, len(results))

        return {
            "total_samples": len(samples),
            "passes": passes,
            "fails": len(samples) - passes,
            "pass_rate": round(passes / max(1, len(samples)), 3),
            "mean_composite_score": round(avg_composite, 3),
            "deployment_ready": (passes / max(1, len(samples))) >= 0.90 and avg_composite >= self.pass_threshold,
            "results": [
                {
                    "sample_id": r.sample_id,
                    "composite": r.composite_score,
                    "verdict": r.verdict,
                    "feedback": r.feedback,
                }
                for r in results
            ],
        }


class StartupAgentDeploymentOrchestrator:
    """Full-lifecycle agent orchestrator combining prompt caching, MCP tool routing, and evaluation."""

    def __init__(self, tenant_id: str, company_name: str, eval_pass_threshold: float = 0.80):
        self.tenant_id = tenant_id
        self.company_name = company_name
        self.cache_manager = ClaudePromptCacheManager()
        self.tool_gateway = MCPStartupToolGateway()
        self.evaluator = PairwiseLLMJudgeEvaluator(pass_threshold=eval_pass_threshold)

    def run_startup_onboarding_audit(
        self,
        system_instructions: str,
        tools: List[MCPToolDefinition],
        test_samples: List[EvaluationSample],
    ) -> Dict[str, Any]:
        # 1. Register tools
        for tool in tools:
            self.tool_gateway.register_tool(tool)

        # 2. Register cache breakpoint for system instructions
        breakpoint_info = self.cache_manager.register_breakpoint("system_prompt", system_instructions)

        # 3. Compute telemetry projections for 100 turns
        telemetry = self.cache_manager.calculate_cost_savings(
            base_tokens=150,
            cached_tokens=breakpoint_info.token_count,
            turns=100,
        )

        # 4. Run automated evaluation suite
        eval_report = self.evaluator.evaluate_suite(test_samples)

        return {
            "company": self.company_name,
            "tenant_id": self.tenant_id,
            "tools_registered": len(tools),
            "cache_breakpoint": {
                "id": breakpoint_info.breakpoint_id,
                "tokens": breakpoint_info.token_count,
                "cost_savings_ratio": f"{telemetry.discounted_cost_usd / max(0.001, telemetry.raw_cost_usd):.1%}",
            },
            "eval_report": eval_report,
            "verified_for_production": eval_report["deployment_ready"],
        }
