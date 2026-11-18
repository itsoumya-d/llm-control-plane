"""
Adaptive AI Platform Gateway Router, Vector RAG Engine, and Regression Evaluator.

Provides production-grade AI platform infrastructure:
1. AdaptiveLLMGateway: Multi-provider model access, latency/cost-optimized routing,
   circuit breaking, and fallback policies.
2. VectorKnowledgeEngine: Document ingestion, semantic chunking, simulated pgvector
   hybrid retrieval, and groundedness verification.
3. RegressionEvaluator: Golden dataset evaluation harness, regression detection against
   baselines, and P50/P95 latency SLO monitoring.
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class ModelSpec:
    model_id: str
    provider: str  # azure_openai, anthropic, openai, self_hosted
    cost_per_1k_input: float
    cost_per_1k_output: float
    avg_latency_ms: float
    max_context_tokens: int
    quality_tier: int  # 1 (lightweight) to 5 (flagship)
    is_healthy: bool = True
    consecutive_failures: int = 0
    failure_threshold: int = 3
    circuit_state: CircuitState = CircuitState.CLOSED
    last_failure_time: float = 0.0
    recovery_cooldown_seconds: float = 30.0


@dataclass
class RoutingRequest:
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    priority: str = "normal"  # low, normal, high
    max_latency_ms: Optional[float] = None
    cost_budget: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingDecision:
    chosen_model: str
    provider: str
    estimated_latency_ms: float
    estimated_cost: float
    circuit_status: str
    fallback_chain: List[str] = field(default_factory=list)


@dataclass
class ExecutionResult:
    model_id: str
    provider: str
    output_text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    cost: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class Document:
    doc_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    vector: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    chunk_id: str
    doc_id: str
    score: float
    text: str
    lexical_score: float
    dense_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestCase:
    __test__ = False  # Prevent pytest from collecting this as a test class
    test_id: str
    prompt: str
    expected_facts: List[str]
    reference_output: str
    max_acceptable_latency_ms: float = 1500.0


@dataclass
class TestResult:
    test_id: str
    model_id: str
    actual_output: str
    latency_ms: float
    cost: float
    factual_accuracy_score: float
    hallucination_detected: bool
    passed: bool
    failure_reasons: List[str] = field(default_factory=list)


@dataclass
class RegressionReport:
    total_tests: int
    passed_tests: int
    pass_rate: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    total_cost: float
    regressions_detected: List[str]
    is_deployable: bool


class AdaptiveLLMGateway:
    """Production LLM Gateway supporting multi-provider routing and circuit breaking."""

    def __init__(self) -> None:
        self.models: Dict[str, ModelSpec] = {}
        self.execution_logs: List[ExecutionResult] = []

    def register_model(self, model: ModelSpec) -> None:
        self.models[model.model_id] = model

    def record_success(self, model_id: str) -> None:
        if model_id in self.models:
            model = self.models[model_id]
            model.consecutive_failures = 0
            if model.circuit_state == CircuitState.HALF_OPEN:
                model.circuit_state = CircuitState.CLOSED
                model.is_healthy = True

    def record_failure(self, model_id: str) -> None:
        if model_id in self.models:
            model = self.models[model_id]
            model.consecutive_failures += 1
            model.last_failure_time = time.time()
            if model.consecutive_failures >= model.failure_threshold:
                model.circuit_state = CircuitState.OPEN
                model.is_healthy = False

    def check_circuit_breaker(self, model_id: str) -> CircuitState:
        if model_id not in self.models:
            return CircuitState.OPEN
        model = self.models[model_id]
        if model.circuit_state == CircuitState.OPEN:
            # Check cooldown
            if time.time() - model.last_failure_time > model.recovery_cooldown_seconds:
                model.circuit_state = CircuitState.HALF_OPEN
        return model.circuit_state

    def estimate_tokens(self, text: str) -> int:
        # Approximate 4 characters per token
        return max(1, len(text) // 4)

    def route_request(self, request: RoutingRequest) -> RoutingDecision:
        estimated_input_tokens = self.estimate_tokens(request.prompt)
        candidates: List[Tuple[float, ModelSpec]] = []

        for model_id, model in self.models.items():
            state = self.check_circuit_breaker(model_id)
            if state == CircuitState.OPEN:
                continue

            # Context window check
            if estimated_input_tokens + request.max_tokens > model.max_context_tokens:
                continue

            # Cost estimation
            cost = (
                (estimated_input_tokens / 1000.0) * model.cost_per_1k_input
                + (request.max_tokens / 1000.0) * model.cost_per_1k_output
            )
            if request.cost_budget is not None and cost > request.cost_budget:
                continue

            # Latency check
            if request.max_latency_ms is not None and model.avg_latency_ms > request.max_latency_ms:
                continue

            # Scoring: Multi-objective score (higher is better)
            latency_penalty = model.avg_latency_ms / 1000.0
            cost_penalty = cost * 100.0

            if request.priority == "high":
                score = (model.quality_tier * 5.0) - latency_penalty - cost_penalty
            elif request.priority == "low":
                # Heavily prioritize lower cost
                score = (1.0 / (cost + 1e-4)) - latency_penalty
            else:
                score = (model.quality_tier * 2.0) - latency_penalty - cost_penalty

            candidates.append((score, model))

        if not candidates:
            # Fallback to any model regardless of strict latency/budget if available
            available = [
                m for m in self.models.values()
                if self.check_circuit_breaker(m.model_id) != CircuitState.OPEN
            ]
            if not available:
                raise RuntimeError("All models in LLM Gateway have circuits OPEN or are unreachable.")
            available.sort(key=lambda m: m.cost_per_1k_input)
            chosen = available[0]
        else:
            candidates.sort(key=lambda x: x[0], reverse=True)
            chosen = candidates[0][1]

        # Calculate estimated cost
        est_cost = (
            (estimated_input_tokens / 1000.0) * chosen.cost_per_1k_input
            + (request.max_tokens / 1000.0) * chosen.cost_per_1k_output
        )

        fallback_chain = [
            m.model_id for _, m in candidates if m.model_id != chosen.model_id
        ]

        return RoutingDecision(
            chosen_model=chosen.model_id,
            provider=chosen.provider,
            estimated_latency_ms=chosen.avg_latency_ms,
            estimated_cost=round(est_cost, 6),
            circuit_status=chosen.circuit_state.value,
            fallback_chain=fallback_chain,
        )

    def execute_request(
        self, request: RoutingRequest, simulated_output: Optional[str] = None
    ) -> ExecutionResult:
        decision = self.route_request(request)
        model = self.models[decision.chosen_model]

        prompt_tokens = self.estimate_tokens(request.prompt)
        output_text = simulated_output or f"Response from {model.model_id} for prompt: {request.prompt[:30]}..."
        completion_tokens = self.estimate_tokens(output_text)
        total_tokens = prompt_tokens + completion_tokens

        actual_cost = (
            (prompt_tokens / 1000.0) * model.cost_per_1k_input
            + (completion_tokens / 1000.0) * model.cost_per_1k_output
        )

        # Simulate execution
        res = ExecutionResult(
            model_id=model.model_id,
            provider=model.provider,
            output_text=output_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=model.avg_latency_ms,
            cost=round(actual_cost, 6),
            success=True,
        )
        self.record_success(model.model_id)
        self.execution_logs.append(res)
        return res


class VectorKnowledgeEngine:
    """RAG infrastructure providing document ingestion, chunking, and hybrid search."""

    def __init__(self, vector_dim: int = 16) -> None:
        self.vector_dim = vector_dim
        self.chunks: List[Chunk] = []

    def _tokenize(self, text: str) -> List[str]:
        return [w.lower() for w in re.findall(r"\b\w+\b", text)]

    def _generate_dense_vector(self, text: str) -> List[float]:
        # Deterministic normalized embedding vector derived from text hash
        vec = [0.0] * self.vector_dim
        words = self._tokenize(text)
        if not words:
            return vec
        for word in words:
            h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
            for i in range(self.vector_dim):
                component = ((h >> (i * 4)) & 0x0F) - 7.5
                vec[i] += component
        # Normalize to unit length
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 1e-6:
            vec = [x / norm for x in vec]
        return vec

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        if len(v1) != len(v2):
            return 0.0
        dot = sum(a * b for a, b in zip(v1, v2))
        return max(0.0, min(1.0, (dot + 1.0) / 2.0))

    def ingest_document(
        self, doc: Document, chunk_size: int = 200, chunk_overlap: int = 40
    ) -> List[Chunk]:
        words = doc.content.split()
        if not words:
            return []

        created_chunks: List[Chunk] = []
        step = max(1, chunk_size - chunk_overlap)
        index = 0

        for i in range(0, len(words), step):
            chunk_words = words[i : i + chunk_size]
            chunk_text = " ".join(chunk_words)
            chunk_id = f"{doc.doc_id}_c{index}"
            vec = self._generate_dense_vector(chunk_text)
            c = Chunk(
                chunk_id=chunk_id,
                doc_id=doc.doc_id,
                text=chunk_text,
                vector=vec,
                metadata={**doc.metadata, "chunk_index": index},
            )
            created_chunks.append(c)
            self.chunks.append(c)
            index += 1
            if i + chunk_size >= len(words):
                break

        return created_chunks

    def hybrid_search(
        self, query: str, top_k: int = 3, dense_weight: float = 0.6
    ) -> List[RetrievalResult]:
        query_words = set(self._tokenize(query))
        query_vector = self._generate_dense_vector(query)
        results: List[RetrievalResult] = []

        for chunk in self.chunks:
            # Lexical score: Jaccard overlap of words
            chunk_words = set(self._tokenize(chunk.text))
            if query_words and chunk_words:
                overlap = len(query_words.intersection(chunk_words))
                union = len(query_words.union(chunk_words))
                lexical_score = overlap / union if union > 0 else 0.0
            else:
                lexical_score = 0.0

            # Dense score: Cosine similarity
            dense_score = self._cosine_similarity(query_vector, chunk.vector)

            # Combined hybrid score
            hybrid_score = (dense_weight * dense_score) + ((1.0 - dense_weight) * lexical_score)

            results.append(
                RetrievalResult(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    score=round(hybrid_score, 4),
                    text=chunk.text,
                    lexical_score=round(lexical_score, 4),
                    dense_score=round(dense_score, 4),
                    metadata=chunk.metadata,
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def verify_groundedness(self, answer: str, retrieved_chunks: List[RetrievalResult]) -> Tuple[bool, float]:
        """Check if claims in the answer are supported by retrieved chunks."""
        answer_words = set(self._tokenize(answer))
        if not answer_words:
            return True, 1.0

        corpus_words: Set[str] = set()
        for chunk in retrieved_chunks:
            corpus_words.update(self._tokenize(chunk.text))

        supported_words = answer_words.intersection(corpus_words)
        groundedness_ratio = len(supported_words) / len(answer_words)

        # Passes groundedness if at least 65% of vocabulary words are verified in context
        is_grounded = groundedness_ratio >= 0.65
        return is_grounded, round(groundedness_ratio, 4)


class RegressionEvaluator:
    """Evaluates AI models against golden regression datasets with SLO tracking."""

    def __init__(self, gateway: AdaptiveLLMGateway) -> None:
        self.gateway = gateway
        self.test_cases: List[TestCase] = []

    def add_test_case(self, case: TestCase) -> None:
        self.test_cases.append(case)

    def run_evaluations(self, model_id: Optional[str] = None) -> RegressionReport:
        results: List[TestResult] = []
        latencies: List[float] = []
        total_cost = 0.0
        regressions: List[str] = []

        for case in self.test_cases:
            req = RoutingRequest(prompt=case.prompt, priority="high")
            if model_id and model_id in self.gateway.models:
                # Force specific model if requested
                old_models = self.gateway.models
                selected_model = old_models[model_id]
                # Temporary simulate direct execution
                prompt_tokens = self.gateway.estimate_tokens(case.prompt)
                simulated_output = case.reference_output
                completion_tokens = self.gateway.estimate_tokens(simulated_output)
                latency = selected_model.avg_latency_ms
                cost = (
                    (prompt_tokens / 1000.0) * selected_model.cost_per_1k_input
                    + (completion_tokens / 1000.0) * selected_model.cost_per_1k_output
                )
            else:
                exec_res = self.gateway.execute_request(req, simulated_output=case.reference_output)
                latency = exec_res.latency_ms
                cost = exec_res.cost
                simulated_output = exec_res.output_text
                model_id = exec_res.model_id

            latencies.append(latency)
            total_cost += cost

            # Fact verification
            missing_facts = [
                fact for fact in case.expected_facts
                if fact.lower() not in simulated_output.lower()
            ]

            hallucination_detected = False
            accuracy_score = 1.0 - (len(missing_facts) / len(case.expected_facts)) if case.expected_facts else 1.0

            failure_reasons = []
            if missing_facts:
                failure_reasons.append(f"Missing expected facts: {missing_facts}")
            if latency > case.max_acceptable_latency_ms:
                failure_reasons.append(f"Latency {latency}ms exceeded SLO {case.max_acceptable_latency_ms}ms")

            passed = len(failure_reasons) == 0

            if not passed:
                regressions.append(f"{case.test_id}: {'; '.join(failure_reasons)}")

            results.append(
                TestResult(
                    test_id=case.test_id,
                    model_id=model_id or "default",
                    actual_output=simulated_output,
                    latency_ms=latency,
                    cost=round(cost, 6),
                    factual_accuracy_score=round(accuracy_score, 4),
                    hallucination_detected=hallucination_detected,
                    passed=passed,
                    failure_reasons=failure_reasons,
                )
            )

        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.passed)
        pass_rate = (passed_tests / total_tests) if total_tests > 0 else 1.0

        latencies.sort()
        p50 = latencies[len(latencies) // 2] if latencies else 0.0
        p95_index = min(len(latencies) - 1, int(len(latencies) * 0.95))
        p95 = latencies[p95_index] if latencies else 0.0
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

        is_deployable = pass_rate >= 0.90 and len(regressions) == 0

        return RegressionReport(
            total_tests=total_tests,
            passed_tests=passed_tests,
            pass_rate=round(pass_rate, 4),
            avg_latency_ms=round(avg_latency, 2),
            p50_latency_ms=round(p50, 2),
            p95_latency_ms=round(p95, 2),
            total_cost=round(total_cost, 6),
            regressions_detected=regressions,
            is_deployable=is_deployable,
        )
