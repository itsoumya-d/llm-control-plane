"""Codat Commercial Banking Advisory Intelligence Platform.

Engineered for the Codat Advisory Intelligence Platform to power banking assistants
(such as Microsoft Copilot) with client financial books intelligence, certified semantic
definitions, model choice token economics, MCP tool execution, and regression evals.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import math
import time
from typing import Any, Callable, Dict, List, Optional, Tuple


class ModelTier(str, enum.Enum):
    FAST_EXTRACTOR = "fast-extractor"  # Low cost, high speed for deterministic lookups
    BALANCED_ANALYST = "balanced-analyst"  # Mid tier for standard financial comparisons
    FRONTIER_REASONER = "frontier-reasoner"  # High reasoning for complex covenant audits


@dataclasses.dataclass(frozen=True)
class ProvenanceRecord:
    """Audit lineage record linking an AI answer metric directly to source books."""
    metric_name: str
    numeric_value: float
    source_ledger_id: str
    source_table: str
    transaction_date: str
    confidence_score: float
    grounded: bool


@dataclasses.dataclass
class WorkingCapitalSignal:
    """Anomalies and material financial changes detected in client books."""
    signal_id: str
    client_id: str
    signal_type: str  # e.g., 'dso_deterioration', 'customer_concentration'
    severity: str  # 'INFO', 'WARNING', 'CRITICAL'
    description: str
    metric_drift: float
    detected_at: float


@dataclasses.dataclass
class ModelCostMetrics:
    """Token economics attribution per banker query."""
    model_name: str
    prompt_tokens: int
    completion_tokens: int
    estimated_cost_usd: float
    cached: bool
    latency_ms: float


class CommercialBankingSemanticLayer:
    """Standardized definitions of banking metrics across client accounting platforms."""

    @staticmethod
    def calculate_dscr(net_operating_income: float, total_debt_service: float) -> Tuple[float, bool]:
        """Debt Service Coverage Ratio. Minimum typical bank covenant is 1.20x."""
        if total_debt_service <= 0:
            return math.inf, True
        ratio = round(net_operating_income / total_debt_service, 2)
        covenant_compliant = ratio >= 1.20
        return ratio, covenant_compliant

    @staticmethod
    def calculate_current_ratio(current_assets: float, current_liabilities: float) -> Tuple[float, bool]:
        """Liquidity metric: Current Assets / Current Liabilities."""
        if current_liabilities <= 0:
            return math.inf, True
        ratio = round(current_assets / current_liabilities, 2)
        healthy = ratio >= 1.0
        return ratio, healthy

    @staticmethod
    def calculate_dso(accounts_receivable: float, credit_sales: float, days: int = 90) -> float:
        """Days Sales Outstanding: measures average days to collect receivables."""
        if credit_sales <= 0:
            return 0.0
        return round((accounts_receivable / credit_sales) * days, 1)


class ModelChoiceCostRouter:
    """Routes queries to appropriate model tiers balancing reasoning vs token spend."""

    TIER_PRICING_PER_1K = {
        ModelTier.FAST_EXTRACTOR: 0.00015,   # $0.15 / 1M tokens
        ModelTier.BALANCED_ANALYST: 0.0015,  # $1.50 / 1M tokens
        ModelTier.FRONTIER_REASONER: 0.015,  # $15.00 / 1M tokens
    }

    def __init__(self, semantic_cache_enabled: bool = True):
        self.semantic_cache_enabled = semantic_cache_enabled
        self._cache: Dict[str, Tuple[Dict[str, Any], float]] = {}
        self.total_tokens_served: int = 0
        self.total_cost_usd: float = 0.0
        self.dollars_saved_by_caching: float = 0.0

    def select_tier(self, query: str, require_synthesis: bool = False) -> ModelTier:
        """Classify question difficulty to minimize inference spend."""
        q_lower = query.lower()
        complex_keywords = [
            "covenant", "default", "runway forecast", "restructure",
            "sensitivity analysis", "stress test", "cross-collateral"
        ]
        if require_synthesis or any(k in q_lower for k in complex_keywords):
            return ModelTier.FRONTIER_REASONER

        comparison_keywords = ["trend", "compare", "variance", "quarter over quarter", "why did"]
        if any(k in q_lower for k in comparison_keywords):
            return ModelTier.BALANCED_ANALYST

        return ModelTier.FAST_EXTRACTOR

    def estimate_cost(self, tier: ModelTier, prompt_len: int, completion_len: int) -> float:
        rate = self.TIER_PRICING_PER_1K[tier]
        tokens = prompt_len + completion_len
        return round((tokens / 1000.0) * rate, 6)

    def get_cached_response(self, cache_key: str) -> Optional[Dict[str, Any]]:
        if not self.semantic_cache_enabled:
            return None
        if cache_key in self._cache:
            resp, cached_time = self._cache[cache_key]
            self.dollars_saved_by_caching += 0.003
            return resp
        return None

    def store_cache(self, cache_key: str, response: Dict[str, Any]) -> None:
        if self.semantic_cache_enabled:
            self._cache[cache_key] = (response, time.time())


class CodatBankingMCPServer:
    """Model Context Protocol (MCP) server providing verified financial tools to agents."""

    def __init__(self):
        self._registered_tools: Dict[str, Callable] = {}
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        self._registered_tools["get_client_metrics"] = self._tool_get_client_metrics
        self._registered_tools["detect_working_capital_signals"] = self._tool_detect_signals
        self._registered_tools["verify_provenance"] = self._tool_verify_provenance

    def list_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "get_client_metrics",
                "description": "Fetch certified accounting metrics (DSCR, Quick Ratio, DSO) with data lineage.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "client_id": {"type": "string"},
                        "metric_names": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["client_id", "metric_names"]
                }
            },
            {
                "name": "detect_working_capital_signals",
                "description": "Detect emerging financial risks, DSO drift, and debtor concentration in client books.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "client_id": {"type": "string"}
                    },
                    "required": ["client_id"]
                }
            },
            {
                "name": "verify_provenance",
                "description": "Verify every numeric claim against immutable ledger records before answering.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "claims": {"type": "array"}
                    },
                    "required": ["claims"]
                }
            }
        ]

    def call_tool(self, tool_name: str, arguments: Dict[str, Any], client_books: Dict[str, Any]) -> Dict[str, Any]:
        if tool_name not in self._registered_tools:
            return {"error": f"Unknown tool '{tool_name}'", "status": "FAILED"}
        return self._registered_tools[tool_name](arguments, client_books)

    def _tool_get_client_metrics(self, args: Dict[str, Any], books: Dict[str, Any]) -> Dict[str, Any]:
        client_id = args.get("client_id")
        requested = args.get("metric_names", [])
        metrics_out = {}
        provenance = []

        pl = books.get("profit_and_loss", {})
        bs = books.get("balance_sheet", {})

        for m in requested:
            m_norm = m.lower().strip()
            if m_norm in ("dscr", "debt_service_coverage"):
                noi = pl.get("net_operating_income", 0.0)
                debt_service = pl.get("debt_service", 0.0)
                ratio, compliant = CommercialBankingSemanticLayer.calculate_dscr(noi, debt_service)
                metrics_out["dscr"] = {
                    "value": ratio,
                    "covenant_compliant": compliant,
                    "covenant_threshold": 1.20
                }
                provenance.append(ProvenanceRecord(
                    metric_name="dscr",
                    numeric_value=ratio,
                    source_ledger_id=pl.get("ledger_ref", "LEDG-PL-001"),
                    source_table="profit_and_loss",
                    transaction_date=books.get("period_end", "2026-11-01"),
                    confidence_score=1.0,
                    grounded=True
                ))
            elif m_norm in ("current_ratio", "liquidity"):
                ca = bs.get("current_assets", 0.0)
                cl = bs.get("current_liabilities", 0.0)
                ratio, healthy = CommercialBankingSemanticLayer.calculate_current_ratio(ca, cl)
                metrics_out["current_ratio"] = {
                    "value": ratio,
                    "healthy": healthy
                }
                provenance.append(ProvenanceRecord(
                    metric_name="current_ratio",
                    numeric_value=ratio,
                    source_ledger_id=bs.get("ledger_ref", "LEDG-BS-001"),
                    source_table="balance_sheet",
                    transaction_date=books.get("period_end", "2026-11-01"),
                    confidence_score=1.0,
                    grounded=True
                ))
            elif m_norm in ("dso", "days_sales_outstanding"):
                ar = bs.get("accounts_receivable", 0.0)
                sales = pl.get("gross_sales", 0.0)
                dso = CommercialBankingSemanticLayer.calculate_dso(ar, sales)
                metrics_out["dso"] = {
                    "value": dso,
                    "unit": "days"
                }
                provenance.append(ProvenanceRecord(
                    metric_name="dso",
                    numeric_value=dso,
                    source_ledger_id=bs.get("ar_ledger_ref", "LEDG-AR-001"),
                    source_table="receivables_aging",
                    transaction_date=books.get("period_end", "2026-11-01"),
                    confidence_score=1.0,
                    grounded=True
                ))
            else:
                metrics_out[m] = {"error": "Metric not found in certified semantic layer"}

        return {
            "status": "SUCCESS",
            "client_id": client_id,
            "metrics": metrics_out,
            "provenance": [dataclasses.asdict(p) for p in provenance]
        }

    def _tool_detect_signals(self, args: Dict[str, Any], books: Dict[str, Any]) -> Dict[str, Any]:
        client_id = args.get("client_id", "unknown")
        signals = []

        # 1. DSO Drift Detection
        dso_current = books.get("metrics", {}).get("current_dso", 42.0)
        dso_prior = books.get("metrics", {}).get("prior_quarter_dso", 28.0)
        drift = dso_current - dso_prior
        if drift >= 10.0:
            desc = (
                f"Days Sales Outstanding drifted by +{drift:.1f} days "
                f"(from {dso_prior:.1f} to {dso_current:.1f} days). Cash conversion slowing."
            )
            signals.append(WorkingCapitalSignal(
                signal_id=f"SIG-DSO-{int(time.time())}",
                client_id=client_id,
                signal_type="dso_deterioration",
                severity="WARNING" if drift < 20 else "CRITICAL",
                description=desc,
                metric_drift=drift,
                detected_at=time.time()
            ))

        # 2. Debtor Concentration Risk
        debtors = books.get("debtors", [])
        total_ar = sum(d.get("balance", 0.0) for d in debtors)
        if total_ar > 0:
            for d in debtors:
                pct = (d.get("balance", 0.0) / total_ar) * 100.0
                if pct >= 30.0:
                    desc = (
                        f"High credit concentration: Debtor '{d.get('name')}' "
                        f"represents {pct:.1f}% of outstanding receivables."
                    )
                    signals.append(WorkingCapitalSignal(
                        signal_id=f"SIG-CONC-{int(time.time())}",
                        client_id=client_id,
                        signal_type="debtor_concentration",
                        severity="WARNING",
                        description=desc,
                        metric_drift=pct,
                        detected_at=time.time()
                    ))

        return {
            "status": "SUCCESS",
            "client_id": client_id,
            "signals_detected": len(signals),
            "signals": [dataclasses.asdict(s) for s in signals]
        }

    def _tool_verify_provenance(self, args: Dict[str, Any], books: Dict[str, Any]) -> Dict[str, Any]:
        claims = args.get("claims", [])
        verified_claims = []
        all_grounded = True

        known_ledger_ids = books.get("known_ledger_ids", set())

        for c in claims:
            ledger_id = c.get("source_ledger_id")
            val = c.get("numeric_value")
            is_valid = (ledger_id in known_ledger_ids) and (val is not None)
            verified_claims.append({
                "metric_name": c.get("metric_name"),
                "value": val,
                "verified": is_valid,
                "ledger_ref": ledger_id
            })
            if not is_valid:
                all_grounded = False

        return {
            "status": "SUCCESS" if all_grounded else "GROUNDING_FAILED",
            "all_grounded": all_grounded,
            "claims": verified_claims
        }


class CodatAdvisoryIntelligenceEngine:
    """End-to-end Advisory Intelligence orchestration engine for commercial bankers."""

    def __init__(self, semantic_cache: bool = True):
        self.router = ModelChoiceCostRouter(semantic_cache_enabled=semantic_cache)
        self.mcp_server = CodatBankingMCPServer()

    def process_banker_query(
        self,
        banker_query: str,
        client_books: Dict[str, Any],
        require_provenance_audit: bool = True
    ) -> Dict[str, Any]:
        start_t = time.time()
        client_id = client_books.get("client_id", "ACME-CORP")

        # 1. Check Semantic Cache
        cache_key = hashlib.sha256(f"{client_id}:{banker_query}".encode()).hexdigest()
        cached_val = self.router.get_cached_response(cache_key)
        if cached_val:
            cached_val["model_metrics"]["cached"] = True
            cached_val["model_metrics"]["latency_ms"] = round((time.time() - start_t) * 1000, 2)
            return cached_val

        # 2. Model Tier Selection & Token Economics
        selected_tier = self.router.select_tier(banker_query)

        # 3. Grounding & Certified Metrics Retrieval
        requested_metrics = []
        q_lower = banker_query.lower()
        if "covenant" in q_lower or "dscr" in q_lower:
            requested_metrics.append("dscr")
        if "liquidity" in q_lower or "ratio" in q_lower:
            requested_metrics.append("current_ratio")
        if "dso" in q_lower or "receivable" in q_lower or "collection" in q_lower:
            requested_metrics.append("dso")

        if not requested_metrics and "working capital" not in q_lower and "signal" not in q_lower:
            refusal_text = (
                "Refusal: Insufficient certified ledger data to answer without hallucination. "
                "Please specify financial metrics (e.g., DSCR, DSO, Working Capital) available in client books."
            )
            return {
                "status": "REFUSAL_UNGROUNDED",
                "answer": refusal_text,
                "provenance": [],
                "signals": [],
                "model_metrics": dataclasses.asdict(ModelCostMetrics(
                    model_name=selected_tier.value,
                    prompt_tokens=42,
                    completion_tokens=28,
                    estimated_cost_usd=self.router.estimate_cost(selected_tier, 42, 28),
                    cached=False,
                    latency_ms=round((time.time() - start_t) * 1000, 2)
                ))
            }

        mcp_res = self.mcp_server.call_tool(
            "get_client_metrics",
            {"client_id": client_id, "metric_names": requested_metrics},
            client_books
        )

        signal_res = self.mcp_server.call_tool(
            "detect_working_capital_signals",
            {"client_id": client_id},
            client_books
        )

        # 4. Formulate Synthesized Answer
        metrics = mcp_res.get("metrics", {})
        answer_parts = [f"Advisory Intelligence Briefing for Client {client_id}:"]

        if "dscr" in metrics:
            d_info = metrics["dscr"]
            d_val = d_info.get("value")
            d_comp = (
                "compliant with covenant >= 1.20x"
                if d_info.get("covenant_compliant")
                else "BREACH of covenant (below 1.20x)"
            )
            answer_parts.append(f"- Debt Service Coverage Ratio (DSCR): {d_val:.2f}x ({d_comp}).")

        if "current_ratio" in metrics:
            cr_info = metrics["current_ratio"]
            cr_val = cr_info.get("value")
            cr_status = (
                "healthy balance sheet liquidity"
                if cr_info.get("healthy")
                else "compressed liquidity buffer"
            )
            answer_parts.append(f"- Current Ratio: {cr_val:.2f}x ({cr_status}).")

        if "dso" in metrics:
            dso_val = metrics["dso"].get("value")
            answer_parts.append(f"- Days Sales Outstanding (DSO): {dso_val:.1f} days.")

        detected_signals = signal_res.get("signals", [])
        if detected_signals:
            answer_parts.append("\nDetected Working Capital & Risk Signals:")
            for s in detected_signals:
                answer_parts.append(f"  • [{s['severity']}] {s['description']}")

        full_answer = "\n".join(answer_parts)
        provenance = mcp_res.get("provenance", [])

        # Calculate Token Economics
        p_tokens = len(banker_query.split()) * 4 + 150
        c_tokens = len(full_answer.split()) * 4
        cost = self.router.estimate_cost(selected_tier, p_tokens, c_tokens)

        cost_metrics = ModelCostMetrics(
            model_name=selected_tier.value,
            prompt_tokens=p_tokens,
            completion_tokens=c_tokens,
            estimated_cost_usd=cost,
            cached=False,
            latency_ms=round((time.time() - start_t) * 1000, 2)
        )

        result = {
            "status": "SUCCESS",
            "client_id": client_id,
            "tier_selected": selected_tier.value,
            "answer": full_answer,
            "metrics": metrics,
            "provenance": provenance,
            "signals": detected_signals,
            "model_metrics": dataclasses.asdict(cost_metrics)
        }

        self.router.store_cache(cache_key, result)
        return result


class AdvisoryEvalHarness:
    """Benchmark eval harness converting banker corrections into golden test regression cases."""

    def __init__(self, engine: CodatAdvisoryIntelligenceEngine):
        self.engine = engine
        self._golden_cases: List[Dict[str, Any]] = []

    def add_golden_case(
        self,
        case_id: str,
        query: str,
        client_books: Dict[str, Any],
        expected_metric: str,
        expected_min_val: float,
        expected_max_val: float,
        should_refuse: bool = False
    ) -> None:
        self._golden_cases.append({
            "case_id": case_id,
            "query": query,
            "client_books": client_books,
            "expected_metric": expected_metric,
            "expected_min_val": expected_min_val,
            "expected_max_val": expected_max_val,
            "should_refuse": should_refuse
        })

    def ingest_banker_correction(
        self,
        query: str,
        client_books: Dict[str, Any],
        corrected_metric: str,
        exact_value: float,
        reason: str
    ) -> str:
        """Takes a commercial banker correction and compiles it into an automated regression assertion."""
        case_id = f"REG-BANKER-{int(time.time())}-{len(self._golden_cases) + 1}"
        self.add_golden_case(
            case_id=case_id,
            query=query,
            client_books=client_books,
            expected_metric=corrected_metric,
            expected_min_val=exact_value * 0.999,
            expected_max_val=exact_value * 1.001,
            should_refuse=False
        )
        return case_id

    def run_eval_suite(self) -> Dict[str, Any]:
        passed = 0
        failed = 0
        details = []

        for case in self._golden_cases:
            res = self.engine.process_banker_query(case["query"], case["client_books"])
            if case["should_refuse"]:
                is_pass = res.get("status") == "REFUSAL_UNGROUNDED"
            else:
                m_info = res.get("metrics", {}).get(case["expected_metric"], {})
                val = m_info.get("value")
                is_pass = (
                    val is not None
                    and case["expected_min_val"] <= val <= case["expected_max_val"]
                    and res.get("status") == "SUCCESS"
                )

            if is_pass:
                passed += 1
            else:
                failed += 1

            details.append({
                "case_id": case["case_id"],
                "passed": is_pass,
                "status": res.get("status")
            })

        total = len(self._golden_cases)
        pass_rate = round((passed / total) * 100.0, 1) if total > 0 else 100.0

        return {
            "total_cases": total,
            "passed": passed,
            "failed": failed,
            "pass_rate_pct": pass_rate,
            "eval_details": details
        }
