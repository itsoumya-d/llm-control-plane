"""Unit and integration tests for Codat Commercial Banking Advisory Intelligence Platform."""

import pytest
from llm_control_plane.codat_advisory_intelligence import (
    AdvisoryEvalHarness,
    CodatAdvisoryIntelligenceEngine,
    CodatBankingMCPServer,
    CommercialBankingSemanticLayer,
    ModelChoiceCostRouter,
    ModelTier,
)


@pytest.fixture
def sample_client_books():
    return {
        "client_id": "BARBICAN-TECH-LTD",
        "period_end": "2026-10-31",
        "known_ledger_ids": {"LEDG-PL-001", "LEDG-BS-001", "LEDG-AR-001"},
        "profit_and_loss": {
            "ledger_ref": "LEDG-PL-001",
            "net_operating_income": 450000.0,
            "debt_service": 250000.0,
            "gross_sales": 1800000.0
        },
        "balance_sheet": {
            "ledger_ref": "LEDG-BS-001",
            "ar_ledger_ref": "LEDG-AR-001",
            "current_assets": 1200000.0,
            "current_liabilities": 750000.0,
            "accounts_receivable": 360000.0
        },
        "metrics": {
            "current_dso": 48.0,
            "prior_quarter_dso": 32.0
        },
        "debtors": [
            {"name": "Client Acme UK", "balance": 180000.0},
            {"name": "Beta Retail", "balance": 90000.0},
            {"name": "Gamma Logistics", "balance": 90000.0}
        ]
    }


def test_commercial_banking_semantic_layer():
    # DSCR: 450k / 250k = 1.80x (Covenant >= 1.20x -> Compliant)
    dscr, compliant = CommercialBankingSemanticLayer.calculate_dscr(450000.0, 250000.0)
    assert dscr == 1.80
    assert compliant is True

    # Breach case: 100k / 100k = 1.00x < 1.20x
    dscr_breach, compliant_breach = CommercialBankingSemanticLayer.calculate_dscr(100000.0, 100000.0)
    assert dscr_breach == 1.00
    assert compliant_breach is False

    # Current ratio: 1.2M / 750k = 1.60x
    cr, healthy = CommercialBankingSemanticLayer.calculate_current_ratio(1200000.0, 750000.0)
    assert cr == 1.60
    assert healthy is True

    # DSO: (360k / 1.8M) * 90 = 18.0 days
    dso = CommercialBankingSemanticLayer.calculate_dso(360000.0, 1800000.0, 90)
    assert dso == 18.0


def test_model_choice_cost_router():
    router = ModelChoiceCostRouter(semantic_cache_enabled=True)

    # Simple metric lookup -> Fast Extractor
    tier1 = router.select_tier("What is the current DSO for Barbican Tech?")
    assert tier1 == ModelTier.FAST_EXTRACTOR

    # Trend / comparison -> Balanced Analyst
    tier2 = router.select_tier("Compare current quarter revenue variance against last quarter")
    assert tier2 == ModelTier.BALANCED_ANALYST

    # Complex covenant / stress test -> Frontier Reasoner
    tier3 = router.select_tier("Perform debt covenant sensitivity analysis if revenue drops 15%")
    assert tier3 == ModelTier.FRONTIER_REASONER

    # Cost estimation
    cost_fast = router.estimate_cost(ModelTier.FAST_EXTRACTOR, 100, 50)
    cost_frontier = router.estimate_cost(ModelTier.FRONTIER_REASONER, 100, 50)
    assert cost_fast < cost_frontier


def test_mcp_server_tools_and_execution(sample_client_books):
    server = CodatBankingMCPServer()
    tools = server.list_tools()
    tool_names = [t["name"] for t in tools]
    assert "get_client_metrics" in tool_names
    assert "detect_working_capital_signals" in tool_names
    assert "verify_provenance" in tool_names

    # Call get_client_metrics
    res = server.call_tool(
        "get_client_metrics",
        {"client_id": "BARBICAN-TECH-LTD", "metric_names": ["dscr", "current_ratio", "dso"]},
        sample_client_books
    )
    assert res["status"] == "SUCCESS"
    assert res["metrics"]["dscr"]["value"] == 1.80
    assert res["metrics"]["current_ratio"]["value"] == 1.60
    assert len(res["provenance"]) == 3
    assert res["provenance"][0]["grounded"] is True


def test_working_capital_anomaly_signals(sample_client_books):
    server = CodatBankingMCPServer()
    res = server.call_tool(
        "detect_working_capital_signals",
        {"client_id": "BARBICAN-TECH-LTD"},
        sample_client_books
    )
    assert res["status"] == "SUCCESS"
    assert res["signals_detected"] >= 2

    signal_types = [s["signal_type"] for s in res["signals"]]
    assert "dso_deterioration" in signal_types
    assert "debtor_concentration" in signal_types


def test_end_to_end_banker_query_and_cache(sample_client_books):
    engine = CodatAdvisoryIntelligenceEngine(semantic_cache=True)
    query = "Check debt covenant status and liquidity ratio for Barbican Tech"

    # Turn 1: Uncached computation
    res1 = engine.process_banker_query(query, sample_client_books)
    assert res1["status"] == "SUCCESS"
    assert "Debt Service Coverage Ratio" in res1["answer"]
    assert res1["metrics"]["dscr"]["value"] == 1.80
    assert res1["model_metrics"]["cached"] is False

    # Turn 2: Cached computation with sub-10ms response
    res2 = engine.process_banker_query(query, sample_client_books)
    assert res2["status"] == "SUCCESS"
    assert res2["model_metrics"]["cached"] is True
    assert res2["metrics"]["dscr"]["value"] == 1.80


def test_deterministic_hallucination_refusal(sample_client_books):
    engine = CodatAdvisoryIntelligenceEngine(semantic_cache=False)
    ungrounded_query = "What is the CEO's favorite holiday destination and stock price target?"

    res = engine.process_banker_query(ungrounded_query, sample_client_books)
    assert res["status"] == "REFUSAL_UNGROUNDED"
    assert "Refusal: Insufficient certified ledger data" in res["answer"]
    assert len(res["provenance"]) == 0


def test_eval_harness_and_banker_correction_ingestion(sample_client_books):
    engine = CodatAdvisoryIntelligenceEngine(semantic_cache=False)
    harness = AdvisoryEvalHarness(engine)

    # Add standard golden test case
    harness.add_golden_case(
        case_id="GOLD-001",
        query="What is the DSCR covenant ratio?",
        client_books=sample_client_books,
        expected_metric="dscr",
        expected_min_val=1.79,
        expected_max_val=1.81
    )

    # Ingest banker correction: relationship manager noted a custom accounting adjustment
    case_id = harness.ingest_banker_correction(
        query="What is the liquidity current ratio for Barbican Tech?",
        client_books=sample_client_books,
        corrected_metric="current_ratio",
        exact_value=1.60,
        reason="Banker verified current assets includes audited prepaid expenses"
    )
    assert case_id.startswith("REG-BANKER-")

    # Run eval suite
    eval_report = harness.run_eval_suite()
    assert eval_report["total_cases"] == 2
    assert eval_report["passed"] == 2
    assert eval_report["failed"] == 0
    assert eval_report["pass_rate_pct"] == 100.0
