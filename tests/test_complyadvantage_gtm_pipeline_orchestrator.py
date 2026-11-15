"""Unit tests for ComplyAdvantage GTM Systems and Applied AI Pipeline Orchestrator."""


from llm_control_plane.complyadvantage_gtm_pipeline_orchestrator import (
    AccountTier,
    CrmWebhookSyncDispatcher,
    DefendableAccountScoringEngine,
    EnrichmentWaterfallEngine,
    LLMAccountIntelligenceAgent,
    ResilientMarketScraper,
    SignalOutboundPlayRunner,
    SignalType,
    TargetAccount,
)


def test_scraper_domain_cleaning_and_deduplication():
    scraper = ResilientMarketScraper(max_retries=2)
    cleaned1 = scraper.clean_domain("https://www.fintech-example.co.uk/about?ref=direct")
    assert cleaned1 == "fintech-example.co.uk"

    account1 = scraper.scrape_account_record("https://neo-pay.uk", "NeoPay")
    assert account1 is not None
    assert account1.domain == "neo-pay.uk"
    assert account1.company_name == "NeoPay"

    # Second scrape of identical domain returns None (deduplication)
    account2 = scraper.scrape_account_record("http://www.neo-pay.uk/pricing", "NeoPay Copy")
    assert account2 is None


def test_scraper_custom_extractor_and_failure_retry():
    scraper = ResilientMarketScraper(max_retries=2, base_delay=0.001)

    def custom_ext(dom):
        return {
            "jurisdiction": "UK",
            "industry": "Crypto / VASP",
            "employee_count": 250,
            "compliance_team_size": 8,
            "cross_border_payments_active": True,
            "recent_regulatory_penalties": 1,
        }

    account = scraper.scrape_account_record("crypto-gateway.io", "CryptoGateway", extractor_fn=custom_ext)
    assert account is not None
    assert account.industry == "Crypto / VASP"
    assert account.employee_count == 250
    assert account.recent_regulatory_penalties == 1

    # Test failure recovery after retry exhaustion
    def failing_ext(dom):
        raise RuntimeError("Network timeout connecting to registry")

    failing_account = scraper.scrape_account_record("timeout-bank.com", "TimeoutBank", extractor_fn=failing_ext)
    assert failing_account is None


def test_enrichment_waterfall_cost_logic():
    engine = EnrichmentWaterfallEngine()
    account = TargetAccount(
        domain="uk-challenger.co.uk",
        company_name="UK Challenger Bank",
        jurisdiction="UK",
        industry="Retail Banking",
        employee_count=150,
        compliance_team_size=5,
        cross_border_payments_active=True,
    )

    results = engine.execute_waterfall(account)
    assert len(results) == 3  # Tier 1 + Tier 2 + Tier 3 (employee_count >= 10)
    assert account.enrichment_status == "fully_enriched"
    assert "registered_legal_entity" in account.metadata
    assert len(account.contacts) == 2
    assert engine.total_spend_usd > 0.20  # $0.002 + $0.05 + $0.15


def test_enrichment_waterfall_skips_contacts_for_micro_accounts():
    engine = EnrichmentWaterfallEngine()
    micro_account = TargetAccount(
        domain="micro-crypto.io",
        company_name="MicroCrypto",
        jurisdiction="UK",
        industry="Crypto",
        employee_count=5,  # Under 10
    )

    results = engine.execute_waterfall(micro_account)
    assert len(results) == 2  # Tier 1 + Tier 2 only
    assert len(micro_account.contacts) == 0


def test_defendable_scoring_engine_tier_1_enterprise():
    engine = DefendableAccountScoringEngine()
    high_risk_enterprise = TargetAccount(
        domain="global-remit.com",
        company_name="GlobalRemit",
        jurisdiction="UK",
        industry="Fintech Payments",
        estimated_arr_usd=50_000_000,
        employee_count=800,
        compliance_team_size=10,  # 10/800 = 0.0125 (< 0.03 -> understaffed, 25pts)
        recent_regulatory_penalties=2,  # 35pts
        cross_border_payments_active=True,  # 20pts
    )

    score = engine.score_account(high_risk_enterprise)
    assert score >= 75.0
    assert high_risk_enterprise.tier == AccountTier.TIER_1_ENTERPRISE
    assert "score_breakdown" in high_risk_enterprise.metadata


def test_defendable_scoring_engine_tier_3_nurture():
    engine = DefendableAccountScoringEngine()
    small_account = TargetAccount(
        domain="local-boutique.com",
        company_name="Local Boutique",
        jurisdiction="US",
        industry="Consulting",
        employee_count=20,
        compliance_team_size=3,  # 3/20 = 15% (well staffed)
        recent_regulatory_penalties=0,
        cross_border_payments_active=False,
    )

    score = engine.score_account(small_account)
    assert score < 50.0
    assert small_account.tier == AccountTier.TIER_3_NURTURE


def test_signal_outbound_play_runner():
    runner = SignalOutboundPlayRunner()
    account = TargetAccount(
        domain="remit-fast.co.uk",
        company_name="RemitFast",
        jurisdiction="UK",
        industry="Payments",
        contacts=[{"name": "Jane MLRO", "email": "jane@remit-fast.co.uk"}],
    )

    # 1. Enforcement signal
    enforce_signal = {
        "type": SignalType.REGULATORY_ENFORCEMENT.value,
        "authority": "FCA UK",
        "description": "FCA fines peer firm for sanctions monitoring failure",
    }
    play1 = runner.evaluate_and_trigger(account, enforce_signal)
    assert play1 is not None
    assert play1["play_name"] == "ENFORCEMENT_ACTION_FAST_TRACK"
    assert play1["primary_recipient"]["name"] == "Jane MLRO"

    # 2. Hiring surge signal
    surge_signal = {
        "type": SignalType.COMPLIANCE_HIRING_SURGE.value,
        "openings_count": 6,
    }
    play2 = runner.evaluate_and_trigger(account, surge_signal)
    assert play2 is not None
    assert play2["play_name"] == "COMPLIANCE_HIRING_SURGE_PLAY"
    assert len(runner.dispatched_plays) == 2


def test_llm_account_intelligence_agent_with_pii_redaction():
    agent = LLMAccountIntelligenceAgent(max_token_budget=1000)
    redacted = agent.redact_pii("Call compliance lead at 077-123-4567 or 123.456.7890 directly.")
    assert "[REDACTED_PHONE]" in redacted
    assert "077-123-4567" not in redacted

    account = TargetAccount(
        domain="tier1-neo.co.uk",
        company_name="Tier1 Neo",
        jurisdiction="UK",
        industry="Fintech",
        employee_count=450,
        compliance_team_size=8,
        score=88.5,
        tier=AccountTier.TIER_1_ENTERPRISE,
        contacts=[{"name": "Arthur Dent", "title": "Head of Financial Crime"}],
    )

    dossier = agent.generate_dossier(account)
    assert dossier["account_domain"] == "tier1-neo.co.uk"
    assert dossier["tier"] == AccountTier.TIER_1_ENTERPRISE.value
    assert dossier["guardrails_passed"] is True
    assert "85% of routine Level 1 AML false positives" in dossier["pitch_hook"]
    assert account.metadata["llm_dossier"] == dossier


def test_crm_webhook_sync_dispatcher_idempotency_and_dlq():
    dispatcher = CrmWebhookSyncDispatcher()
    account = TargetAccount(
        domain="fin-partner.uk",
        company_name="FinPartner UK",
        jurisdiction="UK",
        industry="Payments",
        score=82.0,
        tier=AccountTier.TIER_1_ENTERPRISE,
    )

    # First sync succeeds
    res1 = dispatcher.sync_to_crm(account, target_system="Salesforce", mock_network_success=True)
    assert res1["status"] == "SUCCESS"
    assert "001_" in res1["crm_id"]

    # Duplicate sync skipped idempotently
    res2 = dispatcher.sync_to_crm(account, target_system="Salesforce", mock_network_success=True)
    assert res2["status"] == "SKIPPED_DUPLICATE"

    # Simulated network failure routes payload to DLQ
    failing_account = TargetAccount(
        domain="retry-bank.co.uk",
        company_name="Retry Bank",
        jurisdiction="UK",
        industry="Banking",
        score=78.0,
    )
    res_fail = dispatcher.sync_to_crm(failing_account, target_system="Salesforce", mock_network_success=False)
    assert res_fail["status"] == "DLQ_QUEUED"
    assert len(dispatcher.dlq) == 1
    assert dispatcher.dlq[0]["payload"]["domain"] == "retry-bank.co.uk"
