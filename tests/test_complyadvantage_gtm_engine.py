"""Unit and integration tests for ComplyAdvantage Founding GTM Systems and Applied AI Engine."""

from __future__ import annotations

import pytest

from llm_control_plane.complyadvantage_gtm_engine import (
    AccountTier,
    BiDirectionalCRMSyncReconciler,
    ClaudeGTMPlaySynthesizer,
    ComplyAdvantageGTMEngine,
    EnrichedAccount,
    EnrichmentProvider,
    FinCrimeAccountScorer,
    SignalType,
    SyncAction,
    WaterfallEnricher,
)


class TestWaterfallEnricher:
    def test_enrich_requires_domain(self) -> None:
        enricher = WaterfallEnricher()
        with pytest.raises(ValueError, match="Domain is required"):
            enricher.enrich({"company_name": "TestCorp"})

    def test_enrich_waterfall_cost_and_fallback(self) -> None:
        enricher = WaterfallEnricher()

        # Provider 1: Free tier
        p1 = EnrichmentProvider(
            provider_id="free_registry",
            name="Free Public Registry",
            cost_per_query_usd=0.00,
            latency_ms=10.0,
            confidence_weight=0.7,
            handler=lambda d: {"employee_count": 120, "country": "United Kingdom"},
        )

        # Provider 2: Paid enrichment
        p2 = EnrichmentProvider(
            provider_id="clay_waterfall",
            name="Clay Waterfall Stack",
            cost_per_query_usd=0.05,
            latency_ms=80.0,
            confidence_weight=0.95,
            handler=lambda d: {
                "is_regulated_financial_institution": True,
                "detected_tech_stack": ["Salesforce", "legacy_aml", "HubSpot"],
            },
        )

        # Provider 3: Too expensive for budget
        p3 = EnrichmentProvider(
            provider_id="expensive_credit_bureau",
            name="Credit Bureau VIP",
            cost_per_query_usd=1.00,
            latency_ms=500.0,
            confidence_weight=0.99,
            handler=lambda d: {"credit_rating": "AAA"},
        )

        enricher.register_provider(p1)
        enricher.register_provider(p2)
        enricher.register_provider(p3)

        raw = {"domain": "monzo-challenger.co.uk", "company_name": "Monzo Challenger"}
        account = enricher.enrich(raw, max_budget_usd=0.10)

        assert account.domain == "monzo-challenger.co.uk"
        assert account.employee_count == 120
        assert account.country == "United Kingdom"
        assert account.is_regulated_financial_institution is True
        assert "legacy_aml" in account.detected_tech_stack
        assert account.total_enrichment_cost_usd == 0.05
        assert "free_registry" in account.data_sources
        assert "clay_waterfall" in account.data_sources
        assert "expensive_credit_bureau" not in account.data_sources

        # Check caching
        cached_account = enricher.enrich(raw)
        assert cached_account is account

    def test_enricher_gracefully_handles_provider_exception(self) -> None:
        enricher = WaterfallEnricher()

        def failing_handler(d):
            raise ConnectionError("API rate limit exceeded")

        p_fail = EnrichmentProvider(
            provider_id="failing_api",
            name="Failing API",
            cost_per_query_usd=0.01,
            latency_ms=50.0,
            confidence_weight=0.8,
            handler=failing_handler,
        )
        p_ok = EnrichmentProvider(
            provider_id="ok_api",
            name="OK API",
            cost_per_query_usd=0.02,
            latency_ms=20.0,
            confidence_weight=0.85,
            handler=lambda d: {"country": "Germany"},
        )

        enricher.register_provider(p_fail)
        enricher.register_provider(p_ok)

        account = enricher.enrich({"domain": "fintech-berlin.de"})
        assert account.country == "Germany"
        assert "ok_api" in account.data_sources
        assert "failing_api" not in account.data_sources


class TestFinCrimeAccountScorer:
    def test_score_tier_1_strategic_account(self) -> None:
        scorer = FinCrimeAccountScorer()
        account = EnrichedAccount(
            domain="revolut-peer.com",
            company_name="Revolut Peer Ltd",
            country="United Kingdom",
            employee_count=350,
            industry="Digital Banking / Payments",
            is_regulated_financial_institution=True,
            detected_tech_stack=["legacy_aml", "Salesforce"],
            intent_signals=[
                {"type": SignalType.COMPLIANCE_HIRING.value, "confidence": 1.0},
                {"type": SignalType.CROSS_BORDER_EXPANSION.value, "confidence": 0.9},
            ],
            data_sources=["clay_waterfall"],
            total_enrichment_cost_usd=0.05,
            confidence_score=0.9,
        )

        report = scorer.score(account)
        assert report.total_score >= 80.0
        assert report.tier == AccountTier.TIER_1_STRATEGIC
        assert "Direct Exec Alignment" in report.recommended_play
        assert SignalType.REGULATORY_LICENSING.value in report.signal_breakdown
        assert SignalType.TECH_STACK_LEGACY_AML.value in report.signal_breakdown
        assert "employee_scale_bonus" in report.signal_breakdown

    def test_score_disqualified_account(self) -> None:
        scorer = FinCrimeAccountScorer()
        account = EnrichedAccount(
            domain="small-bakery.local",
            company_name="Small Bakery",
            country="UK",
            employee_count=4,
            industry="Food & Beverage",
            is_regulated_financial_institution=False,
            detected_tech_stack=[],
            intent_signals=[],
            data_sources=["seed"],
            total_enrichment_cost_usd=0.0,
            confidence_score=0.5,
        )

        report = scorer.score(account)
        assert report.total_score == 0.0
        assert report.tier == AccountTier.DISQUALIFIED

    def test_update_weight_tracks_history(self) -> None:
        scorer = FinCrimeAccountScorer()
        scorer.update_weight(SignalType.FUNDING_ROUND.value, 20.0, "Increased predictive correlation in Q4")
        assert scorer.weights[SignalType.FUNDING_ROUND.value] == 20.0
        assert len(scorer.weight_history) == 1
        assert scorer.weight_history[0]["old_weight"] == 10.0


class TestClaudeGTMPlaySynthesizer:
    def test_synthesize_tier_1_strategic_play(self) -> None:
        synthesizer = ClaudeGTMPlaySynthesizer()
        account = EnrichedAccount(
            domain="checkout-competitor.com",
            company_name="Checkout Competitor",
            country="United Kingdom",
            employee_count=500,
            industry="Merchant Acquiring",
            is_regulated_financial_institution=True,
            detected_tech_stack=["legacy_aml"],
            intent_signals=[{"type": SignalType.REGULATORY_LICENSING.value, "confidence": 1.0}],
            data_sources=["seed"],
            total_enrichment_cost_usd=0.02,
            confidence_score=0.92,
        )
        scorer = FinCrimeAccountScorer()
        report = scorer.score(account)

        play = synthesizer.synthesize(account, report)
        assert play.domain == "checkout-competitor.com"
        assert "Chief Compliance Officer" in play.target_persona
        assert "Checkout Competitor" in play.hook
        assert play.guardrail_passed is True
        assert play.confidence_rating > 0.5


class TestBiDirectionalCRMSyncReconciler:
    def test_reconcile_insert_new_record(self) -> None:
        reconciler = BiDirectionalCRMSyncReconciler()
        account = EnrichedAccount(
            domain="new-fintech.io",
            company_name="New Fintech",
            country="UK",
            employee_count=75,
            industry="Fintech",
            is_regulated_financial_institution=True,
            detected_tech_stack=[],
            intent_signals=[],
            data_sources=["seed", "clay"],
            total_enrichment_cost_usd=0.01,
            confidence_score=0.8,
        )
        scorer = FinCrimeAccountScorer()
        report = scorer.score(account)

        action, payload = reconciler.reconcile_account(account, report, crm_record=None)
        assert action == SyncAction.INSERT
        assert payload["Domain__c"] == "new-fintech.io"
        assert payload["GTM_Intent_Score__c"] == report.total_score
        assert len(reconciler.audit_log) == 1

    def test_reconcile_skip_when_synced(self) -> None:
        reconciler = BiDirectionalCRMSyncReconciler()
        account = EnrichedAccount(
            domain="synced-fintech.io",
            company_name="Synced Fintech",
            country="UK",
            employee_count=75,
            industry="Fintech",
            is_regulated_financial_institution=True,
            detected_tech_stack=[],
            intent_signals=[],
            data_sources=["seed"],
            total_enrichment_cost_usd=0.0,
            confidence_score=0.8,
        )
        scorer = FinCrimeAccountScorer()
        report = scorer.score(account)

        crm_existing = {
            "Id": "001XXXXXXXXX",
            "Domain__c": "synced-fintech.io",
            "GTM_Intent_Score__c": report.total_score,
            "GTM_Account_Tier__c": report.tier.value,
            "Recommended_Play__c": report.recommended_play,
        }

        action, payload = reconciler.reconcile_account(account, report, crm_existing)
        assert action == SyncAction.SKIP
        assert payload == {}

    def test_reconcile_update_on_score_change(self) -> None:
        reconciler = BiDirectionalCRMSyncReconciler()
        account = EnrichedAccount(
            domain="updating-fintech.io",
            company_name="Updating Fintech",
            country="UK",
            employee_count=150,
            industry="Fintech",
            is_regulated_financial_institution=True,
            detected_tech_stack=["legacy_aml"],
            intent_signals=[{"type": SignalType.REGULATORY_LICENSING.value, "confidence": 1.0}],
            data_sources=["seed"],
            total_enrichment_cost_usd=0.0,
            confidence_score=0.85,
        )
        scorer = FinCrimeAccountScorer()
        report = scorer.score(account)

        crm_existing = {
            "Id": "001YYYYYYYYY",
            "Domain__c": "updating-fintech.io",
            "GTM_Intent_Score__c": 20.0,  # Old score
            "GTM_Account_Tier__c": AccountTier.TIER_3_NURTURE.value,
            "Recommended_Play__c": "Old Play",
        }

        action, payload = reconciler.reconcile_account(account, report, crm_existing)
        assert action == SyncAction.UPDATE
        assert payload["GTM_Intent_Score__c"] == report.total_score
        assert payload["GTM_Account_Tier__c"] == report.tier.value

    def test_dead_letter_queue_capture(self) -> None:
        reconciler = BiDirectionalCRMSyncReconciler()
        reconciler.handle_sync_error("error-corp.com", {"Domain__c": "error-corp.com"}, "Salesforce API 503")
        assert len(reconciler.dead_letter_queue) == 1
        assert reconciler.dead_letter_queue[0]["domain"] == "error-corp.com"


class TestComplyAdvantageGTMEngineEndToEnd:
    def test_full_pipeline_flow(self) -> None:
        engine = ComplyAdvantageGTMEngine()

        # Add provider to waterfall
        engine.enricher.register_provider(
            EnrichmentProvider(
                provider_id="fca_register",
                name="FCA Register Scraper",
                cost_per_query_usd=0.01,
                latency_ms=15.0,
                confidence_weight=0.98,
                handler=lambda d: {
                    "is_regulated_financial_institution": True,
                    "country": "United Kingdom",
                    "detected_tech_stack": ["legacy_aml", "Salesforce"],
                },
            )
        )

        raw = {
            "domain": "currensea-global.co.uk",
            "company_name": "Currensea Global",
            "employee_count": 80,
            "intent_signals": [
                {"type": SignalType.CROSS_BORDER_EXPANSION.value, "confidence": 1.0},
                {"type": SignalType.COMPLIANCE_HIRING.value, "confidence": 0.95},
            ],
        }

        result = engine.process_target_account(raw)
        assert result["account"].domain == "currensea-global.co.uk"
        assert result["account"].is_regulated_financial_institution is True
        assert result["score"].tier == AccountTier.TIER_1_STRATEGIC
        assert result["score"].total_score >= 80.0
        assert result["play"].guardrail_passed is True
        assert result["sync_action"] == SyncAction.INSERT.value
        assert "Domain__c" in result["sync_payload"]
