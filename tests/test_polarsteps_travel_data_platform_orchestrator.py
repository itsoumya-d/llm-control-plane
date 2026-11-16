"""
Unit tests for Polarsteps Travel Data Platform & Analytics Orchestrator.
"""

from __future__ import annotations

from llm_control_plane.polarsteps_travel_data_platform_orchestrator import (
    DbtAnalyticsMartSynthesizer,
    LLMAnalyticalQueryGuardrail,
    PolarstepsTravelDataPlatformOrchestrator,
    PrivacyPreservingLocationQuantizer,
    SchemaDriftSeverity,
    TravelEventIngestionDeduplicator,
    TravelEventType,
    TravelTelemetryEvent,
)


def test_raw_event_fingerprint_deterministic():
    ev1 = TravelTelemetryEvent(
        event_id="ev_001",
        traveler_id="user_123",
        device_id="dev_abc",
        event_type=TravelEventType.STEP_RECORDED,
        client_timestamp_ms=1700000000000,
        latitude=52.370216,
        longitude=4.895168,
    )
    ev2 = TravelTelemetryEvent(
        event_id="ev_002",
        traveler_id="user_123",
        device_id="dev_xyz",
        event_type=TravelEventType.STEP_RECORDED,
        client_timestamp_ms=1700000000000,
        latitude=52.370216,
        longitude=4.895168,
    )
    assert ev1.compute_fingerprint() == ev2.compute_fingerprint()


def test_gdpr_coordinate_quantizer():
    quantizer = PrivacyPreservingLocationQuantizer(salt="secret_salt_2026", precision_decimals=2)
    ev = TravelTelemetryEvent(
        event_id="ev_001",
        traveler_id="user_john_doe",
        device_id="device_iphone15",
        event_type=TravelEventType.STEP_RECORDED,
        client_timestamp_ms=1700000000000,
        latitude=52.3791823,
        longitude=4.9003819,
        metadata={"step_count": 120, "exact_address": "Damrak 1, Amsterdam", "country_code": "NL"},
    )
    quantized = quantizer.process(ev)
    assert quantized.quantized_lat == 52.37
    assert quantized.quantized_lon == 4.90
    assert quantized.pseudonymized_traveler_id != "user_john_doe"
    assert len(quantized.pseudonymized_traveler_id) == 16
    assert "exact_address" not in quantized.properties
    assert quantized.properties["country_code"] == "NL"


def test_event_deduplication_engine():
    engine = TravelEventIngestionDeduplicator(max_seen_entries=10)
    ev = TravelTelemetryEvent(
        event_id="ev_001",
        traveler_id="u1",
        device_id="d1",
        event_type=TravelEventType.STEP_RECORDED,
        client_timestamp_ms=1700000000000,
        latitude=52.37,
        longitude=4.89,
    )

    ok1, _ = engine.ingest(ev)
    assert ok1 is True

    ok2, _ = engine.ingest(ev)
    assert ok2 is False
    assert engine.total_deduplicated == 1


def test_deduplication_engine_sliding_window_eviction():
    engine = TravelEventIngestionDeduplicator(max_seen_entries=2)
    events = [
        TravelTelemetryEvent(f"e{i}", f"u{i}", "d", TravelEventType.STEP_RECORDED, 1700000000000 + i, 52.0 + i, 4.0)
        for i in range(3)
    ]
    for ev in events:
        engine.ingest(ev)

    # First event should now be evicted from seen set
    re_ev0, _ = engine.ingest(events[0])
    assert re_ev0 is True


def test_data_contract_validator_compatible():
    synthesizer = DbtAnalyticsMartSynthesizer()
    metadata = {"altitude_m": 12.5, "step_count": 450}
    severity, issues = synthesizer.validate_contract(TravelEventType.STEP_RECORDED, metadata)
    assert severity == SchemaDriftSeverity.COMPATIBLE
    assert len(issues) == 0


def test_data_contract_validator_missing_required():
    synthesizer = DbtAnalyticsMartSynthesizer()
    metadata = {"altitude_m": 12.5}
    severity, issues = synthesizer.validate_contract(TravelEventType.STEP_RECORDED, metadata)
    assert severity == SchemaDriftSeverity.BREAKING_MISSING_REQUIRED
    assert any("step_count" in i for i in issues)


def test_data_contract_validator_type_mismatch():
    synthesizer = DbtAnalyticsMartSynthesizer()
    metadata = {"photo_count": "three", "has_caption": True}
    severity, issues = synthesizer.validate_contract(TravelEventType.PHOTO_ATTACHED, metadata)
    assert severity == SchemaDriftSeverity.BREAKING_TYPE_MISMATCH
    assert any("photo_count" in i for i in issues)


def test_dbt_traveler_engagement_mart():
    quantizer = PrivacyPreservingLocationQuantizer(salt="salt_test", precision_decimals=2)
    ev1 = quantizer.process(
        TravelTelemetryEvent(
            "e1", "u1", "d1", TravelEventType.STEP_RECORDED, 1000, 52.37, 4.89,
            {"step_count": 100, "trip_id": "trip_alps"}
        )
    )
    ev2 = quantizer.process(
        TravelTelemetryEvent(
            "e2", "u1", "d1", TravelEventType.PHOTO_ATTACHED, 5000, 52.38, 4.90,
            {"photo_count": 3, "trip_id": "trip_alps"}
        )
    )
    ev3 = quantizer.process(
        TravelTelemetryEvent(
            "e3", "u1", "d1", TravelEventType.TRIP_PUBLISHED, 9000, 52.39, 4.91,
            {"trip_id": "trip_alps"}
        )
    )

    synthesizer = DbtAnalyticsMartSynthesizer()
    synthesizer.load_staging([ev1, ev2, ev3])
    marts = synthesizer.build_traveler_engagement_mart()

    assert len(marts) == 1
    row = marts[0]
    assert row["trip_id"] == "trip_alps"
    assert row["total_steps"] == 100
    assert row["total_photos"] == 3
    assert row["is_published"] is True


def test_dbt_book_conversion_funnel_mart():
    quantizer = PrivacyPreservingLocationQuantizer(salt="salt_funnel", precision_decimals=2)
    ev_preview1 = quantizer.process(
        TravelTelemetryEvent("p1", "u1", "d1", TravelEventType.TRAVEL_BOOK_PREVIEWED, 1000, 52.0, 4.0)
    )
    ev_preview2 = quantizer.process(
        TravelTelemetryEvent("p2", "u2", "d1", TravelEventType.TRAVEL_BOOK_PREVIEWED, 2000, 52.0, 4.0)
    )
    ev_order1 = quantizer.process(
        TravelTelemetryEvent(
            "o1", "u1", "d1", TravelEventType.TRAVEL_BOOK_ORDERED, 3000, 52.0, 4.0, {"gross_amount": 49.50}
        )
    )

    synthesizer = DbtAnalyticsMartSynthesizer()
    synthesizer.load_staging([ev_preview1, ev_preview2, ev_order1])
    funnel = synthesizer.build_book_conversion_funnel_mart()

    assert funnel["total_previews"] == 2
    assert funnel["total_orders"] == 1
    assert funnel["conversion_rate_pct"] == 50.0
    assert funnel["gross_gmv_eur"] == 49.50


def test_ai_query_privacy_guardrail_prompt_audit():
    ok, err = LLMAnalyticalQueryGuardrail.audit_llm_prompt("Summarize top travel countries for user u123")
    assert ok is True
    assert err is None

    bad_ok, bad_err = LLMAnalyticalQueryGuardrail.audit_llm_prompt("Retrieve exact_lat and exact_lon for traveler")
    assert bad_ok is False
    assert "exact_lat" in bad_err


def test_ai_query_privacy_guardrail_sanitize_context():
    context = {
        "traveler_id": "anon_9823",
        "home_address": "Secret Villa, Keizersgracht, Amsterdam",
        "nested": {
            "ip_address": "192.168.1.1",
            "country": "NL",
        },
        "steps": 4200,
    }
    sanitized = LLMAnalyticalQueryGuardrail.sanitize_llm_query_context(context)
    assert "home_address" not in sanitized
    assert "ip_address" not in sanitized["nested"]
    assert sanitized["nested"]["country"] == "NL"
    assert sanitized["steps"] == 4200


def test_polarsteps_orchestrator_end_to_end():
    orchestrator = PolarstepsTravelDataPlatformOrchestrator(salt="e2e_salt")
    batch = [
        TravelTelemetryEvent(
            "ev1", "user_a", "d1", TravelEventType.STEP_RECORDED, 1000, 52.37, 4.89,
            {"step_count": 50, "trip_id": "t1"}
        ),
        TravelTelemetryEvent(
            "ev1_dup", "user_a", "d1", TravelEventType.STEP_RECORDED, 1000, 52.37, 4.89,
            {"step_count": 50, "trip_id": "t1"}
        ),
        TravelTelemetryEvent(
            "ev2", "user_a", "d1", TravelEventType.TRAVEL_BOOK_PREVIEWED, 2000, 52.38, 4.90,
            {"trip_id": "t1"}
        ),
        TravelTelemetryEvent(
            "ev3", "user_a", "d1", TravelEventType.TRAVEL_BOOK_ORDERED, 3000, 52.39, 4.91,
            {"gross_amount": 55.0, "trip_id": "t1"}
        ),
    ]
    res = orchestrator.process_incoming_batch(batch)
    assert res["total_received"] == 4
    assert res["accepted_count"] == 3
    assert res["deduplicated_count"] == 1

    marts = orchestrator.synthesizer.build_traveler_engagement_mart()
    assert len(marts) == 1
    assert marts[0]["total_steps"] == 50

    funnel = orchestrator.synthesizer.build_book_conversion_funnel_mart()
    assert funnel["total_orders"] == 1
    assert funnel["gross_gmv_eur"] == 55.0
