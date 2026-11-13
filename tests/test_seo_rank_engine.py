"""Tests for SEO Rank Engine, SERP Tracker, Core Web Vitals Auditor, and Intent Clusterer."""

from __future__ import annotations

import hashlib
import json

from llm_control_plane.seo_rank_engine import (
    ClientSEOAnalyticsAggregator,
    CoreWebVitalsAuditor,
    DeviceType,
    MetricGrade,
    ProxyConfig,
    SearchIntent,
    SemanticKeywordClusterer,
    SERPRankTracker,
    SERPResult,
    SERPSnapshot,
)


def test_proxy_config_and_rotation():
    p1 = ProxyConfig(host="proxy1.local", port=8080)
    p2 = ProxyConfig(host="proxy2.local", port=8080, auth="user:pass")

    assert p1.url == "https://proxy1.local:8080"
    assert p2.url == "https://user:pass@proxy2.local:8080"

    tracker = SERPRankTracker(proxies=[p1, p2])
    assert tracker.get_next_proxy() == p1
    assert tracker.get_next_proxy() == p2
    assert tracker.get_next_proxy() == p1


def test_user_agent_and_backoff():
    tracker = SERPRankTracker()
    desktop_ua = tracker.get_random_user_agent(DeviceType.DESKTOP)
    mobile_ua = tracker.get_random_user_agent(DeviceType.MOBILE)

    assert isinstance(desktop_ua, str) and len(desktop_ua) > 10
    assert isinstance(mobile_ua, str) and len(mobile_ua) > 10

    # Backoff with full jitter must stay within bounds [0.0, min(max_sec, base * 2^(attempt-1))]
    for attempt in range(1, 6):
        delay = tracker.calculate_backoff_delay(attempt, base_sec=1.0, max_sec=16.0)
        assert 0.0 <= delay <= min(16.0, 1.0 * (2 ** (attempt - 1)))


def test_serp_html_parsing_standard():
    mock_html = """
    <html>
        <body>
            <div class="g">
                <a href="https://example.co.nz/services/seo">
                    <h3>Leading Auckland SEO Services | Example</h3>
                </a>
                <div class="VwiC3b">Award-winning organic search optimisation in Auckland.</div>
            </div>
            <div class="g">
                <a href="https://justseo.co.nz/pricing">
                    <h3>JustSEO Agency Auckland</h3>
                </a>
                <div class="VwiC3b">Data-driven search growth for Kiwi enterprises.</div>
            </div>
        </body>
    </html>
    """
    tracker = SERPRankTracker()
    snapshot = tracker.parse_serp_html(mock_html, keyword="seo auckland", device=DeviceType.DESKTOP)

    assert len(snapshot.results) == 2
    assert snapshot.results[0].rank == 1
    assert snapshot.results[0].domain == "example.co.nz"
    assert snapshot.results[0].title == "Leading Auckland SEO Services | Example"
    assert snapshot.results[1].rank == 2
    assert snapshot.results[1].domain == "justseo.co.nz"


def test_serp_html_parsing_fallback():
    mock_html = """
    <div class="search-wrap">
        <a class="result-link" href="https://competitor.co.nz/audit">Audit Tool</a>
        <p class="result-desc">Free technical SEO audit.</p>
    </div>
    """
    tracker = SERPRankTracker()
    snapshot = tracker.parse_serp_html(mock_html, keyword="free seo audit", device=DeviceType.MOBILE)

    assert len(snapshot.results) == 1
    assert snapshot.results[0].rank == 1
    assert snapshot.results[0].domain == "competitor.co.nz"
    assert snapshot.results[0].title == "Audit Tool"


def test_track_rank_delta():
    tracker = SERPRankTracker()
    snapshot = SERPSnapshot(
        keyword="enterprise seo nz",
        device=DeviceType.DESKTOP,
        timestamp=1700000000.0,
        results=[
            SERPResult(rank=1, title="Alpha", url="https://alpha.nz", snippet="...", domain="alpha.nz"),
            SERPResult(
                rank=2,
                title="JustSEO",
                url="https://justseo.co.nz/enterprise",
                snippet="...",
                domain="justseo.co.nz",
            ),
        ],
    )

    # Improved from rank 5 to rank 2 -> delta +3
    perf = tracker.track_rank(target_domain="justseo.co.nz", snapshot=snapshot, previous_rank=5)
    assert perf["is_ranked"] is True
    assert perf["current_rank"] == 2
    assert perf["previous_rank"] == 5
    assert perf["rank_delta"] == 3
    assert perf["is_top_3"] is True
    assert perf["is_first_page"] is True

    # Test unranked domain
    unranked = tracker.track_rank(target_domain="missingdomain.nz", snapshot=snapshot, previous_rank=4)
    assert unranked["is_ranked"] is False
    assert unranked["current_rank"] is None
    assert unranked["rank_delta"] is None


def test_core_web_vitals_grading():
    assert CoreWebVitalsAuditor.grade_metric("lcp_ms", 2100.0) == MetricGrade.GOOD
    assert CoreWebVitalsAuditor.grade_metric("lcp_ms", 3200.0) == MetricGrade.NEEDS_IMPROVEMENT
    assert CoreWebVitalsAuditor.grade_metric("lcp_ms", 4500.0) == MetricGrade.POOR

    assert CoreWebVitalsAuditor.grade_metric("cls", 0.05) == MetricGrade.GOOD
    assert CoreWebVitalsAuditor.grade_metric("cls", 0.18) == MetricGrade.NEEDS_IMPROVEMENT
    assert CoreWebVitalsAuditor.grade_metric("cls", 0.35) == MetricGrade.POOR


def test_core_web_vitals_audit_perfect():
    audit = CoreWebVitalsAuditor.audit_url(
        url="https://justseo.co.nz",
        lcp_ms=1800.0,
        cls_val=0.04,
        inp_ms=120.0,
        fcp_ms=1100.0,
        ttfb_ms=350.0,
    )
    assert audit["passed_core_web_vitals"] is True
    assert audit["overall_health_score"] == 100.0
    assert len(audit["remediation_actions"]) == 0


def test_core_web_vitals_audit_remediation():
    audit = CoreWebVitalsAuditor.audit_url(
        url="https://client-site.co.nz",
        lcp_ms=4200.0,
        cls_val=0.32,
        inp_ms=650.0,
        fcp_ms=3200.0,
        ttfb_ms=2100.0,
    )
    assert audit["passed_core_web_vitals"] is False
    assert audit["overall_health_score"] < 50.0
    assert len(audit["remediation_actions"]) == 4
    assert any("LCP is 4200.0ms" in r for r in audit["remediation_actions"])
    assert any("CLS is 0.320" in r for r in audit["remediation_actions"])
    assert any("INP is 650.0ms" in r for r in audit["remediation_actions"])
    assert any("TTFB is 2100.0ms" in r for r in audit["remediation_actions"])


def test_search_intent_classification():
    assert SemanticKeywordClusterer.classify_intent("buy seo package auckland") == SearchIntent.TRANSACTIONAL
    assert SemanticKeywordClusterer.classify_intent("seo quote nz") == SearchIntent.TRANSACTIONAL
    assert SemanticKeywordClusterer.classify_intent("best seo companies auckland review") == SearchIntent.COMMERCIAL
    assert SemanticKeywordClusterer.classify_intent("justseo client portal login") == SearchIntent.NAVIGATIONAL
    assert (
        SemanticKeywordClusterer.classify_intent("how to rank higher on google maps nz")
        == SearchIntent.INFORMATIONAL
    )
    assert SemanticKeywordClusterer.classify_intent("what is cumulative layout shift") == SearchIntent.INFORMATIONAL


def test_semantic_keyword_clustering():
    keywords = [
        "seo consultant auckland",
        "best seo consultant in auckland",
        "top auckland seo consultant",
        "ecommerce web design wellington",
        "custom ecommerce website wellington",
        "core web vitals audit service",
    ]
    clusters = SemanticKeywordClusterer.cluster_keywords(keywords, similarity_threshold=0.35)
    assert len(clusters) >= 2
    # Verify related keywords are grouped together
    auckland_seo_group = None
    for members in clusters.values():
        if "seo consultant auckland" in members:
            auckland_seo_group = members
            break
    assert auckland_seo_group is not None
    assert "best seo consultant in auckland" in auckland_seo_group


def test_cannibalization_detection():
    rankings = [
        {"keyword": "auckland seo services", "url": "https://site.nz/seo-services", "rank": 4},
        {"keyword": "auckland seo services", "url": "https://site.nz/blog/auckland-seo", "rank": 9},
        {"keyword": "contact seo team", "url": "https://site.nz/contact", "rank": 1},
    ]
    cannibalized = SemanticKeywordClusterer.detect_cannibalization(rankings)
    assert len(cannibalized) == 1
    assert cannibalized[0]["keyword"] == "auckland seo services"
    assert cannibalized[0]["competing_url_count"] == 2
    assert cannibalized[0]["severity"] == "HIGH"


def test_visibility_index_calculation():
    rankings = [
        {"keyword": "seo nz", "rank": 1, "search_volume": 1000},
        {"keyword": "seo agency", "rank": 2, "search_volume": 1000},
        {"keyword": "seo audit", "rank": 3, "search_volume": 1000},
        {"keyword": "unranked kw", "rank": 25, "search_volume": 1000},
    ]
    # rank 1: 0.316 * 1.0, rank 2: 0.157 * 1.0, rank 3: 0.103 * 1.0 = 0.576 -> rounded 0.58
    vis = ClientSEOAnalyticsAggregator.calculate_visibility_index(rankings)
    assert vis == 0.58


def test_rank_velocity_tracking():
    prev = [
        {"keyword": "kw1", "rank": 10},
        {"keyword": "kw2", "rank": 5},
        {"keyword": "kw3", "rank": 2},
    ]
    curr = [
        {"keyword": "kw1", "rank": 6},   # improved +4
        {"keyword": "kw2", "rank": 8},   # declined -3
        {"keyword": "kw3", "rank": 2},   # unchanged 0
    ]
    velocity = ClientSEOAnalyticsAggregator.calculate_rank_velocity(curr, prev)
    assert velocity["keywords_evaluated"] == 3
    assert velocity["net_rank_delta"] == 1
    assert velocity["positions_improved"] == 1
    assert velocity["positions_declined"] == 1
    assert velocity["positions_stable"] == 1
    assert velocity["velocity_trend"] == "ACCELERATING"


def test_audit_report_generation_and_hash():
    rankings = [{"keyword": "seo agency auckland", "rank": 2, "search_volume": 2500}]
    cwv = CoreWebVitalsAuditor.audit_url("https://client.nz", 1900.0, 0.05, 140.0, 1200.0, 400.0)
    clusters = {"seo agency": ["seo agency auckland"]}

    report = ClientSEOAnalyticsAggregator.generate_audit_report(
        client_name="Tedtech Client Alpha",
        target_domain="client.nz",
        rankings=rankings,
        cwv_audit=cwv,
        clusters=clusters,
        previous_hash="GENESIS_HASH",
    )

    assert "report_id" in report
    assert "report_hash" in report
    assert report["previous_hash"] == "GENESIS_HASH"
    assert report["summary"]["client"] == "Tedtech Client Alpha"
    assert report["summary"]["domain"] == "client.nz"
    assert report["summary"]["cwv_health_score"] == 100.0

    # Cryptographic integrity verification
    canonical_bytes = json.dumps(report["data"], sort_keys=True, separators=(",", ":")).encode("utf-8")
    expected_hash = hashlib.sha256(canonical_bytes).hexdigest()
    assert report["report_hash"] == expected_hash
