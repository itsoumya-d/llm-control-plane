"""Unit tests for Tavily Agentic Web Access Engine & HTTP/2 TLS Fingerprint Orchestrator."""

import pytest

from llm_control_plane.tavily_web_access_engine import (
    AdaptiveFetchScheduler,
    ChallengeType,
    ClientVerificationClassifier,
    FetchMethod,
    HTTP2StreamProfile,
    TavilyWebAccessEngine,
    TLSFingerprintSpec,
)


def test_tls_fingerprint_generation():
    """Verify JA3 and JA4 fingerprint computation against expected structure."""
    spec = TLSFingerprintSpec(
        browser_profile="chrome_desktop_128",
        tls_version="TLS 1.3",
        ciphers=[0x1301, 0x1302, 0x1303, 0xC02B, 0xC02F],
        extensions=[0x0000, 0x0017, 0xFF01, 0x000A, 0x000B],
        supported_groups=[0x001D, 0x0017],
        signature_algorithms=[0x0403, 0x0804],
        alpn_protocols=["h2", "http/1.1"],
    )
    ja3 = spec.compute_ja3()
    ja4 = spec.compute_ja4()

    assert len(ja3) == 32
    assert ja4.startswith("t13d0505h2_")
    assert len(ja4.split("_")) == 3


def test_http2_stream_profile_defaults():
    """Verify HTTP/2 profile enforces pseudo-header order and window parameters."""
    profile = HTTP2StreamProfile()
    assert profile.pseudo_header_order == (":method", ":authority", ":scheme", ":path")
    assert profile.header_table_size == 65536
    assert profile.enable_push is False
    assert profile.max_concurrent_streams == 1000
    assert profile.connection_window_update == 15663105


def test_classifier_cloudflare_turnstile():
    """Detect Cloudflare Turnstile anti-bot challenges."""
    classifier = ClientVerificationClassifier()
    html_challenge = """
    <html>
      <head><title>Just a moment...</title></head>
      <body>
        <div id="cf-turnstile-wrapper" class="cf-turnstile"></div>
        <script src="https://challenges.cloudflare.com/turnstile/v0/api.js"></script>
      </body>
    </html>
    """
    diag = classifier.diagnose(
        status_code=403,
        headers={"server": "cloudflare", "cf-ray": "8e3c1294819a-LHR"},
        body=html_challenge,
    )
    assert diag.requires_browser is True
    assert diag.challenge_type == ChallengeType.CLOUDFLARE_TURNSTILE
    assert diag.recommended_fetch_method == FetchMethod.CDP_HEADLESS_BROWSER
    assert diag.confidence >= 0.95


def test_classifier_datadome_shield():
    """Detect Datadome challenge signatures."""
    classifier = ClientVerificationClassifier()
    diag = classifier.diagnose(
        status_code=403,
        headers={"x-datadome": "protected", "content-type": "text/html"},
        body="<html><body>Please solve the captcha at captcha-delivery.com</body></html>",
    )
    assert diag.requires_browser is True
    assert diag.challenge_type == ChallengeType.DATADOME_SHIELD
    assert diag.recommended_fetch_method == FetchMethod.CDP_HEADLESS_BROWSER


def test_classifier_akamai_bot_manager():
    """Detect Akamai Bot Manager challenge signatures."""
    classifier = ClientVerificationClassifier()
    diag = classifier.diagnose(
        status_code=403,
        headers={"server": "AkamaiGHost"},
        body="<html><script>var _abck = 'test_token'; sec-cpt challenge</script></html>",
    )
    assert diag.requires_browser is True
    assert diag.challenge_type == ChallengeType.AKAMAI_BOT_MANAGER


def test_classifier_js_hydration_shell():
    """Detect 200 OK responses with empty JavaScript hydration root."""
    classifier = ClientVerificationClassifier()
    diag = classifier.diagnose(
        status_code=200,
        headers={"content-type": "text/html"},
        body='<!DOCTYPE html><html><body><div id="root"></div><script src="/bundle.js"></script></body></html>',
    )
    assert diag.requires_browser is True
    assert diag.challenge_type == ChallengeType.JS_HYDRATION_SHELL
    assert diag.recommended_fetch_method == FetchMethod.CDP_HEADLESS_BROWSER


def test_classifier_clean_response():
    """Verify standard informative responses require no browser escalation."""
    classifier = ClientVerificationClassifier()
    diag = classifier.diagnose(
        status_code=200,
        headers={"content-type": "text/html"},
        body="<html><body><h1>API Documentation</h1><p>Comprehensive guide to Tavily Search API.</p></body></html>",
    )
    assert diag.requires_browser is False
    assert diag.challenge_type == ChallengeType.NONE
    assert diag.recommended_fetch_method == FetchMethod.LIGHTWEIGHT_HTTP2


def test_extract_clean_markdown():
    """Verify semantic markdown extraction strips noise and retains headers."""
    engine = TavilyWebAccessEngine()
    raw_html = """
    <html>
      <head>
        <style>.ads { color: red; }</style>
        <script>console.log("tracking");</script>
      </head>
      <body>
        <h1>Tavily Search API</h1>
        <p>Real-time search built for AI agents.</p>
        <div class="footer"><p>Copyright 2026 Tavily Inc.</p></div>
      </body>
    </html>
    """
    md = engine.extract_clean_markdown(raw_html)
    assert "# Tavily Search API" in md
    assert "Real-time search built for AI agents." in md
    assert "console.log" not in md
    assert ".ads" not in md


def test_engine_fetch_lightweight_http2():
    """Verify standard fetch operation via lightweight HTTP/2 transport."""
    engine = TavilyWebAccessEngine()
    result = engine.fetch("https://docs.tavily.com/overview")

    assert result.status_code == 200
    assert result.fetch_method == FetchMethod.LIGHTWEIGHT_HTTP2
    assert result.challenge_bypassed is False
    assert result.clean_markdown_bytes > 0
    assert result.semantic_yield_ratio > 0.0
    assert len(result.audit_hash) == 64


def test_engine_fetch_challenge_escalation():
    """Verify automatic fallback to CDP Headless Browser upon challenge."""
    engine = TavilyWebAccessEngine()
    cf_challenge = (
        403,
        {"server": "cloudflare", "cf-ray": "8e3c1294819a-LHR"},
        "<html><head><title>Just a moment...</title></head><body><div class='cf-turnstile'></div></body></html>",
    )
    result = engine.fetch("https://protected-research.org/paper", simulated_response=cf_challenge)

    assert result.status_code == 200
    assert result.fetch_method == FetchMethod.CDP_HEADLESS_BROWSER
    assert result.challenge_bypassed is True
    assert "CDP Rendered Content" in result.target_url or result.clean_markdown_bytes > 0


def test_engine_invalid_url():
    """Verify invalid URL triggers descriptive ValueError."""
    engine = TavilyWebAccessEngine()
    with pytest.raises(ValueError, match="Invalid target URL"):
        engine.fetch("invalid_url_without_scheme")


def test_adaptive_fetch_scheduler():
    """Verify scheduler caches optimal method and rotates proxies."""
    engine = TavilyWebAccessEngine()
    scheduler = AdaptiveFetchScheduler(engine=engine)

    # 1. Test proxy rotation
    p1 = scheduler.get_next_proxy()
    p2 = scheduler.get_next_proxy()
    p3 = scheduler.get_next_proxy()
    p4 = scheduler.get_next_proxy()
    assert p1 != p2
    assert p2 != p3
    assert p1 == p4  # wrap-around

    # 2. Challenge on first fetch triggers browser caching
    cf_challenge = (
        403,
        {"server": "cloudflare", "cf-ray": "8e3c1294819a-LHR"},
        "<html><body><div class='cf-turnstile'></div></body></html>",
    )
    res1 = scheduler.schedule_fetch("https://spatarget.com/app", simulated_response=cf_challenge)
    assert res1.fetch_method == FetchMethod.CDP_HEADLESS_BROWSER
    assert scheduler.host_method_cache["spatarget.com"] == FetchMethod.CDP_HEADLESS_BROWSER

    # 3. Second fetch reuses cached browser method automatically
    res2 = scheduler.schedule_fetch("https://spatarget.com/dashboard")
    assert res2.fetch_method == FetchMethod.CDP_HEADLESS_BROWSER

    stats = scheduler.host_stats["spatarget.com"]
    assert stats["requests"] == 2
    assert stats["browser_escalations"] == 2
