"""Tests for Verve Yield Optimization & Low-Latency Auction Inference Engine."""

import time
from llm_control_plane.verve_yield_optimization_engine import (
    AuctionBidRequest,
    AuctionFeatureDriftMonitor,
    ShadowAuctionEvalHarness,
    TritonInferenceClientSimulator,
    VerveYieldOptimizationService,
)


def make_request(auction_id: str = "auc_101", ecpm: float = 240.0) -> AuctionBidRequest:
    return AuctionBidRequest(
        auction_id=auction_id,
        publisher_id="pub_dublin_01",
        ad_slot="interstitial_320x480",
        device_type="mobile_app",
        geo_country="IE",
        timestamp=time.time(),
        historical_ecpm_cents=ecpm,
        device_ctr_velocity=1.05,
        advertiser_budget_pacing=1.1,
        floor_price_cents=60.0,
    )


def test_single_auction_scoring_success() -> None:
    service = VerveYieldOptimizationService(sla_timeout_ms=10.0)
    req = make_request()
    res = service.score_auction(req)

    assert res.auction_id == "auc_101"
    assert not res.is_fallback
    assert res.predicted_bid_floor_cents >= 60.0
    assert res.recommended_bid_cents > res.predicted_bid_floor_cents
    assert res.inference_latency_ms < 10.0
    assert res.metadata["model_engine"] == "nvidia_triton_v2"


def test_batch_auction_scoring_micro_batching() -> None:
    service = VerveYieldOptimizationService()
    requests = [make_request(f"auc_{i}", 200.0 + i * 5.0) for i in range(16)]

    results = service.score_auction_batch(requests)
    assert len(results) == 16

    batch_ids = {r.micro_batch_id for r in results}
    assert len(batch_ids) == 1  # All belong to single micro-batch
    assert all(not r.is_fallback for r in results)


def test_drift_monitor_detects_distribution_shift() -> None:
    monitor = AuctionFeatureDriftMonitor(
        baseline_ecpm_mean=200.0,
        baseline_ecpm_std=40.0,
        drift_threshold_sigma=2.0,
    )

    # Normal baseline traffic
    drift_seen = False
    for _ in range(15):
        if monitor.record_observation(205.0):
            drift_seen = True
    assert not drift_seen

    # Surge of anomalous high-yield bid requests (drift injection)
    anomalies_detected = []
    for _ in range(25):
        anomalies_detected.append(monitor.record_observation(450.0))

    assert any(anomalies_detected)
    assert monitor.current_stats["count"] == 40.0
    assert monitor.current_stats["online_mean"] > 250.0


def test_latency_sla_fallback_guard() -> None:
    # Service with ultra-tight SLA that forces fallback
    class SlowTritonSimulator(TritonInferenceClientSimulator):
        def score_batch(self, requests, drift_detected):
            time.sleep(0.01)
            return super().score_batch(requests, drift_detected)

    service = VerveYieldOptimizationService(
        sla_timeout_ms=0.001,  # 1 microsecond timeout guarantees fallback
        triton_client=SlowTritonSimulator(),
    )
    req = make_request("auc_slow")
    res = service.score_auction(req)

    assert res.is_fallback
    assert res.metadata["fallback_reason"] == "sla_budget_exceeded"
    assert res.predicted_bid_floor_cents == req.floor_price_cents


def test_shadow_auction_eval_harness() -> None:
    service = VerveYieldOptimizationService(sla_timeout_ms=20.0)
    harness = ShadowAuctionEvalHarness(service)

    logs = [
        {
            "auction_id": f"hist_{i}",
            "publisher_id": "pub_irish_times",
            "ad_slot": "leaderboard_728x90",
            "device_type": "desktop",
            "geo_country": "IE",
            "timestamp": time.time(),
            "historical_ecpm_cents": 220.0,
            "device_ctr_velocity": 1.1,
            "advertiser_budget_pacing": 1.0,
            "floor_price_cents": 50.0,
            "historical_clearing_price_cents": 190.0,
            "winning_bid_cents": 210.0,
        }
        for i in range(20)
    ]

    eval_report = harness.evaluate_historical_log(logs)

    assert eval_report["total_auctions"] == 20.0
    assert eval_report["fill_rate"] == 1.0
    assert eval_report["revenue_lift_pct"] > 0.0
    assert eval_report["sla_compliance_pct"] == 100.0


def test_shadow_auction_eval_empty_log() -> None:
    service = VerveYieldOptimizationService()
    harness = ShadowAuctionEvalHarness(service)

    eval_report = harness.evaluate_historical_log([])
    assert eval_report["total_auctions"] == 0.0
    assert eval_report["fill_rate"] == 0.0
