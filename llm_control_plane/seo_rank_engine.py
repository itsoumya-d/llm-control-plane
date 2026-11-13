"""SEO Rank Engine, SERP Tracker, Core Web Vitals Auditor, and Intent Clusterer.

Engineered for enterprise search intelligence, automated SERP tracking at scale,
Core Web Vitals diagnostic auditing, and semantic search intent clustering.

Components:
1. SERPRankTracker: Resilient SERP harvesting pipeline with proxy rotation, user-agent
   randomization, exponential backoff with full jitter, and structured HTML parsing.
2. CoreWebVitalsAuditor: Chrome User Experience / Lighthouse aligned Web Vitals evaluation
   (LCP, CLS, INP, FCP, TTFB) with automated remediation advice and overall health scoring.
3. SemanticKeywordClusterer: Lexical and TF-IDF cosine intent classification (Informational,
   Navigational, Commercial, Transactional) and keyword cannibalization detection.
4. ClientSEOAnalyticsAggregator: Search visibility index modeling, ranking velocity delta
   tracking, and SHA-256 tamper-evident client report verification.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import random
import re
import time
from collections import Counter
from enum import Enum
from typing import Any
from urllib.parse import urlparse


class SearchIntent(str, Enum):
    INFORMATIONAL = "INFORMATIONAL"
    NAVIGATIONAL = "NAVIGATIONAL"
    COMMERCIAL = "COMMERCIAL"
    TRANSACTIONAL = "TRANSACTIONAL"


class MetricGrade(str, Enum):
    GOOD = "GOOD"
    NEEDS_IMPROVEMENT = "NEEDS_IMPROVEMENT"
    POOR = "POOR"


class DeviceType(str, Enum):
    DESKTOP = "DESKTOP"
    MOBILE = "MOBILE"


@dataclasses.dataclass
class ProxyConfig:
    """Proxy endpoint configuration for SERP request routing."""

    host: str
    port: int
    protocol: str = "https"
    auth: str | None = None

    @property
    def url(self) -> str:
        if self.auth:
            return f"{self.protocol}://{self.auth}@{self.host}:{self.port}"
        return f"{self.protocol}://{self.host}:{self.port}"


@dataclasses.dataclass
class SERPResult:
    """Individual search engine ranking entry."""

    rank: int
    title: str
    url: str
    snippet: str
    domain: str
    is_featured_snippet: bool = False
    sitelinks: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class SERPSnapshot:
    """Point-in-time snapshot of search results for a keyword query."""

    keyword: str
    device: DeviceType
    timestamp: float
    results: list[SERPResult]
    total_results_estimated: int = 0


class SERPRankTracker:
    """Automated SERP rank tracking pipeline with proxy pooling and anti-bot mitigation."""

    DEFAULT_DESKTOP_UA = [
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64; rv:130.0) Gecko/20100101 Firefox/130.0",
    ]

    DEFAULT_MOBILE_UA = [
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_6 like Mac OS X) AppleWebKit/605.1.15 "
        "(KHTML, like Gecko) Version/17.6 Mobile/15E148 Safari/604.1",
        "Mozilla/5.0 (Linux; Android 14; Pixel 8 Pro) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/128.0.0.0 Mobile Safari/537.36",
    ]

    def __init__(
        self,
        proxies: list[ProxyConfig] | None = None,
        desktop_user_agents: list[str] | None = None,
        mobile_user_agents: list[str] | None = None,
    ) -> None:
        self.proxies = proxies or []
        self._proxy_index = 0
        self.desktop_user_agents = desktop_user_agents or self.DEFAULT_DESKTOP_UA
        self.mobile_user_agents = mobile_user_agents or self.DEFAULT_MOBILE_UA

    def get_next_proxy(self) -> ProxyConfig | None:
        """Round-robin proxy pool selector."""
        if not self.proxies:
            return None
        proxy = self.proxies[self._proxy_index % len(self.proxies)]
        self._proxy_index += 1
        return proxy

    def get_random_user_agent(self, device: DeviceType = DeviceType.DESKTOP) -> str:
        """Fetch randomized valid browser User-Agent header."""
        pool = self.mobile_user_agents if device == DeviceType.MOBILE else self.desktop_user_agents
        return random.choice(pool)

    def calculate_backoff_delay(self, attempt: int, base_sec: float = 1.0, max_sec: float = 30.0) -> float:
        """Compute exponential backoff with full jitter to evade rate limiting."""
        raw_backoff = min(max_sec, base_sec * (2 ** (attempt - 1)))
        return random.uniform(0.0, raw_backoff)

    def parse_serp_html(
        self,
        html_content: str,
        keyword: str,
        device: DeviceType = DeviceType.DESKTOP,
        timestamp: float | None = None,
    ) -> SERPSnapshot:
        """Parse raw SERP HTML into structured ranking items."""
        ts = timestamp if timestamp is not None else time.time()
        results: list[SERPResult] = []

        # Matches standard Google search result blocks or simulated structured cards
        card_pattern = re.compile(
            r'<div class="(?:g|tF2Cxc|serp-card)"[^>]*>.*?'
            r'<a href="(?P<url>https?://[^"]+)"[^>]*>.*?'
            r'<h3[^>]*>(?P<title>[^<]+)</h3>.*?'
            r'(?:<div class="(?:VwiC3b|snippet)"[^>]*>(?P<snippet>[^<]+)</div>)?',
            re.DOTALL,
        )

        rank = 1
        for match in card_pattern.finditer(html_content):
            url = match.group("url")
            title = match.group("title").strip()
            snippet = (match.group("snippet") or "").strip()
            parsed = urlparse(url)
            domain = parsed.netloc.lower()

            results.append(
                SERPResult(
                    rank=rank,
                    title=title,
                    url=url,
                    snippet=snippet,
                    domain=domain,
                    is_featured_snippet=(rank == 1 and "featured" in html_content.lower()),
                )
            )
            rank += 1

        # Fallback heuristic for lightweight or mock markup
        if not results:
            item_pattern = re.compile(
                r'<a class="result-link" href="(?P<url>https?://[^"]+)">'
                r'(?P<title>[^<]+)</a>\s*<p class="result-desc">(?P<snippet>[^<]+)</p>',
                re.DOTALL,
            )
            for match in item_pattern.finditer(html_content):
                url = match.group("url")
                title = match.group("title").strip()
                snippet = match.group("snippet").strip()
                parsed = urlparse(url)
                domain = parsed.netloc.lower()
                results.append(
                    SERPResult(
                        rank=rank,
                        title=title,
                        url=url,
                        snippet=snippet,
                        domain=domain,
                    )
                )
                rank += 1

        return SERPSnapshot(
            keyword=keyword,
            device=device,
            timestamp=ts,
            results=results,
            total_results_estimated=len(results),
        )

    def track_rank(
        self,
        target_domain: str,
        snapshot: SERPSnapshot,
        previous_rank: int | None = None,
    ) -> dict[str, Any]:
        """Assess domain ranking performance and delta against previous baseline."""
        target_domain_clean = target_domain.lower().replace("www.", "")
        matched_result: SERPResult | None = None

        for res in snapshot.results:
            clean_host = res.domain.replace("www.", "")
            if clean_host == target_domain_clean or clean_host.endswith("." + target_domain_clean):
                matched_result = res
                break

        current_rank = matched_result.rank if matched_result else None
        rank_delta: int | None = None
        if current_rank is not None and previous_rank is not None:
            # Positive delta means improvement (e.g. 5 -> 2 is +3 positions)
            rank_delta = previous_rank - current_rank

        return {
            "target_domain": target_domain,
            "keyword": snapshot.keyword,
            "device": snapshot.device.value,
            "is_ranked": current_rank is not None,
            "current_rank": current_rank,
            "previous_rank": previous_rank,
            "rank_delta": rank_delta,
            "is_top_3": current_rank is not None and current_rank <= 3,
            "is_first_page": current_rank is not None and current_rank <= 10,
            "ranking_url": matched_result.url if matched_result else None,
            "title": matched_result.title if matched_result else None,
        }


class CoreWebVitalsAuditor:
    """Chrome Core Web Vitals (CWV) diagnostics, scoring, and automated remediation engine."""

    # Official thresholds (Google Web Vitals 2024 standards)
    THRESHOLDS = {
        "lcp_ms": {"good": 2500.0, "poor": 4000.0},  # Largest Contentful Paint
        "cls": {"good": 0.10, "poor": 0.25},         # Cumulative Layout Shift
        "inp_ms": {"good": 200.0, "poor": 500.0},    # Interaction to Next Paint
        "fcp_ms": {"good": 1800.0, "poor": 3000.0},  # First Contentful Paint
        "ttfb_ms": {"good": 800.0, "poor": 1800.0},  # Time to First Byte
    }

    @classmethod
    def grade_metric(cls, metric_name: str, value: float) -> MetricGrade:
        """Classify a single Web Vital into GOOD, NEEDS_IMPROVEMENT, or POOR."""
        limits = cls.THRESHOLDS.get(metric_name)
        if not limits:
            return MetricGrade.GOOD
        if value <= limits["good"]:
            return MetricGrade.GOOD
        elif value <= limits["poor"]:
            return MetricGrade.NEEDS_IMPROVEMENT
        else:
            return MetricGrade.POOR

    @classmethod
    def audit_url(
        cls,
        url: str,
        lcp_ms: float,
        cls_val: float,
        inp_ms: float,
        fcp_ms: float,
        ttfb_ms: float,
    ) -> dict[str, Any]:
        """Perform comprehensive Core Web Vitals audit and compute overall SEO health score."""
        grades = {
            "lcp": cls.grade_metric("lcp_ms", lcp_ms),
            "cls": cls.grade_metric("cls", cls_val),
            "inp": cls.grade_metric("inp_ms", inp_ms),
            "fcp": cls.grade_metric("fcp_ms", fcp_ms),
            "ttfb": cls.grade_metric("ttfb_ms", ttfb_ms),
        }

        # Health score weighting: LCP (30%), INP (25%), CLS (20%), FCP (15%), TTFB (10%)
        def score_single(grade: MetricGrade) -> float:
            if grade == MetricGrade.GOOD:
                return 100.0
            if grade == MetricGrade.NEEDS_IMPROVEMENT:
                return 60.0
            return 20.0

        health_score = (
            score_single(grades["lcp"]) * 0.30
            + score_single(grades["inp"]) * 0.25
            + score_single(grades["cls"]) * 0.20
            + score_single(grades["fcp"]) * 0.15
            + score_single(grades["ttfb"]) * 0.10
        )

        remediations: list[str] = []
        if grades["lcp"] != MetricGrade.GOOD:
            remediations.append(
                f"LCP is {lcp_ms:.1f}ms (>2500ms). Preload hero image, deploy AVIF/WebP formats, "
                "and ensure edge CDN cache HIT."
            )
        if grades["cls"] != MetricGrade.GOOD:
            remediations.append(
                f"CLS is {cls_val:.3f} (>0.10). Specify explicit width/height on media wrappers; "
                "reserve DOM space for dynamic ad banners."
            )
        if grades["inp"] != MetricGrade.GOOD:
            remediations.append(
                f"INP is {inp_ms:.1f}ms (>200ms). Break up long JavaScript execution blocks (>50ms); "
                "defer non-critical hydrate scripts via web workers."
            )
        if grades["ttfb"] != MetricGrade.GOOD:
            remediations.append(
                f"TTFB is {ttfb_ms:.1f}ms (>800ms). Enable server-side full page caching (e.g. Fastly), "
                "tune SQL queries, or relocate edge workers."
            )

        passed_all_core = (
            grades["lcp"] == MetricGrade.GOOD
            and grades["cls"] == MetricGrade.GOOD
            and grades["inp"] == MetricGrade.GOOD
        )

        return {
            "url": url,
            "overall_health_score": round(health_score, 1),
            "passed_core_web_vitals": passed_all_core,
            "metrics": {
                "lcp_ms": {"value": lcp_ms, "grade": grades["lcp"].value},
                "cls": {"value": cls_val, "grade": grades["cls"].value},
                "inp_ms": {"value": inp_ms, "grade": grades["inp"].value},
                "fcp_ms": {"value": fcp_ms, "grade": grades["fcp"].value},
                "ttfb_ms": {"value": ttfb_ms, "grade": grades["ttfb"].value},
            },
            "remediation_actions": remediations,
        }


class SemanticKeywordClusterer:
    """Semantic clustering, search intent classification, and cannibalization detector."""

    # High-signal intent triggers
    INTENT_TRIGGERS = {
        SearchIntent.TRANSACTIONAL: [
            "buy", "order", "purchase", "cheap", "price", "discount", "coupon",
            "deal", "quote", "hire", "service", "pricing", "cost", "near me",
        ],
        SearchIntent.COMMERCIAL: [
            "best", "top", "review", "vs", "versus", "comparison", "alternative",
            "features", "software", "tool", "agency", "firm", "audit", "platform",
        ],
        SearchIntent.NAVIGATIONAL: [
            "login", "signin", "portal", "dashboard", "account", "website",
            "official", "app", "download", "contact",
        ],
        SearchIntent.INFORMATIONAL: [
            "how", "what", "why", "where", "guide", "tutorial", "tips",
            "learn", "definition", "meaning", "examples", "steps", "strategy",
        ],
    }

    @classmethod
    def classify_intent(cls, keyword: str) -> SearchIntent:
        """Classify search intent from lexical keywords and modifiers."""
        text = keyword.lower().strip()
        tokens = re.findall(r"\w+", text)

        scores: dict[SearchIntent, int] = {
            SearchIntent.TRANSACTIONAL: 0,
            SearchIntent.COMMERCIAL: 0,
            SearchIntent.NAVIGATIONAL: 0,
            SearchIntent.INFORMATIONAL: 0,
        }

        for intent, triggers in cls.INTENT_TRIGGERS.items():
            for trigger in triggers:
                if trigger in text or trigger in tokens:
                    scores[intent] += 2

        # Question words heavily skew to Informational
        if any(text.startswith(q) for q in ["how", "what", "why", "when", "who"]):
            scores[SearchIntent.INFORMATIONAL] += 4

        best_intent = max(scores.items(), key=lambda x: x[1])
        if best_intent[1] > 0:
            return best_intent[0]
        # Default fallback is Informational for long-tail, Commercial for short generic queries
        return SearchIntent.COMMERCIAL if len(tokens) <= 2 else SearchIntent.INFORMATIONAL

    @staticmethod
    def _cosine_similarity(vec1: Counter[str], vec2: Counter[str]) -> float:
        """Compute cosine similarity between two bag-of-words character n-gram profiles."""
        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum(vec1[x] * vec2[x] for x in intersection)
        sum1 = sum(v ** 2 for v in vec1.values())
        sum2 = sum(v ** 2 for v in vec2.values())
        denominator = math.sqrt(sum1) * math.sqrt(sum2)
        if not denominator:
            return 0.0
        return numerator / denominator

    @classmethod
    def _to_ngram_vector(cls, text: str, n: int = 3) -> Counter[str]:
        """Convert string to 3-gram character frequency vector."""
        cleaned = re.sub(r"\s+", " ", text.lower().strip())
        ngrams = [cleaned[i : i + n] for i in range(len(cleaned) - n + 1)]
        return Counter(ngrams)

    @classmethod
    def cluster_keywords(
        cls,
        keywords: list[str],
        similarity_threshold: float = 0.40,
    ) -> dict[str, list[str]]:
        """Cluster keywords into topical silos based on semantic token overlap."""
        clusters: dict[str, list[str]] = {}
        vectors = {kw: cls._to_ngram_vector(kw) for kw in keywords}

        for kw in keywords:
            vec = vectors[kw]
            assigned = False
            for centroid, members in clusters.items():
                sim = cls._cosine_similarity(vec, vectors[centroid])
                if sim >= similarity_threshold:
                    members.append(kw)
                    assigned = True
                    break
            if not assigned:
                clusters[kw] = [kw]

        return clusters

    @classmethod
    def detect_cannibalization(
        cls,
        ranking_data: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Identify keyword cannibalization where multiple distinct URLs rank for the same query."""
        by_keyword: dict[str, list[dict[str, Any]]] = {}
        for entry in ranking_data:
            kw = entry.get("keyword", "")
            by_keyword.setdefault(kw, []).append(entry)

        cannibalized: list[dict[str, Any]] = []
        for kw, entries in by_keyword.items():
            distinct_urls = {e.get("url") for e in entries if e.get("url")}
            if len(distinct_urls) > 1:
                cannibalized.append({
                    "keyword": kw,
                    "competing_url_count": len(distinct_urls),
                    "urls": sorted(list(distinct_urls)),
                    "severity": "HIGH" if any(e.get("rank", 99) <= 10 for e in entries) else "MODERATE",
                })

        return cannibalized


class ClientSEOAnalyticsAggregator:
    """Enterprise visibility aggregation, rank velocity delta, and SHA-256 verifiable reporting."""

    # Industry standard CTR curve by organic position
    CTR_CURVE = {
        1: 0.316,
        2: 0.157,
        3: 0.103,
        4: 0.072,
        5: 0.051,
        6: 0.038,
        7: 0.029,
        8: 0.022,
        9: 0.017,
        10: 0.014,
    }

    @classmethod
    def calculate_visibility_index(cls, rankings: list[dict[str, Any]]) -> float:
        """Compute aggregated Organic Visibility Index across keyword portfolio."""
        total_score = 0.0
        for entry in rankings:
            rank = entry.get("rank")
            search_volume = entry.get("search_volume", 1000)
            if rank and 1 <= rank <= 10:
                ctr = cls.CTR_CURVE.get(rank, 0.01)
                total_score += ctr * (search_volume / 1000.0)
            elif rank and 11 <= rank <= 20:
                total_score += 0.003 * (search_volume / 1000.0)

        return round(total_score, 2)

    @classmethod
    def calculate_rank_velocity(
        cls,
        current_rankings: list[dict[str, Any]],
        previous_rankings: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Compute net rank change velocity across an audit cycle."""
        prev_map = {p["keyword"]: p.get("rank") for p in previous_rankings if "keyword" in p}
        gained = 0
        lost = 0
        unchanged = 0
        net_delta = 0

        for cur in current_rankings:
            kw = cur.get("keyword")
            cur_rank = cur.get("rank")
            prev_rank = prev_map.get(kw)

            if cur_rank is not None and prev_rank is not None:
                delta = prev_rank - cur_rank  # positive = improvement
                net_delta += delta
                if delta > 0:
                    gained += 1
                elif delta < 0:
                    lost += 1
                else:
                    unchanged += 1

        return {
            "keywords_evaluated": len(current_rankings),
            "net_rank_delta": net_delta,
            "positions_improved": gained,
            "positions_declined": lost,
            "positions_stable": unchanged,
            "velocity_trend": "ACCELERATING" if net_delta > 0 else "DECLINING" if net_delta < 0 else "NEUTRAL",
        }

    @classmethod
    def generate_audit_report(
        cls,
        client_name: str,
        target_domain: str,
        rankings: list[dict[str, Any]],
        cwv_audit: dict[str, Any],
        clusters: dict[str, list[str]],
        previous_hash: str = "GENESIS_SEO_HASH",
    ) -> dict[str, Any]:
        """Build a verifiable, SHA-256 cryptographically anchored audit package."""
        timestamp = time.time()
        visibility = cls.calculate_visibility_index(rankings)

        payload = {
            "client_name": client_name,
            "target_domain": target_domain,
            "timestamp": timestamp,
            "visibility_index": visibility,
            "cwv_audit": cwv_audit,
            "keyword_clusters": clusters,
            "rankings_sample": rankings[:10],
            "previous_hash": previous_hash,
        }

        canonical_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        report_hash = hashlib.sha256(canonical_bytes).hexdigest()

        return {
            "report_id": f"seo-audit-{hashlib.md5(f'{client_name}:{timestamp}'.encode()).hexdigest()[:8]}",
            "report_hash": report_hash,
            "previous_hash": previous_hash,
            "summary": {
                "client": client_name,
                "domain": target_domain,
                "visibility_index": visibility,
                "cwv_health_score": cwv_audit.get("overall_health_score"),
                "total_clusters": len(clusters),
            },
            "data": payload,
        }
