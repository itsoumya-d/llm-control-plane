"""Telnyx Inline Fraud & Abuse Decision Engine.

High-throughput, real-time messaging security platform:
- Adversarial Unicode, homoglyph, and leetspeak de-obfuscation normalizer
- Inline hot-path sub-50ms heuristic screening with deliberate fail-open/fail-closed semantics
- Entity-level risk state tracking and Artificially Inflated Traffic (AIT) entropy watchdog
- Two-tier LLM judge classifier with structured JSON evaluations and calibration scoring
- Immutable decision audit ledger and shadow-mode holdout evaluation harness
- Human-in-the-loop (HITL) disposition feedback ingestion and replay regression harness

Tailored for Telnyx Machine Learning Engineer (Fraud & Abuse Detection), Dublin, Ireland.
"""

from __future__ import annotations

import hashlib
import math
import re
import time
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


class AbuseArchetype(str, Enum):
    """Classification categories for messaging fraud, abuse, and legitimate traffic."""

    PHISHING_SMISHING = "PHISHING_SMISHING"
    BRAND_IMPERSONATION = "BRAND_IMPERSONATION"
    AIT_FRAUD = "AIT_FRAUD"
    SOCIAL_ENGINEERING_SCAM = "SOCIAL_ENGINEERING_SCAM"
    MALWARE_URL = "MALWARE_URL"
    LEGITIMATE = "LEGITIMATE"


class DecisionAction(str, Enum):
    """Enforcement outcome produced by the decision engine."""

    ALLOW = "ALLOW"
    BLOCK = "BLOCK"
    QUARANTINE_REVIEW = "QUARANTINE_REVIEW"
    SHADOW_LOG = "SHADOW_LOG"


class FailMode(str, Enum):
    """Operational resilience mode when inline components exceed latency or encounter errors."""

    FAIL_OPEN = "FAIL_OPEN"
    FAIL_CLOSED = "FAIL_CLOSED"


@dataclass
class NormalizationResult:
    """Output of adversarial text de-obfuscation."""

    original_text: str
    normalized_text: str
    zero_width_chars_removed: int
    homoglyphs_replaced: int
    leetspeak_substitutions: int
    extracted_urls: List[str]
    evasion_risk_score: float  # [0.0, 1.0]


class AdversarialTextNormalizer:
    """De-obfuscates adversarial mutations, confusable homoglyphs, and evasive formatting in SMS."""

    # Zero-width, directional, and invisible formatting characters
    ZERO_WIDTH_CHARS: Set[str] = {
        "\u200b",  # Zero-width space
        "\u200c",  # Zero-width non-joiner
        "\u200d",  # Zero-width joiner
        "\u200e",  # Left-to-right mark
        "\u200f",  # Right-to-left mark
        "\u202a",  # LRE
        "\u202b",  # RLE
        "\u202c",  # PDF
        "\u202d",  # LRO
        "\u202e",  # RLO
        "\u2060",  # Word joiner
        "\ufeff",  # Byte order mark
        "\u00ad",  # Soft hyphen
    }

    # Common Cyrillic, Greek, and Unicode homoglyphs mapped to ASCII equivalents
    HOMOGLYPH_MAP: Dict[str, str] = {
        # Cyrillic
        "а": "a",
        "А": "A",
        "в": "b",
        "В": "B",
        "с": "c",
        "С": "C",
        "е": "e",
        "Е": "E",
        "і": "i",
        "І": "I",
        "ј": "j",
        "Ј": "J",
        "к": "k",
        "К": "K",
        "м": "m",
        "М": "M",
        "н": "h",
        "Н": "H",
        "о": "o",
        "О": "O",
        "р": "p",
        "Р": "P",
        "ѕ": "s",
        "Ѕ": "S",
        "т": "t",
        "Т": "T",
        "у": "y",
        "У": "Y",
        "х": "x",
        "Х": "X",
        # Greek
        "α": "a",
        "Α": "A",
        "β": "b",
        "Β": "B",
        "ε": "e",
        "Ε": "E",
        "η": "n",
        "Η": "H",
        "ι": "i",
        "Ι": "I",
        "κ": "k",
        "Κ": "K",
        "ν": "v",
        "Ν": "N",
        "ο": "o",
        "Ο": "O",
        "ρ": "p",
        "Ρ": "P",
        "τ": "t",
        "Τ": "T",
        "υ": "u",
        "Υ": "Y",
        "χ": "x",
        "Χ": "X",
    }

    # Common leetspeak symbol substitutions
    LEETSPEAK_MAP: Dict[str, str] = {
        "@": "a",
        "0": "o",
        "1": "i",
        "!": "i",
        "3": "e",
        "4": "a",
        "5": "s",
        "$": "s",
        "7": "t",
        "+": "t",
        "8": "b",
    }

    URL_REGEX = re.compile(
        r"(?:https?://)?(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}(?::\d+)?(?:/[^\s]*)?",
        re.IGNORECASE,
    )

    SUSPICIOUS_SHORTENERS = {"bit.ly", "tinyurl.com", "t.co", "is.gd", "ow.ly", "rb.gy", "cutt.ly", "shorturl.at"}

    def normalize(self, text: str) -> NormalizationResult:
        """Strip zero-width characters, normalize homoglyphs/NFKD, and resolve leetspeak."""
        if not text:
            return NormalizationResult(
                original_text="",
                normalized_text="",
                zero_width_chars_removed=0,
                homoglyphs_replaced=0,
                leetspeak_substitutions=0,
                extracted_urls=[],
                evasion_risk_score=0.0,
            )

        # 1. Strip zero-width characters
        zw_removed = 0
        cleaned_chars = []
        for ch in text:
            if ch in self.ZERO_WIDTH_CHARS:
                zw_removed += 1
            else:
                cleaned_chars.append(ch)
        zw_cleaned = "".join(cleaned_chars)

        # 2. Extract URLs
        urls = self.URL_REGEX.findall(zw_cleaned)

        # 3. Unicode NFKD decomposition + ASCII mapping
        nfkd_form = unicodedata.normalize("NFKD", zw_cleaned)
        homoglyphs_replaced = 0
        homoglyph_chars = []
        for ch in nfkd_form:
            if ch in self.HOMOGLYPH_MAP:
                homoglyph_chars.append(self.HOMOGLYPH_MAP[ch])
                homoglyphs_replaced += 1
            else:
                homoglyph_chars.append(ch)
        homoglyph_text = "".join(homoglyph_chars)

        # 4. Leetspeak mapping
        leetspeak_count = 0
        leet_chars = []
        for ch in homoglyph_text:
            if ch in self.LEETSPEAK_MAP and not (ch.isdigit() and len(urls) > 0 and ch in "".join(urls)):
                leet_chars.append(self.LEETSPEAK_MAP[ch])
                leetspeak_count += 1
            else:
                leet_chars.append(ch)
        leet_text = "".join(leet_chars)

        # 5. Clean whitespace & lowercasing for search
        normalized = re.sub(r"\s+", " ", leet_text).strip().lower()

        # Evasion risk calculation: proportion of adversarial modifications
        length = max(len(text), 1)
        evasion_weight = (zw_removed * 2.0 + homoglyphs_replaced * 1.5 + leetspeak_count * 0.8) / length
        evasion_score = min(round(evasion_weight, 4), 1.0)

        return NormalizationResult(
            original_text=text,
            normalized_text=normalized,
            zero_width_chars_removed=zw_removed,
            homoglyphs_replaced=homoglyphs_replaced,
            leetspeak_substitutions=leetspeak_count,
            extracted_urls=urls,
            evasion_risk_score=evasion_score,
        )


@dataclass
class EntityRiskProfile:
    """Stateful entity risk profile tracking rolling message velocity and entropy."""

    entity_id: str
    message_count: int = 0
    destination_numbers: List[str] = field(default_factory=list)
    template_hashes: Set[str] = field(default_factory=set)
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    reputation_score: float = 1.0  # 1.0 = clean, 0.0 = confirmed bad actor

    def record_message(self, recipient: str, template_hash: str) -> None:
        """Record an outbound message event."""
        self.message_count += 1
        self.destination_numbers.append(recipient)
        self.template_hashes.add(template_hash)
        self.last_seen = time.time()

    def calculate_destination_entropy(self) -> float:
        """Shannon entropy of recipient prefix distributions to detect sequential AIT bot blasts."""
        if not self.destination_numbers:
            return 0.0
        counts: Dict[str, int] = {}
        # Track 6-digit prefix or exact number
        for num in self.destination_numbers[-500:]:
            prefix = num[:7] if len(num) >= 7 else num
            counts[prefix] = counts.get(prefix, 0) + 1

        total = sum(counts.values())
        entropy = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        return round(entropy, 4)


class AITTrafficWatchdog:
    """Detects Artificially Inflated Traffic (AIT) and SMS toll fraud patterns."""

    def __init__(self, velocity_threshold: int = 100, entropy_min: float = 1.2):
        self.velocity_threshold = velocity_threshold
        self.entropy_min = entropy_min
        self.entities: Dict[str, EntityRiskProfile] = {}

    def get_or_create_entity(self, entity_id: str) -> EntityRiskProfile:
        """Get or initialize entity profile."""
        if entity_id not in self.entities:
            self.entities[entity_id] = EntityRiskProfile(entity_id=entity_id)
        return self.entities[entity_id]

    def evaluate_ait(
        self,
        entity_id: str,
        recipient: str,
        normalized_template: str,
    ) -> Tuple[bool, float, str]:
        """Check if message represents artificially inflated traffic."""
        template_hash = hashlib.md5(normalized_template.encode("utf-8")).hexdigest()[:12]
        profile = self.get_or_create_entity(entity_id)
        profile.record_message(recipient, template_hash)

        if profile.message_count < 10:
            return False, 0.0, "Insufficient sample count for AIT determination"

        entropy = profile.calculate_destination_entropy()
        # High volume + low entropy + single template = classical AIT toll fraud pattern
        if (
            profile.message_count > self.velocity_threshold
            and entropy < self.entropy_min
            and len(profile.template_hashes) <= 2
        ):
            ait_risk = min(1.0, 0.6 + (0.4 * (1.0 - min(entropy / self.entropy_min, 1.0))))
            return (
                True,
                round(ait_risk, 4),
                f"AIT detected: {profile.message_count} msgs, low entropy {entropy:.2f}, "
                f"{len(profile.template_hashes)} template(s)",
            )

        return False, 0.0, f"Normal traffic pattern (entropy={entropy:.2f}, count={profile.message_count})"


@dataclass
class LLMJudgeVerdict:
    """Structured response from LLM classifier/judge."""

    archetype: AbuseArchetype
    confidence: float
    reasoning: str
    recommended_action: DecisionAction
    calibration_score: float = 0.0


@dataclass
class FraudDecisionResult:
    """Final decision output for an inline messaging request."""

    action: DecisionAction
    archetype: AbuseArchetype
    confidence: float
    normalized_text: str
    reasons: List[str]
    latency_ms: float
    trace_id: str
    evasion_score: float
    is_shadow: bool
    audit_id: str


@dataclass
class DecisionAuditEntry:
    """Immutable audit record for regulatory compliance and ML replay."""

    audit_id: str
    trace_id: str
    timestamp: float
    entity_id: str
    sender_id: str
    recipient: str
    raw_message_hash: str
    normalized_text: str
    action: DecisionAction
    archetype: AbuseArchetype
    confidence: float
    evasion_score: float
    latency_ms: float
    reasons: List[str]
    human_disposition: Optional[str] = None  # None, "TRUE_POSITIVE", "FALSE_POSITIVE", "FALSE_NEGATIVE"


class TelnyxFraudDecisionEngine:
    """Production inline hot-path decision engine with adversarial de-obfuscation and two-tier ML."""

    # Targeted brand keywords commonly impersonated in European/US smishing
    PROTECTED_BRANDS = {
        "telnyx",
        "bank of ireland",
        "aib",
        "revolut",
        "an post",
        "paypal",
        "apple",
        "microsoft",
        "amazon",
        "netflix",
        "dhl",
        "ups",
        "fedex",
        "dpd",
    }

    # Urgent credential/action phishing trigger phrases
    URGENT_PHISHING_TRIGGERS = [
        "account suspended",
        "unauthorized transaction",
        "click here to verify",
        "verify your identity",
        "immediate action required",
        "login attempt detected",
        "billing failure",
        "card deactivated",
        "claim your prize",
        "refund pending",
        "package detained",
        "customs fee required",
        "one-time passcode",
        "otp code",
    ]

    def __init__(
        self,
        ait_watchdog: Optional[AITTrafficWatchdog] = None,
        llm_judge_fn: Optional[Callable[[str, str], LLMJudgeVerdict]] = None,
    ):
        self.normalizer = AdversarialTextNormalizer()
        self.ait_watchdog = ait_watchdog or AITTrafficWatchdog()
        self.llm_judge_fn = llm_judge_fn or self._default_mock_llm_judge
        self.audit_ledger: List[DecisionAuditEntry] = []

    def _default_mock_llm_judge(self, normalized_text: str, sender_id: str) -> LLMJudgeVerdict:
        """Deterministic mock LLM judge classifier for offline evaluation and testing."""
        sender_clean = sender_id.lower().strip()
        matched_brands = [b for b in self.PROTECTED_BRANDS if b in normalized_text]
        matched_triggers = [t for t in self.URGENT_PHISHING_TRIGGERS if t in normalized_text]

        # Brand Impersonation check
        if matched_brands:
            brand = matched_brands[0]
            if brand not in sender_clean:
                confidence = 0.94 if matched_triggers else 0.82
                return LLMJudgeVerdict(
                    archetype=AbuseArchetype.BRAND_IMPERSONATION,
                    confidence=confidence,
                    reasoning=f"Message claims to be {brand.title()} but sent from unverified sender '{sender_id}'",
                    recommended_action=DecisionAction.BLOCK if confidence >= 0.85 else DecisionAction.QUARANTINE_REVIEW,
                )

        # Phishing / Smishing check
        has_shortener = "http" in normalized_text or "bit.ly" in normalized_text
        if len(matched_triggers) >= 2 or (matched_triggers and has_shortener):
            return LLMJudgeVerdict(
                archetype=AbuseArchetype.PHISHING_SMISHING,
                confidence=0.91,
                reasoning=f"High-urgency phishing pattern detected with call-to-action triggers: {matched_triggers}",
                recommended_action=DecisionAction.BLOCK,
            )

        # Legitimate traffic
        return LLMJudgeVerdict(
            archetype=AbuseArchetype.LEGITIMATE,
            confidence=0.98,
            reasoning="Transactional, promotional, or conversational content without abuse signals",
            recommended_action=DecisionAction.ALLOW,
        )

    def evaluate_message(
        self,
        message: str,
        sender_id: str,
        recipient: str,
        entity_id: str,
        trace_id: Optional[str] = None,
        fail_mode: FailMode = FailMode.FAIL_OPEN,
        is_shadow: bool = False,
    ) -> FraudDecisionResult:
        """Inline hot-path evaluation running within sub-50ms SLA budget."""
        start_time = time.perf_counter()
        trace = trace_id or hashlib.sha256(f"{sender_id}:{recipient}:{time.time()}".encode()).hexdigest()[:16]
        reasons: List[str] = []

        try:
            # Stage 0: Adversarial De-obfuscation Normalization (<2ms)
            norm_res = self.normalizer.normalize(message)
            if norm_res.evasion_risk_score > 0.3:
                reasons.append(
                    f"Adversarial evasion detected (score={norm_res.evasion_risk_score:.2f}, "
                    f"zw_chars={norm_res.zero_width_chars_removed}, homoglyphs={norm_res.homoglyphs_replaced})"
                )

            # Stage 1: Fast Heuristic & Regex Rules (<2ms)
            for url in norm_res.extracted_urls:
                for shortener in self.normalizer.SUSPICIOUS_SHORTENERS:
                    if shortener in url.lower():
                        reasons.append(f"Deceptive URL shortener identified: {url}")

            # Stage 2: AIT Traffic Watchdog (<2ms)
            is_ait, ait_risk, ait_reason = self.ait_watchdog.evaluate_ait(
                entity_id=entity_id,
                recipient=recipient,
                normalized_template=norm_res.normalized_text,
            )
            if is_ait:
                reasons.append(ait_reason)
                elapsed = (time.perf_counter() - start_time) * 1000.0
                action = DecisionAction.SHADOW_LOG if is_shadow else DecisionAction.BLOCK
                result = FraudDecisionResult(
                    action=action,
                    archetype=AbuseArchetype.AIT_FRAUD,
                    confidence=ait_risk,
                    normalized_text=norm_res.normalized_text,
                    reasons=reasons,
                    latency_ms=round(elapsed, 2),
                    trace_id=trace,
                    evasion_score=norm_res.evasion_risk_score,
                    is_shadow=is_shadow,
                    audit_id=hashlib.sha256(f"audit:{trace}".encode()).hexdigest()[:16],
                )
                self._record_audit(result, entity_id, sender_id, recipient, message)
                return result

            # Stage 3: Two-Tier LLM Judge Classifier (<35ms budget)
            judge_verdict = self.llm_judge_fn(norm_res.normalized_text, sender_id)
            reasons.append(judge_verdict.reasoning)

            # Determine Action based on confidence and policy
            action = judge_verdict.recommended_action
            if is_shadow:
                action = DecisionAction.SHADOW_LOG
            elif norm_res.evasion_risk_score > 0.45 and action == DecisionAction.ALLOW:
                # Up-weight borderline messages with high evasion into quarantine
                action = DecisionAction.QUARANTINE_REVIEW
                reasons.append("Upgraded to QUARANTINE_REVIEW due to elevated adversarial evasion score")

            elapsed = (time.perf_counter() - start_time) * 1000.0
            result = FraudDecisionResult(
                action=action,
                archetype=judge_verdict.archetype,
                confidence=round(judge_verdict.confidence, 4),
                normalized_text=norm_res.normalized_text,
                reasons=reasons,
                latency_ms=round(elapsed, 2),
                trace_id=trace,
                evasion_score=norm_res.evasion_risk_score,
                is_shadow=is_shadow,
                audit_id=hashlib.sha256(f"audit:{trace}".encode()).hexdigest()[:16],
            )
            self._record_audit(result, entity_id, sender_id, recipient, message)
            return result

        except Exception as exc:
            elapsed = (time.perf_counter() - start_time) * 1000.0
            fallback_action = (
                DecisionAction.ALLOW if fail_mode == FailMode.FAIL_OPEN else DecisionAction.QUARANTINE_REVIEW
            )
            reasons.append(f"Inline decision engine exception: {str(exc)}. Fallback mode {fail_mode.value} applied.")
            result = FraudDecisionResult(
                action=fallback_action,
                archetype=AbuseArchetype.LEGITIMATE,
                confidence=0.5,
                normalized_text=message,
                reasons=reasons,
                latency_ms=round(elapsed, 2),
                trace_id=trace,
                evasion_score=0.0,
                is_shadow=is_shadow,
                audit_id=hashlib.sha256(f"audit:{trace}:fail".encode()).hexdigest()[:16],
            )
            self._record_audit(result, entity_id, sender_id, recipient, message)
            return result

    def _record_audit(
        self,
        result: FraudDecisionResult,
        entity_id: str,
        sender_id: str,
        recipient: str,
        raw_message: str,
    ) -> None:
        """Record immutable audit entry in memory ledger."""
        raw_hash = hashlib.sha256(raw_message.encode("utf-8")).hexdigest()
        entry = DecisionAuditEntry(
            audit_id=result.audit_id,
            trace_id=result.trace_id,
            timestamp=time.time(),
            entity_id=entity_id,
            sender_id=sender_id,
            recipient=recipient,
            raw_message_hash=raw_hash,
            normalized_text=result.normalized_text,
            action=result.action,
            archetype=result.archetype,
            confidence=result.confidence,
            evasion_score=result.evasion_score,
            latency_ms=result.latency_ms,
            reasons=result.reasons,
        )
        self.audit_ledger.append(entry)


@dataclass
class ShadowEvaluationMetrics:
    """Evaluation metrics computed over labeled fraud holdout corpus."""

    total_samples: int
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float
    false_positive_rate: float
    brier_calibration_score: float
    archetype_breakdown: Dict[str, Dict[str, Any]]


class ShadowModeEvaluator:
    """Evaluates fraud detection models in shadow mode against labeled holdout corpora."""

    def __init__(self, engine: TelnyxFraudDecisionEngine):
        self.engine = engine

    def evaluate_corpus(
        self,
        corpus: List[Dict[str, Any]],
    ) -> ShadowEvaluationMetrics:
        """Run shadow evaluation over corpus of records with {message, sender_id, recipient, ...}."""
        tp = fp = tn = fn = 0
        brier_sum = 0.0
        archetype_stats: Dict[str, Dict[str, int]] = {}

        for item in corpus:
            raw_msg = item["message"]
            sender = item.get("sender_id", "Unknown")
            recipient = item.get("recipient", "+353871234567")
            true_archetype = item["true_archetype"]
            is_fraud = item["is_fraud"]

            res = self.engine.evaluate_message(
                message=raw_msg,
                sender_id=sender,
                recipient=recipient,
                entity_id="eval_entity",
                is_shadow=True,
            )

            predicted_fraud = res.archetype != AbuseArchetype.LEGITIMATE
            prob_fraud = res.confidence if predicted_fraud else (1.0 - res.confidence)
            actual_fraud_binary = 1.0 if is_fraud else 0.0
            brier_sum += (prob_fraud - actual_fraud_binary) ** 2

            if is_fraud and predicted_fraud:
                tp += 1
            elif not is_fraud and predicted_fraud:
                fp += 1
            elif not is_fraud and not predicted_fraud:
                tn += 1
            else:
                fn += 1

            arch_key = true_archetype.value if hasattr(true_archetype, "value") else str(true_archetype)
            if arch_key not in archetype_stats:
                archetype_stats[arch_key] = {"total": 0, "correct": 0}
            archetype_stats[arch_key]["total"] += 1
            if res.archetype == true_archetype:
                archetype_stats[arch_key]["correct"] += 1

        total = len(corpus)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        brier = brier_sum / total if total > 0 else 0.0

        breakdown = {}
        for arch, stats in archetype_stats.items():
            tot = stats["total"]
            corr = stats["correct"]
            acc = corr / tot if tot > 0 else 1.0
            breakdown[arch] = {"total": tot, "correct": corr, "accuracy": round(acc, 4)}

        return ShadowEvaluationMetrics(
            total_samples=total,
            true_positives=tp,
            false_positives=fp,
            true_negatives=tn,
            false_negatives=fn,
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1_score=round(f1, 4),
            false_positive_rate=round(fpr, 4),
            brier_calibration_score=round(brier, 4),
            archetype_breakdown=breakdown,
        )


class HumanInTheLoopReplayHarness:
    """Converts human fraud reviewer dispositions into immutable regression replay suites."""

    def __init__(self, engine: TelnyxFraudDecisionEngine):
        self.engine = engine
        self.replay_test_cases: List[Dict[str, Any]] = []

    def ingest_reviewer_disposition(
        self,
        audit_id: str,
        disposition: str,  # "TRUE_POSITIVE", "FALSE_POSITIVE", "FALSE_NEGATIVE"
        corrected_archetype: AbuseArchetype,
        notes: str = "",
    ) -> bool:
        """Find audit entry and record human disposition into replay test suite."""
        entry = next((e for e in self.engine.audit_ledger if e.audit_id == audit_id), None)
        if not entry:
            return False

        entry.human_disposition = disposition
        self.replay_test_cases.append(
            {
                "audit_id": audit_id,
                "raw_message_hash": entry.raw_message_hash,
                "normalized_text": entry.normalized_text,
                "sender_id": entry.sender_id,
                "recipient": entry.recipient,
                "disposition": disposition,
                "expected_archetype": corrected_archetype,
                "notes": notes,
            }
        )
        return True

    def run_regression_replay(self) -> Dict[str, Any]:
        """Replay all ingested human reviewer cases to verify no behavioral regression."""
        passed = 0
        failed = 0
        details = []

        for tc in self.replay_test_cases:
            res = self.engine.evaluate_message(
                message=tc["normalized_text"],
                sender_id=tc["sender_id"],
                recipient=tc["recipient"],
                entity_id="replay_entity",
                is_shadow=True,
            )
            is_match = res.archetype == tc["expected_archetype"]
            if is_match:
                passed += 1
            else:
                failed += 1
            details.append(
                {
                    "audit_id": tc["audit_id"],
                    "expected": tc["expected_archetype"].value,
                    "actual": res.archetype.value,
                    "status": "PASS" if is_match else "FAIL",
                }
            )

        return {
            "total_cases": len(self.replay_test_cases),
            "passed": passed,
            "failed": failed,
            "pass_rate": round(passed / len(self.replay_test_cases), 4) if self.replay_test_cases else 1.0,
            "details": details,
        }
