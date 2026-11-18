"""
Polarsteps Travel Data Platform & Analytics Orchestrator.

Models high-volume mobile travel telemetry event stream ingestion, out-of-order
offline sync deduplication, dbt-compatible analytics data mart transformations,
GDPR-compliant GPS coordinate quantization, and privacy-preserving LLM query
guardrails for Polarsteps' 23M+ traveler consumer platform.
"""

from __future__ import annotations

import enum
import hashlib
import hmac
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


class TravelEventType(str, enum.Enum):
    STEP_RECORDED = "STEP_RECORDED"
    TRIP_STARTED = "TRIP_STARTED"
    TRIP_PUBLISHED = "TRIP_PUBLISHED"
    PHOTO_ATTACHED = "PHOTO_ATTACHED"
    TRAVEL_BOOK_PREVIEWED = "TRAVEL_BOOK_PREVIEWED"
    TRAVEL_BOOK_ORDERED = "TRAVEL_BOOK_ORDERED"
    OFFLINE_SYNC_BATCH = "OFFLINE_SYNC_BATCH"


class SchemaDriftSeverity(str, enum.Enum):
    COMPATIBLE = "COMPATIBLE"
    WARNING_OPTIONAL_FIELD = "WARNING_OPTIONAL_FIELD"
    BREAKING_MISSING_REQUIRED = "BREAKING_MISSING_REQUIRED"
    BREAKING_TYPE_MISMATCH = "BREAKING_TYPE_MISMATCH"


@dataclass
class TravelTelemetryEvent:
    event_id: str
    traveler_id: str
    device_id: str
    event_type: TravelEventType
    client_timestamp_ms: int
    latitude: float
    longitude: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    server_received_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    def compute_fingerprint(self) -> str:
        """Deterministic hash for deduplication during offline batch recovery."""
        raw = (
            f"{self.traveler_id}:{self.event_type.value}:{self.client_timestamp_ms}:"
            f"{self.latitude:.5f}:{self.longitude:.5f}"
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass
class QuantizedTravelEvent:
    event_id: str
    pseudonymized_traveler_id: str
    event_type: TravelEventType
    quantized_lat: float
    quantized_lon: float
    client_timestamp_ms: int
    trip_id: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)


class PrivacyPreservingLocationQuantizer:
    """
    Quantizes raw GPS coordinates into privacy-preserving geospatial grid buckets
    and hashes identifiable traveler and device keys using salted HMAC.
    """

    def __init__(self, salt: str = "polarsteps_2026_salt", precision_decimals: int = 2):
        self.salt = salt.encode("utf-8")
        self.precision_decimals = precision_decimals

    def pseudonymize_id(self, raw_id: str) -> str:
        return hmac.new(self.salt, raw_id.encode("utf-8"), hashlib.sha256).hexdigest()[:16]

    def quantize_coordinate(self, coord: float) -> float:
        factor = 10**self.precision_decimals
        return round(math.floor(coord * factor) / factor, self.precision_decimals)

    def process(self, event: TravelTelemetryEvent) -> QuantizedTravelEvent:
        q_lat = self.quantize_coordinate(event.latitude)
        q_lon = self.quantize_coordinate(event.longitude)
        pseudo_id = self.pseudonymize_id(event.traveler_id)
        sanitized_props = {
            k: v for k, v in event.metadata.items() if k not in ("exact_address", "ip_address", "imei", "home_street")
        }
        return QuantizedTravelEvent(
            event_id=event.event_id,
            pseudonymized_traveler_id=pseudo_id,
            event_type=event.event_type,
            quantized_lat=q_lat,
            quantized_lon=q_lon,
            client_timestamp_ms=event.client_timestamp_ms,
            trip_id=str(event.metadata.get("trip_id")) if "trip_id" in event.metadata else None,
            properties=sanitized_props,
        )


class TravelEventIngestionDeduplicator:
    """
    Handles out-of-order offline batch sync deduplication with sliding window memory.
    """

    def __init__(self, max_seen_entries: int = 100_000):
        self._seen_fingerprints: Set[str] = set()
        self._fingerprint_order: List[str] = []
        self._max_entries = max_seen_entries
        self.total_received = 0
        self.total_deduplicated = 0

    def ingest(self, event: TravelTelemetryEvent) -> Tuple[bool, str]:
        self.total_received += 1
        fp = event.compute_fingerprint()
        if fp in self._seen_fingerprints:
            self.total_deduplicated += 1
            return False, fp

        self._seen_fingerprints.add(fp)
        self._fingerprint_order.append(fp)
        if len(self._fingerprint_order) > self._max_entries:
            evicted = self._fingerprint_order.pop(0)
            self._seen_fingerprints.discard(evicted)

        return True, fp

    def ingest_batch(self, events: List[TravelTelemetryEvent]) -> List[TravelTelemetryEvent]:
        accepted = []
        for ev in events:
            ok, _ = self.ingest(ev)
            if ok:
                accepted.append(ev)
        return accepted


class DbtAnalyticsMartSynthesizer:
    """
    Transforms staging quantized travel events into intermediate sessions and
    fact data marts (traveler engagement, travel book checkout funnels).
    """

    EXPECTED_CONTRACTS: Dict[TravelEventType, Dict[str, Tuple[type, ...]]] = {
        TravelEventType.STEP_RECORDED: {
            "altitude_m": (int, float),
            "step_count": (int,),
        },
        TravelEventType.PHOTO_ATTACHED: {
            "photo_count": (int,),
            "has_caption": (bool,),
        },
        TravelEventType.TRAVEL_BOOK_ORDERED: {
            "book_page_count": (int,),
            "currency": (str,),
            "gross_amount": (int, float),
        },
    }

    def __init__(self):
        self._stg_events: List[QuantizedTravelEvent] = []

    def load_staging(self, events: List[QuantizedTravelEvent]) -> None:
        self._stg_events.extend(events)

    def validate_contract(
        self, event_type: TravelEventType, metadata: Dict[str, Any]
    ) -> Tuple[SchemaDriftSeverity, List[str]]:
        expected = self.EXPECTED_CONTRACTS.get(event_type)
        if not expected:
            return SchemaDriftSeverity.COMPATIBLE, []

        issues: List[str] = []
        for key, expected_types in expected.items():
            if key not in metadata:
                issues.append(f"Missing required field '{key}' for event {event_type.value}")
            else:
                val = metadata[key]
                if not isinstance(val, expected_types):
                    issues.append(f"Field '{key}' expected type {expected_types}, got {type(val).__name__}")

        if any("Missing required" in i for i in issues):
            return SchemaDriftSeverity.BREAKING_MISSING_REQUIRED, issues
        if any("expected type" in i for i in issues):
            return SchemaDriftSeverity.BREAKING_TYPE_MISMATCH, issues

        return SchemaDriftSeverity.COMPATIBLE, []

    def build_traveler_engagement_mart(self) -> List[Dict[str, Any]]:
        """
        fct_traveler_engagement mart summarizing trip steps, photos, and duration.
        """
        traveler_trips: Dict[str, Dict[str, Any]] = {}

        for ev in sorted(self._stg_events, key=lambda e: e.client_timestamp_ms):
            user = ev.pseudonymized_traveler_id
            trip = ev.trip_id or "default_trip"
            key = f"{user}:{trip}"

            if key not in traveler_trips:
                traveler_trips[key] = {
                    "pseudonymized_traveler_id": user,
                    "trip_id": trip,
                    "total_steps_logged": 0,
                    "total_photos_attached": 0,
                    "first_event_ms": ev.client_timestamp_ms,
                    "last_event_ms": ev.client_timestamp_ms,
                    "is_published": False,
                }

            row = traveler_trips[key]
            row["last_event_ms"] = max(row["last_event_ms"], ev.client_timestamp_ms)

            if ev.event_type == TravelEventType.STEP_RECORDED:
                steps = ev.properties.get("step_count", 1)
                row["total_steps_logged"] += steps
            elif ev.event_type == TravelEventType.PHOTO_ATTACHED:
                photos = ev.properties.get("photo_count", 1)
                row["total_photos_attached"] += photos
            elif ev.event_type == TravelEventType.TRIP_PUBLISHED:
                row["is_published"] = True

        marts: List[Dict[str, Any]] = []
        for row in traveler_trips.values():
            duration_hours = (row["last_event_ms"] - row["first_event_ms"]) / (1000 * 3600)
            marts.append(
                {
                    "pseudonymized_traveler_id": row["pseudonymized_traveler_id"],
                    "trip_id": row["trip_id"],
                    "total_steps": row["total_steps_logged"],
                    "total_photos": row["total_photos_attached"],
                    "duration_hours": round(duration_hours, 2),
                    "is_published": row["is_published"],
                }
            )
        return marts

    def build_book_conversion_funnel_mart(self) -> Dict[str, Any]:
        """
        fct_travel_book_funnel mart computing preview-to-order conversion rate.
        """
        previewed_users: Set[str] = set()
        ordered_users: Set[str] = set()
        gross_gmv_eur = 0.0

        for ev in self._stg_events:
            user = ev.pseudonymized_traveler_id
            if ev.event_type == TravelEventType.TRAVEL_BOOK_PREVIEWED:
                previewed_users.add(user)
            elif ev.event_type == TravelEventType.TRAVEL_BOOK_ORDERED:
                ordered_users.add(user)
                gross_gmv_eur += float(ev.properties.get("gross_amount", 0.0))

        conversion_rate = (len(ordered_users) / len(previewed_users)) if previewed_users else 0.0
        return {
            "total_previews": len(previewed_users),
            "total_orders": len(ordered_users),
            "conversion_rate_pct": round(conversion_rate * 100, 2),
            "gross_gmv_eur": round(gross_gmv_eur, 2),
        }


class LLMAnalyticalQueryGuardrail:
    """
    Prevents LLM queries and automated analyst reporting agents from extracting
    raw PII or exact geographic coordinates.
    """

    PROHIBITED_TERMS = (
        "exact_lat",
        "exact_lon",
        "home_address",
        "ip_address",
        "mac_address",
        "ssn",
        "passport_number",
        "full_name",
        "phone_number",
    )

    @classmethod
    def sanitize_llm_query_context(cls, raw_context: Dict[str, Any]) -> Dict[str, Any]:
        sanitized = {}
        for k, v in raw_context.items():
            if any(term in k.lower() for term in cls.PROHIBITED_TERMS):
                continue
            if isinstance(v, dict):
                sanitized[k] = cls.sanitize_llm_query_context(v)
            else:
                sanitized[k] = v
        return sanitized

    @classmethod
    def audit_llm_prompt(cls, prompt: str) -> Tuple[bool, Optional[str]]:
        lowered = prompt.lower()
        for term in cls.PROHIBITED_TERMS:
            if term in lowered:
                return False, f"Prompt contains prohibited PII token '{term}' violating GDPR policy"
        return True, None


class PolarstepsTravelDataPlatformOrchestrator:
    """
    Unified coordinator connecting event deduplication, GDPR quantization,
    dbt data mart synthesis, and LLM guardrails for Polarsteps.
    """

    def __init__(self, salt: str = "polarsteps_prod_salt", max_window: int = 50_000):
        self.deduplicator = TravelEventIngestionDeduplicator(max_seen_entries=max_window)
        self.quantizer = PrivacyPreservingLocationQuantizer(salt=salt)
        self.synthesizer = DbtAnalyticsMartSynthesizer()
        self.guardrail = LLMAnalyticalQueryGuardrail()

    def process_incoming_batch(self, events: List[TravelTelemetryEvent]) -> Dict[str, Any]:
        accepted = self.deduplicator.ingest_batch(events)
        quantized = [self.quantizer.process(ev) for ev in accepted]
        self.synthesizer.load_staging(quantized)

        return {
            "total_received": len(events),
            "accepted_count": len(accepted),
            "deduplicated_count": len(events) - len(accepted),
            "quantized_count": len(quantized),
        }
