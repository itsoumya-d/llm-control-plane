"""Cresta Customer Data Platform (CDP) Real-Time Analytics & Conversation Intelligence Engine.

Provides high-throughput columnar event ingestion modeled on ClickHouse architectures,
multi-tenant faceted conversational search with Elasticsearch tenant-isolated routing,
real-time contact center agent coaching triggers, and sub-second latency SLO tracking
for enterprise contact center workloads.
"""

from __future__ import annotations

import datetime
import enum
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


class TurnRole(str, enum.Enum):
    """Role of the participant in a conversational turn."""

    AGENT = "agent"
    CUSTOMER = "customer"
    SYSTEM = "system"


class CoachingActionType(str, enum.Enum):
    """Categorization of real-time AI coaching actions."""

    SUGGESTION = "suggestion"
    ESCALATION = "escalation"
    COMPLIANCE_WARNING = "compliance_warning"
    SCRIPT_GUIDANCE = "script_guidance"


@dataclass
class ConversationTurnEvent:
    """Represents a discrete turn in a contact center customer interaction."""

    conversation_id: str
    turn_id: str
    tenant_id: str
    agent_id: str
    customer_id: str
    timestamp: float
    role: TurnRole
    text: str
    sentiment_score: float  # -1.0 (very negative) to 1.0 (very positive)
    intent: str
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (-1.0 <= self.sentiment_score <= 1.0):
            raise ValueError(f"sentiment_score must be between -1.0 and 1.0, got {self.sentiment_score}")
        if self.latency_ms < 0:
            raise ValueError(f"latency_ms cannot be negative, got {self.latency_ms}")


@dataclass
class CoachingRecommendation:
    """Real-time coaching guidance delivered to an active agent during a call."""

    recommendation_id: str
    conversation_id: str
    turn_id: str
    tenant_id: str
    action_type: CoachingActionType
    guidance_text: str
    confidence_score: float
    trigger_reason: str
    generated_at: float = field(default_factory=time.time)


class ClickHouseColumnarBuffer:
    """High-throughput columnar ingestion buffer modeling ClickHouse MergeTree storage.

    Partitions data by (tenant_id, date) and maintains columnar vectors for vectorized
    analytical queries (e.g., agent performance summaries, sentiment trends, latency percentiles).
    """

    def __init__(self, batch_size: int = 100) -> None:
        self.batch_size = batch_size
        self._buffer: List[ConversationTurnEvent] = []
        # Key: (tenant_id, date_str) -> column name -> List of values
        self._partitions: Dict[Tuple[str, str], Dict[str, List[Any]]] = {}
        self._total_flushed_rows: int = 0

    def append(self, event: ConversationTurnEvent) -> bool:
        """Appends an event to the micro-batch buffer, flushing when batch_size is reached.

        Returns True if a flush occurred, False otherwise.
        """
        self._buffer.append(event)
        if len(self._buffer) >= self.batch_size:
            self.flush()
            return True
        return False

    def flush(self) -> int:
        """Flushes buffered rows into columnar partitioned storage."""
        count = len(self._buffer)
        if count == 0:
            return 0

        for event in self._buffer:
            date_str = datetime.datetime.fromtimestamp(event.timestamp, tz=datetime.timezone.utc).strftime("%Y-%m-%d")
            partition_key = (event.tenant_id, date_str)

            if partition_key not in self._partitions:
                self._partitions[partition_key] = {
                    "turn_id": [],
                    "conversation_id": [],
                    "agent_id": [],
                    "customer_id": [],
                    "timestamp": [],
                    "role": [],
                    "sentiment_score": [],
                    "intent": [],
                    "latency_ms": [],
                }

            cols = self._partitions[partition_key]
            cols["turn_id"].append(event.turn_id)
            cols["conversation_id"].append(event.conversation_id)
            cols["agent_id"].append(event.agent_id)
            cols["customer_id"].append(event.customer_id)
            cols["timestamp"].append(event.timestamp)
            cols["role"].append(event.role.value if isinstance(event.role, TurnRole) else str(event.role))
            cols["sentiment_score"].append(event.sentiment_score)
            cols["intent"].append(event.intent)
            cols["latency_ms"].append(event.latency_ms)

        self._total_flushed_rows += count
        self._buffer.clear()
        return count

    def query_agent_performance(self, tenant_id: str, agent_id: str) -> Dict[str, Any]:
        """Vectorized aggregation over agent turns across all active partitions for a tenant."""
        self.flush()
        sentiments: List[float] = []
        latencies: List[float] = []
        intent_counts: Dict[str, int] = {}
        agent_turns = 0

        for (part_tenant, _), cols in self._partitions.items():
            if part_tenant != tenant_id:
                continue
            for i, a_id in enumerate(cols["agent_id"]):
                if a_id == agent_id:
                    agent_turns += 1
                    sentiments.append(cols["sentiment_score"][i])
                    latencies.append(cols["latency_ms"][i])
                    intent = cols["intent"][i]
                    intent_counts[intent] = intent_counts.get(intent, 0) + 1

        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

        p95_latency = 0.0
        if latencies:
            sorted_lat = sorted(latencies)
            idx = int(math.ceil(0.95 * len(sorted_lat))) - 1
            p95_latency = sorted_lat[max(0, min(idx, len(sorted_lat) - 1))]

        return {
            "tenant_id": tenant_id,
            "agent_id": agent_id,
            "total_turns": agent_turns,
            "avg_sentiment": round(avg_sentiment, 4),
            "avg_latency_ms": round(avg_latency, 2),
            "p95_latency_ms": round(p95_latency, 2),
            "intent_distribution": intent_counts,
        }

    def query_tenant_sentiment_trend(self, tenant_id: str, start_ts: float, end_ts: float) -> List[Dict[str, Any]]:
        """Calculates daily sentiment aggregations between two timestamps."""
        self.flush()
        buckets: Dict[str, List[float]] = {}

        for (part_tenant, date_str), cols in self._partitions.items():
            if part_tenant != tenant_id:
                continue
            for i, ts in enumerate(cols["timestamp"]):
                if start_ts <= ts <= end_ts:
                    if date_str not in buckets:
                        buckets[date_str] = []
                    buckets[date_str].append(cols["sentiment_score"][i])

        result = []
        for date_str in sorted(buckets.keys()):
            scores = buckets[date_str]
            avg_score = sum(scores) / len(scores) if scores else 0.0
            result.append(
                {
                    "date": date_str,
                    "turn_count": len(scores),
                    "avg_sentiment": round(avg_score, 4),
                }
            )
        return result

    def prune_partitions_older_than(self, tenant_id: str, cutoff_ts: float) -> int:
        """Removes partitions where all events are strictly older than cutoff_ts."""
        self.flush()
        keys_to_remove = []
        pruned_records = 0

        for (part_tenant, date_str), cols in self._partitions.items():
            if part_tenant != tenant_id:
                continue
            if not cols["timestamp"]:
                keys_to_remove.append((part_tenant, date_str))
                continue
            if max(cols["timestamp"]) < cutoff_ts:
                pruned_records += len(cols["timestamp"])
                keys_to_remove.append((part_tenant, date_str))

        for key in keys_to_remove:
            del self._partitions[key]

        return pruned_records


class ElasticsearchConversationIndexer:
    """Multi-tenant inverted index for full-text conversation search and faceted discovery.

    Enforces strict tenant isolation via routing keys, preventing cross-tenant data leaks.
    """

    def __init__(self) -> None:
        # Key: tenant_id -> List of indexed document records
        self._tenant_indices: Dict[str, List[Dict[str, Any]]] = {}

    def index_turn(self, event: ConversationTurnEvent) -> str:
        """Indexes a conversational turn into the tenant-isolated index shard."""
        if event.tenant_id not in self._tenant_indices:
            self._tenant_indices[event.tenant_id] = []

        searchable_content = f"{event.text} {event.intent}"
        tokens = set(re.findall(r"[a-zA-Z0-9]+", searchable_content.lower()))
        doc = {
            "turn_id": event.turn_id,
            "conversation_id": event.conversation_id,
            "tenant_id": event.tenant_id,
            "agent_id": event.agent_id,
            "customer_id": event.customer_id,
            "timestamp": event.timestamp,
            "role": event.role.value if isinstance(event.role, TurnRole) else str(event.role),
            "text": event.text,
            "tokens": tokens,
            "sentiment_score": event.sentiment_score,
            "intent": event.intent,
        }
        self._tenant_indices[event.tenant_id].append(doc)
        return event.turn_id

    def search(
        self,
        tenant_id: str,
        query: str,
        intent: Optional[str] = None,
        min_sentiment: Optional[float] = None,
        max_sentiment: Optional[float] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Executes full-text keyword search and faceted filtering strictly within tenant boundary."""
        if tenant_id not in self._tenant_indices:
            return []

        query_tokens = set(re.findall(r"[a-zA-Z0-9]+", query.lower()))
        results: List[Tuple[int, Dict[str, Any]]] = []

        for doc in self._tenant_indices[tenant_id]:
            # Faceted intent filter
            if intent and doc["intent"].lower() != intent.lower():
                continue
            # Faceted sentiment range filters
            if min_sentiment is not None and doc["sentiment_score"] < min_sentiment:
                continue
            if max_sentiment is not None and doc["sentiment_score"] > max_sentiment:
                continue

            # Token overlap score
            match_score = len(doc["tokens"].intersection(query_tokens))
            if query_tokens and match_score == 0:
                continue

            # Clean output record without internal token set
            record = {k: v for k, v in doc.items() if k != "tokens"}
            record["score"] = match_score if query_tokens else 1
            results.append((record["score"], record))

        # Sort descending by match score, then by timestamp descending
        results.sort(key=lambda x: (x[0], x[1]["timestamp"]), reverse=True)
        return [r[1] for r in results[:limit]]

    def faceted_aggregate_intents(self, tenant_id: str) -> Dict[str, int]:
        """Returns aggregation counts of conversation intents within a tenant."""
        if tenant_id not in self._tenant_indices:
            return {}
        counts: Dict[str, int] = {}
        for doc in self._tenant_indices[tenant_id]:
            intent = doc["intent"]
            counts[intent] = counts.get(intent, 0) + 1
        return counts


class CrestaCDPRealtimeEngine:
    """Unified Customer Data Platform (CDP) Real-Time Analytics & Intelligence Engine.

    Coordinates ClickHouse columnar streaming, Elasticsearch multi-tenant search,
    live coaching recommendation triggers, and sub-second latency SLO tracking.
    """

    def __init__(self, buffer_batch_size: int = 50, latency_slo_target_ms: float = 100.0) -> None:
        self.columnar_buffer = ClickHouseColumnarBuffer(batch_size=buffer_batch_size)
        self.search_indexer = ElasticsearchConversationIndexer()
        self.latency_slo_target_ms = latency_slo_target_ms

        self._ingested_count: int = 0
        self._ingest_latencies_ms: List[float] = []
        self._slo_violations: int = 0

    def ingest_turn(self, event: ConversationTurnEvent) -> Tuple[bool, Optional[CoachingRecommendation]]:
        """Ingests a conversation turn into both ClickHouse and Elasticsearch engines.

        Returns (flushed, coaching_recommendation).
        """
        start = time.perf_counter()

        # Ingest to ClickHouse buffer and Elasticsearch search index
        flushed = self.columnar_buffer.append(event)
        self.search_indexer.index_turn(event)

        # Evaluate real-time agent coaching triggers
        recommendation = self._evaluate_coaching_trigger(event)

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self._ingested_count += 1
        self._ingest_latencies_ms.append(elapsed_ms)

        if elapsed_ms > self.latency_slo_target_ms:
            self._slo_violations += 1

        return flushed, recommendation

    def _evaluate_coaching_trigger(self, event: ConversationTurnEvent) -> Optional[CoachingRecommendation]:
        """Rules and semantic heuristic engine generating real-time agent guidance."""
        text_lower = event.text.lower()

        # Trigger 1: Severe customer dissatisfaction / negative sentiment escalation
        if event.role == TurnRole.CUSTOMER and event.sentiment_score <= -0.5:
            return CoachingRecommendation(
                recommendation_id=f"rec-{event.turn_id}-esc",
                conversation_id=event.conversation_id,
                turn_id=event.turn_id,
                tenant_id=event.tenant_id,
                action_type=CoachingActionType.ESCALATION,
                guidance_text=(
                    "Customer sentiment is critically negative. Acknowledge frustration empathetically "
                    "and offer immediate senior supervisor escalation or complimentary credit."
                ),
                confidence_score=0.95,
                trigger_reason="Customer sentiment score critically negative (<= -0.50)",
            )

        # Trigger 2: Cancellation or churn intent
        if event.intent in ("cancel_subscription", "close_account", "churn_risk"):
            return CoachingRecommendation(
                recommendation_id=f"rec-{event.turn_id}-churn",
                conversation_id=event.conversation_id,
                turn_id=event.turn_id,
                tenant_id=event.tenant_id,
                action_type=CoachingActionType.SCRIPT_GUIDANCE,
                guidance_text=(
                    "Customer expressed account cancellation intent. Highlight loyalty retention discount "
                    "and inquire about specific feature or service blockers."
                ),
                confidence_score=0.88,
                trigger_reason=f"Detected high-risk retention intent: '{event.intent}'",
            )

        # Trigger 3: Compliance disclaimer warning for agent
        if event.role == TurnRole.AGENT and any(w in text_lower for w in ["recording", "credit card", "cvv", "ssn"]):
            return CoachingRecommendation(
                recommendation_id=f"rec-{event.turn_id}-comp",
                conversation_id=event.conversation_id,
                turn_id=event.turn_id,
                tenant_id=event.tenant_id,
                action_type=CoachingActionType.COMPLIANCE_WARNING,
                guidance_text=(
                    "Sensitive payment/PII token detected. Verify payment pause recording is active "
                    "prior to reading payment transaction disclosures."
                ),
                confidence_score=0.92,
                trigger_reason="Detected compliance-regulated keyword pattern",
            )

        # Trigger 4: Mild negative sentiment proactive coaching suggestion
        if event.role == TurnRole.CUSTOMER and event.sentiment_score < -0.2:
            return CoachingRecommendation(
                recommendation_id=f"rec-{event.turn_id}-sugg",
                conversation_id=event.conversation_id,
                turn_id=event.turn_id,
                tenant_id=event.tenant_id,
                action_type=CoachingActionType.SUGGESTION,
                guidance_text=(
                    "Customer shows mild hesitation. Summarize the resolution steps and confirm satisfaction."
                ),
                confidence_score=0.75,
                trigger_reason="Moderate negative sentiment detected",
            )

        return None

    def get_telemetry_metrics(self) -> Dict[str, Any]:
        """Computes ingestion throughput, P50/P95 latency, and SLO compliance."""
        p50 = 0.0
        p95 = 0.0
        if self._ingest_latencies_ms:
            sorted_lat = sorted(self._ingest_latencies_ms)
            p50_idx = int(0.50 * len(sorted_lat))
            p95_idx = int(math.ceil(0.95 * len(sorted_lat))) - 1
            p50 = sorted_lat[max(0, min(p50_idx, len(sorted_lat) - 1))]
            p95 = sorted_lat[max(0, min(p95_idx, len(sorted_lat) - 1))]

        return {
            "total_ingested_turns": self._ingested_count,
            "p50_ingestion_latency_ms": round(p50, 3),
            "p95_ingestion_latency_ms": round(p95, 3),
            "latency_slo_target_ms": self.latency_slo_target_ms,
            "slo_violations": self._slo_violations,
            "slo_compliance_pct": (
                round((1.0 - (self._slo_violations / self._ingested_count)) * 100.0, 2)
                if self._ingested_count > 0
                else 100.0
            ),
        }

    def health_check(self) -> Dict[str, Any]:
        """Returns health status of the CDP streaming and search components."""
        return {
            "status": "healthy",
            "clickhouse_buffer_pending": len(self.columnar_buffer._buffer),
            "clickhouse_total_flushed": self.columnar_buffer._total_flushed_rows,
            "elasticsearch_tenants_active": len(self.search_indexer._tenant_indices),
            "timestamp": time.time(),
        }
