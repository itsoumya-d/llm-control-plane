"""Stream Real-Time Messaging & Activity Feeds Storage Orchestrator.

Engineered for Stream's high-scale real-time communication platform (Chat, Activity Feeds, Video, Moderation):
- RealTimeEventFanoutEngine: High-concurrency event publishing, dynamic channel subscriptions,
  and token-bucket rate limiting across multi-tenant feeds.
- PostgresEventPartitionManager: Declarative PostgreSQL range partitioning for high-volume message event
  streams with automated partition lifecycle management and pruning.
- RocksDBLSMCompactionCoordinator: Embedded key-value LSM-tree write buffer management, write stall mitigation,
  and compaction scheduling for low-latency activity feed stores.
- StreamPlatformTelemetryAndSLOTracker: Real-time P99 delivery latency monitoring, throughput analytics,
  and cost-per-message efficiency tracking.
"""

from __future__ import annotations

import collections
import dataclasses
import datetime as dt
import enum
import math
import typing


class DeliveryState(str, enum.Enum):
    PENDING = "PENDING"
    DELIVERED = "DELIVERED"
    RATE_LIMITED = "RATE_LIMITED"
    DROPPED = "DROPPED"


@dataclasses.dataclass(frozen=True)
class StreamMessage:
    message_id: str
    channel_id: str
    tenant_id: str
    sender_id: str
    payload: typing.Dict[str, typing.Any]
    timestamp: dt.datetime
    priority: int = 1


@dataclasses.dataclass
class ChannelSubscription:
    channel_id: str
    subscriber_id: str
    joined_at: dt.datetime
    filter_tags: typing.Set[str] = dataclasses.field(default_factory=set)


@dataclasses.dataclass
class PartitionMetadata:
    table_name: str
    partition_name: str
    start_time: dt.datetime
    end_time: dt.datetime
    row_count_estimate: int = 0
    is_active: bool = True


@dataclasses.dataclass
class CompactionJob:
    job_id: str
    cf_name: str
    level_from: int
    level_to: int
    input_bytes: int
    scheduled_at: dt.datetime
    completed: bool = False


class RealTimeEventFanoutEngine:
    """High-concurrency pub/sub event distribution engine with multi-tenant rate limiting."""

    def __init__(self, tenant_rate_limit_per_sec: int = 1000):
        self.rate_limit = tenant_rate_limit_per_sec
        self._channels: typing.Dict[str, typing.Dict[str, ChannelSubscription]] = collections.defaultdict(dict)
        self._tenant_buckets: typing.Dict[str, typing.Tuple[float, float]] = {}  # tenant_id -> (tokens, last_time)
        self._delivery_log: typing.List[typing.Tuple[StreamMessage, DeliveryState, int]] = []

    def subscribe(
        self,
        channel_id: str,
        subscriber_id: str,
        tags: typing.Optional[typing.Set[str]] = None,
    ) -> ChannelSubscription:
        sub = ChannelSubscription(
            channel_id=channel_id,
            subscriber_id=subscriber_id,
            joined_at=dt.datetime.now(dt.timezone.utc),
            filter_tags=tags or set(),
        )
        self._channels[channel_id][subscriber_id] = sub
        return sub

    def unsubscribe(self, channel_id: str, subscriber_id: str) -> bool:
        if channel_id in self._channels and subscriber_id in self._channels[channel_id]:
            del self._channels[channel_id][subscriber_id]
            return True
        return False

    def get_subscriber_count(self, channel_id: str) -> int:
        return len(self._channels.get(channel_id, {}))

    def _consume_token(self, tenant_id: str, now: float) -> bool:
        tokens, last_time = self._tenant_buckets.get(tenant_id, (float(self.rate_limit), now))
        elapsed = max(0.0, now - last_time)
        tokens = min(float(self.rate_limit), tokens + elapsed * self.rate_limit)

        if tokens >= 1.0:
            self._tenant_buckets[tenant_id] = (tokens - 1.0, now)
            return True
        self._tenant_buckets[tenant_id] = (tokens, now)
        return False

    def publish(self, message: StreamMessage) -> typing.Tuple[DeliveryState, int]:
        now_ts = dt.datetime.now(dt.timezone.utc).timestamp()
        if not self._consume_token(message.tenant_id, now_ts):
            self._delivery_log.append((message, DeliveryState.RATE_LIMITED, 0))
            return DeliveryState.RATE_LIMITED, 0

        subscribers = self._channels.get(message.channel_id, {})
        delivered_count = 0
        msg_tags = set(message.payload.get("tags", []))

        for sub in subscribers.values():
            if not sub.filter_tags or (sub.filter_tags & msg_tags):
                delivered_count += 1

        state = DeliveryState.DELIVERED if delivered_count > 0 else DeliveryState.PENDING
        self._delivery_log.append((message, state, delivered_count))
        return state, delivered_count


class PostgresEventPartitionManager:
    """Manages declarative range partitioning for massive relational event and message stores."""

    def __init__(self, base_table: str = "stream_channel_messages"):
        self.base_table = base_table
        self._partitions: typing.Dict[str, PartitionMetadata] = {}

    def generate_create_partition_sql(self, start: dt.datetime, end: dt.datetime) -> str:
        part_name = f"{self.base_table}_{start.strftime('%Y_%m_%d')}"
        sql = (
            f"CREATE TABLE IF NOT EXISTS {part_name} PARTITION OF {self.base_table} "
            f"FOR VALUES FROM ('{start.isoformat()}') TO ('{end.isoformat()}');"
        )
        self._partitions[part_name] = PartitionMetadata(
            table_name=self.base_table,
            partition_name=part_name,
            start_time=start,
            end_time=end,
            is_active=True,
        )
        return sql

    def list_active_partitions(self) -> typing.List[PartitionMetadata]:
        return [p for p in self._partitions.values() if p.is_active]

    def prune_expired_partitions(self, cutoff: dt.datetime) -> typing.List[str]:
        dropped = []
        for name, meta in list(self._partitions.items()):
            if meta.end_time < cutoff:
                meta.is_active = False
                dropped.append(name)
        return dropped

    def resolve_target_partition(self, event_time: dt.datetime) -> typing.Optional[str]:
        for name, meta in self._partitions.items():
            if meta.is_active and meta.start_time <= event_time < meta.end_time:
                return name
        return None


class RocksDBLSMCompactionCoordinator:
    """Coordinates LSM-tree write buffer sizing, stall detection, and compaction priorities."""

    def __init__(self, max_write_buffer_bytes: int = 64 * 1024 * 1024, stall_threshold_ratio: float = 0.85):
        self.max_buffer = max_write_buffer_bytes
        self.stall_threshold = stall_threshold_ratio
        self._cf_buffers: typing.Dict[str, int] = collections.defaultdict(int)
        self._scheduled_jobs: typing.List[CompactionJob] = []

    def record_write(self, cf_name: str, payload_bytes: int) -> bool:
        """Records write; returns True if write stall alert is triggered."""
        self._cf_buffers[cf_name] += payload_bytes
        used_ratio = self._cf_buffers[cf_name] / self.max_buffer
        return used_ratio >= self.stall_threshold

    def get_buffer_utilization(self, cf_name: str) -> float:
        return min(1.0, self._cf_buffers.get(cf_name, 0) / self.max_buffer)

    def schedule_compaction(self, cf_name: str, level_from: int, level_to: int) -> CompactionJob:
        current_bytes = self._cf_buffers.get(cf_name, 0)
        job = CompactionJob(
            job_id=f"comp_{cf_name}_{level_from}_{level_to}_{len(self._scheduled_jobs)+1}",
            cf_name=cf_name,
            level_from=level_from,
            level_to=level_to,
            input_bytes=current_bytes,
            scheduled_at=dt.datetime.now(dt.timezone.utc),
        )
        self._scheduled_jobs.append(job)
        # Reset buffer on compaction
        self._cf_buffers[cf_name] = 0
        return job

    def calculate_write_amplification(self, total_bytes_written: int, user_payload_bytes: int) -> float:
        if user_payload_bytes <= 0:
            return 1.0
        return round(total_bytes_written / user_payload_bytes, 2)


class StreamPlatformTelemetryAndSLOTracker:
    """Tracks message delivery latency, P99 compliance, and cost efficiency per million events."""

    def __init__(self, target_p99_ms: float = 25.0, target_availability: float = 0.9999):
        self.target_p99 = target_p99_ms
        self.target_avail = target_availability
        self._latencies: typing.List[float] = []
        self._total_events: int = 0
        self._failed_events: int = 0
        self._infra_cost_usd: float = 0.0

    def record_event(self, latency_ms: float, success: bool = True, cost_usd: float = 0.0) -> None:
        self._total_events += 1
        self._infra_cost_usd += cost_usd
        if success:
            self._latencies.append(latency_ms)
        else:
            self._failed_events += 1

    def calculate_p99_latency_ms(self) -> float:
        if not self._latencies:
            return 0.0
        sorted_lats = sorted(self._latencies)
        idx = int(math.ceil(0.99 * len(sorted_lats))) - 1
        return round(sorted_lats[max(0, min(idx, len(sorted_lats) - 1))], 2)

    def calculate_availability(self) -> float:
        if self._total_events == 0:
            return 1.0
        return round((self._total_events - self._failed_events) / self._total_events, 4)

    def calculate_cost_per_million_events(self) -> float:
        if self._total_events == 0:
            return 0.0
        return round((self._infra_cost_usd / self._total_events) * 1_000_000, 3)

    def evaluate_slo_compliance(self) -> typing.Dict[str, typing.Any]:
        p99 = self.calculate_p99_latency_ms()
        avail = self.calculate_availability()
        p99_pass = p99 <= self.target_p99
        avail_pass = avail >= self.target_avail

        return {
            "p99_latency_ms": p99,
            "target_p99_ms": self.target_p99,
            "p99_pass": p99_pass,
            "availability": avail,
            "target_availability": self.target_avail,
            "availability_pass": avail_pass,
            "cost_per_million_usd": self.calculate_cost_per_million_events(),
            "overall_compliant": p99_pass and avail_pass,
        }
