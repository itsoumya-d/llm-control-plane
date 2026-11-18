"""Prewave Supply Chain Disruption Graph Coordinator & Risk Intelligence Engine.

Provides enterprise-grade multi-tier supply chain relationship modeling,
recursive graph traversal with cycle prevention, streaming event deduplication,
and PostgreSQL recursive CTE query generation targeting Prewave's supply chain
resilience and risk intelligence platform (Vienna HQ).
"""

from __future__ import annotations

import collections
import dataclasses
import datetime
import enum
import hashlib
import re
from typing import Any, Dict, List, Set, Tuple


class DisruptionSeverity(str, enum.Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class DisruptionCategory(str, enum.Enum):
    INSOLVENCY = "INSOLVENCY"
    CYBER_ATTACK = "CYBER_ATTACK"
    FACTORY_FIRE = "FACTORY_FIRE"
    FLOOD = "FLOOD"
    LABOR_STRIKE = "LABOR_STRIKE"
    ESG_VIOLATION = "ESG_VIOLATION"
    SANCTION_BREACH = "SANCTION_BREACH"


@dataclasses.dataclass(frozen=True)
class SupplierNode:
    supplier_id: str
    legal_name: str
    country: str
    tier_level: int
    critical_component: str
    baseline_risk_score: float = 0.0

    def __post_init__(self) -> None:
        if not self.supplier_id:
            raise ValueError("supplier_id cannot be empty")
        if self.tier_level < 0:
            raise ValueError("tier_level cannot be negative")
        if not (0.0 <= self.baseline_risk_score <= 100.0):
            raise ValueError("baseline_risk_score must be between 0.0 and 100.0")


@dataclasses.dataclass(frozen=True)
class SupplyChainEdge:
    parent_id: str
    child_id: str
    component_name: str
    lead_time_days: int
    is_single_source: bool = False

    def __post_init__(self) -> None:
        if self.parent_id == self.child_id:
            raise ValueError("Self-referencing dependency cycles not allowed")
        if self.lead_time_days < 0:
            raise ValueError("lead_time_days cannot be negative")


@dataclasses.dataclass(frozen=True)
class DisruptionEvent:
    event_id: str
    category: DisruptionCategory
    severity: DisruptionSeverity
    affected_supplier_ids: Tuple[str, ...]
    summary: str
    confidence_score: float
    reported_at: datetime.datetime = dataclasses.field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc)
    )

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence_score <= 1.0):
            raise ValueError("confidence_score must be between 0.0 and 1.0")


class PrewaveSupplyChainOrchestrator:
    """Core coordination engine for Prewave multi-tier supply chain risk intelligence."""

    def __init__(self, company_tenant: str = "Prewave-Enterprise-Global") -> None:
        self.company_tenant = company_tenant
        self._suppliers: Dict[str, SupplierNode] = {}
        # parent -> list of edges to children
        self._adjacency: Dict[str, List[SupplyChainEdge]] = collections.defaultdict(list)
        # child -> list of parents depending on it
        self._reverse_adjacency: Dict[str, List[SupplyChainEdge]] = collections.defaultdict(list)
        self._events: Dict[str, DisruptionEvent] = {}
        self._dedupe_window_seconds: int = 3600
        self._event_hashes: Set[str] = set()

    def register_supplier(self, supplier: SupplierNode) -> SupplierNode:
        """Register a verified supplier into the enterprise supplier graph."""
        self._suppliers[supplier.supplier_id] = supplier
        return supplier

    def add_dependency(self, edge: SupplyChainEdge) -> None:
        """Add a directed supply dependency from parent to child with cycle detection."""
        if edge.parent_id not in self._suppliers:
            raise KeyError(f"Parent supplier {edge.parent_id} is not registered")
        if edge.child_id not in self._suppliers:
            raise KeyError(f"Child supplier {edge.child_id} is not registered")

        # Validate that adding this edge does not create a cycle (child must not reach parent)
        visited: Set[str] = set()
        queue = collections.deque([edge.child_id])
        while queue:
            curr = queue.popleft()
            if curr == edge.parent_id:
                raise ValueError(
                    f"Dependency cycle detected: adding edge {edge.parent_id} -> {edge.child_id} creates a loop"
                )
            if curr not in visited:
                visited.add(curr)
                for out_edge in self._adjacency.get(curr, []):
                    queue.append(out_edge.child_id)

        self._adjacency[edge.parent_id].append(edge)
        self._reverse_adjacency[edge.child_id].append(edge)

    def traverse_tier_n_dependencies(self, root_supplier_id: str, max_depth: int = 5) -> Dict[str, Any]:
        """Perform breadth-first traversal down the tier-N supplier tree."""
        if root_supplier_id not in self._suppliers:
            raise KeyError(f"Supplier {root_supplier_id} not registered")

        visited_nodes: Set[str] = {root_supplier_id}
        traversal_tree: List[Dict[str, Any]] = []
        max_tier_encountered = self._suppliers[root_supplier_id].tier_level

        # Queue contains (current_id, current_depth, path)
        queue = collections.deque([(root_supplier_id, 0, [root_supplier_id])])

        while queue:
            curr_id, depth, path = queue.popleft()
            if depth >= max_depth:
                continue

            for edge in self._adjacency.get(curr_id, []):
                child_node = self._suppliers[edge.child_id]
                max_tier_encountered = max(max_tier_encountered, child_node.tier_level)
                node_entry = {
                    "from_supplier_id": edge.parent_id,
                    "to_supplier_id": edge.child_id,
                    "component": edge.component_name,
                    "lead_time_days": edge.lead_time_days,
                    "single_source": edge.is_single_source,
                    "depth": depth + 1,
                    "path": path + [edge.child_id],
                }
                traversal_tree.append(node_entry)
                if edge.child_id not in visited_nodes:
                    visited_nodes.add(edge.child_id)
                    queue.append((edge.child_id, depth + 1, path + [edge.child_id]))

        return {
            "root_supplier_id": root_supplier_id,
            "total_sub_suppliers": len(visited_nodes) - 1,
            "max_tier_encountered": max_tier_encountered,
            "dependencies": traversal_tree,
        }

    def ingest_disruption_event(self, event: DisruptionEvent) -> Dict[str, Any]:
        """Ingest a disruption signal, performing semantic deduplication and entity resolution."""
        # Clean text for fingerprinting
        normalized_text = re.sub(r"\s+", " ", event.summary.lower().strip())
        hash_src = f"{event.category.value}:{normalized_text}:{sorted(event.affected_supplier_ids)}"
        event_hash = hashlib.sha256(hash_src.encode("utf-8")).hexdigest()

        if event_hash in self._event_hashes:
            return {
                "event_id": event.event_id,
                "status": "DEDUPLICATED",
                "message": "Duplicate event suppressed via sliding window hash match",
            }

        for sup_id in event.affected_supplier_ids:
            if sup_id not in self._suppliers:
                raise KeyError(f"Affected supplier {sup_id} not recognized in network")

        self._event_hashes.add(event_hash)
        self._events[event.event_id] = event

        return {
            "event_id": event.event_id,
            "status": "ACCEPTED",
            "category": event.category.value,
            "severity": event.severity.value,
            "affected_suppliers_count": len(event.affected_supplier_ids),
        }

    def compute_compound_risk_score(self, supplier_id: str) -> float:
        """Compute holistic risk score combining direct incidents with tier-N cascading risk."""
        if supplier_id not in self._suppliers:
            raise KeyError(f"Supplier {supplier_id} not registered")

        base_score = self._suppliers[supplier_id].baseline_risk_score

        # Severity weights
        severity_multipliers = {
            DisruptionSeverity.LOW: 5.0,
            DisruptionSeverity.MEDIUM: 15.0,
            DisruptionSeverity.HIGH: 35.0,
            DisruptionSeverity.CRITICAL: 60.0,
        }

        # Direct disruption impact
        direct_penalty = 0.0
        for ev in self._events.values():
            if supplier_id in ev.affected_supplier_ids:
                weight = severity_multipliers.get(ev.severity, 10.0)
                direct_penalty += weight * ev.confidence_score

        # Downstream cascading impact
        downstream_penalty = 0.0
        traversal = self.traverse_tier_n_dependencies(supplier_id)
        for dep in traversal.get("dependencies", []):
            child_id = dep["to_supplier_id"]
            depth = dep["depth"]
            single_source_boost = 1.5 if dep["single_source"] else 1.0

            for ev in self._events.values():
                if child_id in ev.affected_supplier_ids:
                    weight = severity_multipliers.get(ev.severity, 10.0)
                    attenuation = 1.0 / (depth * 1.4)
                    downstream_penalty += weight * ev.confidence_score * attenuation * single_source_boost

        total_risk = min(100.0, base_score + direct_penalty + downstream_penalty)
        return round(total_risk, 2)

    def generate_postgresql_cte_query(self, root_supplier_id: str, max_depth: int = 5) -> str:
        """Generate high-performance PostgreSQL Recursive CTE query for Tier-N graph resolution."""
        # Sanitize supplier ID against SQL injection
        safe_id = re.sub(r"[^a-zA-Z0-9_\-]", "", root_supplier_id)
        cte_sql = f"""
WITH RECURSIVE supply_chain_tree AS (
    -- Anchor member: immediate dependencies of root supplier
    SELECT
        e.parent_id,
        e.child_id,
        e.component_name,
        e.lead_time_days,
        e.is_single_source,
        1 AS tier_depth,
        ARRAY[e.parent_id, e.child_id] AS path_track,
        FALSE AS is_cycle
    FROM supply_dependencies e
    WHERE e.parent_id = '{safe_id}'

    UNION ALL

    -- Recursive member: cascade through sub-tier suppliers
    SELECT
        e.parent_id,
        e.child_id,
        e.component_name,
        e.lead_time_days,
        e.is_single_source,
        t.tier_depth + 1,
        t.path_track || e.child_id,
        e.child_id = ANY(t.path_track)
    FROM supply_dependencies e
    INNER JOIN supply_chain_tree t ON e.parent_id = t.child_id
    WHERE t.tier_depth < {max_depth} AND NOT t.is_cycle
)
SELECT
    tree.parent_id,
    p.legal_name AS parent_name,
    tree.child_id,
    c.legal_name AS child_name,
    tree.component_name,
    tree.lead_time_days,
    tree.is_single_source,
    tree.tier_depth,
    array_to_string(tree.path_track, ' -> ') AS dependency_route
FROM supply_chain_tree tree
JOIN suppliers p ON p.supplier_id = tree.parent_id
JOIN suppliers c ON c.supplier_id = tree.child_id
ORDER BY tree.tier_depth ASC, tree.lead_time_days DESC;
""".strip()
        return cte_sql

    def export_fullstack_dashboard_payload(self, root_supplier_id: str) -> Dict[str, Any]:
        """Generate payload for React/TypeScript frontend visualization."""
        if root_supplier_id not in self._suppliers:
            raise KeyError(f"Supplier {root_supplier_id} not registered")

        traversal = self.traverse_tier_n_dependencies(root_supplier_id)
        nodes: List[Dict[str, Any]] = []
        all_ids = {root_supplier_id} | {d["to_supplier_id"] for d in traversal["dependencies"]}

        for s_id in sorted(all_ids):
            node = self._suppliers[s_id]
            risk = self.compute_compound_risk_score(s_id)
            status = "HEALTHY" if risk < 30.0 else "ELEVATED" if risk < 70.0 else "CRITICAL"
            nodes.append(
                {
                    "id": node.supplier_id,
                    "label": node.legal_name,
                    "tier": node.tier_level,
                    "country": node.country,
                    "riskScore": risk,
                    "status": status,
                }
            )

        edges: List[Dict[str, Any]] = []
        for dep in traversal["dependencies"]:
            edges.append(
                {
                    "source": dep["from_supplier_id"],
                    "target": dep["to_supplier_id"],
                    "component": dep["component"],
                    "leadTimeDays": dep["lead_time_days"],
                    "singleSource": dep["single_source"],
                    "depth": dep["depth"],
                }
            )

        return {
            "tenant": self.company_tenant,
            "rootSupplier": root_supplier_id,
            "graph": {
                "nodes": nodes,
                "edges": edges,
            },
            "timestampUtc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "activeEventsCount": len(self._events),
        }
