"""Unit tests for Prewave Supply Chain Disruption Graph Coordinator."""

import pytest

from llm_control_plane.prewave_supply_chain_orchestrator import (
    DisruptionCategory,
    DisruptionEvent,
    DisruptionSeverity,
    PrewaveSupplyChainOrchestrator,
    SupplierNode,
    SupplyChainEdge,
)


@pytest.fixture
def orchestrator() -> PrewaveSupplyChainOrchestrator:
    orc = PrewaveSupplyChainOrchestrator(company_tenant="Prewave-Vienna-Test")
    # Register 4-tier network: OEM -> Tier 1 -> Tier 2 -> Tier 3
    orc.register_supplier(
        SupplierNode(
            supplier_id="OEM-BMW",
            legal_name="BMW Group Manufacturing",
            country="DE",
            tier_level=0,
            critical_component="Final Assembly",
            baseline_risk_score=10.0,
        )
    )
    orc.register_supplier(
        SupplierNode(
            supplier_id="T1-BOSCH",
            legal_name="Bosch Mobility Solutions",
            country="DE",
            tier_level=1,
            critical_component="Braking ECU",
            baseline_risk_score=15.0,
        )
    )
    orc.register_supplier(
        SupplierNode(
            supplier_id="T2-INFINEON",
            legal_name="Infineon Technologies Austria",
            country="AT",
            tier_level=2,
            critical_component="Power Semiconductor Microcontroller",
            baseline_risk_score=12.0,
        )
    )
    orc.register_supplier(
        SupplierNode(
            supplier_id="T3-SILICON",
            legal_name="Silicon Wafer Smelting Corp",
            country="TW",
            tier_level=3,
            critical_component="High Purity Monocrystalline Silicon",
            baseline_risk_score=20.0,
        )
    )

    orc.add_dependency(
        SupplyChainEdge(
            parent_id="OEM-BMW",
            child_id="T1-BOSCH",
            component_name="Braking ECU Module",
            lead_time_days=14,
            is_single_source=False,
        )
    )
    orc.add_dependency(
        SupplyChainEdge(
            parent_id="T1-BOSCH",
            child_id="T2-INFINEON",
            component_name="Automotive Power IC",
            lead_time_days=45,
            is_single_source=True,
        )
    )
    orc.add_dependency(
        SupplyChainEdge(
            parent_id="T2-INFINEON",
            child_id="T3-SILICON",
            component_name="300mm Ingot Slices",
            lead_time_days=60,
            is_single_source=True,
        )
    )
    return orc


def test_supplier_node_validation() -> None:
    with pytest.raises(ValueError, match="supplier_id cannot be empty"):
        SupplierNode(
            supplier_id="",
            legal_name="Invalid",
            country="AT",
            tier_level=1,
            critical_component="Part",
        )

    with pytest.raises(ValueError, match="tier_level cannot be negative"):
        SupplierNode(
            supplier_id="SUP-1",
            legal_name="Invalid",
            country="AT",
            tier_level=-1,
            critical_component="Part",
        )

    with pytest.raises(ValueError, match="baseline_risk_score must be between"):
        SupplierNode(
            supplier_id="SUP-1",
            legal_name="Invalid",
            country="AT",
            tier_level=1,
            critical_component="Part",
            baseline_risk_score=150.0,
        )


def test_cycle_detection(orchestrator: PrewaveSupplyChainOrchestrator) -> None:
    # Attempting to add an edge from T3 back to OEM creates a cycle
    with pytest.raises(ValueError, match="Dependency cycle detected"):
        orchestrator.add_dependency(
            SupplyChainEdge(
                parent_id="T3-SILICON",
                child_id="OEM-BMW",
                component_name="Raw feedback loop",
                lead_time_days=5,
            )
        )

    with pytest.raises(ValueError, match="Self-referencing dependency cycles not allowed"):
        SupplyChainEdge(
            parent_id="T1-BOSCH",
            child_id="T1-BOSCH",
            component_name="Self",
            lead_time_days=1,
        )


def test_unregistered_supplier_error(orchestrator: PrewaveSupplyChainOrchestrator) -> None:
    with pytest.raises(KeyError, match="Parent supplier UNKNOWN is not registered"):
        orchestrator.add_dependency(
            SupplyChainEdge(
                parent_id="UNKNOWN",
                child_id="T1-BOSCH",
                component_name="Part",
                lead_time_days=5,
            )
        )


def test_tier_n_tree_traversal(orchestrator: PrewaveSupplyChainOrchestrator) -> None:
    tree = orchestrator.traverse_tier_n_dependencies("OEM-BMW", max_depth=5)
    assert tree["root_supplier_id"] == "OEM-BMW"
    assert tree["total_sub_suppliers"] == 3
    assert tree["max_tier_encountered"] == 3

    deps = tree["dependencies"]
    assert len(deps) == 3
    assert deps[0]["from_supplier_id"] == "OEM-BMW"
    assert deps[0]["to_supplier_id"] == "T1-BOSCH"
    assert deps[1]["from_supplier_id"] == "T1-BOSCH"
    assert deps[1]["to_supplier_id"] == "T2-INFINEON"
    assert deps[2]["from_supplier_id"] == "T2-INFINEON"
    assert deps[2]["to_supplier_id"] == "T3-SILICON"


def test_disruption_event_ingestion_and_deduplication(
    orchestrator: PrewaveSupplyChainOrchestrator,
) -> None:
    ev = DisruptionEvent(
        event_id="EV-2026-001",
        category=DisruptionCategory.FACTORY_FIRE,
        severity=DisruptionSeverity.CRITICAL,
        affected_supplier_ids=("T3-SILICON",),
        summary="Major chemical fire at semiconductor ingot purification foundry in Hsinchu",
        confidence_score=0.95,
    )
    res = orchestrator.ingest_disruption_event(ev)
    assert res["status"] == "ACCEPTED"
    assert res["category"] == "FACTORY_FIRE"
    assert res["severity"] == "CRITICAL"

    # Ingest duplicate event with whitespace difference
    ev_dup = DisruptionEvent(
        event_id="EV-2026-001-COPY",
        category=DisruptionCategory.FACTORY_FIRE,
        severity=DisruptionSeverity.CRITICAL,
        affected_supplier_ids=("T3-SILICON",),
        summary="Major  chemical fire at semiconductor ingot purification foundry in Hsinchu ",
        confidence_score=0.90,
    )
    res_dup = orchestrator.ingest_disruption_event(ev_dup)
    assert res_dup["status"] == "DEDUPLICATED"


def test_compound_risk_calculation(orchestrator: PrewaveSupplyChainOrchestrator) -> None:
    # Baseline risk for OEM-BMW before any events
    base_risk = orchestrator.compute_compound_risk_score("OEM-BMW")
    assert base_risk == 10.0

    # Ingest disruption event on Tier 3 supplier
    ev = DisruptionEvent(
        event_id="EV-FLOOD-T3",
        category=DisruptionCategory.FLOOD,
        severity=DisruptionSeverity.HIGH,
        affected_supplier_ids=("T3-SILICON",),
        summary="Typhoon flooding severely halts wafer slicing facility",
        confidence_score=0.90,
    )
    orchestrator.ingest_disruption_event(ev)

    t3_risk = orchestrator.compute_compound_risk_score("T3-SILICON")
    # Base 20.0 + (35.0 * 0.90) = 20.0 + 31.5 = 51.5
    assert t3_risk == 51.5

    # OEM-BMW risk should have cascaded upwards
    oem_risk = orchestrator.compute_compound_risk_score("OEM-BMW")
    assert oem_risk > base_risk


def test_postgresql_cte_query_generation(orchestrator: PrewaveSupplyChainOrchestrator) -> None:
    cte = orchestrator.generate_postgresql_cte_query("OEM-BMW", max_depth=4)
    assert "WITH RECURSIVE supply_chain_tree AS" in cte
    assert "WHERE e.parent_id = 'OEM-BMW'" in cte
    assert "WHERE t.tier_depth < 4 AND NOT t.is_cycle" in cte
    assert "JOIN suppliers p ON p.supplier_id = tree.parent_id" in cte


def test_fullstack_dashboard_payload(orchestrator: PrewaveSupplyChainOrchestrator) -> None:
    payload = orchestrator.export_fullstack_dashboard_payload("OEM-BMW")
    assert payload["tenant"] == "Prewave-Vienna-Test"
    assert payload["rootSupplier"] == "OEM-BMW"

    nodes = payload["graph"]["nodes"]
    assert len(nodes) == 4
    node_ids = [n["id"] for n in nodes]
    assert "OEM-BMW" in node_ids
    assert "T1-BOSCH" in node_ids
    assert "T2-INFINEON" in node_ids
    assert "T3-SILICON" in node_ids

    edges = payload["graph"]["edges"]
    assert len(edges) == 3
    assert any(e["singleSource"] is True for e in edges)
