"""Comprehensive unit & integration test suite for Flexport Autonomous Supply Chain Orchestrator."""

from __future__ import annotations

import asyncio

import pytest

from llm_control_plane.flexport_autonomous_supply_chain_orchestrator import (
    CrossDocumentReconciliationAgent,
    CustomsDocumentExtractionEngine,
    DiscrepancySeverity,
    DurableLogisticsStateMachine,
    FlexportAutonomousSupplyChainOrchestrator,
    ISO6346Validator,
    LogisticsDocumentType,
    ModelProvider,
    MultiProviderLLMGateway,
    RegulatoryCustomsComplianceScreener,
    ShipmentMode,
    WorkflowState,
)


def test_iso_6346_container_validator():
    assert ISO6346Validator.is_valid_format("MSKU1234567") is True
    assert ISO6346Validator.is_valid_format("CMAU9876543") is True
    assert ISO6346Validator.is_valid_format("INVALID") is False
    assert ISO6346Validator.is_valid_format("MSK1234567") is False  # 3 chars prefix
    assert ISO6346Validator.is_valid_format("MSKU123456") is False  # 6 digits


def test_customs_document_extraction_engine():
    engine = CustomsDocumentExtractionEngine()
    raw_bl = (
        "OCEAN BILL OF LADING #BL-99201\n"
        "Container ID: MSKU1234567\n"
        "Gross Weight: 14250.0 kg\n"
        "Declared Value: $125,000.00\n"
        "Port of Loading: Shanghai Port -> Port of Discharge: Port of Rotterdam"
    )
    doc = engine.extract_document(
        doc_id="doc_bl_001",
        doc_type=LogisticsDocumentType.BILL_OF_LADING,
        raw_text=raw_bl,
    )
    assert doc.doc_id == "doc_bl_001"
    assert doc.doc_type == LogisticsDocumentType.BILL_OF_LADING
    assert doc.parsed_fields.get("container_id") == "MSKU1234567"
    assert doc.parsed_fields.get("weight_kg") == 14250.0
    assert doc.parsed_fields.get("total_value_usd") == 125000.0
    assert len(doc.checksum_sha256) == 64
    assert doc.confidence_score >= 0.95


def test_cross_document_reconciliation_zero_discrepancy():
    engine = CustomsDocumentExtractionEngine()
    reconciler = CrossDocumentReconciliationAgent(weight_tolerance_percent=2.0)

    doc_bl = engine.extract_document(
        doc_id="bl_1",
        doc_type=LogisticsDocumentType.BILL_OF_LADING,
        raw_text="B/L data",
        injected_fields={"container_id": "MSKU1234567", "weight_kg": 10000.0, "total_value_usd": 50000.0},
    )
    doc_pl = engine.extract_document(
        doc_id="pl_1",
        doc_type=LogisticsDocumentType.PACKING_LIST,
        raw_text="P/L data",
        injected_fields={"container_id": "MSKU1234567", "weight_kg": 10050.0},  # 0.5% delta
    )
    doc_ci = engine.extract_document(
        doc_id="ci_1",
        doc_type=LogisticsDocumentType.COMMERCIAL_INVOICE,
        raw_text="C/I data",
        injected_fields={"total_value_usd": 50000.0},
    )

    reports = reconciler.reconcile([doc_bl, doc_pl, doc_ci])
    assert len(reports) == 0


def test_cross_document_reconciliation_weight_variance():
    engine = CustomsDocumentExtractionEngine()
    reconciler = CrossDocumentReconciliationAgent(weight_tolerance_percent=2.0)

    doc_bl = engine.extract_document(
        doc_id="bl_2",
        doc_type=LogisticsDocumentType.BILL_OF_LADING,
        raw_text="B/L data",
        injected_fields={"container_id": "MSKU1234567", "weight_kg": 12000.0},
    )
    doc_pl = engine.extract_document(
        doc_id="pl_2",
        doc_type=LogisticsDocumentType.PACKING_LIST,
        raw_text="P/L data",
        injected_fields={"container_id": "MSKU1234567", "weight_kg": 10000.0},  # 20% delta
    )

    reports = reconciler.reconcile([doc_bl, doc_pl])
    assert len(reports) == 1
    assert reports[0].field_name == "weight_kg"
    assert reports[0].severity == DiscrepancySeverity.HIGH_CARRIER_HOLD
    delta = reports[0].delta_percentage
    assert delta == 16.67 or delta == 20.0 or delta > 10.0


def test_cross_document_reconciliation_container_mismatch():
    engine = CustomsDocumentExtractionEngine()
    reconciler = CrossDocumentReconciliationAgent()

    doc_bl = engine.extract_document(
        doc_id="bl_3",
        doc_type=LogisticsDocumentType.BILL_OF_LADING,
        raw_text="B/L data",
        injected_fields={"container_id": "MSKU1234567", "weight_kg": 5000.0},
    )
    doc_pl = engine.extract_document(
        doc_id="pl_3",
        doc_type=LogisticsDocumentType.PACKING_LIST,
        raw_text="P/L data",
        injected_fields={"container_id": "CMAU7654321", "weight_kg": 5000.0},
    )

    reports = reconciler.reconcile([doc_bl, doc_pl])
    assert any(r.field_name == "container_id" and r.severity == DiscrepancySeverity.HIGH_CARRIER_HOLD for r in reports)


def test_regulatory_compliance_sanctions_trigger():
    screener = RegulatoryCustomsComplianceScreener()
    cleared, reports = screener.screen_customs(
        hts_code="8471.30.01",
        origin_country="IR",  # Iran - sanctioned
        destination_country="NL",
        has_certificate_of_origin=True,
    )
    assert cleared is False
    assert len(reports) == 1
    assert reports[0].severity == DiscrepancySeverity.CRITICAL_CUSTOMS_SANCTION


def test_regulatory_compliance_origin_certificate_prerequisite():
    screener = RegulatoryCustomsComplianceScreener()
    # 8542.31.00 requires Certificate of Origin
    cleared, reports = screener.screen_customs(
        hts_code="8542.31.00",
        origin_country="TW",
        destination_country="NL",
        has_certificate_of_origin=False,
    )
    assert cleared is False
    assert any(
        r.field_name == "certificate_of_origin" and r.severity == DiscrepancySeverity.HIGH_CARRIER_HOLD for r in reports
    )


def test_llm_gateway_token_cost_and_fallback():
    async def _run():
        gateway = MultiProviderLLMGateway(token_budget_usd=1.0)
        resp, trace = await gateway.execute_task(
            task_name="test_customs_eval",
            prompt="Analyze container manifest for customs declaration",
            preferred_provider=ModelProvider.ANTHROPIC_CLAUDE,
            simulate_failure=True,  # Triggers fallback to GPT-4
        )
        assert trace.provider == ModelProvider.OPENAI_GPT4
        assert trace.cost_usd > 0
        assert trace.latency_ms > 0
        assert gateway.total_cost_usd > 0
        assert "Autonomous Agent Output" in resp

    asyncio.run(_run())


def test_llm_gateway_budget_exceeded():
    async def _run():
        gateway = MultiProviderLLMGateway(token_budget_usd=0.0001)
        # First call succeeds and consumes budget
        await gateway.execute_task("task_1", "small prompt")
        # Second call should fail with budget error
        with pytest.raises(RuntimeError, match="Token budget exceeded"):
            await gateway.execute_task("task_2", "second prompt")

    asyncio.run(_run())


def test_durable_state_machine_audit_chain_and_hitl():
    sm = DurableLogisticsStateMachine("wf_ship_001")
    assert sm.current_state == WorkflowState.SUBMITTED

    sm.record_transition("PARSE_COMPLETE", WorkflowState.DOCUMENTS_PARSED, {"docs": 3})
    assert sm.current_state == WorkflowState.DOCUMENTS_PARSED

    hitl = sm.escalate_hitl(
        severity=DiscrepancySeverity.HIGH_CARRIER_HOLD,
        reason="Container mismatch at port gate",
        assigned_team="amsterdam_customs",
    )
    assert sm.current_state == WorkflowState.HITL_ESCALATED
    assert hitl.resolved is False

    resolved = sm.resolve_hitl(hitl.task_id, resolved_by="soumya.debnath", notes="Physical gate scan verified B/L")
    assert resolved is not None
    assert resolved.resolved is True
    assert sm.current_state == WorkflowState.RECONCILIATION_PASSED
    assert sm.verify_audit_chain() is True


def test_end_to_end_clean_shipment_settlement():
    async def _run():
        orch = FlexportAutonomousSupplyChainOrchestrator(
            workflow_id="flex_ams_1001",
            shipment_mode=ShipmentMode.OCEAN_FCL,
            weight_tolerance_pct=2.0,
        )

        docs = [
            (
                "doc_bl_1001",
                LogisticsDocumentType.BILL_OF_LADING,
                "B/L Ocean Freight",
                {"container_id": "MSKU9988776", "weight_kg": 18500.0, "total_value_usd": 95000.0},
            ),
            (
                "doc_pl_1001",
                LogisticsDocumentType.PACKING_LIST,
                "Packing List",
                {"container_id": "MSKU9988776", "weight_kg": 18500.0},
            ),
            (
                "doc_ci_1001",
                LogisticsDocumentType.COMMERCIAL_INVOICE,
                "Commercial Invoice",
                {"total_value_usd": 95000.0},
            ),
        ]

        result = await orch.ingest_and_process_shipment(
            documents=docs,
            hts_code="8471.30.01",  # Laptops - no cert required
            origin_country="TW",
            destination_country="NL",
            has_certificate_of_origin=False,
        )

        assert result["status"] == "CLEAN_SETTLEMENT"
        assert result["current_state"] == WorkflowState.SETTLED.value
        assert result["audit_chain_valid"] is True
        assert len(result["discrepancies"]) == 0
        assert result["hitl_pending"] == 0
        assert result["total_token_cost_usd"] > 0

    asyncio.run(_run())


def test_end_to_end_discrepancy_hitl_escalation():
    async def _run():
        orch = FlexportAutonomousSupplyChainOrchestrator(
            workflow_id="flex_ams_1002",
            shipment_mode=ShipmentMode.OCEAN_FCL,
            weight_tolerance_pct=2.0,
        )

        docs = [
            (
                "doc_bl_1002",
                LogisticsDocumentType.BILL_OF_LADING,
                "B/L Ocean Freight",
                {"container_id": "MSKU1112223", "weight_kg": 25000.0},
            ),
            (
                "doc_pl_1002",
                LogisticsDocumentType.PACKING_LIST,
                "Packing List",
                {"container_id": "MSKU1112223", "weight_kg": 15000.0},  # 40% weight variance
            ),
        ]

        result = await orch.ingest_and_process_shipment(
            documents=docs,
            hts_code="8471.30.01",
            origin_country="VN",
            destination_country="NL",
        )

        assert result["status"] == "ESCALATED_TO_HUMAN"
        assert result["current_state"] == WorkflowState.HITL_ESCALATED.value
        assert result["audit_chain_valid"] is True
        assert len(result["discrepancies"]) >= 1
        assert result["hitl_pending"] == 1

    asyncio.run(_run())
