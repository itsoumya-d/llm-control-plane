"""
Unit tests for Autonomous AP Invoice Exception Resolution and Reconciliation Engine.
"""

from __future__ import annotations

import pytest

from llm_control_plane.invoice_exception_resolver import (
    ApprovalState,
    ExceptionType,
    InvoiceDocument,
    InvoiceExceptionResolutionEngine,
    InvoiceLineItem,
    PurchaseOrderLine,
    ResolutionAction,
)


@pytest.fixture
def sample_po_lines():
    return {
        "PO-101": PurchaseOrderLine(
            po_line_id="PO-101",
            item_sku="SERVER-RACK-01",
            description="Enterprise Rack Server Unit",
            expected_quantity=5.0,
            expected_unit_price=1200.0,
            expected_gl_account="GL-6100-HARDWARE",
            tolerance_pct=2.0,
        ),
        "PO-102": PurchaseOrderLine(
            po_line_id="PO-102",
            item_sku="CABLE-SFP-10G",
            description="10G Direct Attach SFP+ Cable",
            expected_quantity=20.0,
            expected_unit_price=45.0,
            expected_gl_account="GL-6100-HARDWARE",
            tolerance_pct=2.0,
        ),
    }


def test_clean_three_way_match(sample_po_lines):
    engine = InvoiceExceptionResolutionEngine()
    invoice = InvoiceDocument(
        invoice_id="INV-2026-001",
        vendor_id="VEND-DELL",
        vendor_name="Dell Technologies GmbH",
        po_number="PO-99881",
        currency="EUR",
        line_items=[
            InvoiceLineItem(
                line_number=1,
                description="Enterprise Rack Server Unit",
                quantity=5.0,
                unit_price=1200.0,
                total_amount=6000.0,
                tax_rate=0.20,
                gl_account="GL-6100-HARDWARE",
                po_line_id="PO-101",
            ),
            InvoiceLineItem(
                line_number=2,
                description="10G Direct Attach SFP+ Cable",
                quantity=20.0,
                unit_price=45.0,
                total_amount=900.0,
                tax_rate=0.20,
                gl_account="GL-6100-HARDWARE",
                po_line_id="PO-102",
            ),
        ],
        total_net=6900.0,
        total_tax=1380.0,
        total_gross=8280.0,
    )

    clean, exceptions = engine.reconcile_invoice(invoice, sample_po_lines)
    assert clean is True
    assert len(exceptions) == 0

    res = engine.resolve_exceptions(invoice.invoice_id, exceptions)
    assert res["final_state"] == ApprovalState.APPROVED.value
    assert res["all_auto_resolved"] is True
    assert engine.verify_audit_chain() is True


def test_minor_price_variance_auto_resolution(sample_po_lines):
    engine = InvoiceExceptionResolutionEngine(auto_approve_threshold_usd=50.0)
    invoice = InvoiceDocument(
        invoice_id="INV-2026-002",
        vendor_id="VEND-DELL",
        vendor_name="Dell Technologies GmbH",
        po_number="PO-99881",
        currency="EUR",
        line_items=[
            InvoiceLineItem(
                line_number=1,
                description="10G Direct Attach SFP+ Cable",
                quantity=20.0,
                unit_price=46.0,
                total_amount=920.0,
                tax_rate=0.20,
                gl_account="GL-6100-HARDWARE",
                po_line_id="PO-102",
            ),
        ],
        total_net=920.0,
        total_tax=184.0,
        total_gross=1104.0,
    )

    clean, exceptions = engine.reconcile_invoice(invoice, sample_po_lines)
    assert clean is False
    assert len(exceptions) == 1
    assert exceptions[0].exception_type == ExceptionType.PRICE_VARIANCE
    assert exceptions[0].auto_resolvable is True
    assert (
        exceptions[0].recommended_action
        == ResolutionAction.APPLY_PRICE_TOLERANCE_ADJUSTMENT
    )

    res = engine.resolve_exceptions(invoice.invoice_id, exceptions)
    assert res["final_state"] == ApprovalState.AUTO_RESOLVED.value
    assert res["all_auto_resolved"] is True
    assert engine.verify_audit_chain() is True


def test_major_price_variance_escalation(sample_po_lines):
    engine = InvoiceExceptionResolutionEngine(
        auto_approve_threshold_usd=50.0, max_tolerance_pct=5.0
    )
    invoice = InvoiceDocument(
        invoice_id="INV-2026-003",
        vendor_id="VEND-DELL",
        vendor_name="Dell Technologies GmbH",
        po_number="PO-99881",
        currency="EUR",
        line_items=[
            InvoiceLineItem(
                line_number=1,
                description="Enterprise Rack Server Unit",
                quantity=5.0,
                unit_price=1350.0,
                total_amount=6750.0,
                tax_rate=0.20,
                gl_account="GL-6100-HARDWARE",
                po_line_id="PO-101",
            ),
        ],
        total_net=6750.0,
        total_tax=1350.0,
        total_gross=8100.0,
    )

    clean, exceptions = engine.reconcile_invoice(invoice, sample_po_lines)
    assert clean is False
    assert len(exceptions) == 1
    assert exceptions[0].exception_type == ExceptionType.PRICE_VARIANCE
    assert exceptions[0].auto_resolvable is False
    assert (
        exceptions[0].recommended_action
        == ResolutionAction.ESCALATE_TO_PROCUREMENT
    )

    res = engine.resolve_exceptions(invoice.invoice_id, exceptions)
    assert res["final_state"] == ApprovalState.ESCALATED.value
    assert res["all_auto_resolved"] is False
    assert engine.verify_audit_chain() is True


def test_quantity_mismatch_and_gl_account_drift(sample_po_lines):
    engine = InvoiceExceptionResolutionEngine()
    invoice = InvoiceDocument(
        invoice_id="INV-2026-004",
        vendor_id="VEND-DELL",
        vendor_name="Dell Technologies GmbH",
        po_number="PO-99881",
        currency="EUR",
        line_items=[
            InvoiceLineItem(
                line_number=1,
                description="Enterprise Rack Server Unit",
                quantity=8.0,
                unit_price=1200.0,
                total_amount=9600.0,
                tax_rate=0.20,
                gl_account="GL-9999-MISC",
                po_line_id="PO-101",
            ),
        ],
        total_net=9600.0,
        total_tax=1920.0,
        total_gross=11520.0,
    )

    clean, exceptions = engine.reconcile_invoice(invoice, sample_po_lines)
    assert clean is False
    types = {e.exception_type for e in exceptions}
    assert ExceptionType.QUANTITY_MISMATCH in types
    assert ExceptionType.GL_ACCOUNT_MISMATCH in types

    res = engine.resolve_exceptions(invoice.invoice_id, exceptions)
    assert res["final_state"] == ApprovalState.ESCALATED.value
    assert engine.verify_audit_chain() is True


def test_missing_po_line_and_arithmetic_mismatch(sample_po_lines):
    engine = InvoiceExceptionResolutionEngine()
    invoice = InvoiceDocument(
        invoice_id="INV-2026-005",
        vendor_id="VEND-DELL",
        vendor_name="Dell Technologies GmbH",
        po_number="PO-99881",
        currency="EUR",
        line_items=[
            InvoiceLineItem(
                line_number=1,
                description="Custom GPU Accelerator",
                quantity=1.0,
                unit_price=3500.0,
                total_amount=3500.0,
                tax_rate=0.20,
                gl_account="GL-6100-HARDWARE",
                po_line_id="PO-NON-EXISTENT",
            ),
        ],
        total_net=3000.0,
        total_tax=600.0,
        total_gross=3600.0,
    )

    clean, exceptions = engine.reconcile_invoice(invoice, sample_po_lines)
    assert clean is False
    assert any(e.exception_type == ExceptionType.MISSING_PO_LINE for e in exceptions)
    assert any(e.line_number == 0 for e in exceptions)

    audit_events = engine.get_audit_trail(invoice.invoice_id)
    assert len(audit_events) >= 1
    assert engine.verify_audit_chain() is True
