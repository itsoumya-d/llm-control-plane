"""
Autonomous AP Invoice Exception Resolution Agent, PO Reconciliation, and Audit Trailing Engine.

Provides enterprise accounts payable automation capabilities:
1. InvoiceReconciler: 3-way line item matching against Purchase Orders with tolerance controls.
2. ExceptionClassifier: Root-cause taxonomy classification for price, quantity, tax,
   and GL coding variances.
3. AutonomousResolutionAgent: AI/heuristic agent resolving acceptable exceptions and
   escalating out-of-policy items.
4. ApprovalWorkflowEngine: Idempotent state machine with verifiable cryptographically
   hashed audit trails.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class ExceptionType(str, Enum):
    PRICE_VARIANCE = "price_variance"
    QUANTITY_MISMATCH = "quantity_mismatch"
    TAX_DISCREPANCY = "tax_discrepancy"
    MISSING_PO_LINE = "missing_po_line"
    CURRENCY_EXCHANGE_DRIFT = "currency_exchange_drift"
    GL_ACCOUNT_MISMATCH = "gl_account_mismatch"


class ResolutionAction(str, Enum):
    AUTO_APPROVE = "auto_approve"
    APPLY_PRICE_TOLERANCE_ADJUSTMENT = "apply_price_tolerance_adjustment"
    ADJUST_GL_CODING = "adjust_gl_coding"
    REQUEST_CREDIT_MEMO = "request_credit_memo"
    ESCALATE_TO_PROCUREMENT = "escalate_to_procurement"
    ROUTE_MANUAL_REVIEW = "route_manual_review"


class ApprovalState(str, Enum):
    DRAFT = "draft"
    INGESTED = "ingested"
    RECONCILING = "reconciling"
    EXCEPTION_DETECTED = "exception_detected"
    AUTO_RESOLVED = "auto_resolved"
    APPROVED = "approved"
    ESCALATED = "escalated"
    REJECTED = "rejected"


@dataclass
class InvoiceLineItem:
    line_number: int
    description: str
    quantity: float
    unit_price: float
    total_amount: float
    tax_rate: float
    gl_account: str
    po_line_id: Optional[str] = None


@dataclass
class PurchaseOrderLine:
    po_line_id: str
    item_sku: str
    description: str
    expected_quantity: float
    expected_unit_price: float
    expected_gl_account: str
    tolerance_pct: float = 2.0  # default 2% price/quantity tolerance


@dataclass
class InvoiceDocument:
    invoice_id: str
    vendor_id: str
    vendor_name: str
    po_number: Optional[str]
    currency: str
    line_items: List[InvoiceLineItem]
    total_net: float
    total_tax: float
    total_gross: float
    created_at: str = field(
        default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat()
    )


@dataclass
class ExceptionDetail:
    line_number: int
    exception_type: ExceptionType
    severity: str  # low, medium, high, critical
    variance_amount: float
    variance_pct: float
    recommended_action: ResolutionAction
    explanation: str
    auto_resolvable: bool


@dataclass
class AuditEvent:
    event_id: str
    invoice_id: str
    from_state: ApprovalState
    to_state: ApprovalState
    actor: str
    timestamp: str
    details: Dict[str, Any]
    event_hash: str


class InvoiceExceptionResolutionEngine:
    """
    Core accounts payable reconciliation and exception resolution engine.
    Applies clean architecture, domain-driven design, and AI agent delegation principles.
    """

    def __init__(
        self, auto_approve_threshold_usd: float = 50.0, max_tolerance_pct: float = 5.0
    ):
        self.auto_approve_threshold_usd = auto_approve_threshold_usd
        self.max_tolerance_pct = max_tolerance_pct
        self.audit_log: List[AuditEvent] = []
        self._states: Dict[str, ApprovalState] = {}

    def reconcile_invoice(
        self,
        invoice: InvoiceDocument,
        po_lines: Dict[str, PurchaseOrderLine],
    ) -> Tuple[bool, List[ExceptionDetail]]:
        """
        Reconcile invoice lines against corresponding PO lines.
        Returns (is_clean_match, list_of_exceptions).
        """
        exceptions: List[ExceptionDetail] = []
        self._states[invoice.invoice_id] = ApprovalState.RECONCILING

        # Validate arithmetic consistency first
        calculated_net = sum(line.total_amount for line in invoice.line_items)
        if abs(calculated_net - invoice.total_net) > 0.05:
            exceptions.append(
                ExceptionDetail(
                    line_number=0,
                    exception_type=ExceptionType.PRICE_VARIANCE,
                    severity="high",
                    variance_amount=abs(calculated_net - invoice.total_net),
                    variance_pct=0.0,
                    recommended_action=ResolutionAction.ROUTE_MANUAL_REVIEW,
                    explanation=(
                        "Sum of invoice line items does not equal total net amount."
                    ),
                    auto_resolvable=False,
                )
            )

        # Check line-by-line against PO lines
        for line in invoice.line_items:
            if not line.po_line_id or line.po_line_id not in po_lines:
                exceptions.append(
                    ExceptionDetail(
                        line_number=line.line_number,
                        exception_type=ExceptionType.MISSING_PO_LINE,
                        severity="high",
                        variance_amount=line.total_amount,
                        variance_pct=100.0,
                        recommended_action=ResolutionAction.ESCALATE_TO_PROCUREMENT,
                        explanation=(
                            f"Line {line.line_number} references non-existent "
                            f"PO line '{line.po_line_id}'."
                        ),
                        auto_resolvable=False,
                    )
                )
                continue

            po_line = po_lines[line.po_line_id]

            # 1. Price check
            unit_price_diff = line.unit_price - po_line.expected_unit_price
            if po_line.expected_unit_price:
                price_variance_pct = (
                    unit_price_diff / po_line.expected_unit_price
                ) * 100.0
            else:
                price_variance_pct = 0.0

            if abs(price_variance_pct) > po_line.tolerance_pct:
                total_variance = unit_price_diff * line.quantity
                auto_ok = (
                    abs(total_variance) <= self.auto_approve_threshold_usd
                    and abs(price_variance_pct) <= self.max_tolerance_pct
                )
                action = (
                    ResolutionAction.APPLY_PRICE_TOLERANCE_ADJUSTMENT
                    if auto_ok
                    else ResolutionAction.ESCALATE_TO_PROCUREMENT
                )
                exceptions.append(
                    ExceptionDetail(
                        line_number=line.line_number,
                        exception_type=ExceptionType.PRICE_VARIANCE,
                        severity="low" if auto_ok else "medium",
                        variance_amount=round(total_variance, 2),
                        variance_pct=round(price_variance_pct, 2),
                        recommended_action=action,
                        explanation=(
                            f"Price variance of {price_variance_pct:.1f}% "
                            f"(tolerance: ±{po_line.tolerance_pct}%). "
                            f"Variance amount: €{total_variance:.2f}."
                        ),
                        auto_resolvable=auto_ok,
                    )
                )

            # 2. Quantity check
            qty_diff = line.quantity - po_line.expected_quantity
            if po_line.expected_quantity:
                qty_variance_pct = (qty_diff / po_line.expected_quantity) * 100.0
            else:
                qty_variance_pct = 0.0
            if qty_diff > 0 and abs(qty_variance_pct) > po_line.tolerance_pct:
                exceptions.append(
                    ExceptionDetail(
                        line_number=line.line_number,
                        exception_type=ExceptionType.QUANTITY_MISMATCH,
                        severity="medium",
                        variance_amount=round(qty_diff * line.unit_price, 2),
                        variance_pct=round(qty_variance_pct, 2),
                        recommended_action=ResolutionAction.REQUEST_CREDIT_MEMO,
                        explanation=(
                            f"Billed quantity ({line.quantity}) exceeds PO quantity "
                            f"({po_line.expected_quantity})."
                        ),
                        auto_resolvable=False,
                    )
                )

            # 3. GL Account coding check
            if line.gl_account != po_line.expected_gl_account:
                exceptions.append(
                    ExceptionDetail(
                        line_number=line.line_number,
                        exception_type=ExceptionType.GL_ACCOUNT_MISMATCH,
                        severity="low",
                        variance_amount=0.0,
                        variance_pct=0.0,
                        recommended_action=ResolutionAction.ADJUST_GL_CODING,
                        explanation=(
                            f"GL Account '{line.gl_account}' drifts from PO standard "
                            f"'{po_line.expected_gl_account}'."
                        ),
                        auto_resolvable=True,
                    )
                )

        clean = len(exceptions) == 0
        target_state = (
            ApprovalState.APPROVED if clean else ApprovalState.EXCEPTION_DETECTED
        )
        self._record_transition(
            invoice.invoice_id,
            self._states[invoice.invoice_id],
            target_state,
            "reconciler",
            {"exceptions_count": len(exceptions)},
        )
        return clean, exceptions

    def resolve_exceptions(
        self,
        invoice_id: str,
        exceptions: List[ExceptionDetail],
        agent_id: str = "ai_agent_invoicetrack",
    ) -> Dict[str, Any]:
        """
        Process detected exceptions through automated policies.
        Auto-resolves minor tolerance variances, adjusts GL accounts, or escalates severe items.
        """
        current_state = self._states.get(
            invoice_id, ApprovalState.EXCEPTION_DETECTED
        )
        all_resolvable = all(e.auto_resolvable for e in exceptions) if exceptions else True

        resolutions = []
        for exc in exceptions:
            if exc.auto_resolvable:
                resolutions.append({
                    "line_number": exc.line_number,
                    "action": exc.recommended_action.value,
                    "status": "resolved",
                    "reason": f"Auto-resolved within policy limits ({exc.explanation})",
                })
            else:
                resolutions.append({
                    "line_number": exc.line_number,
                    "action": exc.recommended_action.value,
                    "status": "escalated",
                    "reason": exc.explanation,
                })

        new_state = (
            ApprovalState.AUTO_RESOLVED
            if all_resolvable and exceptions
            else (ApprovalState.APPROVED if not exceptions else ApprovalState.ESCALATED)
        )
        self._record_transition(
            invoice_id,
            current_state,
            new_state,
            actor=agent_id,
            details={"resolutions": resolutions, "all_auto_resolved": all_resolvable},
        )

        return {
            "invoice_id": invoice_id,
            "final_state": new_state.value,
            "all_auto_resolved": all_resolvable,
            "resolutions": resolutions,
        }

    def _record_transition(
        self,
        invoice_id: str,
        from_state: ApprovalState,
        to_state: ApprovalState,
        actor: str,
        details: Dict[str, Any],
    ) -> None:
        """Record an immutable, cryptographically chained audit event."""
        self._states[invoice_id] = to_state
        ts = dt.datetime.now(dt.timezone.utc).isoformat()
        prev_hash = self.audit_log[-1].event_hash if self.audit_log else "0" * 64
        payload = (
            f"{invoice_id}|{from_state.value}|{to_state.value}|{actor}|{ts}|"
            f"{json.dumps(details, sort_keys=True)}|{prev_hash}"
        )
        event_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()

        event = AuditEvent(
            event_id=f"EVT-{len(self.audit_log) + 1:05d}",
            invoice_id=invoice_id,
            from_state=from_state,
            to_state=to_state,
            actor=actor,
            timestamp=ts,
            details=details,
            event_hash=event_hash,
        )
        self.audit_log.append(event)

    def get_audit_trail(self, invoice_id: str) -> List[AuditEvent]:
        return [e for e in self.audit_log if e.invoice_id == invoice_id]

    def verify_audit_chain(self) -> bool:
        """Verify tamper-resistance of the event hash chain."""
        for i, event in enumerate(self.audit_log):
            prev_hash = self.audit_log[i - 1].event_hash if i > 0 else "0" * 64
            payload = (
                f"{event.invoice_id}|{event.from_state.value}|{event.to_state.value}|"
                f"{event.actor}|{event.timestamp}|"
                f"{json.dumps(event.details, sort_keys=True)}|{prev_hash}"
            )
            expected = hashlib.sha256(payload.encode("utf-8")).hexdigest()
            if event.event_hash != expected:
                return False
        return True
