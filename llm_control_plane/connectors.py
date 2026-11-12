"""Enterprise MCP tool connector registry, schema validation, and human approval gates."""

from __future__ import annotations

import hashlib
import hmac
import json
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .audit import AuditLog
from .evals import EvalCase, Score
from .metrics import MetricsRegistry
from .redaction import Redactor

VALID_RISK_LEVELS = ("read", "write", "privileged")
VALID_ARG_TYPES = ("string", "integer", "number", "boolean", "object", "array")


class ConnectorError(Exception):
    """Base error for enterprise MCP connector registration and execution."""


class ConnectorNotFoundError(ConnectorError):
    """Raised when a requested MCP connector tool is not registered."""


class ConnectorSchemaError(ConnectorError):
    """Raised fail-closed when tool arguments violate the declared contract."""

    def __init__(self, tool_name: str, errors: List[str]) -> None:
        super().__init__(f"schema validation failed for tool {tool_name!r}: {'; '.join(errors)}")
        self.tool_name = tool_name
        self.errors = errors


class ApprovalRequiredError(ConnectorError):
    """Raised when a privileged or approval-gated action lacks valid human sign-off."""

    def __init__(self, tool_name: str, connector: str, reason: str) -> None:
        super().__init__(f"human approval required for {connector}.{tool_name}: {reason}")
        self.tool_name = tool_name
        self.connector = connector
        self.reason = reason


@dataclass(frozen=True)
class ToolArgSpec:
    arg_type: str
    required: bool = True
    enum: Optional[Tuple[Any, ...]] = None
    max_length: Optional[int] = None

    def __post_init__(self) -> None:
        if self.arg_type not in VALID_ARG_TYPES:
            raise ValueError(f"unsupported arg_type: {self.arg_type!r}")


@dataclass(frozen=True)
class MCPToolSpec:
    name: str
    connector: str
    description: str
    args: Dict[str, ToolArgSpec] = field(default_factory=dict)
    risk_level: str = "read"
    requires_human_approval: bool = False
    allow_additional_args: bool = False

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.connector.strip():
            raise ValueError("tool name and connector must be non-empty")
        if self.risk_level not in VALID_RISK_LEVELS:
            raise ValueError(f"unsupported risk_level: {self.risk_level!r}")


@dataclass(frozen=True)
class ToolCallRequest:
    tenant_id: str
    tool_name: str
    arguments: Dict[str, Any]
    trace_id: Optional[str] = None
    dry_run: bool = False
    approver_id: Optional[str] = None
    approval_token: Optional[str] = None


@dataclass(frozen=True)
class ToolCallResult:
    tool_name: str
    connector: str
    trace_id: str
    status: str
    output: Any
    redacted_arguments: Dict[str, Any]
    redactions: Dict[str, int]
    audit_seq: int


def _type_matches(value: Any, arg_type: str) -> bool:
    if arg_type == "string":
        return isinstance(value, str)
    if arg_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if arg_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if arg_type == "boolean":
        return isinstance(value, bool)
    if arg_type == "object":
        return isinstance(value, dict)
    if arg_type == "array":
        return isinstance(value, list)
    return False


class ConnectorRegistry:
    """Thread-safe registry gating enterprise MCP connector tool execution."""

    def __init__(
        self,
        audit: Optional[AuditLog] = None,
        metrics: Optional[MetricsRegistry] = None,
        redactor: Optional[Redactor] = None,
        approval_secret: Optional[str] = None,
        redact_pii: bool = True,
        trace_id_factory: Optional[Callable[[], str]] = None,
    ) -> None:
        self.audit = audit or AuditLog()
        self.metrics = metrics or MetricsRegistry()
        self.redactor = redactor or Redactor()
        self._approval_secret = approval_secret
        self.redact_pii = redact_pii
        self._trace_id_factory = trace_id_factory or (lambda: uuid.uuid4().hex)
        self._specs: Dict[str, MCPToolSpec] = {}
        self._handlers: Dict[str, Callable[[Dict[str, Any]], Any]] = {}
        self._lock = threading.Lock()

    def register(self, spec: MCPToolSpec, handler: Callable[[Dict[str, Any]], Any]) -> None:
        with self._lock:
            self._specs[spec.name] = spec
            self._handlers[spec.name] = handler

    def list_specs(self, connector: Optional[str] = None) -> List[MCPToolSpec]:
        with self._lock:
            specs = list(self._specs.values())
        if connector is not None:
            specs = [s for s in specs if s.connector == connector]
        return sorted(specs, key=lambda s: (s.connector, s.name))

    def issue_approval_token(self, tenant_id: str, tool_name: str, approver_id: str) -> str:
        if not self._approval_secret:
            raise ValueError("approval_secret must be configured before issuing approval tokens")
        if not tenant_id.strip() or not tool_name.strip() or not approver_id.strip():
            raise ValueError("tenant_id, tool_name, and approver_id must be non-empty")
        msg = f"{tenant_id}:{tool_name}:{approver_id}".encode("utf-8")
        return hmac.new(self._approval_secret.encode("utf-8"), msg, hashlib.sha256).hexdigest()

    def _verify_approval(self, request: ToolCallRequest) -> Tuple[bool, str]:
        if not request.approver_id or not request.approver_id.strip():
            return False, "missing approver_id"
        if not request.approval_token:
            return False, "missing approval_token"
        if not self._approval_secret:
            return False, "approval verifier is not configured (fail-closed)"
        expected = self.issue_approval_token(request.tenant_id, request.tool_name, request.approver_id)
        if not hmac.compare_digest(expected, request.approval_token):
            return False, "invalid approval_token"
        return True, ""

    def validate_call(self, tool_name: str, arguments: Dict[str, Any]) -> MCPToolSpec:
        with self._lock:
            spec = self._specs.get(tool_name)
        if spec is None:
            raise ConnectorNotFoundError(f"unknown tool {tool_name!r}")
        if not isinstance(arguments, dict):
            raise ConnectorSchemaError(tool_name, ["arguments must be a mapping"])

        errors: List[str] = []
        if not spec.allow_additional_args:
            unknown = sorted(set(arguments) - set(spec.args))
            if unknown:
                errors.append(f"unexpected arguments: {', '.join(unknown)}")

        for key, arg_spec in spec.args.items():
            if key not in arguments or arguments[key] is None:
                if arg_spec.required:
                    errors.append(f"missing required argument {key!r}")
                continue
            val = arguments[key]
            if not _type_matches(val, arg_spec.arg_type):
                errors.append(f"argument {key!r} must be {arg_spec.arg_type}")
                continue
            if arg_spec.enum is not None and val not in arg_spec.enum:
                errors.append(f"argument {key!r} must be one of {list(arg_spec.enum)!r}")
            if arg_spec.max_length is not None and isinstance(val, str) and len(val) > arg_spec.max_length:
                errors.append(f"argument {key!r} exceeds max_length {arg_spec.max_length}")

        if errors:
            raise ConnectorSchemaError(tool_name, errors)
        return spec

    def _redact_value(self, value: Any, totals: Dict[str, int]) -> Any:
        if isinstance(value, str):
            res = self.redactor.redact(value)
            for k, v in res.counts.items():
                totals[k] = totals.get(k, 0) + v
            return res.text
        if isinstance(value, list):
            return [self._redact_value(item, totals) for item in value]
        if isinstance(value, dict):
            return {k: self._redact_value(v, totals) for k, v in value.items()}
        return value

    def execute(self, request: ToolCallRequest) -> ToolCallResult:
        trace_id = request.trace_id or self._trace_id_factory()
        with self._lock:
            spec = self._specs.get(request.tool_name)
            handler = self._handlers.get(request.tool_name)
        if spec is None or handler is None:
            raise ConnectorNotFoundError(f"unknown tool {request.tool_name!r}")
        if not request.tenant_id.strip():
            raise ConnectorSchemaError(request.tool_name, ["tenant_id must be non-empty"])

        try:
            self.validate_call(request.tool_name, request.arguments)
        except ConnectorSchemaError as exc:
            self.metrics.inc(
                "connector_schema_failures_total",
                {"connector": spec.connector, "tool": spec.name},
            )
            self.audit.append(
                "connector.schema_rejected",
                {
                    "trace_id": trace_id,
                    "tenant_id": request.tenant_id,
                    "connector": spec.connector,
                    "tool": spec.name,
                    "errors": exc.errors,
                },
            )
            raise

        redactions: Dict[str, int] = {}
        safe_args = (
            self._redact_value(request.arguments, redactions)
            if self.redact_pii
            else dict(request.arguments)
        )

        needs_approval = spec.requires_human_approval or spec.risk_level == "privileged"
        if needs_approval and not request.dry_run:
            ok, reason = self._verify_approval(request)
            if not ok:
                self.metrics.inc(
                    "connector_escalations_total",
                    {"connector": spec.connector, "tool": spec.name, "status": "denied"},
                )
                self.audit.append(
                    "connector.approval_required",
                    {
                        "trace_id": trace_id,
                        "tenant_id": request.tenant_id,
                        "connector": spec.connector,
                        "tool": spec.name,
                        "reason": reason,
                    },
                )
                raise ApprovalRequiredError(spec.name, spec.connector, reason)

        if request.dry_run:
            sim_output = {
                "simulated": True,
                "connector": spec.connector,
                "tool": spec.name,
                "arguments": safe_args,
            }
            ev = self.audit.append(
                "connector.dry_run",
                {
                    "trace_id": trace_id,
                    "tenant_id": request.tenant_id,
                    "connector": spec.connector,
                    "tool": spec.name,
                    "redactions": redactions,
                },
            )
            self.metrics.inc(
                "connector_calls_total",
                {"connector": spec.connector, "tool": spec.name, "mode": "dry_run"},
            )
            return ToolCallResult(
                tool_name=spec.name,
                connector=spec.connector,
                trace_id=trace_id,
                status="dry_run",
                output=sim_output,
                redacted_arguments=safe_args,
                redactions=redactions,
                audit_seq=ev.seq,
            )

        output = handler(safe_args)
        ev = self.audit.append(
            "connector.executed",
            {
                "trace_id": trace_id,
                "tenant_id": request.tenant_id,
                "connector": spec.connector,
                "tool": spec.name,
                "approver_id": request.approver_id,
                "redactions": redactions,
            },
        )
        self.metrics.inc(
            "connector_calls_total",
            {"connector": spec.connector, "tool": spec.name, "mode": "live"},
        )
        return ToolCallResult(
            tool_name=spec.name,
            connector=spec.connector,
            trace_id=trace_id,
            status="executed",
            output=output,
            redacted_arguments=safe_args,
            redactions=redactions,
            audit_seq=ev.seq,
        )


class ToolTrajectoryScorer:
    """Eval scorer validating ordered agent MCP tool-call trajectories and argument keys."""

    name = "tool_trajectory"

    def score(self, case: EvalCase, output: str, latency_ms: float) -> Score:
        try:
            trajectory = json.loads(output)
        except json.JSONDecodeError as exc:
            return Score(self.name, False, 0.0, f"invalid trajectory json: {exc.msg}")

        if not isinstance(trajectory, list):
            return Score(self.name, False, 0.0, "trajectory must be a JSON list of tool calls")

        expected = case.expected if isinstance(case.expected, dict) else {}
        expected_steps: List[Dict[str, Any]] = list(expected.get("steps") or [])
        forbidden_tools = set(expected.get("forbidden_tools") or [])

        called_names = [step.get("tool") for step in trajectory if isinstance(step, dict)]
        blocked = [t for t in called_names if t in forbidden_tools]
        if blocked:
            return Score(self.name, False, 0.0, f"forbidden tool(s) invoked: {blocked}")

        if len(trajectory) != len(expected_steps):
            return Score(
                self.name,
                False,
                0.0,
                f"expected {len(expected_steps)} tool steps, got {len(trajectory)}",
            )

        issues: List[str] = []
        matched = 0
        for idx, (actual, exp) in enumerate(zip(trajectory, expected_steps)):
            if not isinstance(actual, dict):
                issues.append(f"step {idx}: call entry must be an object")
                continue
            if actual.get("tool") != exp.get("tool"):
                issues.append(f"step {idx}: expected tool {exp.get('tool')!r}, got {actual.get('tool')!r}")
                continue
            args = actual.get("arguments") or {}
            req_keys = exp.get("required_keys") or []
            missing = [k for k in req_keys if k not in args]
            if missing:
                issues.append(f"step {idx}: missing argument keys {missing}")
                continue
            matched += 1

        value = round(matched / len(expected_steps), 4) if expected_steps else 1.0
        return Score(self.name, not issues, value, "; ".join(issues))
