import pytest

from llm_control_plane.audit import AuditLog
from llm_control_plane.connectors import (
    ApprovalRequiredError,
    ConnectorNotFoundError,
    ConnectorRegistry,
    ConnectorSchemaError,
    MCPToolSpec,
    ToolArgSpec,
    ToolCallRequest,
    ToolTrajectoryScorer,
)
from llm_control_plane.evals import EvalCase


def _make_registry() -> ConnectorRegistry:
    registry = ConnectorRegistry(
        audit=AuditLog(),
        approval_secret="ema-test-signing-key",
    )
    registry.register(
        MCPToolSpec(
            name="servicenow.create_ticket",
            connector="servicenow",
            description="Create an IT support incident in ServiceNow.",
            risk_level="write",
            args={
                "requester_email": ToolArgSpec(arg_type="string", required=True, max_length=120),
                "summary": ToolArgSpec(arg_type="string", required=True, max_length=200),
                "priority": ToolArgSpec(arg_type="integer", required=False, enum=(1, 2, 3, 4)),
            },
        ),
        handler=lambda args: {"ticket_id": "INC0042", "received": args},
    )
    registry.register(
        MCPToolSpec(
            name="workday.approve_payroll_run",
            connector="workday",
            description="Approve a payroll settlement batch in Workday.",
            risk_level="privileged",
            requires_human_approval=True,
            args={
                "batch_id": ToolArgSpec(arg_type="string", required=True),
                "confirm": ToolArgSpec(arg_type="boolean", required=True),
            },
        ),
        handler=lambda args: {"batch_id": args["batch_id"], "status": "approved"},
    )
    return registry


def test_list_specs_and_unknown_tool():
    reg = _make_registry()
    assert [s.name for s in reg.list_specs()] == [
        "servicenow.create_ticket",
        "workday.approve_payroll_run",
    ]
    assert [s.name for s in reg.list_specs(connector="workday")] == ["workday.approve_payroll_run"]
    with pytest.raises(ConnectorNotFoundError):
        reg.validate_call("sap.unknown", {})
    with pytest.raises(ConnectorNotFoundError):
        reg.execute(ToolCallRequest(tenant_id="acme", tool_name="sap.unknown", arguments={}))


def test_schema_validation_fail_closed():
    reg = _make_registry()
    with pytest.raises(ConnectorSchemaError) as exc:
        reg.execute(
            ToolCallRequest(
                tenant_id="acme",
                tool_name="servicenow.create_ticket",
                arguments={"summary": "Laptop locked", "priority": 9, "extra_field": "bad"},
            )
        )
    assert any("missing required argument 'requester_email'" in e for e in exc.value.errors)
    assert any("unexpected arguments: extra_field" in e for e in exc.value.errors)
    assert any("must be one of [1, 2, 3, 4]" in e for e in exc.value.errors)
    assert reg.audit.events()[-1].event_type == "connector.schema_rejected"
    assert reg.audit.verify() is True


def test_pii_redaction_and_live_execution():
    reg = _make_registry()
    res = reg.execute(
        ToolCallRequest(
            tenant_id="acme",
            tool_name="servicenow.create_ticket",
            arguments={
                "requester_email": "alice.smith@enterprise.co.uk",
                "summary": "Reset MFA for +44 7700 900123",
                "priority": 2,
            },
            trace_id="trace-snow-1",
        )
    )
    assert res.status == "executed"
    assert res.redacted_arguments["requester_email"] == "[REDACTED_EMAIL]"
    assert "[REDACTED_PHONE]" in res.redacted_arguments["summary"]
    assert res.redactions == {"email": 1, "phone": 1}
    assert res.output["ticket_id"] == "INC0042"
    assert reg.audit.find("trace-snow-1")[0].event_type == "connector.executed"
    assert reg.audit.verify() is True


def test_privileged_human_approval_gate_and_dry_run():
    reg = _make_registry()
    dry = reg.execute(
        ToolCallRequest(
            tenant_id="acme",
            tool_name="workday.approve_payroll_run",
            arguments={"batch_id": "PAY-2026-11", "confirm": True},
            dry_run=True,
        )
    )
    assert dry.status == "dry_run"
    assert dry.output["simulated"] is True

    with pytest.raises(ApprovalRequiredError):
        reg.execute(
            ToolCallRequest(
                tenant_id="acme",
                tool_name="workday.approve_payroll_run",
                arguments={"batch_id": "PAY-2026-11", "confirm": True},
            )
        )

    token = reg.issue_approval_token("acme", "workday.approve_payroll_run", "director-uk-01")
    with pytest.raises(ApprovalRequiredError):
        reg.execute(
            ToolCallRequest(
                tenant_id="acme",
                tool_name="workday.approve_payroll_run",
                arguments={"batch_id": "PAY-2026-11", "confirm": True},
                approver_id="director-uk-01",
                approval_token="tampered-token",
            )
        )

    approved = reg.execute(
        ToolCallRequest(
            tenant_id="acme",
            tool_name="workday.approve_payroll_run",
            arguments={"batch_id": "PAY-2026-11", "confirm": True},
            approver_id="director-uk-01",
            approval_token=token,
        )
    )
    assert approved.status == "executed"
    assert approved.output == {"batch_id": "PAY-2026-11", "status": "approved"}
    assert reg.audit.verify() is True


def test_tool_trajectory_scorer():
    scorer = ToolTrajectoryScorer()
    case = EvalCase(
        id="hr-onboarding-1",
        input="Onboard employee and provision laptop ticket",
        expected={
            "forbidden_tools": ["workday.approve_payroll_run"],
            "steps": [
                {"tool": "servicenow.create_ticket", "required_keys": ["requester_email", "summary"]},
            ],
        },
    )
    ok_out = '[{"tool": "servicenow.create_ticket", "arguments": {"requester_email": "u@x.uk", "summary": "New hire"}}]'
    assert scorer.score(case, ok_out, 10.0).passed is True

    bad_out = '[{"tool": "workday.approve_payroll_run", "arguments": {"batch_id": "1"}}]'
    bad_score = scorer.score(case, bad_out, 10.0)
    assert bad_score.passed is False
    assert "forbidden" in bad_score.detail

    assert scorer.score(case, "not-json", 1.0).passed is False
    assert scorer.score(case, "{}", 1.0).passed is False
    assert scorer.score(case, "[]", 1.0).passed is False
    wrong_tool = '[{"tool": "slack.post", "arguments": {}}]'
    assert scorer.score(case, wrong_tool, 1.0).passed is False
    missing_key = '[{"tool": "servicenow.create_ticket", "arguments": {"summary": "hi"}}]'
    assert scorer.score(case, missing_key, 1.0).passed is False


def test_edge_cases_types_nested_redaction_and_fail_closed_secret():
    with pytest.raises(ValueError):
        ToolArgSpec(arg_type="binary")
    with pytest.raises(ValueError):
        MCPToolSpec(name="", connector="sap", description="bad")
    with pytest.raises(ValueError):
        MCPToolSpec(name="sap.sync", connector="sap", description="bad", risk_level="admin")

    no_secret_reg = ConnectorRegistry()
    with pytest.raises(ValueError):
        no_secret_reg.issue_approval_token("t1", "sap.post", "approver-1")

    spec = MCPToolSpec(
        name="sap.sync_payload",
        connector="sap",
        description="Sync typed record payload",
        risk_level="write",
        requires_human_approval=True,
        args={
            "amount": ToolArgSpec(arg_type="number", required=True),
            "meta": ToolArgSpec(arg_type="object", required=True),
            "notes": ToolArgSpec(arg_type="array", required=True),
            "tag": ToolArgSpec(arg_type="string", required=False, max_length=5),
        },
    )
    no_secret_reg.register(spec, handler=lambda args: args)

    with pytest.raises(ConnectorSchemaError):
        no_secret_reg.execute(
            ToolCallRequest(
                tenant_id="",
                tool_name="sap.sync_payload",
                arguments={"amount": 10, "meta": {}, "notes": []},
            )
        )

    with pytest.raises(ConnectorSchemaError):
        no_secret_reg.execute(
            ToolCallRequest(
                tenant_id="acme",
                tool_name="sap.sync_payload",
                arguments={"amount": True, "meta": "bad", "notes": [], "tag": "too-long-tag"},
            )
        )

    with pytest.raises(ApprovalRequiredError):
        no_secret_reg.execute(
            ToolCallRequest(
                tenant_id="acme",
                tool_name="sap.sync_payload",
                arguments={
                    "amount": 12.5,
                    "meta": {"contact": "ops@acme.co.uk"},
                    "notes": ["ping user@acme.co.uk"],
                },
                approver_id="director",
                approval_token="anything",
            )
        )
    dry = no_secret_reg.execute(
        ToolCallRequest(
            tenant_id="acme",
            tool_name="sap.sync_payload",
            arguments={
                "amount": 12.5,
                "meta": {"contact": "ops@acme.co.uk"},
                "notes": ["ping user@acme.co.uk"],
            },
            dry_run=True,
        )
    )
    assert dry.redactions == {"email": 2}
    assert dry.output["arguments"]["meta"]["contact"] == "[REDACTED_EMAIL]"

