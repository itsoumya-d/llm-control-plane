"""Unit tests for Bitpanda Elastic SecOps Platform."""

from llm_control_plane.bitpanda_elastic_secops_platform import (
    CloudThreatHunter,
    DORAClassification,
    ElasticDetectionEngine,
    ElasticDetectionRule,
    IncidentStatus,
    QueryLanguage,
    SecurityPostureHardeningAuditor,
    SeverityLevel,
    SOARIncidentWorkflowManager,
)


def test_rule_validation():
    # Valid ES|QL rule
    valid_esql = ElasticDetectionRule(
        rule_id="BP-SEC-001",
        name="AWS Root Session Without MFA",
        query=(
            "FROM aws-cloudtrail-* | WHERE event.name == 'ConsoleLogin' "
            "and aws.cloudtrail.user_identity.type == 'Root'"
        ),
        language=QueryLanguage.ESQL,
        severity=SeverityLevel.CRITICAL,
        mitre_attack_ids=["T1078.004"],
        description="Detects AWS root console logins without MFA.",
    )
    ok, err = valid_esql.validate()
    assert ok is True
    assert err is None

    # Invalid ES|QL rule missing pipe
    invalid_esql = ElasticDetectionRule(
        rule_id="BP-SEC-002",
        name="Invalid Rule",
        query="SELECT * FROM table",
        language=QueryLanguage.ESQL,
        severity=SeverityLevel.LOW,
        mitre_attack_ids=["T1078"],
        description="Missing pipe",
    )
    ok, err = invalid_esql.validate()
    assert ok is False
    assert "must follow piped syntax" in err

    # Invalid rule missing MITRE techniques
    missing_mitre = ElasticDetectionRule(
        rule_id="BP-SEC-003",
        name="No MITRE",
        query="FROM logs-* | WHERE status == 500",
        language=QueryLanguage.ESQL,
        severity=SeverityLevel.LOW,
        mitre_attack_ids=[],
        description="No techniques",
    )
    ok, err = missing_mitre.validate()
    assert ok is False
    assert "must map to at least one MITRE" in err


def test_detection_engine_event_evaluation_and_suppression():
    engine = ElasticDetectionEngine()

    rule = ElasticDetectionRule(
        rule_id="BP-AUTH-FAIL",
        name="Multiple Failed Authentication Attempts",
        query="event.category: authentication and event.outcome: failure",
        language=QueryLanguage.KQL,
        severity=SeverityLevel.HIGH,
        mitre_attack_ids=["T1110"],
        description="Detects failed authentication attempts",
        false_positive_filters=[r"automated_health_check_bot"],
    )
    engine.register_rule(rule)

    # Normal matched event
    normal_fail_event = {
        "id": "ev-01",
        "event": {"category": "authentication", "outcome": "failure"},
        "user": {"name": "trader_alice"},
    }
    alerts = engine.evaluate_event(normal_fail_event)
    assert len(alerts) == 1
    assert alerts[0].suppressed is False
    assert alerts[0].severity == SeverityLevel.HIGH
    assert len(engine.emitted_alerts) == 1

    # False positive event matching health check filter
    fp_event = {
        "id": "ev-02",
        "event": {"category": "authentication", "outcome": "failure"},
        "user": {"name": "automated_health_check_bot"},
    }
    fp_alerts = engine.evaluate_event(fp_event)
    assert len(fp_alerts) == 1
    assert fp_alerts[0].suppressed is True
    assert "Matched false-positive filter" in fp_alerts[0].suppression_reason
    assert engine.suppressed_alerts_count == 1


def test_cloud_threat_hunter_aws_iam_anomalies():
    engine = ElasticDetectionEngine()
    hunter = CloudThreatHunter(engine)

    trail_events = [
        {
            "eventName": "ConsoleLogin",
            "userIdentity": {"type": "Root"},
            "additionalEventData": {"MFAUsed": "No"},
        },
        {
            "eventName": "AttachUserPolicy",
            "userIdentity": {"type": "IAMUser", "arn": "arn:aws:iam::123456789012:user/dev"},
            "requestParameters": {
                "userName": "contractor_bob",
                "policyArn": "arn:aws:iam::aws:policy/AdministratorAccess",
            },
        },
    ]

    findings = hunter.hunt_aws_iam_anomalies(trail_events)
    assert len(findings) == 2
    assert findings[0]["technique"] == "T1078.004"
    assert findings[0]["severity"] == SeverityLevel.CRITICAL
    assert findings[1]["technique"] == "T1098"
    assert findings[1]["severity"] == SeverityLevel.HIGH


def test_cloud_threat_hunter_kms_decrypt_spike():
    engine = ElasticDetectionEngine()
    hunter = CloudThreatHunter(engine)

    events = [
        {
            "eventName": "Decrypt",
            "userIdentity": {"arn": "arn:aws:iam::123456789012:role/hot-wallet-signer"},
        }
        for _ in range(6)
    ]
    findings = hunter.hunt_kms_decrypt_spike(events, threshold=5)
    assert len(findings) == 1
    assert findings[0]["technique"] == "T1552.004"
    assert findings[0]["decrypt_count"] == 6
    assert findings[0]["severity"] == SeverityLevel.CRITICAL


def test_soar_incident_workflow_and_dora_compliance():
    soar = SOARIncidentWorkflowManager()

    engine = ElasticDetectionEngine()
    rule = ElasticDetectionRule(
        rule_id="BP-KMS-001",
        name="Crypto Vault Key Exfiltration Suspected",
        query="event.name: Decrypt",
        language=QueryLanguage.KQL,
        severity=SeverityLevel.CRITICAL,
        mitre_attack_ids=["T1552.004"],
        description="Critical custody event",
    )
    engine.register_rule(rule)

    alerts = engine.evaluate_event({"id": "ev-99", "event": {"name": "Decrypt"}})
    alert = alerts[0]

    incident = soar.create_incident_from_alert(alert)
    assert incident.dora_classification == DORAClassification.MAJOR
    assert incident.status == IncidentStatus.DETECTED

    # Enrich IOC
    enrichment = soar.enrich_incident_iocs(incident.incident_id, ip_address="185.220.101.5")
    assert enrichment["tor_exit_node"] is True
    assert enrichment["threat_score"] == 95
    assert incident.status == IncidentStatus.ENRICHED

    # Contain incident
    soar.execute_containment(
        incident.incident_id,
        actions=[
            "Revoke active IAM STS sessions for hot-wallet-signer",
            "Enforce security group ingress restriction to 0.0.0.0/0",
            "Rotate primary HSM master encryption key",
        ],
    )
    assert incident.status == IncidentStatus.CONTAINED
    assert len(incident.containment_actions) == 3
    assert incident.is_dora_sla_compliant() is True


def test_security_posture_hardening_auditor():
    auditor = SecurityPostureHardeningAuditor()

    infra_resources = {
        "s3_buckets": [
            {"name": "bitpanda-tx-receipts", "public_access_block": True},
            {"name": "bitpanda-public-assets", "public_access_block": True},
        ],
        "kms_keys": [
            {"id": "key-master-custody", "rotation_enabled": True},
            {"id": "key-analytics-temp", "rotation_enabled": False},
        ],
        "iam_users": [
            {"name": "secops_admin", "mfa_enabled": True},
            {"name": "build_deployer", "mfa_enabled": True},
        ],
    }

    report = auditor.evaluate_aws_posture(infra_resources)
    assert report["checks_run"] == 6
    assert report["checks_passed"] == 5
    assert report["posture_score_pct"] == 83.3
    assert report["compliant"] is False
    assert len(report["findings"]) == 1
    assert "DORA" in report["findings"][0]["control"]
