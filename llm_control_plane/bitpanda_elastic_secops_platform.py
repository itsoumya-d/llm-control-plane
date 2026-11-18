"""Bitpanda Elastic Security Operations & Cloud SecOps Platform.

Provides enterprise detection engineering with ES|QL and KQL query parsers,
multi-source AWS CloudTrail and SIEM threat hunting, automated SOAR incident
containment workflows, and DORA (Digital Operational Resilience Act) and
ISO 27001 regulatory compliance posture tracking.
"""

from __future__ import annotations

import datetime
import enum
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


class QueryLanguage(str, enum.Enum):
    ESQL = "ES|QL"
    KQL = "KQL"


class SeverityLevel(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IncidentStatus(str, enum.Enum):
    DETECTED = "detected"
    ENRICHED = "enriched"
    CONTAINED = "contained"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"
    CLOSED = "closed"


class DORAClassification(str, enum.Enum):
    NON_MAJOR = "non_major"
    MAJOR = "major"


@dataclass
class ElasticDetectionRule:
    """Detection rule definition supporting both ES|QL and KQL."""

    rule_id: str
    name: str
    query: str
    language: QueryLanguage
    severity: SeverityLevel
    mitre_attack_ids: List[str]
    description: str
    false_positive_filters: List[str] = field(default_factory=list)
    enabled: bool = True
    fidelity_score: float = 1.0  # 0.0 to 1.0
    created_at: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())

    def validate(self) -> Tuple[bool, Optional[str]]:
        """Validate query syntax against language conventions."""
        if not self.rule_id or not self.name or not self.query:
            return False, "Missing required rule identification or query string"

        if self.language == QueryLanguage.ESQL:
            # ES|QL uses piped structure: FROM index | WHERE condition
            if "|" not in self.query and not self.query.strip().upper().startswith("FROM"):
                return False, "ES|QL queries must follow piped syntax starting with FROM"
        elif self.language == QueryLanguage.KQL:
            # KQL syntax check
            if not any(op in self.query for op in (":", "and", "or", "not", "AND", "OR", "NOT")):
                return False, "KQL query must contain field:value or boolean predicates"

        if not self.mitre_attack_ids:
            return False, "Rule must map to at least one MITRE ATT&CK technique"

        return True, None


@dataclass
class SecurityAlert:
    """Security alert emitted by detection engine."""

    alert_id: str
    rule_id: str
    rule_name: str
    severity: SeverityLevel
    timestamp: str
    raw_event: Dict[str, Any]
    matched_conditions: List[str]
    mitre_attack_ids: List[str]
    suppressed: bool = False
    suppression_reason: Optional[str] = None


class ElasticDetectionEngine:
    """Detection engine managing rule execution, noise reduction, and alert emission."""

    def __init__(self) -> None:
        self.rules: Dict[str, ElasticDetectionRule] = {}
        self.emitted_alerts: List[SecurityAlert] = []
        self.suppressed_alerts_count: int = 0

    def register_rule(self, rule: ElasticDetectionRule) -> None:
        valid, err = rule.validate()
        if not valid:
            raise ValueError(f"Invalid detection rule '{rule.rule_id}': {err}")
        self.rules[rule.rule_id] = rule

    def evaluate_event(self, event: Dict[str, Any]) -> List[SecurityAlert]:
        """Evaluate a single security log event against all enabled rules."""
        alerts: List[SecurityAlert] = []
        for rule in self.rules.values():
            if not rule.enabled:
                continue

            matches, matched_fields = self._matches_rule(rule, event)
            if matches:
                # Check false positive filters
                is_fp, fp_reason = self._is_false_positive(rule, event)
                now_str = datetime.datetime.now(datetime.timezone.utc).isoformat()
                raw_seed = f"{rule.rule_id}:{event.get('id', '')}:{now_str}".encode()
                alert_id = hashlib.sha256(raw_seed).hexdigest()[:16]

                alert = SecurityAlert(
                    alert_id=f"alt-{alert_id}",
                    rule_id=rule.rule_id,
                    rule_name=rule.name,
                    severity=rule.severity,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    raw_event=event,
                    matched_conditions=matched_fields,
                    mitre_attack_ids=rule.mitre_attack_ids,
                    suppressed=is_fp,
                    suppression_reason=fp_reason,
                )

                if is_fp:
                    self.suppressed_alerts_count += 1
                else:
                    self.emitted_alerts.append(alert)
                alerts.append(alert)

        return alerts

    def _matches_rule(self, rule: ElasticDetectionRule, event: Dict[str, Any]) -> Tuple[bool, List[str]]:
        matched_fields: List[str] = []
        # Parse basic KQL or ES|QL predicates for simulation
        # E.g. event.category: authentication and event.outcome: failure
        query_text = rule.query.lower()

        # Extract field:value pairs
        pairs = re.findall(r"([\w\.\-]+)\s*[:=]\s*[\"']?([\w\.\-]+)[\"']?", query_text)
        if not pairs:
            # Fallback string check
            for k, v in event.items():
                if str(v).lower() in query_text:
                    matched_fields.append(f"{k}={v}")
            return (len(matched_fields) > 0, matched_fields)

        all_matched = True
        for field_name, expected_val in pairs:
            # Resolve dotted paths
            actual_val = self._resolve_dotted_key(event, field_name)
            if actual_val is None or str(actual_val).lower() != expected_val.lower():
                all_matched = False
                break
            matched_fields.append(f"{field_name}={actual_val}")

        return all_matched, matched_fields

    def _resolve_dotted_key(self, data: Dict[str, Any], path: str) -> Any:
        current = data
        for part in path.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return current

    def _is_false_positive(self, rule: ElasticDetectionRule, event: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        event_str = json.dumps(event)
        for fp_filter in rule.false_positive_filters:
            if re.search(fp_filter, event_str, re.IGNORECASE):
                return True, f"Matched false-positive filter pattern: {fp_filter}"
        return False, None


class CloudThreatHunter:
    """Proactive threat hunting across AWS CloudTrail and Elastic SIEM."""

    def __init__(self, engine: ElasticDetectionEngine) -> None:
        self.engine = engine

    def hunt_aws_iam_anomalies(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify suspicious AWS IAM privilege escalation and root usage without MFA."""
        findings: List[Dict[str, Any]] = []
        for ev in events:
            user_identity = ev.get("userIdentity", {})
            event_name = ev.get("eventName", "")

            # Check root account usage
            if user_identity.get("type") == "Root":
                mfa_used = ev.get("additionalEventData", {}).get("MFAUsed") == "Yes"
                if not mfa_used:
                    findings.append(
                        {
                            "hunt": "AWS Root Account Activity Without MFA",
                            "severity": SeverityLevel.CRITICAL,
                            "technique": "T1078.004",
                            "event": ev,
                            "details": "Root account session executed API calls without multi-factor authentication.",
                        }
                    )

            # Check suspicious policy attachment
            if event_name in ("AttachUserPolicy", "AttachRolePolicy", "PutUserPolicy"):
                policy_arn = ev.get("requestParameters", {}).get("policyArn", "")
                if "AdministratorAccess" in policy_arn:
                    findings.append(
                        {
                            "hunt": "Unauthorized AdministratorAccess Grant",
                            "severity": SeverityLevel.HIGH,
                            "technique": "T1098",
                            "event": ev,
                            "details": (
                                f"AdministratorAccess attached to {ev.get('requestParameters', {}).get('userName')}"
                            ),
                        }
                    )

        return findings

    def hunt_kms_decrypt_spike(self, events: List[Dict[str, Any]], threshold: int = 5) -> List[Dict[str, Any]]:
        """Detect rapid bulk decryption calls indicating potential crypto wallet key extraction."""
        decrypt_counts_by_user: Dict[str, int] = {}
        findings: List[Dict[str, Any]] = []

        for ev in events:
            if ev.get("eventName") == "Decrypt":
                user = ev.get("userIdentity", {}).get("arn", "unknown")
                decrypt_counts_by_user[user] = decrypt_counts_by_user.get(user, 0) + 1

        for user, count in decrypt_counts_by_user.items():
            if count >= threshold:
                findings.append(
                    {
                        "hunt": "KMS Bulk Decryption Anomalous Spike",
                        "severity": SeverityLevel.CRITICAL,
                        "technique": "T1552.004",
                        "user_arn": user,
                        "decrypt_count": count,
                        "threshold": threshold,
                        "details": f"User {user} invoked KMS:Decrypt {count} times within observation window.",
                    }
                )

        return findings


@dataclass
class IncidentRecord:
    """Incident response record tracked by SOAR manager."""

    incident_id: str
    title: str
    severity: SeverityLevel
    status: IncidentStatus
    dora_classification: DORAClassification
    detected_at: str
    contained_at: Optional[str] = None
    containment_actions: List[str] = field(default_factory=list)
    ioc_enrichments: Dict[str, Any] = field(default_factory=dict)
    audit_trail: List[str] = field(default_factory=list)

    def is_dora_sla_compliant(self) -> bool:
        """DORA requires initial notification within 4 hours for major ICT incidents."""
        if self.dora_classification != DORAClassification.MAJOR:
            return True
        if not self.contained_at:
            return True  # in progress
        t0 = datetime.datetime.fromisoformat(self.detected_at)
        t1 = datetime.datetime.fromisoformat(self.contained_at)
        delta_hours = (t1 - t0).total_seconds() / 3600.0
        return delta_hours <= 4.0


class SOARIncidentWorkflowManager:
    """Orchestrates automated containment, enrichment, and DORA incident compliance."""

    def __init__(self) -> None:
        self.incidents: Dict[str, IncidentRecord] = {}

    def create_incident_from_alert(self, alert: SecurityAlert) -> IncidentRecord:
        inc_id = f"inc-{hashlib.sha256(alert.alert_id.encode()).hexdigest()[:12]}"

        # Major classification if critical severity or involves custody/KMS techniques
        is_major = alert.severity == SeverityLevel.CRITICAL or any(
            t in ("T1078.004", "T1552.004", "T1098") for t in alert.mitre_attack_ids
        )

        dora_cls = DORAClassification.MAJOR if is_major else DORAClassification.NON_MAJOR

        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        incident = IncidentRecord(
            incident_id=inc_id,
            title=f"Security Incident: {alert.rule_name}",
            severity=alert.severity,
            status=IncidentStatus.DETECTED,
            dora_classification=dora_cls,
            detected_at=now_iso,
            audit_trail=[f"[{now_iso}] Incident created from alert {alert.alert_id}"],
        )

        self.incidents[inc_id] = incident
        return incident

    def enrich_incident_iocs(self, incident_id: str, ip_address: Optional[str] = None) -> Dict[str, Any]:
        incident = self.incidents[incident_id]
        enrichment: Dict[str, Any] = {}

        if ip_address:
            # Deterministic reputation lookup
            is_tor = ip_address.startswith("185.") or ip_address.startswith("198.")
            enrichment["ip"] = ip_address
            enrichment["tor_exit_node"] = is_tor
            enrichment["asn"] = "AS13335 Cloudflare" if not is_tor else "AS9009 Tor Gateway"
            enrichment["threat_score"] = 95 if is_tor else 15

        incident.ioc_enrichments.update(enrichment)
        incident.status = IncidentStatus.ENRICHED
        now_ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        incident.audit_trail.append(f"[{now_ts}] IOC enrichment completed")
        return enrichment

    def execute_containment(self, incident_id: str, actions: List[str]) -> bool:
        incident = self.incidents[incident_id]
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

        for act in actions:
            incident.containment_actions.append(act)
            incident.audit_trail.append(f"[{now_iso}] Containment action executed: {act}")

        incident.contained_at = now_iso
        incident.status = IncidentStatus.CONTAINED
        return True


class SecurityPostureHardeningAuditor:
    """Evaluates AWS cloud configurations and ISO 27001 / DORA control adherence."""

    def evaluate_aws_posture(self, resources: Dict[str, Any]) -> Dict[str, Any]:
        findings: List[Dict[str, Any]] = []
        checks_run = 0
        checks_passed = 0

        # Check 1: S3 bucket public access block
        for bucket in resources.get("s3_buckets", []):
            checks_run += 1
            if bucket.get("public_access_block") is True:
                checks_passed += 1
            else:
                findings.append(
                    {
                        "resource": bucket.get("name"),
                        "control": "ISO 27001 A.8.24",
                        "issue": "S3 bucket public access is not explicitly blocked.",
                        "severity": SeverityLevel.HIGH,
                    }
                )

        # Check 2: KMS Key Rotation Enabled
        for key in resources.get("kms_keys", []):
            checks_run += 1
            if key.get("rotation_enabled") is True:
                checks_passed += 1
            else:
                findings.append(
                    {
                        "resource": key.get("id"),
                        "control": "DORA ICT Security (KMS Rotation)",
                        "issue": "KMS key automatic annual rotation is disabled.",
                        "severity": SeverityLevel.MEDIUM,
                    }
                )

        # Check 3: IAM MFA enforcement
        for user in resources.get("iam_users", []):
            checks_run += 1
            if user.get("mfa_enabled") is True:
                checks_passed += 1
            else:
                findings.append(
                    {
                        "resource": user.get("name"),
                        "control": "ISO 27001 A.9.4.2",
                        "issue": "IAM user has console access without MFA enforcement.",
                        "severity": SeverityLevel.HIGH,
                    }
                )

        score_pct = round((checks_passed / checks_run * 100.0) if checks_run > 0 else 100.0, 1)

        return {
            "checks_run": checks_run,
            "checks_passed": checks_passed,
            "posture_score_pct": score_pct,
            "compliant": score_pct >= 90.0,
            "findings": findings,
        }
