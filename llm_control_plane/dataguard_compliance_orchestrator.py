"""DataGuard Continuous Compliance & Evidence Verification Control Plane.

Architected for European Information Security (InfoSec), Privacy (GDPR), and
Regulatory Compliance automation (ISO 27001:2022, SOC 2 Type II, NIS2, DORA).
Provides multi-tenant cryptographic evidence chaining, deterministic policy control
evaluations, real-time posture scoring, and agentic AI audit review with untrusted
output quarantine.
"""

from __future__ import annotations

import datetime as dt
import enum
import hashlib
import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ComplianceFramework(str, enum.Enum):
    GDPR = "GDPR"
    ISO_27001 = "ISO_27001_2022"
    SOC2_TYPE2 = "SOC2_TYPE2"
    NIS2 = "NIS2"
    DORA = "DORA"


class ControlStatus(str, enum.Enum):
    COMPLIANT = "COMPLIANT"
    NON_COMPLIANT = "NON_COMPLIANT"
    PARTIALLY_COMPLIANT = "PARTIALLY_COMPLIANT"
    QUARANTINED = "QUARANTINED"
    PENDING_REVIEW = "PENDING_REVIEW"


class EvidenceType(str, enum.Enum):
    INFRASTRUCTURE_TELEMETRY = "INFRASTRUCTURE_TELEMETRY"
    ACCESS_CONTROL_AUDIT = "ACCESS_CONTROL_AUDIT"
    ENCRYPTION_CERTIFICATE = "ENCRYPTION_CERTIFICATE"
    VENDOR_RISK_ASSESSMENT = "VENDOR_RISK_ASSESSMENT"
    INCIDENT_RESPONSE_LOG = "INCIDENT_RESPONSE_LOG"
    BACKUP_RESTORATION_VERIFICATION = "BACKUP_RESTORATION_VERIFICATION"


class ComplianceEvidence(BaseModel):
    evidence_id: str
    tenant_id: str
    framework: ComplianceFramework
    evidence_type: EvidenceType
    source_system: str
    payload: Dict[str, Any]
    collected_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.timezone.utc))
    checksum: Optional[str] = None

    def compute_checksum(self) -> str:
        serialized = json.dumps(
            {
                "evidence_id": self.evidence_id,
                "tenant_id": self.tenant_id,
                "framework": self.framework.value,
                "evidence_type": self.evidence_type.value,
                "source": self.source_system,
                "payload": self.payload,
            },
            sort_keys=True,
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def seal(self) -> ComplianceEvidence:
        self.checksum = self.compute_checksum()
        return self


class EvaluationResult(BaseModel):
    control_id: str
    framework: ComplianceFramework
    status: ControlStatus
    score: float = Field(ge=0.0, le=100.0)
    summary: str
    violations: List[str] = Field(default_factory=list)
    remediation_steps: List[str] = Field(default_factory=list)
    evaluated_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.timezone.utc))


class AuditChainBlock(BaseModel):
    block_index: int
    tenant_id: str
    timestamp: dt.datetime
    evidence_checksums: List[str]
    previous_block_hash: str
    block_hash: str


class AIAuditAssessment(BaseModel):
    assessment_id: str
    evidence_id: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    identified_risks: List[str]
    suggested_controls: List[str]
    is_quarantined: bool = False
    quarantine_reason: Optional[str] = None


class DataGuardComplianceOrchestrator:
    """Core enterprise compliance orchestration engine for DataGuard."""

    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id
        self._evidence_store: Dict[str, ComplianceEvidence] = {}
        self._audit_chain: List[AuditChainBlock] = []
        self._genesis_hash = hashlib.sha256(f"DATAGUARD_GENESIS_{tenant_id}".encode("utf-8")).hexdigest()

    def ingest_evidence(self, evidence: ComplianceEvidence) -> str:
        """Ingests and cryptographically seals a compliance evidence object."""
        if evidence.tenant_id != self.tenant_id:
            raise ValueError(f"Tenant mismatch: expected {self.tenant_id}, got {evidence.tenant_id}")
        sealed = evidence.seal()
        self._evidence_store[sealed.evidence_id] = sealed
        return sealed.checksum or ""

    def evaluate_gdpr_policy(self, evidence_id: str) -> EvaluationResult:
        """Evaluates GDPR Article 32 & 33 compliance (security of processing & breach notification)."""
        evidence = self._evidence_store.get(evidence_id)
        if not evidence or evidence.framework != ComplianceFramework.GDPR:
            raise ValueError(f"Valid GDPR evidence {evidence_id} not found")

        violations: List[str] = []
        remediation: List[str] = []
        score = 100.0

        p = evidence.payload
        # Rule 1: Encryption at rest
        if not p.get("encryption_at_rest", False):
            violations.append("Data storage lacks authenticated AES-256 encryption at rest")
            remediation.append("Enable KMS-managed server-side encryption with key rotation")
            score -= 35.0

        # Rule 2: TLS version requirement (TLS 1.3 required, 1.2 minimum)
        tls_ver = str(p.get("tls_version", "1.0"))
        if tls_ver in ["1.0", "1.1"]:
            violations.append(f"Insecure transit protocol detected: TLS {tls_ver}")
            remediation.append("Deprecate legacy TLS protocols; enforce TLS 1.3 across all endpoints")
            score -= 30.0

        # Rule 3: Data retention policy enforcement
        if not p.get("automated_retention_purge", False):
            violations.append("Missing automated deletion workflow for expired personal data")
            remediation.append("Configure automated retention policies and scheduled purge routines")
            score -= 20.0

        # Rule 4: Access audit logging
        if not p.get("immutable_audit_logging", False):
            violations.append("Access logging is not configured as immutable / write-once")
            remediation.append("Forward access logs to write-once-read-many (WORM) storage")
            score -= 15.0

        score = max(0.0, score)
        if score >= 90.0:
            status = ControlStatus.COMPLIANT
            summary = "GDPR technical security controls satisfy Article 32 mandates"
        elif score >= 60.0:
            status = ControlStatus.PARTIALLY_COMPLIANT
            summary = "GDPR security posture contains non-critical gaps requiring remediation"
        else:
            status = ControlStatus.NON_COMPLIANT
            summary = "Critical GDPR non-compliance detected with direct regulatory liability"

        return EvaluationResult(
            control_id=f"GDPR-ART-32-{evidence.evidence_id}",
            framework=ComplianceFramework.GDPR,
            status=status,
            score=score,
            summary=summary,
            violations=violations,
            remediation_steps=remediation,
        )

    def evaluate_iso27001_policy(self, evidence_id: str) -> EvaluationResult:
        """Evaluates ISO/IEC 27001:2022 Annex A controls (A.5, A.8)."""
        evidence = self._evidence_store.get(evidence_id)
        if not evidence or evidence.framework != ComplianceFramework.ISO_27001:
            raise ValueError(f"Valid ISO 27001 evidence {evidence_id} not found")

        violations: List[str] = []
        remediation: List[str] = []
        score = 100.0
        p = evidence.payload

        # Control A.8.24: Use of cryptography
        if not p.get("crypto_key_rotation_days") or p["crypto_key_rotation_days"] > 90:
            violations.append("Cryptographic keys exceed 90-day rotation threshold")
            remediation.append("Configure automated 90-day key rotation schedule")
            score -= 25.0

        # Control A.8.5: Secure authentication & MFA
        if not p.get("mfa_enforced_for_all_users", False):
            violations.append("Multi-Factor Authentication (MFA) is not enforced for all accounts")
            remediation.append("Mandate hardware-backed WebAuthn or TOTP MFA across all accounts")
            score -= 40.0

        # Control A.8.13: Information backup
        if not p.get("backup_tested_in_last_30_days", False):
            violations.append("No successful backup restoration test logged within 30 days")
            remediation.append("Execute automated disaster recovery backup restoration drill")
            score -= 25.0

        # Control A.5.15: Access control
        if p.get("unassigned_privileged_roles", 0) > 0:
            violations.append(f"{p['unassigned_privileged_roles']} orphaned administrative roles detected")
            remediation.append("Revoke dormant administrative credentials and enforce least privilege")
            score -= 10.0

        score = max(0.0, score)
        if score >= 90.0:
            status = ControlStatus.COMPLIANT
            summary = "ISO 27001:2022 technical controls verified compliant"
        elif score >= 60.0:
            status = ControlStatus.PARTIALLY_COMPLIANT
            summary = "ISO 27001:2022 controls require minor remediation"
        else:
            status = ControlStatus.NON_COMPLIANT
            summary = "ISO 27001:2022 audit failure: critical security controls missing"

        return EvaluationResult(
            control_id=f"ISO27001-A8-{evidence.evidence_id}",
            framework=ComplianceFramework.ISO_27001,
            status=status,
            score=score,
            summary=summary,
            violations=violations,
            remediation_steps=remediation,
        )

    def evaluate_agentic_ai_assessment(
        self,
        evidence_id: str,
        raw_llm_response: Dict[str, Any],
        confidence_threshold: float = 0.85,
    ) -> AIAuditAssessment:
        """Processes AI/LLM compliance findings with untrusted output quarantine."""
        evidence = self._evidence_store.get(evidence_id)
        if not evidence:
            raise ValueError(f"Evidence {evidence_id} not found")

        # Validate structured schema
        confidence = float(raw_llm_response.get("confidence", 0.0))
        risks = raw_llm_response.get("risks", [])
        controls = raw_llm_response.get("controls", [])

        is_quarantined = False
        quarantine_reason = None

        if confidence < confidence_threshold:
            is_quarantined = True
            quarantine_reason = (
                f"Confidence {confidence:.2f} below statutory audit threshold {confidence_threshold:.2f}"
            )
        elif not isinstance(risks, list) or not isinstance(controls, list):
            is_quarantined = True
            quarantine_reason = "Malformed structured AI output: expected lists for risks and controls"
        elif any("hallucination_flag" in str(r).lower() for r in risks):
            is_quarantined = True
            quarantine_reason = "Automated prompt injection or hallucination marker detected in assessment"

        return AIAuditAssessment(
            assessment_id=f"AI-AUDIT-{evidence_id}-{dt.datetime.now(dt.timezone.utc).strftime('%H%M%S')}",
            evidence_id=evidence_id,
            confidence_score=confidence,
            identified_risks=risks if not is_quarantined else [],
            suggested_controls=controls if not is_quarantined else [],
            is_quarantined=is_quarantined,
            quarantine_reason=quarantine_reason,
        )

    def append_audit_block(self, evidence_ids: List[str]) -> AuditChainBlock:
        """Builds and cryptographically seals a tamper-evident audit ledger block."""
        checksums: List[str] = []
        for eid in evidence_ids:
            ev = self._evidence_store.get(eid)
            if not ev or not ev.checksum:
                raise ValueError(f"Evidence {eid} missing or unsealed")
            checksums.append(ev.checksum)

        prev_hash = self._audit_chain[-1].block_hash if self._audit_chain else self._genesis_hash
        block_idx = len(self._audit_chain)
        timestamp = dt.datetime.now(dt.timezone.utc)

        block_payload = {
            "index": block_idx,
            "tenant_id": self.tenant_id,
            "timestamp": timestamp.isoformat(),
            "checksums": sorted(checksums),
            "prev_hash": prev_hash,
        }
        block_hash = hashlib.sha256(json.dumps(block_payload, sort_keys=True).encode("utf-8")).hexdigest()

        block = AuditChainBlock(
            block_index=block_idx,
            tenant_id=self.tenant_id,
            timestamp=timestamp,
            evidence_checksums=sorted(checksums),
            previous_block_hash=prev_hash,
            block_hash=block_hash,
        )
        self._audit_chain.append(block)
        return block

    def verify_audit_chain_integrity(self) -> bool:
        """Verifies the unbroken cryptographic integrity of the entire audit chain."""
        if not self._audit_chain:
            return True

        for i, block in enumerate(self._audit_chain):
            expected_prev = self._genesis_hash if i == 0 else self._audit_chain[i - 1].block_hash
            if block.previous_block_hash != expected_prev:
                return False

            payload = {
                "index": block.block_index,
                "tenant_id": block.tenant_id,
                "timestamp": block.timestamp.isoformat(),
                "checksums": sorted(block.evidence_checksums),
                "prev_hash": expected_prev,
            }
            computed_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
            if block.block_hash != computed_hash:
                return False

        return True

    def calculate_tenant_compliance_posture(self) -> Dict[str, Any]:
        """Calculates aggregated compliance readiness and framework coverage."""
        framework_counts: Dict[str, int] = {}
        for ev in self._evidence_store.values():
            framework_counts[ev.framework.value] = framework_counts.get(ev.framework.value, 0) + 1

        return {
            "tenant_id": self.tenant_id,
            "total_sealed_evidence": len(self._evidence_store),
            "total_audit_blocks": len(self._audit_chain),
            "chain_valid": self.verify_audit_chain_integrity(),
            "framework_evidence_distribution": framework_counts,
        }
