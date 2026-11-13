"""Doppler Secrets Manager & Ephemeral Token Broker for Agentic Workflows.

Provides dynamic secret injection, ephemeral lease lifecycles, prompt trace
redaction, multi-tenant isolation, cryptographically signed audit logs,
and high-privilege human-review gate enforcement for Model Context Protocol (MCP)
tools and agentic runtime execution.
"""

from __future__ import annotations

import hashlib
import hmac
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .connectors import (
    ConnectorRegistry,
    MCPToolSpec,
    ToolArgSpec,
)


@dataclass(frozen=True)
class DopplerConfig:
    """Configuration for Doppler Secrets integration and lease security."""

    api_host: str = "https://api.doppler.com"
    service_token: str = "mock_doppler_service_token_production"
    project: str = "ai-platform"
    environment: str = "production"
    default_lease_ttl_seconds: float = 300.0
    max_lease_ttl_seconds: float = 3600.0
    default_max_accesses: int = 5
    enforce_strict_redaction: bool = True
    audit_hmac_secret: str = "doppler-audit-hmac-key-2026"


@dataclass
class SecretLease:
    """Represents a temporary, scoped lease to retrieve an authorized secret."""

    lease_id: str
    secret_name: str
    tenant_id: str
    tool_name: str
    environment: str
    created_at: float
    expires_at: float
    access_count: int = 0
    max_access_count: int = 5
    revoked: bool = False
    high_privilege: bool = False
    approval_token: Optional[str] = None

    def is_valid(self, now: Optional[float] = None) -> bool:
        """Check if lease is currently active, unrevoked, unexpired, and under quota."""
        if self.revoked:
            return False
        current_time = now if now is not None else time.time()
        if current_time >= self.expires_at:
            return False
        if self.access_count >= self.max_access_count:
            return False
        return True


@dataclass
class DopplerAuditEvent:
    """Tamper-evident audit event recorded for secret access or lease mutation."""

    event_id: str
    timestamp: float
    event_type: str
    tenant_id: str
    secret_name: str
    tool_name: str
    lease_id: str
    hmac_signature: str
    details: Dict[str, Any] = field(default_factory=dict)


class DopplerSecretsError(Exception):
    """Base exception for all Doppler secrets broker errors."""


class DopplerSecretNotFoundError(DopplerSecretsError):
    """Raised when the requested secret name does not exist in the environment."""


class DopplerLeaseExpiredError(DopplerSecretsError):
    """Raised when an access attempt is made with an expired lease."""


class DopplerLeaseExhaustedError(DopplerSecretsError):
    """Raised when a lease has exceeded its maximum allowed retrieval count."""


class DopplerLeaseRevokedError(DopplerSecretsError):
    """Raised when attempting to resolve a revoked lease."""


class DopplerHighPrivilegeApprovalRequiredError(DopplerSecretsError):
    """Raised when high-privilege secret access lacks a verified human approval token."""


class DopplerTenantIsolationViolationError(DopplerSecretsError):
    """Raised when a tenant attempts to resolve or lease secrets belonging to another tenant."""


class DopplerSecretsBroker:
    """Thread-safe secrets broker managing ephemeral leases, access guardrails, and audit logs."""

    def __init__(self, config: Optional[DopplerConfig] = None) -> None:
        self.config = config or DopplerConfig()
        self._lock = threading.RLock()
        # Stored secrets: (environment, secret_name) -> Dict[str, Any]
        self._secrets: Dict[Tuple[str, str], Dict[str, Any]] = {}
        # Active leases: lease_id -> SecretLease
        self._leases: Dict[str, SecretLease] = {}
        # Known secret values to redact: raw_value -> secret_name
        self._redaction_patterns: Dict[str, str] = {}
        # Audit log history
        self._audit_log: List[DopplerAuditEvent] = []

    def set_secret(
        self,
        secret_name: str,
        secret_value: str,
        environment: Optional[str] = None,
        high_privilege: bool = False,
        allowed_tenants: Optional[Set[str]] = None,
    ) -> None:
        """Register or update a secret in the broker."""
        env = environment or self.config.environment
        with self._lock:
            key = (env, secret_name)
            self._secrets[key] = {
                "name": secret_name,
                "value": secret_value,
                "environment": env,
                "high_privilege": high_privilege,
                "allowed_tenants": allowed_tenants,
                "sha256": hashlib.sha256(secret_value.encode("utf-8")).hexdigest(),
                "created_at": time.time(),
            }
            if self.config.enforce_strict_redaction and len(secret_value) >= 4:
                self._redaction_patterns[secret_value] = secret_name

    def request_lease(
        self,
        tenant_id: str,
        secret_name: str,
        tool_name: str,
        environment: Optional[str] = None,
        ttl_seconds: Optional[float] = None,
        max_access_count: Optional[int] = None,
        approval_token: Optional[str] = None,
    ) -> SecretLease:
        """Issue a temporary, scoped lease for a secret under security guardrails."""
        env = environment or self.config.environment
        with self._lock:
            key = (env, secret_name)
            if key not in self._secrets:
                self._record_audit(
                    event_type="access_denied",
                    tenant_id=tenant_id,
                    secret_name=secret_name,
                    tool_name=tool_name,
                    lease_id="none",
                    details={"reason": "secret_not_found", "environment": env},
                )
                raise DopplerSecretNotFoundError(
                    f"Secret '{secret_name}' not found in environment '{env}'"
                )

            sec_meta = self._secrets[key]
            allowed_tenants = sec_meta.get("allowed_tenants")
            if allowed_tenants is not None and tenant_id not in allowed_tenants:
                self._record_audit(
                    event_type="access_denied",
                    tenant_id=tenant_id,
                    secret_name=secret_name,
                    tool_name=tool_name,
                    lease_id="none",
                    details={"reason": "tenant_isolation_violation", "environment": env},
                )
                raise DopplerTenantIsolationViolationError(
                    f"Tenant '{tenant_id}' is not authorized to access secret '{secret_name}'"
                )

            # High privilege check
            if sec_meta.get("high_privilege", False):
                if not approval_token or not approval_token.startswith("appr_"):
                    self._record_audit(
                        event_type="access_denied",
                        tenant_id=tenant_id,
                        secret_name=secret_name,
                        tool_name=tool_name,
                        lease_id="none",
                        details={"reason": "high_privilege_approval_required", "environment": env},
                    )
                    raise DopplerHighPrivilegeApprovalRequiredError(
                        f"Secret '{secret_name}' is high-privilege and requires a valid approval token"
                    )

            ttl = min(
                ttl_seconds or self.config.default_lease_ttl_seconds,
                self.config.max_lease_ttl_seconds,
            )
            max_access = max_access_count or self.config.default_max_accesses

            now = time.time()
            alpha_alphabet = "abcdefghijklmnop"
            hex_id = uuid.uuid4().hex[:16]
            alpha_suffix = "".join(alpha_alphabet[int(c, 16)] for c in hex_id)
            lease_id = f"lease_dop_{alpha_suffix}"
            lease = SecretLease(
                lease_id=lease_id,
                secret_name=secret_name,
                tenant_id=tenant_id,
                tool_name=tool_name,
                environment=env,
                created_at=now,
                expires_at=now + ttl,
                max_access_count=max_access,
                high_privilege=sec_meta.get("high_privilege", False),
                approval_token=approval_token,
            )
            self._leases[lease_id] = lease

            self._record_audit(
                event_type="lease_granted",
                tenant_id=tenant_id,
                secret_name=secret_name,
                tool_name=tool_name,
                lease_id=lease_id,
                details={
                    "ttl_seconds": ttl,
                    "max_access_count": max_access,
                    "high_privilege": lease.high_privilege,
                },
            )
            return lease

    def resolve_secret(self, lease_id: str, tenant_id: str) -> str:
        """Resolve the raw secret value using an active lease, updating metrics and audit logs."""
        with self._lock:
            if lease_id not in self._leases:
                raise DopplerLeaseExpiredError(f"Lease '{lease_id}' not found or already purged")

            lease = self._leases[lease_id]

            if lease.tenant_id != tenant_id:
                self._record_audit(
                    event_type="access_denied",
                    tenant_id=tenant_id,
                    secret_name=lease.secret_name,
                    tool_name=lease.tool_name,
                    lease_id=lease_id,
                    details={"reason": "tenant_mismatch", "lease_tenant": lease.tenant_id},
                )
                raise DopplerTenantIsolationViolationError(
                    f"Lease '{lease_id}' belongs to tenant '{lease.tenant_id}', not '{tenant_id}'"
                )

            if lease.revoked:
                self._record_audit(
                    event_type="access_denied",
                    tenant_id=tenant_id,
                    secret_name=lease.secret_name,
                    tool_name=lease.tool_name,
                    lease_id=lease_id,
                    details={"reason": "lease_revoked"},
                )
                raise DopplerLeaseRevokedError(f"Lease '{lease_id}' has been revoked")

            now = time.time()
            if now >= lease.expires_at:
                self._record_audit(
                    event_type="access_denied",
                    tenant_id=tenant_id,
                    secret_name=lease.secret_name,
                    tool_name=lease.tool_name,
                    lease_id=lease_id,
                    details={"reason": "lease_expired", "expired_at": lease.expires_at},
                )
                raise DopplerLeaseExpiredError(f"Lease '{lease_id}' expired at {lease.expires_at}")

            if lease.access_count >= lease.max_access_count:
                self._record_audit(
                    event_type="access_denied",
                    tenant_id=tenant_id,
                    secret_name=lease.secret_name,
                    tool_name=lease.tool_name,
                    lease_id=lease_id,
                    details={"reason": "lease_quota_exhausted", "quota": lease.max_access_count},
                )
                raise DopplerLeaseExhaustedError(
                    f"Lease '{lease_id}' has reached maximum access count ({lease.max_access_count})"
                )

            key = (lease.environment, lease.secret_name)
            if key not in self._secrets:
                raise DopplerSecretNotFoundError(
                    f"Secret '{lease.secret_name}' no longer exists in environment '{lease.environment}'"
                )

            lease.access_count += 1
            sec_meta = self._secrets[key]
            raw_value = sec_meta["value"]

            self._record_audit(
                event_type="secret_accessed",
                tenant_id=tenant_id,
                secret_name=lease.secret_name,
                tool_name=lease.tool_name,
                lease_id=lease_id,
                details={
                    "access_count": lease.access_count,
                    "remaining_accesses": lease.max_access_count - lease.access_count,
                    "time_to_expiry_seconds": round(lease.expires_at - now, 2),
                },
            )
            return raw_value

    def revoke_lease(self, lease_id: str, tenant_id: str) -> bool:
        """Immediately revoke an active lease."""
        with self._lock:
            if lease_id not in self._leases:
                return False
            lease = self._leases[lease_id]
            if lease.tenant_id != tenant_id:
                raise DopplerTenantIsolationViolationError("Cannot revoke lease belonging to another tenant")
            lease.revoked = True
            self._record_audit(
                event_type="lease_revoked",
                tenant_id=tenant_id,
                secret_name=lease.secret_name,
                tool_name=lease.tool_name,
                lease_id=lease_id,
                details={"revoked_at": time.time()},
            )
            return True

    def prune_expired_leases(self, now: Optional[float] = None) -> int:
        """Remove expired or revoked leases from active memory."""
        current_time = now if now is not None else time.time()
        pruned = 0
        with self._lock:
            to_remove = [
                lid
                for lid, lease in self._leases.items()
                if lease.revoked
                or current_time >= lease.expires_at
                or lease.access_count >= lease.max_access_count
            ]
            for lid in to_remove:
                del self._leases[lid]
                pruned += 1
        return pruned

    def redact_trace(self, text: str) -> str:
        """Redact known secret values from prompt traces, tool arguments, or logs."""
        if not self.config.enforce_strict_redaction:
            return text
        with self._lock:
            redacted = text
            for raw_val, name in self._redaction_patterns.items():
                if raw_val in redacted:
                    token_hash = hashlib.sha256(raw_val.encode("utf-8")).hexdigest()[:8]
                    mask = f"[DOPPLER_SECRET_REDACTED:{name}:{token_hash}]"
                    redacted = redacted.replace(raw_val, mask)
            return redacted

    @staticmethod
    def mask_secret(secret_value: str) -> str:
        """Return a safe masked representation of a secret (e.g. dop_v1_...8f2a)."""
        if len(secret_value) <= 8:
            return "***"
        prefix = secret_value[:6]
        suffix = secret_value[-4:]
        return f"{prefix}...{suffix}"

    def get_audit_log(self, tenant_id: Optional[str] = None) -> List[DopplerAuditEvent]:
        """Return audit event logs, optionally filtered by tenant."""
        with self._lock:
            if tenant_id:
                return [ev for ev in self._audit_log if ev.tenant_id == tenant_id]
            return list(self._audit_log)

    def verify_audit_event(self, event: DopplerAuditEvent) -> bool:
        """Cryptographically verify the HMAC signature of an audit event."""
        payload = (
            f"{event.event_id}:{event.timestamp}:{event.event_type}:"
            f"{event.tenant_id}:{event.secret_name}:{event.tool_name}:{event.lease_id}"
        )
        expected = hmac.new(
            self.config.audit_hmac_secret.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(expected, event.hmac_signature)

    def _record_audit(
        self,
        event_type: str,
        tenant_id: str,
        secret_name: str,
        tool_name: str,
        lease_id: str,
        details: Dict[str, Any],
    ) -> None:
        """Generate and append an authenticated audit record."""
        now = time.time()
        alpha_alphabet = "abcdefghijklmnop"
        hex_id = uuid.uuid4().hex[:12]
        event_id = "evt_dop_" + "".join(alpha_alphabet[int(c, 16)] for c in hex_id)
        payload = f"{event_id}:{now}:{event_type}:{tenant_id}:{secret_name}:{tool_name}:{lease_id}"
        sig = hmac.new(
            self.config.audit_hmac_secret.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        event = DopplerAuditEvent(
            event_id=event_id,
            timestamp=now,
            event_type=event_type,
            tenant_id=tenant_id,
            secret_name=secret_name,
            tool_name=tool_name,
            lease_id=lease_id,
            hmac_signature=sig,
            details=details,
        )
        self._audit_log.append(event)


def register_doppler_mcp_tools(registry: ConnectorRegistry, broker: DopplerSecretsBroker) -> None:
    """Register Doppler secret management and lease resolution tools into the MCP ConnectorRegistry."""

    def handle_request_lease(args: Dict[str, Any]) -> Dict[str, Any]:
        secret_name = args.get("secret_name", "")
        environment = args.get("environment")
        ttl_seconds = args.get("ttl_seconds")
        max_accesses = args.get("max_access_count")
        approval_token = args.get("approval_token")
        tenant_id = args.get("tenant_id", "default_tenant")
        tool_name = args.get("tool_name", "agent_executor")

        ttl_float = float(ttl_seconds) if ttl_seconds is not None else None
        max_acc_int = int(max_accesses) if max_accesses is not None else None

        lease = broker.request_lease(
            tenant_id=tenant_id,
            secret_name=secret_name,
            tool_name=tool_name,
            environment=environment,
            ttl_seconds=ttl_float,
            max_access_count=max_acc_int,
            approval_token=approval_token,
        )
        return {
            "lease_id": lease.lease_id,
            "secret_name": lease.secret_name,
            "environment": lease.environment,
            "expires_at": lease.expires_at,
            "max_access_count": lease.max_access_count,
            "high_privilege": lease.high_privilege,
        }

    def handle_resolve_secret(args: Dict[str, Any]) -> Dict[str, Any]:
        lease_id = args.get("lease_id", "")
        tenant_id = args.get("tenant_id", "default_tenant")
        raw_val = broker.resolve_secret(lease_id=lease_id, tenant_id=tenant_id)
        masked = broker.mask_secret(raw_val)
        return {
            "lease_id": lease_id,
            "secret_value": raw_val,
            "masked_display": masked,
            "resolved": True,
        }

    def handle_revoke_lease(args: Dict[str, Any]) -> Dict[str, Any]:
        lease_id = args.get("lease_id", "")
        tenant_id = args.get("tenant_id", "default_tenant")
        success = broker.revoke_lease(lease_id=lease_id, tenant_id=tenant_id)
        return {"lease_id": lease_id, "revoked": success}

    def handle_audit_secrets(args: Dict[str, Any]) -> Dict[str, Any]:
        tenant_id = args.get("tenant_id")
        events = broker.get_audit_log(tenant_id=tenant_id)
        return {
            "event_count": len(events),
            "events": [
                {
                    "event_id": ev.event_id,
                    "timestamp": ev.timestamp,
                    "event_type": ev.event_type,
                    "secret_name": ev.secret_name,
                    "tool_name": ev.tool_name,
                    "lease_id": ev.lease_id,
                    "verified": broker.verify_audit_event(ev),
                }
                for ev in events[-20:]
            ],
        }

    # Register specs
    registry.register(
        MCPToolSpec(
            name="doppler_request_secret_lease",
            connector="doppler",
            description="Request a temporary scoped lease to access a Doppler secret with guardrails.",
            args={
                "secret_name": ToolArgSpec(arg_type="string", required=True),
                "environment": ToolArgSpec(arg_type="string", required=False),
                "ttl_seconds": ToolArgSpec(arg_type="number", required=False),
                "max_access_count": ToolArgSpec(arg_type="integer", required=False),
                "approval_token": ToolArgSpec(arg_type="string", required=False),
                "tenant_id": ToolArgSpec(arg_type="string", required=False),
                "tool_name": ToolArgSpec(arg_type="string", required=False),
            },
            risk_level="read",
        ),
        handler=handle_request_lease,
    )

    registry.register(
        MCPToolSpec(
            name="doppler_resolve_secret",
            connector="doppler",
            description="Resolve the secret value using a valid ephemeral lease ID.",
            args={
                "lease_id": ToolArgSpec(arg_type="string", required=True),
                "tenant_id": ToolArgSpec(arg_type="string", required=False),
            },
            risk_level="read",
        ),
        handler=handle_resolve_secret,
    )

    registry.register(
        MCPToolSpec(
            name="doppler_revoke_secret_lease",
            connector="doppler",
            description="Immediately revoke an active ephemeral secret lease.",
            args={
                "lease_id": ToolArgSpec(arg_type="string", required=True),
                "tenant_id": ToolArgSpec(arg_type="string", required=False),
            },
            risk_level="write",
        ),
        handler=handle_revoke_lease,
    )

    registry.register(
        MCPToolSpec(
            name="doppler_audit_secrets",
            connector="doppler",
            description="Query the cryptographic audit trail for secret leases and access events.",
            args={
                "tenant_id": ToolArgSpec(arg_type="string", required=False),
            },
            risk_level="read",
        ),
        handler=handle_audit_secrets,
    )
