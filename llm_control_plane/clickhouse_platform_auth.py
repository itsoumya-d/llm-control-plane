"""ClickHouse Cloud Platform Auth: Unified ReBAC authorization, SCIM 2.0 provisioning, and audit telemetry.

Implements Zanzibar-style relationship-based access control (ReBAC), enterprise SCIM 2.0 user
directory lifecycle synchronization, short-lived JWT token issuance with instant revocation,
and tamper-evident access decision audit logging across ClickHouse Cloud services.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Set, Tuple


class PrincipalType(str, Enum):
    """Subject type initiating access requests."""

    USER = "user"
    SERVICE_ACCOUNT = "service_account"
    API_KEY = "api_key"
    GROUP = "group"


class ResourceType(str, Enum):
    """Target resource types across ClickHouse Cloud platform."""

    ORGANIZATION = "organization"
    WORKSPACE = "workspace"
    SERVICE = "service"
    DATABASE = "database"
    TABLE = "table"
    CLICKPIPE = "clickpipe"


class Permission(str, Enum):
    """Fine-grained operations governed by authorization policies."""

    READ = "read"
    WRITE = "write"
    ADMIN = "admin"
    EXECUTE_QUERY = "execute_query"
    MANAGE_MEMBERS = "manage_members"
    CREATE_SERVICE = "create_service"
    DELETE = "delete"


@dataclass(frozen=True)
class RelationTuple:
    """Zanzibar-style authorization relationship tuple: object#relation@subject."""

    resource_type: ResourceType
    resource_id: str
    relation: str
    subject_type: PrincipalType
    subject_id: str


@dataclass
class AuthorizationDecision:
    """Evaluation result for a permission check."""

    allowed: bool
    principal_id: str
    resource_type: ResourceType
    resource_id: str
    permission: Permission
    evaluation_latency_ms: float
    decision_reason: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class SCIMUser:
    """SCIM 2.0 Enterprise User entity representation (RFC 7644)."""

    user_id: str
    external_id: str
    user_name: str
    email: str
    active: bool
    groups: Set[str] = field(default_factory=set)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class JWTToken:
    """Short-lived access token with cryptographically hashed signature."""

    token_id: str
    principal_id: str
    organization_id: str
    scopes: List[str]
    expires_at_epoch: float
    signature_hash: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_expired(self) -> bool:
        """Check whether the token has passed its expiration time."""
        return time.time() >= self.expires_at_epoch


class ClickHouseReBACEngine:
    """Zanzibar-inspired relationship-based access control engine with hierarchical inheritance."""

    def __init__(self) -> None:
        self.tuples: Set[RelationTuple] = set()
        self.resource_hierarchy: Dict[Tuple[ResourceType, str], Tuple[ResourceType, str]] = {}
        # Caches evaluation results: (subject_id, permission, res_type, res_id) -> bool
        self._decision_cache: Dict[Tuple[str, Permission, ResourceType, str], bool] = {}

    def add_relation(
        self,
        resource_type: ResourceType,
        resource_id: str,
        relation: str,
        subject_type: PrincipalType,
        subject_id: str,
    ) -> None:
        """Add a direct relationship tuple to the authorization graph."""
        rel_tuple = RelationTuple(
            resource_type=resource_type,
            resource_id=resource_id,
            relation=relation,
            subject_type=subject_type,
            subject_id=subject_id,
        )
        self.tuples.add(rel_tuple)
        self._decision_cache.clear()

    def remove_relation(
        self,
        resource_type: ResourceType,
        resource_id: str,
        relation: str,
        subject_type: PrincipalType,
        subject_id: str,
    ) -> None:
        """Remove a relationship tuple from the authorization graph."""
        rel_tuple = RelationTuple(
            resource_type=resource_type,
            resource_id=resource_id,
            relation=relation,
            subject_type=subject_type,
            subject_id=subject_id,
        )
        self.tuples.discard(rel_tuple)
        self._decision_cache.clear()

    def set_parent_resource(
        self,
        child_type: ResourceType,
        child_id: str,
        parent_type: ResourceType,
        parent_id: str,
    ) -> None:
        """Establish hierarchical containment (e.g., Table in Database in Service in Org)."""
        self.resource_hierarchy[(child_type, child_id)] = (parent_type, parent_id)
        self._decision_cache.clear()

    def check_permission(
        self,
        principal_id: str,
        permission: Permission,
        resource_type: ResourceType,
        resource_id: str,
    ) -> AuthorizationDecision:
        """Evaluate whether a principal holds a permission on a target resource."""
        t0 = time.perf_counter()
        cache_key = (principal_id, permission, resource_type, resource_id)

        if cache_key in self._decision_cache:
            allowed = self._decision_cache[cache_key]
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return AuthorizationDecision(
                allowed=allowed,
                principal_id=principal_id,
                resource_type=resource_type,
                resource_id=resource_id,
                permission=permission,
                evaluation_latency_ms=latency_ms,
                decision_reason="Cached authorization decision",
            )

        allowed, reason = self._evaluate_permission(principal_id, permission, resource_type, resource_id)
        self._decision_cache[cache_key] = allowed
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return AuthorizationDecision(
            allowed=allowed,
            principal_id=principal_id,
            resource_type=resource_type,
            resource_id=resource_id,
            permission=permission,
            evaluation_latency_ms=latency_ms,
            decision_reason=reason,
        )

    def _evaluate_permission(
        self,
        principal_id: str,
        permission: Permission,
        res_type: ResourceType,
        res_id: str,
    ) -> Tuple[bool, str]:
        # 1. Direct role mapping on target resource
        for tup in self.tuples:
            if (
                tup.resource_type == res_type
                and tup.resource_id == res_id
                and tup.subject_id == principal_id
            ):
                if tup.relation == "owner" or tup.relation == "admin":
                    return True, f"Direct {tup.relation} grant on {res_type.value}:{res_id}"
                if permission in (Permission.READ, Permission.EXECUTE_QUERY) and tup.relation in (
                    "viewer",
                    "member",
                    "editor",
                ):
                    return True, f"Direct {tup.relation} grant on {res_type.value}:{res_id}"
                if permission == Permission.WRITE and tup.relation == "editor":
                    return True, f"Direct editor grant on {res_type.value}:{res_id}"

        # 2. Hierarchical inheritance from parent resource
        parent_key = (res_type, res_id)
        if parent_key in self.resource_hierarchy:
            parent_type, parent_id = self.resource_hierarchy[parent_key]
            parent_allowed, parent_reason = self._evaluate_permission(
                principal_id, permission, parent_type, parent_id
            )
            if parent_allowed:
                return True, f"Inherited from parent {parent_type.value}:{parent_id} ({parent_reason})"

        return False, f"No policy or role relation grants {permission.value} on {res_type.value}:{res_id}"


class ClickHouseSCIMManager:
    """Manages enterprise SCIM 2.0 user directory provisioning and automated de-authorization."""

    def __init__(self, rebac_engine: ClickHouseReBACEngine) -> None:
        self.rebac = rebac_engine
        self.users: Dict[str, SCIMUser] = {}
        self.revoked_principals: Set[str] = set()

    def provision_user(self, user_id: str, external_id: str, user_name: str, email: str) -> SCIMUser:
        """Provision a new enterprise user via SCIM 2.0 webhook/API."""
        user = SCIMUser(
            user_id=user_id,
            external_id=external_id,
            user_name=user_name,
            email=email,
            active=True,
        )
        self.users[user_id] = user
        self.revoked_principals.discard(user_id)
        return user

    def deprovision_user(self, user_id: str) -> None:
        """Immediately deprovision a user and invalidate all permissions cluster-wide."""
        if user_id not in self.users:
            raise KeyError(f"SCIM User {user_id} does not exist.")

        user = self.users[user_id]
        user.active = False
        user.updated_at = datetime.now(timezone.utc)
        self.revoked_principals.add(user_id)

        # Evict direct ReBAC relations
        tuples_to_remove = [t for t in self.rebac.tuples if t.subject_id == user_id]
        for t in tuples_to_remove:
            self.rebac.tuples.remove(t)
        self.rebac._decision_cache.clear()

    def assign_group(self, user_id: str, group_name: str) -> None:
        """Synchronize enterprise IdP group membership."""
        if user_id not in self.users:
            raise KeyError(f"SCIM User {user_id} does not exist.")
        self.users[user_id].groups.add(group_name)

    def is_user_active(self, user_id: str) -> bool:
        """Check whether a user is active and not revoked."""
        if user_id in self.revoked_principals:
            return False
        user = self.users.get(user_id)
        return user.active if user else False


class ClickHouseTokenIssuer:
    """Issues short-lived JWT-style access tokens with distributed revocation checks."""

    def __init__(self, scim_manager: ClickHouseSCIMManager, secret_key: str = "clickhouse-cloud-secret") -> None:
        self.scim = scim_manager
        self.secret_key = secret_key
        self.token_blacklist: Set[str] = set()

    def issue_token(
        self,
        principal_id: str,
        organization_id: str,
        scopes: List[str],
        ttl_seconds: int = 900,
    ) -> JWTToken:
        """Issue a short-lived access token with SHA-256 signature."""
        if not self.scim.is_user_active(principal_id):
            raise PermissionError(f"Cannot issue token for inactive or revoked principal {principal_id}.")

        token_id = hashlib.sha256(f"{principal_id}:{time.time()}:{organization_id}".encode()).hexdigest()[:16]
        expires_at = time.time() + ttl_seconds
        sig_data = f"{token_id}:{principal_id}:{organization_id}:{expires_at}:{self.secret_key}"
        signature_hash = hashlib.sha256(sig_data.encode()).hexdigest()

        return JWTToken(
            token_id=token_id,
            principal_id=principal_id,
            organization_id=organization_id,
            scopes=scopes,
            expires_at_epoch=expires_at,
            signature_hash=signature_hash,
        )

    def validate_token(self, token: JWTToken) -> bool:
        """Validate token authenticity, expiration, and blacklist status."""
        if token.token_id in self.token_blacklist:
            return False
        if token.is_expired:
            return False
        if not self.scim.is_user_active(token.principal_id):
            return False

        expected_sig = hashlib.sha256(
            f"{token.token_id}:{token.principal_id}:{token.organization_id}:{token.expires_at_epoch}:{self.secret_key}".encode()
        ).hexdigest()
        return token.signature_hash == expected_sig

    def revoke_token(self, token_id: str) -> None:
        """Add a token to the immediate revocation blacklist."""
        self.token_blacklist.add(token_id)


class AccessDecisionTelemetry:
    """High-throughput audit telemetry capturing tamper-evident authorization decision logs."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []
        self._prev_hash: str = "0" * 64

    def record_decision(self, decision: AuthorizationDecision) -> str:
        """Record an access decision into the append-only cryptographic hash chain."""
        payload = (
            f"{self._prev_hash}:{decision.principal_id}:{decision.permission.value}:"
            f"{decision.resource_type.value}:{decision.resource_id}:{decision.allowed}:"
            f"{decision.timestamp.isoformat()}"
        )
        event_hash = hashlib.sha256(payload.encode()).hexdigest()

        event = {
            "hash": event_hash,
            "prev_hash": self._prev_hash,
            "allowed": decision.allowed,
            "principal_id": decision.principal_id,
            "permission": decision.permission.value,
            "resource": f"{decision.resource_type.value}:{decision.resource_id}",
            "latency_ms": round(decision.evaluation_latency_ms, 3),
            "reason": decision.decision_reason,
            "timestamp": decision.timestamp.isoformat(),
        }
        self.events.append(event)
        self._prev_hash = event_hash
        return event_hash

    def verify_integrity(self) -> bool:
        """Verify the cryptographic integrity of the audit chain."""
        prev = "0" * 64
        for ev in self.events:
            payload = (
                f"{prev}:{ev['principal_id']}:{ev['permission']}:{ev['resource']}:"
                f"{ev['allowed']}:{ev['timestamp']}"
            )
            expected = hashlib.sha256(payload.encode()).hexdigest()
            if ev["hash"] != expected:
                return False
            prev = ev["hash"]
        return True
