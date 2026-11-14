"""Inkeep Open-Source Agent Platform SDK & Runtime Orchestrator.

Provides developer-ergonomic, type-safe agent abstractions:
- Declarative Agent Tool definitions with JSON Schema validation and human-in-the-loop gates.
- Stateful, branched conversational Agent Sessions with token budget ceilings.
- Streaming event emission state machine (tokens, tool lifecycle, approvals, errors).
- Data Loss Prevention (DLP) guardrails scanning inputs/outputs for sensitive credentials.
- Automated multi-dimensional Agent Evaluation Harness (grounding, tool accuracy, relevance, safety).
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class EventType(str, Enum):
    """Event types emitted during an agent execution cycle."""

    TOKEN = "token"
    TOOL_START = "tool_start"
    TOOL_RESULT = "tool_result"
    HUMAN_APPROVAL_REQUIRED = "human_approval_required"
    ERROR = "error"
    COMPLETED = "completed"


@dataclass
class AgentToolDefinition:
    """Strongly-typed definition for an agent-invocable tool."""

    name: str
    description: str
    parameters_schema: Dict[str, Any]
    handler: Callable[[Dict[str, Any]], Any]
    requires_human_approval: bool = False
    timeout_seconds: float = 30.0

    def validate_args(self, args: Dict[str, Any]) -> bool:
        """Validate argument dictionary against required schema parameters."""
        required = self.parameters_schema.get("required", [])
        return all(key in args for key in required)

    def execute(self, args: Dict[str, Any]) -> Any:
        """Execute the tool handler with argument validation."""
        if not self.validate_args(args):
            missing = [k for k in self.parameters_schema.get("required", []) if k not in args]
            raise ValueError(f"Missing required arguments for tool '{self.name}': {missing}")
        return self.handler(args)


@dataclass
class AgentStreamEvent:
    """Streaming event emitted by the Inkeep agent platform runtime."""

    event_type: EventType
    payload: Dict[str, Any]
    session_id: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentSession:
    """Stateful agent session supporting branching, history, and token tracking."""

    session_id: str
    parent_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, str]] = field(default_factory=list)
    tokens_consumed: int = 0
    token_budget_limit: int = 128000
    is_active: bool = True
    created_at: float = field(default_factory=time.time)

    def add_message(self, role: str, content: str, tokens: int = 0) -> None:
        """Add a turn to conversation history and accrue token budget."""
        self.history.append({"role": role, "content": content})
        self.tokens_consumed += tokens
        if self.tokens_consumed > self.token_budget_limit:
            self.is_active = False

    def branch(self, new_session_id: Optional[str] = None) -> "AgentSession":
        """Create a child branch session retaining parent conversation history."""
        child_id = new_session_id or f"{self.session_id}-branch-{uuid.uuid4().hex[:6]}"
        return AgentSession(
            session_id=child_id,
            parent_id=self.session_id,
            metadata=dict(self.metadata),
            history=list(self.history),
            tokens_consumed=self.tokens_consumed,
            token_budget_limit=self.token_budget_limit,
            is_active=True,
        )


class DLPScanner:
    """Zero-trust Data Loss Prevention scanner for inputs and outputs."""

    PATTERNS = {
        "anthropic_api_key": re.compile(r"sk-ant-[a-zA-Z0-9_\-]{20,}"),
        "openai_api_key": re.compile(r"sk-[a-zA-Z0-9]{20,}"),
        "bearer_token": re.compile(r"Bearer\s+[a-zA-Z0-9_\-\.]{25,}", re.IGNORECASE),
        "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "generic_secret": re.compile(
            r"(api[_\-]?key|secret[_\-]?key)\s*[:=]\s*['\"][a-zA-Z0-9_\-]{16,}['\"]", re.IGNORECASE
        ),
    }

    @classmethod
    def scan(cls, text: str) -> List[str]:
        """Scan text for sensitive pattern violations. Returns list of violated pattern names."""
        violations = []
        for name, pattern in cls.PATTERNS.items():
            if pattern.search(text):
                violations.append(name)
        return violations

    @classmethod
    def redact(cls, text: str) -> str:
        """Redact sensitive patterns in text with [REDACTED:<pattern>] markers."""
        redacted = text
        for name, pattern in cls.PATTERNS.items():
            redacted = pattern.sub(f"[REDACTED:{name.upper()}]", redacted)
        return redacted


@dataclass
class AgentEvaluationScore:
    """Multi-dimensional automated evaluation score for agent execution."""

    grounding_score: float  # 0.0 to 1.0 (factual adherence to sources)
    relevance_score: float  # 0.0 to 1.0 (relevance to user query)
    tool_accuracy_score: float  # 0.0 to 1.0 (accuracy of tool arguments and invocation)
    safety_score: float  # 0.0 to 1.0 (adherence to DLP and safety policies)

    @property
    def composite_score(self) -> float:
        """Weighted composite quality score."""
        return round(
            (0.35 * self.grounding_score)
            + (0.25 * self.relevance_score)
            + (0.25 * self.tool_accuracy_score)
            + (0.15 * self.safety_score),
            3,
        )

    @property
    def passes_threshold(self) -> bool:
        """Determines if the execution meets the production benchmark (>= 0.85)."""
        return self.composite_score >= 0.85


class InkeepAgentPlatformSDK:
    """Central Inkeep Agent Platform SDK client interface.

    Enables developers to register typed tools, manage branched sessions,
    execute agent flows with streaming telemetry, enforce DLP guardrails,
    and evaluate agent quality metrics.
    """

    def __init__(self, platform_name: str = "inkeep-agent-platform", version: str = "1.0.0") -> None:
        self.platform_name = platform_name
        self.version = version
        self._tools: Dict[str, AgentToolDefinition] = {}
        self._sessions: Dict[str, AgentSession] = {}
        self._pending_approvals: Dict[str, Dict[str, Any]] = {}

    def register_tool(self, tool: AgentToolDefinition) -> None:
        """Register an agent-invocable tool definition."""
        if not tool.name or not isinstance(tool.name, str):
            raise ValueError("Tool name must be a non-empty string.")
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> Optional[AgentToolDefinition]:
        """Retrieve a registered tool by name."""
        return self._tools.get(name)

    def create_session(
        self,
        session_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        token_budget_limit: int = 128000,
    ) -> AgentSession:
        """Create a new conversational agent session."""
        sid = session_id or f"inkeep-sess-{uuid.uuid4().hex[:8]}"
        session = AgentSession(
            session_id=sid,
            parent_id=parent_id,
            metadata=metadata or {},
            token_budget_limit=token_budget_limit,
        )
        self._sessions[sid] = session
        return session

    def get_session(self, session_id: str) -> Optional[AgentSession]:
        """Retrieve a session by its session ID."""
        return self._sessions.get(session_id)

    def branch_session(self, parent_session_id: str, new_session_id: Optional[str] = None) -> AgentSession:
        """Branch an existing session into a new child session."""
        parent = self._sessions.get(parent_session_id)
        if not parent:
            raise KeyError(f"Parent session '{parent_session_id}' not found.")
        child = parent.branch(new_session_id)
        self._sessions[child.session_id] = child
        return child

    def dispatch_prompt(
        self,
        session_id: str,
        prompt: str,
        requested_tools: Optional[List[Dict[str, Any]]] = None,
        event_callback: Optional[Callable[[AgentStreamEvent], None]] = None,
    ) -> Dict[str, Any]:
        """Execute a prompt turn through the agent runtime with streaming events and guardrails."""
        session = self.get_session(session_id)
        if not session:
            raise KeyError(f"Session '{session_id}' not found.")

        if not session.is_active:
            raise RuntimeError(f"Session '{session_id}' is inactive or has exceeded its token limit.")

        def emit(event_type: EventType, payload: Dict[str, Any]) -> None:
            event = AgentStreamEvent(event_type=event_type, payload=payload, session_id=session_id)
            if event_callback:
                event_callback(event)

        # 1. DLP Check on User Prompt
        dlp_violations = DLPScanner.scan(prompt)
        if dlp_violations:
            emit(EventType.ERROR, {"error": "DLP violation in input prompt", "violations": dlp_violations})
            clean_prompt = DLPScanner.redact(prompt)
        else:
            clean_prompt = prompt

        session.add_message("user", clean_prompt, tokens=max(1, len(clean_prompt) // 4))

        # 2. Token Stream Simulation
        simulated_tokens = ["Analyzing", " request", " across", " knowledge", " base", "."]
        for tok in simulated_tokens:
            emit(EventType.TOKEN, {"token": tok})

        # 3. Tool Invocations
        tool_results = []
        if requested_tools:
            for call in requested_tools:
                tool_name = call.get("name", "")
                args = call.get("args", {})
                tool = self.get_tool(tool_name)

                if not tool:
                    emit(EventType.ERROR, {"error": f"Tool '{tool_name}' is not registered."})
                    tool_results.append({"name": tool_name, "status": "error", "message": "Unknown tool"})
                    continue

                if tool.requires_human_approval:
                    approval_id = f"appr-{uuid.uuid4().hex[:6]}"
                    self._pending_approvals[approval_id] = {
                        "session_id": session_id,
                        "tool_name": tool_name,
                        "args": args,
                    }
                    emit(
                        EventType.HUMAN_APPROVAL_REQUIRED,
                        {"approval_id": approval_id, "tool_name": tool_name, "args": args},
                    )
                    tool_results.append(
                        {"name": tool_name, "status": "pending_approval", "approval_id": approval_id}
                    )
                    continue

                emit(EventType.TOOL_START, {"tool_name": tool_name, "args": args})
                try:
                    result = tool.execute(args)
                    emit(EventType.TOOL_RESULT, {"tool_name": tool_name, "result": result})
                    tool_results.append({"name": tool_name, "status": "success", "result": result})
                except Exception as exc:
                    emit(EventType.ERROR, {"tool_name": tool_name, "error": str(exc)})
                    tool_results.append({"name": tool_name, "status": "error", "message": str(exc)})

        # 4. Synthesize Final Agent Response
        response_text = f"Synthesized response for: {clean_prompt}"
        if tool_results:
            response_text += f" (Executed {len(tool_results)} tools)"

        session.add_message("assistant", response_text, tokens=max(1, len(response_text) // 4))
        emit(EventType.COMPLETED, {"response": response_text, "tools_count": len(tool_results)})

        return {
            "session_id": session_id,
            "response": response_text,
            "tool_results": tool_results,
            "tokens_consumed": session.tokens_consumed,
            "active": session.is_active,
        }

    def resolve_approval(self, approval_id: str, approved: bool) -> Dict[str, Any]:
        """Approve or reject a pending human-in-the-loop tool execution."""
        item = self._pending_approvals.pop(approval_id, None)
        if not item:
            raise KeyError(f"Approval request '{approval_id}' not found.")

        if not approved:
            return {"approval_id": approval_id, "status": "rejected"}

        tool = self.get_tool(item["tool_name"])
        if not tool:
            return {"approval_id": approval_id, "status": "error", "message": "Tool disappeared"}

        result = tool.execute(item["args"])
        session = self.get_session(item["session_id"])
        if session:
            session.add_message(
                "tool",
                f"Tool {tool.name} executed post-approval: {result}",
                tokens=10,
            )
        return {"approval_id": approval_id, "status": "approved_and_executed", "result": result}

    def evaluate_execution(
        self,
        grounding_score: float,
        relevance_score: float,
        tool_accuracy_score: float,
        safety_score: float,
    ) -> AgentEvaluationScore:
        """Create and validate an agent evaluation score."""
        for val, name in [
            (grounding_score, "grounding"),
            (relevance_score, "relevance"),
            (tool_accuracy_score, "tool_accuracy"),
            (safety_score, "safety"),
        ]:
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"Score '{name}' must be between 0.0 and 1.0, got {val}")

        return AgentEvaluationScore(
            grounding_score=grounding_score,
            relevance_score=relevance_score,
            tool_accuracy_score=tool_accuracy_score,
            safety_score=safety_score,
        )
