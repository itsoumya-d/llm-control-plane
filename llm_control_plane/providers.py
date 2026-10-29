"""Provider protocol and deterministic fakes used by tests and demos."""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Protocol

from .types import (
    CompletionRequest,
    ProviderError,
    ProviderResponse,
    Usage,
)


class LLMProvider(Protocol):
    name: str

    def complete(self, request: CompletionRequest) -> ProviderResponse:
        ...

    def health(self) -> bool:
        ...


class FakeProvider:
    """Scriptable provider: optional failures and responses, then deterministic text."""

    def __init__(
        self,
        name: str,
        responses: Optional[List[str]] = None,
        failures: Optional[List[Optional[ProviderError]]] = None,
        latency_ms: float = 1.0,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.name = name
        self.responses = list(responses or [])
        self.failures = list(failures or [])
        self.latency_ms = latency_ms
        self.clock = clock
        self.calls: List[CompletionRequest] = []
        self.healthy = True

    def complete(self, request: CompletionRequest) -> ProviderResponse:
        self.calls.append(request)
        if self.failures:
            failure = self.failures.pop(0)
            if failure is not None:
                raise failure
        if not self.healthy:
            raise ProviderError(self.name, "provider is unhealthy", retryable=True)

        text = self.responses.pop(0) if self.responses else f"ok:{request.model}"
        prompt_tokens = sum(len(message.content.split()) for message in request.messages)
        completion_tokens = len(text.split())
        return ProviderResponse(
            text=text,
            model=request.model,
            provider=self.name,
            usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
            latency_ms=self.latency_ms,
            raw={"scripted": True},
        )

    def health(self) -> bool:
        return self.healthy


class EchoProvider:
    """Trivial provider that echoes the last user message."""

    name = "echo"

    def complete(self, request: CompletionRequest) -> ProviderResponse:
        last = request.messages[-1].content if request.messages else ""
        return ProviderResponse(
            text=last,
            model=request.model,
            provider=self.name,
            usage=Usage(prompt_tokens=len(last.split()), completion_tokens=len(last.split())),
            latency_ms=0.0,
        )

    def health(self) -> bool:
        return True


class RuleProvider:
    """Small keyword-routed provider used by the demo and API smoke tests."""

    def __init__(
        self,
        name: str,
        rules: Dict[str, str],
        default: str = "unknown",
        latency_ms: float = 0.5,
    ) -> None:
        self.name = name
        self.rules = dict(rules)
        self.default = default
        self.latency_ms = latency_ms
        self.calls: List[CompletionRequest] = []

    def complete(self, request: CompletionRequest) -> ProviderResponse:
        self.calls.append(request)
        last = request.messages[-1].content if request.messages else ""
        text = self.default
        for keyword, response in self.rules.items():
            if keyword.lower() in last.lower():
                text = response
                break
        return ProviderResponse(
            text=text,
            model=request.model,
            provider=self.name,
            usage=Usage(prompt_tokens=len(last.split()), completion_tokens=len(text.split())),
            latency_ms=self.latency_ms,
        )

    def health(self) -> bool:
        return True


def provider_map(*providers: LLMProvider) -> Dict[str, LLMProvider]:
    return {provider.name: provider for provider in providers}
