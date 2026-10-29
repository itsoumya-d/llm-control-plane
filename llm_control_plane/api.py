"""FastAPI surface for the gateway, evals, health, and Prometheus metrics."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from .evals import (
    ContainsScorer,
    Dataset,
    EvalCase,
    EvalRunner,
    ExactMatchScorer,
    JsonSchemaScorer,
    NumericToleranceScorer,
    render_markdown,
)
from .gateway import LLMGateway
from .types import ChatMessage, CompletionRequest, ProviderError, RateLimitExceeded


class MessageIn(BaseModel):
    role: str = Field(default="user")
    content: str


class ChatIn(BaseModel):
    tenant_id: str
    model: str = "smart"
    messages: List[MessageIn]
    temperature: float = 0.0
    max_tokens: int = 512
    use_case: str = "default"
    trace_id: Optional[str] = None


class EvalCaseIn(BaseModel):
    id: str
    input: str
    expected: Any
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DatasetIn(BaseModel):
    name: str
    version: str = "v1"
    cases: List[EvalCaseIn]


class EvalRunIn(BaseModel):
    dataset: DatasetIn
    model_alias: str = "smart"
    tenant_id: str
    use_case: str = "eval"
    scorers: List[str] = Field(default_factory=lambda: ["contains"])


SCORERS = {
    "exact_match": ExactMatchScorer,
    "contains": ContainsScorer,
    "json_schema": JsonSchemaScorer,
    "numeric_tolerance": NumericToleranceScorer,
}


def create_app(gateway: Optional[LLMGateway] = None) -> FastAPI:
    if gateway is None:
        from .demo import build_demo_gateway

        gateway = build_demo_gateway()

    app = FastAPI(
        title="LLM Control Plane",
        version="1.0.0",
        description="Provider-agnostic LLM gateway and evaluation control plane.",
    )

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {
            "status": "ok",
            "providers": gateway.health(),
            "audit_events": len(gateway.audit),
            "audit_chain_valid": gateway.audit.verify(),
        }

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics() -> str:
        return gateway.metrics.prometheus_text()

    @app.post("/v1/chat/completions")
    def chat(body: ChatIn) -> Any:
        request = CompletionRequest(
            tenant_id=body.tenant_id,
            model=body.model,
            messages=[ChatMessage(role=message.role, content=message.content) for message in body.messages],
            temperature=body.temperature,
            max_tokens=body.max_tokens,
            use_case=body.use_case,
            trace_id=body.trace_id,
        )
        try:
            result = gateway.complete(request)
        except RateLimitExceeded as exc:
            return JSONResponse(
                status_code=429,
                content={"error": "rate_limited", "retry_after_seconds": exc.retry_after_seconds},
                headers={"Retry-After": f"{exc.retry_after_seconds:.3f}"},
            )
        except ProviderError as exc:
            return JSONResponse(status_code=502, content={"error": "provider_error", "detail": exc.message})
        return {
            "text": result.text,
            "provider": result.provider,
            "model": result.model,
            "usage": {
                "prompt_tokens": result.usage.prompt_tokens,
                "completion_tokens": result.usage.completion_tokens,
                "total_tokens": result.usage.total_tokens,
            },
            "cost_usd": result.cost_usd,
            "latency_ms": result.latency_ms,
            "trace_id": result.trace_id,
            "attempts": result.attempts,
            "redactions": result.redactions,
        }

    @app.post("/v1/evals/runs")
    def run_eval(body: EvalRunIn) -> Any:
        dataset = Dataset(
            name=body.dataset.name,
            version=body.dataset.version,
            cases=[
                EvalCase(
                    id=case.id,
                    input=case.input,
                    expected=case.expected,
                    metadata=case.metadata,
                )
                for case in body.dataset.cases
            ],
        )
        scorers = [SCORERS[name]() for name in body.scorers if name in SCORERS]
        if not scorers:
            return JSONResponse(status_code=400, content={"error": "no_known_scorers"})
        run = EvalRunner(gateway, scorers).run(
            dataset,
            model_alias=body.model_alias,
            tenant_id=body.tenant_id,
            use_case=body.use_case,
        )
        return {"run": run.to_dict(), "markdown": render_markdown(run)}

    return app


app = create_app()
