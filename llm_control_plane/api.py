"""FastAPI surface for the gateway, evals, health, and Prometheus metrics."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from .connectors import (
    ApprovalRequiredError,
    ConnectorNotFoundError,
    ConnectorRegistry,
    ConnectorSchemaError,
    ToolCallRequest,
    ToolTrajectoryScorer,
)
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
from .optimal_research_engine import (
    CardSortPlacement,
    QualitativeObservation,
    ResearchObservationClusterer,
    StudyType,
    ThematicSynthesisEngine,
    TreejackNavigationScorer,
    TreeTestTaskResult,
    UserResearchStudyManager,
)
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


class ConnectorCallIn(BaseModel):
    tenant_id: str
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    trace_id: Optional[str] = None
    dry_run: bool = False
    approver_id: Optional[str] = None
    approval_token: Optional[str] = None


class CreateStudyIn(BaseModel):
    study_id: str
    name: str
    study_type: str
    metadata: Optional[Dict[str, Any]] = None


class CardSortPlacementIn(BaseModel):
    participant_id: str
    card_id: str
    card_label: str
    category_id: str
    category_label: str


class CardSortAnalyzeIn(BaseModel):
    study_id: str
    cards: List[str]
    placements: List[CardSortPlacementIn]
    cluster_threshold: float = 0.5


class TreeTestResultIn(BaseModel):
    participant_id: str
    task_id: str
    selected_node_id: str
    path_taken: List[str]
    time_spent_seconds: float
    success: bool
    direct: bool


class TreeTestEvaluateIn(BaseModel):
    study_id: str
    task_id: str
    target_node_id: str
    results: List[TreeTestResultIn]


class QualitativeObservationIn(BaseModel):
    observation_id: str
    session_id: str
    raw_text: str
    tags: List[str] = Field(default_factory=list)
    participant_quote: Optional[str] = None
    sentiment: str = "neutral"
    impact_weight: float = 1.0


class ReframerSynthesizeIn(BaseModel):
    study_id: str
    observations: List[QualitativeObservationIn]
    min_theme_size: int = 1


SCORERS = {
    "exact_match": ExactMatchScorer,
    "contains": ContainsScorer,
    "json_schema": JsonSchemaScorer,
    "numeric_tolerance": NumericToleranceScorer,
    "tool_trajectory": ToolTrajectoryScorer,
}


def create_app(
    gateway: Optional[LLMGateway] = None,
    connectors: Optional[ConnectorRegistry] = None,
) -> FastAPI:
    if gateway is None:
        from .demo import build_demo_gateway

        gateway = build_demo_gateway()
    registry = connectors or ConnectorRegistry(audit=gateway.audit, metrics=gateway.metrics)

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

    @app.get("/v1/connectors")
    def list_connectors(connector: Optional[str] = None) -> Dict[str, Any]:
        specs = registry.list_specs(connector=connector)
        return {
            "tools": [
                {
                    "name": spec.name,
                    "connector": spec.connector,
                    "description": spec.description,
                    "risk_level": spec.risk_level,
                    "requires_human_approval": spec.requires_human_approval,
                }
                for spec in specs
            ]
        }

    @app.post("/v1/connectors/execute")
    def execute_connector(body: ConnectorCallIn) -> Any:
        req = ToolCallRequest(
            tenant_id=body.tenant_id,
            tool_name=body.tool_name,
            arguments=body.arguments,
            trace_id=body.trace_id,
            dry_run=body.dry_run,
            approver_id=body.approver_id,
            approval_token=body.approval_token,
        )
        try:
            result = registry.execute(req)
        except ConnectorNotFoundError as exc:
            return JSONResponse(status_code=404, content={"error": "tool_not_found", "detail": str(exc)})
        except ConnectorSchemaError as exc:
            return JSONResponse(status_code=422, content={"error": "schema_validation_failed", "errors": exc.errors})
        except ApprovalRequiredError as exc:
            return JSONResponse(status_code=403, content={"error": "approval_required", "detail": exc.reason})
        return {
            "tool_name": result.tool_name,
            "connector": result.connector,
            "trace_id": result.trace_id,
            "status": result.status,
            "output": result.output,
            "redactions": result.redactions,
            "audit_seq": result.audit_seq,
        }

    research_manager = UserResearchStudyManager()

    @app.post("/v1/research/studies")
    def create_research_study(body: CreateStudyIn) -> Dict[str, Any]:
        try:
            stype = StudyType(body.study_type)
        except ValueError:
            return JSONResponse(status_code=400, content={"error": "invalid_study_type"})
        study = research_manager.create_study(
            study_id=body.study_id,
            name=body.name,
            study_type=stype,
            metadata=body.metadata,
        )
        return {"study": study}

    @app.post("/v1/research/cardsort/analyze")
    def analyze_card_sort(body: CardSortAnalyzeIn) -> Dict[str, Any]:
        placements = [
            CardSortPlacement(
                participant_id=p.participant_id,
                card_id=p.card_id,
                card_label=p.card_label,
                category_id=p.category_id,
                category_label=p.category_label,
            )
            for p in body.placements
        ]
        participants = len({p.participant_id for p in placements})
        cooc = ResearchObservationClusterer.calculate_cooccurrence_matrix(
            body.cards, placements
        )
        sim = ResearchObservationClusterer.calculate_similarity_matrix(
            body.cards, cooc, participants
        )
        clusters = ResearchObservationClusterer.build_hierarchical_clusters(
            body.cards, sim, body.cluster_threshold
        )
        return {
            "study_id": body.study_id,
            "total_participants": participants,
            "cooccurrence_matrix": cooc,
            "similarity_matrix": sim,
            "clusters": clusters,
        }

    @app.post("/v1/research/treejack/evaluate")
    def evaluate_tree_test(body: TreeTestEvaluateIn) -> Dict[str, Any]:
        results = [
            TreeTestTaskResult(
                participant_id=r.participant_id,
                task_id=r.task_id,
                selected_node_id=r.selected_node_id,
                path_taken=r.path_taken,
                time_spent_seconds=r.time_spent_seconds,
                success=r.success,
                direct=r.direct,
            )
            for r in body.results
        ]
        metrics = TreejackNavigationScorer.evaluate_task(
            body.task_id, body.target_node_id, results
        )
        first_clicks = TreejackNavigationScorer.identify_first_click_paths(results)
        return {
            "study_id": body.study_id,
            "metrics": metrics,
            "first_click_distribution": first_clicks,
        }

    @app.post("/v1/research/reframer/synthesize")
    def synthesize_reframer(body: ReframerSynthesizeIn) -> Dict[str, Any]:
        observations = [
            QualitativeObservation(
                observation_id=o.observation_id,
                session_id=o.session_id,
                raw_text=o.raw_text,
                tags=o.tags,
                participant_quote=o.participant_quote,
                sentiment=o.sentiment,
                impact_weight=o.impact_weight,
            )
            for o in body.observations
        ]
        themes = ThematicSynthesisEngine.extract_thematic_clusters(
            observations, min_theme_size=body.min_theme_size
        )
        return {
            "study_id": body.study_id,
            "total_observations": len(observations),
            "themes": themes,
        }

    return app


app = create_app()
