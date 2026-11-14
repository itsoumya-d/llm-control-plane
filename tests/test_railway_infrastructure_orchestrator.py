"""Tests for Railway Infrastructure Orchestrator & Durable Workflow Engine."""

import asyncio
import time

import pytest

from llm_control_plane.railway_infrastructure_orchestrator import (
    CanvasCoordinate,
    ClickHouseLogBuffer,
    ClickHouseLogEvent,
    InfrastructureNode,
    InfrastructureTopologyGraph,
    NixpacksBuildPlan,
    NodeType,
    TemporalVolumeMigrationWorkflow,
    VolumeMigrationSpec,
    WorkflowStatus,
)


def test_canvas_coordinate_distance():
    c1 = CanvasCoordinate(0.0, 0.0)
    c2 = CanvasCoordinate(3.0, 4.0)
    assert c1.distance_to(c2) == 5.0


def test_infrastructure_node_secret_masking():
    node = InfrastructureNode(
        id="srv-backend",
        name="Backend API",
        node_type=NodeType.SERVICE,
        region="us-west1",
        environment_variables={
            "DATABASE_URL": "postgres://user:password123@db:5432/main",
            "API_KEY": "sk-proj-supersecretkey9999",
            "PORT": "8080",
            "AUTH_SECRET": "jwt-private-token-4321",
        },
        canvas_pos=CanvasCoordinate(100.0, 200.0),
    )
    masked = node.mask_secrets()
    assert masked["PORT"] == "8080"
    assert "password123" not in masked["DATABASE_URL"]
    assert masked["API_KEY"].endswith("9999")
    assert "sk-proj" not in masked["API_KEY"]
    assert masked["AUTH_SECRET"].startswith("••••••••")


def test_topology_graph_dag_ordering():
    graph = InfrastructureTopologyGraph(project_id="proj-railway-alpha")

    db = InfrastructureNode(id="db-pg", name="Postgres", node_type=NodeType.DATABASE, region="us-west1")
    cache = InfrastructureNode(id="redis-cache", name="Redis", node_type=NodeType.DATABASE, region="us-west1")
    api = InfrastructureNode(id="srv-api", name="Core API", node_type=NodeType.SERVICE, region="us-west1")
    frontend = InfrastructureNode(id="srv-web", name="Next.js Web", node_type=NodeType.SERVICE, region="us-west1")

    graph.add_node(db)
    graph.add_node(cache)
    graph.add_node(api)
    graph.add_node(frontend)

    # API depends on DB and Cache; Frontend depends on API
    graph.add_dependency("srv-api", "db-pg")
    graph.add_dependency("srv-api", "redis-cache")
    graph.add_dependency("srv-web", "srv-api")

    order = graph.get_deployment_order()
    # db and redis must precede api; api must precede frontend
    assert order.index("db-pg") < order.index("srv-api")
    assert order.index("redis-cache") < order.index("srv-api")
    assert order.index("srv-api") < order.index("srv-web")


def test_topology_graph_cycle_rejection():
    graph = InfrastructureTopologyGraph(project_id="proj-cyclical")
    n1 = InfrastructureNode(id="n1", name="Service 1", node_type=NodeType.SERVICE, region="us-east1")
    n2 = InfrastructureNode(id="n2", name="Service 2", node_type=NodeType.SERVICE, region="us-east1")
    graph.add_node(n1)
    graph.add_node(n2)

    graph.add_dependency("n1", "n2")
    with pytest.raises(ValueError, match="creates a circular cycle"):
        graph.add_dependency("n2", "n1")

    # Verify self-dependency rejection
    with pytest.raises(ValueError, match="cannot depend on itself"):
        graph.add_dependency("n1", "n1")


def test_nixpacks_build_plan_hash_and_validation():
    plan = NixpacksBuildPlan(
        name="fastapi-app",
        providers=["python", "pip"],
        pkgs=["python311", "gcc"],
        apt_pkgs=["libpq-dev"],
        build_cmd="pip install -r requirements.txt",
        start_cmd="uvicorn main:app --host 0.0.0.0 --port $PORT",
    )
    assert plan.validate() is True
    hash1 = plan.generate_plan_hash()
    assert len(hash1) == 64

    # Idempotent hash
    assert hash1 == plan.generate_plan_hash()

    # Invalid plan without start command
    invalid_plan = NixpacksBuildPlan(
        name="bad-app",
        providers=["node"],
        pkgs=["nodejs"],
        apt_pkgs=[],
        build_cmd="npm run build",
        start_cmd="   ",
    )
    with pytest.raises(ValueError, match="non-empty start_cmd"):
        invalid_plan.validate()


def test_temporal_volume_migration_success():
    spec = VolumeMigrationSpec(
        service_id="srv-telemetry",
        volume_id="vol-storage-88",
        source_region="us-west1",
        target_region="eu-central1",
        volume_size_gb=100,
    )
    workflow = TemporalVolumeMigrationWorkflow(spec)
    success = asyncio.run(workflow.execute())
    assert success is True
    assert workflow.status == WorkflowStatus.COMPLETED
    assert workflow.snapshot_id is not None
    assert workflow.replicated_volume_id is not None
    assert len(workflow.execution_log) == 7


def test_temporal_volume_migration_compensation_on_failure():
    spec = VolumeMigrationSpec(
        service_id="srv-payment-ledger",
        volume_id="vol-ledger-01",
        source_region="us-west1",
        target_region="ap-southeast1",
        volume_size_gb=250,
    )
    workflow = TemporalVolumeMigrationWorkflow(spec)
    # Simulate network partition during cross-region replication
    success = asyncio.run(workflow.execute(simulate_failure_step=WorkflowStatus.DATA_REPLICATED))
    assert success is False
    assert workflow.status == WorkflowStatus.COMPENSATED

    # Verify compensation log recorded
    log_messages = [msg for status, msg in workflow.execution_log]
    assert any("Simulated cross-region network partition" in msg for msg in log_messages)
    assert any("Re-attaching source volume" in msg for msg in log_messages)


def test_clickhouse_log_buffer_batching_and_query():
    buffer = ClickHouseLogBuffer(batch_size=10, flush_interval_sec=0.1)

    async def run_scenario():
        for i in range(9):
            flushed = await buffer.emit(
                ClickHouseLogEvent(
                    timestamp_ns=time.time_ns(),
                    project_id="proj-alpha",
                    service_id="srv-web",
                    deployment_id="dep-123",
                    level="INFO" if i % 2 == 0 else "ERROR",
                    message=f"Log event #{i}",
                )
            )
            assert flushed is None

        # 10th event triggers flush of 10 events
        flushed = await buffer.emit(
            ClickHouseLogEvent(
                timestamp_ns=time.time_ns(),
                project_id="proj-alpha",
                service_id="srv-web",
                deployment_id="dep-123",
                level="INFO",
                message="Log event #9",
            )
        )
        assert flushed == 10
        assert len(buffer.flushed_batches) == 1

    asyncio.run(run_scenario())

    # Query filtered results
    errors = buffer.query(service_id="srv-web", level="ERROR")
    assert len(errors) == 4
    for err in errors:
        assert err.level == "ERROR"

    infos = buffer.query(service_id="srv-web", level="INFO")
    assert len(infos) == 6
