"""Unit tests for Onehouse Kubernetes Lakehouse Infrastructure Operator."""

import pytest

from llm_control_plane.onehouse_k8s_lakehouse_operator import (
    LakehouseResourceLimits,
    LakehouseServiceSpec,
    OnehouseK8sLakehouseReconciler,
    ReconciliationState,
    TableServiceType,
)


@pytest.fixture
def lakehouse_spec():
    return LakehouseServiceSpec(
        service_id="telemetry-events-lakehouse",
        table_name="raw_telemetry_events",
        database="production_lake",
        service_type=TableServiceType.COMPACTION,
        min_workers=2,
        max_workers=8,
        resource_limits=LakehouseResourceLimits(cpu_cores=8.0, memory_gib=32.0),
        commit_backlog_threshold=100,
        parquet_skew_threshold_mb=512.0,
        checkpoint_storage_uri="s3://onehouse-test-checkpoints/hudi/",
    )


def test_reconciler_initialization(lakehouse_spec):
    reconciler = OnehouseK8sLakehouseReconciler(lakehouse_spec)
    assert reconciler.spec.service_id == "telemetry-events-lakehouse"
    assert reconciler.current_state == ReconciliationState.PENDING
    assert len(reconciler.workers) == 0


def test_reconciliation_cycle_provisions_min_workers(lakehouse_spec):
    reconciler = OnehouseK8sLakehouseReconciler(lakehouse_spec)
    status = reconciler.reconcile(current_backlog=10, file_skew_mb=50.0)

    assert status.state == ReconciliationState.RUNNING
    assert status.active_workers == 2
    assert status.reconciliation_epoch == 1
    assert len(status.conditions) >= 2


def test_compaction_trigger_and_autoscaling(lakehouse_spec):
    reconciler = OnehouseK8sLakehouseReconciler(lakehouse_spec)
    # Trigger compaction via backlog > 100
    status = reconciler.reconcile(current_backlog=250, file_skew_mb=100.0)

    assert status.state == ReconciliationState.COMPACTING
    assert status.active_workers > 2
    assert status.active_workers <= lakehouse_spec.max_workers


def test_compaction_trigger_via_file_skew(lakehouse_spec):
    reconciler = OnehouseK8sLakehouseReconciler(lakehouse_spec)
    # Trigger compaction via file skew > 512 MB
    status = reconciler.reconcile(current_backlog=20, file_skew_mb=1024.0)

    assert status.state == ReconciliationState.COMPACTING
    assert status.active_workers >= 4


def test_compaction_succeeded_state_transition(lakehouse_spec):
    reconciler = OnehouseK8sLakehouseReconciler(lakehouse_spec)
    reconciler.reconcile(current_backlog=200, file_skew_mb=100.0)
    assert reconciler.current_state == ReconciliationState.COMPACTING

    # Backlog resolved to 0
    status = reconciler.reconcile(current_backlog=0, file_skew_mb=50.0)
    assert status.state == ReconciliationState.SUCCEEDED
    assert status.completed_commits == 1


def test_spot_node_preemption_handling(lakehouse_spec):
    reconciler = OnehouseK8sLakehouseReconciler(lakehouse_spec)
    reconciler.reconcile(current_backlog=50, file_skew_mb=50.0)

    target_node = list(reconciler.workers.values())[0].node_name
    result = reconciler.handle_spot_preemption(target_node)

    assert result["node_name"] == target_node
    assert result["drained_pod_count"] >= 1
    assert len(result["checkpoints"]) >= 1

    # Ensure minimum active workers were preserved after compensation
    active = [w for w in reconciler.workers.values() if not w.is_draining]
    assert len(active) >= lakehouse_spec.min_workers


def test_compute_autoscaling_demand_boundaries(lakehouse_spec):
    reconciler = OnehouseK8sLakehouseReconciler(lakehouse_spec)

    # Minimum bound
    low_demand = reconciler.compute_autoscaling_demand(backlog=10, file_skew_mb=50.0)
    assert low_demand == lakehouse_spec.min_workers

    # Maximum bound cap
    extreme_demand = reconciler.compute_autoscaling_demand(backlog=10000, file_skew_mb=50000.0)
    assert extreme_demand == lakehouse_spec.max_workers
