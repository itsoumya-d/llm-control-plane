"""Tests for Ray Core distributed cluster executor, object store, and actor pool."""

import pytest

from llm_control_plane import (
    RayActorPool,
    RayActorState,
    RayClusterNode,
    RayCoreClusterExecutor,
    RayObjectRef,
    RayObjectStore,
    RayObjectStoreError,
    RayPlacementStrategy,
    RaySchedulingError,
)


class CounterActor:
    """Stateful worker actor for test verification."""

    def __init__(self, initial_count: int = 0):
        self.count = initial_count

    def increment(self, amount: int = 1) -> int:
        self.count += amount
        return self.count

    def get_count(self) -> int:
        return self.count


def add_numbers(a: int, b: int) -> int:
    return a + b


def multiply_numbers(a: int, b: int) -> int:
    return a * b


def test_object_store_put_get_and_stats():
    store = RayObjectStore(capacity_bytes=1024 * 1024)
    data = {"prompt": "Analyze cluster bottlenecks", "tokens": 42}

    ref = store.put(data, owner_node_id="node_1")
    assert isinstance(ref, RayObjectRef)
    assert ref.owner_node_id == "node_1"
    assert ref.size_bytes > 0

    retrieved = store.get(ref)
    assert retrieved == data
    # Ensure deepcopy isolation
    retrieved["tokens"] = 999
    assert store.get(ref)["tokens"] == 42

    stats = store.get_memory_stats()
    assert stats["active_objects"] == 1
    assert stats["pinned_objects"] == 0
    assert stats["used_bytes"] == ref.size_bytes


def test_object_store_pinning_and_lru_eviction():
    # Capacity 200 bytes
    store = RayObjectStore(capacity_bytes=200)

    # Put object 1 (estimated 80 bytes) and pin it
    ref1 = store.put("pinned_payload_alpha", owner_node_id="node_1", pinned=True, estimated_size=80)
    assert store.is_pinned(ref1)

    # Put object 2 (80 bytes) unpinned
    ref2 = store.put("unpinned_payload_beta", owner_node_id="node_1", pinned=False, estimated_size=80)
    assert not store.is_pinned(ref2)

    # Access ref1 so ref2 is the LRU object
    _ = store.get(ref1)

    # Put object 3 (80 bytes) -> should evict/spill unpinned ref2
    ref3 = store.put("unpinned_payload_gamma", owner_node_id="node_1", pinned=False, estimated_size=80)

    stats = store.get_memory_stats()
    assert stats["spilled_objects"] == 1

    # Spilled ref2 can still be retrieved transparently
    assert store.get(ref2) == "unpinned_payload_beta"
    assert store.get(ref1) == "pinned_payload_alpha"
    assert store.get(ref3) == "unpinned_payload_gamma"

    # Unpin ref1
    store.unpin(ref1)
    assert not store.is_pinned(ref1)


def test_object_store_out_of_memory_error():
    # Capacity 100 bytes
    store = RayObjectStore(capacity_bytes=100)
    # Put 90 bytes pinned
    _ = store.put("large_pinned_data", pinned=True, estimated_size=90)

    # Attempt to put another 50 bytes while all memory is pinned
    with pytest.raises(RayObjectStoreError, match="Out of object store memory"):
        store.put("excess_data", estimated_size=50)

    # Attempt to put object larger than total capacity
    with pytest.raises(RayObjectStoreError, match="exceeds total capacity"):
        store.put("huge_data", estimated_size=500)


def test_cluster_node_resource_accounting():
    node = RayClusterNode(node_id="node_gpu_1", resources={"CPU": 16.0, "GPU": 2.0, "MEM": 32768.0})

    req = {"CPU": 4.0, "GPU": 1.0}
    assert node.can_accommodate(req)
    assert node.allocate(req)

    avail = node.get_available_resources()
    assert avail["CPU"] == 12.0
    assert avail["GPU"] == 1.0

    # Allocate remaining GPU
    assert node.allocate({"GPU": 1.0})
    # Cannot allocate more GPU
    assert not node.can_accommodate({"GPU": 1.0})
    assert not node.allocate({"GPU": 1.0})

    # Release
    node.release(req)
    avail_after = node.get_available_resources()
    assert avail_after["CPU"] == 16.0
    assert avail_after["GPU"] == 1.0


def test_placement_group_pack_and_spread():
    executor = RayCoreClusterExecutor()
    executor.register_node("node_a", resources={"CPU": 8.0, "GPU": 2.0})
    executor.register_node("node_b", resources={"CPU": 8.0, "GPU": 2.0})

    # Test PACK strategy
    bundles_pack = [{"CPU": 2.0, "GPU": 1.0}, {"CPU": 2.0, "GPU": 1.0}]
    pg_pack = executor.create_placement_group(RayPlacementStrategy.PACK, bundles_pack)
    assert pg_pack.is_created
    # Both bundles packed on the same node
    assert len(set(pg_pack.assigned_nodes)) == 1

    # Test SPREAD strategy
    bundles_spread = [{"CPU": 2.0}, {"CPU": 2.0}]
    pg_spread = executor.create_placement_group(RayPlacementStrategy.SPREAD, bundles_spread)
    assert pg_spread.is_created
    # Spread across 2 different nodes
    assert len(set(pg_spread.assigned_nodes)) == 2


def test_placement_group_strict_failures():
    executor = RayCoreClusterExecutor()
    executor.register_node("node_small_1", resources={"CPU": 2.0})
    executor.register_node("node_small_2", resources={"CPU": 2.0})

    # STRICT_PACK requesting 2 bundles of 2.0 CPU each -> total 4.0 CPU needed on 1 node
    with pytest.raises(RaySchedulingError, match="Cannot satisfy STRICT_PACK"):
        executor.create_placement_group(
            RayPlacementStrategy.STRICT_PACK,
            [{"CPU": 2.0}, {"CPU": 2.0}],
        )

    # STRICT_SPREAD requesting 3 bundles with only 2 nodes
    with pytest.raises(RaySchedulingError, match="Cannot satisfy STRICT_SPREAD"):
        executor.create_placement_group(
            RayPlacementStrategy.STRICT_SPREAD,
            [{"CPU": 1.0}, {"CPU": 1.0}, {"CPU": 1.0}],
        )


def test_actor_pool_lifecycle_and_restart():
    pool = RayActorPool()
    handle = pool.create_actor(CounterActor, initial_count=10, max_restarts=2)
    assert handle.state == RayActorState.ALIVE
    handle.node_id = "worker_node_1"

    # Test method invocation
    res = handle.invoke("increment", 5)
    assert res == 15
    assert handle.invoke("get_count") == 15

    # Simulate node failure and automated actor restart
    restarted = pool.handle_node_failure("worker_node_1")
    assert handle.actor_id in restarted
    assert handle.restart_count == 1
    assert handle.state == RayActorState.ALIVE
    # Re-initialized to initial count
    assert handle.invoke("get_count") == 10

    # Exhaust restarts
    handle.restart()
    assert handle.restart_count == 2
    assert not handle.restart()
    assert handle.state == RayActorState.DEAD


def test_scheduler_data_locality_dispatch():
    executor = RayCoreClusterExecutor()
    executor.register_node("node_storage", resources={"CPU": 4.0})
    executor.register_node("node_compute", resources={"CPU": 4.0})

    # Put an initial object on node_storage
    data_ref = executor.object_store.put(100, owner_node_id="node_storage")

    # Dispatch task with data_ref as argument
    # Scheduler should prioritize node_storage where data_ref is resident
    out_ref = executor.submit_task(add_numbers, data_ref, 50)
    assert out_ref.owner_node_id == "node_storage"
    assert executor.object_store.get(out_ref) == 150


def test_scheduler_ray_wait_and_lineage_recovery():
    executor = RayCoreClusterExecutor()
    executor.register_node("node_main", resources={"CPU": 8.0})

    ref1 = executor.submit_task(add_numbers, 10, 20)
    ref2 = executor.submit_task(multiply_numbers, 5, 6)

    ready, not_ready = executor.ray_wait([ref1, ref2], num_returns=2)
    assert len(ready) == 2
    assert len(not_ready) == 0

    # Test lineage recovery: simulate object ref loss
    del executor.object_store._objects[ref1.object_id]
    assert ref1.object_id not in executor.object_store._objects

    # Recover using recorded lineage
    recovered_ref = executor.recover_object(ref1)
    assert executor.object_store.get(recovered_ref) == 30


def test_scheduler_capacity_exhaustion_error():
    executor = RayCoreClusterExecutor()
    executor.register_node("node_tiny", resources={"CPU": 1.0})

    with pytest.raises(RaySchedulingError, match="No cluster node can accommodate"):
        executor.submit_task(add_numbers, 1, 2, resource_requirements={"CPU": 4.0})


def test_cluster_status_and_heartbeat():
    executor = RayCoreClusterExecutor()
    executor.register_node("node_alpha", resources={"CPU": 4.0, "MEM": 8192.0})
    executor.heartbeat("node_alpha")

    status = executor.get_cluster_status()
    assert status["active_nodes"] == 1
    assert status["total_nodes"] == 1
    assert status["total_resources"]["CPU"] == 4.0

    executor.unregister_node("node_alpha")
    status_after = executor.get_cluster_status()
    assert status_after["active_nodes"] == 0
