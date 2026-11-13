"""Ray Core distributed cluster executor, object store, and actor pool.

Provides distributed task scheduling, data-locality dispatching, Plasma-style
shared memory object store emulation with LRU eviction and pinning, placement
group allocation, and fault-tolerant actor lifecycle management for high-throughput
distributed workloads and LLM inference clusters.
"""

from __future__ import annotations

import copy
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


class RayActorState(str, Enum):
    """Lifecycle states of a Ray worker actor."""

    CREATED = "CREATED"
    ALIVE = "ALIVE"
    BUSY = "BUSY"
    RESTARTING = "RESTARTING"
    DEAD = "DEAD"


class RayPlacementStrategy(str, Enum):
    """Placement group scheduling strategies."""

    PACK = "PACK"
    SPREAD = "SPREAD"
    STRICT_PACK = "STRICT_PACK"
    STRICT_SPREAD = "STRICT_SPREAD"


@dataclass(frozen=True)
class RayObjectRef:
    """Immutable reference to an object stored in the distributed object store."""

    object_id: str = field(default_factory=lambda: f"obj_{uuid.uuid4().hex[:12]}")
    owner_node_id: str = "node_local"
    size_bytes: int = 0

    def __repr__(self) -> str:
        return f"RayObjectRef(id={self.object_id}, owner={self.owner_node_id}, size={self.size_bytes}B)"


@dataclass
class RayTaskSpec:
    """Specification of a distributed task to be scheduled and executed."""

    task_id: str = field(default_factory=lambda: f"task_{uuid.uuid4().hex[:12]}")
    func: Optional[Callable[..., Any]] = None
    args: Tuple[Any, ...] = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    resource_requirements: Dict[str, float] = field(default_factory=lambda: {"CPU": 1.0})
    target_actor_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)


class RayObjectStoreError(Exception):
    """Raised when an object store operation fails."""


class RaySchedulingError(Exception):
    """Raised when a task or placement group cannot be scheduled."""


class RayObjectStore:
    """In-memory object store with Plasma shared-memory semantics.

    Features:
    - Zero-copy reference tracking with object IDs.
    - Reference pinning to protect active working sets from eviction.
    - LRU eviction of unpinned objects when memory threshold is reached.
    - Spilling simulation to external persistent storage.
    """

    def __init__(self, capacity_bytes: int = 64 * 1024 * 1024):
        self._capacity_bytes = capacity_bytes
        self._used_bytes = 0
        self._objects: Dict[str, Any] = {}
        self._metadata: Dict[str, RayObjectRef] = {}
        self._pinned: Set[str] = set()
        self._access_order: List[str] = []
        self._spilled: Dict[str, Any] = {}
        self._lock = threading.RLock()

    @property
    def capacity_bytes(self) -> int:
        return self._capacity_bytes

    @property
    def used_bytes(self) -> int:
        with self._lock:
            return self._used_bytes

    def put(
        self,
        value: Any,
        owner_node_id: str = "node_local",
        pinned: bool = False,
        estimated_size: Optional[int] = None,
    ) -> RayObjectRef:
        """Stores a Python object in the local object store and returns a reference."""
        with self._lock:
            size = estimated_size or max(len(str(value).encode("utf-8")), 64)
            if size > self._capacity_bytes:
                raise RayObjectStoreError(
                    f"Object size {size}B exceeds total capacity {self._capacity_bytes}B"
                )

            self._evict_if_needed(size)

            ref = RayObjectRef(
                object_id=f"obj_{uuid.uuid4().hex[:12]}",
                owner_node_id=owner_node_id,
                size_bytes=size,
            )

            self._objects[ref.object_id] = copy.deepcopy(value)
            self._metadata[ref.object_id] = ref
            self._used_bytes += size
            self._access_order.append(ref.object_id)

            if pinned:
                self._pinned.add(ref.object_id)

            return ref

    def get(self, ref: RayObjectRef, timeout: Optional[float] = None) -> Any:
        """Retrieves the value of a referenced object from memory or spilled storage."""
        start_time = time.time()
        while True:
            with self._lock:
                oid = ref.object_id
                if oid in self._objects:
                    # Update LRU ordering
                    if oid in self._access_order:
                        self._access_order.remove(oid)
                    self._access_order.append(oid)
                    return copy.deepcopy(self._objects[oid])

                if oid in self._spilled:
                    # Retrieve from spill storage
                    return copy.deepcopy(self._spilled[oid])

            if timeout is not None and (time.time() - start_time) >= timeout:
                raise TimeoutError(f"Timed out waiting for object ref {ref.object_id}")
            time.sleep(0.01)

    def pin(self, ref: RayObjectRef) -> None:
        """Pins an object to prevent it from being evicted."""
        with self._lock:
            if ref.object_id in self._objects:
                self._pinned.add(ref.object_id)

    def unpin(self, ref: RayObjectRef) -> None:
        """Unpins an object so it may be evicted when memory pressure occurs."""
        with self._lock:
            self._pinned.discard(ref.object_id)

    def is_pinned(self, ref: RayObjectRef) -> bool:
        with self._lock:
            return ref.object_id in self._pinned

    def _evict_if_needed(self, required_bytes: int) -> None:
        """Evicts unpinned objects in LRU order to make room for required bytes."""
        while (self._used_bytes + required_bytes) > self._capacity_bytes:
            evicted_any = False
            for oid in list(self._access_order):
                if oid not in self._pinned:
                    ref = self._metadata[oid]
                    # Simulate spilling to persistent storage before eviction
                    self._spilled[oid] = self._objects.pop(oid)
                    self._used_bytes -= ref.size_bytes
                    self._access_order.remove(oid)
                    evicted_any = True
                    break

            if not evicted_any:
                raise RayObjectStoreError(
                    f"Out of object store memory: cannot free {required_bytes}B; "
                    f"all {len(self._pinned)} objects are pinned"
                )

    def get_memory_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "capacity_bytes": self._capacity_bytes,
                "used_bytes": self._used_bytes,
                "free_bytes": self._capacity_bytes - self._used_bytes,
                "active_objects": len(self._objects),
                "pinned_objects": len(self._pinned),
                "spilled_objects": len(self._spilled),
            }


@dataclass
class RayClusterNode:
    """Represents a physical or virtual compute node in the Ray cluster."""

    node_id: str
    resources: Dict[str, float] = field(default_factory=lambda: {"CPU": 8.0, "GPU": 0.0, "MEM": 16384.0})
    allocated: Dict[str, float] = field(default_factory=lambda: {"CPU": 0.0, "GPU": 0.0, "MEM": 0.0})
    alive: bool = True
    last_heartbeat: float = field(default_factory=time.time)

    def can_accommodate(self, requirements: Dict[str, float]) -> bool:
        if not self.alive:
            return False
        for res, required_val in requirements.items():
            available = self.resources.get(res, 0.0) - self.allocated.get(res, 0.0)
            if available < required_val:
                return False
        return True

    def allocate(self, requirements: Dict[str, float]) -> bool:
        if not self.can_accommodate(requirements):
            return False
        for res, val in requirements.items():
            self.allocated[res] = self.allocated.get(res, 0.0) + val
        return True

    def release(self, requirements: Dict[str, float]) -> None:
        for res, val in requirements.items():
            self.allocated[res] = max(0.0, self.allocated.get(res, 0.0) - val)

    def get_available_resources(self) -> Dict[str, float]:
        return {
            res: max(0.0, total - self.allocated.get(res, 0.0))
            for res, total in self.resources.items()
        }


@dataclass
class RayPlacementGroup:
    """Gang-scheduling bundle abstraction for actor and worker placement."""

    group_id: str
    strategy: RayPlacementStrategy
    bundles: List[Dict[str, float]]
    assigned_nodes: List[str] = field(default_factory=list)
    is_created: bool = False


class RayActorHandle:
    """Client proxy for interacting with a stateful remote actor."""

    def __init__(
        self,
        actor_id: str,
        actor_class: type,
        init_args: Tuple[Any, ...],
        init_kwargs: Dict[str, Any],
        resources: Dict[str, float],
        max_restarts: int = 3,
    ):
        self.actor_id = actor_id
        self.actor_class = actor_class
        self.init_args = init_args
        self.init_kwargs = init_kwargs
        self.resources = resources
        self.max_restarts = max_restarts
        self.restart_count = 0
        self.state = RayActorState.CREATED
        self.node_id: Optional[str] = None
        self.instance: Any = None
        self._lock = threading.RLock()

    def instantiate(self) -> None:
        with self._lock:
            self.instance = self.actor_class(*self.init_args, **self.init_kwargs)
            self.state = RayActorState.ALIVE

    def invoke(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            if self.state != RayActorState.ALIVE:
                raise RuntimeError(f"Actor {self.actor_id} is in invalid state {self.state}")
            self.state = RayActorState.BUSY
            try:
                method = getattr(self.instance, method_name)
                result = method(*args, **kwargs)
                return result
            finally:
                self.state = RayActorState.ALIVE

    def restart(self) -> bool:
        with self._lock:
            if self.restart_count >= self.max_restarts:
                self.state = RayActorState.DEAD
                return False
            self.restart_count += 1
            self.state = RayActorState.RESTARTING
            self.instantiate()
            return True


class RayActorPool:
    """Manages actor lifecycle, placement, leasing, and automated failure recovery."""

    def __init__(self):
        self._actors: Dict[str, RayActorHandle] = {}
        self._lock = threading.RLock()

    def create_actor(
        self,
        actor_class: type,
        *init_args: Any,
        resources: Optional[Dict[str, float]] = None,
        max_restarts: int = 3,
        **init_kwargs: Any,
    ) -> RayActorHandle:
        with self._lock:
            actor_id = f"actor_{uuid.uuid4().hex[:10]}"
            handle = RayActorHandle(
                actor_id=actor_id,
                actor_class=actor_class,
                init_args=init_args,
                init_kwargs=init_kwargs,
                resources=resources or {"CPU": 1.0},
                max_restarts=max_restarts,
            )
            handle.instantiate()
            self._actors[actor_id] = handle
            return handle

    def get_actor(self, actor_id: str) -> Optional[RayActorHandle]:
        with self._lock:
            return self._actors.get(actor_id)

    def list_actors(self) -> List[RayActorHandle]:
        with self._lock:
            return list(self._actors.values())

    def handle_node_failure(self, dead_node_id: str) -> List[str]:
        """Restarts actors assigned to a failed node on surviving nodes."""
        restarted_actor_ids: List[str] = []
        with self._lock:
            for handle in self._actors.values():
                if handle.node_id == dead_node_id and handle.state != RayActorState.DEAD:
                    success = handle.restart()
                    if success:
                        restarted_actor_ids.append(handle.actor_id)
        return restarted_actor_ids


class RayCoreClusterExecutor:
    """Distributed scheduling and task execution engine for Ray Core workloads.

    Features:
    - Data-locality aware dispatching (prefers nodes with input object references).
    - Decentralized scheduling with cluster resource accounting.
    - Placement group reservations (PACK, SPREAD).
    - Asynchronous task futures with ray_wait semantics.
    - Task lineage tracking for deterministic fault recovery.
    """

    def __init__(self, object_store: Optional[RayObjectStore] = None):
        self.object_store = object_store or RayObjectStore()
        self.actor_pool = RayActorPool()
        self._nodes: Dict[str, RayClusterNode] = {}
        self._placement_groups: Dict[str, RayPlacementGroup] = {}
        self._lineage: Dict[str, RayTaskSpec] = {}
        self._lock = threading.RLock()

    def register_node(
        self,
        node_id: str,
        resources: Optional[Dict[str, float]] = None,
    ) -> RayClusterNode:
        with self._lock:
            node = RayClusterNode(
                node_id=node_id,
                resources=resources or {"CPU": 8.0, "GPU": 0.0, "MEM": 16384.0},
            )
            self._nodes[node_id] = node
            return node

    def unregister_node(self, node_id: str) -> None:
        with self._lock:
            if node_id in self._nodes:
                self._nodes[node_id].alive = False
                self.actor_pool.handle_node_failure(node_id)

    def heartbeat(self, node_id: str) -> None:
        with self._lock:
            if node_id in self._nodes:
                self._nodes[node_id].last_heartbeat = time.time()
                self._nodes[node_id].alive = True

    def create_placement_group(
        self,
        strategy: RayPlacementStrategy,
        bundles: List[Dict[str, float]],
    ) -> RayPlacementGroup:
        """Allocates a placement group across cluster nodes according to strategy."""
        with self._lock:
            group_id = f"pg_{uuid.uuid4().hex[:10]}"
            pg = RayPlacementGroup(group_id=group_id, strategy=strategy, bundles=bundles)

            assigned_nodes: List[str] = []
            alive_nodes = [n for n in self._nodes.values() if n.alive]

            if strategy in (RayPlacementStrategy.PACK, RayPlacementStrategy.STRICT_PACK):
                # Attempt to pack all bundles on the first node with sufficient capacity
                packed_node = None
                for node in alive_nodes:
                    combined_req: Dict[str, float] = {}
                    for b in bundles:
                        for r, v in b.items():
                            combined_req[r] = combined_req.get(r, 0.0) + v
                    if node.can_accommodate(combined_req):
                        packed_node = node
                        break

                if packed_node:
                    for b in bundles:
                        packed_node.allocate(b)
                        assigned_nodes.append(packed_node.node_id)
                elif strategy == RayPlacementStrategy.STRICT_PACK:
                    raise RaySchedulingError("Cannot satisfy STRICT_PACK: no single node has capacity for all bundles")
                else:
                    # Fallback to greedy per-bundle allocation
                    for b in bundles:
                        scheduled = False
                        for node in alive_nodes:
                            if node.allocate(b):
                                assigned_nodes.append(node.node_id)
                                scheduled = True
                                break
                        if not scheduled:
                            raise RaySchedulingError("Failed to allocate bundles under PACK strategy")

            elif strategy in (RayPlacementStrategy.SPREAD, RayPlacementStrategy.STRICT_SPREAD):
                used_nodes: Set[str] = set()
                for b in bundles:
                    candidate = None
                    for node in alive_nodes:
                        if node.node_id not in used_nodes and node.can_accommodate(b):
                            candidate = node
                            break
                    if candidate:
                        candidate.allocate(b)
                        assigned_nodes.append(candidate.node_id)
                        used_nodes.add(candidate.node_id)
                    elif strategy == RayPlacementStrategy.STRICT_SPREAD:
                        raise RaySchedulingError("Cannot satisfy STRICT_SPREAD: insufficient unique nodes")
                    else:
                        # Fallback to any node with capacity
                        for node in alive_nodes:
                            if node.allocate(b):
                                assigned_nodes.append(node.node_id)
                                break

            pg.assigned_nodes = assigned_nodes
            pg.is_created = True
            self._placement_groups[group_id] = pg
            return pg

    def submit_task(
        self,
        func: Callable[..., Any],
        *args: Any,
        resource_requirements: Optional[Dict[str, float]] = None,
        preferred_node_id: Optional[str] = None,
        **kwargs: Any,
    ) -> RayObjectRef:
        """Dispatches a task with data-locality awareness and returns a RayObjectRef."""
        with self._lock:
            reqs = resource_requirements or {"CPU": 1.0}
            task = RayTaskSpec(
                func=func,
                args=args,
                kwargs=kwargs,
                resource_requirements=reqs,
            )

            # Data-locality optimization: check if any arg is a RayObjectRef resident on a node
            locality_node_id = preferred_node_id
            if not locality_node_id:
                for arg in args:
                    if isinstance(arg, RayObjectRef) and arg.owner_node_id in self._nodes:
                        if self._nodes[arg.owner_node_id].can_accommodate(reqs):
                            locality_node_id = arg.owner_node_id
                            break

            target_node = None
            if locality_node_id and locality_node_id in self._nodes:
                candidate = self._nodes[locality_node_id]
                if candidate.can_accommodate(reqs):
                    target_node = candidate

            if not target_node:
                # Select node with highest remaining CPU
                candidates = [n for n in self._nodes.values() if n.can_accommodate(reqs)]
                if not candidates:
                    raise RaySchedulingError(f"No cluster node can accommodate task requirements {reqs}")
                candidates.sort(key=lambda n: n.get_available_resources().get("CPU", 0.0), reverse=True)
                target_node = candidates[0]

            # Allocate resources
            target_node.allocate(reqs)

            # Resolve RayObjectRef arguments from object store
            resolved_args = []
            for arg in args:
                if isinstance(arg, RayObjectRef):
                    resolved_args.append(self.object_store.get(arg))
                else:
                    resolved_args.append(arg)

            resolved_kwargs = {}
            for k, v in kwargs.items():
                if isinstance(v, RayObjectRef):
                    resolved_kwargs[k] = self.object_store.get(v)
                else:
                    resolved_kwargs[k] = v

            try:
                # Execute task
                result = func(*resolved_args, **resolved_kwargs)
            finally:
                target_node.release(reqs)

            # Store result in object store on target node
            out_ref = self.object_store.put(result, owner_node_id=target_node.node_id)
            self._lineage[out_ref.object_id] = task
            return out_ref

    def ray_wait(
        self,
        object_refs: List[RayObjectRef],
        num_returns: int = 1,
        timeout: Optional[float] = None,
    ) -> Tuple[List[RayObjectRef], List[RayObjectRef]]:
        """Separates ready object references from not-ready references."""
        start = time.time()
        ready: List[RayObjectRef] = []
        not_ready: List[RayObjectRef] = list(object_refs)

        while len(ready) < num_returns and not_ready:
            for ref in list(not_ready):
                with self.object_store._lock:
                    if ref.object_id in self.object_store._objects or ref.object_id in self.object_store._spilled:
                        ready.append(ref)
                        not_ready.remove(ref)
                        if len(ready) >= num_returns:
                            break

            if len(ready) >= num_returns:
                break

            if timeout is not None and (time.time() - start) >= timeout:
                break
            time.sleep(0.01)

        return ready, not_ready

    def recover_object(self, ref: RayObjectRef) -> RayObjectRef:
        """Reconstructs a lost object ref using its recorded lineage."""
        with self._lock:
            oid = ref.object_id
            if oid not in self._lineage:
                raise RayObjectStoreError(f"Cannot recover object {oid}: no lineage found")

            task = self._lineage[oid]
            if not task.func:
                raise RayObjectStoreError(f"Cannot recover object {oid}: task func is missing")

            new_ref = self.submit_task(
                task.func,
                *task.args,
                resource_requirements=task.resource_requirements,
                **task.kwargs,
            )
            return new_ref

    def get_cluster_status(self) -> Dict[str, Any]:
        with self._lock:
            total_res: Dict[str, float] = {}
            allocated_res: Dict[str, float] = {}
            active_node_count = 0

            for node in self._nodes.values():
                if node.alive:
                    active_node_count += 1
                    for r, val in node.resources.items():
                        total_res[r] = total_res.get(r, 0.0) + val
                    for r, val in node.allocated.items():
                        allocated_res[r] = allocated_res.get(r, 0.0) + val

            return {
                "active_nodes": active_node_count,
                "total_nodes": len(self._nodes),
                "total_resources": total_res,
                "allocated_resources": allocated_res,
                "placement_groups": len(self._placement_groups),
                "object_store": self.object_store.get_memory_stats(),
            }
