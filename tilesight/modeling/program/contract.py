"""Public description contract for the unified TileSight/TileLang interface.

This module defines the typed program description: primitives
(``Load``/``Store``/``Compute``/``Sync``), regions (``Sequence``/``Loop``/
``Pipeline``), tile accesses, launch work units, and ``Program``.

Every object is a frozen dataclass.  Validation runs in ``__post_init__`` and
errors always name the offending object path, field, and value.  Nothing in
this module knows about costs, caches, or architectures; those live in the
sibling analysis modules.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence as SeqT, Tuple, Union

from ..errors import ModelingValidationError
from ..ir import Timing, freeze_attrs, validate_identifier
from ..ops.schema import SemanticOpSpec, is_semantic_op

PROGRAM_SCHEMA_VERSION = "tilesight.program/1"

PUBLIC_STORAGES = ("global", "smem", "register", "tmem")
INTERNAL_STORAGE = {"global": "ddr", "smem": "smem", "register": "register", "tmem": "tmem"}
MOVE_ENGINES = ("tma", "cp_async", "ldst", "unbound")
COMPUTE_ENGINES = ("tensor", "cuda", "sfu")
SYNC_KINDS = ("barrier", "wait", "arrive")
EVENT_KINDS = ("start", "done")
ACTOR_ORDERS = ("issue", "completion")
LOOP_SITES = ("prologue", "body", "epilogue")
II_MODES = ("inherit", "resource_ii", "periodic_best", "periodic_worst")
DISPATCH_ORDERS = ("linear_block_id",)
ARCH_CAPABILITIES = ("tmem", "tcgen05", "wgmma", "l1_5")


class ContractError(ModelingValidationError):
    """Raised when a typed description violates the public contract."""


def _fail(path: str, field_name: str, value: Any, message: str) -> "ContractError":
    return ContractError(
        "%s.%s=%r: %s" % (path, field_name, value, message)
    )


def _ident(value: Any, path: str, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise _fail(path, field_name, value, "must be a non-empty string")
    if "/" in value:
        raise _fail(path, field_name, value, "must not contain '/'")
    return value


def _nonneg_int(value: Any, path: str, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise _fail(path, field_name, value, "must be a non-negative integer")
    return value


def _pos_int(value: Any, path: str, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise _fail(path, field_name, value, "must be a positive integer")
    return value


def _nonneg_float(value: Any, path: str, field_name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise _fail(path, field_name, value, "must be numeric")
    if not math.isfinite(result) or result < 0.0:
        raise _fail(path, field_name, value, "must be finite and non-negative")
    return result


def _pos_float(value: Any, path: str, field_name: str) -> float:
    result = _nonneg_float(value, path, field_name)
    if result <= 0.0:
        raise _fail(path, field_name, value, "must be positive")
    return result


def _shape(value: Any, path: str, field_name: str) -> Tuple[int, ...]:
    try:
        result = tuple(value)
    except TypeError:
        raise _fail(path, field_name, value, "must be a sequence of positive integers")
    if not result:
        raise _fail(path, field_name, value, "must not be empty")
    for item in result:
        if not isinstance(item, int) or isinstance(item, bool) or item <= 0:
            raise _fail(path, field_name, value, "must contain positive integers")
    return result


def _product(values: SeqT[int]) -> int:
    result = 1
    for item in values:
        result *= item
    return result


# ---------------------------------------------------------------------------
# Tile access and identity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Projection:
    """Tile identity keeps the named axes; projected-away axes reuse the tile.

    ``work_axes`` may name launch work axes and loop ids.  Order is kept as
    written so the identity key is stable and human-readable.
    """

    work_axes: Tuple[str, ...]

    def __post_init__(self) -> None:
        axes = tuple(self.work_axes)
        for axis in axes:
            _ident(axis, "Projection", "work_axes")
        if len(axes) != len(set(axes)):
            raise _fail("Projection", "work_axes", axes, "axes must be unique")
        object.__setattr__(self, "work_axes", axes)

    def canonical(self) -> Dict[str, Any]:
        return {"kind": "projection", "work_axes": list(self.work_axes)}


@dataclass(frozen=True)
class TileAccess:
    """One tile touched by every execution of the owning load/store.

    ``tile_shape`` is the allocated tile.  ``valid_shape`` is the masked
    effective extent (defaults to the tile).  The logical payload is derived
    from the valid extent; ``transaction_bytes`` defaults to the full tile
    bytes actually moved.  ``tensor_shape`` optionally bounds the access.
    """

    tensor: str
    index_map: Projection
    tile_shape: Tuple[int, ...]
    element_bytes: float
    valid_shape: Optional[Tuple[int, ...]] = None
    tensor_shape: Optional[Tuple[int, ...]] = None
    transaction_bytes: Optional[float] = None
    allocation_bytes: Optional[float] = None
    access_id: Optional[str] = None

    def __post_init__(self) -> None:
        path = "TileAccess(%s)" % self.tensor
        _ident(self.tensor, path, "tensor")
        if not isinstance(self.index_map, Projection):
            raise _fail(path, "index_map", self.index_map, "must be Projection")
        shape = _shape(self.tile_shape, path, "tile_shape")
        object.__setattr__(self, "tile_shape", shape)
        object.__setattr__(
            self, "element_bytes", _pos_float(self.element_bytes, path, "element_bytes")
        )
        if self.valid_shape is not None:
            valid = _shape(self.valid_shape, path, "valid_shape")
            if len(valid) != len(shape) or any(v > t for v, t in zip(valid, shape)):
                raise _fail(path, "valid_shape", valid, "must fit inside tile_shape")
            object.__setattr__(self, "valid_shape", valid)
        if self.tensor_shape is not None:
            tensor_shape = _shape(self.tensor_shape, path, "tensor_shape")
            if len(tensor_shape) != len(shape):
                raise _fail(path, "tensor_shape", tensor_shape, "rank must equal tile rank")
            if any(v > t for v, t in zip(self.effective_shape, tensor_shape)):
                raise _fail(
                    path, "valid_shape", self.effective_shape,
                    "effective access exceeds tensor_shape",
                )
            object.__setattr__(self, "tensor_shape", tensor_shape)
        if self.transaction_bytes is not None:
            transaction = _pos_float(self.transaction_bytes, path, "transaction_bytes")
            if transaction + 1.0e-9 < self.payload_bytes:
                raise _fail(
                    path, "transaction_bytes", transaction,
                    "cannot be smaller than the logical payload",
                )
            object.__setattr__(self, "transaction_bytes", transaction)
        if self.allocation_bytes is not None:
            allocation = _pos_float(self.allocation_bytes, path, "allocation_bytes")
            if allocation + 1.0e-9 < self.tile_bytes:
                raise _fail(
                    path, "allocation_bytes", allocation,
                    "cannot be smaller than the tile bytes",
                )
            object.__setattr__(self, "allocation_bytes", allocation)
        if self.access_id is not None:
            _ident(self.access_id, path, "access_id")

    @property
    def effective_shape(self) -> Tuple[int, ...]:
        return self.tile_shape if self.valid_shape is None else self.valid_shape

    @property
    def tile_bytes(self) -> float:
        return float(_product(self.tile_shape)) * self.element_bytes

    @property
    def payload_bytes(self) -> float:
        """Logical (mask-effective) bytes of one execution."""

        return float(_product(self.effective_shape)) * self.element_bytes

    @property
    def executed_bytes(self) -> float:
        """Bytes actually moved by one execution."""

        return self.tile_bytes if self.transaction_bytes is None else self.transaction_bytes

    @property
    def effective_allocation_bytes(self) -> float:
        return self.tile_bytes if self.allocation_bytes is None else self.allocation_bytes

    def canonical(self) -> Dict[str, Any]:
        return {
            "tensor": self.tensor,
            "index_map": self.index_map.canonical(),
            "tile_shape": list(self.tile_shape),
            "element_bytes": self.element_bytes,
            "valid_shape": None if self.valid_shape is None else list(self.valid_shape),
            "tensor_shape": None if self.tensor_shape is None else list(self.tensor_shape),
            "transaction_bytes": self.transaction_bytes,
            "allocation_bytes": self.allocation_bytes,
            "access_id": self.access_id,
        }


# ---------------------------------------------------------------------------
# Events, buffers, and primitives
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EventRef:
    """The ``start`` or ``done`` event of an op inside the same pipeline."""

    op_id: str
    kind: str = "done"

    def __post_init__(self) -> None:
        _ident(self.op_id, "EventRef", "op_id")
        if self.kind not in EVENT_KINDS:
            raise _fail("EventRef(%s)" % self.op_id, "kind", self.kind,
                        "must be one of %s" % ", ".join(EVENT_KINDS))


def start(op: Union[str, "Op"]) -> EventRef:
    return EventRef(op if isinstance(op, str) else op.op_id, "start")


def done(op: Union[str, "Op"]) -> EventRef:
    return EventRef(op if isinstance(op, str) else op.op_id, "done")


@dataclass(frozen=True)
class Buffer:
    """Named on-chip storage owned by one pipeline."""

    name: str
    storage: str
    shape: Tuple[int, ...] = field(default_factory=tuple)
    dtype: str = ""
    slots: int = 1
    execution_scope: str = "cta"

    def __post_init__(self) -> None:
        path = "Buffer(%s)" % self.name
        _ident(self.name, path, "name")
        if self.storage not in ("smem", "register", "tmem", "fragment"):
            raise _fail(path, "storage", self.storage,
                        "must be smem, register, tmem, or fragment")
        object.__setattr__(self, "shape", tuple(self.shape))
        _pos_int(self.slots, path, "slots")
        _ident(self.execution_scope, path, "execution_scope")


class Op:
    """Marker base for the four primitives."""

    op_id: str
    actor: str
    timing: Optional[Timing]
    metadata: Tuple[Tuple[str, Any], ...]

    @property
    def kind(self) -> str:
        raise NotImplementedError


def _op_common(op: "Op", path: str) -> None:
    _ident(op.op_id, path, "op_id")
    _ident(op.actor, path, "actor")
    if op.timing is not None and not isinstance(op.timing, Timing):
        raise _fail(path, "timing", op.timing, "must be Timing or None")
    object.__setattr__(op, "metadata", freeze_attrs(op.metadata))
    object.__setattr__(op, "reads", tuple(op.reads))
    object.__setattr__(op, "writes", tuple(op.writes))
    for name in op.reads + op.writes:
        _ident(name, path, "reads/writes")


@dataclass(frozen=True)
class Load(Op):
    """Move data from ``source`` to ``destination`` with one engine."""

    op_id: str
    source: str
    destination: str
    engine: str = "unbound"
    access: Optional[TileAccess] = None
    bytes: Optional[float] = None
    actor: str = "default"
    reads: Tuple[str, ...] = field(default_factory=tuple)
    writes: Tuple[str, ...] = field(default_factory=tuple)
    timing: Optional[Timing] = None
    latency_extra_s: Optional[float] = None
    metadata: Tuple[Tuple[str, Any], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _validate_move(self, "Load")

    @property
    def kind(self) -> str:
        return "load"


@dataclass(frozen=True)
class Store(Op):
    op_id: str
    source: str
    destination: str
    engine: str = "unbound"
    access: Optional[TileAccess] = None
    bytes: Optional[float] = None
    actor: str = "default"
    reads: Tuple[str, ...] = field(default_factory=tuple)
    writes: Tuple[str, ...] = field(default_factory=tuple)
    timing: Optional[Timing] = None
    latency_extra_s: Optional[float] = None
    metadata: Tuple[Tuple[str, Any], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _validate_move(self, "Store")

    @property
    def kind(self) -> str:
        return "store"


def _validate_move(op: Any, label: str) -> None:
    path = "%s(%s)" % (label, op.op_id)
    _op_common(op, path)
    for name in ("source", "destination"):
        value = getattr(op, name)
        if value not in PUBLIC_STORAGES:
            raise _fail(path, name, value, "must be one of %s" % ", ".join(PUBLIC_STORAGES))
    if op.source == op.destination:
        raise _fail(path, "destination", op.destination, "must differ from source")
    if op.engine not in MOVE_ENGINES:
        raise _fail(path, "engine", op.engine, "must be one of %s" % ", ".join(MOVE_ENGINES))
    if op.access is not None and not isinstance(op.access, TileAccess):
        raise _fail(path, "access", op.access, "must be TileAccess or None")
    if op.bytes is not None:
        object.__setattr__(op, "bytes", _pos_float(op.bytes, path, "bytes"))
    if op.access is None and op.bytes is None:
        raise _fail(path, "bytes", None, "needs a TileAccess or explicit bytes")
    if op.access is not None and op.bytes is not None:
        raise _fail(path, "bytes", op.bytes, "explicit bytes conflict with TileAccess")
    if op.latency_extra_s is not None:
        object.__setattr__(
            op, "latency_extra_s", _nonneg_float(op.latency_extra_s, path, "latency_extra_s")
        )
    if label == "Load" and op.destination == "global":
        raise _fail(path, "destination", op.destination, "a load cannot target global")
    if label == "Store" and op.source == "global":
        raise _fail(path, "source", op.source, "a store cannot read from global")


def move_payload_bytes(op: Union[Load, Store]) -> float:
    """Logical payload bytes of one execution."""

    return op.bytes if op.access is None else op.access.payload_bytes


def move_executed_bytes(op: Union[Load, Store]) -> float:
    return op.bytes if op.access is None else op.access.executed_bytes


def touches_global(op: Op) -> bool:
    return isinstance(op, (Load, Store)) and "global" in (op.source, op.destination)


@dataclass(frozen=True)
class Compute(Op):
    """One execution of a semantic operation on one engine.

    ``spec`` describes the *executed* (padded) work of one execution.
    ``effective_fraction`` scales it to mask-effective useful work.
    """

    op_id: str
    spec: SemanticOpSpec
    engine: Optional[str] = None
    effective_fraction: float = 1.0
    actor: str = "default"
    reads: Tuple[str, ...] = field(default_factory=tuple)
    writes: Tuple[str, ...] = field(default_factory=tuple)
    timing: Optional[Timing] = None
    latency_extra_s: Optional[float] = None
    metadata: Tuple[Tuple[str, Any], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        path = "Compute(%s)" % self.op_id
        _op_common(self, path)
        if not is_semantic_op(self.spec):
            raise _fail(path, "spec", self.spec, "must be a semantic op spec")
        if self.engine is not None and self.engine not in COMPUTE_ENGINES:
            raise _fail(path, "engine", self.engine,
                        "must be one of %s" % ", ".join(COMPUTE_ENGINES))
        fraction = _nonneg_float(self.effective_fraction, path, "effective_fraction")
        if fraction > 1.0:
            raise _fail(path, "effective_fraction", fraction, "must be in [0, 1]")
        object.__setattr__(self, "effective_fraction", fraction)
        if self.latency_extra_s is not None:
            object.__setattr__(
                self, "latency_extra_s",
                _nonneg_float(self.latency_extra_s, path, "latency_extra_s"),
            )

    @property
    def kind(self) -> str:
        return "compute"

    @property
    def executed_flops(self) -> float:
        return float(sum(item.semantic_flops for item in self.spec.work_items))

    @property
    def useful_flops(self) -> float:
        return self.executed_flops * self.effective_fraction


@dataclass(frozen=True)
class Sync(Op):
    """barrier/wait/arrive; produces dependencies, zero cost by default."""

    op_id: str
    sync_kind: str = "barrier"
    actor: str = "default"
    participants: Tuple[str, ...] = field(default_factory=tuple)
    reads: Tuple[str, ...] = field(default_factory=tuple)
    writes: Tuple[str, ...] = field(default_factory=tuple)
    timing: Optional[Timing] = None
    latency_extra_s: Optional[float] = None
    metadata: Tuple[Tuple[str, Any], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        path = "Sync(%s)" % self.op_id
        _op_common(self, path)
        if self.sync_kind not in SYNC_KINDS:
            raise _fail(path, "sync_kind", self.sync_kind,
                        "must be one of %s" % ", ".join(SYNC_KINDS))
        object.__setattr__(self, "participants", tuple(self.participants))
        for item in self.participants:
            _ident(item, path, "participants")
        if self.latency_extra_s is not None:
            object.__setattr__(
                self, "latency_extra_s",
                _nonneg_float(self.latency_extra_s, path, "latency_extra_s"),
            )

    @property
    def kind(self) -> str:
        return "sync"


def is_op(value: Any) -> bool:
    return isinstance(value, (Load, Store, Compute, Sync))


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Actor:
    """One issue role of a pipeline with its ops and cyclic program order."""

    name: str
    ops: Tuple[Op, ...]
    sequence: Tuple[Tuple[str, int], ...] = field(default_factory=tuple)
    order: str = "issue"
    resource: Optional[str] = None
    execution_scope: str = "unspecified"

    def __post_init__(self) -> None:
        path = "Actor(%s)" % self.name
        _ident(self.name, path, "name")
        ops = tuple(self.ops)
        if not ops or not all(is_op(item) for item in ops):
            raise _fail(path, "ops", ops, "must be a non-empty tuple of ops")
        for op in ops:
            if op.actor != self.name:
                raise _fail(path, "ops", op.op_id,
                            "op.actor=%r must equal the owning actor" % op.actor)
        names = [op.op_id for op in ops]
        if len(names) != len(set(names)):
            raise _fail(path, "ops", names, "op ids must be unique")
        object.__setattr__(self, "ops", ops)
        sequence = tuple((str(a), int(b)) for a, b in self.sequence)
        for op_id, _offset in sequence:
            if op_id not in names:
                raise _fail(path, "sequence", op_id, "is not an op of this actor")
        seq_ids = [item[0] for item in sequence]
        if len(seq_ids) != len(set(seq_ids)):
            raise _fail(path, "sequence", seq_ids, "an op may appear once")
        object.__setattr__(self, "sequence", sequence)
        if self.order not in ACTOR_ORDERS:
            raise _fail(path, "order", self.order, "must be issue or completion")
        if self.resource is not None:
            _ident(self.resource, path, "resource")
        _ident(self.execution_scope, path, "execution_scope")


@dataclass(frozen=True)
class Dependency:
    source: EventRef
    target: EventRef
    distance: int = 0
    lag_s: float = 0.0
    name: str = ""

    def __post_init__(self) -> None:
        path = "Dependency(%s->%s)" % (self.source.op_id, self.target.op_id)
        if not isinstance(self.source, EventRef) or not isinstance(self.target, EventRef):
            raise _fail(path, "source/target", None, "must be EventRef")
        if self.target.kind != "start":
            raise _fail(path, "target", self.target.kind, "must target a start event")
        if not isinstance(self.distance, int) or isinstance(self.distance, bool):
            raise _fail(path, "distance", self.distance, "must be an integer")
        object.__setattr__(self, "lag_s", _nonneg_float(self.lag_s, path, "lag_s"))
        if self.name:
            _ident(self.name, path, "name")


@dataclass(frozen=True)
class Carry:
    """A loop-carried state with the recurrence it induces."""

    state: str
    source: EventRef
    target: EventRef
    distance: int = 1
    storage: Optional[str] = None
    lag_s: float = 0.0

    def __post_init__(self) -> None:
        path = "Carry(%s)" % self.state
        _ident(self.state, path, "state")
        if not isinstance(self.source, EventRef) or not isinstance(self.target, EventRef):
            raise _fail(path, "source/target", None, "must be EventRef")
        if self.target.kind != "start":
            raise _fail(path, "target", self.target.kind, "must target a start event")
        _pos_int(self.distance, path, "distance")
        if self.storage is not None:
            _ident(self.storage, path, "storage")
        object.__setattr__(self, "lag_s", _nonneg_float(self.lag_s, path, "lag_s"))


@dataclass(frozen=True)
class BufferSlots:
    """Bounded acquire/release lifetime that lowers to a FIFO token buffer."""

    buffer: str
    acquire: EventRef
    release: EventRef
    capacity: Optional[int] = None

    def __post_init__(self) -> None:
        path = "BufferSlots(%s)" % self.buffer
        _ident(self.buffer, path, "buffer")
        if not isinstance(self.acquire, EventRef) or not isinstance(self.release, EventRef):
            raise _fail(path, "acquire/release", None, "must be EventRef")
        if self.capacity is not None:
            _pos_int(self.capacity, path, "capacity")


STORAGE_GROUP_KINDS = ("alias", "ring")


@dataclass(frozen=True)
class StorageGroup:
    """Buffers that share one physical on-chip allocation.

    ``alias``: the members occupy the *same* slot one after another inside one
    lifetime (e.g. P written over S); the group holds ``slots`` such lifetimes.
    ``ring``: the members are allocated round-robin, in the listed order, from one
    FIFO pool of ``slots`` slots (e.g. an aliased K/V ring issued K, V, K, V ...).
    A slot is as large as the largest member.  The group is what capacity
    accounting counts; the matching lifetime constraints are ``BufferSlots`` /
    ``Dependency`` entries of the pipeline (``KernelBuilder`` derives them).
    """

    name: str
    kind: str
    members: Tuple[str, ...]
    slots: int = 1

    def __post_init__(self) -> None:
        path = "StorageGroup(%s)" % self.name
        _ident(self.name, path, "name")
        if self.kind not in STORAGE_GROUP_KINDS:
            raise _fail(path, "kind", self.kind, "must be one of %s" % ", ".join(STORAGE_GROUP_KINDS))
        members = tuple(self.members)
        if len(members) < 2 or len(set(members)) != len(members):
            raise _fail(path, "members", members, "needs at least two distinct buffers")
        object.__setattr__(self, "members", members)
        _pos_int(self.slots, path, "slots")


@dataclass(frozen=True)
class ResourceOrder:
    """Explicit cyclic arbitration order of one exclusive resource."""

    resource: str
    sequence: Tuple[Tuple[str, int], ...]

    def __post_init__(self) -> None:
        path = "ResourceOrder(%s)" % self.resource
        _ident(self.resource, path, "resource")
        sequence = tuple((str(a), int(b)) for a, b in self.sequence)
        if not sequence:
            raise _fail(path, "sequence", sequence, "must not be empty")
        object.__setattr__(self, "sequence", sequence)


@dataclass(frozen=True)
class Pipeline:
    """Steady-state software pipeline; only valid as ``Loop.body``."""

    actors: Tuple[Actor, ...]
    buffers: Tuple[Buffer, ...] = field(default_factory=tuple)
    dependencies: Tuple[Dependency, ...] = field(default_factory=tuple)
    carries: Tuple[Carry, ...] = field(default_factory=tuple)
    buffer_slots: Tuple[BufferSlots, ...] = field(default_factory=tuple)
    resource_orders: Tuple[ResourceOrder, ...] = field(default_factory=tuple)
    stages: Optional[int] = None
    storage_groups: Tuple[StorageGroup, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        path = "Pipeline"
        actors = tuple(self.actors)
        if not actors or not all(isinstance(item, Actor) for item in actors):
            raise _fail(path, "actors", actors, "must be a non-empty tuple of Actor")
        names = [item.name for item in actors]
        if len(names) != len(set(names)):
            raise _fail(path, "actors", names, "actor names must be unique")
        object.__setattr__(self, "actors", actors)
        for name in ("buffers", "dependencies", "carries", "buffer_slots", "resource_orders", "storage_groups"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        expected = (
            ("buffers", Buffer), ("dependencies", Dependency), ("carries", Carry),
            ("buffer_slots", BufferSlots), ("resource_orders", ResourceOrder), ("storage_groups", StorageGroup),
        )
        for name, typ in expected:
            if not all(isinstance(item, typ) for item in getattr(self, name)):
                raise _fail(path, name, getattr(self, name), "contains a wrong type")
        buffer_names = [item.name for item in self.buffers]
        if len(buffer_names) != len(set(buffer_names)):
            raise _fail(path, "buffers", buffer_names, "buffer names must be unique")
        op_ids = [op.op_id for actor in actors for op in actor.ops]
        if len(op_ids) != len(set(op_ids)):
            raise _fail(path, "actors", op_ids, "op ids must be unique across actors")
        known_ops = set(op_ids)
        known_buffers = set(buffer_names)
        for op in self.ops:
            for name in op.reads + op.writes:
                if name not in known_buffers:
                    raise _fail("Pipeline/%s" % op.op_id, "reads/writes", name,
                                "references an undeclared buffer")
        for dep in self.dependencies:
            for ref in (dep.source, dep.target):
                if ref.op_id not in known_ops:
                    raise _fail(path, "dependencies", ref.op_id, "references unknown op")
        for carry in self.carries:
            for ref in (carry.source, carry.target):
                if ref.op_id not in known_ops:
                    raise _fail(path, "carries", ref.op_id, "references unknown op")
            if carry.storage is not None and carry.storage not in known_buffers:
                raise _fail(path, "carries", carry.storage, "references unknown buffer")
        carry_names = [item.state for item in self.carries]
        if len(carry_names) != len(set(carry_names)):
            raise _fail(path, "carries", carry_names, "state names must be unique")
        slot_names = []
        for item in self.buffer_slots:
            if item.buffer not in known_buffers:
                raise _fail(path, "buffer_slots", item.buffer, "references unknown buffer")
            for ref in (item.acquire, item.release):
                if ref.op_id not in known_ops:
                    raise _fail(path, "buffer_slots", ref.op_id, "references unknown op")
            slot_names.append(item.buffer)
        if len(slot_names) != len(set(slot_names)):
            raise _fail(path, "buffer_slots", slot_names, "one lifetime per buffer")
        for order in self.resource_orders:
            for op_id, _offset in order.sequence:
                if op_id not in known_ops:
                    raise _fail(path, "resource_orders", op_id, "references unknown op")
        resources = [item.resource for item in self.resource_orders]
        if len(resources) != len(set(resources)):
            raise _fail(path, "resource_orders", resources, "one order per resource")
        if self.stages is not None:
            _pos_int(self.stages, path, "stages")
        by_name = {item.name: item for item in self.buffers}
        grouped = []  # type: List[str]
        for group in self.storage_groups:
            storages = set()
            for member in group.members:
                if member not in by_name:
                    raise _fail(path, "storage_groups", member, "references unknown buffer")
                storages.add(by_name[member].storage)
                grouped.append(member)
            if len(storages) != 1 or storages & {"register", "fragment"}:
                raise _fail(path, "storage_groups", group.name, "members must live in the same smem or tmem storage")
        if len(grouped) != len(set(grouped)):
            raise _fail(path, "storage_groups", grouped, "a buffer belongs to at most one storage group")

    def storage_bytes(self) -> Dict[str, float]:
        """Declared on-chip bytes per storage kind for one work unit (groups counted once)."""

        from ..ops.schema import DTypeSpec

        def size(item: Buffer) -> float:
            if not item.shape or not item.dtype:
                return 0.0
            return float(_product(item.shape)) * DTypeSpec(item.dtype).storage_bytes

        by_name = {item.name: item for item in self.buffers}
        totals = {}  # type: Dict[str, float]
        grouped = set()
        for group in self.storage_groups:
            members = [by_name[name] for name in group.members]
            grouped.update(group.members)
            storage = members[0].storage
            totals[storage] = totals.get(storage, 0.0) + group.slots * max(size(m) for m in members)
        for item in self.buffers:
            if item.name not in grouped and item.storage in ("smem", "tmem"):
                totals[item.storage] = totals.get(item.storage, 0.0) + item.slots * size(item)
        return totals

    @property
    def ops(self) -> Tuple[Op, ...]:
        return tuple(op for actor in self.actors for op in actor.ops)


# ---------------------------------------------------------------------------
# Regions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FromWorkGroup:
    """Loop trip count bound from the launch's current work group."""

    def canonical(self) -> Dict[str, Any]:
        return {"kind": "from_work_group"}


@dataclass(frozen=True)
class Sequence:
    """Explicit serial order of child regions."""

    children: Tuple[Any, ...]
    name: Optional[str] = None

    def __post_init__(self) -> None:
        path = "Sequence(%s)" % (self.name or "?")
        children = tuple(self.children)
        if not children:
            raise _fail(path, "children", children, "must not be empty")
        for child in children:
            if isinstance(child, Pipeline):
                raise _fail(path, "children", "Pipeline",
                            "Pipeline may only be the direct body of a Loop")
            if not (is_op(child) or isinstance(child, (Sequence, Loop))):
                raise _fail(path, "children", child, "must be op, Sequence, or Loop")
        object.__setattr__(self, "children", children)
        if self.name is not None:
            _ident(self.name, path, "name")


@dataclass(frozen=True)
class Loop:
    """Repeat ``body`` ``trip_count`` times, with one-time prologue/epilogue."""

    loop_id: str
    trip_count: Union[int, FromWorkGroup]
    body: Any
    prologue: Optional[Any] = None
    epilogue: Optional[Any] = None
    ii_mode: str = "inherit"
    boundary_anchor_op: Optional[str] = None

    def __post_init__(self) -> None:
        path = "Loop(%s)" % self.loop_id
        _ident(self.loop_id, path, "loop_id")
        if isinstance(self.trip_count, FromWorkGroup):
            pass
        else:
            _nonneg_int(self.trip_count, path, "trip_count")
        if not (is_op(self.body) or isinstance(self.body, (Sequence, Loop, Pipeline))):
            raise _fail(path, "body", self.body, "must be op, Sequence, Loop, or Pipeline")
        for site in ("prologue", "epilogue"):
            value = getattr(self, site)
            if value is None:
                continue
            if isinstance(value, Pipeline):
                raise _fail(path, site, "Pipeline",
                            "Pipeline may only be the direct body of a Loop")
            if not (is_op(value) or isinstance(value, (Sequence, Loop))):
                raise _fail(path, site, value, "must be op, Sequence, Loop, or None")
        if self.ii_mode not in II_MODES:
            raise _fail(path, "ii_mode", self.ii_mode, "must be one of %s" % ", ".join(II_MODES))
        if self.boundary_anchor_op is not None:
            _ident(self.boundary_anchor_op, path, "boundary_anchor_op")
            if not isinstance(self.body, Pipeline):
                raise _fail(path, "boundary_anchor_op", self.boundary_anchor_op,
                            "only meaningful for a Pipeline body")
            if self.boundary_anchor_op not in {op.op_id for op in self.body.ops}:
                raise _fail(path, "boundary_anchor_op", self.boundary_anchor_op,
                            "is not an op of the pipeline body")

    @property
    def is_periodic(self) -> bool:
        return isinstance(self.body, Pipeline)


Region = Union[Op, Sequence, Loop]


def is_region(value: Any) -> bool:
    return is_op(value) or isinstance(value, (Sequence, Loop))


@dataclass(frozen=True)
class OpSite:
    """Where one op sits: launch, enclosing loop chain, and the op."""

    launch: str
    chain: Tuple[Tuple[str, str], ...]  # ((loop_id, site), ...) outermost first
    op: Op
    in_pipeline: bool

    @property
    def loop_ids(self) -> Tuple[str, ...]:
        return tuple(loop_id for loop_id, _site in self.chain)

    @property
    def path(self) -> str:
        return "/".join((self.launch,) + self.loop_ids + (self.op.op_id,))

    @property
    def region_site(self) -> str:
        return "/".join("%s:%s" % item for item in self.chain) or "-"


def walk_ops(region: Any, launch: str, chain: Tuple[Tuple[str, str], ...] = ()) -> List[OpSite]:
    """Return every op with its loop chain, in declaration order."""

    result = []  # type: List[OpSite]
    if is_op(region):
        result.append(OpSite(launch, chain, region, False))
    elif isinstance(region, Sequence):
        for child in region.children:
            result.extend(walk_ops(child, launch, chain))
    elif isinstance(region, Loop):
        if region.prologue is not None:
            result.extend(walk_ops(region.prologue, launch, chain + ((region.loop_id, "prologue"),)))
        body_chain = chain + ((region.loop_id, "body"),)
        if isinstance(region.body, Pipeline):
            for op in region.body.ops:
                result.append(OpSite(launch, body_chain, op, True))
        else:
            result.extend(walk_ops(region.body, launch, body_chain))
        if region.epilogue is not None:
            result.extend(walk_ops(region.epilogue, launch, chain + ((region.loop_id, "epilogue"),)))
    else:
        raise ContractError("walk_ops: unexpected region %r" % (region,))
    return result


def walk_loops(region: Any, chain: Tuple[str, ...] = ()) -> List[Tuple[Tuple[str, ...], Loop]]:
    """Return every loop with its enclosing loop-id chain, outermost first."""

    result = []  # type: List[Tuple[Tuple[str, ...], Loop]]
    if isinstance(region, Sequence):
        for child in region.children:
            result.extend(walk_loops(child, chain))
    elif isinstance(region, Loop):
        result.append((chain, region))
        inner = chain + (region.loop_id,)
        for site in (region.prologue, region.epilogue):
            if site is not None:
                result.extend(walk_loops(site, inner))
        if not isinstance(region.body, Pipeline):
            result.extend(walk_loops(region.body, inner))
    return result


# ---------------------------------------------------------------------------
# Launch, work units, program
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkAxis:
    name: str
    extent: int

    def __post_init__(self) -> None:
        _ident(self.name, "WorkAxis", "name")
        _pos_int(self.extent, "WorkAxis(%s)" % self.name, "extent")


@dataclass(frozen=True)
class WorkGroup:
    """Per-group loop trip counts and optional op overrides."""

    loop_trip_counts: Tuple[Tuple[str, int], ...] = field(default_factory=tuple)
    region_variant: Optional[str] = None
    effective_fractions: Tuple[Tuple[str, float], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        path = "WorkGroup"
        items = _pairs(self.loop_trip_counts, path, "loop_trip_counts")
        for loop_id, count in items:
            _ident(loop_id, path, "loop_trip_counts")
            _nonneg_int(count, path, "loop_trip_counts[%s]" % loop_id)
        object.__setattr__(self, "loop_trip_counts", items)
        if self.region_variant is not None:
            _ident(self.region_variant, path, "region_variant")
        fractions = _pairs(self.effective_fractions, path, "effective_fractions")
        for op_id, fraction in fractions:
            _ident(op_id, path, "effective_fractions")
            value = _nonneg_float(fraction, path, "effective_fractions[%s]" % op_id)
            if value > 1.0:
                raise _fail(path, "effective_fractions", value, "must be in [0, 1]")
        object.__setattr__(
            self, "effective_fractions",
            tuple((op_id, float(fraction)) for op_id, fraction in fractions),
        )

    def trip_count(self, loop_id: str) -> Optional[int]:
        for name, count in self.loop_trip_counts:
            if name == loop_id:
                return count
        return None

    def effective_fraction(self, op_id: str) -> Optional[float]:
        for name, fraction in self.effective_fractions:
            if name == op_id:
                return fraction
        return None


def _pairs(value: Any, path: str, field_name: str) -> Tuple[Tuple[str, Any], ...]:
    if isinstance(value, Mapping):
        items = tuple(value.items())
    else:
        try:
            items = tuple(tuple(item) for item in value)
        except TypeError:
            raise _fail(path, field_name, value, "must be a mapping or key/value pairs")
    keys = [item[0] for item in items]
    if len(keys) != len(set(keys)):
        raise _fail(path, field_name, keys, "keys must be unique")
    return tuple(sorted(items))


@dataclass(frozen=True)
class WorkRun:
    group_id: str
    count: int

    def __post_init__(self) -> None:
        _ident(self.group_id, "WorkRun", "group_id")
        _pos_int(self.count, "WorkRun(%s)" % self.group_id, "count")


@dataclass(frozen=True)
class WorkUnits:
    """The single source of truth for spatial work grouping and order."""

    groups: Tuple[Tuple[str, WorkGroup], ...]
    order: Tuple[WorkRun, ...]

    def __post_init__(self) -> None:
        path = "WorkUnits"
        groups = _pairs(self.groups, path, "groups")
        for group_id, group in groups:
            _ident(group_id, path, "groups")
            if not isinstance(group, WorkGroup):
                raise _fail(path, "groups[%s]" % group_id, group, "must be WorkGroup")
        object.__setattr__(self, "groups", groups)
        order = tuple(self.order)
        if not order or not all(isinstance(item, WorkRun) for item in order):
            raise _fail(path, "order", order, "must be a non-empty tuple of WorkRun")
        known = {group_id for group_id, _group in groups}
        for run in order:
            if run.group_id not in known:
                raise _fail(path, "order", run.group_id, "references an unknown group")
        object.__setattr__(self, "order", order)

    @property
    def total_count(self) -> int:
        return sum(run.count for run in self.order)

    def group(self, group_id: str) -> WorkGroup:
        for name, group in self.groups:
            if name == group_id:
                return group
        raise KeyError(group_id)

    def group_counts(self) -> Dict[str, int]:
        counts = {group_id: 0 for group_id, _group in self.groups}
        for run in self.order:
            counts[run.group_id] += run.count
        return counts

    def instance_group_ids(self) -> List[str]:
        result = []  # type: List[str]
        for run in self.order:
            result.extend([run.group_id] * run.count)
        return result


@dataclass(frozen=True)
class Persistent:
    """Persistent scheduler: explicit work-instance partitions per CTA.

    ``partitions[i]`` is a tuple of ``(start, stop)`` half-open instance
    index ranges in the launch work sequence owned by CTA ``i``.
    """

    partitions: Tuple[Tuple[Tuple[int, int], ...], ...]

    def __post_init__(self) -> None:
        path = "Persistent"
        partitions = tuple(tuple(tuple(r) for r in part) for part in self.partitions)
        if not partitions:
            raise _fail(path, "partitions", partitions, "must not be empty")
        for part in partitions:
            for item in part:
                if len(item) != 2 or any(
                    not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in item
                ) or item[0] >= item[1]:
                    raise _fail(path, "partitions", item, "ranges must be (start, stop) with start < stop")
        object.__setattr__(self, "partitions", partitions)

    @property
    def cta_count(self) -> int:
        return len(self.partitions)

    def validate_coverage(self, total: int) -> None:
        seen = [0] * total
        for part in self.partitions:
            for begin, end in part:
                if end > total:
                    raise _fail("Persistent", "partitions", (begin, end),
                                "exceeds work sequence length %d" % total)
                for index in range(begin, end):
                    seen[index] += 1
        bad = [index for index, count in enumerate(seen) if count != 1]
        if bad:
            raise _fail("Persistent", "partitions", bad[:8],
                        "instances must be covered exactly once")


class BlockMapper:
    """Decides in which order the work blocks of a launch are dispatched.

    This is the single interface through which cache/L2 modelling, tail-group
    derivation and spatial scheduling learn the launch order.  A mapper maps
    the work-grid extents to the sequence of block coordinates (one tuple per
    work instance, in dispatch order).  Row panels are only one common swizzle;
    any other policy is expressed by another mapper, or materialised with
    ``ExplicitOrder.from_function``.  Mappers are frozen and serialisable so a
    ``Program`` snapshot records the assumed launch order.
    """

    kind = "abstract"

    def coordinates(self, extents: SeqT[int]) -> List[Tuple[int, ...]]:
        raise NotImplementedError

    def canonical(self) -> Dict[str, Any]:
        raise NotImplementedError

    def panel_parameters(self) -> Optional[Tuple[int, int, str]]:
        """``(row_panel, column_panel, raster_axis)`` when the order is a panel walk the
        reduction-axis cache backend can reproduce; ``None`` sends cache modelling to the
        explicit instance trace, which accepts every mapper."""

        return None


def _row_major(extents: SeqT[int]) -> List[Tuple[int, ...]]:
    result = []
    for index in range(_product(extents)):
        coords = []
        remaining = index
        for extent in reversed(tuple(extents)):
            coords.append(remaining % extent)
            remaining //= extent
        result.append(tuple(reversed(coords)))
    return result


@dataclass(frozen=True)
class LinearBlockId(BlockMapper):
    """``blockIdx`` order: row-major over the work axes, last axis fastest."""

    kind = "linear_block_id"

    def coordinates(self, extents: SeqT[int]) -> List[Tuple[int, ...]]:
        return _row_major(extents)

    def canonical(self) -> Dict[str, Any]:
        return {"kind": self.kind}


@dataclass(frozen=True)
class PanelSwizzle(BlockMapper):
    """Panel (row/column swizzle) order over a rank-2 work grid.

    The first work axis is walked in strides of ``column_panel`` and the second
    in strides of ``row_panel``; ``raster_axis`` picks which panel index
    advances first.  One common swizzle, not the only one.
    """

    row_panel: int
    column_panel: int
    raster_axis: str = "along_m"
    kind = "panel_swizzle"

    def __post_init__(self) -> None:
        _pos_int(self.row_panel, "PanelSwizzle", "row_panel")
        _pos_int(self.column_panel, "PanelSwizzle", "column_panel")
        if self.raster_axis not in ("along_m", "along_n", "legacy"):
            raise _fail("PanelSwizzle", "raster_axis", self.raster_axis, "must be along_m, along_n or legacy")

    def coordinates(self, extents: SeqT[int]) -> List[Tuple[int, ...]]:
        extents = tuple(extents)
        if len(extents) != 2:
            raise ContractError("PanelSwizzle needs exactly two work axes, got %d" % len(extents))
        grid_m, grid_n = extents
        result = []
        m_start = n_start = 0
        while m_start < grid_m and n_start < grid_n:
            m_end = min(m_start + self.column_panel, grid_m)
            n_end = min(n_start + self.row_panel, grid_n)
            # Same walk as cache.traversal._panel_coordinates: "legacy" is the historical
            # pipeline-wave order (n fastest inside a panel, M panels advance first).
            if self.raster_axis in ("along_n", "legacy"):
                result.extend((m, n) for m in range(m_start, m_end) for n in range(n_start, n_end))
            else:
                result.extend((m, n) for n in range(n_start, n_end) for m in range(m_start, m_end))
            if self.raster_axis in ("along_m", "legacy"):
                m_start = m_end
                if m_start >= grid_m:
                    m_start, n_start = 0, n_end
            else:
                n_start = n_end
                if n_start >= grid_n:
                    n_start, m_start = 0, m_end
        return result

    def canonical(self) -> Dict[str, Any]:
        return {"kind": self.kind, "row_panel": self.row_panel, "column_panel": self.column_panel,
                "raster_axis": self.raster_axis}

    def panel_parameters(self) -> Optional[Tuple[int, int, str]]:
        return (self.row_panel, self.column_panel, self.raster_axis)


@dataclass(frozen=True)
class ExplicitOrder(BlockMapper):
    """A fully materialised launch order (any swizzle/LPT/scheduler policy)."""

    order: Tuple[Tuple[int, ...], ...]
    label: str = "explicit"
    kind = "explicit_order"

    def __post_init__(self) -> None:
        order = tuple(tuple(int(v) for v in item) for item in self.order)
        if not order:
            raise _fail("ExplicitOrder", "order", order, "must not be empty")
        if len(set(order)) != len(order):
            raise _fail("ExplicitOrder", "order", "...", "a block appears more than once")
        object.__setattr__(self, "order", order)
        _ident(self.label, "ExplicitOrder", "label")

    @classmethod
    def from_function(cls, label: str, function: Any, extents: SeqT[int]) -> "ExplicitOrder":
        """Materialise ``function(extents) -> iterable of coordinates`` (e.g. a scheduler's LPT order)."""

        return cls(tuple(tuple(item) for item in function(tuple(extents))), label)

    def coordinates(self, extents: SeqT[int]) -> List[Tuple[int, ...]]:
        expected = set(_row_major(extents))
        if set(self.order) != expected:
            raise ContractError("ExplicitOrder(%s) is not a permutation of the %r work grid" % (self.label, tuple(extents)))
        return list(self.order)

    def canonical(self) -> Dict[str, Any]:
        return {"kind": self.kind, "label": self.label, "order": [list(item) for item in self.order]}


@dataclass(frozen=True)
class CutlassSm90(BlockMapper):
    """Block id -> tile coordinate of the CUTLASS SM90 static persistent tile scheduler.

    Mirrors ``PersistentTileSchedulerSm90::get_work_idx_m_and_n`` and
    ``PersistentTileSchedulerSm90Params::initialize`` (cutlass/gemm/kernel/
    sm90_tile_scheduler.hpp, tile_scheduler_params.h): the tile grid is padded to a
    multiple of ``swizzle x cluster`` per axis, the swizzle size follows
    ``get_log_swizzle_size`` and ``raster_order='heuristic'`` follows
    ``get_rasterization_order``.  The work grid is ``(m, n)`` or ``(l, m, n)``;
    linear block ids run in launch order, and ids that land in the padding carry
    no work and are dropped.
    """

    cluster_m: int = 1
    cluster_n: int = 1
    max_swizzle_size: int = 1
    raster_order: str = "heuristic"
    kind = "cutlass_sm90"

    def __post_init__(self) -> None:
        for name in ("cluster_m", "cluster_n"):
            value = _pos_int(getattr(self, name), "CutlassSm90", name)
            if value & (value - 1):
                raise _fail("CutlassSm90", name, value, "must be a power of two (FastDivmodU64Pow2)")
        if not isinstance(self.max_swizzle_size, int) or isinstance(self.max_swizzle_size, bool) or self.max_swizzle_size < 0:
            raise _fail("CutlassSm90", "max_swizzle_size", self.max_swizzle_size, "must be a non-negative int")
        if self.raster_order not in ("heuristic", "along_m", "along_n"):
            raise _fail("CutlassSm90", "raster_order", self.raster_order, "must be heuristic, along_m or along_n")

    @staticmethod
    def log_swizzle_size(tiles_m: int, tiles_n: int, max_swizzle_size: int) -> int:
        smallest = min(tiles_m, tiles_n)
        if max_swizzle_size >= 8 and smallest >= 6:
            return 3
        if max_swizzle_size >= 4 and smallest >= 3:
            return 2
        if max_swizzle_size >= 2 and smallest >= 2:
            return 1
        return 0

    def resolve(self, extents: SeqT[int]) -> Dict[str, Any]:
        """Scheduler parameters CUTLASS derives for this tile grid."""

        extents = tuple(extents)
        if len(extents) not in (2, 3):
            raise ContractError("CutlassSm90 needs an (m, n) or (l, m, n) work grid, got %r" % (extents,))
        tiles_m, tiles_n = extents[-2], extents[-1]
        log_swizzle = self.log_swizzle_size(tiles_m, tiles_n, self.max_swizzle_size)
        swizzle = 1 << log_swizzle
        padded_m = -(-tiles_m // (swizzle * self.cluster_m)) * swizzle * self.cluster_m
        padded_n = -(-tiles_n // (swizzle * self.cluster_n)) * swizzle * self.cluster_n
        raster = self.raster_order
        if raster == "heuristic":
            raster = "along_m" if padded_n > padded_m else "along_n"
        return {"log_swizzle_size": log_swizzle, "swizzle_size": swizzle, "raster_order": raster,
                "padded_m": padded_m, "padded_n": padded_n, "batch": extents[0] if len(extents) == 3 else 1,
                "padding_blocks": (padded_m * padded_n - tiles_m * tiles_n) * (extents[0] if len(extents) == 3 else 1)}

    def work_index(self, linear_idx: int, extents: SeqT[int]) -> Tuple[int, int, int]:
        """``(l, m, n)`` of one linear block id (may lie in the padding)."""

        info = self.resolve(extents)
        along_n = info["raster_order"] == "along_n"
        shape_major, shape_minor = (self.cluster_n, self.cluster_m) if along_n else (self.cluster_m, self.cluster_n)
        blk_major = (info["padded_n"] // self.cluster_n) if along_n else (info["padded_m"] // self.cluster_m)
        log_swizzle = info["log_swizzle_size"]
        work_l, remainder = divmod(linear_idx, info["padded_m"] * info["padded_n"])
        blk_per_grid_dim, cluster_minor_offset = divmod(remainder, shape_minor)
        cluster_id, cluster_major_offset = divmod(blk_per_grid_dim, shape_major)
        offset = cluster_id & ((1 << log_swizzle) - 1)
        extra = cluster_id >> log_swizzle
        minor_div_swizzle, cluster_idx_major = divmod(extra, blk_major)
        cluster_idx_minor = minor_div_swizzle * (1 << log_swizzle) + offset
        minor = cluster_idx_minor * shape_minor + cluster_minor_offset
        major = cluster_idx_major * shape_major + cluster_major_offset
        return (work_l, minor, major) if along_n else (work_l, major, minor)

    def coordinates(self, extents: SeqT[int]) -> List[Tuple[int, ...]]:
        extents = tuple(extents)
        info = self.resolve(extents)
        tiles_m, tiles_n = extents[-2], extents[-1]
        result = []
        for linear_idx in range(info["padded_m"] * info["padded_n"] * info["batch"]):
            work_l, m, n = self.work_index(linear_idx, extents)
            if m < tiles_m and n < tiles_n:
                result.append((work_l, m, n) if len(extents) == 3 else (m, n))
        return result

    def canonical(self) -> Dict[str, Any]:
        return {"kind": self.kind, "cluster_m": self.cluster_m, "cluster_n": self.cluster_n,
                "max_swizzle_size": self.max_swizzle_size, "raster_order": self.raster_order}


PanelDispatch = PanelSwizzle  # earlier spelling

_MAPPER_FACTORIES = {}  # type: Dict[str, Any]


def register_block_mapper(kind: str, factory: Any) -> None:
    """Make a custom mapper restorable from a snapshot: ``factory(canonical_dict) -> BlockMapper``."""

    _ident(kind, "register_block_mapper", "kind")
    if kind in ("linear_block_id", "panel_swizzle", "panel", "explicit_order", "cutlass_sm90"):
        raise ContractError("block mapper kind %r is built in" % kind)
    _MAPPER_FACTORIES[kind] = factory


def validate_block_order(extents: SeqT[int], ordered: Any, label: str) -> List[Tuple[int, ...]]:
    """Check a mapper's output: every block once, integer coordinates of the right rank, in range."""

    extents = tuple(extents)
    result = []
    for position, item in enumerate(ordered):
        if not isinstance(item, (tuple, list)) or len(item) != len(extents):
            raise ContractError("block mapper %s: item %d = %r is not a rank-%d coordinate"
                                % (label, position, item, len(extents)))
        for value, extent in zip(item, extents):
            if not isinstance(value, int) or isinstance(value, bool):
                raise ContractError("block mapper %s: item %d = %r has a non-integer coordinate" % (label, position, item))
            if not 0 <= value < extent:
                raise ContractError("block mapper %s: item %d = %r is outside the %r work grid"
                                    % (label, position, tuple(item), extents))
        result.append(tuple(item))
    if len(result) != _product(extents) or len(set(result)) != len(result):
        raise ContractError("block mapper %s must return every one of the %d work blocks exactly once (got %d, %d distinct)"
                            % (label, _product(extents), len(result), len(set(result))))
    return result


def as_mapper(value: Any) -> BlockMapper:
    if isinstance(value, BlockMapper):
        return value
    if value == "linear_block_id":
        return LinearBlockId()
    raise ContractError("dispatch_order=%r: must be 'linear_block_id' or a BlockMapper" % (value,))


def mapper_from_canonical(data: Any) -> Any:
    if isinstance(data, str):
        return data
    kind = data.get("kind")
    if kind == "linear_block_id":
        return LinearBlockId()
    if kind in ("panel_swizzle", "panel"):
        return PanelSwizzle(data["row_panel"], data["column_panel"], data.get("raster_axis", "along_m"))
    if kind == "explicit_order":
        return ExplicitOrder(tuple(tuple(item) for item in data["order"]), data.get("label", "explicit"))
    if kind == "cutlass_sm90":
        return CutlassSm90(data["cluster_m"], data["cluster_n"], data["max_swizzle_size"], data["raster_order"])
    if kind in _MAPPER_FACTORIES:
        return as_mapper(_MAPPER_FACTORIES[kind](data))
    raise ContractError("unknown block mapper kind %r (register it with register_block_mapper)" % kind)


def dispatch_coordinates(extents: SeqT[int], dispatch_order: Any) -> List[Tuple[int, ...]]:
    """Work-instance coordinates in dispatch order (one tuple per instance)."""

    return as_mapper(dispatch_order).coordinates(extents)


@dataclass(frozen=True)
class Launch:
    """One kernel launch: body, spatial work, and launch shape."""

    name: str
    body: Any
    work_axes: Tuple[WorkAxis, ...]
    threads: int = 128
    grid: Optional[Tuple[int, ...]] = None
    cluster: Tuple[int, ...] = (1, 1, 1)
    residency: Union[str, int] = "auto"
    work_units: Optional[WorkUnits] = None
    scheduler: Union[str, Persistent] = "static_dispatch"
    dispatch_order: Union[str, BlockMapper] = "linear_block_id"
    requires_arch: Tuple[str, ...] = field(default_factory=tuple)
    metadata: Tuple[Tuple[str, Any], ...] = field(default_factory=tuple)
    buffers: Tuple[Buffer, ...] = field(default_factory=tuple)               # launch-scope allocations (any region)
    storage_groups: Tuple[StorageGroup, ...] = field(default_factory=tuple)  # alias/ring groups of launch-scope buffers

    def __post_init__(self) -> None:
        path = "Launch(%s)" % self.name
        _ident(self.name, path, "name")
        requires = tuple(self.requires_arch)
        for item in requires:
            if item not in ARCH_CAPABILITIES:
                raise _fail(path, "requires_arch", item, "must be one of %s" % ", ".join(ARCH_CAPABILITIES))
        object.__setattr__(self, "requires_arch", requires)
        if not is_region(self.body):
            if isinstance(self.body, Pipeline):
                raise _fail(path, "body", "Pipeline",
                            "wrap a pipeline in Loop(trip_count=...) to use it as a body")
            raise _fail(path, "body", self.body, "must be op, Sequence, or Loop")
        axes = tuple(self.work_axes)
        if not axes or not all(isinstance(item, WorkAxis) for item in axes):
            raise _fail(path, "work_axes", axes, "must be a non-empty tuple of WorkAxis")
        axis_names = [item.name for item in axes]
        if len(axis_names) != len(set(axis_names)):
            raise _fail(path, "work_axes", axis_names, "axis names must be unique")
        object.__setattr__(self, "work_axes", axes)
        _pos_int(self.threads, path, "threads")
        if self.grid is not None:
            object.__setattr__(self, "grid", _shape(self.grid, path, "grid"))
        cluster = _shape(self.cluster, path, "cluster")
        object.__setattr__(self, "cluster", cluster)
        if not (self.residency == "auto" or (
            isinstance(self.residency, int) and not isinstance(self.residency, bool)
            and self.residency > 0
        )):
            raise _fail(path, "residency", self.residency, "must be 'auto' or a positive int")
        if self.work_units is not None and not isinstance(self.work_units, WorkUnits):
            raise _fail(path, "work_units", self.work_units, "must be WorkUnits or None")
        mapper = as_mapper(self.dispatch_order)
        extents = [axis.extent for axis in axes]
        try:
            ordered = validate_block_order(extents, mapper.coordinates(extents), str(mapper.kind))
        except ContractError as error:
            raise _fail(path, "dispatch_order", getattr(mapper, "kind", type(mapper).__name__), str(error))
        builtin = (LinearBlockId, PanelSwizzle, ExplicitOrder, CutlassSm90)
        if type(mapper) not in builtin and mapper.kind not in _MAPPER_FACTORIES:
            # An unregistered custom mapper is frozen into the order it produced, so the
            # launch (and its snapshot) no longer depends on user code.
            label = "".join(c if c.isalnum() or c == "_" else "_" for c in str(mapper.kind)) or "custom"
            if not (label[0].isalpha() or label[0] == "_"):
                label = "m_" + label
            object.__setattr__(self, "dispatch_order", ExplicitOrder(tuple(ordered), label))
        if isinstance(self.scheduler, str):
            if self.scheduler != "static_dispatch":
                raise _fail(path, "scheduler", self.scheduler,
                            "must be 'static_dispatch' or Persistent(...)")
        elif not isinstance(self.scheduler, Persistent):
            raise _fail(path, "scheduler", self.scheduler,
                        "must be 'static_dispatch' or Persistent(...)")
        object.__setattr__(self, "metadata", freeze_attrs(self.metadata))
        object.__setattr__(self, "buffers", tuple(self.buffers))
        object.__setattr__(self, "storage_groups", tuple(self.storage_groups))
        if not all(isinstance(item, Buffer) for item in self.buffers):
            raise _fail(path, "buffers", self.buffers, "must contain Buffer")
        if not all(isinstance(item, StorageGroup) for item in self.storage_groups):
            raise _fail(path, "storage_groups", self.storage_groups, "must contain StorageGroup")
        self._validate_structure()
        self.allocations()      # one identity per buffer/group name across the launch

    # -- on-chip allocations -------------------------------------------------

    def allocations(self) -> Tuple[Dict[str, Buffer], Dict[str, StorageGroup]]:
        """Every on-chip allocation of one work unit, by name.

        An allocation is identified by its buffer name and lives for the whole work
        unit (a TileLang ``alloc_*`` at kernel scope): a buffer used by several loops,
        a prologue or a plain sequence is one allocation.  Declarations come from
        ``Launch.buffers`` and from every pipeline; the same name must always describe
        the same storage, shape, dtype and slot count.
        """

        path = "Launch(%s)" % self.name
        buffers = {}  # type: Dict[str, Buffer]
        groups = {}  # type: Dict[str, StorageGroup]
        sources = [(self.buffers, self.storage_groups)]
        sources += [(loop.body.buffers, loop.body.storage_groups) for loop in self.loops() if isinstance(loop.body, Pipeline)]
        for declared, declared_groups in sources:
            for item in declared:
                known = buffers.setdefault(item.name, item)
                if (known.storage, tuple(known.shape), known.dtype, known.slots) != (item.storage, tuple(item.shape), item.dtype, item.slots):
                    raise _fail(path, "buffers", item.name, "is declared twice with different storage/shape/dtype/slots")
            for group in declared_groups:
                known_group = groups.setdefault(group.name, group)
                if known_group != group:
                    raise _fail(path, "storage_groups", group.name, "is declared twice with different members/slots")
        members = [member for group in groups.values() for member in group.members]
        if len(members) != len(set(members)):
            raise _fail(path, "storage_groups", members, "a buffer belongs to at most one storage group")
        for group in groups.values():
            missing = [member for member in group.members if member not in buffers]
            if missing:
                raise _fail(path, "storage_groups", group.name, "references undeclared buffers %s" % missing)
        return buffers, groups

    def storage_bytes(self) -> Dict[str, float]:
        """Declared SMEM/TMEM bytes of one work unit: each allocation once, alias/ring groups as ``slots x largest member``."""

        from ..ops.schema import DTypeSpec

        def size(item: Buffer) -> float:
            if not item.shape or not item.dtype:
                return 0.0
            return float(_product(item.shape)) * DTypeSpec(item.dtype).storage_bytes

        buffers, groups = self.allocations()
        totals = {}  # type: Dict[str, float]
        grouped = set()
        for group in groups.values():
            members = [buffers[name] for name in group.members]
            grouped.update(group.members)
            totals[members[0].storage] = totals.get(members[0].storage, 0.0) + group.slots * max(size(m) for m in members)
        for item in buffers.values():
            if item.name not in grouped and item.storage in ("smem", "tmem"):
                totals[item.storage] = totals.get(item.storage, 0.0) + item.slots * size(item)
        return totals

    def unsized_buffers(self) -> Tuple[str, ...]:
        """Buffer names that ops use but no declaration sizes (their storage cannot be checked)."""

        buffers, _groups = self.allocations()
        used = []  # type: List[str]
        for op in self.ops():
            for name in tuple(op.reads) + tuple(op.writes):
                known = buffers.get(name)
                if (known is None or not known.shape or not known.dtype) and name not in used:
                    used.append(name)
        return tuple(used)

    # -- derived views ------------------------------------------------------

    @property
    def work_count(self) -> int:
        return _product(item.extent for item in self.work_axes)

    @property
    def work_axis_names(self) -> Tuple[str, ...]:
        return tuple(item.name for item in self.work_axes)

    @property
    def physical_grid(self) -> Tuple[int, ...]:
        return self.grid if self.grid is not None else (self.work_count,)

    @property
    def cluster_size(self) -> int:
        return _product(self.cluster)

    @property
    def mapper(self) -> BlockMapper:
        return as_mapper(self.dispatch_order)

    def instance_coordinates(self) -> List[Tuple[int, ...]]:
        """Coordinates (one per work axis) of every work instance, in the mapper's dispatch order."""

        return self.mapper.coordinates([axis.extent for axis in self.work_axes])

    def effective_work_units(self) -> WorkUnits:
        """Return the declared work units, or one default group."""

        if self.work_units is not None:
            return self.work_units
        return WorkUnits(
            groups=(("default", WorkGroup()),),
            order=(WorkRun("default", self.work_count),),
        )

    def loops(self) -> Tuple[Loop, ...]:
        return tuple(loop for _chain, loop in walk_loops(self.body))

    def op_sites(self) -> Tuple[OpSite, ...]:
        return tuple(walk_ops(self.body, self.name))

    def ops(self) -> Tuple[Op, ...]:
        return tuple(site.op for site in self.op_sites())

    def loop(self, loop_id: str) -> Loop:
        for item in self.loops():
            if item.loop_id == loop_id:
                return item
        raise KeyError(loop_id)

    def bound_trip_count(self, loop: Loop, group: WorkGroup) -> int:
        """Resolve a loop's trip count for one work group."""

        override = group.trip_count(loop.loop_id)
        if isinstance(loop.trip_count, FromWorkGroup):
            if override is None:
                raise ContractError(
                    "Launch(%s)/Loop(%s): FromWorkGroup() has no trip count in the work group"
                    % (self.name, loop.loop_id)
                )
            return override
        if override is not None and override != loop.trip_count:
            raise ContractError(
                "Launch(%s)/Loop(%s): fixed trip_count=%d conflicts with work-group override %d"
                % (self.name, loop.loop_id, loop.trip_count, override)
            )
        return loop.trip_count

    # -- validation -----------------------------------------------------------

    def _validate_structure(self) -> None:
        path = "Launch(%s)" % self.name
        loops = walk_loops(self.body)
        loop_ids = [loop.loop_id for _chain, loop in loops]
        if len(loop_ids) != len(set(loop_ids)):
            raise _fail(path, "body", loop_ids, "loop ids must be unique within a launch")
        sites = walk_ops(self.body, self.name)
        op_ids = [site.op.op_id for site in sites]
        if len(op_ids) != len(set(op_ids)):
            dupes = sorted({x for x in op_ids if op_ids.count(x) > 1})
            raise _fail(path, "body", dupes, "op ids must be unique within a launch")
        access_ids = []
        for site in sites:
            access = getattr(site.op, "access", None)
            if access is not None and access.access_id is not None:
                access_ids.append(access.access_id)
        if len(access_ids) != len(set(access_ids)):
            raise _fail(path, "body", access_ids, "explicit access ids must be unique")
        # Tile access axes must name launch work axes or enclosing loop ids.
        axis_names = set(self.work_axis_names)
        for site in sites:
            access = getattr(site.op, "access", None)
            if access is None:
                continue
            allowed = axis_names | set(site.loop_ids)
            for axis in access.index_map.work_axes:
                if axis not in allowed:
                    raise _fail(
                        "%s/%s" % (path, site.op.op_id), "access.index_map", axis,
                        "must name a launch work axis or an enclosing loop id",
                    )
        # Allocation identity: same tensor => same allocation bytes.
        allocation = {}  # type: Dict[str, float]
        for site in sites:
            access = getattr(site.op, "access", None)
            if access is None:
                continue
            previous = allocation.get(access.tensor)
            if previous is not None and abs(previous - access.effective_allocation_bytes) > 1e-9:
                raise _fail(
                    "%s/%s" % (path, site.op.op_id), "access.allocation_bytes",
                    access.effective_allocation_bytes,
                    "tensor %s already declared allocation %r" % (access.tensor, previous),
                )
            allocation[access.tensor] = access.effective_allocation_bytes
        # Work-unit binding.
        units = self.effective_work_units()
        if units.total_count != self.work_count:
            raise _fail(
                path, "work_units.order", units.total_count,
                "must sum to the work-axis product %d" % self.work_count,
            )
        known_loops = set(loop_ids)
        for group_id, group in units.groups:
            for loop_id, _count in group.loop_trip_counts:
                if loop_id not in known_loops:
                    raise _fail(path, "work_units.groups[%s]" % group_id, loop_id,
                                "names a loop that is not in the launch body")
            for op_id, _fraction in group.effective_fractions:
                if op_id not in set(op_ids):
                    raise _fail(path, "work_units.groups[%s]" % group_id, op_id,
                                "names an op that is not in the launch body")
            for _chain, loop in loops:
                self.bound_trip_count(loop, group)
        if isinstance(self.scheduler, Persistent):
            self.scheduler.validate_coverage(units.total_count)
        if _product(self.physical_grid) < 1:
            raise _fail(path, "grid", self.physical_grid, "must launch at least one CTA")


@dataclass(frozen=True)
class LaunchEdge:
    """Dependency between two launches with an optional explicit gap."""

    source: str
    target: str
    gap_s: float = 0.0

    def __post_init__(self) -> None:
        path = "LaunchEdge(%s->%s)" % (self.source, self.target)
        _ident(self.source, path, "source")
        _ident(self.target, path, "target")
        if self.source == self.target:
            raise _fail(path, "target", self.target, "must differ from source")
        object.__setattr__(self, "gap_s", _nonneg_float(self.gap_s, path, "gap_s"))


@dataclass(frozen=True)
class Program:
    """Ordered launches plus inter-launch edges."""

    launches: Tuple[Launch, ...]
    edges: Tuple[LaunchEdge, ...] = field(default_factory=tuple)
    name: str = "program"
    metadata: Tuple[Tuple[str, Any], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        path = "Program(%s)" % self.name
        _ident(self.name, path, "name")
        launches = tuple(self.launches)
        if not launches or not all(isinstance(item, Launch) for item in launches):
            raise _fail(path, "launches", launches, "must be a non-empty tuple of Launch")
        names = [item.name for item in launches]
        if len(names) != len(set(names)):
            raise _fail(path, "launches", names, "launch names must be unique")
        object.__setattr__(self, "launches", launches)
        edges = tuple(self.edges)
        if not all(isinstance(item, LaunchEdge) for item in edges):
            raise _fail(path, "edges", edges, "must contain LaunchEdge")
        index = {name: position for position, name in enumerate(names)}
        for edge in edges:
            if edge.source not in index or edge.target not in index:
                raise _fail(path, "edges", (edge.source, edge.target),
                            "references an unknown launch")
            if index[edge.source] >= index[edge.target]:
                raise _fail(path, "edges", (edge.source, edge.target),
                            "must follow the declared sequential launch order")
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "metadata", freeze_attrs(self.metadata))

    def launch(self, name: str) -> Launch:
        for item in self.launches:
            if item.name == name:
                return item
        raise KeyError(name)

    def op_sites(self) -> Tuple[OpSite, ...]:
        return tuple(site for launch in self.launches for site in launch.op_sites())

    def op_paths(self) -> Tuple[str, ...]:
        return tuple(site.path for site in self.op_sites())

    def region_paths(self) -> Tuple[str, ...]:
        result = []
        for launch in self.launches:
            result.append(launch.name)
            for chain, loop in walk_loops(launch.body):
                result.append("/".join((launch.name,) + chain + (loop.loop_id,)))
        return tuple(result)


__all__ = [
    "StorageGroup", "STORAGE_GROUP_KINDS",
    "CutlassSm90", "register_block_mapper", "validate_block_order",
    "ACTOR_ORDERS", "ARCH_CAPABILITIES", "COMPUTE_ENGINES", "Actor", "Buffer", "BufferSlots", "Carry",
    "Compute", "ContractError", "Dependency", "EventRef", "FromWorkGroup",
    "INTERNAL_STORAGE", "Launch", "LaunchEdge", "Load", "Loop", "MOVE_ENGINES",
    "BlockMapper", "ExplicitOrder", "LinearBlockId", "Op", "OpSite", "PROGRAM_SCHEMA_VERSION", "PUBLIC_STORAGES",
    "PanelDispatch", "PanelSwizzle", "Persistent", "as_mapper", "mapper_from_canonical",
    "Pipeline", "Program", "Projection", "Region", "ResourceOrder", "Sequence",
    "Store", "Sync", "TileAccess", "WorkAxis", "WorkGroup", "WorkRun", "WorkUnits",
    "dispatch_coordinates", "done", "is_op", "is_region", "move_executed_bytes", "move_payload_bytes",
    "start", "touches_global", "walk_loops", "walk_ops",
]
