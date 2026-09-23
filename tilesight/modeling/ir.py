"""Frozen, hashable IR for the experimental TileSight modeling frontend.

The classes in this module contain no builder state.  Mutable construction is
isolated in :mod:`tilesight.modeling.frontend`; ``Kernel.build()`` snapshots it
into this IR before validation or lowering.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

from .errors import ModelingValidationError


_MEMORY_LEVELS = frozenset(("ddr", "l2", "smem", "tmem", "register"))
_HANDOFF_EXECUTION_SCOPES = frozenset(
    ("thread", "warp", "warpgroup", "cta", "cta_group")
)
_HANDOFF_SCOPE_BY_MEMORY = {
    "register": frozenset(("thread", "warp", "warpgroup")),
    "smem": frozenset(("cta",)),
    "tmem": frozenset(("warpgroup", "cta", "cta_group")),
}
_ACTOR_EXECUTION_SCOPES = frozenset(
    ("unspecified", "thread", "warp", "warpgroup", "cta", "cta_group")
)
_OWNERSHIP_SCOPES = frozenset(
    ("thread", "warp", "warpgroup", "cta", "cta_group")
)


def canonical_memory_level(value: Any) -> str:
    """Return the canonical name used by fusion traffic accounting.

    ``global`` is accepted only as an ergonomic input alias.  Frozen IR and
    reports always use ``ddr`` so there is one unambiguous off-chip level.
    """

    value = validate_identifier(value, "memory level").lower()
    if value == "global":
        value = "ddr"
    if value not in _MEMORY_LEVELS:
        raise ModelingValidationError(
            "memory level must be one of %s" % ", ".join(sorted(_MEMORY_LEVELS))
        )
    return value


def validate_identifier(value: Any, label: str, allow_empty: bool = False) -> str:
    """Validate a user-visible identifier used in stable owner paths."""

    if not isinstance(value, str):
        raise ModelingValidationError("%s must be a string" % label)
    if not allow_empty and not value:
        raise ModelingValidationError("%s must not be empty" % label)
    if "/" in value:
        raise ModelingValidationError("%s must not contain '/'" % label)
    return value


def freeze_value(value: Any) -> Any:
    """Recursively turn common metadata containers into hashable values."""

    if isinstance(value, Mapping):
        return tuple(
            sorted(
                (
                    validate_identifier(key, "metadata key"),
                    freeze_value(item),
                )
                for key, item in value.items()
            )
        )
    if isinstance(value, (list, tuple)):
        return tuple(freeze_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((freeze_value(item) for item in value), key=repr))
    try:
        hash(value)
    except TypeError as error:
        raise ModelingValidationError(
            "metadata value %r is not hashable or recursively freezable" % (value,)
        ) from error
    return value


def freeze_attrs(value: Any) -> Tuple[Tuple[str, Any], ...]:
    """Normalize mapping or pair-sequence metadata into a sorted tuple."""

    if value is None:
        return tuple()
    if isinstance(value, Mapping):
        items = value.items()
    else:
        try:
            items = tuple(value)
        except TypeError as error:
            raise ModelingValidationError("attrs must be a mapping or key/value pairs") from error
    normalized = tuple(
        (validate_identifier(key, "attribute key"), freeze_value(item))
        for key, item in items
    )
    keys = [key for key, _ in normalized]
    if len(keys) != len(set(keys)):
        raise ModelingValidationError("attrs contains duplicate keys")
    return tuple(sorted(normalized))


def _finite_nonnegative(value: float, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ModelingValidationError("%s must be numeric" % label) from error
    if not math.isfinite(result) or result < 0.0:
        raise ModelingValidationError("%s must be finite and non-negative" % label)
    return result


@dataclass(frozen=True)
class ResourceTiming:
    """Static service reservation in SI seconds for an exclusive resource."""

    resource: str
    service_time: float
    offset: float = 0.0

    def __post_init__(self) -> None:
        validate_identifier(self.resource, "resource name")
        service_time = _finite_nonnegative(self.service_time, "service_time")
        if service_time <= 0.0:
            raise ModelingValidationError("service_time must be positive")
        object.__setattr__(self, "service_time", service_time)
        object.__setattr__(self, "offset", _finite_nonnegative(self.offset, "offset"))


@dataclass(frozen=True)
class Timing:
    """Completion latency and resource service intervals, all in SI seconds."""

    latency: float
    resources: Tuple[ResourceTiming, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        latency = _finite_nonnegative(self.latency, "latency")
        object.__setattr__(self, "latency", latency)
        resources = tuple(self.resources)
        if not all(isinstance(resource, ResourceTiming) for resource in resources):
            raise ModelingValidationError("Timing.resources must contain ResourceTiming")
        names = [resource.resource for resource in resources]
        if len(names) != len(set(names)):
            raise ModelingValidationError(
                "a phase may reserve each exclusive resource at most once"
            )
        for resource in resources:
            end = resource.offset + resource.service_time
            tolerance = 1.0e-15 + 1.0e-12 * max(abs(end), abs(latency))
            if end > latency + tolerance:
                raise ModelingValidationError(
                    "resource %s reservation ends after phase latency"
                    % resource.resource
                )
        object.__setattr__(self, "resources", resources)


@dataclass(frozen=True)
class Work:
    """Architecture-independent semantic work attached to a phase."""

    kind: str
    flops: float = 0.0
    bytes: float = 0.0
    attrs: Tuple[Tuple[str, Any], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        validate_identifier(self.kind, "work kind")
        object.__setattr__(self, "flops", _finite_nonnegative(self.flops, "flops"))
        object.__setattr__(self, "bytes", _finite_nonnegative(self.bytes, "bytes"))
        object.__setattr__(self, "attrs", freeze_attrs(self.attrs))

    @classmethod
    def copy(cls, bytes: float, **attrs: Any) -> "Work":
        return cls("copy", bytes=bytes, attrs=attrs)

    @classmethod
    def mma(cls, flops: float, **attrs: Any) -> "Work":
        return cls("mma", flops=flops, attrs=attrs)

    @classmethod
    def pointwise(cls, flops: float, **attrs: Any) -> "Work":
        return cls("pointwise", flops=flops, attrs=attrs)

    @classmethod
    def reduce(cls, flops: float, **attrs: Any) -> "Work":
        return cls("reduce", flops=flops, attrs=attrs)

    @classmethod
    def opaque(cls, kind: str = "opaque", **attrs: Any) -> "Work":
        return cls(kind, attrs=attrs)


@dataclass(frozen=True)
class ExecutionDomain:
    """One actor/storage ownership domain inside a logical CTA program.

    ``instances_per_cta`` is deliberately separate from launch residency.
    For example, two producer warps use ``scope='warp', instances_per_cta=2``;
    four resident CTAs then create eight warp-domain replicas per SM.  A
    CTA-group domain is the exception: its per-SM replicas are divided by the
    launch cluster size by the liveness pass.
    """

    scope: str
    instances_per_cta: int = 1
    members_per_instance: int = 1
    provenance: str = "explicit"

    def __post_init__(self) -> None:
        validate_identifier(self.scope, "execution-domain scope")
        if self.scope not in _OWNERSHIP_SCOPES:
            raise ModelingValidationError(
                "execution-domain scope must be one of %s"
                % ", ".join(sorted(_OWNERSHIP_SCOPES))
            )
        if (
            not isinstance(self.instances_per_cta, int)
            or isinstance(self.instances_per_cta, bool)
            or self.instances_per_cta <= 0
        ):
            raise ModelingValidationError(
                "execution-domain instances_per_cta must be a positive integer"
            )
        if (
            not isinstance(self.members_per_instance, int)
            or isinstance(self.members_per_instance, bool)
            or self.members_per_instance <= 0
        ):
            raise ModelingValidationError(
                "execution-domain members_per_instance must be a positive integer"
            )
        validate_identifier(self.provenance, "execution-domain provenance")


@dataclass(frozen=True)
class OwnershipMap:
    """Optional value/storage ownership proof beyond a coarse scope string.

    ``layout`` names the distributed layout and ``index_map`` records a stable,
    hashable index/lane mapping.  Two maps are proof-compatible only when all
    three semantic fields match; merely sharing ``scope='warp'`` is not a
    same-lane proof.
    """

    domain: ExecutionDomain
    layout: str
    index_map: Tuple[Tuple[str, Any], ...] = field(default_factory=tuple)
    provenance: str = "explicit"

    def __post_init__(self) -> None:
        if not isinstance(self.domain, ExecutionDomain):
            raise ModelingValidationError(
                "ownership map domain must be ExecutionDomain"
            )
        validate_identifier(self.layout, "ownership layout")
        object.__setattr__(self, "index_map", freeze_attrs(self.index_map))
        validate_identifier(self.provenance, "ownership-map provenance")

    def compatible_with(self, other: Any) -> bool:
        return (
            isinstance(other, OwnershipMap)
            and self.domain.scope == other.domain.scope
            and self.domain.instances_per_cta == other.domain.instances_per_cta
            and self.domain.members_per_instance == other.domain.members_per_instance
            and self.layout == other.layout
            and self.index_map == other.index_map
        )

    def summary(self) -> Dict[str, Any]:
        return {
            "scope": self.domain.scope,
            "instances_per_cta": self.domain.instances_per_cta,
            "members_per_instance": self.domain.members_per_instance,
            "domain_provenance": self.domain.provenance,
            "layout": self.layout,
            "index_map": dict(self.index_map),
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class Buffer:
    """A named logical storage object owned by one periodic loop."""

    name: str
    scope: str
    owner: str
    shape: Tuple[Any, ...] = field(default_factory=tuple)
    dtype: str = ""
    slots: int = 1
    execution_scope: str = "cta"
    alias_group: Optional[str] = None
    ownership: Optional[OwnershipMap] = None

    def __post_init__(self) -> None:
        validate_identifier(self.name, "buffer name")
        validate_identifier(self.scope, "buffer storage scope")
        validate_identifier(self.dtype, "buffer dtype", allow_empty=True)
        validate_identifier(self.execution_scope, "buffer execution scope")
        if self.alias_group is not None:
            validate_identifier(self.alias_group, "buffer alias_group")
        if self.ownership is not None:
            if not isinstance(self.ownership, OwnershipMap):
                raise ModelingValidationError(
                    "buffer ownership must be OwnershipMap or None"
                )
            if self.ownership.domain.scope != self.execution_scope:
                raise ModelingValidationError(
                    "buffer ownership domain scope must equal execution_scope"
                )
        if not isinstance(self.owner, str) or not self.owner:
            raise ModelingValidationError("buffer owner must be an internal owner string")
        object.__setattr__(self, "shape", tuple(freeze_value(item) for item in self.shape))
        if not isinstance(self.slots, int) or isinstance(self.slots, bool) or self.slots <= 0:
            raise ModelingValidationError("buffer slots must be a positive integer")

    @property
    def qualified_name(self) -> str:
        return "%s/%s" % (self.owner, self.name)


@dataclass(frozen=True)
class TensorValue:
    """A logical value crossing one or more modeled fragments.

    ``bytes`` is the exact payload size used by traffic accounting.  Shape and
    dtype remain descriptive because symbolic shapes cannot always be reduced
    to bytes at frontend construction time.
    """

    name: str
    owner: str
    shape: Tuple[Any, ...]
    dtype: str
    bytes: float
    layout: str
    ownership: Optional[OwnershipMap] = None

    def __post_init__(self) -> None:
        validate_identifier(self.name, "tensor value name")
        validate_identifier(self.owner, "tensor value owner")
        validate_identifier(self.dtype, "tensor value dtype")
        validate_identifier(self.layout, "tensor value layout")
        if self.ownership is not None and not isinstance(
            self.ownership, OwnershipMap
        ):
            raise ModelingValidationError(
                "tensor value ownership must be OwnershipMap or None"
            )
        if self.ownership is not None and self.ownership.layout != self.layout:
            raise ModelingValidationError(
                "tensor value ownership layout must equal value layout"
            )
        object.__setattr__(self, "shape", tuple(freeze_value(x) for x in self.shape))
        size = _finite_nonnegative(self.bytes, "tensor value bytes")
        if size <= 0.0:
            raise ModelingValidationError("tensor value bytes must be positive")
        object.__setattr__(self, "bytes", size)

    @property
    def qualified_name(self) -> str:
        return "%s/%s" % (self.owner, self.name)


@dataclass(frozen=True)
class Event:
    """The start or completion event of a phase."""

    phase_name: str
    owner: str
    kind: str

    def __post_init__(self) -> None:
        validate_identifier(self.phase_name, "event phase name")
        if not isinstance(self.owner, str) or not self.owner:
            raise ModelingValidationError("event owner must be an internal owner string")
        if self.kind not in ("start", "done"):
            raise ModelingValidationError("event kind must be 'start' or 'done'")


@dataclass(frozen=True)
class MemoryAccess:
    """One exact read or write traffic item at one canonical memory level."""

    name: str
    owner: str
    value: TensorValue
    mode: str
    memory_level: str
    bytes: float
    event: Event
    fragment: str
    layout: str
    accounting: str = "exact"

    def __post_init__(self) -> None:
        validate_identifier(self.name, "memory access name")
        validate_identifier(self.owner, "memory access owner")
        if not isinstance(self.value, TensorValue):
            raise ModelingValidationError("memory access value must be TensorValue")
        if self.value.owner != self.owner:
            raise ModelingValidationError("memory access value belongs to another kernel")
        mode = validate_identifier(self.mode, "memory access mode").lower()
        if mode not in ("read", "write"):
            raise ModelingValidationError("memory access mode must be 'read' or 'write'")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(
            self, "memory_level", canonical_memory_level(self.memory_level)
        )
        size = _finite_nonnegative(self.bytes, "memory access bytes")
        if size <= 0.0:
            raise ModelingValidationError("memory access bytes must be positive")
        object.__setattr__(self, "bytes", size)
        if not isinstance(self.event, Event):
            raise ModelingValidationError("memory access event must be Event")
        validate_identifier(self.fragment, "memory access fragment")
        validate_identifier(self.layout, "memory access layout")
        if self.accounting not in ("exact", "generic_equal_bytes"):
            raise ModelingValidationError(
                "memory access accounting must be 'exact' or 'generic_equal_bytes'"
            )


@dataclass(frozen=True)
class LayoutMapping:
    """Explicit, lossless layout mapping for an eliminated materialization."""

    producer_layout: str
    via_layout: str
    consumer_layout: str
    kind: str = "identity"
    bijective: bool = True
    attrs: Tuple[Tuple[str, Any], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        validate_identifier(self.producer_layout, "producer layout")
        validate_identifier(self.via_layout, "handoff storage layout")
        validate_identifier(self.consumer_layout, "consumer layout")
        if self.kind not in ("identity", "reshape", "transpose", "swizzle", "custom"):
            raise ModelingValidationError("unsupported layout mapping kind %r" % self.kind)
        if not isinstance(self.bijective, bool) or not self.bijective:
            raise ModelingValidationError(
                "traffic elimination requires an explicitly bijective layout mapping"
            )
        if self.kind == "identity" and not (
            self.producer_layout == self.via_layout == self.consumer_layout
        ):
            raise ModelingValidationError(
                "identity layout mapping requires identical producer/via/consumer layouts"
            )
        object.__setattr__(self, "attrs", freeze_attrs(self.attrs))

    @classmethod
    def identity(cls, layout: str) -> "LayoutMapping":
        return cls(layout, layout, layout)


@dataclass(frozen=True)
class Handoff:
    """Replace explicit baseline stores/loads with one on-chip transfer.

    A handoff writes ``bytes`` once into ``via`` storage and reads it once per
    distinct consumer.  The referenced baseline accesses are removed exactly;
    no cache or traffic inference is hidden in this IR.
    """

    name: str
    owner: str
    value: TensorValue
    baseline_writes: Tuple[MemoryAccess, ...]
    baseline_reads: Tuple[MemoryAccess, ...]
    via: str
    bytes: float
    execution_scope: str
    layout: LayoutMapping
    acquire: Event
    releases: Tuple[Event, ...]
    baseline_storage: str = "ddr"
    producer_source: str = "register"
    consumer_targets: Tuple[str, ...] = field(default_factory=lambda: ("register",))
    iteration_distance: int = 0
    slots: int = 1
    alias_group: Optional[str] = None
    ownership: Optional[OwnershipMap] = None

    def __post_init__(self) -> None:
        validate_identifier(self.name, "handoff name")
        validate_identifier(self.owner, "handoff owner")
        if not isinstance(self.value, TensorValue) or self.value.owner != self.owner:
            raise ModelingValidationError("handoff value belongs to another kernel")
        writes = tuple(self.baseline_writes)
        reads = tuple(self.baseline_reads)
        if not writes or not reads:
            raise ModelingValidationError(
                "handoff needs explicit baseline write and read accesses"
            )
        if not all(isinstance(x, MemoryAccess) for x in writes + reads):
            raise ModelingValidationError("handoff accesses must be MemoryAccess")
        if any(x.mode != "write" for x in writes) or any(x.mode != "read" for x in reads):
            raise ModelingValidationError(
                "handoff baseline_writes/reads contain the wrong access mode"
            )
        if any(x.owner != self.owner or x.value != self.value for x in writes + reads):
            raise ModelingValidationError(
                "handoff accesses must reference its value in the same kernel"
            )
        names = [x.name for x in writes + reads]
        if len(names) != len(set(names)):
            raise ModelingValidationError("handoff baseline accesses must be unique")
        write_levels = [x.memory_level for x in writes]
        if len(write_levels) != len(set(write_levels)):
            raise ModelingValidationError(
                "handoff may eliminate at most one baseline write per memory level"
            )
        size = _finite_nonnegative(self.bytes, "handoff bytes")
        if size <= 0.0:
            raise ModelingValidationError("handoff bytes must be positive")
        tolerance = 1.0e-9 * max(size, self.value.bytes)
        if size > self.value.bytes + tolerance:
            raise ModelingValidationError("handoff bytes cannot exceed tensor value bytes")
        if any(
            x.accounting == "generic_equal_bytes"
            and abs(x.bytes - size) > tolerance
            for x in writes + reads
        ):
            raise ModelingValidationError(
                "generic equal-byte baseline access differs from handoff payload bytes"
            )
        object.__setattr__(self, "bytes", size)
        via = canonical_memory_level(self.via)
        if via in ("ddr", "l2"):
            raise ModelingValidationError(
                "fused handoff via must be register, smem, or tmem"
            )
        object.__setattr__(self, "via", via)
        validate_identifier(self.execution_scope, "handoff execution scope")
        if self.execution_scope not in _HANDOFF_EXECUTION_SCOPES:
            raise ModelingValidationError(
                "handoff execution scope must be one of %s"
                % ", ".join(sorted(_HANDOFF_EXECUTION_SCOPES))
            )
        if self.execution_scope not in _HANDOFF_SCOPE_BY_MEMORY[via]:
            raise ModelingValidationError(
                "%s handoff does not support execution_scope=%s"
                % (via, self.execution_scope)
            )
        if not isinstance(self.layout, LayoutMapping):
            raise ModelingValidationError("handoff layout must be LayoutMapping")
        producer_layouts = set(x.layout for x in writes)
        consumer_layouts = set(x.layout for x in reads)
        if producer_layouts != {self.layout.producer_layout}:
            raise ModelingValidationError(
                "handoff layout producer side does not match baseline writes"
            )
        if consumer_layouts != {self.layout.consumer_layout}:
            raise ModelingValidationError(
                "handoff layout consumer side does not match baseline reads"
            )
        if not isinstance(self.acquire, Event):
            raise ModelingValidationError("handoff acquire must be Event")
        if any(x.event != self.acquire for x in writes):
            raise ModelingValidationError(
                "handoff acquire must equal every baseline producer event"
            )
        if any(x.event.kind != "start" for x in reads):
            raise ModelingValidationError(
                "handoff baseline consumer reads must occur at phase.start"
            )
        releases = tuple(self.releases)
        if not releases or not all(isinstance(x, Event) for x in releases):
            raise ModelingValidationError("handoff releases must contain Events")
        if any(x.kind != "done" for x in releases):
            raise ModelingValidationError("handoff release events must be phase.done")
        consumer_order = []
        for access in reads:
            key = (access.fragment, access.event.owner, access.event.phase_name)
            if key not in consumer_order:
                consumer_order.append(key)
        release_phase_order = [(x.owner, x.phase_name) for x in releases]
        expected_release_order = [(owner, phase) for _fragment, owner, phase in consumer_order]
        if release_phase_order != expected_release_order:
            raise ModelingValidationError(
                "handoff needs one release per consumer access in consumer order"
            )
        object.__setattr__(
            self, "baseline_storage", canonical_memory_level(self.baseline_storage)
        )
        if self.baseline_storage == "l2":
            raise ModelingValidationError(
                "L2 is passive cache traffic, not logical materialization storage"
            )
        if self.baseline_storage not in set(write_levels):
            raise ModelingValidationError(
                "handoff baseline storage must have explicit baseline traffic"
            )
        object.__setattr__(
            self, "producer_source", canonical_memory_level(self.producer_source)
        )
        if self.producer_source == "l2":
            raise ModelingValidationError(
                "L2 is passive cache traffic, not a programmed producer source"
            )
        consumer_targets = tuple(
            canonical_memory_level(x) for x in self.consumer_targets
        )
        if "l2" in consumer_targets:
            raise ModelingValidationError(
                "L2 is passive cache traffic, not a programmed consumer target"
            )
        if len(consumer_targets) != len(consumer_order):
            raise ModelingValidationError(
                "handoff needs one target memory level per consumer access"
            )
        object.__setattr__(self, "consumer_targets", consumer_targets)
        if not isinstance(self.iteration_distance, int) or isinstance(
            self.iteration_distance, bool
        ):
            raise ModelingValidationError("handoff iteration_distance must be an integer")
        if self.iteration_distance != 0:
            raise ModelingValidationError(
                "v0 handoff supports same-iteration consumers only; "
                "cross-iteration reuse needs distance-aware token lowering"
            )
        if not isinstance(self.slots, int) or isinstance(self.slots, bool) or self.slots <= 0:
            raise ModelingValidationError("handoff slots must be a positive integer")
        if self.alias_group is not None:
            validate_identifier(self.alias_group, "handoff alias_group")
        if self.ownership is not None:
            if not isinstance(self.ownership, OwnershipMap):
                raise ModelingValidationError(
                    "handoff ownership must be OwnershipMap or None"
                )
            if self.ownership.domain.scope != self.execution_scope:
                raise ModelingValidationError(
                    "handoff ownership domain scope must equal execution_scope"
                )
            if self.ownership.layout != self.layout.via_layout:
                raise ModelingValidationError(
                    "handoff ownership layout must equal handoff via layout"
                )
        object.__setattr__(self, "baseline_writes", writes)
        object.__setattr__(self, "baseline_reads", reads)
        object.__setattr__(self, "releases", releases)


@dataclass(frozen=True)
class FusionPlan:
    """A structurally explicit collection of materialization replacements."""

    name: str
    owner: str
    handoffs: Tuple[Handoff, ...]

    def __post_init__(self) -> None:
        validate_identifier(self.name, "fusion plan name")
        validate_identifier(self.owner, "fusion plan owner")
        handoffs = tuple(self.handoffs)
        if not handoffs or not all(isinstance(x, Handoff) for x in handoffs):
            raise ModelingValidationError("fusion plan needs at least one Handoff")
        if any(x.owner != self.owner for x in handoffs):
            raise ModelingValidationError("fusion plan handoff belongs to another kernel")
        names = [x.name for x in handoffs]
        if len(names) != len(set(names)):
            raise ModelingValidationError("fusion plan contains duplicate handoff names")
        object.__setattr__(self, "handoffs", handoffs)


@dataclass(frozen=True)
class Occurrence:
    """A phase position in a source-faithful unrolled schedule window.

    ``window_offset`` labels which logical loop iteration the source schedule
    placed in this local window.  It is not a producer/consumer dependency
    distance; semantic buffer flow remains same-iteration unless stated by an
    explicit carry/dependency.
    """

    phase: "Phase"
    window_offset: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.phase, Phase):
            raise ModelingValidationError("occurrence phase must be Phase")
        if not isinstance(self.window_offset, int) or isinstance(
            self.window_offset, bool
        ):
            raise ModelingValidationError("occurrence window_offset must be an integer")

    @property
    def iteration_offset(self) -> int:
        """Compatibility alias; new frontend code should use ``window_offset``."""

        return self.window_offset

@dataclass(frozen=True)
class Phase:
    """One semantic unit of work owned by an actor."""

    name: str
    actor: str
    owner: str
    work: Work
    timing: Optional[Timing] = None
    reads: Tuple[Buffer, ...] = field(default_factory=tuple)
    writes: Tuple[Buffer, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        validate_identifier(self.name, "phase name")
        validate_identifier(self.actor, "phase actor")
        if not isinstance(self.owner, str) or not self.owner:
            raise ModelingValidationError("phase owner must be an internal owner string")
        if not isinstance(self.work, Work):
            raise ModelingValidationError("phase work must be Work")
        if self.timing is not None and not isinstance(self.timing, Timing):
            raise ModelingValidationError("phase timing must be Timing or None")
        reads = tuple(self.reads)
        writes = tuple(self.writes)
        for buffer in reads + writes:
            if not isinstance(buffer, Buffer):
                raise ModelingValidationError("phase reads/writes must contain Buffer")
            if buffer.owner != self.owner:
                raise ModelingValidationError(
                    "phase %s references buffer %s from another loop"
                    % (self.name, buffer.name)
                )
        object.__setattr__(self, "reads", reads)
        object.__setattr__(self, "writes", writes)

    def at(self, window_offset: int) -> Occurrence:
        """Place this phase at ``window_offset`` in an actor's cyclic window."""

        return Occurrence(self, window_offset)

    @property
    def start(self) -> Event:
        return Event(self.name, self.owner, "start")

    @property
    def done(self) -> Event:
        return Event(self.name, self.owner, "done")


@dataclass(frozen=True)
class State:
    """A named loop-carried value, optionally bound to physical storage."""

    name: str
    owner: str
    storage: Optional[Buffer] = None

    def __post_init__(self) -> None:
        validate_identifier(self.name, "state name")
        if not isinstance(self.owner, str) or not self.owner:
            raise ModelingValidationError("state owner must be an internal owner string")
        if self.storage is not None:
            if not isinstance(self.storage, Buffer):
                raise ModelingValidationError("state storage must be Buffer or None")
            if self.storage.owner != self.owner:
                raise ModelingValidationError("state storage belongs to another periodic loop")


@dataclass(frozen=True)
class Dependency:
    """An explicit event dependency in a periodic loop."""

    source: Event
    target: Event
    iteration_distance: int = 0
    lag: float = 0.0
    name: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.source, Event) or not isinstance(self.target, Event):
            raise ModelingValidationError("dependency endpoints must be Event")
        if self.source.owner != self.target.owner:
            raise ModelingValidationError("dependency endpoints belong to different loops")
        if not isinstance(self.iteration_distance, int) or isinstance(
            self.iteration_distance, bool
        ):
            raise ModelingValidationError("dependency iteration_distance must be an integer")
        object.__setattr__(self, "lag", _finite_nonnegative(self.lag, "dependency lag"))
        validate_identifier(self.name, "dependency name", allow_empty=True)


@dataclass(frozen=True)
class StateCarry:
    """A dependency labelled by the state carried across iterations."""

    state: State
    source: Event
    target: Event
    iteration_distance: int
    lag: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.state, State):
            raise ModelingValidationError("carry state must be State")
        if not isinstance(self.source, Event) or not isinstance(self.target, Event):
            raise ModelingValidationError("carry endpoints must be Event")
        if self.source.owner != self.state.owner or self.target.owner != self.state.owner:
            raise ModelingValidationError("carry state/events belong to different loops")
        if not isinstance(self.iteration_distance, int) or isinstance(
            self.iteration_distance, bool
        ) or self.iteration_distance <= 0:
            raise ModelingValidationError("carry iteration_distance must be positive")
        object.__setattr__(self, "lag", _finite_nonnegative(self.lag, "carry lag"))


@dataclass(frozen=True)
class Lifetime:
    """Semantic acquire-to-release lifetime for one buffer."""

    buffer: Buffer
    acquire: Event
    release: Event
    minimum_residence: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.buffer, Buffer):
            raise ModelingValidationError("lifetime buffer must be Buffer")
        if not isinstance(self.acquire, Event) or not isinstance(self.release, Event):
            raise ModelingValidationError("lifetime endpoints must be Event")
        if self.acquire.owner != self.buffer.owner or self.release.owner != self.buffer.owner:
            raise ModelingValidationError("lifetime buffer/events belong to different loops")
        object.__setattr__(
            self,
            "minimum_residence",
            _finite_nonnegative(self.minimum_residence, "minimum_residence"),
        )


@dataclass(frozen=True)
class PipelineBuffer:
    """A bounded repeated lifetime that lowers to a FIFO TokenBuffer."""

    buffer: Buffer
    acquire: Event
    release: Event
    capacity: int
    minimum_residence: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.buffer, Buffer):
            raise ModelingValidationError("pipeline buffer must be Buffer")
        if not isinstance(self.acquire, Event) or not isinstance(self.release, Event):
            raise ModelingValidationError("pipeline-buffer endpoints must be Event")
        if self.acquire.owner != self.buffer.owner or self.release.owner != self.buffer.owner:
            raise ModelingValidationError(
                "pipeline buffer/events belong to different loops"
            )
        if not isinstance(self.capacity, int) or isinstance(self.capacity, bool):
            raise ModelingValidationError("pipeline-buffer capacity must be an integer")
        if self.capacity <= 0 or self.capacity > self.buffer.slots:
            raise ModelingValidationError(
                "pipeline-buffer capacity must be between 1 and buffer.slots"
            )
        object.__setattr__(
            self,
            "minimum_residence",
            _finite_nonnegative(self.minimum_residence, "minimum_residence"),
        )


@dataclass(frozen=True)
class ActorIR:
    name: str
    serial_resource: Optional[str]
    phases: Tuple[Phase, ...]
    sequence: Tuple[Occurrence, ...] = field(default_factory=tuple)
    order: str = "issue"
    execution_scope: str = "unspecified"
    execution_domain: Optional[ExecutionDomain] = None

    def __post_init__(self) -> None:
        validate_identifier(self.name, "actor name")
        if self.serial_resource is not None:
            validate_identifier(self.serial_resource, "serial_resource")
        object.__setattr__(self, "phases", tuple(self.phases))
        object.__setattr__(self, "sequence", tuple(self.sequence))
        if self.order not in ("issue", "completion"):
            raise ModelingValidationError("actor sequence order must be 'issue' or 'completion'")
        validate_identifier(self.execution_scope, "actor execution_scope")
        if self.execution_scope not in _ACTOR_EXECUTION_SCOPES:
            raise ModelingValidationError(
                "actor execution_scope must be one of %s"
                % ", ".join(sorted(_ACTOR_EXECUTION_SCOPES))
            )
        if self.execution_domain is not None:
            if not isinstance(self.execution_domain, ExecutionDomain):
                raise ModelingValidationError(
                    "actor execution_domain must be ExecutionDomain or None"
                )
            if self.execution_scope == "unspecified":
                raise ModelingValidationError(
                    "actor with execution_domain needs an explicit execution_scope"
                )
            if self.execution_domain.scope != self.execution_scope:
                raise ModelingValidationError(
                    "actor execution_domain scope must equal execution_scope"
                )
        if not all(isinstance(phase, Phase) for phase in self.phases):
            raise ModelingValidationError("ActorIR.phases must contain Phase")
        if not all(isinstance(item, Occurrence) for item in self.sequence):
            raise ModelingValidationError("ActorIR.sequence must contain Occurrence")


@dataclass(frozen=True)
class ResourceSequenceIR:
    """An independent cyclic arbitration order for one exclusive resource."""

    resource: str
    sequence: Tuple[Occurrence, ...]
    source: str = "expert"

    def __post_init__(self) -> None:
        validate_identifier(self.resource, "resource-sequence resource")
        validate_identifier(self.source, "resource-sequence source")
        object.__setattr__(self, "sequence", tuple(self.sequence))
        if not self.sequence or not all(
            isinstance(item, Occurrence) for item in self.sequence
        ):
            raise ModelingValidationError(
                "resource sequence must contain Phase.at(window_offset) occurrences"
            )


@dataclass(frozen=True)
class PeriodicLoopIR:
    name: str
    owner: str
    iterations: Any
    stages: Optional[int]
    buffers: Tuple[Buffer, ...]
    actors: Tuple[ActorIR, ...]
    dependencies: Tuple[Dependency, ...]
    states: Tuple[State, ...]
    carries: Tuple[StateCarry, ...]
    lifetimes: Tuple[Lifetime, ...]
    pipeline_buffers: Tuple[PipelineBuffer, ...]
    resource_sequences: Tuple[ResourceSequenceIR, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        validate_identifier(self.name, "periodic loop name")
        if not isinstance(self.owner, str) or not self.owner:
            raise ModelingValidationError("loop owner must be an internal owner string")
        for field_name in (
            "buffers",
            "actors",
            "dependencies",
            "states",
            "carries",
            "lifetimes",
            "pipeline_buffers",
            "resource_sequences",
        ):
            object.__setattr__(self, field_name, tuple(getattr(self, field_name)))
        object.__setattr__(self, "iterations", freeze_value(self.iterations))
        if self.stages is not None and (
            not isinstance(self.stages, int)
            or isinstance(self.stages, bool)
            or self.stages <= 0
        ):
            raise ModelingValidationError("loop stages must be a positive integer or None")
        expected_types = (
            ("buffers", Buffer),
            ("actors", ActorIR),
            ("dependencies", Dependency),
            ("states", State),
            ("carries", StateCarry),
            ("lifetimes", Lifetime),
            ("pipeline_buffers", PipelineBuffer),
            ("resource_sequences", ResourceSequenceIR),
        )
        for field_name, expected_type in expected_types:
            if not all(
                isinstance(item, expected_type) for item in getattr(self, field_name)
            ):
                raise ModelingValidationError(
                    "PeriodicLoopIR.%s contains an invalid item" % field_name
                )
        for actor in self.actors:
            if actor.serial_resource is None:
                continue
            matches = [
                item
                for item in self.resource_sequences
                if item.source == "actor_%s" % actor.name
                and item.resource == actor.serial_resource
                and item.sequence == actor.sequence
            ]
            if len(matches) != 1:
                raise ModelingValidationError(
                    "serial_resource compatibility alias must map to exactly one "
                    "equivalent resource sequence"
                )

    @property
    def phases(self) -> Tuple[Phase, ...]:
        return tuple(phase for actor in self.actors for phase in actor.phases)

    def summary(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "iterations": self.iterations,
            "stages": self.stages,
            "buffers": tuple(buffer.name for buffer in self.buffers),
            "buffer_specs": tuple(
                {
                    "name": buffer.name,
                    "storage_scope": buffer.scope,
                    "execution_scope": buffer.execution_scope,
                    "shape": buffer.shape,
                    "dtype": buffer.dtype,
                    "slots": buffer.slots,
                    "alias_group": buffer.alias_group,
                    "ownership": None
                    if buffer.ownership is None
                    else buffer.ownership.summary(),
                }
                for buffer in self.buffers
            ),
            "phases": tuple(phase.name for phase in self.phases),
            "phase_specs": tuple(
                {
                    "name": phase.name,
                    "actor": phase.actor,
                    "work": {
                        "kind": phase.work.kind,
                        "flops": phase.work.flops,
                        "bytes": phase.work.bytes,
                        "attrs": dict(phase.work.attrs),
                    },
                    "timing": None
                    if phase.timing is None
                    else {
                        "latency_s": phase.timing.latency,
                        "resources": tuple(
                            {
                                "resource": item.resource,
                                "service_time_s": item.service_time,
                                "offset_s": item.offset,
                            }
                            for item in phase.timing.resources
                        ),
                    },
                    "reads": tuple(buffer.name for buffer in phase.reads),
                    "writes": tuple(buffer.name for buffer in phase.writes),
                }
                for phase in self.phases
            ),
            "actor_specs": tuple(
                {
                    "name": actor.name,
                    "execution_scope": actor.execution_scope,
                    "execution_domain": None
                    if actor.execution_domain is None
                    else {
                        "scope": actor.execution_domain.scope,
                        "instances_per_cta": actor.execution_domain.instances_per_cta,
                        "members_per_instance": actor.execution_domain.members_per_instance,
                        "provenance": actor.execution_domain.provenance,
                    },
                    "serial_resource": actor.serial_resource,
                    "order": actor.order,
                    "phases": tuple(phase.name for phase in actor.phases),
                }
                for actor in self.actors
            ),
            "actor_sequences": tuple(
                {
                    "actor": actor.name,
                    "execution_scope": actor.execution_scope,
                    "execution_domain": None
                    if actor.execution_domain is None
                    else {
                        "scope": actor.execution_domain.scope,
                        "instances_per_cta": actor.execution_domain.instances_per_cta,
                        "members_per_instance": actor.execution_domain.members_per_instance,
                        "provenance": actor.execution_domain.provenance,
                    },
                    "serial_resource": actor.serial_resource,
                    "order": actor.order,
                    "cyclic": True,
                    "occurrences": tuple(
                        {
                            "phase": occurrence.phase.name,
                            "window_offset": occurrence.window_offset,
                        }
                        for occurrence in actor.sequence
                    ),
                }
                for actor in self.actors
                if actor.sequence
            ),
            "resource_sequences": tuple(
                {
                    "resource": item.resource,
                    "source": item.source,
                    "cyclic": True,
                    "occurrences": tuple(
                        {
                            "phase": occurrence.phase.name,
                            "window_offset": occurrence.window_offset,
                        }
                        for occurrence in item.sequence
                    ),
                }
                for item in self.resource_sequences
            ),
            "dependencies": tuple(
                {
                    "source": dependency.source.phase_name,
                    "source_event": dependency.source.kind,
                    "target": dependency.target.phase_name,
                    "target_event": dependency.target.kind,
                    "iteration_distance": dependency.iteration_distance,
                    "lag_s": dependency.lag,
                    "name": dependency.name,
                }
                for dependency in self.dependencies
            ),
            "states": tuple(
                {
                    "name": state.name,
                    "storage": None if state.storage is None else state.storage.name,
                }
                for state in self.states
            ),
            "carries": tuple(
                {
                    "state": carry.state.name,
                    "state_storage": None
                    if carry.state.storage is None
                    else carry.state.storage.name,
                    "source": carry.source.phase_name,
                    "source_event": carry.source.kind,
                    "target": carry.target.phase_name,
                    "target_event": carry.target.kind,
                    "iteration_distance": carry.iteration_distance,
                    "lag_s": carry.lag,
                }
                for carry in self.carries
            ),
            "lifetimes": tuple(
                {
                    "buffer": lifetime.buffer.name,
                    "acquire": lifetime.acquire.phase_name,
                    "acquire_event": lifetime.acquire.kind,
                    "release": lifetime.release.phase_name,
                    "release_event": lifetime.release.kind,
                    "minimum_residence_s": lifetime.minimum_residence,
                }
                for lifetime in self.lifetimes
            ),
            "pipeline_buffers": tuple(
                {
                    "buffer": item.buffer.name,
                    "capacity": item.capacity,
                    "acquire": item.acquire.phase_name,
                    "release": item.release.phase_name,
                }
                for item in self.pipeline_buffers
            ),
        }


@dataclass(frozen=True)
class LaunchIR:
    name: str
    work_grid: Tuple[Any, ...]
    physical_grid: Tuple[Any, ...]
    threads: int
    cluster: Tuple[int, ...]
    residency: Any
    scheduler: str
    swizzle: Tuple[Tuple[str, Any], ...]
    periodic_loops: Tuple[PeriodicLoopIR, ...]

    def __post_init__(self) -> None:
        validate_identifier(self.name, "launch name")
        validate_identifier(self.scheduler, "launch scheduler")
        object.__setattr__(
            self,
            "work_grid",
            tuple(freeze_value(item) for item in self.work_grid),
        )
        object.__setattr__(
            self,
            "physical_grid",
            tuple(freeze_value(item) for item in self.physical_grid),
        )
        object.__setattr__(self, "cluster", tuple(self.cluster))
        object.__setattr__(self, "residency", freeze_value(self.residency))
        object.__setattr__(self, "swizzle", freeze_attrs(self.swizzle))
        object.__setattr__(self, "periodic_loops", tuple(self.periodic_loops))
        if not self.work_grid or not self.physical_grid:
            raise ModelingValidationError("launch work/physical grids must not be empty")
        if not (
            self.residency == "auto"
            or (
                isinstance(self.residency, int)
                and not isinstance(self.residency, bool)
                and self.residency > 0
            )
        ):
            raise ModelingValidationError(
                "launch residency must be 'auto' or a positive CTA count"
            )
        if not isinstance(self.threads, int) or isinstance(self.threads, bool) or self.threads <= 0:
            raise ModelingValidationError("launch threads must be a positive integer")
        if not self.cluster or any(
            not isinstance(item, int) or isinstance(item, bool) or item <= 0
            for item in self.cluster
        ):
            raise ModelingValidationError("launch cluster dimensions must be positive integers")
        if not all(isinstance(loop, PeriodicLoopIR) for loop in self.periodic_loops):
            raise ModelingValidationError("LaunchIR.periodic_loops contains an invalid item")

    @property
    def grid(self) -> Tuple[Any, ...]:
        """Backward-compatible spelling for the logical work grid."""

        return self.work_grid


@dataclass(frozen=True)
class KernelIR:
    name: str
    params: Tuple[Tuple[str, Any], ...]
    launches: Tuple[LaunchIR, ...]
    tensor_values: Tuple[TensorValue, ...] = field(default_factory=tuple)
    memory_accesses: Tuple[MemoryAccess, ...] = field(default_factory=tuple)
    fusion_plans: Tuple[FusionPlan, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        validate_identifier(self.name, "kernel name")
        object.__setattr__(self, "params", freeze_attrs(self.params))
        object.__setattr__(self, "launches", tuple(self.launches))
        object.__setattr__(self, "tensor_values", tuple(self.tensor_values))
        object.__setattr__(self, "memory_accesses", tuple(self.memory_accesses))
        object.__setattr__(self, "fusion_plans", tuple(self.fusion_plans))
        if not self.launches:
            raise ModelingValidationError("KernelIR needs at least one launch")
        if not all(isinstance(launch, LaunchIR) for launch in self.launches):
            raise ModelingValidationError("KernelIR.launches contains an invalid item")
        if not all(isinstance(x, TensorValue) for x in self.tensor_values):
            raise ModelingValidationError("KernelIR.tensor_values contains an invalid item")
        if not all(isinstance(x, MemoryAccess) for x in self.memory_accesses):
            raise ModelingValidationError("KernelIR.memory_accesses contains an invalid item")
        if not all(isinstance(x, FusionPlan) for x in self.fusion_plans):
            raise ModelingValidationError("KernelIR.fusion_plans contains an invalid item")
        value_names = [x.name for x in self.tensor_values]
        access_names = [x.name for x in self.memory_accesses]
        plan_names = [x.name for x in self.fusion_plans]
        for label, names in (
            ("tensor value", value_names),
            ("memory access", access_names),
            ("fusion plan", plan_names),
        ):
            if len(names) != len(set(names)):
                raise ModelingValidationError("duplicate %s name in KernelIR" % label)
        if any(x.owner != self.name for x in self.tensor_values):
            raise ModelingValidationError("KernelIR tensor value belongs to another kernel")
        values = frozenset(self.tensor_values)
        if any(x.owner != self.name or x.value not in values for x in self.memory_accesses):
            raise ModelingValidationError("KernelIR memory access references an unknown value")
        events = frozenset(
            Event(phase.name, phase.owner, kind)
            for launch in self.launches
            for loop in launch.periodic_loops
            for phase in loop.phases
            for kind in ("start", "done")
        )
        if any(x.event not in events for x in self.memory_accesses):
            raise ModelingValidationError("KernelIR memory access references an unknown event")
        phase_actors = {
            (phase.owner, phase.name): actor
            for launch in self.launches
            for loop in launch.periodic_loops
            for actor in loop.actors
            for phase in actor.phases
        }
        accesses = frozenset(self.memory_accesses)
        eliminated = []
        loop_launch = {
            loop.owner: launch.name
            for launch in self.launches
            for loop in launch.periodic_loops
        }
        for plan in self.fusion_plans:
            if plan.owner != self.name:
                raise ModelingValidationError("fusion plan belongs to another kernel")
            for handoff in plan.handoffs:
                if handoff.value not in values:
                    raise ModelingValidationError("handoff references an unknown value")
                used = handoff.baseline_writes + handoff.baseline_reads
                if any(x not in accesses for x in used):
                    raise ModelingValidationError(
                        "handoff references an unknown baseline memory access"
                    )
                eliminated.extend(used)
                owners = (handoff.acquire.owner,) + tuple(x.owner for x in handoff.releases)
                launch_names = {loop_launch.get(owner) for owner in owners}
                if None in launch_names or len(launch_names) != 1:
                    raise ModelingValidationError(
                        "handoff acquire/releases must belong to one fused launch"
                    )
                if handoff.via == "register":
                    participant_actors = {
                        phase_actors[(handoff.acquire.owner, handoff.acquire.phase_name)]
                    }
                    participant_actors.update(
                        phase_actors[(event.owner, event.phase_name)]
                        for event in handoff.releases
                    )
                    explicit_domains = {
                        actor.execution_domain for actor in participant_actors
                    }
                    ownership_compatible = (
                        handoff.ownership is not None
                        and handoff.value.ownership is not None
                        and handoff.ownership.compatible_with(
                            handoff.value.ownership
                        )
                    )
                    shared_explicit_domain = (
                        None not in explicit_domains
                        and len(explicit_domains) == 1
                        and handoff.ownership is not None
                        and next(iter(explicit_domains))
                        == handoff.ownership.domain
                    )
                    if len(participant_actors) != 1 and not (
                        ownership_compatible and shared_explicit_domain
                    ):
                        raise ModelingValidationError(
                            "register handoff requires one declared actor ownership "
                            "domain; cross-actor proof needs one shared explicit "
                            "ExecutionDomain and compatible OwnershipMap"
                        )
                    if any(
                        actor.execution_scope != handoff.execution_scope
                        for actor in participant_actors
                    ):
                        raise ModelingValidationError(
                            "register handoff execution_scope must equal the actor's "
                            "explicit execution_scope declaration"
                        )
        handoff_names = [
            handoff.name
            for plan in self.fusion_plans
            for handoff in plan.handoffs
        ]
        if len(handoff_names) != len(set(handoff_names)):
            raise ModelingValidationError(
                "handoff names must be globally unique within one KernelIR"
            )
        eliminated_names = [x.name for x in eliminated]
        if len(eliminated_names) != len(set(eliminated_names)):
            raise ModelingValidationError(
                "a baseline memory access cannot be eliminated by multiple handoffs"
            )

    def periodic_loop(self, name: str) -> PeriodicLoopIR:
        matches = [
            loop
            for launch in self.launches
            for loop in launch.periodic_loops
            if loop.name == name
        ]
        if not matches:
            raise ModelingValidationError("unknown periodic loop %r" % name)
        if len(matches) != 1:
            raise ModelingValidationError(
                "periodic loop name %r is ambiguous across launches" % name
            )
        return matches[0]

    def lower_periodic(self, loop: str, oracle: Optional[Any] = None) -> Any:
        from .lowering import lower_periodic

        return lower_periodic(self, loop, oracle=oracle)

    def summary(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "params": dict(self.params),
            "tensor_values": tuple(
                {
                    "name": value.name,
                    "shape": value.shape,
                    "dtype": value.dtype,
                    "bytes": value.bytes,
                    "layout": value.layout,
                    "ownership": None
                    if value.ownership is None
                    else value.ownership.summary(),
                }
                for value in self.tensor_values
            ),
            "memory_accesses": tuple(
                {
                    "name": access.name,
                    "value": access.value.name,
                    "mode": access.mode,
                    "memory_level": access.memory_level,
                    "bytes": access.bytes,
                    "phase": access.event.phase_name,
                    "event": access.event.kind,
                    "fragment": access.fragment,
                    "layout": access.layout,
                    "accounting": access.accounting,
                }
                for access in self.memory_accesses
            ),
            "fusion_plans": tuple(
                {
                    "name": plan.name,
                    "handoffs": tuple(
                        {
                            "name": handoff.name,
                            "value": handoff.value.name,
                            "via": handoff.via,
                            "bytes": handoff.bytes,
                            "execution_scope": handoff.execution_scope,
                            "iteration_distance": handoff.iteration_distance,
                            "slots": handoff.slots,
                            "alias_group": handoff.alias_group,
                            "ownership": None
                            if handoff.ownership is None
                            else handoff.ownership.summary(),
                            "baseline_storage": handoff.baseline_storage,
                            "producer_source": handoff.producer_source,
                            "consumer_targets": handoff.consumer_targets,
                            "layout_mapping": {
                                "producer": handoff.layout.producer_layout,
                                "via": handoff.layout.via_layout,
                                "consumer": handoff.layout.consumer_layout,
                                "kind": handoff.layout.kind,
                                "bijective": handoff.layout.bijective,
                                "attrs": dict(handoff.layout.attrs),
                            },
                            "acquire": {
                                "phase": handoff.acquire.phase_name,
                                "event": handoff.acquire.kind,
                            },
                            "releases": tuple(
                                {"phase": event.phase_name, "event": event.kind}
                                for event in handoff.releases
                            ),
                            "baseline_writes": tuple(
                                x.name for x in handoff.baseline_writes
                            ),
                            "baseline_reads": tuple(
                                x.name for x in handoff.baseline_reads
                            ),
                        }
                        for handoff in plan.handoffs
                    ),
                }
                for plan in self.fusion_plans
            ),
            "launches": tuple(
                {
                    "name": launch.name,
                    "grid": launch.work_grid,
                    "work_grid": launch.work_grid,
                    "physical_grid": launch.physical_grid,
                    "threads": launch.threads,
                    "cluster": launch.cluster,
                    "residency": launch.residency,
                    "scheduler": launch.scheduler,
                    "swizzle": dict(launch.swizzle),
                    "periodic_loops": tuple(
                        loop.summary() for loop in launch.periodic_loops
                    ),
                }
                for launch in self.launches
            ),
        }
