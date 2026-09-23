"""Ergonomic builders for the frozen modeling IR."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from .errors import ModelingValidationError
from .ir import (
    ActorIR,
    Buffer,
    Dependency,
    ExecutionDomain,
    Event,
    FusionPlan,
    Handoff,
    KernelIR,
    LaunchIR,
    LayoutMapping,
    Lifetime,
    MemoryAccess,
    Occurrence,
    OwnershipMap,
    PeriodicLoopIR,
    Phase,
    PipelineBuffer,
    ResourceSequenceIR,
    State,
    StateCarry,
    TensorValue,
    Timing,
    Work,
    canonical_memory_level,
    freeze_attrs,
    freeze_value,
    validate_identifier,
)


@dataclass(frozen=True)
class Materialization:
    """Convenience grouping for generic equal-byte baseline traffic.

    It is builder-side sugar rather than a separate frozen KernelIR node;
    ``writes`` and ``reads`` are ordinary exact-location ``MemoryAccess``
    nodes retained by the built kernel.
    """

    name: str
    value: TensorValue
    writes: Tuple[MemoryAccess, ...]
    reads: Tuple[MemoryAccess, ...]
    storage: str = "ddr"
    producer_source: str = "register"
    consumer_targets: Tuple[str, ...] = ("register",)
    transfer_bytes: Optional[float] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "writes", tuple(self.writes))
        object.__setattr__(self, "reads", tuple(self.reads))
        object.__setattr__(self, "storage", canonical_memory_level(self.storage))
        if self.storage == "l2":
            raise ModelingValidationError(
                "L2 is passive cache traffic, not logical materialization storage"
            )
        object.__setattr__(
            self, "producer_source", canonical_memory_level(self.producer_source)
        )
        if self.producer_source == "l2":
            raise ModelingValidationError(
                "L2 is passive cache traffic, not a programmed producer source"
            )
        object.__setattr__(
            self,
            "consumer_targets",
            tuple(canonical_memory_level(x) for x in self.consumer_targets),
        )
        if "l2" in self.consumer_targets:
            raise ModelingValidationError(
                "L2 is passive cache traffic, not a programmed consumer target"
            )
        if self.transfer_bytes is not None:
            try:
                size = float(self.transfer_bytes)
            except (TypeError, ValueError) as error:
                raise ModelingValidationError("transfer_bytes must be numeric") from error
            if not math.isfinite(size) or size <= 0.0 or size > self.value.bytes:
                raise ModelingValidationError(
                    "transfer_bytes must be positive and no larger than value bytes"
                )
            object.__setattr__(self, "transfer_bytes", size)

    @property
    def bytes(self) -> float:
        return self.value.bytes if self.transfer_bytes is None else self.transfer_bytes


def legacy_level_path(descriptor: Sequence[Any], *, role: str) -> Tuple[str, ...]:
    """Translate only the documented legacy ``mem_levels`` descriptors.

    Legacy source code defines the first three entries as nested DDR, SMEM,
    and register presence flags and the final entry as element bytes.  L2 is
    not an array position; it is a passive cache modeled separately.  The
    unusual descriptors used by a few specialized models do not share this
    documented meaning and are rejected instead of guessed.
    """

    values = tuple(descriptor)
    if len(values) != 4:
        raise ModelingValidationError(
            "legacy mem_levels descriptor must be [ddr, smem, register, element_bytes]"
        )
    flags = values[:3]
    mapping = {
        (1, 1, 1): ("ddr", "smem", "register"),
        (0, 1, 1): ("smem", "register"),
        (0, 0, 1): ("register",),
    }
    if flags not in mapping:
        raise ModelingValidationError(
            "undocumented legacy mem_levels flags %r; provide a named memory path"
            % (flags,)
        )
    try:
        element_bytes = float(values[3])
    except (TypeError, ValueError) as error:
        raise ModelingValidationError("legacy element bytes must be numeric") from error
    if not math.isfinite(element_bytes) or element_bytes <= 0.0:
        raise ModelingValidationError("legacy element bytes must be finite and positive")
    if role not in ("read", "write"):
        raise ModelingValidationError("legacy path role must be 'read' or 'write'")
    path = mapping[flags]
    return path if role == "read" else tuple(reversed(path))


def _positive_integer(value: int, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ModelingValidationError("%s must be a positive integer" % label)
    return value


def _distance(value: int, label: str = "distance") -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ModelingValidationError("%s must be an integer" % label)
    return value


def _positive_distance(value: int, label: str = "distance") -> int:
    result = _distance(value, label)
    if result <= 0:
        raise ModelingValidationError("%s must be positive" % label)
    return result


def _lag(value: float, label: str = "lag") -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ModelingValidationError("%s must be numeric" % label) from error
    if not math.isfinite(result) or result < 0.0:
        raise ModelingValidationError("%s must be finite and non-negative" % label)
    return result


class Kernel:
    """Mutable construction façade that snapshots to :class:`KernelIR`."""

    def __init__(self, name: str, params: Optional[Mapping[str, Any]] = None) -> None:
        self.name = validate_identifier(name, "kernel name")
        self._params = freeze_attrs(params or {})
        self._launches = []  # type: List[Launch]
        self._tensor_values = []  # type: List[TensorValue]
        self._memory_accesses = []  # type: List[MemoryAccess]
        self._fusion_plans = []  # type: List[Fusion]

    def tensor_value(
        self,
        name: str,
        *,
        shape: Sequence[Any],
        dtype: str,
        bytes: float,
        layout: str,
        ownership: Optional[OwnershipMap] = None,
    ) -> TensorValue:
        """Declare a logical payload that may cross fragment boundaries."""

        if any(value.name == name for value in self._tensor_values):
            raise ModelingValidationError("duplicate tensor value name %r" % name)
        result = TensorValue(
            name=name,
            owner=self.name,
            shape=tuple(shape),
            dtype=dtype,
            bytes=bytes,
            layout=layout,
            ownership=ownership,
        )
        self._tensor_values.append(result)
        return result

    def _check_tensor_value(self, value: TensorValue) -> None:
        if (
            not isinstance(value, TensorValue)
            or value.owner != self.name
            or value not in self._tensor_values
        ):
            raise ModelingValidationError("tensor value belongs to another kernel")

    @staticmethod
    def _as_event(value: Union[Event, Phase], default_kind: str) -> Event:
        if isinstance(value, Phase):
            return value.done if default_kind == "done" else value.start
        if not isinstance(value, Event):
            raise ModelingValidationError("expected a Phase event")
        return value

    def memory_access(
        self,
        name: str,
        value: TensorValue,
        *,
        mode: str,
        memory_level: str,
        bytes: float,
        event: Union[Event, Phase],
        fragment: str,
        layout: Optional[str] = None,
        accounting: str = "exact",
    ) -> MemoryAccess:
        """Declare one exact per-level traffic item.

        Direct calls default to ``accounting='exact'``.  The
        :meth:`materialize` helper marks its replicated per-level entries as
        ``generic_equal_bytes`` so the simplifying assumption is visible in
        IR and reports.
        """

        self._check_tensor_value(value)
        if any(access.name == name for access in self._memory_accesses):
            raise ModelingValidationError("duplicate memory access name %r" % name)
        canonical_mode = validate_identifier(mode, "memory access mode").lower()
        result = MemoryAccess(
            name=name,
            owner=self.name,
            value=value,
            mode=canonical_mode,
            memory_level=memory_level,
            bytes=bytes,
            event=self._as_event(
                event, "done" if canonical_mode == "write" else "start"
            ),
            fragment=fragment,
            layout=value.layout if layout is None else layout,
            accounting=accounting,
        )
        self._memory_accesses.append(result)
        return result

    def materialize(
        self,
        name: str,
        value: TensorValue,
        *,
        producer: Union[Event, Phase],
        consumer: Union[Event, Phase, Sequence[Union[Event, Phase]]],
        producer_fragment: str,
        consumer_fragment: Union[str, Sequence[str]],
        storage: str = "ddr",
        producer_source: str = "register",
        consumer_target: Union[str, Sequence[str]] = "register",
        levels: Optional[Sequence[str]] = None,
        bytes: Optional[float] = None,
        producer_layout: Optional[str] = None,
        consumer_layout: Optional[str] = None,
    ) -> Materialization:
        """Describe a logical producer-store/consumer-load materialization.

        The primary frontend names the programmed storage endpoint and the
        directed source/target memories.  L2 is passive, so ``storage='ddr'``
        lowers to generic equal-byte DDR and L2 traffic without asking users
        to encode a cache as program-visible storage.  ``levels`` is an
        advanced compatibility override for already-derived traffic.  Use
        :meth:`memory_access` directly when calibrated traffic differs by
        level or consumer.
        """

        validate_identifier(name, "materialization name")
        self._check_tensor_value(value)
        if isinstance(consumer, (Event, Phase)):
            consumers = (consumer,)
        else:
            consumers = tuple(consumer)
        if not consumers:
            raise ModelingValidationError("materialization needs at least one consumer")
        if isinstance(consumer_fragment, str):
            consumer_fragments = (consumer_fragment,) * len(consumers)
        else:
            consumer_fragments = tuple(consumer_fragment)
        if len(consumer_fragments) != len(consumers):
            raise ModelingValidationError(
                "consumer_fragment must provide one fragment per consumer"
            )
        storage = canonical_memory_level(storage)
        producer_source = canonical_memory_level(producer_source)
        if storage == "l2" or producer_source == "l2":
            raise ModelingValidationError(
                "L2 is passive cache traffic; use a programmed source/destination storage"
            )
        if isinstance(consumer_target, str):
            consumer_targets = (consumer_target,) * len(consumers)
        else:
            consumer_targets = tuple(consumer_target)
        if len(consumer_targets) != len(consumers):
            raise ModelingValidationError(
                "consumer_target must provide one memory level per consumer"
            )
        consumer_targets = tuple(canonical_memory_level(x) for x in consumer_targets)
        if "l2" in consumer_targets:
            raise ModelingValidationError(
                "L2 is passive cache traffic, not a programmed consumer target"
            )
        if levels is None:
            traffic_levels = ("ddr", "l2") if storage == "ddr" else (storage,)
        else:
            traffic_levels = tuple(canonical_memory_level(x) for x in levels)
            if not traffic_levels:
                raise ModelingValidationError("materialization levels must not be empty")
        if len(traffic_levels) != len(set(traffic_levels)):
            raise ModelingValidationError("materialization traffic levels must be unique")
        transfer_bytes = value.bytes if bytes is None else bytes
        write_event = self._as_event(producer, "done")
        read_events = tuple(self._as_event(item, "start") for item in consumers)
        producer_layout = value.layout if producer_layout is None else producer_layout
        consumer_layout = value.layout if consumer_layout is None else consumer_layout
        writes = []
        reads = []
        for level in traffic_levels:
            writes.append(
                self.memory_access(
                    "%s_%s_write" % (name, str(level).lower()),
                    value,
                    mode="write",
                    memory_level=level,
                    bytes=transfer_bytes,
                    event=write_event,
                    fragment=producer_fragment,
                    layout=producer_layout,
                    accounting="generic_equal_bytes",
                )
            )
            for index, (read_event, fragment) in enumerate(
                zip(read_events, consumer_fragments)
            ):
                reads.append(
                    self.memory_access(
                        "%s_%s_read%d" % (name, str(level).lower(), index),
                        value,
                        mode="read",
                        memory_level=level,
                        bytes=transfer_bytes,
                        event=read_event,
                        fragment=fragment,
                        layout=consumer_layout,
                        accounting="generic_equal_bytes",
                    )
                )
        return Materialization(
            name,
            value,
            tuple(writes),
            tuple(reads),
            storage,
            producer_source,
            consumer_targets,
            transfer_bytes,
        )

    def exact_materialization(
        self,
        name: str,
        value: TensorValue,
        *,
        writes: Sequence[MemoryAccess],
        reads: Sequence[MemoryAccess],
        storage: str,
        producer_source: str,
        consumer_target: Union[str, Sequence[str]],
        transfer_bytes: Optional[float] = None,
    ) -> Materialization:
        """Group calibrated direct accesses into one baseline transfer."""

        validate_identifier(name, "materialization name")
        self._check_tensor_value(value)
        writes = tuple(writes)
        reads = tuple(reads)
        known = frozenset(self._memory_accesses)
        if not writes or not reads or any(x not in known for x in writes + reads):
            raise ModelingValidationError(
                "exact materialization accesses must be nonempty and belong to this kernel"
            )
        if any(x.value != value or x.mode != "write" for x in writes):
            raise ModelingValidationError(
                "exact materialization writes must write the selected value"
            )
        if any(x.value != value or x.mode != "read" for x in reads):
            raise ModelingValidationError(
                "exact materialization reads must read the selected value"
            )
        consumers = []
        for access in reads:
            key = (access.fragment, access.event.owner, access.event.phase_name)
            if key not in consumers:
                consumers.append(key)
        if isinstance(consumer_target, str):
            consumer_targets = (consumer_target,) * len(consumers)
        else:
            consumer_targets = tuple(consumer_target)
        if len(consumer_targets) != len(consumers):
            raise ModelingValidationError(
                "consumer_target must provide one memory level per consumer access"
            )
        return Materialization(
            name=name,
            value=value,
            writes=writes,
            reads=reads,
            storage=storage,
            producer_source=producer_source,
            consumer_targets=consumer_targets,
            transfer_bytes=transfer_bytes,
        )

    def fusion(self, name: str) -> "Fusion":
        """Create one explicit fragment-fusion transformation plan."""

        if any(plan.name == name for plan in self._fusion_plans):
            raise ModelingValidationError("duplicate fusion plan name %r" % name)
        result = Fusion(self, name)
        self._fusion_plans.append(result)
        return result

    def launch(
        self,
        *,
        grid: Optional[Sequence[Any]] = None,
        work_grid: Optional[Sequence[Any]] = None,
        physical_grid: Optional[Sequence[Any]] = None,
        threads: int,
        cluster: Sequence[int] = (1, 1, 1),
        resident_ctas: Optional[int] = None,
        residency: Any = "auto",
        scheduler: str = "static",
        swizzle: Optional[Mapping[str, Any]] = None,
        name: str = "main",
    ) -> "Launch":
        validate_identifier(name, "launch name")
        if any(launch.name == name for launch in self._launches):
            raise ModelingValidationError("duplicate launch name %r" % name)
        if grid is not None and work_grid is not None:
            raise ModelingValidationError("use either work_grid or its grid alias, not both")
        if work_grid is None:
            work_grid = grid
        if work_grid is None:
            raise ModelingValidationError("launch work_grid must be provided")
        result = Launch(
            self,
            name=name,
            grid=work_grid,
            physical_grid=physical_grid,
            threads=threads,
            cluster=cluster,
            resident_ctas=resident_ctas,
            residency=residency,
            scheduler=scheduler,
            swizzle=swizzle,
        )
        self._launches.append(result)
        return result

    def build(self) -> KernelIR:
        if not self._launches:
            raise ModelingValidationError("kernel needs at least one launch")
        result = KernelIR(
            name=self.name,
            params=self._params,
            launches=tuple(launch._build() for launch in self._launches),
            tensor_values=tuple(self._tensor_values),
            memory_accesses=tuple(self._memory_accesses),
            fusion_plans=tuple(plan._build() for plan in self._fusion_plans),
        )
        try:
            hash(result)
        except TypeError as error:
            raise ModelingValidationError("built KernelIR is not hashable") from error
        return result

    def lower_periodic(self, loop: str, oracle: Optional[Any] = None) -> Any:
        return self.build().lower_periodic(loop, oracle=oracle)

    def summary(self) -> Dict[str, Any]:
        return self.build().summary()


class Fusion:
    """Mutable builder for a frozen :class:`FusionPlan`."""

    def __init__(self, kernel: Kernel, name: str) -> None:
        self.kernel = kernel
        self.name = validate_identifier(name, "fusion plan name")
        self._handoffs = []  # type: List[Handoff]

    def __enter__(self) -> "Fusion":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        return False

    def handoff(
        self,
        name: str,
        baseline: Materialization,
        *,
        via: str,
        execution_scope: str,
        layout: Optional[LayoutMapping] = None,
        iteration_distance: int = 0,
        release: Optional[
            Union[Event, Phase, Sequence[Union[Event, Phase]]]
        ] = None,
        slots: int = 1,
        alias_group: Optional[str] = None,
        ownership: Optional[OwnershipMap] = None,
    ) -> Handoff:
        """Replace ``baseline`` traffic with an on-chip multi-consumer handoff."""

        if any(item.name == name for item in self._handoffs):
            raise ModelingValidationError("duplicate handoff name %r" % name)
        if not isinstance(baseline, Materialization):
            raise ModelingValidationError("handoff baseline must be Materialization")
        self.kernel._check_tensor_value(baseline.value)
        known = frozenset(self.kernel._memory_accesses)
        if any(x not in known for x in baseline.writes + baseline.reads):
            raise ModelingValidationError("handoff baseline belongs to another kernel")
        if layout is None:
            producer_layouts = set(x.layout for x in baseline.writes)
            consumer_layouts = set(x.layout for x in baseline.reads)
            if len(producer_layouts) != 1 or len(consumer_layouts) != 1:
                raise ModelingValidationError(
                    "non-uniform layouts require an explicit LayoutMapping"
                )
            producer_layout = next(iter(producer_layouts))
            consumer_layout = next(iter(consumer_layouts))
            if producer_layout != consumer_layout:
                raise ModelingValidationError(
                    "different producer/consumer layouts require an explicit LayoutMapping"
                )
            layout = LayoutMapping.identity(producer_layout)
        if release is None:
            release_events = []
            seen = set()
            for access in baseline.reads:
                key = (
                    access.fragment,
                    access.event.owner,
                    access.event.phase_name,
                )
                if key not in seen:
                    seen.add(key)
                    release_events.append(
                        Event(access.event.phase_name, access.event.owner, "done")
                    )
        else:
            if isinstance(release, (Event, Phase)):
                release = (release,)
            release_events = [
                self.kernel._as_event(item, "done") for item in tuple(release)
            ]
        effective_ownership = ownership
        if (
            effective_ownership is None
            and baseline.value.ownership is not None
            and baseline.value.ownership.layout == layout.via_layout
        ):
            effective_ownership = baseline.value.ownership
        result = Handoff(
            name=name,
            owner=self.kernel.name,
            value=baseline.value,
            baseline_writes=baseline.writes,
            baseline_reads=baseline.reads,
            via=via,
            bytes=baseline.bytes,
            execution_scope=execution_scope,
            layout=layout,
            acquire=baseline.writes[0].event,
            releases=tuple(release_events),
            baseline_storage=baseline.storage,
            producer_source=baseline.producer_source,
            consumer_targets=baseline.consumer_targets,
            iteration_distance=_distance(iteration_distance, "iteration_distance"),
            slots=_positive_integer(slots, "handoff slots"),
            alias_group=alias_group,
            ownership=effective_ownership,
        )
        self._handoffs.append(result)
        return result

    def _build(self) -> FusionPlan:
        return FusionPlan(self.name, self.kernel.name, tuple(self._handoffs))


class Launch:
    """One kernel launch and its spatial CTA-concurrency description."""

    def __init__(
        self,
        kernel: Kernel,
        *,
        name: str,
        grid: Sequence[Any],
        physical_grid: Optional[Sequence[Any]],
        threads: int,
        cluster: Sequence[int],
        resident_ctas: Optional[int],
        residency: Any,
        scheduler: str,
        swizzle: Optional[Mapping[str, Any]],
    ) -> None:
        self.kernel = kernel
        self.name = validate_identifier(name, "launch name")
        self.work_grid = tuple(freeze_value(item) for item in grid)
        if not self.work_grid:
            raise ModelingValidationError("launch grid must not be empty")
        if physical_grid is None:
            self.physical_grid = self.work_grid
        else:
            self.physical_grid = tuple(freeze_value(item) for item in physical_grid)
            if not self.physical_grid:
                raise ModelingValidationError("physical_grid must not be empty")
        self.threads = _positive_integer(threads, "threads")
        self.cluster = tuple(_positive_integer(item, "cluster dimension") for item in cluster)
        if not self.cluster:
            raise ModelingValidationError("cluster must not be empty")
        if resident_ctas is not None:
            resident_ctas = _positive_integer(resident_ctas, "resident_ctas")
            if residency != "auto":
                raise ModelingValidationError(
                    "use either resident_ctas or residency, not both"
                )
            residency = resident_ctas
        residency = freeze_value(residency)
        if not (
            residency == "auto"
            or (
                isinstance(residency, int)
                and not isinstance(residency, bool)
                and residency > 0
            )
        ):
            raise ModelingValidationError("residency must be 'auto' or a positive CTA count")
        validate_identifier(scheduler, "scheduler")
        self.residency = residency
        self.scheduler = scheduler
        self.swizzle = freeze_attrs(swizzle or {})
        self._periodic_loops = []  # type: List[PeriodicLoop]

    def __enter__(self) -> "Launch":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        return False

    def periodic(
        self,
        name: str,
        *,
        iterations: Optional[Any] = None,
        stages: Optional[int] = None,
    ) -> "PeriodicLoop":
        if any(loop.name == name for loop in self._periodic_loops):
            raise ModelingValidationError("duplicate periodic loop name %r" % name)
        result = PeriodicLoop(
            self,
            name=name,
            iterations=iterations,
            stages=stages,
        )
        self._periodic_loops.append(result)
        return result

    pipeline = periodic

    def _build(self) -> LaunchIR:
        if not self._periodic_loops:
            raise ModelingValidationError("launch %s needs a periodic loop" % self.name)
        return LaunchIR(
            name=self.name,
            work_grid=self.work_grid,
            physical_grid=self.physical_grid,
            threads=self.threads,
            cluster=self.cluster,
            residency=self.residency,
            scheduler=self.scheduler,
            swizzle=self.swizzle,
            periodic_loops=tuple(loop._build() for loop in self._periodic_loops),
        )


class PeriodicLoop:
    """A temporally repeated loop body inside one CTA/cluster program."""

    def __init__(
        self,
        launch: Launch,
        *,
        name: str,
        iterations: Optional[Any],
        stages: Optional[int],
    ) -> None:
        validate_identifier(name, "periodic loop name")
        if stages is not None:
            stages = _positive_integer(stages, "stages")
        self.launch = launch
        self.name = name
        self.iterations = freeze_value(iterations)
        self.stages = stages
        self.owner = "%s/%s/%s" % (launch.kernel.name, launch.name, name)
        self._buffers = []  # type: List[Buffer]
        self._actors = []  # type: List[Actor]
        self._phase_names = {}  # type: Dict[str, Phase]
        self._dependencies = []  # type: List[Dependency]
        self._states = []  # type: List[State]
        self._carries = []  # type: List[StateCarry]
        self._lifetimes = []  # type: List[Lifetime]
        self._pipeline_buffers = []  # type: List[PipelineBuffer]
        self._resource_sequences = []  # type: List[ResourceSequenceIR]

    def __enter__(self) -> "PeriodicLoop":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        return False

    def buffer(
        self,
        name: str,
        *,
        scope: str,
        shape: Sequence[Any] = (),
        dtype: str = "",
        slots: int = 1,
        execution_scope: str = "cta",
        alias_group: Optional[str] = None,
        ownership: Optional[OwnershipMap] = None,
    ) -> Buffer:
        if any(buffer.name == name for buffer in self._buffers):
            raise ModelingValidationError("duplicate buffer name %r" % name)
        result = Buffer(
            name=name,
            scope=scope,
            owner=self.owner,
            shape=tuple(shape),
            dtype=dtype,
            slots=slots,
            execution_scope=execution_scope,
            alias_group=alias_group,
            ownership=ownership,
        )
        self._buffers.append(result)
        return result

    def actor(
        self,
        name: str,
        *,
        serial_resource: Optional[str] = None,
        execution_scope: str = "unspecified",
        execution_domain: Optional[ExecutionDomain] = None,
    ) -> "Actor":
        validate_identifier(name, "actor name")
        if any(actor.name == name for actor in self._actors):
            raise ModelingValidationError("duplicate actor name %r" % name)
        if serial_resource is not None:
            validate_identifier(serial_resource, "serial_resource")
        if execution_domain is not None:
            if not isinstance(execution_domain, ExecutionDomain):
                raise ModelingValidationError(
                    "actor execution_domain must be ExecutionDomain or None"
                )
            if execution_scope == "unspecified":
                execution_scope = execution_domain.scope
            elif execution_scope != execution_domain.scope:
                raise ModelingValidationError(
                    "actor execution_domain scope must equal execution_scope"
                )
        result = Actor(
            self,
            name=name,
            serial_resource=serial_resource,
            execution_scope=execution_scope,
            execution_domain=execution_domain,
        )
        self._actors.append(result)
        return result

    def resource_sequence(
        self, resource: str, *occurrences: Occurrence
    ) -> ResourceSequenceIR:
        """Declare expert global arbitration order, independent of actor order."""

        return self._register_resource_sequence(
            resource,
            occurrences,
            source="expert",
        )

    def _event(self, value: Union[Event, Phase], default_kind: str) -> Event:
        if isinstance(value, Phase):
            value = value.done if default_kind == "done" else value.start
        if not isinstance(value, Event):
            raise ModelingValidationError("expected a Phase event")
        if value.owner != self.owner:
            raise ModelingValidationError("event belongs to another periodic loop")
        if value.phase_name not in self._phase_names:
            raise ModelingValidationError("event references unknown phase %r" % value.phase_name)
        return value

    def after(
        self,
        source: Union[Event, Phase],
        target: Union[Event, Phase],
        *,
        distance: int = 0,
        lag: float = 0.0,
        name: str = "",
    ) -> Dependency:
        """Add event dependency with ``lag`` measured after ``source`` event."""

        source_event = self._event(source, "done")
        target_event = self._event(target, "start")
        if target_event.kind != "start":
            raise ModelingValidationError(
                "v0 dependencies must target a phase start event"
            )
        result = Dependency(
            source=source_event,
            target=target_event,
            iteration_distance=_distance(distance),
            lag=_lag(lag),
            name=name,
        )
        self._dependencies.append(result)
        return result

    def state(self, name: str, *, storage: Optional[Buffer] = None) -> State:
        if not name:
            raise ModelingValidationError("state name must not be empty")
        if any(state.name == name for state in self._states):
            raise ModelingValidationError("duplicate state name %r" % name)
        if storage is not None:
            self._check_buffer(storage)
        result = State(name=name, owner=self.owner, storage=storage)
        self._states.append(result)
        return result

    def carry(
        self,
        state: State,
        *,
        source: Union[Event, Phase],
        target: Union[Event, Phase],
        distance: int,
        lag: float = 0.0,
    ) -> StateCarry:
        """Carry state forward by a positive iteration distance."""

        if (
            not isinstance(state, State)
            or state.owner != self.owner
            or state not in self._states
        ):
            raise ModelingValidationError("carry state belongs to another periodic loop")
        source_event = self._event(source, "done")
        target_event = self._event(target, "start")
        if target_event.kind != "start":
            raise ModelingValidationError("v0 carries must target a phase start event")
        result = StateCarry(
            state=state,
            source=source_event,
            target=target_event,
            iteration_distance=_positive_distance(distance, "carry distance"),
            lag=_lag(lag, "carry lag"),
        )
        self._carries.append(result)
        return result

    def lifetime(
        self,
        buffer: Buffer,
        *,
        acquire: Union[Event, Phase],
        release: Union[Event, Phase],
        minimum_residence: float = 0.0,
    ) -> Lifetime:
        self._check_buffer(buffer)
        result = Lifetime(
            buffer=buffer,
            acquire=self._event(acquire, "start"),
            release=self._event(release, "done"),
            minimum_residence=_lag(minimum_residence, "minimum_residence"),
        )
        self._lifetimes.append(result)
        return result

    def pipeline_buffer(
        self,
        buffer: Buffer,
        *,
        acquire: Union[Event, Phase],
        release: Union[Event, Phase],
        capacity: Optional[int] = None,
        minimum_residence: float = 0.0,
    ) -> PipelineBuffer:
        self._check_buffer(buffer)
        if capacity is None:
            capacity = buffer.slots
        capacity = _positive_integer(capacity, "pipeline-buffer capacity")
        if capacity > buffer.slots:
            raise ModelingValidationError(
                "pipeline-buffer capacity must not exceed buffer.slots"
            )
        acquire_event = self._event(acquire, "start")
        release_event = self._event(release, "done")
        residence = _lag(minimum_residence, "minimum_residence")
        result = PipelineBuffer(
            buffer=buffer,
            acquire=acquire_event,
            release=release_event,
            capacity=capacity,
            minimum_residence=residence,
        )
        self._pipeline_buffers.append(result)
        self._lifetimes.append(
            Lifetime(buffer, acquire_event, release_event, residence)
        )
        return result

    def _check_buffer(self, buffer: Buffer) -> None:
        if not isinstance(buffer, Buffer) or buffer.owner != self.owner:
            raise ModelingValidationError("buffer belongs to another periodic loop")
        if buffer not in self._buffers:
            raise ModelingValidationError("unknown buffer %r" % buffer.name)

    def _register_phase(self, phase: Phase) -> None:
        if phase.name in self._phase_names:
            raise ModelingValidationError("duplicate phase name %r" % phase.name)
        self._phase_names[phase.name] = phase

    def _register_resource_sequence(
        self,
        resource: str,
        occurrences: Sequence[Occurrence],
        *,
        source: str,
    ) -> ResourceSequenceIR:
        validate_identifier(resource, "resource sequence resource")
        if any(item.resource == resource for item in self._resource_sequences):
            raise ModelingValidationError(
                "resource %s already has a cyclic resource sequence" % resource
            )
        occurrences = tuple(occurrences)
        if not occurrences:
            raise ModelingValidationError("resource sequence must not be empty")
        for occurrence in occurrences:
            if not isinstance(occurrence, Occurrence):
                raise ModelingValidationError(
                    "resource_sequence expects Phase.at(window_offset) occurrences"
                )
            phase = self._phase_names.get(occurrence.phase.name)
            if phase is None or phase != occurrence.phase:
                raise ModelingValidationError(
                    "resource-sequence phase belongs to another periodic loop"
                )
        names = [occurrence.phase.name for occurrence in occurrences]
        if len(names) != len(set(names)):
            raise ModelingValidationError(
                "resource sequence must list each phase at most once"
            )
        result = ResourceSequenceIR(
            resource=resource,
            sequence=occurrences,
            source=source,
        )
        self._resource_sequences.append(result)
        return result

    def _build(self) -> PeriodicLoopIR:
        if not self._actors:
            raise ModelingValidationError("periodic loop %s needs an actor" % self.name)
        actors = tuple(actor._build() for actor in self._actors)
        if not any(actor.phases for actor in actors):
            raise ModelingValidationError("periodic loop %s needs a phase" % self.name)
        return PeriodicLoopIR(
            name=self.name,
            owner=self.owner,
            iterations=self.iterations,
            stages=self.stages,
            buffers=tuple(self._buffers),
            actors=actors,
            dependencies=tuple(self._dependencies),
            states=tuple(self._states),
            carries=tuple(self._carries),
            lifetimes=tuple(self._lifetimes),
            pipeline_buffers=tuple(self._pipeline_buffers),
            resource_sequences=tuple(self._resource_sequences),
        )


class Actor:
    """A logical issue role within one periodic loop."""

    def __init__(
        self,
        loop: PeriodicLoop,
        *,
        name: str,
        serial_resource: Optional[str],
        execution_scope: str,
        execution_domain: Optional[ExecutionDomain],
    ) -> None:
        self.loop = loop
        self.name = name
        self.serial_resource = serial_resource
        self.execution_scope = execution_scope
        self.execution_domain = execution_domain
        self._phases = []  # type: List[Phase]
        self._sequence = tuple()  # type: Tuple[Occurrence, ...]
        self._order = "issue"

    def __enter__(self) -> "Actor":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        return False

    def phase(
        self,
        name: str,
        *,
        work: Work,
        timing: Optional[Timing] = None,
        reads: Sequence[Buffer] = (),
        writes: Sequence[Buffer] = (),
    ) -> Phase:
        reads = tuple(reads)
        writes = tuple(writes)
        for buffer in reads + writes:
            self.loop._check_buffer(buffer)
        result = Phase(
            name=name,
            actor=self.name,
            owner=self.loop.owner,
            work=work,
            timing=timing,
            reads=reads,
            writes=writes,
        )
        self.loop._register_phase(result)
        self._phases.append(result)
        return result

    @staticmethod
    def _programmed_storage(value: str, label: str) -> str:
        level = canonical_memory_level(value)
        if level == "l2":
            raise ModelingValidationError(
                "%s cannot be L2 because caches are not programmed endpoints" % label
            )
        return level

    def load(
        self,
        name: str,
        *,
        bytes: float,
        source: str,
        destination: str,
        engine: str,
        timing: Optional[Timing] = None,
        reads: Sequence[Buffer] = (),
        writes: Sequence[Buffer] = (),
        attrs: Optional[Mapping[str, Any]] = None,
    ) -> Phase:
        """Create a load work primitive with explicit storage endpoints."""

        if engine not in ("tma", "cp_async", "ldst"):
            raise ModelingValidationError(
                "load engine must be 'tma', 'cp_async', or 'ldst'"
            )
        metadata = dict(attrs or {})
        metadata.update(
            {
                "source": self._programmed_storage(source, "load source"),
                "destination": self._programmed_storage(
                    destination, "load destination"
                ),
                "engine": engine,
            }
        )
        return self.phase(
            name,
            work=Work("load", bytes=bytes, attrs=metadata),
            timing=timing,
            reads=reads,
            writes=writes,
        )

    def store(
        self,
        name: str,
        *,
        bytes: float,
        source: str,
        destination: str,
        engine: str,
        timing: Optional[Timing] = None,
        reads: Sequence[Buffer] = (),
        writes: Sequence[Buffer] = (),
        attrs: Optional[Mapping[str, Any]] = None,
    ) -> Phase:
        """Create a store work primitive with explicit storage endpoints."""

        if engine not in ("tma", "cp_async", "ldst"):
            raise ModelingValidationError(
                "store engine must be 'tma', 'cp_async', or 'ldst'"
            )
        metadata = dict(attrs or {})
        metadata.update(
            {
                "source": self._programmed_storage(source, "store source"),
                "destination": self._programmed_storage(
                    destination, "store destination"
                ),
                "engine": engine,
            }
        )
        return self.phase(
            name,
            work=Work("store", bytes=bytes, attrs=metadata),
            timing=timing,
            reads=reads,
            writes=writes,
        )

    def compute(
        self,
        name: str,
        *,
        flops: float,
        engine: str,
        op: str = "compute",
        timing: Optional[Timing] = None,
        reads: Sequence[Buffer] = (),
        writes: Sequence[Buffer] = (),
        attrs: Optional[Mapping[str, Any]] = None,
    ) -> Phase:
        """Create tensor, CUDA/vector, or SFU compute work."""

        if engine not in ("tensor", "cuda", "vector", "sfu"):
            raise ModelingValidationError(
                "compute engine must be 'tensor', 'cuda', 'vector', or 'sfu'"
            )
        validate_identifier(op, "compute operation")
        metadata = dict(attrs or {})
        requested_engine = engine
        engine = "cuda" if engine == "vector" else engine
        metadata.update({"engine": engine, "op": op})
        if requested_engine != engine:
            metadata["requested_engine"] = requested_engine
        return self.phase(
            name,
            work=Work("compute", flops=flops, attrs=metadata),
            timing=timing,
            reads=reads,
            writes=writes,
        )

    def sequence(
        self,
        *occurrences: Occurrence,
        order: str = "issue",
        resource: Optional[str] = None,
    ) -> None:
        """Declare cyclic actor order, optionally also fixing resource order.

        ``Phase.at(window_offset)`` labels source positions in a local,
        unrolled scheduling window.  It does not set dataflow distance.
        ``order='issue'`` emits start-to-start zero-lag edges, while
        ``order='completion'`` emits done-to-start edges.  ``resource=...``
        independently registers a fixed hardware arbitration order.  The
        final-to-first wrap is implicit.
        """

        if self._sequence:
            raise ModelingValidationError("actor %s already has a sequence" % self.name)
        if not occurrences:
            raise ModelingValidationError("actor sequence must not be empty")
        if order not in ("issue", "completion"):
            raise ModelingValidationError("sequence order must be 'issue' or 'completion'")
        if resource is not None:
            validate_identifier(resource, "sequence resource")
        if self.serial_resource is not None:
            if resource is not None and resource != self.serial_resource:
                raise ModelingValidationError(
                    "sequence resource conflicts with serial_resource compatibility alias"
                )
            if resource is None:
                resource = self.serial_resource
        for occurrence in occurrences:
            if not isinstance(occurrence, Occurrence):
                raise ModelingValidationError(
                    "actor.sequence expects Phase.at(window_offset) occurrences"
                )
            if occurrence.phase not in self._phases:
                raise ModelingValidationError(
                    "sequence phase %s is not owned by actor %s"
                    % (occurrence.phase.name, self.name)
                )
        names = [occurrence.phase.name for occurrence in occurrences]
        if len(names) != len(set(names)):
            raise ModelingValidationError(
                "v0 actor sequences cannot repeat a phase; create distinct phases"
            )
        if resource is not None:
            self.loop._register_resource_sequence(
                resource,
                occurrences,
                source="actor_%s" % self.name,
            )
        self._sequence = tuple(occurrences)
        self._order = order

    def _build(self) -> ActorIR:
        if self.serial_resource is not None and not self._sequence:
            raise ModelingValidationError(
                "serial_resource compatibility alias requires actor.sequence"
            )
        return ActorIR(
            name=self.name,
            serial_resource=self.serial_resource,
            phases=tuple(self._phases),
            sequence=self._sequence,
            order=self._order,
            execution_scope=self.execution_scope,
            execution_domain=self.execution_domain,
        )
