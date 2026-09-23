"""Lower the frozen frontend IR to TileSight's existing ``PeriodicDAG``."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from tilesight.modeling._pipeline.periodic_schedule import (
    Dependency as BackendDependency,
    FixedResourceOrder,
    PeriodicDAG,
    Phase as BackendPhase,
    ResourceUse,
    TokenBuffer,
)

from .errors import ModelingValidationError, TimingResolutionError
from .ir import ActorIR, Event, Handoff, KernelIR, Occurrence, PeriodicLoopIR, Phase, Timing


def _resolve_timings(
    loop: PeriodicLoopIR, oracle: Optional[Any]
) -> Dict[str, Timing]:
    result = {}
    for phase in loop.phases:
        # A per-phase Timing is an explicit override and therefore takes
        # precedence over any model-bank oracle.
        timing = phase.timing
        if timing is None:
            if oracle is None:
                raise TimingResolutionError(
                    "phase %s has no static Timing; pass a CostOracle to lower_periodic"
                    % phase.name
                )
            try:
                timing = oracle.resolve(phase)
            except (AttributeError, TypeError) as error:
                raise TimingResolutionError(
                    "CostOracle must provide resolve(phase) -> Timing"
                ) from error
        if not isinstance(timing, Timing):
            raise TimingResolutionError(
                "timing oracle returned %r for phase %s, expected Timing"
                % (type(timing).__name__, phase.name)
            )
        result[phase.name] = timing
    return result


def _backend_iteration_offsets(loop: PeriodicLoopIR) -> Dict[str, int]:
    offsets = {phase.name: 0 for phase in loop.phases}
    known_phases = {phase.name: phase for phase in loop.phases}
    placed = {}  # type: Dict[str, int]

    def record(sequence: Sequence[Occurrence]) -> None:
        for occurrence in sequence:
            name = occurrence.phase.name
            if known_phases.get(name) != occurrence.phase:
                raise ModelingValidationError(
                    "sequence phase %s belongs to another periodic loop" % name
                )
            previous = placed.get(name)
            if previous is not None and previous != occurrence.window_offset:
                raise ModelingValidationError(
                    "phase %s has conflicting window offsets %d and %d"
                    % (name, previous, occurrence.window_offset)
                )
            placed[name] = occurrence.window_offset
            offsets[name] = occurrence.window_offset

    for actor in loop.actors:
        record(actor.sequence)
    for resource_sequence in loop.resource_sequences:
        record(resource_sequence.sequence)
    return offsets


def _resource_users(
    loop: PeriodicLoopIR, timings: Mapping[str, Timing]
) -> Dict[str, Tuple[str, ...]]:
    result = {}  # type: Dict[str, List[str]]
    for phase in loop.phases:
        for use in timings[phase.name].resources:
            result.setdefault(use.resource, []).append(phase.name)
    return {resource: tuple(users) for resource, users in result.items()}


def _fixed_resource_orders(
    loop: PeriodicLoopIR, timings: Mapping[str, Timing]
) -> Tuple[FixedResourceOrder, ...]:
    users_by_resource = _resource_users(loop, timings)
    result = []
    seen_resources = set()
    for resource_sequence in loop.resource_sequences:
        resource = resource_sequence.resource
        if resource in seen_resources:
            raise ModelingValidationError(
                "resource %s has more than one fixed sequence" % resource
            )
        seen_resources.add(resource)
        actual_users = users_by_resource.get(resource, tuple())
        if not actual_users:
            raise ModelingValidationError(
                "resource sequence names %s, but no phase timing uses it" % resource
            )
        phases_without_resource = [
            occurrence.phase.name
            for occurrence in resource_sequence.sequence
            if not any(
                use.resource == resource
                for use in timings[occurrence.phase.name].resources
            )
        ]
        if phases_without_resource:
            raise ModelingValidationError(
                "every occurrence in resource sequence %s must use it; "
                "missing from phase timings: %s"
                % (resource, phases_without_resource)
            )
        ordered_users = tuple(
            occurrence.phase.name for occurrence in resource_sequence.sequence
        )
        if set(ordered_users) != set(actual_users) or len(ordered_users) != len(
            actual_users
        ):
            missing = sorted(set(actual_users) - set(ordered_users))
            extra = sorted(set(ordered_users) - set(actual_users))
            raise ModelingValidationError(
                "resource sequence does not cover every %s user exactly once; "
                "missing=%s extra=%s"
                % (resource, missing, extra)
            )
        result.append(FixedResourceOrder(resource, ordered_users))
    return tuple(result)


def _event_offset(event: Event, timings: Mapping[str, Timing]) -> float:
    if event.kind == "start":
        return 0.0
    return timings[event.phase_name].latency


class _DependencyCollector:
    def __init__(self, timings: Mapping[str, Timing]) -> None:
        self.timings = timings
        self.items = []  # type: List[BackendDependency]
        self._keys = set()

    def add(
        self,
        source: Event,
        target: Event,
        distance: int,
        lag: float,
        name: str,
    ) -> None:
        if target.kind != "start":
            raise ModelingValidationError(
                "v0 lowering supports dependencies only to phase start events"
            )
        delay = _event_offset(source, self.timings)
        delay += lag
        key = (source.phase_name, target.phase_name, distance, delay)
        if key in self._keys:
            return
        self._keys.add(key)
        self.items.append(
            BackendDependency(
                source.phase_name,
                target.phase_name,
                iteration_distance=distance,
                min_delay=delay,
                name=name,
            )
        )


def _add_actor_sequences(
    collector: _DependencyCollector, actors: Sequence[ActorIR]
) -> None:
    for actor in actors:
        sequence = actor.sequence
        if not sequence:
            continue
        for index, current in enumerate(sequence):
            following = sequence[(index + 1) % len(sequence)]
            wraps = 1 if index + 1 == len(sequence) else 0
            distance = (
                wraps
                + following.window_offset
                - current.window_offset
            )
            source = (
                current.phase.start
                if actor.order == "issue"
                else current.phase.done
            )
            collector.add(
                source,
                following.phase.start,
                distance,
                0.0,
                "actor:%s:cyclic:%s->%s"
                % (actor.name, current.phase.name, following.phase.name),
            )


def _add_buffer_flows(
    collector: _DependencyCollector, loop: PeriodicLoopIR
) -> None:
    writers = {}  # type: Dict[str, List[Phase]]
    readers = {}  # type: Dict[str, List[Phase]]
    for phase in loop.phases:
        for buffer in phase.writes:
            writers.setdefault(buffer.qualified_name, []).append(phase)
        for buffer in phase.reads:
            readers.setdefault(buffer.qualified_name, []).append(phase)

    for buffer in loop.buffers:
        buffer_writers = writers.get(buffer.qualified_name, [])
        unique_writers = {phase.name: phase for phase in buffer_writers}
        if len(unique_writers) > 1:
            raise ModelingValidationError(
                "buffer %s has multiple writers %s; v0 requires one semantic producer"
                % (buffer.name, sorted(unique_writers))
            )
        if not unique_writers:
            # External/read-only or explicitly synchronized storage.
            continue
        producer = next(iter(unique_writers.values()))
        for consumer in readers.get(buffer.qualified_name, []):
            if consumer.name == producer.name:
                continue
            # Window placement describes actor/resource issue order only.
            # Buffer value flow is deliberately semantic same-iteration (0),
            # independent of producer/consumer ``window_offset`` labels.
            collector.add(
                producer.done,
                consumer.start,
                0,
                0.0,
                "buffer:%s:%s->%s"
                % (buffer.name, producer.name, consumer.name),
            )


def _handoffs_for_loop(kernel: KernelIR, loop: PeriodicLoopIR) -> Tuple[Handoff, ...]:
    result = []
    for plan in kernel.fusion_plans:
        for handoff in plan.handoffs:
            owners = {handoff.acquire.owner}
            owners.update(event.owner for event in handoff.releases)
            if loop.owner not in owners:
                continue
            if owners != {loop.owner}:
                raise ModelingValidationError(
                    "handoff %s crosses periodic loops and cannot lower into one PeriodicDAG"
                    % handoff.name
                )
            result.append(handoff)
    return tuple(result)


def _lower_dependencies(
    kernel: KernelIR, loop: PeriodicLoopIR, timings: Mapping[str, Timing]
) -> Tuple[BackendDependency, ...]:
    collector = _DependencyCollector(timings)
    for dependency in loop.dependencies:
        collector.add(
            dependency.source,
            dependency.target,
            dependency.iteration_distance,
            dependency.lag,
            dependency.name,
        )
    for carry in loop.carries:
        collector.add(
            carry.source,
            carry.target,
            carry.iteration_distance,
            carry.lag,
            "state:%s" % carry.state.name,
        )
    _add_actor_sequences(collector, loop.actors)
    _add_buffer_flows(collector, loop)
    for handoff in _handoffs_for_loop(kernel, loop):
        seen_consumers = set()
        for access in handoff.baseline_reads:
            key = (access.fragment, access.event.owner, access.event.phase_name)
            if key in seen_consumers:
                continue
            seen_consumers.add(key)
            collector.add(
                handoff.acquire,
                access.event,
                handoff.iteration_distance,
                0.0,
                "handoff:%s:data_ready:%s" % (handoff.name, access.fragment),
            )
    return tuple(collector.items)


def _lower_token_buffers(
    kernel: KernelIR, loop: PeriodicLoopIR, timings: Mapping[str, Timing]
) -> Tuple[TokenBuffer, ...]:
    seen = set()
    result = []
    for item in loop.pipeline_buffers:
        if item.buffer.qualified_name in seen:
            raise ModelingValidationError(
                "buffer %s has more than one pipeline lifetime" % item.buffer.name
            )
        seen.add(item.buffer.qualified_name)
        result.append(
            TokenBuffer(
                name=item.buffer.name,
                acquire=item.acquire.phase_name,
                release=item.release.phase_name,
                capacity=item.capacity,
                acquire_offset=_event_offset(item.acquire, timings),
                release_offset=_event_offset(item.release, timings),
                minimum_residence=item.minimum_residence,
            )
        )
    for handoff in _handoffs_for_loop(kernel, loop):
        for index, release in enumerate(handoff.releases):
            result.append(
                TokenBuffer(
                    # Frontend identifiers cannot contain '/', so this
                    # internal namespace cannot collide with buffer-derived
                    # token names.
                    name="__fusion__/%s/consumer%d" % (handoff.name, index),
                    acquire=handoff.acquire.phase_name,
                    release=release.phase_name,
                    capacity=handoff.slots,
                    acquire_offset=_event_offset(handoff.acquire, timings),
                    release_offset=_event_offset(release, timings),
                )
            )
    return tuple(result)


def lower_periodic(
    kernel: KernelIR,
    loop: str,
    oracle: Optional[Any] = None,
) -> PeriodicDAG:
    """Lower one named temporal loop; launch/grid concurrency stays in IR."""

    if not isinstance(kernel, KernelIR):
        raise ModelingValidationError("lower_periodic expects a frozen KernelIR")
    loop_ir = kernel.periodic_loop(loop)
    timings = _resolve_timings(loop_ir, oracle)
    offsets = _backend_iteration_offsets(loop_ir)
    phases = tuple(
        BackendPhase(
            phase.name,
            latency=timings[phase.name].latency,
            resources=tuple(
                ResourceUse(
                    resource.resource,
                    service_time=resource.service_time,
                    offset=resource.offset,
                )
                for resource in timings[phase.name].resources
            ),
            iteration_offset=offsets[phase.name],
        )
        for phase in loop_ir.phases
    )
    return PeriodicDAG(
        phases=phases,
        dependencies=_lower_dependencies(kernel, loop_ir, timings),
        token_buffers=_lower_token_buffers(kernel, loop_ir, timings),
        fixed_resource_orders=_fixed_resource_orders(loop_ir, timings),
    )


__all__ = ["lower_periodic"]
