"""Lower one ``Launch`` (under one work group) to ``KernelIR`` + region tree.

The existing recursive region evaluator and ``PeriodicDAG`` lowering are
reused unchanged: every ``Pipeline`` becomes a frontend periodic loop, every
op becomes a ``Phase`` named by its ``op_id``, and the region tree mirrors
``Sequence``/``Loop`` structure with ``PeriodicAxisIR`` on pipeline loops.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..errors import ModelingValidationError
from ..frontend import Kernel
from ..ir import KernelIR, LaunchIR, Phase, Work
from ..regions import LoopRegion, PeriodicAxisIR, PhaseRegion, SequenceRegion
from .contract import (
    Compute, INTERNAL_STORAGE, Launch, Load, Loop, Op, OpSite, Pipeline, Sequence,
    Store, Sync, WorkGroup, is_op, move_payload_bytes,
)
from .options import Options


class LoweringError(ModelingValidationError):
    pass


@dataclass
class LoweredLaunch:
    launch: Launch
    group_id: str
    group: WorkGroup
    kernel: KernelIR
    region: Any
    phases: Dict[str, Phase]
    sites: Dict[str, OpSite]
    trip_counts: Tuple[Tuple[str, int], ...]
    loop_paths: Dict[str, str] = field(default_factory=dict)


def _work_for(op: Op) -> Work:
    if isinstance(op, Load):
        return Work("load", bytes=move_payload_bytes(op), attrs={
            "source": INTERNAL_STORAGE[op.source], "destination": INTERNAL_STORAGE[op.destination],
            "engine": op.engine,
        })
    if isinstance(op, Store):
        return Work("store", bytes=move_payload_bytes(op), attrs={
            "source": INTERNAL_STORAGE[op.source], "destination": INTERNAL_STORAGE[op.destination],
            "engine": op.engine,
        })
    if isinstance(op, Compute):
        return Work("compute", flops=op.executed_flops, attrs={
            "engine": op.engine or "spec", "op": op.spec.op_id,
        })
    if isinstance(op, Sync):
        return Work("sync", attrs={"sync_kind": op.sync_kind})
    raise LoweringError("unknown op %r" % (op,))


def _lower_pipeline(kernel_launch: Any, loop: Loop, trip_count: int, phases: Dict[str, Phase]) -> None:
    pipeline = loop.body  # type: Pipeline
    periodic = kernel_launch.periodic(loop.loop_id, iterations=trip_count, stages=pipeline.stages)
    buffers = {}
    for buffer in pipeline.buffers:
        buffers[buffer.name] = periodic.buffer(
            buffer.name, scope=buffer.storage, shape=buffer.shape, dtype=buffer.dtype,
            slots=buffer.slots, execution_scope=buffer.execution_scope,
        )
    actor_builders = {}
    for actor in pipeline.actors:
        builder = periodic.actor(actor.name, execution_scope=actor.execution_scope)
        actor_builders[actor.name] = builder
        for op in actor.ops:
            phase = builder.phase(
                op.op_id,
                work=_work_for(op),
                timing=op.timing,
                reads=tuple(buffers[name] for name in op.reads),
                writes=tuple(buffers[name] for name in op.writes),
            )
            phases[op.op_id] = phase
    for actor in pipeline.actors:
        if actor.sequence:
            actor_builders[actor.name].sequence(
                *[phases[op_id].at(offset) for op_id, offset in actor.sequence],
                order=actor.order,
                resource=actor.resource,
            )

    def event(ref: Any) -> Any:
        phase = phases[ref.op_id]
        return phase.start if ref.kind == "start" else phase.done

    for dep in pipeline.dependencies:
        periodic.after(event(dep.source), event(dep.target), distance=dep.distance,
                       lag=dep.lag_s, name=dep.name)
    for carry in pipeline.carries:
        state = periodic.state(
            carry.state, storage=None if carry.storage is None else buffers[carry.storage]
        )
        periodic.carry(state, source=event(carry.source), target=event(carry.target),
                       distance=carry.distance, lag=carry.lag_s)
    for item in pipeline.buffer_slots:
        periodic.pipeline_buffer(
            buffers[item.buffer], acquire=event(item.acquire), release=event(item.release),
            capacity=item.capacity,
        )
    for order in pipeline.resource_orders:
        periodic.resource_sequence(
            order.resource, *[phases[op_id].at(offset) for op_id, offset in order.sequence]
        )


class _RegionBuilder:
    def __init__(self, launch: Launch, group: WorkGroup, options: Options, phases: Dict[str, Phase]):
        self.launch = launch
        self.group = group
        self.options = options
        self.phases = phases
        self.serial_owner = "%s/__serial__" % launch.name
        self.sequence_counter = 0
        self.loop_paths = {}  # type: Dict[str, str]

    def phase_for(self, op: Op) -> Phase:
        found = self.phases.get(op.op_id)
        if found is None:
            found = Phase(
                name=op.op_id, actor=op.actor, owner=self.serial_owner,
                work=_work_for(op), timing=op.timing,
            )
            self.phases[op.op_id] = found
        return found

    def build(self, region: Any, path: str) -> Any:
        if is_op(region):
            return PhaseRegion(self.phase_for(region))
        if isinstance(region, Sequence):
            name = region.name
            if name is None:
                self.sequence_counter += 1
                name = "seq%d" % self.sequence_counter
            children = tuple(self.build(child, path) for child in region.children)
            return SequenceRegion(name, children)
        if isinstance(region, Loop):
            loop_path = "%s/%s" % (path, region.loop_id)
            self.loop_paths[region.loop_id] = loop_path
            trip = self.launch.bound_trip_count(region, self.group)
            prologue = None if region.prologue is None else self.build(region.prologue, loop_path)
            epilogue = None if region.epilogue is None else self.build(region.epilogue, loop_path)
            if isinstance(region.body, Pipeline):
                leaves = tuple(PhaseRegion(self.phases[op.op_id]) for op in region.body.ops)
                body = SequenceRegion("pipeline", leaves) if len(leaves) > 1 else leaves[0]
                axis = PeriodicAxisIR(
                    loop_name=region.loop_id,
                    ii_mode=region.ii_mode,
                    boundary_anchor_phase=region.boundary_anchor_op,
                    search_config=self.options.search_config,
                )
            else:
                body = self.build(region.body, loop_path)
                axis = None
            return LoopRegion(
                region.loop_id, trip_count=trip, body=body, prologue=prologue,
                epilogue=epilogue, periodic_axis=axis,
            )
        raise LoweringError("cannot lower region %r" % (region,))


def lower_launch(launch: Launch, group_id: str, group: WorkGroup, options: Options) -> LoweredLaunch:
    """Build the KernelIR and region tree of ``launch`` for one work group."""

    kernel = Kernel("program_%s" % launch.name)
    phases = {}  # type: Dict[str, Phase]
    trip_counts = []
    pipelines = [loop for loop in launch.loops() if isinstance(loop.body, Pipeline)]
    for loop in launch.loops():
        trip_counts.append((loop.loop_id, launch.bound_trip_count(loop, group)))
    residency = launch.residency
    if pipelines:
        with kernel.launch(
            name=launch.name,
            work_grid=launch.physical_grid,
            physical_grid=launch.physical_grid,
            threads=launch.threads,
            cluster=launch.cluster,
            residency=residency,
            scheduler="static" if isinstance(launch.scheduler, str) else "persistent",
        ) as kernel_launch:
            for loop in pipelines:
                _lower_pipeline(kernel_launch, loop, launch.bound_trip_count(loop, group), phases)
        kernel_ir = kernel.build()
    else:
        kernel_ir = KernelIR(
            name=kernel.name,
            params=tuple(),
            launches=(LaunchIR(
                name=launch.name, work_grid=launch.physical_grid, physical_grid=launch.physical_grid,
                threads=launch.threads, cluster=launch.cluster, residency=residency,
                scheduler="static" if isinstance(launch.scheduler, str) else "persistent",
                swizzle=tuple(), periodic_loops=tuple(),
            ),),
        )
    builder = _RegionBuilder(launch, group, options, phases)
    region = builder.build(launch.body, launch.name)
    if isinstance(region, PhaseRegion):
        region = SequenceRegion("body", (region,))
    sites = {site.op.op_id: site for site in launch.op_sites()}
    return LoweredLaunch(
        launch=launch, group_id=group_id, group=group, kernel=kernel_ir, region=region,
        phases=phases, sites=sites, trip_counts=tuple(sorted(trip_counts)),
        loop_paths=builder.loop_paths,
    )


__all__ = ["LoweredLaunch", "LoweringError", "lower_launch"]
