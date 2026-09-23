"""Exact distinct reuse trace over the *real* work-instance sequence.

This path expands every work instance of a launch with its own work-group
trip counts and walks the declared program order.  Instances that share a
wave (static dispatch) or a persistent CTA stream advance in lockstep.  A
``TileAccess`` identity is ``(tensor, projected key)`` where a projected axis
is either a launch work axis (instance coordinate) or an enclosing loop id
(current iteration); an access that does not project on its loop is
loop-invariant and correctly reuses the same allocation every iteration.

Reuse distances use the existing weighted distinct LRU stack and SDCM hit
model.  Traffic follows the write-through / allocate / no-RFO defaults of
``cache.legacy_volume._traffic_for_bin``.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from tilesight.util.sdcm import sdcm

from ..cache.ir import CacheLevelConfig
from .traffic import Traffic
from ..cache.tile_reuse_distance import _WeightedLRUStack
from ..errors import ModelingError
from .contract import (
    Launch, Load, Loop, Persistent, Pipeline, Sequence, Store, TileAccess, WorkUnits,
    is_op, touches_global,
)


class ExplicitTraceError(ModelingError):
    pass


@dataclass
class _Stats:
    """Per-stratum accumulators.

    Deterministic quantities (request count, payload/transaction bytes per
    request, direction) are exact.  Only the hit probabilities are
    Horvitz-Thompson estimates; the byte fields are derived from the clamped
    rates so rates and bytes always describe the same estimate.
    """

    requests: float = 0.0
    weight_sum: float = 0.0
    l1_5_hits: float = 0.0
    l2_served: float = 0.0
    ddr_misses: float = 0.0
    payload_bytes: float = 0.0
    transaction_bytes: float = 0.0
    is_read: bool = True
    has_l1_5: bool = False
    finalized: bool = False

    def finalize(self) -> None:
        """Turn Horvitz-Thompson sums into one hierarchy-consistent rate estimate.

        Per event the three probabilities sum to one, so their weighted sums
        divided by the total weight are rates in ``[0, 1]`` that sum to one
        (L1.5 hit + L2 served + DDR miss = 1) and DDR reads never exceed L2
        read requests.  This is a ratio estimate: consistent, and exactly the
        trace value without sampling, but not exactly unbiased for a finite
        sample.
        """

        if self.finalized:
            return
        total = self.weight_sum
        if total <= 0.0:
            self.l1_5_hits = self.l2_served = self.ddr_misses = 0.0
        else:
            scale = self.requests / total
            # Floating-point guard only: the HT sums are bounded by total.
            self.l1_5_hits = min(self.l1_5_hits * scale, self.requests)
            self.l2_served = min(self.l2_served * scale, self.requests - self.l1_5_hits)
            self.ddr_misses = max(self.requests - self.l1_5_hits - self.l2_served, 0.0)
        self.finalized = True

    @property
    def l1_5_rate(self) -> float:
        return self.l1_5_hits / self.requests if self.requests > 0 else 0.0

    @property
    def l2_rate(self) -> float:
        return self.l2_served / self.requests if self.requests > 0 else 0.0

    @property
    def ddr_rate(self) -> float:
        return self.ddr_misses / self.requests if self.requests > 0 else 0.0

    def cache_traffic(self) -> Traffic:
        """Cumulative traffic of the stratum, derived from exact counts and estimated rates."""

        if not self.finalized:
            self.finalize()
        n = self.requests
        payload = self.payload_bytes * n
        transaction = self.transaction_bytes * n
        amplification = max(transaction - payload, 0.0)
        if self.is_read:
            return Traffic(
                payload_read_bytes=payload,
                l1_5_read_request_bytes=transaction if self.has_l1_5 else 0.0,
                l2_read_request_bytes=transaction * (1.0 - self.l1_5_rate),
                ddr_read_bytes=transaction * self.ddr_rate,
                transaction_amplification_bytes=amplification,
            )
        return Traffic(
            payload_write_bytes=payload,
            l1_5_write_request_bytes=transaction if self.has_l1_5 else 0.0,
            l2_write_request_bytes=transaction,
            ddr_write_bytes=transaction,
            transaction_amplification_bytes=amplification,
        )


@dataclass(frozen=True)
class ExplicitTraceResult:
    per_op: Dict[str, _Stats]
    per_op_group: Dict[Tuple[str, str], _Stats]
    events: int
    sampled_events: int
    exact: bool
    notes: Tuple[str, ...]


def _instance_events(
    region: Any, launch: Launch, group: Any, loop_index: Dict[str, int],
) -> Iterator[Tuple[Any, Dict[str, int]]]:
    if is_op(region):
        if touches_global(region) and region.access is not None:
            yield region, dict(loop_index)
        return
    if isinstance(region, Sequence):
        for child in region.children:
            yield from _instance_events(child, launch, group, loop_index)
        return
    if isinstance(region, Loop):
        trips = launch.bound_trip_count(region, group)
        if region.prologue is not None:
            yield from _instance_events(region.prologue, launch, group, loop_index)
        for iteration in range(trips):
            loop_index[region.loop_id] = iteration
            if isinstance(region.body, Pipeline):
                for op in region.body.ops:
                    if touches_global(op) and op.access is not None:
                        yield op, dict(loop_index)
            else:
                yield from _instance_events(region.body, launch, group, loop_index)
        loop_index.pop(region.loop_id, None)
        if region.epilogue is not None:
            yield from _instance_events(region.epilogue, launch, group, loop_index)
        return
    raise ExplicitTraceError("unexpected region %r" % (region,))


def _streams(
    launch: Launch, units: WorkUnits, slots: int,
) -> Iterator[List[List[int]]]:
    """Yield concurrent instance streams: one wave at a time."""

    total = units.total_count
    if isinstance(launch.scheduler, Persistent):
        yield [
            [index for begin, end in part for index in range(begin, end)]
            for part in launch.scheduler.partitions
        ]
        return
    for start in range(0, total, slots):
        yield [[index] for index in range(start, min(start + slots, total))]


def _sampling_plan(
    counts: Dict[Tuple[str, str], int], sample_budget: Optional[int], seed: int,
) -> Dict[Tuple[str, str], Tuple[int, int, float]]:
    """Stratified systematic sampling per (op, work group).

    The per-op budget is split across groups proportionally to their event
    counts (at least one sample per group).  Each stratum gets a seeded phase
    offset; every event's inclusion probability is ``1/period`` regardless of
    the phase, so the Horvitz-Thompson weight is exactly ``period``.  Totals
    are therefore unbiased over the phase distribution; the exact event count
    is used as the request denominator instead of the weight sum.
    """

    per_op = {}  # type: Dict[str, int]
    for (op_id, _group), count in counts.items():
        per_op[op_id] = per_op.get(op_id, 0) + count
    plan = {}
    for (op_id, group_id), count in counts.items():
        if count == 0:
            continue
        if sample_budget is None or per_op[op_id] <= sample_budget:
            plan[(op_id, group_id)] = (1, 0, 1.0)
            continue
        budget = max(1, int(round(sample_budget * count / float(per_op[op_id]))))
        period = max(1, int(math.ceil(count / float(budget))))
        token = "%d:%s:%s" % (seed, op_id, group_id)
        offset = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:16], 16) % period
        if offset >= count:
            offset = 0
        plan[(op_id, group_id)] = (period, offset, float(period))
    return plan


def run_explicit_trace(
    launch: Launch,
    units: WorkUnits,
    slots: int,
    residency: int,
    l2: CacheLevelConfig,
    l1_5: Optional[CacheLevelConfig],
    l1_5_group_size: int,
    sample_budget: Optional[int] = None,
    seed: int = 0,
) -> ExplicitTraceResult:
    extents = [axis.extent for axis in launch.work_axes]
    axis_names = launch.work_axis_names
    group_ids = units.instance_group_ids()
    groups = dict(units.groups)
    # Counting pass for the sampling plan and stack sizing.
    counts = {}  # type: Dict[Tuple[str, str], int]
    total_events = 0
    for index, group_id in enumerate(group_ids):
        for op, _loops in _instance_events(launch.body, launch, groups[group_id], {}):
            counts[(op.op_id, group_id)] = counts.get((op.op_id, group_id), 0) + 1
            total_events += 1
    plan = _sampling_plan(counts, sample_budget, seed)
    exact = all(period == 1 for period, _offset, _weight in plan.values())

    has_l1_5 = l1_5 is not None
    group_count = 1
    if has_l1_5:
        logical_sms = max(slots // max(residency, 1), 1)
        group_count = max((logical_sms + l1_5_group_size - 1) // l1_5_group_size, 1)
    l2_stack = _WeightedLRUStack(total_events + 8)
    l1_5_stacks = [_WeightedLRUStack(total_events + 8) for _ in range(group_count)]
    l2_unit = l2.unit_bytes
    l1_5_unit = l1_5.unit_bytes if has_l1_5 else l2_unit
    per_op = {}  # type: Dict[str, _Stats]
    per_op_group = {}  # type: Dict[Tuple[str, str], _Stats]
    occurrence = {}  # type: Dict[Tuple[str, str], int]
    sampled = 0
    coordinates = launch.instance_coordinates()

    def identity_key(access: TileAccess, coordinate: Tuple[int, ...], loops: Dict[str, int]) -> Tuple[int, ...]:
        key = []
        for axis in access.index_map.work_axes:
            if axis in loops:
                key.append(loops[axis])
            else:
                key.append(coordinate[axis_names.index(axis)])
        return tuple(key)

    def observe(op: Any, group_id: str, slot: int, coordinate: Tuple[int, ...], loops: Dict[str, int]) -> None:
        nonlocal sampled
        access = op.access
        identity = (access.tensor, identity_key(access, coordinate, loops))
        is_read = isinstance(op, Load)
        allocation = access.effective_allocation_bytes
        l2_distance, l2_cold = l2_stack.access(identity, allocation / l2_unit, True)
        if has_l1_5:
            sm = slot // max(residency, 1)
            group = min(sm // l1_5_group_size, group_count - 1)
            l1_5_distance, l1_5_cold = l1_5_stacks[group].access(identity, allocation / l1_5_unit, True)
        stratum = (op.op_id, group_id)
        index = occurrence.get(stratum, 0)
        occurrence[stratum] = index + 1
        period, offset, weight = plan[stratum]
        if index < offset or (index - offset) % period != 0:
            return
        sampled += 1
        p_l1_5 = 0.0
        if has_l1_5 and not l1_5_cold:
            p_l1_5 = min(max(float(sdcm(l1_5_distance, l1_5.associativity, l1_5.capacity_units)), 0.0), 1.0)
        p_l2 = 0.0
        if not l2_cold:
            p_l2 = min(max(float(sdcm(l2_distance, l2.associativity, l2.capacity_units)), 0.0), 1.0)
        served = (1.0 - p_l1_5) * p_l2
        miss = (1.0 - p_l1_5) * (1.0 - p_l2)
        for stats in (per_op.setdefault(op.op_id, _Stats()),
                      per_op_group.setdefault((op.op_id, group_id), _Stats())):
            stats.weight_sum += weight
            stats.l1_5_hits += weight * p_l1_5
            stats.l2_served += weight * served
            stats.ddr_misses += weight * miss
            stats.payload_bytes = access.payload_bytes
            stats.transaction_bytes = access.executed_bytes
            stats.is_read = is_read
            stats.has_l1_5 = has_l1_5

    for streams in _streams(launch, units, slots):
        generators = []
        for slot, instances in enumerate(streams):
            def gen(instances=instances, slot=slot):
                for index in instances:
                    group_id = group_ids[index]
                    coordinate = coordinates[index]
                    for op, loops in _instance_events(launch.body, launch, groups[group_id], {}):
                        yield op, group_id, slot, coordinate, loops
            generators.append(gen())
        active = list(range(len(generators)))
        while active:
            still = []
            for position in active:
                try:
                    op, group_id, slot, coordinate, loops = next(generators[position])
                except StopIteration:
                    continue
                observe(op, group_id, slot, coordinate, loops)
                still.append(position)
            active = still
    # Exact request counts per stratum; per-op results are the sums of the
    # finalized group results so cost groups and reports always close.
    for (op_id, group_id), count in counts.items():
        group_stats = per_op_group.setdefault((op_id, group_id), _Stats())
        group_stats.requests += count
    for stats in per_op_group.values():
        stats.finalize()
    per_op = {}
    for (op_id, _group_id), stats in per_op_group.items():
        target = per_op.setdefault(op_id, _Stats())
        target.requests += stats.requests
        target.weight_sum += stats.requests
        target.l1_5_hits += stats.l1_5_hits
        target.l2_served += stats.l2_served
        target.ddr_misses += stats.ddr_misses
        target.payload_bytes = stats.payload_bytes
        target.transaction_bytes = stats.transaction_bytes
        target.is_read = stats.is_read
        target.has_l1_5 = stats.has_l1_5
    for stats in per_op.values():
        stats.finalize()
    notes = (
        "explicit trace over %d work instances in %s order (%d global access events, %d observed%s)"
        % (units.total_count, "persistent streams" if isinstance(launch.scheduler, Persistent)
           else "waves of %d lockstep instances" % slots, total_events, sampled,
           "" if exact else "; stratified systematic sampling per (op, work group), seed %d, "
           "Horvitz-Thompson weights = period; rates are ratio estimates (sum to one) and bytes derive from them" % seed),
        "cache state between launches: cold",
    )
    return ExplicitTraceResult(per_op, per_op_group, total_events, sampled, exact, notes)


__all__ = ["ExplicitTraceError", "ExplicitTraceResult", "run_explicit_trace"]
