"""Spatial dispatch of work units onto SM slots (list scheduling)."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from ..errors import ModelingError
from .contract import Persistent

MAX_RECORDED_UNITS = 4096


class SpatialError(ModelingError):
    pass


@dataclass(frozen=True)
class SpatialResult:
    kernel_body_s: float
    slot_completion_s: Tuple[float, ...]
    critical_unit: Optional[int]
    critical_group: Optional[str]
    waves: int
    tail_work_units: int
    scope: str
    scheduler: str
    unit_timeline: Tuple[Tuple[int, int, str, float, float], ...] = field(default_factory=tuple)
    timeline_complete: bool = True


def list_schedule(
    group_ids: Sequence[str],
    cost_by_group: Mapping[str, float],
    slots: int,
    tail_units: int = 0,
    tail_cost_by_group: Optional[Mapping[str, float]] = None,
) -> SpatialResult:
    """Assign each work unit (in dispatch order) to the earliest free slot.

    With ``tail_units`` the last ``tail_units`` work units form the partial final
    wave: they start only after every earlier unit has finished (static dispatch
    launches wave by wave) and take their cost from ``tail_cost_by_group``.
    """

    if slots <= 0:
        raise SpatialError("list scheduling needs at least one slot")
    total = len(group_ids)
    if total == 0:
        raise SpatialError("launch has no work units")
    heap = [(0.0, index) for index in range(min(slots, total))]
    heapq.heapify(heap)
    completion = [0.0] * len(heap)
    critical = (0.0, None, None)  # type: Tuple[float, Optional[int], Optional[str]]
    timeline = []  # type: List[Tuple[int, int, str, float, float]]
    record = total <= MAX_RECORDED_UNITS
    if tail_units and tail_cost_by_group is None:
        raise SpatialError("tail_units needs tail_cost_by_group")
    tail_begin = total - tail_units if tail_units else total
    for index, group_id in enumerate(group_ids):
        if index == tail_begin:
            barrier = max(completion)
            heap = [(barrier, slot) for slot in range(len(completion))]
            heapq.heapify(heap)
        start, slot = heapq.heappop(heap)
        costs = tail_cost_by_group if index >= tail_begin else cost_by_group
        end = start + costs[group_id]
        completion[slot] = end
        heapq.heappush(heap, (end, slot))
        if end > critical[0]:
            critical = (end, index, group_id)
        if record:
            timeline.append((index, slot, group_id, start, end))
    body = max(completion) if completion else 0.0
    waves = (total + slots - 1) // slots
    tail = total % slots
    return SpatialResult(
        kernel_body_s=body,
        slot_completion_s=tuple(completion),
        critical_unit=critical[1],
        critical_group=critical[2],
        waves=waves,
        tail_work_units=tail,
        scope="list_schedule_tail_active_sms" if tail_units else "list_schedule_full_wave_costs",
        scheduler="static_dispatch",
        unit_timeline=tuple(timeline),
        timeline_complete=record,
    )


def persistent_schedule(
    group_ids: Sequence[str],
    cost_by_group: Mapping[str, float],
    persistent: Persistent,
    slots: int,
) -> SpatialResult:
    """Each persistent CTA executes its explicit partition in order."""

    total = len(group_ids)
    persistent.validate_coverage(total)
    if persistent.cta_count > slots:
        raise SpatialError(
            "persistent scheduler declares %d CTAs but only %d slots are resident"
            % (persistent.cta_count, slots)
        )
    completion = []
    critical = (0.0, None, None)  # type: Tuple[float, Optional[int], Optional[str]]
    timeline = []  # type: List[Tuple[int, int, str, float, float]]
    record = total <= MAX_RECORDED_UNITS
    for cta, part in enumerate(persistent.partitions):
        clock = 0.0
        for begin, end in part:
            for index in range(begin, end):
                group_id = group_ids[index]
                start = clock
                clock += cost_by_group[group_id]
                if record:
                    timeline.append((index, cta, group_id, start, clock))
                if clock > critical[0]:
                    critical = (clock, index, group_id)
        completion.append(clock)
    return SpatialResult(
        kernel_body_s=max(completion),
        slot_completion_s=tuple(completion),
        critical_unit=critical[1],
        critical_group=critical[2],
        waves=1,
        tail_work_units=0,
        scope="persistent_partition_full_wave_costs",
        scheduler="persistent",
        unit_timeline=tuple(timeline),
        timeline_complete=record,
    )


__all__ = ["MAX_RECORDED_UNITS", "SpatialError", "SpatialResult", "list_schedule", "persistent_schedule"]
