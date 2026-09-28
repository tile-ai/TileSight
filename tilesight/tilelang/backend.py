"""Architecture profiles, analysis and the composition of resident CTAs."""

from __future__ import annotations

import importlib
from collections import defaultdict
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Dict, Tuple

from ..modeling import program as sight
from ..modeling.native_executor import _resident_group_bound
from ..modeling.native_policy import LaunchTopology
from ..modeling.program.spatial import list_schedule


def arch(name: str = "H200_SXM", profile: str = "set_to_microbench") -> Any:
    module = importlib.import_module("tilesight.arch.%s" % name.lower())
    instance = getattr(module, name)()
    result = getattr(instance, profile)()
    return result if result is not None else instance


def options(ii_mode: str = "periodic_best") -> Any:
    return sight.Options(ii_mode=ii_mode, cache="fast", cache_sample_budget=None)


def analyze(program: Any, arch_obj: Any, *, context: Any = None, ii_mode: str = "periodic_best") -> Any:
    return sight.analyze(program, arch_obj, options=options(ii_mode), context=context)


@dataclass(frozen=True)
class Estimate:
    """Kernel body time with the rule that composed resident CTAs.

    ``single`` is TileSight's analysis of one CTA per SM at full SM rates. With
    more than one resident CTA, the point estimate composes them with TileSight's
    native per-SM resource bound (``native_executor._resident_group_bound``, as
    in ``model_native(multi_cta_policy="resource_bound")``) and dispatches work
    units with TileSight's ``list_schedule``. ``upper_s`` is the program layer's
    ``equal_share`` analysis, where resident CTAs never overlap different
    resources; the truth lies between the two.

    A launch with fewer CTAs than one wave holds is spread over the SMs first, as
    TileSight's native executor does (its ``effective_residency``):
    ``min(residency, ceil(active CTAs / active SMs))`` CTAs share an SM.
    """

    kernel_body_s: float
    upper_s: float
    residency: int
    composition: str
    single: Any
    shared: Any = None
    group_s: Tuple[Tuple[str, float], ...] = field(default=())
    effective_residency: int = 1


def estimate(build: Callable[[int], Any], arch_obj: Any, residency: int, *, context: Any = None,
             ii_mode: str = "periodic_best") -> Estimate:
    program_one = build(1)
    single = analyze(program_one, arch_obj, context=context, ii_mode=ii_mode)
    launch_one = single.launches["main"]
    active = min(launch_one.work_units, launch_one.physical_ctas, launch_one.sm_count * residency)
    effective = min(residency, -(-active // min(launch_one.sm_count, active)))
    if effective <= 1:
        return Estimate(launch_one.kernel_body_s, launch_one.kernel_body_s, residency, "single_cta", single)
    shared = analyze(build(effective), arch_obj, context=context, ii_mode=ii_mode)
    groups = {g.group_id: g for g in launch_one.groups}
    service: Dict[str, Dict[str, float]] = {gid: defaultdict(float) for gid in groups}
    for path in single.ops:
        op = single.ops[path]
        for cost in op.cost_groups:
            gid = cost.key.work_group_id
            if gid in groups and cost.key.wave_kind == "full" and groups[gid].count:
                for resource, seconds in cost.service_s:
                    service[gid][resource] += seconds * cost.count / groups[gid].count
    topology = LaunchTopology(kind="resident")
    per_unit = {}
    for gid, summary in groups.items():
        evaluation = SimpleNamespace(total_s=summary.per_unit_s, resource_service_s=tuple(service[gid].items()))
        per_unit[gid] = _resident_group_bound(evaluation, effective, topology, "resource_bound")[0]
    sequence = program_one.launches[0].effective_work_units().instance_group_ids()
    spatial = list_schedule(sequence, per_unit, launch_one.sm_count * effective)
    return Estimate(spatial.kernel_body_s, shared.launches["main"].kernel_body_s, residency,
                    "resident_group_resource_bound_nonconstructive", single, shared,
                    tuple(sorted(per_unit.items())), effective)
