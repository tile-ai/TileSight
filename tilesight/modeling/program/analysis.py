"""``analyze(program, arch, options, context) -> AnalysisResult``."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from typing import Any, Dict, List, Mapping, Optional, Tuple

from tilesight.modeling._pipeline.periodic_schedule import PeriodicScheduleError

from ..errors import ModelingError
from ..native_executor import NativeModelError, NativeModelOptions, RegionEvaluation, evaluate_region
from .traffic import Traffic, traffic_shares
from .cache_binding import CacheView, access_id_for, build_cache_view
from .contract import Compute, Launch, Load, OpSite, Persistent, Pipeline, Program, Store, move_executed_bytes, move_payload_bytes
from .generic_oracle import BindingContext, BoundOracle, GenericOracle, UnsupportedCostError, WaveView
from .lowering import LoweredLaunch, lower_launch
from .options import Context, Options
from .results import (
    AnalysisResult, CacheReport, CompletionInterval, CostGroup, CostGroupKey, Diagnostics,
    IIComponents, Index, LaunchResult, MODEL_VERSION, OpResult, ProgramResult, RegionGroupResult,
    RegionResult, ServiceInterval, TrafficInterval, WorkGroupSummary, scale_traffic,
)
from .snapshot import launch_workload_digest, program_to_snapshot, program_workload_digest
from .spatial import SpatialResult, list_schedule, persistent_schedule


class UnsupportedError(ModelingError):
    """Raised when the requested analysis cannot be produced honestly."""


def _digest(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _multiplicity(site: OpSite, trip_counts: Mapping[str, int]) -> int:
    result = 1
    for loop_id, where in site.chain:
        if where == "body":
            result *= trip_counts[loop_id]
    return result


def _find_evaluation(node: RegionEvaluation, name: str) -> Optional[RegionEvaluation]:
    if node.kind == "loop" and node.name == name:
        return node
    for child in node.children:
        found = _find_evaluation(child, name)
        if found is not None:
            return found
    return None


def _ii_components(node: RegionEvaluation, mode: str) -> Optional[IIComponents]:
    if node.selected_ii_s is None and node.resource_ii_s is None:
        return None
    return IIComponents(
        resource_ii_s=node.resource_ii_s,
        recurrence_ii_s=node.recurrence_ii_s,
        credit_ii_s=node.credit_ii_s,
        selected_ii_s=node.selected_ii_s,
        best_ii_s=node.best_ii_s,
        worst_ii_s=node.worst_ii_s,
        scope=node.ii_scope,
        mode=mode,
        search_complete=(None if node.best_ii_s is None else ("model_exhaustive" in node.ii_scope)),
        witness=node.witness_phase_starts or None,
    )


def _service_trace(node: RegionEvaluation) -> Tuple[ServiceInterval, ...]:
    return tuple(
        ServiceInterval(
            path=item.path, op_id=item.phase, resource=item.resource,
            offset_s=item.offset_s, service_time_s=item.service_time_s,
            repeats=tuple((r.label, r.count, r.stride_s) for r in item.repeats),
            ordering=item.ordering,
        )
        for item in node.resource_trace
    )


def _completion_trace(
    node: RegionEvaluation, path: str, base: Optional[float], repeats: Tuple[Tuple[str, int, float], ...],
    bound: BoundOracle, sites: Mapping[str, OpSite], out: List[CompletionInterval],
) -> None:
    """Record one completion event per op execution, independent of resources.

    ``base`` is the region start (``None`` when no witness fixes it).  Serial
    loops compress iterations into ``repeats``; periodic loops use the
    witness phase starts with stride II.
    """

    if node.kind == "phase":
        site = sites.get(node.name)
        cost = bound.cost(node.name) if node.name in bound.sites else None
        out.append(CompletionInterval(
            path=path, op_id=node.name, actor=site.op.actor if site else "?",
            start_s=base, completion_latency_s=node.total_s if cost is None else cost.completion_latency_s,
            repeats=repeats,
        ))
        return
    if node.kind == "empty":
        return
    if node.kind == "sequence":
        offset = 0.0
        for child in node.children:
            child_base = None if base is None else base + offset
            _completion_trace(child, "%s/%s" % (path, child.name), child_base, repeats, bound, sites, out)
            offset += child.total_s
        return
    if node.kind == "loop":
        children = list(node.children)
        prologue = children[0]
        epilogue = children[-1]
        body = children[1] if len(children) == 3 else None
        trip = node.trip_count or 0
        _completion_trace(prologue, "%s/prologue" % path, base, repeats, bound, sites, out)
        body_base = None if base is None else base + prologue.total_s
        if body is not None and trip > 0:
            if node.finite_event_starts:
                # Finite source schedule: one real event per (op, iteration).
                for name, iteration, start in node.finite_event_starts:
                    site = sites.get(name)
                    out.append(CompletionInterval(
                        path="%s/body/%s" % (path, name), op_id=name,
                        actor=site.op.actor if site else "?",
                        start_s=None if body_base is None else body_base + start,
                        completion_latency_s=bound.cost(name).completion_latency_s if name in bound.sites else 0.0,
                        repeats=repeats,
                    ))
            elif node.selected_ii_s is not None and node.strategy.startswith("periodic"):
                starts = dict(node.witness_phase_starts)
                loop_repeats = repeats + ((node.name, trip, node.selected_ii_s),)
                for leaf in _leaves(body):
                    start = None
                    if starts and body_base is not None:
                        start = body_base + starts.get(leaf.name, 0.0)
                    site = sites.get(leaf.name)
                    out.append(CompletionInterval(
                        path="%s/body/%s" % (path, leaf.name), op_id=leaf.name,
                        actor=site.op.actor if site else "?", start_s=start,
                        completion_latency_s=bound.cost(leaf.name).completion_latency_s if leaf.name in bound.sites else leaf.total_s,
                        repeats=loop_repeats,
                    ))
            else:
                loop_repeats = repeats + ((node.name, trip, body.total_s),)
                _completion_trace(body, "%s/body" % path, body_base, loop_repeats, bound, sites, out)
        epilogue_base = None if base is None else base + node.total_s - epilogue.total_s
        _completion_trace(epilogue, "%s/epilogue" % path, epilogue_base, repeats, bound, sites, out)
        return
    raise UnsupportedError("completion trace: unexpected region kind %r" % node.kind)


def _leaves(node: RegionEvaluation) -> List[RegionEvaluation]:
    if node.kind == "phase":
        return [node]
    result = []
    for child in node.children:
        result.extend(_leaves(child))
    return result


def _effective_ii_mode(loop: Any, options: Options) -> str:
    """The II mode that actually governs one loop: its own ``ii_mode`` unless that is ``inherit``."""

    mode = getattr(loop, "ii_mode", "inherit")
    return options.ii_mode if mode in (None, "inherit") else mode


def _region_group(
    node: RegionEvaluation, path: str, group_id: str, bound: BoundOracle, sites: Mapping[str, OpSite],
    options: Options, traffic_total: Traffic = Traffic(), ii_mode: Optional[str] = None,
) -> RegionGroupResult:
    completions = []  # type: List[CompletionInterval]
    _completion_trace(node, path, 0.0, tuple(), bound, sites, completions)
    return RegionGroupResult(
        traffic_total=traffic_total,
        work_group_id=group_id,
        trip_count=node.trip_count,
        total_s=node.total_s, first_s=node.first_s, steady_s=node.steady_s, drain_s=node.drain_s,
        strategy=node.strategy, estimate_scope=node.estimate_scope,
        ii=_ii_components(node, options.ii_mode if ii_mode is None else ii_mode),
        periodic_dag_digest=getattr(node, "periodic_dag_digest", None),
        boundary_policy=options.boundary_policy if node.kind == "loop" and node.selected_ii_s is not None else "serial",
        service_trace=_service_trace(node),
        completion_trace=tuple(completions),
        diagnostics=node.diagnostics,
    )


def _check_arch_requirements(launch: Launch, arch: Any, view: Any) -> None:
    """Reject an architecture that lacks a capability the launch declares."""

    checks = {
        "tmem": (
            view.tmem_bandwidth is not None and (getattr(arch, "tmem_capacity_per_sm", 0) or 0) > 0,
            "Tensor Memory (tmem_bandwidth > 0 and tmem_capacity_per_sm > 0)",
        ),
        "tcgen05": (bool(getattr(arch, "support_utcmma", False)), "tcgen05/UTCMMA support (arch.support_utcmma)"),
        "wgmma": (bool(getattr(arch, "support_wgmma", False)), "WGMMA support (arch.support_wgmma)"),
        "l1_5": (view.l1_5_group_size > 0 and view.l1_5_bandwidth is not None, "an L1.5 level"),
    }
    missing = [
        "%s: %s" % (name, checks[name][1]) for name in launch.requires_arch if not checks[name][0]
    ]
    if missing:
        raise UnsupportedError(
            "%s requires %s, which %s does not provide: %s"
            % (launch.name, ", ".join(launch.requires_arch), view.arch_id, "; ".join(missing))
        )


def _unexecuted_group(key: CostGroupKey) -> CostGroup:
    return CostGroup(
        key=key, count=0, traffic_per_execution=Traffic(), service_s=tuple(),
        completion_latency_s=0.0, latency_extra_s=0.0, traffic_source="not_executed",
        provenance=(("cost_source", "not_executed_zero_count"),),
    )


def _analyze_launch(
    launch: Launch,
    arch: Any,
    options: Options,
    context: Context,
    oracle: GenericOracle,
    options_digest: str,
    context_digest: str,
) -> Tuple[LaunchResult, List[Tuple[str, RegionResult]], List[Tuple[str, OpResult]], List[CacheReport], Diagnostics]:
    unsupported = []  # type: List[str]
    approximations = []  # type: List[str]
    overridden = []  # type: List[str]
    notes = []  # type: List[str]
    view = oracle.view
    sm_count = view.sm_count
    overridden.extend(view.overrides)
    _check_arch_requirements(launch, arch, view)

    # Residency.
    residency_override = context.residency(launch.name)
    if residency_override is not None:
        residency, residency_source = residency_override, "context_override"
        overridden.append("residency[%s]=%d from Context" % (launch.name, residency_override))
    elif isinstance(launch.residency, int):
        residency, residency_source = launch.residency, "launch_explicit"
    else:
        residency, residency_source = 1, "conservative_auto_fallback"
        approximations.append(
            "%s: residency='auto' falls back to 1 resident CTA per SM (occupancy derivation not implemented)"
            % launch.name
        )
    if options.resident_sharing == "schedule" and residency > 1:
        raise UnsupportedError(
            "%s: resident_sharing='schedule' with residency %d needs the shared-resource "
            "scheduler, which this entry point does not provide; use 'equal_share'"
            % (launch.name, residency)
        )
    cluster_size = launch.cluster_size
    logical_sms = sm_count
    if cluster_size > 1:
        logical_sms = sm_count // cluster_size
        if logical_sms <= 0:
            raise UnsupportedError("%s: cluster %d is larger than the device" % (launch.name, cluster_size))
        approximations.append(
            "%s: cluster of %d CTAs is treated as one work unit on %d cluster slots; "
            "no DSM/multicast cost is modelled" % (launch.name, cluster_size, logical_sms)
        )
    slots = logical_sms * residency
    units = launch.effective_work_units()
    group_sequence = units.instance_group_ids()
    total_units = units.total_count
    if isinstance(launch.scheduler, Persistent):
        concurrent = launch.scheduler.cta_count
        if concurrent > slots:
            raise UnsupportedError(
                "%s: persistent scheduler declares %d CTAs but only %d slots are resident"
                % (launch.name, concurrent, slots)
            )
        slots = concurrent
    else:
        concurrent = min(total_units, slots)
    active_sms = min(logical_sms, max(1, math.ceil(concurrent / residency)))
    if active_sms < logical_sms:
        approximations.append(
            "%s: %d concurrent work units fill only %d of %d SM slots; device bandwidth is shared by active SMs"
            % (launch.name, concurrent, active_sms, logical_sms)
        )
    wave = WaveView(
        wave_kind="full", sm_count=sm_count, active_sms=active_sms, residency=residency,
        work_units=total_units,
    )
    tail_units = 0
    tail_wave = None  # type: Optional[WaveView]
    if (options.tail_policy == "tail_active_sms" and not isinstance(launch.scheduler, Persistent)
            and total_units > slots and total_units % slots):
        tail_active = min(logical_sms, max(1, math.ceil((total_units % slots) / residency)))
        if tail_active < active_sms:
            tail_units = total_units % slots
            tail_wave = WaveView(
                wave_kind="tail", sm_count=sm_count, active_sms=tail_active, residency=residency,
                work_units=tail_units,
            )
    if tail_wave is None:
        approximations.append(
            "%s: tail_policy=%r prices every work unit with the full-wave context%s"
            % (launch.name, options.tail_policy,
               "" if options.tail_policy == "full_wave_costs" else " (no partial final wave)")
        )
    else:
        approximations.append(
            "%s: tail_policy='tail_active_sms': the last %d work units form the final wave, start after "
            "every full wave has finished and share pooled DDR/L2 bandwidth among %d active SMs; "
            "their cache traffic is the launch-wide estimate"
            % (launch.name, tail_units, tail_wave.active_sms)
        )

    # Declared on-chip storage of one work unit against the per-SM capacity.
    capacities = {"smem": getattr(arch, "configurable_smem_capacity", None), "tmem": getattr(arch, "tmem_capacity_per_sm", None)}
    for loop in launch.loops():
        if not isinstance(loop.body, Pipeline):
            continue
        declared_hints = [dep.name[len("hint:"):] for dep in loop.body.dependencies if dep.name.startswith("hint:")]
        offsets = ["%s=%+d" % (op_id, offset) for actor in loop.body.actors for op_id, offset in actor.sequence if offset]
        if declared_hints or offsets:
            notes.append("%s: loop %s declared schedule constraints: barriers [%s]; window offsets [%s]; actor issue orders %s"
                         % (launch.name, loop.loop_id, ", ".join(declared_hints), ", ".join(offsets),
                            {actor.name: [op_id for op_id, _o in actor.sequence] for actor in loop.body.actors
                             if len(actor.sequence) > 1}))
    # One allocation per buffer name for the whole work unit (loops, sequences, prologues and epilogues alike).
    declared = launch.storage_bytes()
    unsized = launch.unsized_buffers()
    if unsized:
        notes.append("%s: buffers %s have no sized declaration (Launch.buffers / Pipeline.buffers); their storage is not "
                     "part of the capacity check" % (launch.name, list(unsized)))
    allocations, allocation_groups = launch.allocations()
    grouped_members = {member for group in allocation_groups.values() for member in group.members}
    slot_text = ["%s=%d" % (item.name, item.slots) for item in allocations.values()
                 if item.storage in ("smem", "tmem") and item.name not in grouped_members]
    slot_text += ["%s(%s: %s)=%d" % (group.name, group.kind, "+".join(group.members), group.slots)
                  for group in allocation_groups.values()]
    if slot_text:
        notes.append("%s: effective on-chip slots: %s%s" % (
            launch.name, ", ".join(slot_text),
            "".join("; loop %s pipeline depth %d" % (loop.loop_id, loop.body.stages)
                    for loop in launch.loops() if isinstance(loop.body, Pipeline) and loop.body.stages)))
    for storage, amount in sorted(declared.items()):
        capacity = capacities.get(storage)
        if not isinstance(capacity, (int, float)) or capacity <= 0:
            notes.append("%s: declares %.0f bytes of %s per work unit; the architecture gives no %s capacity to check against"
                         % (launch.name, amount, storage, storage))
            continue
        per_unit = float(capacity) / residency
        text = ("%s: one work unit declares %.0f bytes (%.1f KiB) of %s but an SM provides %.0f bytes (%.1f KiB)%s; "
                "alias buffers that share a slot (KernelBuilder.alias) or share a ring (KernelBuilder.ring)"
                % (launch.name, amount, amount / 1024.0, storage, per_unit, per_unit / 1024.0,
                   "" if residency == 1 else " per resident unit"))
        if amount > per_unit + 1e-9:
            if options.capacity_policy == "error":
                raise UnsupportedError(text)
            unsupported.append(text)
        else:
            notes.append("%s: declared %s %.1f KiB of %.1f KiB per work unit" % (launch.name, storage, amount / 1024.0, per_unit / 1024.0))

    # Cache/traffic over the real instance sequence (once per launch).
    cache_view = build_cache_view(launch, arch, options, context, slots, residency, sm_count)
    unsupported.extend(cache_view.unsupported)
    approximations.extend(cache_view.approximations)
    cache_reports = [cache_view.report] if cache_view.report is not None else []

    def _unsupported(error: Exception) -> UnsupportedError:
        message = str(error)
        if cache_view.unsupported:
            message += " | cache binding: " + "; ".join(cache_view.unsupported)
        return UnsupportedError(message)

    # Per-group evaluation.
    cost_by_group = {}  # type: Dict[str, float]
    lowered_by_group = {}  # type: Dict[str, LoweredLaunch]
    bound_by_group = {}  # type: Dict[str, BoundOracle]
    evaluation_by_group = {}  # type: Dict[str, RegionEvaluation]
    binding_by_group = {}  # type: Dict[str, BindingContext]
    counts = units.group_counts()
    active_groups = [(group_id, group) for group_id, group in units.groups if counts[group_id] > 0]
    for group_id, _group in units.groups:
        if counts[group_id] == 0:
            approximations.append(
                "%s: work group %r has no instances in work_units.order; it is not evaluated" % (launch.name, group_id)
            )
    tail_counts = {}  # type: Dict[str, int]
    for group_id in (group_sequence[total_units - tail_units:] if tail_units else ()):
        tail_counts[group_id] = tail_counts.get(group_id, 0) + 1
    tail_cost_by_group = {}  # type: Dict[str, float]
    tail_bound_by_group = {}  # type: Dict[str, BoundOracle]
    tail_binding_by_group = {}  # type: Dict[str, BindingContext]
    passes = [(group_id, group, wave) for group_id, group in active_groups]
    passes += [(group_id, group, tail_wave) for group_id, group in active_groups if group_id in tail_counts]
    for group_id, group, group_wave in passes:
        is_tail = group_wave is tail_wave and tail_wave is not None
        lowered = lower_launch(launch, group_id, group, options)
        binding = BindingContext(
            launch=launch.name, group_id=group_id, trip_counts=lowered.trip_counts,
            effective_fractions=group.effective_fractions, wave=group_wave,
            cache_scenario=cache_view.scenario, arch_id="%s#%s" % (view.arch_id, view.arch_digest[:16]),
            options_digest=options_digest, context_digest=context_digest,
        )
        bound = oracle.bind(lowered.sites, binding, cache_view.scoped(group_id))
        native_options = NativeModelOptions(
            ii_mode=options.ii_mode,
            unroll_threshold=options.unroll_threshold,
            max_inline_iterations=max(options.unroll_threshold, 64),
            max_finite_iterations=options.max_finite_iterations,
            boundary_policy=options.boundary_policy,
            launch_name=launch.name,
            kernel_launch_s=options.kernel_launch_s,
            host_dispatch_s=options.host_dispatch_s,
            periodic_topology_trials=options.periodic_topology_trials,
            periodic_topology_seed=options.periodic_topology_seed,
        )
        try:
            evaluation = evaluate_region(lowered.kernel, lowered.region, native_options, oracle=bound)
        except UnsupportedCostError as error:
            raise _unsupported(error) from error
        except (NativeModelError, ModelingError, PeriodicScheduleError) as error:
            raise UnsupportedError("%s/%s: %s" % (launch.name, group_id, error)) from error
        trip_counts = dict(lowered.trip_counts)
        for op_id, site in lowered.sites.items():
            if _multiplicity(site, trip_counts) > 0:
                try:
                    bound.cost(op_id)
                except UnsupportedCostError as error:
                    raise _unsupported(error) from error
        if is_tail:
            tail_cost_by_group[group_id] = evaluation.total_s
            tail_bound_by_group[group_id] = bound
            tail_binding_by_group[group_id] = binding
            continue
        cost_by_group[group_id] = evaluation.total_s
        lowered_by_group[group_id] = lowered
        bound_by_group[group_id] = bound
        evaluation_by_group[group_id] = evaluation
        binding_by_group[group_id] = binding

    # Spatial dispatch.
    if isinstance(launch.scheduler, Persistent):
        spatial = persistent_schedule(group_sequence, cost_by_group, launch.scheduler, slots)
    else:
        spatial = list_schedule(group_sequence, cost_by_group, slots, tail_units, tail_cost_by_group or None)
    notes.append("%s: dispatch order assumed '%s'" % (launch.name, launch.dispatch_order))
    physical_ctas = 1
    for extent in launch.physical_grid:
        physical_ctas *= extent
    body_s = spatial.kernel_body_s
    total_s = body_s + options.kernel_launch_s + options.host_dispatch_s

    # Op results with cost groups.
    primary_group = max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
    op_results = []  # type: List[Tuple[str, OpResult]]
    op_group_traffic = {}  # type: Dict[Tuple[str, str], Traffic]
    op_loops = {}  # type: Dict[str, Tuple[str, ...]]
    for site in launch.op_sites():
        op_loops[site.op.op_id] = site.loop_ids
        op = site.op
        groups = []
        total_count = 0
        traffic_total = Traffic()
        useful_flops = executed_flops = useful_bytes = executed_bytes = 0.0
        access_ids = []
        for group_id, group in units.groups:
            if counts[group_id] == 0:
                groups.append(_unexecuted_group(CostGroupKey(
                    work_group_id=group_id, region_site=site.region_site, wave_kind="not_applicable",
                    binding_context_id="not_evaluated",
                )))
                continue
            lowered = lowered_by_group[group_id]
            trip_counts = dict(lowered.trip_counts)
            multiplicity = _multiplicity(site, trip_counts)
            count = multiplicity * counts[group_id]
            key = CostGroupKey(
                work_group_id=group_id, region_site=site.region_site, wave_kind="full",
                binding_context_id=binding_by_group[group_id].binding_context_id,
            )
            if count == 0:
                groups.append(_unexecuted_group(key))
                continue
            group_traffic = Traffic()
            tail_count = multiplicity * tail_counts.get(group_id, 0)
            for wave_kind, wave_count, wave_bound, wave_binding in (
                ("full", count - tail_count, bound_by_group[group_id], binding_by_group[group_id]),
                ("tail", tail_count, tail_bound_by_group.get(group_id), tail_binding_by_group.get(group_id)),
            ):
                if wave_count <= 0 or wave_bound is None:
                    continue
                cost = wave_bound.cost(op.op_id)
                groups.append(CostGroup(
                    key=CostGroupKey(
                        work_group_id=group_id, region_site=site.region_site, wave_kind=wave_kind,
                        binding_context_id=wave_binding.binding_context_id,
                    ),
                    count=wave_count, traffic_per_execution=cost.traffic_per_execution,
                    service_s=cost.service_s, completion_latency_s=cost.completion_latency_s,
                    latency_extra_s=cost.latency_extra_s, traffic_source=cost.traffic_source,
                    provenance=cost.provenance,
                ))
                group_traffic = group_traffic.add(scale_traffic(cost.traffic_per_execution, float(wave_count)))
            total_count += count
            op_group_traffic[(op.op_id, group_id)] = group_traffic
            traffic_total = traffic_total.add(group_traffic)
            if isinstance(op, Compute):
                fraction = group.effective_fraction(op.op_id)
                if fraction is None:
                    fraction = op.effective_fraction
                executed_flops += op.executed_flops * count
                useful_flops += op.executed_flops * fraction * count
            elif isinstance(op, (Load, Store)):
                fraction = group.effective_fraction(op.op_id)
                if fraction is None:
                    fraction = 1.0
                executed_bytes += move_executed_bytes(op) * count
                useful_bytes += move_payload_bytes(op) * fraction * count
        if isinstance(op, (Load, Store)) and op.access is not None:
            access_ids.append(access_id_for(site))
        op_results.append((site.path, OpResult(
            path=site.path, op_id=op.op_id, kind=op.kind, launch=launch.name,
            region_site=site.region_site, count=total_count,
            useful_flops=useful_flops, executed_flops=executed_flops,
            useful_bytes=useful_bytes, executed_bytes=executed_bytes,
            traffic_total=traffic_total, cost_groups=tuple(groups),
            access_ids=tuple(access_ids),
            traffic_interval_total=TrafficInterval(low=traffic_total, high=traffic_total),
            provenance=(
                ("count_rule", "sum over work groups of group_count x enclosing body trip counts"),
                ("traffic_rule", "traffic_total = sum(count_g x traffic_per_execution_g)"),
                ("cache_path", cache_view.path),
            ),
        )))

    def _traffic_under(loop_id: Optional[str], group_id: Optional[str]) -> Traffic:
        total = Traffic()
        for (op_id, gid), traffic in op_group_traffic.items():
            if group_id is not None and gid != group_id:
                continue
            if loop_id is not None and loop_id not in op_loops[op_id]:
                continue
            total = total.add(traffic)
        return total

    launch_traffic = _traffic_under(None, None)

    # Region results.
    region_results = []  # type: List[Tuple[str, RegionResult]]
    launch_groups = []
    for group_id, _group in active_groups:
        launch_groups.append(_region_group(
            evaluation_by_group[group_id], launch.name, group_id, bound_by_group[group_id],
            lowered_by_group[group_id].sites, options, _traffic_under(None, group_id),
        ))
    summary = evaluation_by_group[primary_group]
    top_children = tuple(
        lowered_by_group[primary_group].loop_paths[loop.loop_id]
        for loop in launch.loops()
        if "/" not in lowered_by_group[primary_group].loop_paths[loop.loop_id][len(launch.name) + 1:]
    )
    region_results.append((launch.name, RegionResult(
        path=launch.name, kind="launch_body", launch=launch.name, summary_group_id=primary_group,
        total_s=summary.total_s, first_s=summary.first_s, steady_s=summary.steady_s,
        drain_s=summary.drain_s, trip_count=None, ii=None,
        estimate_scope=summary.estimate_scope, boundary_policy=options.boundary_policy,
        groups=tuple(launch_groups), children=top_children,
        diagnostics=("summary fields use work group %r (largest count); see groups" % primary_group,),
        traffic_total=launch_traffic,
    )))
    for loop in launch.loops():
        path = lowered_by_group[primary_group].loop_paths[loop.loop_id]
        groups = []
        for group_id, _group in active_groups:
            node = _find_evaluation(evaluation_by_group[group_id], loop.loop_id)
            if node is None:
                continue
            groups.append(_region_group(
                node, path, group_id, bound_by_group[group_id], lowered_by_group[group_id].sites, options,
                _traffic_under(loop.loop_id, group_id), ii_mode=_effective_ii_mode(loop, options),
            ))
        node = _find_evaluation(evaluation_by_group[primary_group], loop.loop_id)
        if node is None:
            continue
        prefix = path + "/"
        children = tuple(
            other_path for other_id, other_path in lowered_by_group[primary_group].loop_paths.items()
            if other_path.startswith(prefix) and "/" not in other_path[len(prefix):]
        )
        region_results.append((path, RegionResult(
            path=path, kind="loop", launch=launch.name, summary_group_id=primary_group,
            total_s=node.total_s, first_s=node.first_s, steady_s=node.steady_s, drain_s=node.drain_s,
            trip_count=node.trip_count, ii=_ii_components(node, _effective_ii_mode(loop, options)),
            estimate_scope=node.estimate_scope,
            boundary_policy=options.boundary_policy if loop.is_periodic else "serial",
            groups=tuple(groups), children=children,
            diagnostics=node.diagnostics + (
                "summary fields use work group %r; per-group detail in groups" % primary_group,
            ),
            traffic_total=_traffic_under(loop.loop_id, None),
        )))

    # Per-op shares within the innermost region and the launch (program scope is added later).
    loop_paths = lowered_by_group[primary_group].loop_paths
    shared_ops = []  # type: List[Tuple[str, OpResult]]
    for path, op in op_results:
        shares = list(traffic_shares(op.traffic_total, "launch", launch.name, launch_traffic))
        loops = op_loops[op.op_id]
        if loops:
            inner = loops[-1]
            shares.extend(traffic_shares(op.traffic_total, "region", loop_paths[inner], _traffic_under(inner, None)))
        shared_ops.append((path, replace(op, traffic_shares=tuple(shares))))
    op_results = shared_ops

    launch_result = LaunchResult(
        name=launch.name,
        kernel_body_s=body_s,
        launch_overhead_s=options.kernel_launch_s,
        host_dispatch_s=options.host_dispatch_s,
        total_s=total_s,
        work_units=total_units,
        physical_ctas=physical_ctas,
        sm_count=sm_count,
        active_sms=active_sms,
        residency=residency,
        residency_source=residency_source,
        slots=slots,
        waves=spatial.waves,
        tail_work_units=spatial.tail_work_units,
        spatial_scope=spatial.scope,
        scheduler=spatial.scheduler,
        groups=tuple(
            WorkGroupSummary(
                group_id=group_id, count=counts[group_id], per_unit_s=cost_by_group[group_id],
                trip_counts=lowered_by_group[group_id].trip_counts,
            )
            for group_id, _group in active_groups
        ),
        slot_completion_s=spatial.slot_completion_s,
        critical_work_unit=spatial.critical_unit,
        critical_group_id=spatial.critical_group,
        traffic_total=launch_traffic,
        provenance=(
            ("wave_context", wave.canonical()),
            ("cluster_size", cluster_size),
            ("logical_slots", slots),
            ("concurrent_work_units", concurrent),
            ("dispatch_order", launch.dispatch_order),
            ("tail_policy", options.tail_policy),
            ("tail_wave_context", tail_wave.canonical() if tail_wave is not None else None),
            ("tail_per_unit_s", tuple(sorted(tail_cost_by_group.items()))),
            ("resident_sharing", options.resident_sharing),
            ("cache_path", cache_view.path),
            ("requires_arch", launch.requires_arch),
            ("unit_timeline", spatial.unit_timeline),
            ("unit_timeline_complete", spatial.timeline_complete),
            ("launch_accounting", "exactly_once"),
        ),
    )
    diagnostics = Diagnostics(
        unsupported=tuple(unsupported), approximations=tuple(approximations),
        overridden_parameters=tuple(overridden), notes=tuple(notes),
    )
    return launch_result, region_results, op_results, cache_reports, diagnostics


def _analyze_once(
    program: Program, arch: Any, options: Options, context: Context, scenario: str,
) -> AnalysisResult:
    if not isinstance(program, Program):
        raise UnsupportedError("analyze expects a Program")
    oracle = GenericOracle(arch, options, context)
    options_digest = _digest(options.canonical())
    context_digest = _digest(context.canonical())
    input_digest = _digest({
        "program": program_to_snapshot(program), "options": options.canonical(),
        "context": context.canonical(), "arch": oracle.view.arch_id,
        "arch_digest": oracle.view.arch_digest,
    })
    launches = []
    regions = []
    ops = []
    cache = []
    unsupported = []  # type: List[str]
    approximations = []  # type: List[str]
    overridden = []  # type: List[str]
    notes = []  # type: List[str]
    for launch in program.launches:
        result, launch_regions, launch_ops, reports, diagnostics = _analyze_launch(
            launch, arch, options, context, oracle, options_digest, context_digest,
        )
        launches.append((launch.name, result))
        regions.extend(launch_regions)
        ops.extend(launch_ops)
        cache.extend(reports)
        unsupported.extend(diagnostics.unsupported)
        approximations.extend(diagnostics.approximations)
        overridden.extend(diagnostics.overridden_parameters)
        notes.extend(diagnostics.notes)
    gap_total = sum(edge.gap_s for edge in program.edges)
    total_s = sum(item.total_s for _name, item in launches) + gap_total
    program_traffic = Traffic()
    for _path, op in ops:
        program_traffic = program_traffic.add(op.traffic_total)
    ops = [
        (path, replace(op, traffic_shares=op.traffic_shares
                       + traffic_shares(op.traffic_total, "program", program.name, program_traffic)))
        for path, op in ops
    ]
    if len(program.launches) > 1:
        notes.append("launches are priced sequentially; no inter-launch overlap (PDL) is modelled")
    program_result = ProgramResult(
        name=program.name, total_s=total_s,
        launch_order=tuple(launch.name for launch in program.launches),
        inter_launch_gap_s=gap_total,
        scope="full_program_sequential_launches",
        aggregation="sum(launch.total_s) + sum(edge.gap_s); launch.total_s = body + launch_overhead + host_dispatch",
        measurement_scope="sum_kernel_elapsed_plus_launch_overhead",
        traffic_total=program_traffic,
        edges=tuple((edge.source, edge.target, edge.gap_s) for edge in program.edges),
    )
    workload_digests = [("program:" + program.name, program_workload_digest(program))]
    workload_digests.extend((launch.name, launch_workload_digest(launch)) for launch in program.launches)
    return AnalysisResult(
        program=program_result,
        launches=Index(launches), regions=Index(regions), ops=Index(ops), cache=tuple(cache),
        scenarios=Index(()), selected_scenario=scenario,
        diagnostics=Diagnostics(
            unsupported=tuple(dict.fromkeys(unsupported)),
            approximations=tuple(dict.fromkeys(approximations)),
            overridden_parameters=tuple(dict.fromkeys(overridden)),
            notes=tuple(dict.fromkeys(notes)),
            provenance=(
                ("model_version", MODEL_VERSION),
                ("arch", oracle.view.arch_id),
                ("arch_digest", oracle.view.arch_digest),
                ("options", options.canonical()),
                ("context", context.canonical()),
                ("input_digest", input_digest),
                ("units", "seconds, Hz, bytes"),
                ("latency_extra_default", "0 unless op.latency_extra_s or Context.latency_table"),
                ("cache_bound_policy", options.cache_bound_policy),
                ("cache_scenario", scenario),
            ),
        ),
        input_digest=input_digest,
        workload_digests=tuple(workload_digests),
    )


def analyze(
    program: Program,
    arch: Any,
    *,
    options: Optional[Options] = None,
    context: Optional[Context] = None,
) -> AnalysisResult:
    """Analyze a typed program: cache/traffic, per-op costs, regions, launches."""

    options = options or Options()
    context = context or Context()
    if not isinstance(options, Options):
        raise UnsupportedError("options must be Options")
    if not isinstance(context, Context):
        raise UnsupportedError("context must be Context")
    main = _analyze_once(program, arch, options, context, "conservative")
    if options.cache_bound_policy == "scenarios":
        # Interval widths are zero in this version (no truncated sampling yet), so
        # the low/high scenarios share one computation and are labelled as such.
        low = _analyze_once(program, arch, options, context, "low")
        high = low
        notes = main.diagnostics.notes + (
            "cache scenarios low/high share one computation: interval width is zero "
            "(no truncation-bound sampling in this version)",
        )
        diagnostics = Diagnostics(
            unsupported=main.diagnostics.unsupported,
            approximations=main.diagnostics.approximations,
            overridden_parameters=main.diagnostics.overridden_parameters,
            notes=notes, provenance=main.diagnostics.provenance,
        )
        return AnalysisResult(
            program=main.program, launches=main.launches, regions=main.regions, ops=main.ops,
            cache=main.cache, scenarios=Index((("low", low), ("high", high))),
            selected_scenario="conservative", diagnostics=diagnostics,
            input_digest=main.input_digest, workload_digests=main.workload_digests,
        )
    return main


__all__ = ["UnsupportedError", "analyze"]
