"""Bind ``TileAccess`` declarations to the reuse-distance cache model.

One ``CacheView`` is built per launch over the real work-instance sequence
(``WorkUnits.order``).  Two evaluation paths exist:

* ``reduction_axis`` (``cache='fast'`` when expressible): the existing
  stable-shadow backend over the work-axis tile grid.  It requires one
  homogeneous access loop (same trip count in every group) whose in-loop
  accesses all vary with the loop, and no other loop-nested access.
* ``explicit_trace`` (``cache='trace'``, and ``fast`` fallback): the exact
  distinct-reuse trace over every instance with its own trip counts and the
  declared program order (see ``explicit_trace.py``).  Loop-invariant
  accesses, ragged/causal groups, and persistent streams are honoured.

Explicit ``Context.traffic_assumptions`` override the *priced* traffic of an
access but never remove it from cache history.  Anything that cannot be
modelled is listed in ``CacheView.unsupported``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..cache.api import model_cache
from ..cache.ir import (
    CacheAccessIR, CacheLevelConfig, CacheProblem, ExplicitTraversal, PanelTraversal, Projection as CacheProjection,
    ReductionConfig, RowMajorTraversal, SamplingConfig, TensorTileRegion, TileGrid,
    WritePolicy,
)
from .traffic import Traffic
from ..errors import ModelingError, ModelingValidationError
from .contract import Launch, Load, OpSite, Persistent, Store, WorkUnits, move_executed_bytes, move_payload_bytes, touches_global
from .explicit_trace import ExplicitTraceError, run_explicit_trace
from .options import Context, Options
from .results import CacheAccessReport, CacheReport, TrafficInterval

DEFAULT_REUSE_UNIT_BYTES = 8192.0


class CacheBindingError(ModelingError):
    pass


@dataclass(frozen=True)
class AccessTraffic:
    access_id: str
    op_id: str
    op_path: str
    tensor: str
    mode: str
    per_execution: Traffic
    interval: TrafficInterval
    request_count: float
    l1_5_hit_rate: Optional[float]
    l2_served_rate: Optional[float]
    ddr_miss_rate: Optional[float]
    source: str
    scope: str
    rates_source: str = "cache_model"


@dataclass(frozen=True)
class CacheView:
    """Per-launch traffic: ``accesses`` per op (launch average) and per group."""

    launch: str
    scenario: str
    mode: str
    path: str
    accesses: Tuple[Tuple[str, AccessTraffic], ...]
    group_accesses: Tuple[Tuple[Tuple[str, str], AccessTraffic], ...]
    report: Optional[CacheReport]
    unsupported: Tuple[str, ...] = field(default_factory=tuple)
    approximations: Tuple[str, ...] = field(default_factory=tuple)
    problem_digest: Optional[str] = None

    def traffic_for(self, op_id: str, group_id: Optional[str] = None) -> Optional[AccessTraffic]:
        if group_id is not None:
            found = dict(self.group_accesses).get((op_id, group_id))
            if found is not None:
                return found
        return dict(self.accesses).get(op_id)

    def scoped(self, group_id: str) -> "GroupCacheView":
        return GroupCacheView(self, group_id)


@dataclass(frozen=True)
class GroupCacheView:
    view: CacheView
    group_id: str

    @property
    def scenario(self) -> str:
        return self.view.scenario

    def traffic_for(self, op_id: str) -> Optional[AccessTraffic]:
        return self.view.traffic_for(op_id, self.group_id)


def access_id_for(site: OpSite) -> str:
    access = getattr(site.op, "access", None)
    if access is not None and access.access_id is not None:
        return access.access_id
    return site.op.op_id


def _assumed_miss(site: OpSite) -> Traffic:
    op = site.op
    payload = move_payload_bytes(op)
    executed = move_executed_bytes(op)
    if isinstance(op, Load):
        return Traffic(payload_read_bytes=payload, l2_read_request_bytes=executed, ddr_read_bytes=executed)
    return Traffic(payload_write_bytes=payload, l2_write_request_bytes=executed, ddr_write_bytes=executed)


def on_chip_traffic(op: Any) -> Traffic:
    """Endpoint bytes of one move execution (source read, destination written)."""

    return Traffic.on_chip_move(op.source, op.destination, move_executed_bytes(op))


def _exact_interval(traffic: Traffic, approximation: Optional[str]) -> TrafficInterval:
    return TrafficInterval(low=traffic, high=traffic, structural_approximation=approximation)


def _entry(site: OpSite, traffic: Traffic, source: str, scope: str, structural: Optional[str],
           requests: float = 0.0, l1_5: Optional[float] = None, l2: Optional[float] = None,
           ddr: Optional[float] = None, rates_source: str = "cache_model") -> AccessTraffic:
    op = site.op
    traffic = traffic.with_on_chip_defaults(on_chip_traffic(op))
    return AccessTraffic(
        rates_source=rates_source,
        access_id=access_id_for(site), op_id=op.op_id, op_path=site.path,
        tensor=op.access.tensor if op.access is not None else "?",
        mode="read" if isinstance(op, Load) else "write",
        per_execution=traffic, interval=_exact_interval(traffic, structural),
        request_count=requests, l1_5_hit_rate=l1_5, l2_served_rate=l2, ddr_miss_rate=ddr,
        source=source, scope=scope,
    )


def _level_configs(arch: Any, unit_bytes: float) -> Tuple[CacheLevelConfig, Optional[CacheLevelConfig], int]:
    l2_capacity = getattr(arch, "l2_capacity", None)
    if not isinstance(l2_capacity, (int, float)) or l2_capacity <= 0:
        raise CacheBindingError("arch lacks l2_capacity")
    try:
        l2 = CacheLevelConfig(
            "l2", float(l2_capacity), unit_bytes,
            int(getattr(arch, "l2_associativity", 8) or 8), int(getattr(arch, "l2_cacheline_bytes", 128) or 128),
        )
    except ModelingValidationError as error:
        raise CacheBindingError("L2 config: %s" % error) from error
    group_size = int(getattr(arch, "l1_5_group_size", 0) or 0)
    l1_5 = None
    if group_size > 0:
        try:
            l1_5 = CacheLevelConfig(
                "l1_5", float(getattr(arch, "l1_5_capacity_per_group")), unit_bytes,
                int(getattr(arch, "l1_5_associativity", 8) or 8),
                int(getattr(arch, "l1_5_cacheline_bytes", 128) or 128),
            )
        except (ModelingValidationError, AttributeError, TypeError) as error:
            raise CacheBindingError("L1.5 config: %s" % error) from error
    return l2, l1_5, group_size


def executed_sites(launch: Launch, units: WorkUnits, sites: List[OpSite]) -> Tuple[List[OpSite], List[OpSite]]:
    """Split sites into those executed at least once and those never executed."""

    loops = {loop.loop_id: loop for loop in launch.loops()}
    executed = []
    skipped = []
    counts = units.group_counts()
    for site in sites:
        runs = 0
        for group_id, group in units.groups:
            multiplicity = counts[group_id]
            for loop_id, where in site.chain:
                if where == "body":
                    multiplicity *= launch.bound_trip_count(loops[loop_id], group)
            runs += multiplicity
        (executed if runs > 0 else skipped).append(site)
    return executed, skipped


def _reduction_axis_expressible(launch: Launch, units: WorkUnits, sites: List[OpSite]) -> Tuple[bool, str]:
    """Can the homogeneous reduction-axis backend represent this launch?"""

    if isinstance(launch.scheduler, Persistent):
        return False, "persistent scheduler needs the explicit instance trace"
    if len(units.groups) > 1:
        return False, "%d work groups need per-group attribution from the explicit instance trace" % len(units.groups)
    loops = {loop.loop_id: loop for loop in launch.loops()}
    access_loops = set()
    for site in sites:
        body_loops = [loop_id for loop_id, where in site.chain if where == "body"]
        if len(body_loops) > 1:
            return False, "%s sits inside nested loop bodies %s" % (site.path, body_loops)
        for loop_id in body_loops:
            access_loops.add(loop_id)
            if loop_id not in site.op.access.index_map.work_axes:
                return False, "%s is loop-invariant inside loop %s" % (site.path, loop_id)
        for axis in site.op.access.index_map.work_axes:
            if axis in loops and axis not in body_loops:
                return False, "%s projects on loop %s from outside its body" % (site.path, axis)
    if not access_loops:
        return True, "no loop accesses"
    if len(access_loops) > 1:
        loop_trips = {}
        for group_id, group in units.groups:
            for loop_id in access_loops:
                loop_trips.setdefault(loop_id, set()).add(launch.bound_trip_count(loops[loop_id], group))
        if len({tuple(sorted(v)) for v in loop_trips.values()}) != 1:
            return False, "access loops %s have different trip counts" % sorted(access_loops)
    trips = set()
    for group_id, group in units.groups:
        for loop_id in access_loops:
            trips.add(launch.bound_trip_count(loops[loop_id], group))
    if len(trips) != 1:
        return False, "work groups bind different trip counts %s to the access loop" % sorted(trips)
    if 0 in trips:
        return False, "access loop has zero trips; only prologue/epilogue accesses execute"
    return True, "homogeneous single access loop"


def build_cache_view(
    launch: Launch,
    arch: Any,
    options: Options,
    context: Context,
    slots: int,
    residency: int,
    sm_count: int,
) -> CacheView:
    units = launch.effective_work_units()
    loops_by_id = {loop.loop_id: loop for loop in launch.loops()}
    moves = [site for site in launch.op_sites() if isinstance(site.op, (Load, Store))]
    all_sites = [site for site in moves if touches_global(site.op)]
    sites, skipped = executed_sites(launch, units, all_sites)
    modelled = [site for site in sites if site.op.access is not None]
    accesses = {}  # type: Dict[str, AccessTraffic]
    group_accesses = {}  # type: Dict[Tuple[str, str], AccessTraffic]
    unsupported = []  # type: List[str]
    approximations = []  # type: List[str]
    report = None
    problem_digest = None
    path = "none"
    for site in skipped:
        approximations.append("%s: never executed (zero trip count); no traffic modelled" % site.path)
    # On-chip moves: endpoint bytes only.
    for site in moves:
        if touches_global(site.op):
            continue
        accesses[site.op.op_id] = _entry(site, Traffic(), "on_chip_endpoints", "per_execution_endpoint_bytes", None,
                                         rates_source="not_applicable_on_chip")
        for group_id, _group in units.groups:
            group_accesses[(site.op.op_id, group_id)] = accesses[site.op.op_id]

    if options.cache != "none" and modelled:
        expressible, reason = _reduction_axis_expressible(launch, units, modelled)
        use_reduction = options.cache == "fast" and expressible
        path = "reduction_axis" if use_reduction else "explicit_trace"
        if options.cache == "fast" and not expressible:
            approximations.append(
                "%s: cache='fast' fell back to the explicit instance trace (%s)" % (launch.name, reason)
            )
        try:
            if use_reduction:
                report, problem_digest, notes = _reduction_axis(
                    launch, units, modelled, arch, options, slots, sm_count, accesses, group_accesses,
                )
            else:
                report, notes = _explicit(
                    launch, units, modelled, arch, options, slots, residency, accesses, group_accesses,
                )
            approximations.extend(notes)
        except (CacheBindingError, ExplicitTraceError) as error:
            unsupported.append("cache[%s]: %s" % (launch.name, error))
    elif options.cache == "none" and modelled:
        approximations.append("cache='none': TileAccess declarations were not evaluated for %s" % launch.name)

    # Explicit assumptions override the priced traffic of any move (global or
    # on-chip); cache history is kept and the model prediction is retained.
    model_predictions = dict(accesses)  # type: Dict[str, AccessTraffic]
    overridden = {}  # type: Dict[str, str]
    for site in moves:
        found = context.traffic_assumption(access_id_for(site), site.path)
        if found is None:
            continue
        key, assumption = found
        source = "explicit_assumption:%s:%s" % (key, assumption.source)
        if site.op.op_id in accesses:
            approximations.append(
                "%s: priced traffic overridden by %s; the access stays in cache history" % (site.path, key)
            )
        overridden[site.op.op_id] = key
        if assumption.hit_rate_l2 is not None:
            rates = dict(l1_5=0.0, l2=assumption.hit_rate_l2, ddr=1.0 - assumption.hit_rate_l2,
                         rates_source="explicit_assumption:hit_rate_l2 (no L1.5 term declared)")
        elif not touches_global(site.op):
            rates = dict(rates_source="not_applicable_on_chip")
        else:
            rates = dict(rates_source="unknown: assumption declares traffic without hit rates")
        accesses[site.op.op_id] = _entry(site, assumption.traffic, source, "caller_declared_per_execution", None, **rates)
        for group_id, _group in units.groups:
            group_accesses[(site.op.op_id, group_id)] = accesses[site.op.op_id]

    for site in sites:
        if site.op.op_id in accesses:
            continue
        if options.missing_traffic_policy == "assume_miss":
            traffic = _assumed_miss(site)
            accesses[site.op.op_id] = _entry(
                site, traffic, "assumed_all_miss_no_cache_model", "policy_assume_miss", "assumed_all_miss",
                l1_5=0.0, l2=0.0, ddr=1.0, rates_source="policy_assume_miss",
            )
            approximations.append("%s: traffic assumed all-miss (no cache model result)" % site.path)
        else:
            unsupported.append("%s: no traffic available (no cache result, no explicit assumption)" % site.path)
    # Tail tiles: work-group effective fractions scale the logical payload of
    # each move here, so cost groups, reports and totals all see one binding.
    counts = units.group_counts()
    for site in moves:
        op_id = site.op.op_id
        if op_id not in accesses:
            continue
        executions = {}  # type: Dict[str, int]
        fractions = {}  # type: Dict[str, float]
        for group_id, group in units.groups:
            multiplicity = counts[group_id]
            for loop_id, where in site.chain:
                if where == "body":
                    multiplicity *= launch.bound_trip_count(loops_by_id[loop_id], group)
            executions[group_id] = multiplicity
            fraction = group.effective_fraction(op_id)
            fractions[group_id] = 1.0 if fraction is None else fraction
        if all(f >= 1.0 for f in fractions.values()):
            continue
        for group_id in fractions:
            entry = group_accesses.get((op_id, group_id), accesses[op_id])
            if fractions[group_id] < 1.0:
                group_accesses[(op_id, group_id)] = _with_payload_fraction(entry, fractions[group_id])
            else:
                group_accesses[(op_id, group_id)] = entry
        total = sum(executions.values())
        if total > 0:
            average = sum(executions[g] * fractions[g] for g in fractions) / float(total)
            accesses[op_id] = _with_payload_fraction(accesses[op_id], average)
            approximations.append(
                "%s: tail tiles scale the logical payload (average useful fraction %.4f over %d executions)"
                % (site.path, average, total)
            )
    if report is not None:
        report = _final_report(report, accesses, model_predictions, overridden)
    return CacheView(
        launch=launch.name, scenario="conservative", mode=options.cache, path=path,
        accesses=tuple(sorted(accesses.items())), group_accesses=tuple(sorted(group_accesses.items())),
        report=report, unsupported=tuple(unsupported), approximations=tuple(approximations),
        problem_digest=problem_digest,
    )


def _with_payload_fraction(entry: AccessTraffic, fraction: float) -> AccessTraffic:
    """Scale only the logical payload of a tail tile; executed bytes are unchanged."""

    traffic = entry.per_execution
    values = {name: getattr(traffic, name) for name in traffic.__dataclass_fields__ if name != "unknown"}
    payload = traffic.payload_read_bytes + traffic.payload_write_bytes
    values["payload_read_bytes"] *= fraction
    values["payload_write_bytes"] *= fraction
    values["transaction_amplification_bytes"] = traffic.transaction_amplification_bytes + payload * (1.0 - fraction)
    scaled = Traffic(unknown=traffic.unknown, **values)
    return AccessTraffic(
        access_id=entry.access_id, op_id=entry.op_id, op_path=entry.op_path, tensor=entry.tensor, mode=entry.mode,
        per_execution=scaled, interval=_exact_interval(scaled, entry.interval.structural_approximation),
        request_count=entry.request_count, l1_5_hit_rate=entry.l1_5_hit_rate, l2_served_rate=entry.l2_served_rate,
        ddr_miss_rate=entry.ddr_miss_rate, source=entry.source, scope=entry.scope, rates_source=entry.rates_source,
    )


def _final_report(
    report: CacheReport, accesses: Dict[str, AccessTraffic],
    predictions: Dict[str, AccessTraffic], overridden: Dict[str, str],
) -> CacheReport:
    """Re-derive per-access/per-tensor rows from the finally bound traffic.

    The uncalibrated cache prediction is retained per access when a caller
    assumption replaced it, so reports and pricing never disagree silently.
    """

    rows = []
    per_tensor = {}  # type: Dict[str, Dict[str, float]]
    for row in report.per_access:
        op_id = row.op_path.split("/")[-1]
        final = accesses.get(op_id)
        if final is None:
            rows.append(row)
            continue
        prediction = predictions.get(op_id)
        calibrated = op_id in overridden
        rows.append(CacheAccessReport(
            access_id=row.access_id, op_path=row.op_path, tensor=row.tensor, mode=row.mode,
            request_count=row.request_count,
            l1_5_hit_rate=final.l1_5_hit_rate, l2_served_rate=final.l2_served_rate,
            ddr_miss_rate=final.ddr_miss_rate,
            traffic_per_request=final.per_execution, interval_per_request=final.interval,
            scope=final.scope, source=final.source, calibrated=calibrated,
            model_traffic_per_request=(None if prediction is None or not calibrated else prediction.per_execution),
            model_rates=(None if prediction is None or not calibrated else (
                ("l1_5_hit_rate", prediction.l1_5_hit_rate), ("l2_served_rate", prediction.l2_served_rate),
                ("ddr_miss_rate", prediction.ddr_miss_rate))),
            rates_source=final.rates_source,
        ))
        latest = rows[-1]
        _tensor_bucket(per_tensor, row.tensor, row.mode, row.request_count,
                       None if latest.l1_5_hit_rate is None else row.request_count * latest.l1_5_hit_rate,
                       None if latest.l2_served_rate is None else row.request_count * latest.l2_served_rate,
                       None if latest.ddr_miss_rate is None else row.request_count * latest.ddr_miss_rate,
                       final.per_execution.scale(row.request_count))
    return CacheReport(
        launch=report.launch, mode=report.mode, scenario=report.scenario, backend=report.backend,
        exact=report.exact, reduction_fidelity=report.reduction_fidelity,
        histogram_digest=report.histogram_digest, problem_digest=report.problem_digest,
        per_access=tuple(rows), per_tensor=tuple(sorted(per_tensor.items())),
        assumptions=report.assumptions + (
            ("per_access/per_tensor rows use the finally priced traffic; overridden accesses keep "
             "model_traffic_per_request",) if overridden else ()
        ),
        sampled_requests=report.sampled_requests, represented_requests=report.represented_requests,
    )


def _tensor_bucket(per_tensor: Dict[str, Dict[str, Any]], tensor: str, mode: str, requests: float,
                   l1_5_hits: Optional[float], l2_served: Optional[float], ddr_misses: Optional[float],
                   traffic: Traffic) -> None:
    bucket = per_tensor.setdefault(tensor, {
        "read_requests": 0.0, "write_requests": 0.0, "l1_5_hits": 0.0, "l2_served": 0.0, "ddr_misses": 0.0,
        "requests_with_unknown_rates": 0.0,
        "ddr_read_bytes": 0.0, "ddr_write_bytes": 0.0, "l2_read_request_bytes": 0.0, "l2_write_request_bytes": 0.0,
    })
    bucket["read_requests" if mode == "read" else "write_requests"] += requests
    if l1_5_hits is None or l2_served is None or ddr_misses is None:
        bucket["requests_with_unknown_rates"] += requests
    else:
        bucket["l1_5_hits"] += l1_5_hits
        bucket["l2_served"] += l2_served
        bucket["ddr_misses"] += ddr_misses
    for name in ("ddr_read_bytes", "ddr_write_bytes", "l2_read_request_bytes", "l2_write_request_bytes"):
        value = traffic.value(name)
        if bucket[name] is None or value is None:
            bucket[name] = None
        else:
            bucket[name] += value


def _reuse_unit(options: Options, allocations: List[float], notes: List[str]) -> float:
    if options.cache_reuse_unit_bytes is not None:
        return options.cache_reuse_unit_bytes
    unit = min(min(allocations), DEFAULT_REUSE_UNIT_BYTES)
    notes.append("reuse distance unit = %g bytes (tiles occupy allocation/unit units)" % unit)
    return unit


def _explicit(
    launch: Launch, units: WorkUnits, sites: List[OpSite], arch: Any, options: Options,
    slots: int, residency: int, accesses: Dict[str, AccessTraffic],
    group_accesses: Dict[Tuple[str, str], AccessTraffic],
) -> Tuple[CacheReport, List[str]]:
    notes = []  # type: List[str]
    unit = _reuse_unit(options, [s.op.access.effective_allocation_bytes for s in sites], notes)
    l2, l1_5, group_size = _level_configs(arch, unit)
    budget = None if options.cache == "trace" else options.cache_sample_budget
    result = run_explicit_trace(launch, units, slots, residency, l2, l1_5, group_size, budget, options.cache_seed)
    notes.extend(result.notes)
    site_by_op = {site.op.op_id: site for site in sites}
    reports = []
    per_tensor = {}  # type: Dict[str, Dict[str, float]]
    for op_id, stats in sorted(result.per_op.items()):
        site = site_by_op[op_id]
        requests = stats.requests
        per_execution = stats.cache_traffic().scale(1.0 / requests) if requests > 0 else Traffic()
        rates = (stats.l1_5_rate, stats.l2_rate, stats.ddr_rate)
        source = "cache_%s:explicit_trace%s" % (options.cache, "" if result.exact else "_sampled")
        accesses[op_id] = _entry(site, per_execution, source, "average_over_launch_requests", None,
                                 requests, *rates)
        reports.append(CacheAccessReport(
            access_id=access_id_for(site), op_path=site.path, tensor=site.op.access.tensor,
            mode=accesses[op_id].mode, request_count=requests, l1_5_hit_rate=rates[0],
            l2_served_rate=rates[1], ddr_miss_rate=rates[2], traffic_per_request=per_execution,
            interval_per_request=accesses[op_id].interval, scope="per_request_average_over_launch",
        ))
        _tensor_bucket(per_tensor, site.op.access.tensor, accesses[op_id].mode, requests,
                       stats.l1_5_hits, stats.l2_served, stats.ddr_misses, stats.cache_traffic())
    for (op_id, group_id), stats in result.per_op_group.items():
        site = site_by_op[op_id]
        requests = stats.requests
        if requests <= 0:
            continue
        per_execution = stats.cache_traffic().scale(1.0 / requests)
        group_accesses[(op_id, group_id)] = _entry(
            site, per_execution, accesses[op_id].source, "average_over_group_requests", None,
            requests, stats.l1_5_rate, stats.l2_rate, stats.ddr_rate,
        )
    report = CacheReport(
        launch=launch.name, mode=options.cache, scenario="conservative",
        backend="program-explicit-trace-v1", exact=result.exact,
        reduction_fidelity="exact_instance_trace", histogram_digest="", problem_digest="",
        per_access=tuple(reports), per_tensor=tuple(sorted(per_tensor.items())),
        assumptions=tuple(notes), sampled_requests=result.sampled_events,
        represented_requests=result.events,
    )
    return report, notes


def _reduction_axis(
    launch: Launch, units: WorkUnits, sites: List[OpSite], arch: Any, options: Options,
    slots: int, sm_count: int, accesses: Dict[str, AccessTraffic],
    group_accesses: Dict[Tuple[str, str], AccessTraffic],
) -> Tuple[CacheReport, str, List[str]]:
    notes = []  # type: List[str]
    axis_names = launch.work_axis_names
    axis_index = {name: i for i, name in enumerate(axis_names)}
    loops = {loop.loop_id: loop for loop in launch.loops()}
    access_loops = {}  # type: Dict[str, int]
    group0 = units.groups[0][1]
    for site in sites:
        for loop_id, where in site.chain:
            if where == "body":
                access_loops[loop_id] = launch.bound_trip_count(loops[loop_id], group0)
    inner_iterations = next(iter(access_loops.values())) if access_loops else 1
    if inner_iterations == 0:
        raise CacheBindingError("inner loop trip count is zero; nothing to model")
    if len(access_loops) > 1:
        notes.append("loops %s share one representative iteration in the cache model" % sorted(access_loops))
    regions = {}  # type: Dict[str, TensorTileRegion]
    cache_accesses = []  # type: List[CacheAccessIR]
    for site in sites:
        access = site.op.access
        kept = tuple(axis_index[a] for a in access.index_map.work_axes if a in axis_index)
        region = TensorTileRegion(
            value=access.tensor, index_map=CacheProjection(kept), tile_shape=access.tile_shape,
            element_bytes=access.element_bytes, payload_bytes=access.payload_bytes,
            allocation_bytes=access.effective_allocation_bytes,
        )
        previous = regions.get(access.tensor)
        if previous is not None and (
            previous.index_map.canonical() != region.index_map.canonical()
            or abs(previous.allocation_bytes - region.allocation_bytes) > 1e-9
        ):
            raise CacheBindingError(
                "%s: tensor %s is accessed with a different identity elsewhere" % (site.path, access.tensor)
            )
        regions[access.tensor] = region
        in_inner = any(where == "body" for _l, where in site.chain)
        if isinstance(site.op, Load):
            if inner_iterations > 1 and not in_inner:
                notes.append("%s: load outside the inner loop is placed at the representative iteration" % site.path)
            cache_accesses.append(CacheAccessIR.load(site.op.op_id, region, transaction_bytes=access.executed_bytes))
        else:
            placement = "tile" if (inner_iterations == 1 or in_inner) else "legacy_wave_boundary"
            if placement == "legacy_wave_boundary":
                notes.append("%s: store outside the inner loop is placed after the wave's inner loop" % site.path)
            cache_accesses.append(CacheAccessIR.store(
                site.op.op_id, region, transaction_bytes=access.executed_bytes,
                write_policy=WritePolicy(propagation="write_through"), include_in_hit_rate=False,
                placement=placement,
            ))
    unit = _reuse_unit(options, [r.allocation_bytes for r in regions.values()], notes)
    l2, l1_5, group_size = _level_configs(arch, unit)
    grid = TileGrid(tuple(axis.extent for axis in launch.work_axes), axes=axis_names)
    sampling = SamplingConfig(seed=options.cache_seed, sample_budget=options.cache_sample_budget)
    if inner_iterations > 1:
        if options.cache_representative_k >= inner_iterations:
            raise CacheBindingError("cache_representative_k=%d must be smaller than inner extent %d"
                                    % (options.cache_representative_k, inner_iterations))
        reduction = ReductionConfig(mode="stable_shadow_cohort", representative_k=options.cache_representative_k)
    else:
        reduction = ReductionConfig()
    # The launch order is the mapper's and a wave is ``slots`` concurrent work units --
    # exactly what the explicit trace and the spatial scheduler consume.  The lazy
    # row-major/panel traversals are used only when they describe that same sequence.
    wave_size = max(min(slots, launch.work_count), 1)
    panel = launch.mapper.panel_parameters()
    traversal = None
    if launch.mapper.kind == "linear_block_id":
        traversal = RowMajorTraversal(wave_size=wave_size, sm_count=sm_count)
    elif panel is not None and len(launch.work_axes) == 2:
        row_panel, column_panel, raster_axis = panel
        candidate = PanelTraversal(row_panel=row_panel, sm_count=sm_count, column_panel=column_panel,
                                   raster_axis=raster_axis)
        if candidate.wave_size == wave_size:
            traversal = candidate
    if traversal is None:
        traversal = ExplicitTraversal(coordinates=tuple(launch.instance_coordinates()), wave_size=wave_size,
                                      sm_count=sm_count, label=launch.mapper.kind)
    notes.append("traversal: launch order from block mapper %s, waves of %d concurrent work units (%s)"
                 % (launch.mapper.kind, wave_size, traversal.canonical()["kind"]))
    if wave_size > 1:
        notes.append("within a wave the %d concurrent units reach the cache in a seeded random order "
                     "(backend model of unordered concurrent SMs); the result is an estimate, not an exact trace"
                     % wave_size)
    try:
        problem = CacheProblem(
            grid=grid, accesses=tuple(cache_accesses),
            traversal=traversal,
            l2=l2, l1_5=l1_5, l1_5_group_size=group_size, inner_iterations=inner_iterations,
            reduction=reduction, sampling=sampling,
        )
        result = model_cache(problem)
    except ModelingValidationError as error:
        raise CacheBindingError(str(error)) from error
    notes.append("cache state between launches: cold")
    structural = "stable_shadow_representative_iteration" if inner_iterations > 1 else None
    reports = []
    per_tensor = {}  # type: Dict[str, Dict[str, float]]
    for site in sites:
        name = site.op.op_id
        item = result.by_name(name)
        requests = item.request_count
        per_execution = (Traffic.from_cache_access(item.traffic, item.mode).scale(1.0 / requests)
                         if requests > 0 else Traffic())
        source = "cache_fast:%s" % result.histogram.reduction_fidelity
        accesses[name] = _entry(site, per_execution, source, "average_over_cache_model_requests", structural,
                                requests, item.l1_5_hit_rate, item.l2_served_rate, item.ddr_miss_rate)
        for group_id, _group in units.groups:
            group_accesses[(name, group_id)] = accesses[name]
        reports.append(CacheAccessReport(
            access_id=access_id_for(site), op_path=site.path, tensor=site.op.access.tensor, mode=item.mode,
            request_count=requests, l1_5_hit_rate=item.l1_5_hit_rate, l2_served_rate=item.l2_served_rate,
            ddr_miss_rate=item.ddr_miss_rate, traffic_per_request=per_execution,
            interval_per_request=accesses[name].interval, scope="per_request_average",
        ))
        _tensor_bucket(per_tensor, site.op.access.tensor, item.mode, requests,
                       requests * item.l1_5_hit_rate, requests * item.l2_served_rate,
                       requests * item.ddr_miss_rate, Traffic.from_cache_access(item.traffic, item.mode))
    report = CacheReport(
        launch=launch.name, mode=options.cache, scenario="conservative", backend=result.backend,
        exact=bool(result.histogram.exact and wave_size == 1 and inner_iterations == 1),
        reduction_fidelity=result.histogram.reduction_fidelity,
        histogram_digest=result.histogram.reuse_digest, problem_digest=problem.digest,
        per_access=tuple(reports), per_tensor=tuple(sorted(per_tensor.items())),
        assumptions=tuple(notes) + tuple(result.diagnostics),
        sampled_requests=result.histogram.sampled_requests, represented_requests=result.histogram.represented_requests,
    )
    return report, problem.digest, notes


def cache_problem_for(launch: Launch, arch: Any, options: Options, slots: int, sm_count: int) -> CacheProblem:
    """Expose the reduction-axis ``CacheProblem`` used by ``fast`` (for same-source regressions)."""

    units = launch.effective_work_units()
    sites = [site for site in launch.op_sites() if touches_global(site.op) and site.op.access is not None]
    expressible, reason = _reduction_axis_expressible(launch, units, sites)
    if not expressible:
        raise CacheBindingError(reason)
    holder = {}  # type: Dict[str, Any]
    original = model_cache

    def capture(problem):
        holder["problem"] = problem
        return original(problem)

    globals()["model_cache"] = capture
    try:
        _reduction_axis(launch, units, sites, arch, options, slots, sm_count, {}, {})
    finally:
        globals()["model_cache"] = original
    return holder["problem"]


__all__ = [
    "AccessTraffic", "CacheBindingError", "CacheView", "GroupCacheView", "access_id_for",
    "build_cache_view", "cache_problem_for", "executed_sites", "on_chip_traffic",
]
