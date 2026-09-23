"""Report-only storage, ownership, and fusion analysis for KernelIR."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .full_model import (
    AliasInterference,
    FeasibilityStatus,
    FusionReport,
    LivenessInterval,
    LivenessReport,
    MemoryTraffic,
    OwnershipAssessment,
    ScopeCapacityReport,
    StoragePeak,
    ValueTrafficPath,
)
from .errors import ModelingValidationError
from .ir import Event, ExecutionDomain, KernelIR, OwnershipMap


_DTYPE_BYTES = {
    "bool": 1,
    "int8": 1,
    "uint8": 1,
    "fp8": 1,
    "float8": 1,
    "bf16": 2,
    "fp16": 2,
    "float16": 2,
    "int16": 2,
    "fp32": 4,
    "float32": 4,
    "int32": 4,
    "fp64": 8,
    "float64": 8,
    "int64": 8,
}


def _buffer_unit_bytes(buffer: Any) -> Optional[float]:
    dtype_bytes = _DTYPE_BYTES.get(buffer.dtype.lower())
    if dtype_bytes is None:
        return None
    elements = 1
    for extent in buffer.shape:
        if not isinstance(extent, int) or isinstance(extent, bool) or extent < 0:
            return None
        elements *= extent
    return float(elements * dtype_bytes)


@dataclass(frozen=True)
class _DomainInfo:
    scope: str
    instances_per_cta: float
    members_per_instance: int
    provenance: str


@dataclass(frozen=True)
class _StorageItem:
    name: str
    storage_scope: str
    execution_scope: str
    alias_group: Optional[str]
    unit_bytes: float
    slots: int
    domain: _DomainInfo


@dataclass(frozen=True)
class _LifetimeSpec:
    storage: _StorageItem
    acquire: Event
    releases: Tuple[Event, ...]
    iteration_distance: int
    minimum_residence: float
    source_kind: str


def _legacy_domain(scope: str) -> _DomainInfo:
    """Conservative compatibility mapping; it does not infer actor width."""

    return _DomainInfo(
        scope=scope,
        instances_per_cta=1.0,
        members_per_instance=1,
        provenance="legacy_unspecified",
    )


def _domain_from_ownership(
    ownership: Optional[OwnershipMap], execution_scope: str
) -> _DomainInfo:
    if ownership is None:
        return _legacy_domain(execution_scope)
    domain = ownership.domain
    return _DomainInfo(
        scope=domain.scope,
        instances_per_cta=domain.instances_per_cta,
        members_per_instance=domain.members_per_instance,
        provenance=ownership.provenance,
    )


def _actor_domain(actor: Any) -> Optional[_DomainInfo]:
    domain = getattr(actor, "execution_domain", None)
    if isinstance(domain, ExecutionDomain):
        return _DomainInfo(
            domain.scope,
            domain.instances_per_cta,
            domain.members_per_instance,
            domain.provenance,
        )
    if actor.execution_scope == "unspecified":
        return None
    return _legacy_domain(actor.execution_scope)


def _buffer_domain(loop: Any, buffer: Any) -> _DomainInfo:
    if buffer.ownership is not None:
        return _domain_from_ownership(buffer.ownership, buffer.execution_scope)
    participants = []
    for actor in loop.actors:
        if any(
            buffer in phase.reads or buffer in phase.writes
            for phase in actor.phases
        ):
            domain = _actor_domain(actor)
            if domain is not None and domain.scope == buffer.execution_scope:
                participants.append(domain)
    unique = set(participants)
    if len(unique) == 1:
        return next(iter(unique))
    return _legacy_domain(buffer.execution_scope)


def _handoff_domain(kernel: KernelIR, handoff: Any) -> _DomainInfo:
    ownership = handoff.ownership or handoff.value.ownership
    if ownership is not None:
        return _domain_from_ownership(ownership, handoff.execution_scope)
    participant_domains = []
    events = (handoff.acquire,) + handoff.releases
    for event in events:
        loop = next(
            loop
            for launch in kernel.launches
            for loop in launch.periodic_loops
            if loop.owner == event.owner
        )
        actor = next(
            actor
            for actor in loop.actors
            if any(phase.name == event.phase_name for phase in actor.phases)
        )
        domain = _actor_domain(actor)
        if domain is not None and domain.scope == handoff.execution_scope:
            participant_domains.append(domain)
    unique = set(participant_domains)
    if len(unique) == 1:
        return next(iter(unique))
    return _legacy_domain(handoff.execution_scope)


def _ownership_assessment(kernel: KernelIR, handoff: Any) -> OwnershipAssessment:
    phase_actors = {
        (phase.owner, phase.name): actor
        for launch in kernel.launches
        for loop in launch.periodic_loops
        for actor in loop.actors
        for phase in actor.phases
    }
    actors = {
        phase_actors[(event.owner, event.phase_name)]
        for event in (handoff.acquire,) + handoff.releases
    }
    actor_domains = {actor.execution_domain for actor in actors}
    storage_map = handoff.ownership
    value_map = handoff.value.ownership
    compatible_maps = (
        storage_map is not None
        and value_map is not None
        and storage_map.compatible_with(value_map)
    )
    compatible_domain = (
        None not in actor_domains
        and len(actor_domains) == 1
        and storage_map is not None
        and next(iter(actor_domains)).scope == storage_map.domain.scope
        and next(iter(actor_domains)).instances_per_cta
        == storage_map.domain.instances_per_cta
        and next(iter(actor_domains)).members_per_instance
        == storage_map.domain.members_per_instance
    )
    if actors and compatible_maps and compatible_domain:
        status = "proven_compatible"
        diagnostic = (
            "producer/consumers share one explicit execution domain and a "
            "compatible layout/index ownership map"
        )
    elif actors and all(
        actor.execution_scope == handoff.execution_scope for actor in actors
    ):
        status = "user_asserted"
        diagnostic = (
            "scope/actor ownership is user asserted; a compatible explicit "
            "OwnershipMap is required for a same-lane proof"
        )
    else:
        status = "unknown"
        diagnostic = "producer/consumer ownership compatibility is unknown"
    return OwnershipAssessment(
        storage_name="handoff_%s" % handoff.name,
        execution_scope=handoff.execution_scope,
        status=status,
        diagnostic=diagnostic,
    )


def _resolve_witnesses(
    kernel: KernelIR,
    witness: Optional[Any],
    witnesses: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    if witness is not None and witnesses is not None:
        raise ModelingValidationError("use either witness or witnesses, not both")
    loops = [
        loop for launch in kernel.launches for loop in launch.periodic_loops
    ]
    if witness is not None:
        if len(loops) != 1:
            raise ModelingValidationError(
                "a singular liveness witness requires exactly one periodic loop"
            )
        result = {loops[0].owner: witness}
    else:
        result = {}
        supplied = {} if witnesses is None else dict(witnesses)
        for loop in loops:
            by_owner = supplied.get(loop.owner)
            by_name = supplied.get(loop.name)
            if by_owner is not None and by_name is not None and by_owner is not by_name:
                raise ModelingValidationError(
                    "conflicting liveness witnesses for loop %s" % loop.name
                )
            selected = by_owner if by_owner is not None else by_name
            if selected is not None:
                result[loop.owner] = selected
    for owner, selected in result.items():
        ii = getattr(selected, "ii", None)
        starts = getattr(selected, "phase_starts", None)
        if (
            not isinstance(ii, (int, float))
            or isinstance(ii, bool)
            or not math.isfinite(float(ii))
            or float(ii) <= 0.0
            or not isinstance(starts, Mapping)
        ):
            raise ModelingValidationError(
                "liveness witness for %s needs positive ii and phase_starts" % owner
            )
    return result


def _phase_latencies(loop: Any, oracle: Optional[Any]) -> Dict[str, float]:
    result = {}
    for phase in loop.phases:
        timing = phase.timing
        if timing is None and oracle is not None:
            timing = oracle.resolve(phase)
        if timing is None:
            raise ModelingValidationError(
                "witness liveness needs Timing or oracle for phase %s" % phase.name
            )
        result[phase.name] = timing.latency
    return result


def _event_time(
    event: Event,
    iteration: int,
    starts: Mapping[str, float],
    latencies: Mapping[str, float],
    ii: float,
) -> float:
    if event.phase_name not in starts:
        raise ModelingValidationError(
            "liveness witness omits phase %s" % event.phase_name
        )
    value = float(starts[event.phase_name]) + iteration * ii
    if event.kind == "done":
        value += latencies[event.phase_name]
    return value


def _expand_lifetimes(
    specifications: Sequence[_LifetimeSpec],
    witness: Any,
    latencies: Mapping[str, float],
) -> Tuple[LivenessInterval, ...]:
    ii = float(witness.ii)
    starts = {name: float(value) for name, value in witness.phase_starts.items()}
    if not starts:
        raise ModelingValidationError("liveness witness phase_starts must not be empty")
    # Difference-constraint witnesses are translation invariant. Normalize one
    # representative phase into [0, II) so the central period is deterministic.
    shift = math.floor(min(starts.values()) / ii) * ii
    starts = {name: value - shift for name, value in starts.items()}
    base = []
    padding = 3
    for item in specifications:
        acquire = _event_time(item.acquire, 0, starts, latencies, ii)
        releases = []
        for event in item.releases:
            value = _event_time(
                event,
                item.iteration_distance,
                starts,
                latencies,
                ii,
            )
            while value + 1.0e-15 < acquire + item.minimum_residence:
                value += ii
            releases.append(value)
        release = max(releases)
        base.append((item, acquire, release))
        padding = max(
            padding,
            int(math.ceil(max(0.0, release - acquire) / ii)) + item.storage.slots + 2,
        )

    intervals = []
    for item, acquire, release in base:
        for iteration in range(-padding, padding + 1):
            start = acquire + iteration * ii
            end = release + iteration * ii
            if end <= 0.0 or start >= ii or end <= start:
                continue
            intervals.append(
                LivenessInterval(
                    storage_name=item.storage.name,
                    source_kind=item.source_kind,
                    storage_scope=item.storage.storage_scope,
                    execution_scope=item.storage.execution_scope,
                    iteration=iteration,
                    start_s=start,
                    end_s=end,
                    bytes_per_instance=item.storage.unit_bytes,
                    alias_group=item.storage.alias_group,
                )
            )
    return tuple(intervals)


def _cluster_size(launch: Any) -> int:
    result = 1
    for extent in launch.cluster:
        result *= int(extent)
    return result


def _replicas_per_sm(
    domain: _DomainInfo, resident_ctas_per_sm: float, cluster_size: int
) -> float:
    if domain.scope == "cta_group":
        return (
            resident_ctas_per_sm
            * domain.instances_per_cta
            / float(cluster_size)
        )
    return resident_ctas_per_sm * domain.instances_per_cta


def _scope_capacity(arch: Optional[Any], storage_scope: str) -> Optional[float]:
    if arch is None:
        return None
    candidates = {
        "smem": ("configurable_smem_capacity", "smem_capacity_per_sm"),
        "shared": ("configurable_smem_capacity", "smem_capacity_per_sm"),
        "tmem": ("tmem_capacity_per_sm",),
        "register": (
            "register_capacity_per_sm",
            "register_file_bytes_per_sm",
        ),
        "fragment": (
            "register_capacity_per_sm",
            "register_file_bytes_per_sm",
        ),
    }.get(storage_scope, tuple())
    for name in candidates:
        value = getattr(arch, name, None)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
            return float(value)
    return None


def _evaluate_group(
    items: Sequence[_StorageItem],
    intervals: Sequence[LivenessInterval],
    static_allocation: bool,
    resident_ctas_per_sm: float,
    cluster_size: int,
) -> Tuple[float, float, float, float, Tuple[AliasInterference, ...], str]:
    dynamic_names = {item.storage_name for item in intervals}
    fallback_names = {
        item.name for item in items if item.name not in dynamic_names
    }
    boundaries = {0.0, 1.0}
    # Use a normalized period for sweep positions; caller-provided interval
    # endpoints are divided by II before entering through this local closure.
    normalized = []
    for interval in intervals:
        start = max(0.0, interval.start_s)
        end = min(1.0, interval.end_s)
        if end > start:
            boundaries.update((start, end))
            normalized.append((interval, start, end))
    ordered = sorted(boundaries)
    sample_points = [
        (left + right) * 0.5
        for left, right in zip(ordered, ordered[1:])
        if right > left
    ] or [0.5]

    peak_lower = 0.0
    peak_selected = 0.0
    peak_upper = 0.0
    peak_aggregate = 0.0
    interference = {}
    credit_overflow = False
    for point in sample_points:
        live_iterations = defaultdict(set)
        for interval, start, end in normalized:
            if start <= point < end:
                live_iterations[interval.storage_name].add(interval.iteration)
        values = {}
        weighted = {}
        for item in items:
            count = item.slots if static_allocation or item.name in fallback_names else len(
                live_iterations.get(item.name, ())
            )
            if count > item.slots:
                credit_overflow = True
            count = min(count, item.slots)
            value = item.unit_bytes * count
            values[item.name] = value
            weighted[item.name] = value * _replicas_per_sm(
                item.domain, resident_ctas_per_sm, cluster_size
            )

        lower = max(values.values(), default=0.0)
        upper = sum(values.values())
        selected = 0.0
        aggregate = 0.0
        alias_members = defaultdict(list)
        for item in items:
            if item.alias_group is None:
                selected += values[item.name]
                aggregate += weighted[item.name]
            elif values[item.name] > 0.0:
                alias_members[item.alias_group].append(item)
        for group, members in alias_members.items():
            selected += max(values[item.name] for item in members)
            aggregate += max(weighted[item.name] for item in members)
            distinct = tuple(
                sorted(
                    item.name
                    for item in members
                    if live_iterations.get(item.name)
                )
            )
            if len(distinct) > 1:
                live_bytes = sum(
                    next(item.unit_bytes for item in members if item.name == name)
                    * min(
                        len(live_iterations.get(name, ())),
                        next(item.slots for item in members if item.name == name),
                    )
                    for name in distinct
                )
                previous = interference.get(group)
                if previous is None or live_bytes > previous[0]:
                    interference[group] = (live_bytes, point, distinct)
        peak_lower = max(peak_lower, lower)
        peak_selected = max(peak_selected, selected)
        peak_upper = max(peak_upper, upper)
        peak_aggregate = max(peak_aggregate, aggregate)

    records = tuple(
        AliasInterference(
            alias_group=group,
            storage_scope=items[0].storage_scope,
            execution_scope=items[0].execution_scope,
            time_s=point,
            members=members,
            live_bytes_per_instance=live_bytes,
        )
        for group, (live_bytes, point, members) in sorted(interference.items())
    )
    if static_allocation:
        allocation_kind = "static_allocation"
    elif fallback_names:
        allocation_kind = "witness_with_static_fallback"
    else:
        allocation_kind = "witness_dynamic"
    if credit_overflow:
        allocation_kind += "_credit_overflow"
    return (
        peak_lower,
        peak_selected,
        peak_upper,
        peak_aggregate,
        records,
        allocation_kind,
    )


def analyze_liveness(
    kernel: KernelIR,
    arch: Optional[Any] = None,
    resident_ctas_per_sm: Optional[float] = None,
    witness: Optional[Any] = None,
    witnesses: Optional[Mapping[str, Any]] = None,
    oracle: Optional[Any] = None,
) -> LivenessReport:
    """Compute static or witness-expanded report-only storage peaks.

    SMEM is a static allocation even when a witness is available. Register,
    fragment, and TMEM storage use event intervals from ``phase_starts + i*II``
    when supplied. Capacity and alias interference are observations only;
    ``guard_eligible`` is always false.
    """

    witness_by_owner = _resolve_witnesses(kernel, witness, witnesses)
    launch_peaks = defaultdict(list)  # type: Dict[Tuple[str, str], list]
    capacity_by_group = defaultdict(list)
    all_intervals = []  # type: List[LivenessInterval]
    all_interference = []  # type: List[AliasInterference]
    ownership_assessments = []  # type: List[OwnershipAssessment]
    unknown = []
    loop_launch = {
        loop.owner: launch.name
        for launch in kernel.launches
        for loop in launch.periodic_loops
    }
    handoffs_by_launch = defaultdict(list)
    handoff_count = 0
    for plan in kernel.fusion_plans:
        for handoff in plan.handoffs:
            handoff_count += 1
            ownership_assessments.append(_ownership_assessment(kernel, handoff))
            launch_name = loop_launch[handoff.acquire.owner]
            handoffs_by_launch[launch_name].append(handoff)
    handoff_by_loop = defaultdict(list)
    for plan in kernel.fusion_plans:
        for handoff in plan.handoffs:
            if all(event.owner == handoff.acquire.owner for event in handoff.releases):
                handoff_by_loop[handoff.acquire.owner].append(handoff)
    replicas = 1.0 if resident_ctas_per_sm is None else float(resident_ctas_per_sm)
    if not math.isfinite(replicas) or replicas <= 0.0:
        raise ModelingValidationError("resident_ctas_per_sm must be positive")

    for launch in kernel.launches:
        groups = defaultdict(list)  # type: Dict[Tuple[str, str], List[_StorageItem]]
        lifetime_specs = defaultdict(list)
        for loop in launch.periodic_loops:
            items_by_buffer = {}
            for buffer in loop.buffers:
                size = _buffer_unit_bytes(buffer)
                if size is None:
                    unknown.append(buffer.qualified_name)
                    continue
                item = _StorageItem(
                    name="%s:%s:%s" % (launch.name, loop.name, buffer.name),
                    storage_scope=buffer.scope,
                    execution_scope=buffer.execution_scope,
                    alias_group=buffer.alias_group,
                    unit_bytes=size,
                    slots=buffer.slots,
                    domain=_buffer_domain(loop, buffer),
                )
                items_by_buffer[buffer] = item
                groups[(buffer.scope, buffer.execution_scope)].append(item)
            for lifetime in loop.lifetimes:
                item = items_by_buffer.get(lifetime.buffer)
                if item is not None:
                    lifetime_specs[loop.owner].append(
                        _LifetimeSpec(
                            item,
                            lifetime.acquire,
                            (lifetime.release,),
                            0,
                            lifetime.minimum_residence,
                            "buffer",
                        )
                    )
            for carry in loop.carries:
                if carry.state.storage is None:
                    continue
                item = items_by_buffer.get(carry.state.storage)
                if item is not None:
                    lifetime_specs[loop.owner].append(
                        _LifetimeSpec(
                            item,
                            carry.source,
                            (carry.target,),
                            carry.iteration_distance,
                            0.0,
                            "state",
                        )
                    )
            for handoff in handoff_by_loop.get(loop.owner, ()):
                item = _StorageItem(
                    name="%s:%s:handoff_%s"
                    % (launch.name, loop.name, handoff.name),
                    storage_scope=handoff.via,
                    execution_scope=handoff.execution_scope,
                    alias_group=handoff.alias_group,
                    unit_bytes=handoff.bytes,
                    slots=handoff.slots,
                    domain=_handoff_domain(kernel, handoff),
                )
                groups[(handoff.via, handoff.execution_scope)].append(item)
                lifetime_specs[loop.owner].append(
                    _LifetimeSpec(
                        item,
                        handoff.acquire,
                        handoff.releases,
                        handoff.iteration_distance,
                        0.0,
                        "handoff",
                    )
                )

        same_loop_handoffs = {
            handoff
            for values in handoff_by_loop.values()
            for handoff in values
        }
        for handoff in handoffs_by_launch.get(launch.name, ()):
            if handoff in same_loop_handoffs:
                continue
            item = _StorageItem(
                name="%s:handoff_%s" % (launch.name, handoff.name),
                storage_scope=handoff.via,
                execution_scope=handoff.execution_scope,
                alias_group=handoff.alias_group,
                unit_bytes=handoff.bytes,
                slots=handoff.slots,
                domain=_handoff_domain(kernel, handoff),
            )
            groups[(handoff.via, handoff.execution_scope)].append(item)

        launch_intervals = []
        for loop in launch.periodic_loops:
            selected_witness = witness_by_owner.get(loop.owner)
            if selected_witness is None:
                continue
            expanded = _expand_lifetimes(
                lifetime_specs.get(loop.owner, ()),
                selected_witness,
                _phase_latencies(loop, oracle),
            )
            ii = float(selected_witness.ii)
            # Normalize to [0, 1] only for the peak sweep. Retain SI seconds in
            # the public interval report.
            launch_intervals.extend(expanded)
            all_intervals.extend(expanded)

        cluster_size = _cluster_size(launch)
        for key, items in groups.items():
            matching = [
                interval
                for interval in launch_intervals
                if (interval.storage_scope, interval.execution_scope) == key
            ]
            witness_iis = {
                float(item.ii)
                for owner, item in witness_by_owner.items()
                if owner in {loop.owner for loop in launch.periodic_loops}
            }
            if len(witness_iis) == 1:
                ii = next(iter(witness_iis))
                normalized = tuple(
                    LivenessInterval(
                        storage_name=item.storage_name,
                        source_kind=item.source_kind,
                        storage_scope=item.storage_scope,
                        execution_scope=item.execution_scope,
                        iteration=item.iteration,
                        start_s=item.start_s / ii,
                        end_s=item.end_s / ii,
                        bytes_per_instance=item.bytes_per_instance,
                        alias_group=item.alias_group,
                    )
                    for item in matching
                )
            else:
                normalized = tuple()
            static = key[0] in ("smem", "shared")
            lower, selected, upper, aggregate, interference, kind = _evaluate_group(
                items,
                normalized,
                static_allocation=static,
                resident_ctas_per_sm=replicas,
                cluster_size=cluster_size,
            )
            launch_peaks[key].append((lower, selected, upper, kind))
            capacity_by_group[key].append(
                (aggregate, selected, items, cluster_size, kind)
            )
            time_scale = next(iter(witness_iis)) if len(witness_iis) == 1 else 0.0
            all_interference.extend(
                AliasInterference(
                    alias_group=item.alias_group,
                    storage_scope=item.storage_scope,
                    execution_scope=item.execution_scope,
                    time_s=item.time_s * time_scale,
                    members=item.members,
                    live_bytes_per_instance=item.live_bytes_per_instance,
                )
                for item in interference
            )

    peaks = []
    for (storage_scope, execution_scope), values in sorted(launch_peaks.items()):
        # KernelIR launches execute sequentially. Buffers within one launch may
        # coexist, while allocation peaks across launches combine by max.
        lower = max(value[0] for value in values)
        selected = max(value[1] for value in values)
        upper = max(value[2] for value in values)
        kinds = {value[3] for value in values}
        peaks.append(
            StoragePeak(
                storage_scope=storage_scope,
                execution_scope=execution_scope,
                lower_bytes=lower,
                selected_bytes=selected,
                upper_bytes=upper,
                confidence=(
                    "witness_interval_peak"
                    if kinds == {"witness_dynamic"}
                    else "explicit_alias_envelope"
                ),
            )
        )

    diagnostics = ["report-only: this result is not a fusion feasibility guard"]
    if witness_by_owner:
        diagnostics.append(
            "witness-aware register/TMEM intervals expanded across adjacent iterations"
        )
        diagnostics.append("SMEM remains a static allocation independent of liveness")
    else:
        diagnostics.append("static storage envelope only; no schedule witness supplied")
    if handoff_count:
        diagnostics.append(
            "included %d explicit handoff storage lifetime(s) from acquire through all releases"
            % handoff_count
        )
    status = FeasibilityStatus.UNKNOWN
    capacity_reports = []
    cta_static_violations = []
    for (storage_scope, execution_scope), values in sorted(capacity_by_group.items()):
        aggregate, selected, items, cluster_size, kind = max(
            values, key=lambda value: value[0]
        )
        capacity = _scope_capacity(arch, storage_scope)
        exceeds = None if capacity is None else aggregate > capacity
        domains = {item.domain for item in items}
        if len(domains) == 1:
            domain = next(iter(domains))
            instances = domain.instances_per_cta
            members = domain.members_per_instance
            provenance = domain.provenance
            replica_count = _replicas_per_sm(domain, replicas, cluster_size)
        else:
            instances = aggregate / selected / replicas if selected > 0.0 else 0.0
            members = 1
            provenance = "mixed_domains"
            replica_count = aggregate / selected if selected > 0.0 else 0.0
        capacity_reports.append(
            ScopeCapacityReport(
                storage_scope=storage_scope,
                execution_scope=execution_scope,
                allocation_kind=kind,
                instances_per_cta=instances,
                members_per_instance=members,
                resident_ctas_per_sm=replicas,
                cluster_size=cluster_size,
                replicas_per_sm=replica_count,
                per_replica_peak_bytes=selected,
                aggregate_peak_bytes=aggregate,
                capacity_bytes=capacity,
                exceeds_capacity=exceeds,
                ownership_provenance=provenance,
                confidence=(
                    "witness_report_only"
                    if kind.startswith("witness")
                    else "static_report_only"
                ),
            )
        )
        if exceeds:
            diagnostics.append(
                "%s/%s aggregate %.0f B across %.3g replicas exceeds %.0f B/SM"
                % (storage_scope, execution_scope, aggregate, replica_count, capacity)
            )
            if execution_scope == "cta" and kind == "static_allocation":
                cta_static_violations.append((storage_scope, execution_scope))
    if cta_static_violations:
        # Preserve the legacy narrow proof while keeping the report explicitly
        # ineligible as a model guard.
        status = FeasibilityStatus.PROVEN_INFEASIBLE
    if any(item.execution_scope != "cta" for item in capacity_reports):
        diagnostics.append(
            "non-CTA replica scopes are capacity-reported but remain report-only"
        )
    if all(item.exceeds_capacity is not True for item in capacity_reports):
        diagnostics.append(
            "declared storage fits known capacities where available; other "
            "feasibility conditions remain unknown"
        )
    if all_interference:
        diagnostics.append(
            "%d explicit alias group(s) have overlapping live members"
            % len(all_interference)
        )
    if unknown:
        diagnostics.append("unknown sizes: %s" % ", ".join(sorted(unknown)))
    witness_iis = {float(item.ii) for item in witness_by_owner.values()}
    return LivenessReport(
        status=status,
        peaks=tuple(peaks),
        diagnostics=tuple(diagnostics),
        intervals=tuple(all_intervals),
        alias_interference=tuple(all_interference),
        capacity=tuple(capacity_reports),
        ownership=tuple(ownership_assessments),
        witness_ii_s=next(iter(witness_iis)) if len(witness_iis) == 1 else None,
        guard_eligible=False,
    )


def analyze_fusion(kernel: KernelIR) -> FusionReport:
    """Mechanically apply explicit handoffs to declared per-level traffic.

    ``PROVEN_FEASIBLE`` here has a deliberately narrow meaning: every
    materialization replacement is structurally valid, lossless under its
    declared layout map, exactly accounted, and every register handoff has an
    explicit same-domain/same-index ownership proof. Register scope without a
    compatible ``OwnershipMap`` remains ``UNKNOWN`` while retaining complete
    traffic accounting. This does not claim fused latency or storage capacity.
    """

    alias_count = sum(
        1
        for launch in kernel.launches
        for loop in launch.periodic_loops
        for buffer in loop.buffers
        if buffer.alias_group is not None
    )
    if not kernel.fusion_plans:
        return FusionReport(
            status=FeasibilityStatus.NOT_REQUESTED,
            diagnostics=(
                "no explicit FusionPlan is present in KernelIR",
                "observed %d explicit alias-buffer declarations" % alias_count,
                "fusion analysis was not requested; this is not an unknown plan",
            ),
        )

    baseline = defaultdict(float)
    eliminated = defaultdict(float)
    introduced = defaultdict(float)
    for access in kernel.memory_accesses:
        baseline[(access.memory_level, access.mode)] += access.bytes

    value_paths = []
    generic_count = 0
    handoff_count = 0
    register_ownership = []
    for plan in kernel.fusion_plans:
        for handoff in plan.handoffs:
            handoff_count += 1
            if handoff.via == "register":
                register_ownership.append(
                    (handoff.name, _ownership_assessment(kernel, handoff))
                )
            accesses = handoff.baseline_writes + handoff.baseline_reads
            generic_count += sum(
                access.accounting == "generic_equal_bytes" for access in accesses
            )
            for access in accesses:
                eliminated[(access.memory_level, access.mode)] += access.bytes
            introduced[(handoff.via, "write")] += handoff.bytes
            introduced[(handoff.via, "read")] += handoff.bytes * len(
                handoff.releases
            )

            consumer_fragments = []
            seen_consumers = set()
            for access in handoff.baseline_reads:
                key = (
                    access.fragment,
                    access.event.owner,
                    access.event.phase_name,
                )
                if key not in seen_consumers:
                    seen_consumers.add(key)
                    consumer_fragments.append(access.fragment)

            def _path(*levels: str) -> Tuple[str, ...]:
                result = []
                for level in levels:
                    if not result or result[-1] != level:
                        result.append(level)
                return tuple(result)

            value_paths.append(
                ValueTrafficPath(
                    value=handoff.value.name,
                    producer_fragment=handoff.baseline_writes[0].fragment,
                    consumer_fragments=tuple(consumer_fragments),
                    baseline_store_path=_path(
                        handoff.producer_source, handoff.baseline_storage
                    ),
                    baseline_load_paths=tuple(
                        _path(handoff.baseline_storage, target)
                        for target in handoff.consumer_targets
                    ),
                    fused_handoff_paths=tuple(
                        _path(handoff.producer_source, handoff.via, target)
                        for target in handoff.consumer_targets
                    ),
                    iteration_distance=handoff.iteration_distance,
                )
            )

    levels = sorted(
        set(level for level, _mode in baseline)
        | set(level for level, _mode in eliminated)
        | set(level for level, _mode in introduced)
    )
    traffic = []
    for level in levels:
        values = {}
        for mode in ("read", "write"):
            base = baseline[(level, mode)]
            remove = eliminated[(level, mode)]
            add = introduced[(level, mode)]
            values["baseline_%s_bytes" % mode] = base
            values["eliminated_%s_bytes" % mode] = remove
            values["introduced_%s_bytes" % mode] = add
            values["fused_%s_bytes" % mode] = base - remove + add
        traffic.append(MemoryTraffic(memory_level=level, **values))

    ddr = next((item for item in traffic if item.memory_level == "ddr"), None)
    eliminated_ddr = 0.0 if ddr is None else (
        ddr.eliminated_read_bytes + ddr.eliminated_write_bytes
    )
    unproven_register = [
        (name, assessment)
        for name, assessment in register_ownership
        if assessment.status != "proven_compatible"
    ]
    status = (
        FeasibilityStatus.UNKNOWN
        if unproven_register
        else FeasibilityStatus.PROVEN_FEASIBLE
    )
    diagnostics = [
        "explicit handoff/traffic transformation is structurally closed",
        "this status does not model fused latency or prove storage capacity",
        "eliminated_global_bytes means eliminated DDR read+write traffic",
    ]
    if unproven_register:
        diagnostics.append(
            "register ownership is not proven; traffic accounting remains complete"
        )
        diagnostics.extend(
            "register handoff %s ownership=%s: %s"
            % (name, assessment.status, assessment.diagnostic)
            for name, assessment in unproven_register
        )
    elif register_ownership:
        diagnostics.append(
            "all register handoffs have compatible explicit OwnershipMap proofs"
        )
    if generic_count:
        diagnostics.append(
            "%d access entries use materialize()'s generic equal-byte DDR/L2 assumption; "
            "use direct memory_access() for calibrated per-level traffic" % generic_count
        )
    return FusionReport(
        status=status,
        diagnostics=tuple(diagnostics),
        eliminated_global_bytes=eliminated_ddr,
        traffic=tuple(traffic),
        value_paths=tuple(value_paths),
        handoff_count=handoff_count,
    )


__all__ = ["analyze_fusion", "analyze_liveness"]
