"""Opt-in recursive region evaluator and native whole-kernel wave executor.

This module does not replace the calibrated legacy adapters. It provides a
truthfully scoped path for new/fused kernels whose costs have been bound to a
recursive region tree.
"""

from __future__ import annotations

import hashlib
import math
from collections import deque
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

from tilesight.modeling._pipeline.periodic_schedule import (
    IIScheduleMode,
    PeriodicDAG,
    compute_lower_bounds,
    schedule_periodic_dag,
    select_steady_ii,
)

from .errors import ModelingError, ModelingValidationError, TimingResolutionError
from .full_model import FeasibilityStatus, FusionReport, LivenessReport
from .ir import KernelIR, Phase, Timing, freeze_attrs, validate_identifier
from .liveness import analyze_fusion, analyze_liveness
from .native_policy import (
    BOUNDARY_POLICIES,
    WAVE_POLICIES,
    LaunchTopology,
    SmallTileLatencyConfig,
    WaveCostContext,
)
from .resident_schedule import (
    ResidentScheduleConfig,
    ResourceScope,
    schedule_resident_ctas,
)
from .regions import (
    LoopRegion,
    PhaseRegion,
    SequenceRegion,
    contains_loop,
    is_region,
    phase_regions,
)


class NativeModelError(ModelingError):
    """Raised when a native recursive model would make an unsafe assumption."""


class FusionAwareCostResolver(Protocol):
    consumes_fusion_traffic: bool
    consumes_cache_traffic: bool

    def resolve_fused(
        self,
        phase: Phase,
        kernel: KernelIR,
        fusion: FusionReport,
    ) -> Timing:
        """Resolve a phase after consuming explicit fusion/cache traffic."""


class ContextBoundCostResolver(Protocol):
    """Optional cost-oracle hook for active-SM/tail rebinding."""

    def bind_context(self, context: WaveCostContext) -> Any:
        """Return an oracle view bound to one concrete spatial wave."""


@dataclass(frozen=True)
class TraceRepeat:
    """One compressed repetition axis in a resource trace."""

    label: str
    count: int
    stride_s: float

    def __post_init__(self) -> None:
        validate_identifier(self.label, "trace repeat label")
        if not isinstance(self.count, int) or isinstance(self.count, bool) or self.count <= 0:
            raise ModelingValidationError("trace repeat count must be positive")
        if not math.isfinite(self.stride_s) or self.stride_s < 0.0:
            raise ModelingValidationError("trace repeat stride_s must be non-negative")


@dataclass(frozen=True)
class ResourceTraceEntry:
    """A service interval, optionally compressed over nested loop axes."""

    path: str
    phase: str
    resource: str
    offset_s: Optional[float]
    service_time_s: float
    repeats: Tuple[TraceRepeat, ...] = field(default_factory=tuple)
    ordering: str = "known"

    def __post_init__(self) -> None:
        if not self.path:
            raise ModelingValidationError("resource trace path must not be empty")
        validate_identifier(self.phase, "resource trace phase")
        validate_identifier(self.resource, "resource trace resource")
        if self.offset_s is not None and (
            not math.isfinite(self.offset_s) or self.offset_s < 0.0
        ):
            raise ModelingValidationError("resource trace offset_s must be non-negative")
        if not math.isfinite(self.service_time_s) or self.service_time_s <= 0.0:
            raise ModelingValidationError("resource trace service_time_s must be positive")
        object.__setattr__(self, "repeats", tuple(self.repeats))
        if not all(isinstance(item, TraceRepeat) for item in self.repeats):
            raise ModelingValidationError("resource trace repeats must contain TraceRepeat")
        if self.ordering not in ("known", "aggregate_only"):
            raise ModelingValidationError(
                "resource trace ordering must be 'known' or 'aggregate_only'"
            )

    @property
    def total_service_s(self) -> float:
        repetitions = 1
        for repeat in self.repeats:
            repetitions *= repeat.count
        return self.service_time_s * repetitions


@dataclass(frozen=True)
class RegionEvaluation:
    """Recursive timing decomposition for one region."""

    name: str
    kind: str
    total_s: float
    first_s: float
    steady_s: float
    drain_s: float
    strategy: str
    estimate_scope: str
    trip_count: Optional[int] = None
    selected_ii_s: Optional[float] = None
    resource_ii_s: Optional[float] = None
    recurrence_ii_s: Optional[float] = None
    credit_ii_s: Optional[float] = None
    best_ii_s: Optional[float] = None
    worst_ii_s: Optional[float] = None
    ii_scope: str = "not_applicable"
    witness_loop_name: Optional[str] = None
    witness_phase_starts: Tuple[Tuple[str, float], ...] = field(default_factory=tuple)
    periodic_dag_digest: Optional[str] = None   # identity of the solved PeriodicDAG (phases, costs, constraints)
    children: Tuple["RegionEvaluation", ...] = field(default_factory=tuple)
    resource_trace: Tuple[ResourceTraceEntry, ...] = field(default_factory=tuple)
    diagnostics: Tuple[str, ...] = field(default_factory=tuple)
    finite_event_starts: Tuple[Tuple[str, int, float], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        validate_identifier(self.name, "region evaluation name")
        object.__setattr__(
            self,
            "finite_event_starts",
            tuple((str(n), int(i), float(t)) for n, i, t in self.finite_event_starts),
        )
        validate_identifier(self.kind, "region evaluation kind")
        validate_identifier(self.strategy, "region evaluation strategy")
        validate_identifier(self.estimate_scope, "region estimate scope")
        validate_identifier(self.ii_scope, "region II scope")
        if self.witness_loop_name is not None:
            validate_identifier(self.witness_loop_name, "witness loop name")
        object.__setattr__(
            self,
            "witness_phase_starts",
            tuple((str(name), float(value)) for name, value in self.witness_phase_starts),
        )
        if bool(self.witness_phase_starts) != bool(self.witness_loop_name):
            raise ModelingValidationError(
                "witness loop name and phase starts must be present together"
            )
        if self.witness_phase_starts and self.selected_ii_s is None:
            raise ModelingValidationError("witness phase starts require selected_ii_s")
        for name in ("total_s", "first_s", "steady_s", "drain_s"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ModelingValidationError("%s must be finite and non-negative" % name)
            object.__setattr__(self, name, value)
        for name in (
            "selected_ii_s",
            "resource_ii_s",
            "recurrence_ii_s",
            "credit_ii_s",
            "best_ii_s",
            "worst_ii_s",
        ):
            value = getattr(self, name)
            if value is not None:
                value = float(value)
                if not math.isfinite(value) or value < 0.0:
                    raise ModelingValidationError("%s must be non-negative" % name)
                object.__setattr__(self, name, value)
        expected = self.first_s + self.steady_s + self.drain_s
        tolerance = 1.0e-15 + 1.0e-10 * max(self.total_s, expected)
        if abs(self.total_s - expected) > tolerance:
            raise ModelingValidationError(
                "region total_s must equal first_s + steady_s + drain_s"
            )
        if self.trip_count is not None and (
            not isinstance(self.trip_count, int)
            or isinstance(self.trip_count, bool)
            or self.trip_count < 0
        ):
            raise ModelingValidationError("region trip_count must be non-negative")
        object.__setattr__(self, "children", tuple(self.children))
        object.__setattr__(self, "resource_trace", tuple(self.resource_trace))
        object.__setattr__(self, "diagnostics", tuple(str(x) for x in self.diagnostics))

    @property
    def resource_service_s(self) -> Tuple[Tuple[str, float], ...]:
        totals = {}  # type: Dict[str, float]
        for item in self.resource_trace:
            totals[item.resource] = totals.get(item.resource, 0.0) + item.total_service_s
        return tuple(sorted(totals.items()))


@dataclass(frozen=True)
class NativeModelOptions:
    """Controls recursive summarization and the spatial wave executor."""

    ii_mode: str = "periodic_best"
    unroll_threshold: int = 4
    max_inline_iterations: int = 64
    max_finite_iterations: int = 4096
    cost_source: str = "phase_timing"
    launch_name: str = "main"
    boundary_policy: str = "periodic_witness"
    wave_policy: str = "native"
    launch_topology: Optional[LaunchTopology] = None
    small_tile: Optional[SmallTileLatencyConfig] = None
    resident_schedule: Optional[ResidentScheduleConfig] = None
    resident_ctas_per_sm: Optional[int] = None
    multi_cta_policy: str = "reject"
    liveness_policy: str = "require_witness"
    kernel_launch_s: float = 2.0e-6
    host_dispatch_s: float = 0.0
    reject_proven_infeasible: bool = False
    periodic_topology_trials: int = 0
    periodic_topology_seed: int = 0

    def __post_init__(self) -> None:
        if self.ii_mode not in tuple(item.value for item in IIScheduleMode):
            raise ModelingValidationError("unsupported native ii_mode %r" % self.ii_mode)
        for name in (
            "unroll_threshold",
            "max_inline_iterations",
            "max_finite_iterations",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ModelingValidationError("%s must be a non-negative integer" % name)
        if self.max_inline_iterations < self.unroll_threshold:
            raise ModelingValidationError(
                "max_inline_iterations must be >= unroll_threshold"
            )
        if self.max_finite_iterations <= 0:
            raise ModelingValidationError(
                "max_finite_iterations must be a positive integer"
            )
        if self.cost_source not in (
            "phase_timing",
            "explicit_fused_static_timing",
            "fusion_aware_oracle",
        ):
            raise ModelingValidationError("unsupported native cost_source")
        validate_identifier(self.launch_name, "native launch_name")
        if self.boundary_policy not in BOUNDARY_POLICIES:
            raise ModelingValidationError(
                "boundary_policy must be one of %s"
                % ", ".join(BOUNDARY_POLICIES)
            )
        if self.wave_policy not in WAVE_POLICIES:
            raise ModelingValidationError(
                "wave_policy must be one of %s" % ", ".join(WAVE_POLICIES)
            )
        if self.launch_topology is not None and not isinstance(
            self.launch_topology, LaunchTopology
        ):
            raise ModelingValidationError(
                "launch_topology must be LaunchTopology or None"
            )
        if self.small_tile is not None and not isinstance(
            self.small_tile, SmallTileLatencyConfig
        ):
            raise ModelingValidationError(
                "small_tile must be SmallTileLatencyConfig or None"
            )
        if self.resident_schedule is not None and not isinstance(
            self.resident_schedule, ResidentScheduleConfig
        ):
            raise ModelingValidationError(
                "resident_schedule must be ResidentScheduleConfig or None"
            )
        if self.resident_ctas_per_sm is not None and (
            not isinstance(self.resident_ctas_per_sm, int)
            or isinstance(self.resident_ctas_per_sm, bool)
            or self.resident_ctas_per_sm <= 0
        ):
            raise ModelingValidationError("resident_ctas_per_sm must be positive")
        if self.multi_cta_policy not in (
            "reject",
            "resource_bound",
            "constructive_best",
            "constructive_worst",
        ):
            raise ModelingValidationError(
                "multi_cta_policy must be reject, resource_bound, "
                "constructive_best, or constructive_worst"
            )
        if (
            self.multi_cta_policy.startswith("constructive_")
            and self.resident_schedule is None
        ):
            raise ModelingValidationError(
                "constructive resident policy requires resident_schedule"
            )
        if self.liveness_policy not in ("require_witness", "report_static"):
            raise ModelingValidationError(
                "liveness_policy must be 'require_witness' or 'report_static'"
            )
        if not isinstance(self.reject_proven_infeasible, bool):
            raise ModelingValidationError("reject_proven_infeasible must be bool")
        for name in ("kernel_launch_s", "host_dispatch_s"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ModelingValidationError("%s must be non-negative" % name)
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class NativeGridResult:
    """Logical work and physical CTA counts for the static native launch."""

    work_units: int
    physical_programs: int
    active_work_units: int
    active_physical_ctas: int
    waves: int
    full_waves: int
    tail_work_units: int
    tail_factor: float

    def __post_init__(self) -> None:
        for name in (
            "work_units",
            "physical_programs",
            "active_work_units",
            "active_physical_ctas",
            "waves",
            "full_waves",
            "tail_work_units",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ModelingValidationError("%s must be a non-negative integer" % name)
        if not math.isfinite(self.tail_factor) or self.tail_factor < 0.0:
            raise ModelingValidationError("tail_factor must be non-negative")


@dataclass(frozen=True)
class NativeModelResult:
    region: RegionEvaluation
    per_work_unit_s: float
    resident_group_s: float
    tail_wave_s: float
    kernel_body_s: float
    launch_s: float
    host_dispatch_s: float
    total_s: float
    resident_ctas_per_sm: int
    effective_resident_ctas_per_sm: int
    sm_count: int
    grid: NativeGridResult
    spatial_scope: str
    liveness: LivenessReport
    fusion: FusionReport
    provenance: Tuple[Tuple[str, Any], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.region, RegionEvaluation):
            raise ModelingValidationError("native region must be RegionEvaluation")
        if not isinstance(self.grid, NativeGridResult):
            raise ModelingValidationError("native grid must be NativeGridResult")
        if not isinstance(self.liveness, LivenessReport):
            raise ModelingValidationError("native liveness must be LivenessReport")
        if not isinstance(self.fusion, FusionReport):
            raise ModelingValidationError("native fusion must be FusionReport")
        for name in (
            "per_work_unit_s",
            "resident_group_s",
            "tail_wave_s",
            "kernel_body_s",
            "launch_s",
            "host_dispatch_s",
            "total_s",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ModelingValidationError("%s must be non-negative" % name)
            object.__setattr__(self, name, value)
        expected = self.kernel_body_s + self.launch_s + self.host_dispatch_s
        tolerance = 1.0e-15 + 1.0e-10 * max(expected, self.total_s)
        if abs(expected - self.total_s) > tolerance:
            raise ModelingValidationError(
                "native total_s must equal body + one launch + dispatch"
            )
        if (
            self.resident_ctas_per_sm <= 0
            or self.effective_resident_ctas_per_sm <= 0
            or self.effective_resident_ctas_per_sm > self.resident_ctas_per_sm
            or self.sm_count <= 0
        ):
            raise ModelingValidationError("native occupancy and SM count must be positive")
        validate_identifier(self.spatial_scope, "native spatial_scope")
        object.__setattr__(self, "provenance", freeze_attrs(self.provenance))


class _TimingResolver:
    def __init__(
        self,
        kernel: KernelIR,
        fusion: FusionReport,
        options: NativeModelOptions,
        oracle: Optional[Any],
        wave_context: Optional[WaveCostContext] = None,
    ) -> None:
        self.kernel = kernel
        self.fusion = fusion
        self.options = options
        self.oracle = oracle
        self.wave_context = wave_context
        if kernel.fusion_plans:
            if fusion.status is not FeasibilityStatus.PROVEN_FEASIBLE:
                raise NativeModelError(
                    "native fused cost binding requires a structurally proven FusionPlan; "
                    "got %s" % fusion.status.value
                )
            if options.cost_source == "phase_timing":
                raise NativeModelError(
                    "native fused modeling requires cost_source="
                    "'explicit_fused_static_timing' or a fusion-aware oracle"
                )
            if options.cost_source == "fusion_aware_oracle":
                if oracle is None or not bool(
                    getattr(oracle, "consumes_fusion_traffic", False)
                ) or not bool(getattr(oracle, "consumes_cache_traffic", False)):
                    raise NativeModelError(
                        "fusion-aware oracle must declare consumes_fusion_traffic=True "
                        "and consumes_cache_traffic=True"
                    )
                if not callable(getattr(oracle, "resolve_fused", None)):
                    raise NativeModelError(
                        "fusion-aware oracle must implement resolve_fused(phase,kernel,fusion)"
                    )

    def resolve(self, phase: Phase) -> Timing:
        if self.kernel.fusion_plans and self.options.cost_source == "fusion_aware_oracle":
            timing = self.oracle.resolve_fused(phase, self.kernel, self.fusion)
        else:
            timing = phase.timing
            if timing is None and self.oracle is not None:
                timing = self.oracle.resolve(phase)
        if not isinstance(timing, Timing):
            raise TimingResolutionError(
                "native phase %s needs Timing from its phase or selected cost resolver"
                % phase.name
            )
        small_tile = self.options.small_tile
        if (
            small_tile is not None
            and phase.name in small_tile.load_phases
            and timing.latency < small_tile.fixed_load_latency_s
        ):
            timing = replace(timing, latency=small_tile.fixed_load_latency_s)
        return timing


def _bind_oracle_context(
    oracle: Optional[Any], context: WaveCostContext
) -> Tuple[Optional[Any], str]:
    """Bind a cost oracle to one wave without mutating the original object."""

    if oracle is None:
        return None, "unavailable_no_oracle"
    binder = getattr(oracle, "bind_context", None)
    if not callable(binder):
        return oracle, "unavailable_no_bind_context"
    bound = binder(context)
    if bound is None:
        raise NativeModelError("cost oracle bind_context() returned None")
    if not callable(getattr(bound, "resolve", None)) and not callable(
        getattr(bound, "resolve_fused", None)
    ):
        raise NativeModelError(
            "cost oracle bind_context() must return a timing resolver"
        )
    if context.topology.kind == "cooperative_cluster" and not callable(
        getattr(bound, "resolve", None)
    ):
        raise NativeModelError(
            "cooperative_cluster bind_context() must return an oracle with "
            "resolve(phase)"
        )
    return bound, "bound"


def _zero(name: str) -> RegionEvaluation:
    return RegionEvaluation(
        name=name,
        kind="empty",
        total_s=0.0,
        first_s=0.0,
        steady_s=0.0,
        drain_s=0.0,
        strategy="empty",
        estimate_scope="exact",
    )


def _shift_trace(
    trace: Sequence[ResourceTraceEntry], delta_s: float
) -> Tuple[ResourceTraceEntry, ...]:
    return tuple(
        replace(item, offset_s=None if item.offset_s is None else item.offset_s + delta_s)
        for item in trace
    )


def _append_repeat(
    trace: Sequence[ResourceTraceEntry], repeat: TraceRepeat
) -> Tuple[ResourceTraceEntry, ...]:
    return tuple(replace(item, repeats=item.repeats + (repeat,)) for item in trace)


def _evaluate_phase(
    region: PhaseRegion,
    path: str,
    resolver: _TimingResolver,
) -> RegionEvaluation:
    timing = resolver.resolve(region.phase)
    trace = tuple(
        ResourceTraceEntry(
            path=path,
            phase=region.phase.name,
            resource=item.resource,
            offset_s=item.offset,
            service_time_s=item.service_time,
        )
        for item in timing.resources
    )
    return RegionEvaluation(
        name=region.name,
        kind="phase",
        total_s=timing.latency,
        first_s=timing.latency,
        steady_s=0.0,
        drain_s=0.0,
        strategy="bound_phase",
        estimate_scope="exact_static_or_oracle",
        resource_trace=trace,
    )


def _evaluate_sequence(
    region: SequenceRegion,
    path: str,
    context: "_EvaluationContext",
    inside_periodic_body: bool,
) -> RegionEvaluation:
    children = []
    trace = []
    offset = 0.0
    for child in region.children:
        result = _evaluate(
            child,
            "%s/%s" % (path, child.name),
            context,
            inside_periodic_body=inside_periodic_body,
        )
        children.append(result)
        trace.extend(_shift_trace(result.resource_trace, offset))
        offset += result.total_s
    total = sum(item.total_s for item in children)
    return RegionEvaluation(
        name=region.name,
        kind="sequence",
        total_s=total,
        first_s=sum(item.first_s for item in children),
        steady_s=sum(item.steady_s for item in children),
        drain_s=sum(item.drain_s for item in children),
        strategy="serial_sequence",
        estimate_scope="composed",
        children=tuple(children),
        resource_trace=tuple(trace),
    )


def _small_tile_dag(
    dag: PeriodicDAG,
    loop_name: str,
    context: "_EvaluationContext",
) -> PeriodicDAG:
    """Apply explicit latency/credit caps to the ordinary PeriodicDAG.

    This transformation preserves the frontend's existing token lifetimes.  A
    credit cap is rejected when no matching token exists; the policy never
    invents a FIFO or a data dependency.
    """

    policy = context.options.small_tile
    if policy is None:
        return dag
    phase_by_name = {phase.name: phase for phase in dag.phases}
    selected = tuple(name for name in policy.load_phases if name in phase_by_name)
    if not selected:
        return dag
    old_latency = {name: phase_by_name[name].latency for name in selected}
    new_latency = {
        name: max(old_latency[name], policy.fixed_load_latency_s)
        for name in selected
    }
    phases = tuple(
        replace(phase, latency=new_latency[phase.name])
        if phase.name in new_latency
        else phase
        for phase in dag.phases
    )
    dependencies = []
    for dependency in dag.dependencies:
        delay = dependency.min_delay
        if (
            dependency.source in new_latency
            and delay is not None
            and delay >= old_latency[dependency.source]
        ):
            delay += new_latency[dependency.source] - old_latency[dependency.source]
            dependency = replace(dependency, min_delay=delay)
        dependencies.append(dependency)
    tokens = []
    capped = []
    for token in dag.token_buffers:
        acquire_offset = token.acquire_offset
        release_offset = token.release_offset
        if (
            token.acquire in new_latency
            and acquire_offset >= old_latency[token.acquire]
        ):
            acquire_offset += new_latency[token.acquire] - old_latency[token.acquire]
        if (
            token.release in new_latency
            and release_offset is not None
            and release_offset >= old_latency[token.release]
        ):
            release_offset += new_latency[token.release] - old_latency[token.release]
        capacity = token.capacity
        if policy.outstanding_credits > 0 and token.acquire in selected:
            capacity = min(capacity, policy.outstanding_credits)
            capped.append((token.name, token.capacity, capacity))
        tokens.append(
            replace(
                token,
                acquire_offset=acquire_offset,
                release_offset=release_offset,
                capacity=capacity,
            )
        )
    if policy.outstanding_credits > 0:
        phases_with_token = {token.acquire for token in tokens if token.acquire in selected}
        missing = sorted(set(selected) - phases_with_token)
        if missing:
            raise NativeModelError(
                "small-tile outstanding credits require existing PipelineBuffer "
                "tokens for every selected load phase; missing=%s" % missing
            )
    context.small_tile_records.append(
        {
            "loop": loop_name,
            "load_phases": selected,
            "fixed_load_latency_s": policy.fixed_load_latency_s,
            "effective_load_latencies_s": tuple(
                sorted((name, new_latency[name]) for name in selected)
            ),
            "token_credit_caps": tuple(capped),
            "credit_path": "existing_periodic_dag_token_buffers",
        }
    )
    return PeriodicDAG(
        phases=phases,
        dependencies=tuple(dependencies),
        token_buffers=tuple(tokens),
        fixed_resource_orders=dag.fixed_resource_orders,
    )


def _axis_dag(axis: Any, context: "_EvaluationContext") -> PeriodicDAG:
    if axis.dag is not None:
        dag = axis.dag
        loop_name = axis.liveness_loop_name or "explicit_dag"
    else:
        dag = context.kernel.lower_periodic(axis.loop_name, oracle=context.resolver)
        loop_name = axis.loop_name
    return _small_tile_dag(dag, loop_name, context)


def _validate_periodic_body(
    region: LoopRegion,
    dag: PeriodicDAG,
    context: "_EvaluationContext",
) -> Mapping[str, Timing]:
    if contains_loop(region.body):
        raise NativeModelError(
            "periodic macro %s contains an inner loop; inline it into an explicit "
            "flat PeriodicDAG or keep the parent serial" % region.name
        )
    leaves = phase_regions(region.body)
    names = [item.phase.name for item in leaves]
    if len(names) != len(set(names)):
        raise NativeModelError("periodic macro body repeats a phase name")
    dag_names = [item.name for item in dag.phases]
    if set(names) != set(dag_names) or len(names) != len(dag_names):
        raise NativeModelError(
            "periodic macro body phases must exactly match its PeriodicDAG; "
            "body=%s dag=%s" % (sorted(names), sorted(dag_names))
        )
    timings = {item.phase.name: context.resolver.resolve(item.phase) for item in leaves}
    for phase in dag.phases:
        timing = timings[phase.name]
        tolerance = 1.0e-15 + 1.0e-10 * max(phase.latency, timing.latency)
        if abs(phase.latency - timing.latency) > tolerance:
            raise NativeModelError(
                "PeriodicDAG latency for %s does not match bound region timing"
                % phase.name
            )
        expected_resources = tuple(
            (item.resource, item.service_time, item.offset) for item in timing.resources
        )
        actual_resources = tuple(
            (item.resource, item.service_time, item.offset) for item in phase.resources
        )
        if expected_resources != actual_resources:
            raise NativeModelError(
                "PeriodicDAG resources for %s do not match bound region timing"
                % phase.name
            )
    return timings


def _macro_trace(
    path: str,
    region: LoopRegion,
    dag: PeriodicDAG,
    timings: Mapping[str, Timing],
    starts: Optional[Mapping[str, float]],
    ii: float,
) -> Tuple[ResourceTraceEntry, ...]:
    repeat = TraceRepeat(region.name, region.trip_count, ii)
    origin = 0.0 if starts is None else min(starts.values())
    result = []
    for phase in dag.phases:
        for resource in timings[phase.name].resources:
            offset = None
            ordering = "aggregate_only"
            if starts is not None:
                offset = starts[phase.name] - origin + resource.offset
                ordering = "known"
            result.append(
                ResourceTraceEntry(
                    path="%s/%s" % (path, phase.name),
                    phase=phase.name,
                    resource=resource.resource,
                    offset_s=offset,
                    service_time_s=resource.service_time,
                    repeats=(repeat,),
                    ordering=ordering,
                )
            )
    return tuple(result)


def _finite_trace(
    path: str,
    dag: PeriodicDAG,
    timings: Mapping[str, Timing],
    event_starts: Sequence[Tuple[str, int, float]],
) -> Tuple[ResourceTraceEntry, ...]:
    result = []
    for name, _iteration, start in event_starts:
        for resource in timings[name].resources:
            result.append(
                ResourceTraceEntry(
                    path="%s/%s" % (path, name),
                    phase=name,
                    resource=resource.resource,
                    offset_s=start + resource.offset,
                    service_time_s=resource.service_time,
                    ordering="known",
                )
            )
    return tuple(result)


_PERIODIC_SOLUTIONS = {}  # type: Dict[Any, Any]


def periodic_dag_digest(dag: PeriodicDAG) -> str:
    """Stable identity of one PeriodicDAG: its phases with costs, dependencies, tokens and fixed orders."""

    return hashlib.sha256(repr(dag).encode("utf-8")).hexdigest()


def _schedule_periodic_dag_cached(dag: PeriodicDAG, search_config: Any, topology_trials: int = 0,
                                  topology_seed: int = 0) -> Any:
    """The solver is deterministic in its inputs; work groups that share one PeriodicDAG search once.

    ``topology_trials > 0`` adds seeded zero-distance topological-order proposals
    (``periodic_proposals``) to the scheduler's own beam.
    """

    def solve() -> Any:
        if topology_trials <= 0:
            return schedule_periodic_dag(dag, search_config)
        from .periodic_proposals import topology_order_proposals

        return schedule_periodic_dag(
            dag, search_config, extra_orderings=topology_order_proposals(dag, topology_seed, topology_trials))

    try:
        key = (dag, search_config, topology_trials, topology_seed)
        hash(key)
    except TypeError:
        return solve()
    if key not in _PERIODIC_SOLUTIONS:
        if len(_PERIODIC_SOLUTIONS) >= 256:
            _PERIODIC_SOLUTIONS.clear()
        _PERIODIC_SOLUTIONS[key] = solve()
    return _PERIODIC_SOLUTIONS[key]


def _legacy_stage_terms(
    region: LoopRegion,
    dag: PeriodicDAG,
    context: "_EvaluationContext",
) -> Tuple[int, float, float]:
    policy = region.periodic_axis.legacy_stage_boundary
    if policy is None:
        raise NativeModelError(
            "boundary_policy='legacy_stage' requires an explicit "
            "PeriodicAxisIR.legacy_stage_boundary resource partition"
        )
    loop_name = (
        region.periodic_axis.loop_name
        or region.periodic_axis.liveness_loop_name
    )
    if loop_name is None:
        raise NativeModelError(
            "legacy_stage boundary requires a bound KernelIR loop with stages"
        )
    try:
        loop = context.kernel.periodic_loop(loop_name)
    except (KeyError, ModelingValidationError) as error:
        raise NativeModelError(
            "legacy_stage boundary cannot resolve KernelIR loop %r" % loop_name
        ) from error
    if loop.stages is None:
        raise NativeModelError(
            "legacy_stage boundary requires PeriodicLoopIR.stages"
        )
    totals = {}  # type: Dict[str, float]
    for phase in dag.phases:
        if phase.latency > 0.0 and not phase.resources:
            raise NativeModelError(
                "legacy_stage cannot classify positive-latency phase %s without "
                "an explicit resource reservation" % phase.name
            )
        phase_groups = set()
        for use in phase.resources:
            totals[use.resource] = totals.get(use.resource, 0.0) + use.service_time
            if use.resource in policy.memory_resources:
                phase_groups.add("memory")
            if use.resource in policy.compute_resources:
                phase_groups.add("compute")
        if len(phase_groups) > 1:
            raise NativeModelError(
                "legacy_stage phase %s couples memory and compute resources; "
                "the two-lane calibrated boundary is not applicable" % phase.name
            )
    classified = set(policy.memory_resources).union(policy.compute_resources)
    unknown = sorted(set(totals) - classified)
    missing = sorted(classified - set(totals))
    if unknown or missing:
        raise NativeModelError(
            "legacy_stage resource partition must exactly cover PeriodicDAG "
            "resources; unclassified=%s unused_declared=%s" % (unknown, missing)
        )
    memory_s = max(totals[name] for name in policy.memory_resources)
    compute_s = max(totals[name] for name in policy.compute_resources)
    if memory_s <= 0.0 or compute_s <= 0.0:
        raise NativeModelError("legacy_stage memory/compute service must be positive")
    return loop.stages, memory_s, compute_s


@dataclass(frozen=True)
class _FiniteConstraint:
    source: int
    target: int
    weight: float
    label: str


@dataclass(frozen=True)
class _FinitePeriodicBoundary:
    makespan_s: float
    warmup_s: float
    steady_s: float
    drain_s: float
    event_count: int
    constraint_count: int
    event_starts: Tuple[Tuple[str, int, float], ...] = field(default_factory=tuple)


def _finite_periodic_boundary(
    dag: PeriodicDAG,
    iterations: int,
    witness: Any,
    anchor_phase: str,
) -> _FinitePeriodicBoundary:
    """Schedule the finite source window for one periodic resource order.

    A periodic witness proves a cyclic steady schedule, but its affine
    ``phase + i*II`` expansion still carries predecessor constraints from
    iterations before zero and after the final iteration.  A real kernel
    starts with empty pipeline slots and exits only after the final consumer
    releases them.  This helper expands exactly the requested iterations,
    drops out-of-window recurrence/token/resource arcs, and computes the
    earliest source-faithful finite makespan for the witness' resource order.

    The expansion is linear in ``iterations * (phases + constraints)``.  It
    deliberately rejects a cyclic finite graph instead of silently applying
    the periodic macro approximation; current frontend GEMM/CTA2 pipelines
    lower to an acyclic source graph.
    """

    if iterations <= 0:
        raise NativeModelError("finite periodic boundary needs positive iterations")
    phases = tuple(dag.phases)
    phase_index = {phase.name: index for index, phase in enumerate(phases)}
    if anchor_phase not in phase_index:
        raise NativeModelError(
            "finite boundary anchor %r is not a PeriodicDAG phase" % anchor_phase
        )
    phase_count = len(phases)
    event_count = iterations * phase_count

    def node(iteration: int, phase: int) -> int:
        return iteration * phase_count + phase

    constraints = []  # type: List[_FiniteConstraint]

    for dependency in dag.dependencies:
        source_phase = phase_index[dependency.source]
        target_phase = phase_index[dependency.target]
        delay = dependency.min_delay
        if delay is None:
            delay = phases[source_phase].latency
        for source_iteration in range(iterations):
            target_iteration = source_iteration + dependency.iteration_distance
            if 0 <= target_iteration < iterations:
                constraints.append(
                    _FiniteConstraint(
                        node(source_iteration, source_phase),
                        node(target_iteration, target_phase),
                        float(delay),
                        dependency.name
                        or "%s->%s" % (dependency.source, dependency.target),
                    )
                )

    for token in dag.token_buffers:
        acquire_phase = phase_index[token.acquire]
        release_phase = phase_index[token.release]
        release_offset = token.release_offset
        if release_offset is None:
            release_offset = phases[release_phase].latency
        lifetime_weight = (
            token.acquire_offset + token.minimum_residence - release_offset
        )
        reuse_weight = release_offset - token.acquire_offset
        for iteration in range(iterations):
            constraints.append(
                _FiniteConstraint(
                    node(iteration, acquire_phase),
                    node(iteration, release_phase),
                    lifetime_weight,
                    "token:%s:lifetime" % token.name,
                )
            )
            reuse_iteration = iteration + token.capacity
            if reuse_iteration < iterations:
                constraints.append(
                    _FiniteConstraint(
                        node(iteration, release_phase),
                        node(reuse_iteration, acquire_phase),
                        reuse_weight,
                        "token:%s:reuse" % token.name,
                    )
                )

    uses_by_resource = {}  # type: Dict[str, Dict[str, Any]]
    for phase in phases:
        for use in phase.resources:
            uses_by_resource.setdefault(use.resource, {})[phase.name] = use
    resource_orders = dict(witness.resource_orders)
    for resource, uses in uses_by_resource.items():
        order = tuple(resource_orders.get(resource, tuple()))
        if set(order) != set(uses) or len(order) != len(uses):
            raise NativeModelError(
                "finite boundary witness does not order every %s user" % resource
            )
        for position, current_name in enumerate(order):
            following_name = order[(position + 1) % len(order)]
            current_phase = phase_index[current_name]
            following_phase = phase_index[following_name]
            wraps = 1 if position + 1 == len(order) else 0
            distance = (
                wraps
                + phases[following_phase].iteration_offset
                - phases[current_phase].iteration_offset
            )
            current_use = uses[current_name]
            following_use = uses[following_name]
            weight = (
                current_use.offset
                + current_use.service_time
                - following_use.offset
            )
            for source_iteration in range(iterations):
                target_iteration = source_iteration + distance
                if 0 <= target_iteration < iterations:
                    constraints.append(
                        _FiniteConstraint(
                            node(source_iteration, current_phase),
                            node(target_iteration, following_phase),
                            weight,
                            "resource:%s:%s->%s"
                            % (resource, current_name, following_name),
                        )
                    )

    outgoing = [[] for _ in range(event_count)]  # type: List[List[_FiniteConstraint]]
    indegree = [0] * event_count
    for edge in constraints:
        outgoing[edge.source].append(edge)
        indegree[edge.target] += 1
    ready = deque(index for index, degree in enumerate(indegree) if degree == 0)
    starts = [0.0] * event_count
    visited = 0
    while ready:
        source = ready.popleft()
        visited += 1
        for edge in outgoing[source]:
            candidate = starts[source] + edge.weight
            if candidate > starts[edge.target]:
                starts[edge.target] = candidate
            indegree[edge.target] -= 1
            if indegree[edge.target] == 0:
                ready.append(edge.target)
    if visited != event_count:
        raise NativeModelError(
            "finite periodic expansion contains a constraint cycle; this source "
            "boundary mode currently requires an acyclic unrolled graph"
        )

    makespan = max(
        starts[node(iteration, phase_index_value)] + phase.latency
        for iteration in range(iterations)
        for phase_index_value, phase in enumerate(phases)
    )
    anchor = phase_index[anchor_phase]
    first_anchor = starts[node(0, anchor)]
    last_anchor = starts[node(iterations - 1, anchor)]
    warmup = first_anchor
    steady = max(last_anchor - first_anchor, 0.0)
    drain = max(makespan - last_anchor, 0.0)
    tolerance = 1.0e-15 + 1.0e-10 * max(makespan, warmup + steady + drain)
    if abs(makespan - (warmup + steady + drain)) > tolerance:
        raise NativeModelError("finite periodic boundary decomposition is inconsistent")
    return _FinitePeriodicBoundary(
        makespan_s=makespan,
        warmup_s=warmup,
        steady_s=steady,
        drain_s=drain,
        event_count=event_count,
        constraint_count=len(constraints),
        event_starts=tuple(
            (phase.name, iteration, starts[node(iteration, phase_index_value)])
            for iteration in range(iterations)
            for phase_index_value, phase in enumerate(phases)
        ),
    )


def _evaluate_periodic_macro(
    region: LoopRegion,
    path: str,
    context: "_EvaluationContext",
    prologue: RegionEvaluation,
    epilogue: RegionEvaluation,
) -> RegionEvaluation:
    axis = region.periodic_axis
    dag = _axis_dag(axis, context)
    previous_dag = context.periodic_dags.get(region.name)
    if previous_dag is not None and previous_dag != dag:
        raise NativeModelError(
            "recursive evaluation produced conflicting PeriodicDAGs for region %s"
            % region.name
        )
    context.periodic_dags[region.name] = dag
    timings = _validate_periodic_body(region, dag, context)
    body_serial = _evaluate(
        region.body,
        "%s/body" % path,
        context,
        inside_periodic_body=True,
    )
    bounds = compute_lower_bounds(dag)
    necessary_ii = bounds.overall
    mode = context.options.ii_mode if axis.ii_mode == "inherit" else axis.ii_mode
    selection = select_steady_ii(
        necessary_ii,
        mode,
        solve_periodic=lambda: _schedule_periodic_dag_cached(
            dag, axis.search_config, context.options.periodic_topology_trials, context.options.periodic_topology_seed),
    )
    starts = None if selection.witness is None else selection.witness.phase_starts
    if starts is None:
        span = body_serial.total_s
        estimate_scope = "necessary_steady_bound_with_serial_fill_drain"
        diagnostics = [
            "resource_ii is a necessary steady lower bound and has no schedule witness",
            "first/drain use the explicit serial body span",
        ]
    else:
        origin = min(starts.values())
        span = max(
            starts[phase.name] - origin + timings[phase.name].latency
            for phase in dag.phases
        )
        estimate_scope = "constructive_periodic_schedule"
        diagnostics = []
    boundary_approx = False
    trace_ii = selection.ii
    selected_ii = selection.ii
    finite_event_starts = tuple()
    if context.options.boundary_policy == "legacy_stage":
        stages, memory_s, compute_s = _legacy_stage_terms(region, dag, context)
        depth = max(stages - 1, 0)
        if stages <= 1:
            # The calibrated stage-1 equation deliberately has no load/compute
            # overlap.  A slower II remains compatible with a constructive
            # witness found at the PeriodicDAG's smaller feasible II.
            selected_ii = memory_s + compute_s
            trace_ii = selected_ii
            first_core = 0.0
            steady = region.trip_count * selected_ii
            drain_core = 0.0
        else:
            calibrated_ii = max(memory_s, compute_s)
            tolerance = 1.0e-15 + 1.0e-10 * max(calibrated_ii, selection.ii)
            if abs(selection.ii - calibrated_ii) > tolerance:
                raise NativeModelError(
                    "legacy_stage assumes steady II=max(memory,compute), but "
                    "PeriodicDAG selected II %.9g differs from calibrated %.9g; "
                    "a recurrence/credit/order constraint is outside this policy"
                    % (selection.ii, calibrated_ii)
                )
            first_core = depth * memory_s
            steady = max(region.trip_count - depth, 0) * calibrated_ii
            drain_core = depth * compute_s
        first = prologue.total_s + first_core
        drain = drain_core + epilogue.total_s
        estimate_scope = "calibrated_legacy_stage_boundary"
        diagnostics.extend(
            (
                "legacy_stage uses explicit memory/compute resource sets and "
                "PeriodicLoopIR.stages",
                "legacy_stage preserves the calibrated (stages-1) fill/drain "
                "equation; it is not a finite PeriodicDAG makespan",
            )
        )
        strategy = "periodic_macro_legacy_stage"
    elif context.options.boundary_policy == "finite_witness":
        if selection.witness is None:
            raise NativeModelError(
                "boundary_policy='finite_witness' requires a constructive "
                "periodic best/worst witness"
            )
        if region.trip_count > context.options.max_finite_iterations:
            raise NativeModelError(
                "finite boundary for %s needs %d iterations, exceeding "
                "max_finite_iterations=%d"
                % (
                    region.name,
                    region.trip_count,
                    context.options.max_finite_iterations,
                )
            )
        anchor = axis.boundary_anchor_phase
        if anchor is None:
            raise NativeModelError(
                "boundary_policy='finite_witness' requires "
                "PeriodicAxisIR.boundary_anchor_phase"
            )
        finite = _finite_periodic_boundary(
            dag,
            region.trip_count,
            selection.witness,
            anchor,
        )
        first_core = finite.warmup_s
        steady = finite.steady_s
        drain_core = finite.drain_s
        first = prologue.total_s + first_core
        drain = drain_core + epilogue.total_s
        finite_event_starts = finite.event_starts
        estimate_scope = "constructive_finite_source_schedule"
        diagnostics.extend(
            (
                "finite_witness expands the real iteration window with empty "
                "initial token slots and no post-loop resource wrap",
                "warmup ends at the first %s start; drain runs from the final "
                "%s start through all phase completions" % (anchor, anchor),
                "finite expansion scheduled %d events and %d constraints"
                % (finite.event_count, finite.constraint_count),
            )
        )
        strategy = "periodic_finite_witness"
    else:
        boundary_approx = region.trip_count <= context.options.unroll_threshold
        if boundary_approx:
            diagnostics.append(
                "finite periodic unroll is not implemented; the constructive steady "
                "witness is clipped to this short-loop boundary"
            )
        first_core = min(span, selection.ii)
        drain_core = span - first_core
        steady = (region.trip_count - 1) * selection.ii
        first = prologue.total_s + first_core
        drain = drain_core + epilogue.total_s
        strategy = (
            "periodic_macro_boundary_approx"
            if boundary_approx
            else "periodic_macro"
        )
    trace = list(prologue.resource_trace)
    if finite_event_starts:
        # Finite source schedule: one service interval per (phase, iteration)
        # at its real start; no periodic repeat descriptor.
        trace.extend(
            _shift_trace(
                _finite_trace(path, dag, timings, finite_event_starts),
                prologue.total_s,
            )
        )
    else:
        trace.extend(
            _shift_trace(
                _macro_trace(path, region, dag, timings, starts, trace_ii),
                prologue.total_s,
            )
        )
    trace.extend(
        _shift_trace(
            epilogue.resource_trace,
            prologue.total_s + first_core + steady + drain_core,
        )
    )
    witness_loop_name = None
    witness_phase_starts = tuple()
    if starts is not None and axis.liveness_loop_name is not None:
        liveness_loop = context.kernel.periodic_loop(axis.liveness_loop_name)
        liveness_names = {phase.name for phase in liveness_loop.phases}
        if liveness_names != set(starts):
            raise NativeModelError(
                "liveness loop %s phases do not exactly match the periodic witness"
                % axis.liveness_loop_name
            )
        origin = min(starts.values())
        witness_loop_name = axis.liveness_loop_name
        witness_phase_starts = tuple(
            sorted((name, float(value) - origin) for name, value in starts.items())
        )
    return RegionEvaluation(
        name=region.name,
        kind="loop",
        total_s=first + steady + drain,
        first_s=first,
        steady_s=steady,
        drain_s=drain,
        strategy=strategy,
        estimate_scope=(
            "%s_boundary_approx" % estimate_scope
            if boundary_approx
            else estimate_scope
        ),
        trip_count=region.trip_count,
        selected_ii_s=selected_ii,
        resource_ii_s=bounds.resource_ii,
        recurrence_ii_s=bounds.recurrence_ii,
        credit_ii_s=bounds.credit_ii,
        best_ii_s=selection.best_ii,
        worst_ii_s=selection.worst_ii,
        ii_scope=selection.label,
        witness_loop_name=witness_loop_name,
        witness_phase_starts=witness_phase_starts,
        periodic_dag_digest=periodic_dag_digest(dag),
        children=(prologue, body_serial, epilogue),
        resource_trace=tuple(trace),
        diagnostics=tuple(diagnostics),
        finite_event_starts=finite_event_starts,
    )


def _evaluate_serial_loop(
    region: LoopRegion,
    path: str,
    context: "_EvaluationContext",
    prologue: RegionEvaluation,
    epilogue: RegionEvaluation,
    inline: bool,
) -> RegionEvaluation:
    if region.trip_count == 0:
        total = prologue.total_s + epilogue.total_s
        return RegionEvaluation(
            name=region.name,
            kind="loop",
            total_s=total,
            first_s=prologue.total_s,
            steady_s=0.0,
            drain_s=epilogue.total_s,
            strategy="zero_trip",
            estimate_scope="exact",
            trip_count=0,
            children=(prologue, epilogue),
            resource_trace=prologue.resource_trace
            + _shift_trace(epilogue.resource_trace, prologue.total_s),
        )
    body = _evaluate(
        region.body,
        "%s/body" % path,
        context,
        inside_periodic_body=False,
    )
    first = prologue.total_s + body.first_s
    steady = body.steady_s + (region.trip_count - 1) * body.total_s
    drain = body.drain_s + epilogue.total_s
    trace = list(prologue.resource_trace)
    if inline:
        for index in range(region.trip_count):
            trace.extend(
                _shift_trace(
                    body.resource_trace,
                    prologue.total_s + index * body.total_s,
                )
            )
        strategy = "inline_serial"
    else:
        trace.extend(
            _shift_trace(
                _append_repeat(
                    body.resource_trace,
                    TraceRepeat(region.name, region.trip_count, body.total_s),
                ),
                prologue.total_s,
            )
        )
        strategy = "serial_repeat_macro"
    trace.extend(
        _shift_trace(
            epilogue.resource_trace,
            prologue.total_s + region.trip_count * body.total_s,
        )
    )
    return RegionEvaluation(
        name=region.name,
        kind="loop",
        total_s=first + steady + drain,
        first_s=first,
        steady_s=steady,
        drain_s=drain,
        strategy=strategy,
        estimate_scope="exact_serial_region_semantics",
        trip_count=region.trip_count,
        children=(prologue, body, epilogue),
        resource_trace=tuple(trace),
        diagnostics=(
            "serial parent permits a lossless repeat macro"
            if not inline
            else "trip_count is within the explicit inline threshold",
        ),
    )


def _evaluate_loop(
    region: LoopRegion,
    path: str,
    context: "_EvaluationContext",
    inside_periodic_body: bool,
) -> RegionEvaluation:
    if inside_periodic_body:
        raise NativeModelError(
            "loop %s is nested inside a periodic parent body; a macro would hide "
            "shared-resource arbitration and this v0 does not flatten it" % region.name
        )
    prologue = (
        _zero("%s_prologue" % region.name)
        if region.prologue is None
        else _evaluate(
            region.prologue,
            "%s/prologue" % path,
            context,
            inside_periodic_body=False,
        )
    )
    epilogue = (
        _zero("%s_epilogue" % region.name)
        if region.epilogue is None
        else _evaluate(
            region.epilogue,
            "%s/epilogue" % path,
            context,
            inside_periodic_body=False,
        )
    )
    if region.trip_count == 0:
        return _evaluate_serial_loop(
            region,
            path,
            context,
            prologue,
            epilogue,
            inline=True,
        )
    if region.periodic_axis is not None:
        if region.periodic_axis.summary_policy == "inline":
            raise NativeModelError(
                "periodic summary_policy='inline' requires a finite periodic "
                "scheduler, which native v0 has not implemented"
            )
        return _evaluate_periodic_macro(region, path, context, prologue, epilogue)
    serial_inline = region.trip_count <= context.options.unroll_threshold
    if serial_inline and region.trip_count > context.options.max_inline_iterations:
        raise NativeModelError(
            "loop %s exceeds max_inline_iterations" % region.name
        )
    return _evaluate_serial_loop(
        region,
        path,
        context,
        prologue,
        epilogue,
        inline=serial_inline,
    )


@dataclass
class _EvaluationContext:
    kernel: KernelIR
    fusion: FusionReport
    options: NativeModelOptions
    resolver: _TimingResolver
    small_tile_records: list = field(default_factory=list)
    periodic_dags: dict = field(default_factory=dict)


def _evaluate(
    region: Any,
    path: str,
    context: _EvaluationContext,
    *,
    inside_periodic_body: bool,
) -> RegionEvaluation:
    if isinstance(region, PhaseRegion):
        return _evaluate_phase(region, path, context.resolver)
    if isinstance(region, SequenceRegion):
        return _evaluate_sequence(region, path, context, inside_periodic_body)
    if isinstance(region, LoopRegion):
        return _evaluate_loop(region, path, context, inside_periodic_body)
    raise ModelingValidationError("native evaluator expected a recursive region")


def evaluate_region(
    kernel: KernelIR,
    region: Any,
    options: Optional[NativeModelOptions] = None,
    oracle: Optional[Any] = None,
) -> RegionEvaluation:
    """Bind costs and recursively evaluate one region tree."""

    if not isinstance(kernel, KernelIR):
        raise ModelingValidationError("evaluate_region expects KernelIR")
    if not is_region(region):
        raise ModelingValidationError("evaluate_region expects a recursive region")
    if options is None:
        options = NativeModelOptions()
    if not isinstance(options, NativeModelOptions):
        raise ModelingValidationError("evaluate_region options must be NativeModelOptions")
    _validate_small_tile_targets(region, options)
    fusion = analyze_fusion(kernel)
    resolver = _TimingResolver(kernel, fusion, options, oracle)
    context = _EvaluationContext(kernel, fusion, options, resolver)
    result = _evaluate(region, region.name, context, inside_periodic_body=False)
    _validate_small_tile_token_bindings(context, options)
    return result


def _grid_product(shape: Sequence[Any], label: str) -> int:
    result = 1
    for extent in shape:
        if not isinstance(extent, int) or isinstance(extent, bool) or extent <= 0:
            raise NativeModelError("native %s must have concrete positive extents" % label)
        result *= extent
    return result


def _validate_static_topology_grids(
    launch: Any, topology: LaunchTopology
) -> str:
    """Validate whether grids count CTAs or cooperative logical work units."""

    raw_work_grid = tuple(launch.work_grid)
    raw_physical_grid = tuple(launch.physical_grid)
    raw_cluster = tuple(launch.cluster)
    rank = max(len(raw_work_grid), len(raw_physical_grid), len(raw_cluster))

    def pad(values: Tuple[Any, ...]) -> Tuple[Any, ...]:
        return values + (1,) * (rank - len(values))

    work_grid = pad(raw_work_grid)
    physical_grid = pad(raw_physical_grid)
    cluster = pad(raw_cluster)
    if topology.kind == "cooperative_cluster":
        expected = tuple(
            work_extent * cluster_extent
            for work_extent, cluster_extent in zip(work_grid, cluster)
        )
        semantics = "logical_supertile_work_grid_cluster_expanded_cta_grid"
    else:
        # single/resident work units are CTAs. Ordinary clusters group those
        # CTA work units for placement but do not combine them into a supertile.
        expected = work_grid
        semantics = "cta_work_grid_equals_physical_cta_grid"
    if physical_grid != expected:
        raise NativeModelError(
            "native %s topology requires physical_grid=%r for work_grid=%r and "
            "cluster=%r; got %r"
            % (
                topology.kind,
                expected,
                work_grid,
                cluster,
                physical_grid,
            )
        )
    return semantics


def _validate_small_tile_targets(
    region: Any, options: NativeModelOptions
) -> None:
    policy = options.small_tile
    if policy is None or not policy.load_phases:
        return
    owners_by_name = {}  # type: Dict[str, set]
    for leaf in phase_regions(region):
        owners_by_name.setdefault(leaf.phase.name, set()).add(leaf.phase.owner)
    missing = sorted(set(policy.load_phases) - set(owners_by_name))
    ambiguous = sorted(
        name
        for name in policy.load_phases
        if len(owners_by_name.get(name, set())) > 1
    )
    if missing or ambiguous:
        raise NativeModelError(
            "small-tile load phase names must resolve uniquely in RegionIR; "
            "missing=%s ambiguous=%s" % (missing, ambiguous)
        )


def _validate_small_tile_token_bindings(
    context: "_EvaluationContext", options: NativeModelOptions
) -> None:
    policy = options.small_tile
    if policy is None or policy.outstanding_credits <= 0:
        return
    bound = {
        phase
        for record in context.small_tile_records
        for phase in record["load_phases"]
    }
    missing = sorted(set(policy.load_phases) - bound)
    if missing:
        raise NativeModelError(
            "small-tile outstanding credits apply only to phases in a bound "
            "PeriodicDAG with existing PipelineBuffer tokens; missing=%s" % missing
        )


def _decision_trace(result: RegionEvaluation, path: str = "") -> Tuple[Any, ...]:
    current_path = result.name if not path else "%s/%s" % (path, result.name)
    items = [
        {
            "path": current_path,
            "kind": result.kind,
            "strategy": result.strategy,
            "estimate_scope": result.estimate_scope,
            "trip_count": result.trip_count,
            "ii_scope": result.ii_scope,
            "selected_ii_s": result.selected_ii_s,
        }
    ]
    for child in result.children:
        items.extend(_decision_trace(child, current_path))
    return tuple(items)


def _periodic_results(result: RegionEvaluation) -> Tuple[RegionEvaluation, ...]:
    items = []
    if result.strategy.startswith("periodic_macro"):
        items.append(result)
    for child in result.children:
        items.extend(_periodic_results(child))
    return tuple(items)


def _liveness_witnesses(
    result: RegionEvaluation,
    policy: str,
) -> Tuple[Mapping[str, Any], Tuple[str, ...]]:
    witnesses = {}
    missing = []
    for item in _periodic_results(result):
        if item.witness_loop_name is None or not item.witness_phase_starts:
            missing.append(item.name)
            continue
        witness = SimpleNamespace(
            ii=item.selected_ii_s,
            phase_starts=dict(item.witness_phase_starts),
        )
        previous = witnesses.get(item.witness_loop_name)
        if previous is not None and (
            previous.ii != witness.ii
            or previous.phase_starts != witness.phase_starts
        ):
            raise NativeModelError(
                "conflicting recursive witnesses for loop %s"
                % item.witness_loop_name
            )
        witnesses[item.witness_loop_name] = witness
    if missing and policy == "require_witness":
        raise NativeModelError(
            "native full liveness requires constructive witnesses for every "
            "periodic macro; missing=%s. Use periodic_best/worst and set "
            "PeriodicAxisIR.liveness_loop_name, or explicitly select "
            "liveness_policy='report_static'." % sorted(missing)
        )
    return witnesses, tuple(sorted(missing))


def _collect_region_bindings(region: Any, path: str = "") -> Tuple[Any, ...]:
    current_path = region.name if not path else "%s/%s" % (path, region.name)
    result = []
    if isinstance(region, LoopRegion):
        if region.periodic_axis is not None:
            binding = (
                region.periodic_axis.loop_name
                or region.periodic_axis.liveness_loop_name
            )
            result.append((binding, region, current_path))
        if region.prologue is not None:
            result.extend(
                _collect_region_bindings(region.prologue, current_path + "/prologue")
            )
        result.extend(_collect_region_bindings(region.body, current_path + "/body"))
        if region.epilogue is not None:
            result.extend(
                _collect_region_bindings(region.epilogue, current_path + "/epilogue")
            )
    elif isinstance(region, SequenceRegion):
        for child in region.children:
            result.extend(_collect_region_bindings(child, current_path))
    return tuple(result)


def _validate_full_region_coverage(
    kernel: KernelIR,
    launch: Any,
    region: Any,
) -> Tuple[Tuple[Any, ...], Tuple[str, ...]]:
    """Fail closed unless RegionIR is an exact full-launch phase definition."""

    selected_loops = tuple(launch.periodic_loops)
    loops_by_name = {loop.name: loop for loop in selected_loops}
    if len(loops_by_name) != len(selected_loops):
        raise NativeModelError("selected launch has ambiguous periodic loop names")
    selected_phases = {
        (phase.owner, phase.name): phase
        for loop in selected_loops
        for phase in loop.phases
    }
    all_kernel_phases = {
        (phase.owner, phase.name): phase
        for item in kernel.launches
        for loop in item.periodic_loops
        for phase in loop.phases
    }
    all_kernel_owners = {
        loop.owner for item in kernel.launches for loop in item.periodic_loops
    }
    counts = {}  # type: Dict[Tuple[str, str], int]
    standalone = []
    for leaf in phase_regions(region):
        phase = leaf.phase
        key = (phase.owner, phase.name)
        expected = all_kernel_phases.get(key)
        if expected is None:
            if phase.owner in all_kernel_owners:
                raise NativeModelError(
                    "RegionIR invents unknown phase %s/%s inside a KernelIR loop owner"
                    % key
                )
            standalone.append("%s/%s" % key)
            continue
        if key not in selected_phases:
            raise NativeModelError(
                "RegionIR references kernel-owned phase %s/%s from another launch"
                % key
            )
        if phase != expected:
            raise NativeModelError(
                "RegionIR phase %s/%s mutates the selected launch definition" % key
            )
        counts[key] = counts.get(key, 0) + 1

    missing = sorted(
        "%s/%s" % key for key in selected_phases if counts.get(key, 0) == 0
    )
    duplicates = sorted(
        "%s/%s" % key for key, count in counts.items() if count != 1
    )
    if missing or duplicates:
        raise NativeModelError(
            "full RegionIR must cover every selected-launch phase exactly once; "
            "missing=%s duplicate_or_repeated=%s" % (missing, duplicates)
        )

    bindings = {}  # type: Dict[str, Tuple[LoopRegion, str]]
    for loop_name, loop_region, path in _collect_region_bindings(region):
        if loop_name is None:
            raise NativeModelError(
                "full periodic RegionIR %s needs a KernelIR loop binding" % path
            )
        if loop_name not in loops_by_name:
            raise NativeModelError(
                "RegionIR periodic binding %r is not in selected launch %s"
                % (loop_name, launch.name)
            )
        if loop_name in bindings:
            raise NativeModelError(
                "KernelIR loop %s is bound by multiple RegionIR loops" % loop_name
            )
        bindings[loop_name] = (loop_region, path)

    records = []
    for loop in selected_loops:
        iterations = loop.iterations
        if (
            not isinstance(iterations, int)
            or isinstance(iterations, bool)
            or iterations < 0
        ):
            raise NativeModelError(
                "native full coverage requires concrete non-negative KernelIR "
                "iterations for loop %s" % loop.name
            )
        binding = bindings.get(loop.name)
        if binding is None:
            if iterations != 1:
                raise NativeModelError(
                    "KernelIR loop %s has %d iterations but no PeriodicAxisIR binding"
                    % (loop.name, iterations)
                )
            records.append(
                {
                    "loop": loop.name,
                    "kernel_iterations": iterations,
                    "region_trip_count": 1,
                    "binding": "one_shot_structural",
                    "phase_count": len(loop.phases),
                }
            )
            continue
        loop_region, path = binding
        if loop_region.trip_count != iterations:
            raise NativeModelError(
                "RegionIR trip_count=%d disagrees with KernelIR loop %s iterations=%d"
                % (loop_region.trip_count, loop.name, iterations)
            )
        expected_keys = {(phase.owner, phase.name) for phase in loop.phases}
        body_keys = []
        for leaf in phase_regions(loop_region.body):
            key = (leaf.phase.owner, leaf.phase.name)
            if key in selected_phases:
                body_keys.append(key)
        if len(body_keys) != len(set(body_keys)) or set(body_keys) != expected_keys:
            raise NativeModelError(
                "PeriodicAxisIR binding for %s must contain exactly that loop's "
                "phases in its body" % loop.name
            )
        records.append(
            {
                "loop": loop.name,
                "kernel_iterations": iterations,
                "region_trip_count": loop_region.trip_count,
                "binding": path,
                "phase_count": len(loop.phases),
            }
        )
    return tuple(records), tuple(sorted(standalone))


def _cluster_oracle_contract(
    topology: LaunchTopology,
    oracle: Optional[Any],
    launch: Any,
) -> Tuple[Tuple[str, Any], ...]:
    """Validate cluster semantics without inventing a generic DSM channel."""

    if topology.kind == "ordinary_cluster":
        raise NativeModelError(
            "ordinary_cluster is typed as independent CTA work tiles with "
            "placement/L1.5-group locality, but native v0 has no cluster issue/tail "
            "executor. It is therefore rejected rather than treated as resident "
            "CTAs or a cooperative supertile"
        )
    if topology.kind != "cooperative_cluster":
        return tuple()
    if oracle is None or not bool(
        getattr(oracle, "supports_cooperative_cluster", False)
    ):
        raise NativeModelError(
            "cooperative_cluster requires a cluster-aware cost oracle; native v0 "
            "will not reuse the resident-CTA resource bound"
        )
    if not callable(getattr(oracle, "bind_context", None)):
        raise NativeModelError(
            "cooperative_cluster requires a callable bind_context(context); "
            "declaration flags alone cannot price cluster timing"
        )
    static_phases = sorted(
        "%s.%s" % (loop.name, phase.name)
        for loop in launch.periodic_loops
        for phase in loop.phases
        if phase.timing is not None
    )
    if static_phases:
        raise NativeModelError(
            "cooperative_cluster native v1 requires all selected periodic phases "
            "to have timing=None so the context-bound oracle consumes cluster "
            "operand/barrier/issue costs; static Phase.timing found for %s"
            % static_phases
        )
    expected_size = getattr(oracle, "cooperative_cluster_size", None)
    if expected_size is not None and expected_size != topology.cluster_size:
        raise NativeModelError(
            "cluster-aware oracle cooperative_cluster_size does not match launch"
        )
    required = (
        "consumes_cluster_operand_plan",
        "consumes_pair_tensor_throughput",
        "consumes_per_sm_local_smem",
        "consumes_cluster_issue_latency",
    )
    missing = [name for name in required if not bool(getattr(oracle, name, False))]
    if topology.peer_smem_resources and not bool(
        getattr(oracle, "consumes_peer_smem_traffic", False)
    ):
        missing.append("consumes_peer_smem_traffic")
    if missing:
        raise NativeModelError(
            "cluster-aware oracle is missing required declarations %s" % sorted(missing)
        )
    return (
        ("operand_plan", topology.operand_plan),
        ("per_sm_local_smem", True),
        ("pair_tensor_throughput", "oracle_consumed"),
        ("cluster_barrier_s", topology.cluster_barrier_s),
        ("cluster_issue_s", topology.cluster_issue_s),
        ("tma_multicast", topology.tma_multicast),
        ("peer_smem_resources", topology.peer_smem_resources),
        ("generic_dsm_bandwidth", "not_created"),
    )


def _wave_context(
    kind: str,
    topology: LaunchTopology,
    physical_sm_count: int,
    logical_slot_count: int,
    active_slots: int,
    residency: int,
    work_units: int,
) -> WaveCostContext:
    active_physical = active_slots
    if topology.kind == "cooperative_cluster":
        active_physical *= topology.cluster_size
    return WaveCostContext(
        wave_kind=kind,
        physical_sm_count=physical_sm_count,
        logical_slot_count=logical_slot_count,
        active_physical_sms=active_physical,
        active_logical_slots=active_slots,
        resident_work_units_per_slot=residency,
        work_units=work_units,
        topology=topology,
    )


def _evaluate_wave_region(
    kernel: KernelIR,
    region: Any,
    fusion: FusionReport,
    options: NativeModelOptions,
    oracle: Optional[Any],
    wave_context: WaveCostContext,
) -> Tuple[RegionEvaluation, _EvaluationContext, str]:
    bound_oracle, binding = _bind_oracle_context(oracle, wave_context)
    resolver = _TimingResolver(
        kernel,
        fusion,
        options,
        bound_oracle,
        wave_context=wave_context,
    )
    context = _EvaluationContext(kernel, fusion, options, resolver)
    result = _evaluate(region, region.name, context, inside_periodic_body=False)
    return result, context, binding


def _resident_group_bound(
    result: RegionEvaluation,
    residency: int,
    topology: LaunchTopology,
    multi_cta_policy: str,
) -> Tuple[float, str]:
    if topology.kind == "cooperative_cluster":
        return result.total_s, "cooperative_cluster_cluster_aware_oracle"
    if residency <= 1:
        return result.total_s, "single_cta_constructive_or_region_scope"
    if topology.kind != "resident":
        raise NativeModelError(
            "multiple work units per SM require LaunchTopology(kind='resident')"
        )
    if multi_cta_policy == "reject":
        raise NativeModelError(
            "native v0 cannot constructively arbitrate multiple resident CTAs; "
            "set multi_cta_policy='resource_bound' for a labelled nonconstructive "
            "per-SM resource bound, or use the calibrated legacy GEMM executor"
        )
    service_bound = max(
        (
            service * residency
            for _resource, service in result.resource_service_s
        ),
        default=0.0,
    )
    return (
        max(result.total_s, service_bound),
        "resident_group_resource_bound_nonconstructive",
    )


def _constructive_resident_group(
    result: RegionEvaluation,
    context: _EvaluationContext,
    residency: int,
    options: NativeModelOptions,
    oracle: Optional[Any],
) -> Tuple[float, str, Mapping[str, Any]]:
    """Construct one finite independent-CTA group from a product-DAG witness."""

    if residency <= 1:
        return result.total_s, "single_cta_constructive_or_region_scope", {}
    if residency > 4:
        raise NativeModelError(
            "constructive resident scheduler supports at most four CTAs per SM"
        )
    if options.boundary_policy != "periodic_witness":
        raise NativeModelError(
            "constructive resident scheduling currently requires "
            "boundary_policy='periodic_witness'; legacy_stage is a calibrated "
            "two-lane equation, not a product-DAG finite witness"
        )
    periodic = _periodic_results(result)
    if len(periodic) != 1 or periodic[0] is not result:
        raise NativeModelError(
            "constructive resident v1 requires the root RegionIR to be exactly one "
            "flat periodic macro with optional serial prologue/epilogue"
        )
    dag = context.periodic_dags.get(result.name)
    if dag is None:
        raise NativeModelError("constructive resident path lost its PeriodicDAG")
    if any(phase.iteration_offset != 0 for phase in dag.phases):
        raise NativeModelError(
            "constructive resident finite clipping currently requires zero "
            "PeriodicDAG iteration offsets"
        )
    base_config = options.resident_schedule
    if base_config is None:
        raise NativeModelError("constructive resident policy needs a schedule config")
    if base_config.resident_ctas != residency:
        if base_config.fixed_orders:
            raise NativeModelError(
                "tail residency differs from ResidentScheduleConfig and fixed_orders "
                "cannot be resized implicitly"
            )
        config = replace(base_config, resident_ctas=residency)
    else:
        config = base_config

    dag_resources = {
        use.resource for phase in dag.phases for use in phase.resources
    }
    declared_resources = set(config.resource_scopes)
    if declared_resources != dag_resources:
        raise NativeModelError(
            "constructive resident scheduling requires an explicit scope for every "
            "PeriodicDAG resource; missing=%s extra=%s"
            % (
                sorted(dag_resources - declared_resources),
                sorted(declared_resources - dag_resources),
            )
        )
    external = tuple(
        sorted(
            resource
            for resource, scope in config.resource_scopes.items()
            if scope is ResourceScope.EXTERNAL_PRICED
        )
    )
    external_names = set()
    for resource in dag_resources:
        normalized = resource.lower().replace("_", "").replace("-", "")
        if normalized.startswith(("ddr", "dram", "l2")):
            external_names.add(resource)
    improperly_shared = sorted(
        resource
        for resource in external_names
        if config.resource_scopes[resource] is not ResourceScope.EXTERNAL_PRICED
    )
    if improperly_shared:
        raise NativeModelError(
            "device-global DDR/L2 resources must be external_priced for a resident "
            "product DAG; resources=%s" % improperly_shared
        )
    if external:
        if not callable(getattr(oracle, "bind_context", None)) or not bool(
            getattr(oracle, "consumes_external_priced_resources", False)
        ):
            raise NativeModelError(
                "external_priced resident resources require a context-bound oracle "
                "that declares consumes_external_priced_resources=True"
            )
        declared_external = set(
            getattr(oracle, "external_priced_resources", tuple())
        )
        if not set(external).issubset(declared_external):
            raise NativeModelError(
                "context-bound oracle does not declare every external-priced "
                "resource; missing=%s" % sorted(set(external) - declared_external)
            )

    schedule = schedule_resident_ctas(dag, config)
    witness = (
        schedule.best
        if options.multi_cta_policy == "constructive_best"
        else schedule.worst
    )
    phase_latency = {phase.name: phase.latency for phase in dag.phases}
    starts = witness.phase_starts
    origin = min(starts.values())
    trip_count = result.trip_count
    if trip_count is None or trip_count <= 0:
        raise NativeModelError(
            "constructive resident periodic region needs positive trip_count"
        )
    core_span = max(
        start
        - origin
        + (trip_count - 1) * witness.group_period
        + phase_latency[phase_name]
        for (_cta, phase_name), start in starts.items()
    )
    prologue = result.children[0].total_s
    epilogue = result.children[2].total_s
    # Product scheduling currently covers only the periodic core.  Serializing
    # each CTA's explicit one-shot boundary is a conservative constructive
    # extension and is named as such in provenance.
    group_s = residency * prologue + core_span + residency * epilogue
    metadata = {
        "resident_ctas": residency,
        "selected_endpoint": options.multi_cta_policy,
        "candidate": witness.candidate,
        "group_period_s": witness.group_period,
        "effective_work_interval_s": witness.effective_work_interval,
        "finite_core_span_s": core_span,
        "boundary_extension": "serial_per_cta_prologue_epilogue",
        "search_scope": schedule.search_scope,
        "search_complete": schedule.search_complete,
        "candidates_explored": schedule.candidates_explored,
        "external_priced_resources": schedule.external_priced_resources,
        "phase_starts": tuple(sorted(witness.phase_starts.items())),
        "resource_orders": tuple(sorted(witness.resource_orders.items(), key=repr)),
    }
    scope = "resident_group_%s_product_dag" % options.multi_cta_policy
    return group_s, scope, metadata


def model_native(
    kernel: KernelIR,
    region: Any,
    arch: Any,
    options: Optional[NativeModelOptions] = None,
    oracle: Optional[Any] = None,
) -> NativeModelResult:
    """Run recursive per-work cost, occupancy/grid waves, and one launch."""

    if not isinstance(kernel, KernelIR):
        raise ModelingValidationError("model_native expects KernelIR")
    if not is_region(region):
        raise ModelingValidationError("model_native expects a recursive region")
    if options is None:
        options = NativeModelOptions()
    if not isinstance(options, NativeModelOptions):
        raise ModelingValidationError("model_native options must be NativeModelOptions")
    _validate_small_tile_targets(region, options)
    launches = [item for item in kernel.launches if item.name == options.launch_name]
    if len(launches) != 1:
        raise NativeModelError(
            "native executor requires exactly one launch named %r" % options.launch_name
        )
    launch = launches[0]
    coverage_records, standalone_phases = _validate_full_region_coverage(
        kernel, launch, region
    )
    cluster_size = _grid_product(launch.cluster, "cluster")
    sm_count = getattr(arch, "sm_count", None)
    if not isinstance(sm_count, int) or isinstance(sm_count, bool) or sm_count <= 0:
        raise NativeModelError("architecture must provide positive integer sm_count")
    if options.resident_ctas_per_sm is not None:
        residency = options.resident_ctas_per_sm
        residency_source = "explicit_option"
    elif isinstance(launch.residency, int) and not isinstance(launch.residency, bool):
        residency = launch.residency
        residency_source = "launch_ir"
    else:
        residency = 1
        residency_source = "conservative_auto_fallback"
    if (
        options.multi_cta_policy.startswith("constructive_")
        and options.resident_schedule.resident_ctas != residency
    ):
        raise NativeModelError(
            "ResidentScheduleConfig.resident_ctas must match configured launch "
            "residency; tail residency is resized only after this full-wave check"
        )
    work_units = _grid_product(launch.work_grid, "work_grid")
    physical_programs = _grid_product(launch.physical_grid, "physical_grid")
    if physical_programs < work_units:
        raise NativeModelError(
            "native v0 does not infer a persistent scheduler when physical_grid "
            "contains fewer programs than work_grid"
        )
    topology = options.launch_topology
    topology_source = "explicit_option"
    if cluster_size > 1:
        if topology is None:
            raise NativeModelError(
                "cluster_size>1 requires an explicit LaunchTopology distinguishing "
                "ordinary_cluster from cooperative_cluster"
            )
        if topology.cluster_size != cluster_size:
            raise NativeModelError(
                "LaunchTopology.cluster_size disagrees with LaunchIR.cluster"
            )
        if topology.kind not in ("ordinary_cluster", "cooperative_cluster"):
            raise NativeModelError(
                "clustered LaunchIR requires ordinary_cluster or cooperative_cluster"
            )
        logical_slot_count = sm_count // cluster_size
        if logical_slot_count <= 0:
            raise NativeModelError("cluster is larger than the architecture")
    else:
        if topology is not None and topology.kind in (
            "ordinary_cluster",
            "cooperative_cluster",
        ):
            raise NativeModelError(
                "cluster topology requires LaunchIR.cluster_size>1"
            )
        logical_slot_count = sm_count
    wave_capacity = logical_slot_count * residency
    active_work_units = min(work_units, physical_programs, wave_capacity)
    if active_work_units <= 0:
        raise NativeModelError("native launch has no active work units")
    full_waves = work_units // wave_capacity
    tail = work_units % wave_capacity
    waves = full_waves + (1 if tail else 0)
    active_slots = min(logical_slot_count, active_work_units)
    effective_residency = min(
        residency,
        (active_work_units + active_slots - 1) // active_slots,
    )
    if topology is None:
        topology = (
            LaunchTopology.resident()
            if effective_residency > 1
            else LaunchTopology.single()
        )
        topology_source = "derived_from_effective_residency"
    if topology.kind == "single" and effective_residency > 1:
        raise NativeModelError(
            "LaunchTopology(kind='single') conflicts with effective residency > 1"
        )
    if topology.kind == "cooperative_cluster" and residency != 1:
        raise NativeModelError(
            "native cooperative_cluster currently requires one logical work unit "
            "resident per cluster slot"
        )
    spatial_grid_semantics = _validate_static_topology_grids(launch, topology)
    active_physical_ctas = active_work_units * (
        topology.cluster_size
        if topology.kind == "cooperative_cluster"
        else 1
    )
    if active_physical_ctas > physical_programs:
        raise NativeModelError(
            "native active physical CTA count exceeds physical_grid launch count"
        )
    cluster_contract = _cluster_oracle_contract(topology, oracle, launch)

    fusion = analyze_fusion(kernel)
    if full_waves > 0:
        primary_slots = logical_slot_count
        primary_residency = residency
        primary_units = wave_capacity
        primary_kind = "full"
    else:
        primary_units = tail
        primary_slots = min(logical_slot_count, primary_units)
        primary_residency = min(
            residency,
            (primary_units + primary_slots - 1) // primary_slots,
        )
        primary_kind = "underfilled"
    primary_wave_context = _wave_context(
        primary_kind,
        topology,
        sm_count,
        logical_slot_count,
        primary_slots,
        primary_residency,
        primary_units,
    )
    result, context, primary_binding = _evaluate_wave_region(
        kernel,
        region,
        fusion,
        options,
        oracle,
        primary_wave_context,
    )
    _validate_small_tile_token_bindings(context, options)
    witnesses, missing_witnesses = _liveness_witnesses(
        result, options.liveness_policy
    )
    liveness = analyze_liveness(
        kernel,
        arch=arch,
        resident_ctas_per_sm=effective_residency,
        witnesses=witnesses or None,
        oracle=context.resolver,
    )
    if (
        options.reject_proven_infeasible
        and liveness.status is FeasibilityStatus.PROVEN_INFEASIBLE
        and bool(getattr(liveness, "guard_eligible", False))
    ):
        raise NativeModelError("native launch is proven storage-infeasible: %s" % (
            "; ".join(liveness.diagnostics),
        ))

    tail_units = tail
    tail_factor = float(tail_units) / float(wave_capacity)
    resident_schedule_records = []
    if options.multi_cta_policy.startswith("constructive_"):
        if topology.kind != "resident" or cluster_size != 1:
            raise NativeModelError(
                "constructive resident scheduling is only valid for "
                "LaunchTopology(kind='resident') with cluster_size=1"
            )
        resident_group_s, spatial_scope, schedule_record = (
            _constructive_resident_group(
                result,
                context,
                primary_residency,
                options,
                oracle,
            )
        )
        if schedule_record:
            resident_schedule_records.append(("primary", schedule_record))
    else:
        resident_group_s, spatial_scope = _resident_group_bound(
            result,
            primary_residency,
            topology,
            options.multi_cta_policy,
        )
    tail_context = None
    tail_binding = "not_applicable_no_tail"
    tail_context_records = tuple()
    if tail == 0:
        tail_wave_s = 0.0
        tail_active_slots = logical_slot_count
        tail_residency = residency
    elif full_waves == 0:
        tail_wave_s = resident_group_s
        tail_context = primary_wave_context
        tail_binding = primary_binding
        tail_active_slots = primary_slots
        tail_residency = primary_residency
    else:
        tail_active_slots = min(logical_slot_count, tail)
        tail_residency = min(
            residency,
            (tail + tail_active_slots - 1) // tail_active_slots,
        )
        tail_kind = (
            "underfilled"
            if tail_active_slots < logical_slot_count
            else "tail"
        )
        tail_context = _wave_context(
            tail_kind,
            topology,
            sm_count,
            logical_slot_count,
            tail_active_slots,
            tail_residency,
            tail,
        )
        tail_result, tail_evaluation_context, tail_binding = _evaluate_wave_region(
            kernel,
            region,
            fusion,
            options,
            oracle,
            tail_context,
        )
        tail_context_records = tuple(tail_evaluation_context.small_tile_records)
        if options.multi_cta_policy.startswith("constructive_"):
            tail_wave_s, _tail_scope, schedule_record = (
                _constructive_resident_group(
                    tail_result,
                    tail_evaluation_context,
                    tail_residency,
                    options,
                    oracle,
                )
            )
            if schedule_record:
                resident_schedule_records.append(("tail", schedule_record))
        else:
            tail_wave_s, _tail_scope = _resident_group_bound(
                tail_result,
                tail_residency,
                topology,
                options.multi_cta_policy,
            )

    underfilled_adjustment = "disabled"
    if (
        tail > 0
        and tail_active_slots < logical_slot_count
        and options.small_tile is not None
        and options.small_tile.adjusts_underfilled_wave
    ):
        tail_wave_s = (
            tail_wave_s / options.small_tile.underfilled_efficiency
            + options.small_tile.underfilled_admission_s
        )
        underfilled_adjustment = "explicit_small_tile_policy"

    head_penalty = 1.0
    if options.wave_policy == "legacy_calibrated" and result.first_s > 0.0:
        head_penalty = 1.0 + 0.1 * result.first_s / max(
            resident_group_s, 1.0e-30
        )
    if options.wave_policy == "legacy_calibrated":
        if full_waves > 0:
            body_s = (
                resident_group_s * head_penalty
                + max(full_waves - 1, 0) * resident_group_s
                + tail_wave_s
            )
        else:
            body_s = tail_wave_s * head_penalty
    else:
        body_s = full_waves * resident_group_s + tail_wave_s
    total_s = body_s + options.kernel_launch_s + options.host_dispatch_s
    grid = NativeGridResult(
        work_units=work_units,
        physical_programs=physical_programs,
        active_work_units=active_work_units,
        active_physical_ctas=active_physical_ctas,
        waves=waves,
        full_waves=full_waves,
        tail_work_units=tail_units,
        tail_factor=tail_factor,
    )
    if kernel.fusion_plans and options.cost_source == "explicit_fused_static_timing":
        fusion_binding = "user_asserted_consumed_by_static_timing"
        cache_binding = "user_asserted_consumed_by_static_timing"
    elif kernel.fusion_plans and options.cost_source == "fusion_aware_oracle":
        fusion_binding = "oracle_declared_consumed"
        cache_binding = "oracle_declared_consumed"
    else:
        fusion_binding = "not_applicable"
        cache_binding = "not_connected_native_v0"
    dynamic_phases = tuple(
        sorted(
            phase.name
            for loop in launch.periodic_loops
            for phase in loop.phases
            if phase.timing is None
        )
    )
    if not dynamic_phases:
        context_rebind_effect = "static_phase_timings_override_oracle"
    elif primary_binding == "bound":
        context_rebind_effect = "context_bound_for_dynamic_phases"
    else:
        context_rebind_effect = "dynamic_phases_resolved_without_context_rebind"
    small_tile_payload = None
    if options.small_tile is not None:
        small_tile_payload = {
            "load_phases": options.small_tile.load_phases,
            "fixed_load_latency_s": options.small_tile.fixed_load_latency_s,
            "outstanding_credits": options.small_tile.outstanding_credits,
            "underfilled_admission_s": (
                options.small_tile.underfilled_admission_s
            ),
            "underfilled_efficiency": (
                options.small_tile.underfilled_efficiency
            ),
        }
    provenance = {
        "executor": "native_recursive_v0",
        "cost_source": options.cost_source,
        "fusion_status": fusion.status.value,
        "fusion_traffic_binding": fusion_binding,
        "cache_traffic_binding": cache_binding,
        "residency_source": residency_source,
        "configured_resident_ctas_per_sm": residency,
        "effective_resident_ctas_per_sm": effective_residency,
        "boundary_policy": options.boundary_policy,
        "max_finite_iterations": options.max_finite_iterations,
        "wave_policy": options.wave_policy,
        "wave_policy_equation": (
            "legacy_first_wave_head_penalty"
            if options.wave_policy == "legacy_calibrated"
            else "native_homogeneous_work_units"
        ),
        "head_penalty": head_penalty,
        "wave_capacity_work_units": wave_capacity,
        "logical_slot_count": logical_slot_count,
        "launch_topology_source": topology_source,
        "launch_topology": {
            "kind": topology.kind,
            "cluster_size": topology.cluster_size,
            "tma_multicast": topology.tma_multicast,
            "operand_plan": topology.operand_plan,
            "peer_smem_resources": topology.peer_smem_resources,
        },
        "spatial_grid_semantics": spatial_grid_semantics,
        "cluster_cost_contract": cluster_contract,
        "primary_wave_context": primary_wave_context,
        "tail_wave_context": tail_context,
        "primary_context_binding": primary_binding,
        "tail_context_binding": tail_binding,
        "context_rebind_effect": context_rebind_effect,
        "context_dynamic_phases": dynamic_phases,
        "small_tile_policy": small_tile_payload,
        "small_tile_periodic_bindings": tuple(context.small_tile_records),
        "tail_small_tile_periodic_bindings": tail_context_records,
        "underfilled_adjustment": underfilled_adjustment,
        "multi_cta_policy": options.multi_cta_policy,
        "resident_schedule_records": tuple(resident_schedule_records),
        "resident_witness_liveness": (
            "single_cta_witness_scaled_by_residency_not_cross_cta_phase_sweep"
            if resident_schedule_records
            else "not_applicable"
        ),
        "spatial_scope": spatial_scope,
        "liveness_policy": options.liveness_policy,
        "liveness_witness_loops": tuple(sorted(witnesses)),
        "liveness_static_fallback_regions": missing_witnesses,
        "device_global_resource_arbitration": "must_be_consumed_by_cost_binding",
        "launch_accounting": "exactly_once",
        "ii_mode_default": options.ii_mode,
        "region_decisions": _decision_trace(result),
        "region_loop_coverage": coverage_records,
        "region_standalone_phases": standalone_phases,
        "units": "SI_seconds",
    }
    return NativeModelResult(
        region=result,
        per_work_unit_s=result.total_s,
        resident_group_s=resident_group_s,
        tail_wave_s=tail_wave_s,
        kernel_body_s=body_s,
        launch_s=options.kernel_launch_s,
        host_dispatch_s=options.host_dispatch_s,
        total_s=total_s,
        resident_ctas_per_sm=residency,
        effective_resident_ctas_per_sm=effective_residency,
        sm_count=sm_count,
        grid=grid,
        spatial_scope=spatial_scope,
        liveness=liveness,
        fusion=fusion,
        provenance=provenance,
    )


__all__ = [
    "ContextBoundCostResolver",
    "FusionAwareCostResolver",
    "NativeGridResult",
    "NativeModelError",
    "NativeModelOptions",
    "NativeModelResult",
    "RegionEvaluation",
    "ResourceTraceEntry",
    "TraceRepeat",
    "evaluate_region",
    "model_native",
]
