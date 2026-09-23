"""Public analysis result types (one ``AnalysisResult``, versioned snapshot)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from typing import Any, Dict, Mapping, Optional, Tuple

from ..errors import ModelingValidationError
from .traffic import LEVEL_FIELDS, Traffic, TrafficShare, traffic_shares

RESULT_SCHEMA_VERSION = "tilesight.analysis_result/2"
MODEL_VERSION = "generic_analyze_v2_rw_split"


def traffic_to_dict(traffic: Traffic) -> Dict[str, float]:
    return traffic.to_dict()


def traffic_from_dict(data: Mapping[str, Any]) -> Traffic:
    return Traffic.from_dict(data)


def scale_traffic(traffic: Traffic, factor: float) -> Traffic:
    return traffic.scale(factor)


def traffic_close(a: Traffic, b: Traffic, tolerance: float = 1e-6) -> bool:
    return a.close_to(b, tolerance)


@dataclass(frozen=True)
class TrafficInterval:
    """Lower/upper traffic of one access; width is zero when exact."""

    low: Traffic
    high: Traffic
    truncation_source: Optional[str] = None
    sampling_confidence_interval: Optional[Tuple[float, float]] = None
    structural_approximation: Optional[str] = None

    @property
    def exact(self) -> bool:
        return self.truncation_source is None and traffic_close(self.low, self.high)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "low": traffic_to_dict(self.low),
            "high": traffic_to_dict(self.high),
            "truncation_source": self.truncation_source,
            "sampling_confidence_interval": self.sampling_confidence_interval,
            "structural_approximation": self.structural_approximation,
        }


@dataclass(frozen=True)
class CostGroupKey:
    work_group_id: str
    region_site: str
    wave_kind: str
    binding_context_id: str

    def __post_init__(self) -> None:
        if self.wave_kind not in ("full", "tail", "not_applicable"):
            raise ModelingValidationError("CostGroupKey.wave_kind=%r is invalid" % self.wave_kind)

    def to_dict(self) -> Dict[str, str]:
        return {
            "work_group_id": self.work_group_id,
            "region_site": self.region_site,
            "wave_kind": self.wave_kind,
            "binding_context_id": self.binding_context_id,
        }


@dataclass(frozen=True)
class CostGroup:
    """Cost of *one execution* of an op under one binding context."""

    key: CostGroupKey
    count: int
    traffic_per_execution: Traffic
    service_s: Tuple[Tuple[str, float], ...]
    completion_latency_s: float
    latency_extra_s: float
    traffic_source: str
    provenance: Tuple[Tuple[str, Any], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.count, int) or self.count < 0:
            raise ModelingValidationError("CostGroup.count must be a non-negative integer")
        object.__setattr__(self, "service_s", tuple((str(k), float(v)) for k, v in self.service_s))
        for name in ("completion_latency_s", "latency_extra_s"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ModelingValidationError("CostGroup.%s must be finite and non-negative" % name)

    @property
    def service_total_s(self) -> float:
        return sum(value for _name, value in self.service_s)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key.to_dict(),
            "count": self.count,
            "traffic_per_execution": traffic_to_dict(self.traffic_per_execution),
            "service_s": list(self.service_s),
            "completion_latency_s": self.completion_latency_s,
            "latency_extra_s": self.latency_extra_s,
            "traffic_source": self.traffic_source,
            "provenance": _jsonable(dict(self.provenance)),
        }


@dataclass(frozen=True)
class OpResult:
    """One primitive summed over all its executions.

    ``executed_*`` is the work/bytes of the full tile every time the op runs.
    ``useful_*`` is the part that lies inside the *valid range*, and the two
    kinds of op derive that range differently:

    * ``Compute``: the valid range of its output as propagated from the plain
      inputs it reads (tail tiles, broadcast axes, in-place old values); the
      causal/tail work-group fraction applies on top.
    * ``Load``/``Store``: the mask of the **tensor slice** it moves
      (``TileAccess.valid_shape`` × loop/tail fractions).  A store's
      ``useful_bytes`` therefore says how many destination elements lie inside
      the tensor -- it does *not* certify that every value written was itself
      computed from valid inputs.  Writing a full 8-element tile into an
      8-element tensor is 8 useful elements even when the producing compute
      had only 5 useful elements; into a 5-element tensor it is 5.
    """

    path: str
    op_id: str
    kind: str
    launch: str
    region_site: str
    count: int
    useful_flops: float
    executed_flops: float
    useful_bytes: float
    executed_bytes: float
    traffic_total: Traffic
    cost_groups: Tuple[CostGroup, ...]
    access_ids: Tuple[str, ...] = field(default_factory=tuple)
    traffic_interval_total: Optional[TrafficInterval] = None
    traffic_shares: Tuple[TrafficShare, ...] = field(default_factory=tuple)
    provenance: Tuple[Tuple[str, Any], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "traffic_shares", tuple(self.traffic_shares))
        total = sum(group.count for group in self.cost_groups)
        if total != self.count:
            raise ModelingValidationError(
                "OpResult(%s): count %d != sum of cost-group counts %d" % (self.path, self.count, total)
            )
        expected = Traffic()
        for group in self.cost_groups:
            expected = expected.add(scale_traffic(group.traffic_per_execution, float(group.count)))
        if not traffic_close(expected, self.traffic_total):
            raise ModelingValidationError(
                "OpResult(%s): traffic_total does not equal sum(count x traffic_per_execution)" % self.path
            )

    @property
    def cost_group_index(self) -> Dict[str, CostGroup]:
        return {"%s|%s|%s" % (g.key.work_group_id, g.key.region_site, g.key.wave_kind): g for g in self.cost_groups}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "op_id": self.op_id,
            "kind": self.kind,
            "launch": self.launch,
            "region_site": self.region_site,
            "count": self.count,
            "useful_flops": self.useful_flops,
            "executed_flops": self.executed_flops,
            "useful_bytes": self.useful_bytes,
            "executed_bytes": self.executed_bytes,
            "traffic_total": traffic_to_dict(self.traffic_total),
            "traffic_interval_total": None if self.traffic_interval_total is None else self.traffic_interval_total.to_dict(),
            "cost_groups": [g.to_dict() for g in self.cost_groups],
            "access_ids": list(self.access_ids),
            "traffic_shares": [x.to_dict() for x in self.traffic_shares],
            "provenance": _jsonable(dict(self.provenance)),
        }

    def share(self, field: str, scope: str) -> Optional[TrafficShare]:
        for item in self.traffic_shares:
            if item.field == field and item.scope == scope:
                return item
        return None

    def breakdown(self, scope: str = "launch") -> Tuple[Tuple[str, str, Optional[float], Optional[float]], ...]:
        """(level, direction, cumulative bytes or None if unknown, share within ``scope``) rows."""

        rows = []
        for level, read, write in LEVEL_FIELDS:
            for direction, name in (("read", read), ("write", write)):
                item = self.share(name, scope)
                rows.append((level, direction, self.traffic_total.value(name),
                             None if item is None else item.share))
        return tuple(rows)


@dataclass(frozen=True)
class IIComponents:
    resource_ii_s: Optional[float]
    recurrence_ii_s: Optional[float]
    credit_ii_s: Optional[float]
    selected_ii_s: Optional[float]
    best_ii_s: Optional[float]
    worst_ii_s: Optional[float]
    scope: str
    mode: str
    search_complete: Optional[bool] = None
    witness: Optional[Tuple[Tuple[str, float], ...]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "resource_ii_s": self.resource_ii_s,
            "recurrence_ii_s": self.recurrence_ii_s,
            "credit_ii_s": self.credit_ii_s,
            "selected_ii_s": self.selected_ii_s,
            "best_ii_s": self.best_ii_s,
            "worst_ii_s": self.worst_ii_s,
            "scope": self.scope,
            "mode": self.mode,
            "search_complete": self.search_complete,
            "witness": None if self.witness is None else list(self.witness),
        }


@dataclass(frozen=True)
class ServiceInterval:
    """One resource service interval inside a region timeline."""

    path: str
    op_id: str
    resource: str
    offset_s: Optional[float]
    service_time_s: float
    repeats: Tuple[Tuple[str, int, float], ...] = field(default_factory=tuple)
    ordering: str = "known"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path, "op_id": self.op_id, "resource": self.resource,
            "offset_s": self.offset_s, "service_time_s": self.service_time_s,
            "repeats": [list(r) for r in self.repeats], "ordering": self.ordering,
        }


@dataclass(frozen=True)
class CompletionInterval:
    """Start/completion of one op execution inside a region timeline."""

    path: str
    op_id: str
    actor: str
    start_s: Optional[float]
    completion_latency_s: float
    repeats: Tuple[Tuple[str, int, float], ...] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path, "op_id": self.op_id, "actor": self.actor,
            "start_s": self.start_s, "completion_latency_s": self.completion_latency_s,
            "repeats": [list(r) for r in self.repeats],
        }


@dataclass(frozen=True)
class RegionGroupResult:
    """One loop/region evaluated under one work group."""

    work_group_id: str
    trip_count: Optional[int]
    total_s: float
    first_s: float
    steady_s: float
    drain_s: float
    strategy: str
    estimate_scope: str
    ii: Optional[IIComponents]
    boundary_policy: str
    service_trace: Tuple[ServiceInterval, ...] = field(default_factory=tuple)
    completion_trace: Tuple[CompletionInterval, ...] = field(default_factory=tuple)
    diagnostics: Tuple[str, ...] = field(default_factory=tuple)
    traffic_total: Traffic = field(default_factory=Traffic)
    periodic_dag_digest: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "traffic_total": traffic_to_dict(self.traffic_total),
            "periodic_dag_digest": self.periodic_dag_digest,
            "work_group_id": self.work_group_id,
            "trip_count": self.trip_count,
            "total_s": self.total_s, "first_s": self.first_s,
            "steady_s": self.steady_s, "drain_s": self.drain_s,
            "strategy": self.strategy, "estimate_scope": self.estimate_scope,
            "ii": None if self.ii is None else self.ii.to_dict(),
            "boundary_policy": self.boundary_policy,
            "service_trace": [x.to_dict() for x in self.service_trace],
            "completion_trace": [x.to_dict() for x in self.completion_trace],
            "diagnostics": list(self.diagnostics),
        }


@dataclass(frozen=True)
class RegionResult:
    path: str
    kind: str  # launch_body | loop
    launch: str
    summary_group_id: str
    total_s: float
    first_s: float
    steady_s: float
    drain_s: float
    trip_count: Optional[int]
    ii: Optional[IIComponents]
    estimate_scope: str
    boundary_policy: str
    groups: Tuple[RegionGroupResult, ...]
    children: Tuple[str, ...] = field(default_factory=tuple)
    diagnostics: Tuple[str, ...] = field(default_factory=tuple)
    traffic_total: Traffic = field(default_factory=Traffic)

    def group(self, group_id: str) -> RegionGroupResult:
        for item in self.groups:
            if item.work_group_id == group_id:
                return item
        raise KeyError(group_id)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path, "kind": self.kind, "launch": self.launch,
            "summary_group_id": self.summary_group_id,
            "total_s": self.total_s, "first_s": self.first_s,
            "steady_s": self.steady_s, "drain_s": self.drain_s,
            "trip_count": self.trip_count,
            "ii": None if self.ii is None else self.ii.to_dict(),
            "estimate_scope": self.estimate_scope,
            "boundary_policy": self.boundary_policy,
            "groups": [g.to_dict() for g in self.groups],
            "children": list(self.children),
            "diagnostics": list(self.diagnostics),
            "traffic_total": traffic_to_dict(self.traffic_total),
        }


@dataclass(frozen=True)
class WorkGroupSummary:
    group_id: str
    count: int
    per_unit_s: float
    trip_counts: Tuple[Tuple[str, int], ...]

    def to_dict(self) -> Dict[str, Any]:
        return {"group_id": self.group_id, "count": self.count,
                "per_unit_s": self.per_unit_s, "trip_counts": list(self.trip_counts)}


@dataclass(frozen=True)
class LaunchResult:
    name: str
    kernel_body_s: float
    launch_overhead_s: float
    host_dispatch_s: float
    total_s: float
    work_units: int
    physical_ctas: int
    sm_count: int
    active_sms: int
    residency: int
    residency_source: str
    slots: int
    waves: int
    tail_work_units: int
    spatial_scope: str
    scheduler: str
    groups: Tuple[WorkGroupSummary, ...]
    slot_completion_s: Tuple[float, ...]
    critical_work_unit: Optional[int]
    critical_group_id: Optional[str]
    provenance: Tuple[Tuple[str, Any], ...] = field(default_factory=tuple)
    traffic_total: Traffic = field(default_factory=Traffic)

    def __post_init__(self) -> None:
        expected = self.kernel_body_s + self.launch_overhead_s + self.host_dispatch_s
        if abs(expected - self.total_s) > 1e-12 + 1e-9 * max(expected, self.total_s):
            raise ModelingValidationError("LaunchResult.total_s must equal body + overhead + dispatch")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "kernel_body_s": self.kernel_body_s,
            "launch_overhead_s": self.launch_overhead_s,
            "host_dispatch_s": self.host_dispatch_s,
            "total_s": self.total_s,
            "work_units": self.work_units,
            "physical_ctas": self.physical_ctas,
            "sm_count": self.sm_count,
            "active_sms": self.active_sms,
            "residency": self.residency,
            "residency_source": self.residency_source,
            "slots": self.slots,
            "waves": self.waves,
            "tail_work_units": self.tail_work_units,
            "spatial_scope": self.spatial_scope,
            "scheduler": self.scheduler,
            "groups": [g.to_dict() for g in self.groups],
            "slot_completion_s": list(self.slot_completion_s),
            "critical_work_unit": self.critical_work_unit,
            "critical_group_id": self.critical_group_id,
            "provenance": _jsonable(dict(self.provenance)),
            "traffic_total": traffic_to_dict(self.traffic_total),
        }


@dataclass(frozen=True)
class ProgramResult:
    name: str
    total_s: float
    launch_order: Tuple[str, ...]
    inter_launch_gap_s: float
    scope: str
    aggregation: str
    measurement_scope: str
    traffic_total: Traffic = field(default_factory=Traffic)
    edges: Tuple[Tuple[str, str, float], ...] = field(default_factory=tuple)

    def gap_before(self, launch: str) -> float:
        return sum(gap for _source, target, gap in self.edges if target == launch)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name, "total_s": self.total_s,
            "launch_order": list(self.launch_order),
            "inter_launch_gap_s": self.inter_launch_gap_s,
            "edges": [list(item) for item in self.edges],
            "scope": self.scope, "aggregation": self.aggregation,
            "measurement_scope": self.measurement_scope,
            "traffic_total": traffic_to_dict(self.traffic_total),
        }


@dataclass(frozen=True)
class CacheAccessReport:
    access_id: str
    op_path: str
    tensor: str
    mode: str
    request_count: float
    l1_5_hit_rate: Optional[float]
    l2_served_rate: Optional[float]
    ddr_miss_rate: Optional[float]
    traffic_per_request: Traffic
    interval_per_request: TrafficInterval
    scope: str
    source: str = "cache_model"
    calibrated: bool = False
    model_traffic_per_request: Optional[Traffic] = None
    model_rates: Optional[Tuple[Tuple[str, float], ...]] = None
    rates_source: str = "cache_model"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "access_id": self.access_id, "op_path": self.op_path, "tensor": self.tensor,
            "mode": self.mode, "request_count": self.request_count,
            "l1_5_hit_rate": self.l1_5_hit_rate, "l2_served_rate": self.l2_served_rate,
            "ddr_miss_rate": self.ddr_miss_rate,
            "traffic_per_request": traffic_to_dict(self.traffic_per_request),
            "interval_per_request": self.interval_per_request.to_dict(),
            "scope": self.scope, "source": self.source, "calibrated": self.calibrated,
            "model_traffic_per_request": (
                None if self.model_traffic_per_request is None else traffic_to_dict(self.model_traffic_per_request)
            ),
            "model_rates": None if self.model_rates is None else dict(self.model_rates),
            "rates_source": self.rates_source,
        }


@dataclass(frozen=True)
class CacheReport:
    launch: str
    mode: str
    scenario: str
    backend: str
    exact: bool
    reduction_fidelity: str
    histogram_digest: str
    problem_digest: str
    per_access: Tuple[CacheAccessReport, ...]
    per_tensor: Tuple[Tuple[str, Dict[str, float]], ...]
    assumptions: Tuple[str, ...]
    sampled_requests: int
    represented_requests: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "launch": self.launch, "mode": self.mode, "scenario": self.scenario,
            "backend": self.backend, "exact": self.exact,
            "reduction_fidelity": self.reduction_fidelity,
            "histogram_digest": self.histogram_digest,
            "problem_digest": self.problem_digest,
            "per_access": [x.to_dict() for x in self.per_access],
            "per_tensor": [[k, dict(v)] for k, v in self.per_tensor],
            "assumptions": list(self.assumptions),
            "sampled_requests": self.sampled_requests,
            "represented_requests": self.represented_requests,
        }


@dataclass(frozen=True)
class Diagnostics:
    unsupported: Tuple[str, ...] = field(default_factory=tuple)
    approximations: Tuple[str, ...] = field(default_factory=tuple)
    overridden_parameters: Tuple[str, ...] = field(default_factory=tuple)
    notes: Tuple[str, ...] = field(default_factory=tuple)
    provenance: Tuple[Tuple[str, Any], ...] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "unsupported": list(self.unsupported),
            "approximations": list(self.approximations),
            "overridden_parameters": list(self.overridden_parameters),
            "notes": list(self.notes),
            "provenance": _jsonable(dict(self.provenance)),
        }


class Index(Mapping):
    """Ordered, immutable ``path -> item`` mapping (``result.ops["main/k/mma"]``)."""

    __slots__ = ("_items", "_keys")

    def __init__(self, items: Any) -> None:
        pairs = tuple((str(key), value) for key, value in (items.items() if isinstance(items, Mapping) else items))
        keys = [key for key, _value in pairs]
        if len(keys) != len(set(keys)):
            raise ModelingValidationError("Index keys must be unique")
        self._items = dict(pairs)
        self._keys = tuple(keys)

    def __getitem__(self, key: str) -> Any:
        return self._items[key]

    def __iter__(self):
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)

    def __repr__(self) -> str:
        return "Index(%r)" % (self._keys,)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Mapping) and dict(self.items()) == dict(other.items())

    def __hash__(self) -> int:
        return hash(self._keys)


@dataclass(frozen=True)
class AnalysisResult:
    program: ProgramResult
    launches: Index
    regions: Index
    ops: Index
    cache: Tuple[CacheReport, ...]
    scenarios: Index
    diagnostics: Diagnostics
    selected_scenario: str
    input_digest: str
    workload_digests: Tuple[Tuple[str, str], ...] = field(default_factory=tuple)
    model_version: str = MODEL_VERSION

    def __post_init__(self) -> None:
        for name in ("launches", "regions", "ops", "scenarios"):
            value = getattr(self, name)
            if not isinstance(value, Index):
                object.__setattr__(self, name, Index(value))
        object.__setattr__(self, "cache", tuple(self.cache))
        object.__setattr__(self, "workload_digests", tuple(self.workload_digests))

    def launch(self, name: str) -> LaunchResult:
        return self.launches[name]

    def region(self, path: str) -> RegionResult:
        return self.regions[path]

    def op(self, path: str) -> OpResult:
        return self.ops[path]

    def scenario(self, name: str) -> "AnalysisResult":
        return self.scenarios[name]

    def workload_digest(self, target: str) -> Optional[str]:
        """Digest of a launch name, or of ``"program:<name>"`` (program and launch names may be equal)."""

        return dict(self.workload_digests).get(target)

    @property
    def program_workload_digest(self) -> Optional[str]:
        return self.workload_digest("program:" + self.program.name)

    def to_snapshot(self) -> Dict[str, Any]:
        return {
            "schema": RESULT_SCHEMA_VERSION,
            "model_version": self.model_version,
            "input_digest": self.input_digest,
            "workload_digests": [list(item) for item in self.workload_digests],
            "selected_scenario": self.selected_scenario,
            "program": self.program.to_dict(),
            "launches": [[name, item.to_dict()] for name, item in self.launches.items()],
            "regions": [[path, item.to_dict()] for path, item in self.regions.items()],
            "ops": [[path, item.to_dict()] for path, item in self.ops.items()],
            "cache": [item.to_dict() for item in self.cache],
            "scenarios": [[name, item.to_snapshot()] for name, item in self.scenarios.items()],
            "diagnostics": self.diagnostics.to_dict(),
        }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dataclass_fields__"):
        return {f.name: _jsonable(getattr(value, f.name)) for f in fields(value)}
    return repr(value)


__all__ = [
    "AnalysisResult", "Index", "CacheAccessReport", "CacheReport", "CompletionInterval",
    "CostGroup", "CostGroupKey", "Diagnostics", "IIComponents", "LaunchResult",
    "MODEL_VERSION", "OpResult", "ProgramResult", "RESULT_SCHEMA_VERSION",
    "RegionGroupResult", "RegionResult", "ServiceInterval", "Traffic", "TrafficShare",
    "TrafficInterval", "WorkGroupSummary", "scale_traffic", "traffic_close", "traffic_shares",
    "traffic_from_dict", "traffic_to_dict",
]
