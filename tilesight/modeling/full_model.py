"""Typed full-kernel result contract for the experimental modeling frontend."""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Protocol, Tuple

from .errors import ModelingError, ModelingValidationError
from .ir import KernelIR, freeze_attrs, freeze_value, validate_identifier


def _nonnegative(value: float, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ModelingValidationError("%s must be numeric" % label) from error
    if not math.isfinite(result) or result < 0.0:
        raise ModelingValidationError("%s must be finite and non-negative" % label)
    return result


def _finite(value: float, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ModelingValidationError("%s must be numeric" % label) from error
    if not math.isfinite(result):
        raise ModelingValidationError("%s must be finite" % label)
    return result


class FullModelError(ModelingError):
    """Raised when no complete-kernel adapter can model a KernelIR."""


class FeasibilityStatus(str, Enum):
    NOT_REQUESTED = "not_requested"
    PROVEN_FEASIBLE = "proven_feasible"
    PROVEN_INFEASIBLE = "proven_infeasible"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class FullModelOptions:
    """Options shared by complete GEMM/FA adapters.

    ``kernel_launch_s=None`` selects the unified one-time 2 us kernel launch.
    GEMM callers may pass zero to recover its legacy body-only convention;
    audited FA3 keeps its source-scoped fixed 2 us launch. Adapter-specific
    options remain hashable key/value metadata.
    """

    ii_mode: str = "resource_ii"
    grid_policy: str = "model_default"
    search_profile: str = "reference"
    kernel_launch_s: Optional[float] = None
    host_dispatch_s: float = 0.0
    feasibility_policy: str = "report"
    adapter_options: Tuple[Tuple[str, Any], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.ii_mode not in ("resource_ii", "periodic_best", "periodic_worst"):
            raise ModelingValidationError("unsupported ii_mode %r" % self.ii_mode)
        validate_identifier(self.grid_policy, "grid_policy")
        validate_identifier(self.search_profile, "search_profile")
        if self.kernel_launch_s is not None:
            object.__setattr__(
                self,
                "kernel_launch_s",
                _nonnegative(self.kernel_launch_s, "kernel_launch_s"),
            )
        object.__setattr__(
            self,
            "host_dispatch_s",
            _nonnegative(self.host_dispatch_s, "host_dispatch_s"),
        )
        if self.feasibility_policy != "report":
            raise ModelingValidationError(
                "v0 full modeling supports feasibility_policy='report' only"
            )
        object.__setattr__(self, "adapter_options", freeze_attrs(self.adapter_options))

    def adapter_dict(self) -> Mapping[str, Any]:
        return dict(self.adapter_options)


@dataclass(frozen=True)
class TimingBreakdown:
    kernel_body_s: float
    launch_s: float
    host_dispatch_s: float
    total_s: float
    prologue_s: float = 0.0
    steady_s: float = 0.0
    epilogue_s: float = 0.0
    per_tile_s: Optional[float] = None
    component_scope: str = "kernel_critical_path"

    def __post_init__(self) -> None:
        for name in (
            "kernel_body_s",
            "launch_s",
            "host_dispatch_s",
            "total_s",
            "prologue_s",
            "steady_s",
            "epilogue_s",
        ):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name))
        if self.per_tile_s is not None:
            object.__setattr__(
                self, "per_tile_s", _nonnegative(self.per_tile_s, "per_tile_s")
            )
        validate_identifier(self.component_scope, "component_scope")
        expected = self.kernel_body_s + self.launch_s + self.host_dispatch_s
        tolerance = 1.0e-15 + 1.0e-10 * max(abs(expected), abs(self.total_s))
        if abs(self.total_s - expected) > tolerance:
            raise ModelingValidationError(
                "total_s must equal kernel_body_s + launch_s + host_dispatch_s"
            )


@dataclass(frozen=True)
class IISummary:
    selected_s: Optional[float] = None
    resource_lower_bound_s: Optional[float] = None
    best_s: Optional[float] = None
    worst_s: Optional[float] = None
    scope: str = "not_available"
    constructive: Optional[bool] = None
    search_complete: Optional[bool] = None

    def __post_init__(self) -> None:
        for name in (
            "selected_s",
            "resource_lower_bound_s",
            "best_s",
            "worst_s",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _nonnegative(value, name))
        validate_identifier(self.scope, "II scope")


@dataclass(frozen=True)
class OccupancyResult:
    resident_ctas_per_sm: Optional[float] = None
    smem_bytes_per_cta: Optional[float] = None
    register_value: Optional[float] = None
    register_unit: str = "unknown"
    warps_per_cta: Optional[float] = None
    cluster_size: int = 1

    def __post_init__(self) -> None:
        for name in (
            "resident_ctas_per_sm",
            "smem_bytes_per_cta",
            "register_value",
            "warps_per_cta",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _nonnegative(value, name))
        validate_identifier(self.register_unit, "register_unit")
        if not isinstance(self.cluster_size, int) or self.cluster_size <= 0:
            raise ModelingValidationError("cluster_size must be a positive integer")


@dataclass(frozen=True)
class GridResult:
    work_grid: Tuple[Any, ...]
    physical_grid: Tuple[Any, ...]
    scheduler: str
    total_work_units: Optional[float] = None
    waves: Optional[float] = None
    tail_factor: Optional[float] = None
    critical_work_units: Optional[float] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "work_grid", tuple(freeze_value(self.work_grid)))
        object.__setattr__(
            self, "physical_grid", tuple(freeze_value(self.physical_grid))
        )
        validate_identifier(self.scheduler, "grid scheduler")
        for name in ("total_work_units", "waves", "tail_factor", "critical_work_units"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _nonnegative(value, name))


@dataclass(frozen=True)
class StoragePeak:
    storage_scope: str
    execution_scope: str
    lower_bytes: float
    selected_bytes: float
    upper_bytes: float
    confidence: str

    def __post_init__(self) -> None:
        validate_identifier(self.storage_scope, "storage_scope")
        validate_identifier(self.execution_scope, "execution_scope")
        validate_identifier(self.confidence, "storage confidence")
        for name in ("lower_bytes", "selected_bytes", "upper_bytes"):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name))
        if not self.lower_bytes <= self.selected_bytes <= self.upper_bytes:
            raise ModelingValidationError("storage peak must satisfy lower <= selected <= upper")


@dataclass(frozen=True)
class LivenessInterval:
    """One witness-expanded logical value lifetime in SI seconds."""

    storage_name: str
    source_kind: str
    storage_scope: str
    execution_scope: str
    iteration: int
    start_s: float
    end_s: float
    bytes_per_instance: float
    alias_group: Optional[str] = None

    def __post_init__(self) -> None:
        from .ir import validate_identifier

        for value, label in (
            (self.storage_name, "liveness storage_name"),
            (self.source_kind, "liveness source_kind"),
            (self.storage_scope, "liveness storage_scope"),
            (self.execution_scope, "liveness execution_scope"),
        ):
            validate_identifier(value, label)
        if self.alias_group is not None:
            validate_identifier(self.alias_group, "liveness alias_group")
        if not isinstance(self.iteration, int) or isinstance(self.iteration, bool):
            raise ModelingValidationError("liveness iteration must be an integer")
        object.__setattr__(self, "start_s", _finite(self.start_s, "start_s"))
        object.__setattr__(self, "end_s", _finite(self.end_s, "end_s"))
        if self.end_s < self.start_s:
            raise ModelingValidationError("liveness interval end must not precede start")
        object.__setattr__(
            self,
            "bytes_per_instance",
            _nonnegative(self.bytes_per_instance, "bytes_per_instance"),
        )


@dataclass(frozen=True)
class AliasInterference:
    """Concurrent distinct members of one declared physical alias group."""

    alias_group: str
    storage_scope: str
    execution_scope: str
    time_s: float
    members: Tuple[str, ...]
    live_bytes_per_instance: float

    def __post_init__(self) -> None:
        from .ir import validate_identifier

        validate_identifier(self.alias_group, "alias interference group")
        validate_identifier(self.storage_scope, "alias interference storage_scope")
        validate_identifier(self.execution_scope, "alias interference execution_scope")
        object.__setattr__(self, "time_s", _finite(self.time_s, "time_s"))
        members = tuple(self.members)
        if len(members) < 2:
            raise ModelingValidationError(
                "alias interference requires at least two live members"
            )
        for member in members:
            validate_identifier(member, "alias interference member")
        object.__setattr__(self, "members", members)
        object.__setattr__(
            self,
            "live_bytes_per_instance",
            _nonnegative(self.live_bytes_per_instance, "live_bytes_per_instance"),
        )


@dataclass(frozen=True)
class ScopeCapacityReport:
    """Per-SM capacity accounting after ownership-domain replication."""

    storage_scope: str
    execution_scope: str
    allocation_kind: str
    instances_per_cta: float
    members_per_instance: int
    resident_ctas_per_sm: float
    cluster_size: int
    replicas_per_sm: float
    per_replica_peak_bytes: float
    aggregate_peak_bytes: float
    capacity_bytes: Optional[float]
    exceeds_capacity: Optional[bool]
    ownership_provenance: str
    confidence: str

    def __post_init__(self) -> None:
        from .ir import validate_identifier

        for value, label in (
            (self.storage_scope, "capacity storage_scope"),
            (self.execution_scope, "capacity execution_scope"),
            (self.allocation_kind, "capacity allocation_kind"),
            (self.ownership_provenance, "capacity ownership_provenance"),
            (self.confidence, "capacity confidence"),
        ):
            validate_identifier(value, label)
        for name in (
            "instances_per_cta",
            "resident_ctas_per_sm",
            "replicas_per_sm",
            "per_replica_peak_bytes",
            "aggregate_peak_bytes",
        ):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name))
        if (
            not isinstance(self.members_per_instance, int)
            or isinstance(self.members_per_instance, bool)
            or self.members_per_instance <= 0
        ):
            raise ModelingValidationError(
                "capacity members_per_instance must be a positive integer"
            )
        if not isinstance(self.cluster_size, int) or self.cluster_size <= 0:
            raise ModelingValidationError("capacity cluster_size must be positive")
        if self.capacity_bytes is not None:
            object.__setattr__(
                self,
                "capacity_bytes",
                _nonnegative(self.capacity_bytes, "capacity_bytes"),
            )
        if self.exceeds_capacity is not None and not isinstance(
            self.exceeds_capacity, bool
        ):
            raise ModelingValidationError(
                "exceeds_capacity must be bool or None"
            )


@dataclass(frozen=True)
class OwnershipAssessment:
    """Whether a handoff has a same-domain, same-index ownership proof."""

    storage_name: str
    execution_scope: str
    status: str
    diagnostic: str

    def __post_init__(self) -> None:
        from .ir import validate_identifier

        validate_identifier(self.storage_name, "ownership assessment storage_name")
        validate_identifier(self.execution_scope, "ownership assessment scope")
        if self.status not in ("proven_compatible", "user_asserted", "unknown"):
            raise ModelingValidationError(
                "ownership assessment status must be proven_compatible, "
                "user_asserted, or unknown"
            )
        object.__setattr__(self, "diagnostic", str(self.diagnostic))


@dataclass(frozen=True)
class LivenessReport:
    status: FeasibilityStatus = FeasibilityStatus.UNKNOWN
    peaks: Tuple[StoragePeak, ...] = field(default_factory=tuple)
    diagnostics: Tuple[str, ...] = field(default_factory=tuple)
    intervals: Tuple[LivenessInterval, ...] = field(default_factory=tuple)
    alias_interference: Tuple[AliasInterference, ...] = field(default_factory=tuple)
    capacity: Tuple[ScopeCapacityReport, ...] = field(default_factory=tuple)
    ownership: Tuple[OwnershipAssessment, ...] = field(default_factory=tuple)
    witness_ii_s: Optional[float] = None
    guard_eligible: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "peaks", tuple(self.peaks))
        object.__setattr__(self, "diagnostics", tuple(str(x) for x in self.diagnostics))
        object.__setattr__(self, "intervals", tuple(self.intervals))
        object.__setattr__(self, "alias_interference", tuple(self.alias_interference))
        object.__setattr__(self, "capacity", tuple(self.capacity))
        object.__setattr__(self, "ownership", tuple(self.ownership))
        for values, expected, label in (
            (self.peaks, StoragePeak, "peaks"),
            (self.intervals, LivenessInterval, "intervals"),
            (self.alias_interference, AliasInterference, "alias_interference"),
            (self.capacity, ScopeCapacityReport, "capacity"),
            (self.ownership, OwnershipAssessment, "ownership"),
        ):
            if not all(isinstance(item, expected) for item in values):
                raise ModelingValidationError(
                    "LivenessReport.%s contains an invalid item" % label
                )
        if self.witness_ii_s is not None:
            value = _nonnegative(self.witness_ii_s, "witness_ii_s")
            if value <= 0.0:
                raise ModelingValidationError("witness_ii_s must be positive")
            object.__setattr__(self, "witness_ii_s", value)
        if self.guard_eligible is not False:
            raise ModelingValidationError(
                "witness-aware liveness v1 is report-only and not guard eligible"
            )


@dataclass(frozen=True)
class MemoryTraffic:
    """Per-level read/write bytes before and after a fusion plan."""

    memory_level: str
    baseline_read_bytes: float = 0.0
    baseline_write_bytes: float = 0.0
    fused_read_bytes: float = 0.0
    fused_write_bytes: float = 0.0
    eliminated_read_bytes: float = 0.0
    eliminated_write_bytes: float = 0.0
    introduced_read_bytes: float = 0.0
    introduced_write_bytes: float = 0.0

    def __post_init__(self) -> None:
        from .ir import canonical_memory_level

        object.__setattr__(
            self, "memory_level", canonical_memory_level(self.memory_level)
        )
        for name in (
            "baseline_read_bytes",
            "baseline_write_bytes",
            "fused_read_bytes",
            "fused_write_bytes",
            "eliminated_read_bytes",
            "eliminated_write_bytes",
            "introduced_read_bytes",
            "introduced_write_bytes",
        ):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name))
        for mode in ("read", "write"):
            baseline = getattr(self, "baseline_%s_bytes" % mode)
            fused = getattr(self, "fused_%s_bytes" % mode)
            eliminated = getattr(self, "eliminated_%s_bytes" % mode)
            introduced = getattr(self, "introduced_%s_bytes" % mode)
            expected = baseline - eliminated + introduced
            tolerance = 1.0e-9 * max(1.0, baseline, fused, eliminated, introduced)
            if eliminated > baseline + tolerance or abs(fused - expected) > tolerance:
                raise ModelingValidationError(
                    "memory traffic must satisfy fused = baseline - eliminated + introduced"
                )


@dataclass(frozen=True)
class ValueTrafficPath:
    """Human-readable directed materialization and fused-handoff paths."""

    value: str
    producer_fragment: str
    consumer_fragments: Tuple[str, ...]
    baseline_store_path: Tuple[str, ...]
    baseline_load_paths: Tuple[Tuple[str, ...], ...]
    fused_handoff_paths: Tuple[Tuple[str, ...], ...]
    iteration_distance: int = 0

    def __post_init__(self) -> None:
        from .ir import canonical_memory_level

        validate_identifier(self.value, "traffic path value")
        validate_identifier(self.producer_fragment, "traffic path producer")
        consumers = tuple(self.consumer_fragments)
        if not consumers:
            raise ModelingValidationError("traffic path needs a consumer")
        for consumer in consumers:
            validate_identifier(consumer, "traffic path consumer")
        object.__setattr__(self, "consumer_fragments", consumers)
        store_path = tuple(canonical_memory_level(x) for x in self.baseline_store_path)
        if not store_path:
            raise ModelingValidationError("baseline_store_path must not be empty")
        object.__setattr__(self, "baseline_store_path", store_path)
        for name in ("baseline_load_paths", "fused_handoff_paths"):
            paths = tuple(
                tuple(canonical_memory_level(x) for x in path)
                for path in getattr(self, name)
            )
            if len(paths) != len(consumers) or any(not path for path in paths):
                raise ModelingValidationError(
                    "%s must provide one nonempty path per consumer" % name
                )
            object.__setattr__(self, name, paths)
        if not isinstance(self.iteration_distance, int) or isinstance(
            self.iteration_distance, bool
        ):
            raise ModelingValidationError("traffic path iteration_distance must be integer")


@dataclass(frozen=True)
class FusionReport:
    status: FeasibilityStatus = FeasibilityStatus.NOT_REQUESTED
    diagnostics: Tuple[str, ...] = field(default_factory=tuple)
    eliminated_global_bytes: float = 0.0
    traffic: Tuple[MemoryTraffic, ...] = field(default_factory=tuple)
    value_paths: Tuple[ValueTrafficPath, ...] = field(default_factory=tuple)
    handoff_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "diagnostics", tuple(str(x) for x in self.diagnostics))
        object.__setattr__(
            self,
            "eliminated_global_bytes",
            _nonnegative(self.eliminated_global_bytes, "eliminated_global_bytes"),
        )
        object.__setattr__(self, "traffic", tuple(self.traffic))
        object.__setattr__(self, "value_paths", tuple(self.value_paths))
        if not all(isinstance(x, MemoryTraffic) for x in self.traffic):
            raise ModelingValidationError("FusionReport.traffic must contain MemoryTraffic")
        if not all(isinstance(x, ValueTrafficPath) for x in self.value_paths):
            raise ModelingValidationError(
                "FusionReport.value_paths must contain ValueTrafficPath"
            )
        if not isinstance(self.handoff_count, int) or isinstance(self.handoff_count, bool):
            raise ModelingValidationError("handoff_count must be an integer")
        if self.handoff_count < 0:
            raise ModelingValidationError("handoff_count must be non-negative")
        ddr = next((x for x in self.traffic if x.memory_level == "ddr"), None)
        expected = 0.0 if ddr is None else (
            ddr.eliminated_read_bytes + ddr.eliminated_write_bytes
        )
        tolerance = 1.0e-9 * max(1.0, expected, self.eliminated_global_bytes)
        if abs(self.eliminated_global_bytes - expected) > tolerance:
            raise ModelingValidationError(
                "eliminated_global_bytes must equal eliminated DDR read+write traffic"
            )


@dataclass(frozen=True)
class FullModelResult:
    adapter: str
    timing: TimingBreakdown
    ii: IISummary
    occupancy: OccupancyResult
    grid: GridResult
    resource_metrics: Tuple[Tuple[str, Any], ...] = field(default_factory=tuple)
    liveness: LivenessReport = field(default_factory=LivenessReport)
    fusion: FusionReport = field(default_factory=FusionReport)
    legacy: Tuple[Tuple[str, Any], ...] = field(default_factory=tuple)
    provenance: Tuple[Tuple[str, Any], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        validate_identifier(self.adapter, "adapter")
        object.__setattr__(self, "resource_metrics", freeze_attrs(self.resource_metrics))
        object.__setattr__(self, "legacy", freeze_attrs(self.legacy))
        object.__setattr__(self, "provenance", freeze_attrs(self.provenance))

    @property
    def total_s(self) -> float:
        return self.timing.total_s


class FullKernelAdapter(Protocol):
    def model(
        self,
        kernel: KernelIR,
        arch: Any,
        options: FullModelOptions,
    ) -> FullModelResult:
        """Return a complete-kernel estimate for ``kernel``."""


_BUILTIN_ADAPTERS = {
    "gemm_pipeline_wave": (
        "tilesight.modeling.adapters.gemm",
        "model_gemm",
    ),
    "fa3": (
        "tilesight.modeling.adapters.flash_attention",
        "model_fa3",
    ),
    "fa4": (
        "tilesight.modeling.adapters.flash_attention",
        "model_fa4",
    ),
}


def model(
    kernel: KernelIR,
    arch: Any,
    options: Optional[FullModelOptions] = None,
) -> FullModelResult:
    """Run the registered complete-kernel adapter selected by IR metadata."""

    if not isinstance(kernel, KernelIR):
        raise ModelingValidationError("model expects a frozen KernelIR")
    if options is None:
        options = FullModelOptions()
    params = dict(kernel.params)
    adapter_name = params.get("model_adapter")
    if adapter_name not in _BUILTIN_ADAPTERS:
        raise FullModelError(
            "KernelIR params must select one of %s with model_adapter; got %r"
            % (sorted(_BUILTIN_ADAPTERS), adapter_name)
        )
    module_name, function_name = _BUILTIN_ADAPTERS[adapter_name]
    module = importlib.import_module(module_name)
    function = getattr(module, function_name)
    return function(kernel, arch, options)


__all__ = [
    "AliasInterference",
    "FeasibilityStatus",
    "FullKernelAdapter",
    "FullModelError",
    "FullModelOptions",
    "FullModelResult",
    "FusionReport",
    "GridResult",
    "IISummary",
    "LivenessReport",
    "LivenessInterval",
    "MemoryTraffic",
    "OccupancyResult",
    "OwnershipAssessment",
    "ScopeCapacityReport",
    "StoragePeak",
    "TimingBreakdown",
    "ValueTrafficPath",
    "model",
]
