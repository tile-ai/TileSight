"""Semantic GEMM cost binding and two explicit evaluation profiles.

This module is intentionally separate from :mod:`adapters.gemm`.  The latter
is a strict compatibility bridge that invokes the historical monolithic GEMM
entry point.  Here, a :class:`~tilesight.modeling.ops.GemmOpSpec`
provides operation semantics and a frozen :class:`KernelIR` provides the
schedule.  ``BoundGemmCostOracle`` binds both to architecture/cache costs.

Two profiles consume that same binding:

``legacy_calibrated``
    Reuses the existing low-level occupancy, pipeline, and wave policies, but
    assembles their typed inputs locally and never calls
    ``calculate_matmul_pipeline_wave``.

``native_periodic``
    Lowers the cost-bound schedule to ``PeriodicDAG``, searches a constructive
    steady-state witness, and reports the natural finite-loop fill/drain span.
    It deliberately does not invent a multi-CTA/full-grid latency from a
    single-CTA witness.

The default cache policy is the deterministic stable shadow-cohort model with
an 8 KiB reuse-distance unit.  ``legacy_rng`` is an explicit parity-only cache
policy for comparing the decomposed implementation with the old monolith.
"""

from __future__ import annotations

import math
import copy
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from tilesight.modeling._pipeline.occupancy import compute_occupancy
from tilesight.modeling._pipeline.periodic_schedule import (
    IIScheduleSelection,
    PeriodicDAG,
    ScheduleEnvelope,
    SearchConfig,
    compute_lower_bounds,
    schedule_periodic_dag,
    select_steady_ii,
)
from tilesight.modeling._pipeline.pipeline_overlap import (
    compute_pipeline_tile_latency_with_occupancy,
    resources_to_times,
)
from tilesight.modeling._pipeline.resource_types import (
    PerIterationResources,
    PrologueEpilogueResources,
    TileResources,
)
from tilesight.modeling._pipeline.wave_model import (
    compute_wave_adjusted_latency,
)

from ..cache import (
    CacheProblem,
    CacheResult,
    ProgressJitterConfig,
    ReuseHistogram,
    SamplingConfig,
    build_reuse_histogram,
    evaluate_histogram,
    gemm_cache_problem,
)
from ..errors import ModelingValidationError
from ..ir import KernelIR, LaunchIR, PeriodicLoopIR, ResourceTiming, Timing
from ..ops import (
    GemmOpSpec,
    ThroughputFallbackPolicy,
    ThroughputRequest,
    ThroughputResolution,
    as_dtype,
    resolve_throughput,
)


_LEGACY_L2_TWO_PART_CORES = frozenset(("A100", "H100", "B200", "A100_LUT"))
_SUPPORTED_MMA_TYPES = frozenset(("wmma", "wgmma"))
_PROFILES = frozenset(("legacy_calibrated", "native_periodic"))
_CACHE_POLICIES = frozenset(("stable_shadow", "legacy_rng"))


def _positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ModelingValidationError("%s must be a positive integer" % label)
    return value


def _positive_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ModelingValidationError("%s must be numeric" % label) from error
    if not math.isfinite(result) or result <= 0.0:
        raise ModelingValidationError("%s must be finite and positive" % label)
    return result


def _int_tuple(value: Any, label: str, length: int) -> Tuple[int, ...]:
    try:
        result = tuple(value)
    except TypeError as error:
        raise ModelingValidationError("%s must be a sequence" % label) from error
    if len(result) != length:
        raise ModelingValidationError(
            "%s must have exactly %d dimensions" % (label, length)
        )
    return tuple(_positive_int(item, "%s dimension" % label) for item in result)


def _attrs(**items: Any) -> Tuple[Tuple[str, Any], ...]:
    """Small hashable provenance helper (values are scalar/tuple here)."""

    return tuple(sorted(items.items()))


@dataclass(frozen=True)
class GeneralGemmOptions:
    """Policy choices for semantic GEMM binding/evaluation.

    ``ii_mode=None`` selects ``resource_ii`` for ``legacy_calibrated`` and
    ``periodic_best`` for ``native_periodic``.  The latter requires a witness;
    resource-only mode is therefore rejected rather than used as a latency.
    """

    profile: str = "legacy_calibrated"
    cache_policy: str = "stable_shadow"
    cache_reuse_unit_bytes: float = 8.0 * 1024.0
    cache_seed: int = 0
    cache_sample_budget: Optional[int] = 512
    cache_histogram: Optional[ReuseHistogram] = None
    representative_k: int = 0
    progress_jitter: ProgressJitterConfig = field(
        default_factory=ProgressJitterConfig
    )
    tensor_throughput_fallback: ThroughputFallbackPolicy = field(
        default_factory=ThroughputFallbackPolicy
    )
    ii_mode: Optional[str] = None
    search_config: SearchConfig = field(default_factory=SearchConfig)

    def __post_init__(self) -> None:
        if self.profile not in _PROFILES:
            raise ModelingValidationError(
                "general GEMM profile must be one of %s" % sorted(_PROFILES)
            )
        if self.cache_policy not in _CACHE_POLICIES:
            raise ModelingValidationError(
                "general GEMM cache_policy must be one of %s"
                % sorted(_CACHE_POLICIES)
            )
        object.__setattr__(
            self,
            "cache_reuse_unit_bytes",
            _positive_float(self.cache_reuse_unit_bytes, "cache reuse unit"),
        )
        if (
            not isinstance(self.cache_seed, int)
            or isinstance(self.cache_seed, bool)
            or self.cache_seed < 0
            or self.cache_seed >= 2 ** 32
        ):
            raise ModelingValidationError("cache_seed must fit uint32")
        if self.cache_sample_budget is not None:
            _positive_int(self.cache_sample_budget, "cache_sample_budget")
        if self.cache_histogram is not None and not isinstance(
            self.cache_histogram, ReuseHistogram
        ):
            raise ModelingValidationError(
                "cache_histogram must be ReuseHistogram or None"
            )
        if (
            not isinstance(self.representative_k, int)
            or isinstance(self.representative_k, bool)
            or self.representative_k < 0
        ):
            raise ModelingValidationError(
                "representative_k must be a non-negative integer"
            )
        if not isinstance(self.progress_jitter, ProgressJitterConfig):
            raise ModelingValidationError(
                "progress_jitter must be ProgressJitterConfig"
            )
        if not isinstance(
            self.tensor_throughput_fallback, ThroughputFallbackPolicy
        ):
            raise ModelingValidationError(
                "tensor_throughput_fallback must be ThroughputFallbackPolicy"
            )
        if self.ii_mode not in (None, "resource_ii", "periodic_best", "periodic_worst"):
            raise ModelingValidationError(
                "ii_mode must be resource_ii, periodic_best, periodic_worst, or None"
            )
        if not isinstance(self.search_config, SearchConfig):
            raise ModelingValidationError("search_config must be SearchConfig")

    @property
    def selected_ii_mode(self) -> str:
        if self.ii_mode is not None:
            return self.ii_mode
        if self.profile == "native_periodic":
            return "periodic_best"
        return "resource_ii"


@dataclass(frozen=True)
class GemmScheduleSpec:
    """Validated connection between one semantic GEMM and one schedule loop."""

    launch_name: str
    loop_name: str
    tile: Tuple[int, int, int]
    warp_tile: Tuple[int, int, int]
    stages: int
    grid: Tuple[int, int]
    iterations: int
    threads: int
    mma_type: str
    mem_levels: Tuple[Tuple[str, Tuple[float, ...]], ...]
    row_panel: int
    column_panel: Optional[int]
    raster_axis: str
    load_a_phase: str
    load_b_phase: str
    mma_phase: str
    copy_resource: str
    tensor_resource: str

    def mem_level_dict(self) -> Dict[str, Tuple[float, ...]]:
        return dict(self.mem_levels)


@dataclass(frozen=True)
class GemmCacheSummary:
    """Cache rates used by the cost equations, plus backend identity."""

    backend: str
    problem_digest: Optional[str]
    l1_5_hit_rate: float
    l2_hit_rate: float
    ddr_miss_rate: float
    reuse_unit_bytes: float
    reduction_fidelity: str
    histogram_reused: bool = False
    sampling_mode: str = "exact"
    whole_grid_ddr_read_bytes: Optional[float] = None
    whole_grid_ddr_write_bytes: Optional[float] = None
    representative_grid_ddr_read_bytes: Optional[float] = None


@dataclass(frozen=True)
class GemmPerIterationCost:
    """One CTA's resource demand for one K-tile iteration."""

    logical_a_copy_bytes: float
    logical_b_copy_bytes: float
    l1_5_io_bytes: float
    l2_io_bytes: float
    ddr_io_bytes: float
    smem_io_bytes: float
    tensor_flops: float


@dataclass(frozen=True)
class GemmEpilogueCost:
    """One output tile's final store demand."""

    ddr_io_bytes: float
    l2_io_bytes: float
    l1_5_io_bytes: float
    smem_io_bytes: float


@dataclass(frozen=True)
class GemmFootprint:
    smem_bytes_per_cta: float
    legacy_registers_per_thread: float
    warps_per_cta: int
    resident_ctas_per_sm: int


@dataclass(frozen=True)
class GemmServiceTimes:
    """Full-wave per-SM service times in SI seconds."""

    ddr_s: float
    l2_s: float
    l1_5_s: float
    smem_s: float
    memory_s: float
    tensor_s: float
    resource_ii_s: float


@dataclass(frozen=True)
class GemmCostBreakdown:
    """Typed output of semantic/schedule/architecture cost binding."""

    schedule: GemmScheduleSpec
    cache: GemmCacheSummary
    per_iteration: GemmPerIterationCost
    epilogue: GemmEpilogueCost
    footprint: GemmFootprint
    service: GemmServiceTimes
    tensor_throughput: ThroughputResolution
    compute_overhead: float
    provenance: Tuple[Tuple[str, Any], ...]

    @property
    def steady_resource_ii_s(self) -> float:
        return self.service.resource_ii_s


@dataclass(frozen=True)
class GemmNaturalBoundary:
    """Finite-loop span obtained by clipping a periodic witness naturally."""

    first_start_s: float
    last_completion_s: float
    latency_s: float
    occurrence_count: int
    iterations: int


@dataclass(frozen=True)
class LegacyCalibratedGemmResult:
    """Decomposed result of the old low-level occupancy/pipeline/wave policy."""

    per_tile_latency_s: float
    kernel_body_s: float
    ddr_util: float
    compute_util: float
    l2_hit_rate: float
    resident_ctas_per_sm: int
    waves: float
    prologue_s: float
    steady_time_per_iter_s: float
    epilogue_s: float
    mem_time_per_iter_s: float
    compute_time_per_iter_s: float
    wave_info: Tuple[Tuple[str, Any], ...]
    provenance: Tuple[Tuple[str, Any], ...]


@dataclass(frozen=True)
class GeneralGemmResult:
    """Result shared by the calibrated and native-periodic profiles."""

    profile: str
    breakdown: GemmCostBreakdown
    dag: PeriodicDAG
    ii: IIScheduleSelection
    envelope: Optional[ScheduleEnvelope]
    natural_boundary: Optional[GemmNaturalBoundary]
    legacy_calibrated: Optional[LegacyCalibratedGemmResult]
    provenance: Tuple[Tuple[str, Any], ...]

    @property
    def kernel_body_s(self) -> Optional[float]:
        if self.legacy_calibrated is None:
            return None
        return self.legacy_calibrated.kernel_body_s


def _select_launch_loop(kernel: KernelIR) -> Tuple[LaunchIR, PeriodicLoopIR]:
    params = dict(kernel.params)
    launch_name = params.get("launch")
    if launch_name is None:
        if len(kernel.launches) != 1:
            raise ModelingValidationError(
                "general GEMM needs one launch or params['launch']"
            )
        launch = kernel.launches[0]
    else:
        matches = [item for item in kernel.launches if item.name == launch_name]
        if len(matches) != 1:
            raise ModelingValidationError("GEMM launch selection is ambiguous")
        launch = matches[0]

    loop_name = params.get("mainloop")
    if loop_name is None:
        if len(launch.periodic_loops) != 1:
            raise ModelingValidationError(
                "general GEMM needs one periodic loop or params['mainloop']"
            )
        loop = launch.periodic_loops[0]
    else:
        matches = [item for item in launch.periodic_loops if item.name == loop_name]
        if len(matches) != 1:
            raise ModelingValidationError("GEMM mainloop selection is ambiguous")
        loop = matches[0]
    return launch, loop


def _canonical_mem_levels(
    op: GemmOpSpec, params: Mapping[str, Any]
) -> Tuple[Tuple[str, Tuple[float, ...]], ...]:
    raw = params.get("mem_levels")
    dtype_bytes = {
        "in1": op.a_storage_dtype.storage_bytes,
        "in2": op.b_storage_dtype.storage_bytes,
        "out1": op.result_storage_dtype.storage_bytes,
    }
    if raw is None:
        mapping = {
            name: (1.0, 1.0, 1.0, float(size))
            for name, size in dtype_bytes.items()
        }
    else:
        try:
            mapping = dict(raw)
        except (TypeError, ValueError) as error:
            raise ModelingValidationError("GEMM mem_levels must be a mapping") from error
    result = []
    for name in ("in1", "in2", "out1"):
        if name not in mapping:
            raise ModelingValidationError("GEMM mem_levels is missing %s" % name)
        try:
            values = tuple(float(item) for item in mapping[name])
        except (TypeError, ValueError) as error:
            raise ModelingValidationError(
                "GEMM mem_levels[%s] must be numeric" % name
            ) from error
        if len(values) < 4:
            raise ModelingValidationError(
                "GEMM mem_levels[%s] must have at least four entries" % name
            )
        if any(not math.isfinite(item) or item < 0.0 for item in values):
            raise ModelingValidationError("GEMM mem_levels must be non-negative")
        if any(item not in (0.0, 1.0) for item in values[:3]):
            raise ModelingValidationError(
                "GEMM mem_levels compatibility path flags must be zero or one"
            )
        if not math.isclose(values[-1], dtype_bytes[name], rel_tol=0.0, abs_tol=1e-12):
            raise ModelingValidationError(
                "GEMM mem_levels[%s] byte width disagrees with semantic storage dtype"
                % name
            )
        result.append((name, values))
    return tuple(result)


def _phase_resource(phase: Any, fallback: str) -> str:
    if phase.timing is None or not phase.timing.resources:
        engine = dict(phase.work.attrs).get("engine")
        if isinstance(engine, str) and engine:
            return engine
        return fallback
    names = tuple(item.resource for item in phase.timing.resources)
    if len(names) != 1:
        raise ModelingValidationError(
            "general GEMM phase %s must use one schedule resource hint" % phase.name
        )
    return names[0]


def _identify_gemm_phases(
    loop: PeriodicLoopIR,
    tile: Tuple[int, int, int],
    a_dtype: Any,
    b_dtype: Any,
) -> Tuple[Any, Any, Any]:
    a_dtype = as_dtype(a_dtype)
    b_dtype = as_dtype(b_dtype)
    copy_phases = [
        phase for phase in loop.phases if phase.work.kind in ("copy", "load")
    ]
    mma_phases = []
    for phase in loop.phases:
        attrs = dict(phase.work.attrs)
        legacy_mma = phase.work.kind == "mma"
        object_mma = (
            phase.work.kind == "compute"
            and attrs.get("engine") == "tensor"
            and attrs.get("op") in ("mma", "gemm")
        )
        if legacy_mma or object_mma:
            mma_phases.append(phase)
    if len(copy_phases) != 2 or len(mma_phases) != 1:
        raise ModelingValidationError(
            "general GEMM mainloop needs exactly two copy/load phases and one "
            "tensor MMA/GEMM compute phase"
        )
    tm, tn, tk = tile
    expected = {
        "a": float(tm * tk * a_dtype.storage_bytes),
        "b": float(tk * tn * b_dtype.storage_bytes),
    }
    expected_dtype = {"a": a_dtype, "b": b_dtype}

    by_shape = {}
    for phase in copy_phases:
        shapes = tuple(tuple(buffer.shape) for buffer in phase.writes)
        if (tm, tk) in shapes:
            by_shape["a"] = phase
        if (tk, tn) in shapes:
            by_shape["b"] = phase
    if len(by_shape) != 2 or by_shape["a"] == by_shape["b"]:
        by_name = {phase.name.lower(): phase for phase in copy_phases}
        a_matches = [phase for name, phase in by_name.items() if name.endswith("a")]
        b_matches = [phase for name, phase in by_name.items() if name.endswith("b")]
        if len(a_matches) == 1 and len(b_matches) == 1:
            by_shape = {"a": a_matches[0], "b": b_matches[0]}
        else:
            ordered = sorted(copy_phases, key=lambda phase: phase.name)
            if expected["a"] == expected["b"]:
                # Equal square tiles are semantically symmetric for this cost
                # binding.  A stable name order avoids hidden object identity.
                by_shape = {"a": ordered[0], "b": ordered[1]}
            else:
                raise ModelingValidationError(
                    "cannot identify GEMM A/B copy phases from buffer shapes or names"
                )
    for key, phase in by_shape.items():
        if not math.isclose(
            phase.work.bytes, expected[key], rel_tol=0.0, abs_tol=1e-12
        ):
            raise ModelingValidationError(
                "GEMM %s copy Work.bytes disagrees with semantic tile" % key.upper()
            )
        if (
            len(phase.writes) != 1
            or as_dtype(phase.writes[0].dtype) != expected_dtype[key]
        ):
            raise ModelingValidationError(
                "GEMM %s load buffer dtype disagrees with semantic storage dtype"
                % key.upper()
            )
        attrs = dict(phase.work.attrs)
        if (
            "dtype" in attrs
            and as_dtype(attrs["dtype"]) != expected_dtype[key]
        ):
            raise ModelingValidationError(
                "GEMM %s load Work dtype disagrees with semantic storage dtype"
                % key.upper()
            )
        if phase.work.kind == "load":
            if (
                attrs.get("source") != "ddr"
                or attrs.get("destination") != "smem"
                or attrs.get("engine") not in ("tma", "cp_async", "ldst")
            ):
                raise ModelingValidationError(
                    "object GEMM load phases must explicitly load ddr->smem "
                    "through tma/cp_async/ldst"
                )
    return by_shape["a"], by_shape["b"], mma_phases[0]


def extract_general_gemm_schedule(
    op: GemmOpSpec, kernel: KernelIR, arch: Any
) -> GemmScheduleSpec:
    """Validate a non-clustered GEMM schedule against semantic operation data."""

    if not isinstance(op, GemmOpSpec):
        raise ModelingValidationError("general GEMM expects GemmOpSpec")
    if not isinstance(kernel, KernelIR):
        raise ModelingValidationError("general GEMM expects KernelIR")
    if getattr(arch, "core", None) == "B200":
        raise ModelingValidationError(
            "general_gemm v1 does not bind B200 CTA-pair/cluster semantics"
        )
    params = dict(kernel.params)
    launch, loop = _select_launch_loop(kernel)
    if tuple(launch.cluster) != (1, 1, 1):
        raise ModelingValidationError(
            "general_gemm v1 supports only cluster=(1,1,1); cluster topology "
            "is not resident CTAs/SM"
        )
    if kernel.fusion_plans:
        raise ModelingValidationError(
            "general_gemm standalone profiles do not consume FusionPlan traffic"
        )

    shape_param = params.get("shape", params.get("op_shape"))
    semantic_shape = (op.m, op.n, op.k)
    if shape_param is not None and _int_tuple(shape_param, "GEMM shape", 3) != semantic_shape:
        raise ModelingValidationError(
            "KernelIR GEMM shape disagrees with GemmOpSpec; semantic op is source of truth"
        )
    tile_value = params.get("tile", params.get("tb_shape"))
    warp_value = params.get("warp_tile", params.get("wp_shape"))
    if tile_value is None or warp_value is None:
        raise ModelingValidationError(
            "general GEMM schedule params require tile and warp_tile"
        )
    tile = _int_tuple(tile_value, "GEMM tile", 3)
    warp = _int_tuple(warp_value, "GEMM warp_tile", 3)
    for axis, tile_extent, warp_extent in zip("MNK", tile, warp):
        if tile_extent % warp_extent:
            raise ModelingValidationError(
                "GEMM tile %s must be divisible by warp_tile %s" % (axis, axis)
            )
    if loop.stages is None:
        raise ModelingValidationError("general GEMM mainloop needs stages")
    stages = _positive_int(loop.stages, "GEMM stages")
    iterations = int(math.ceil(float(op.k) / tile[2]))
    if loop.iterations != iterations:
        raise ModelingValidationError(
            "GEMM loop iterations must equal ceil(K/tile_K)=%d" % iterations
        )
    if op.batch != int(params.get("batch", op.batch)):
        raise ModelingValidationError("KernelIR batch disagrees with GemmOpSpec")

    grid = (
        int(math.ceil(float(op.m) / tile[0])),
        int(math.ceil(float(op.n) / tile[1])),
    )
    if tuple(launch.work_grid) != (grid[0], grid[1], 1):
        raise ModelingValidationError("GEMM work_grid disagrees with semantic shape/tile")
    if tuple(launch.physical_grid) != (grid[0], grid[1], 1):
        raise ModelingValidationError(
            "general_gemm v1 requires a static one-CTA-per-work-tile grid"
        )
    if launch.scheduler != "static" or launch.residency != "auto":
        raise ModelingValidationError(
            "general_gemm v1 requires static scheduling and auto occupancy"
        )
    expected_warps = int(tile[0] / warp[0] * tile[1] / warp[1])
    if launch.threads != expected_warps * 32:
        raise ModelingValidationError("GEMM launch threads disagree with warp tiling")

    mma_type = str(params.get("mma_type", "wgmma"))
    if mma_type not in _SUPPORTED_MMA_TYPES:
        raise ModelingValidationError(
            "general_gemm v1 supports wmma/wgmma only; B200 CTA-pair is separate"
        )
    mem_levels = _canonical_mem_levels(op, params)
    load_a, load_b, mma = _identify_gemm_phases(
        loop,
        tile,
        op.a_storage_dtype,
        op.b_storage_dtype,
    )
    if (
        op.a_storage_dtype != op.compute_dtype
        or op.b_storage_dtype != op.compute_dtype
    ):
        raise ModelingValidationError(
            "general_gemm v1 requires A/B storage_dtype == compute_dtype; "
            "a storage-to-compute cast must be an explicit phase and is not "
            "supported by this three-phase adapter"
        )
    expected_flops = float(2 * tile[0] * tile[1] * tile[2])
    if not math.isclose(mma.work.flops, expected_flops, rel_tol=0.0, abs_tol=1e-12):
        raise ModelingValidationError(
            "GEMM MMA Work.flops must equal 2*tile_M*tile_N*tile_K"
        )
    mma_attrs = dict(mma.work.attrs)
    missing_mma_attrs = [
        name
        for name in ("input_dtype", "accumulator_dtype")
        if name not in mma_attrs
    ]
    if missing_mma_attrs:
        raise ModelingValidationError(
            "GEMM MMA Work attrs must explicitly provide %s"
            % ", ".join(missing_mma_attrs)
        )
    if as_dtype(mma_attrs["input_dtype"]) != op.compute_dtype:
        raise ModelingValidationError(
            "GEMM MMA input_dtype disagrees with semantic compute_dtype"
        )
    if as_dtype(mma_attrs["accumulator_dtype"]) != op.accumulation_dtype:
        raise ModelingValidationError(
            "GEMM MMA accumulator_dtype disagrees with semantic accumulation_dtype"
        )
    if "shape" in mma_attrs and tuple(mma_attrs["shape"]) != tile:
        raise ModelingValidationError(
            "GEMM MMA shape attr disagrees with the scheduled tile"
        )
    if len(mma.writes) != 1 or as_dtype(mma.writes[0].dtype) != op.accumulation_dtype:
        raise ModelingValidationError(
            "GEMM MMA accumulator buffer dtype disagrees with semantic accumulation_dtype"
        )
    if "dtype" in params and as_dtype(params["dtype"]) != op.compute_dtype:
        raise ModelingValidationError(
            "KernelIR dtype disagrees with semantic GEMM compute_dtype"
        )
    if (
        "accumulator_dtype" in params
        and as_dtype(params["accumulator_dtype"]) != op.accumulation_dtype
    ):
        raise ModelingValidationError(
            "KernelIR accumulator_dtype disagrees with semantic GEMM accumulation_dtype"
        )
    copy_resources = {
        _phase_resource(load_a, "copy"),
        _phase_resource(load_b, "copy"),
    }
    if len(copy_resources) != 1:
        raise ModelingValidationError(
            "legacy-compatible GEMM binding needs one shared copy resource"
        )
    tensor_resource = _phase_resource(mma, "tensor")

    row_panel = _positive_int(params.get("row_panel", 1), "GEMM row_panel")
    column_panel = params.get("column_panel")
    if column_panel is not None:
        column_panel = _positive_int(column_panel, "GEMM column_panel")
    raster_axis = str(params.get("raster_axis", "legacy"))
    if raster_axis not in ("legacy", "along_m", "along_n"):
        raise ModelingValidationError("unsupported GEMM raster_axis")
    return GemmScheduleSpec(
        launch_name=launch.name,
        loop_name=loop.name,
        tile=tile,
        warp_tile=warp,
        stages=stages,
        grid=grid,
        iterations=iterations,
        threads=launch.threads,
        mma_type=mma_type,
        mem_levels=mem_levels,
        row_panel=row_panel,
        column_panel=column_panel,
        raster_axis=raster_axis,
        load_a_phase=load_a.name,
        load_b_phase=load_b.name,
        mma_phase=mma.name,
        copy_resource=next(iter(copy_resources)),
        tensor_resource=tensor_resource,
    )


def _stable_cache_problem(
    op: GemmOpSpec,
    schedule: GemmScheduleSpec,
    arch: Any,
    options: GeneralGemmOptions,
) -> CacheProblem:
    mem_levels = schedule.mem_level_dict()
    if schedule.iterations <= options.representative_k:
        raise ModelingValidationError(
            "representative_k must be smaller than GEMM K-tile iterations"
        )
    return gemm_cache_problem(
        op.m,
        op.n,
        op.k,
        schedule.tile[0],
        schedule.tile[1],
        schedule.tile[2],
        arch.l2_capacity,
        arch.sm_count,
        mem_levels,
        schedule.row_panel,
        l1_5_group_size=getattr(arch, "l1_5_group_size", 0),
        l1_5_capacity_per_group=getattr(
            arch, "l1_5_capacity_per_group", 180 * 1024
        ),
        l1_5_associativity=getattr(arch, "l1_5_associativity", 8),
        l1_5_cacheline_bytes=getattr(arch, "l1_5_cacheline_bytes", 128),
        column_panel=schedule.column_panel,
        raster_axis=schedule.raster_axis,
        reuse_unit_bytes=options.cache_reuse_unit_bytes,
        reduction_mode="stable_shadow_cohort",
        representative_k=options.representative_k,
        progress_jitter=options.progress_jitter,
        sampling=SamplingConfig(
            seed=options.cache_seed,
            sample_budget=options.cache_sample_budget,
        ),
    )


def build_general_gemm_cache_problem(
    op: GemmOpSpec,
    kernel: KernelIR,
    arch: Any,
    options: Optional[GeneralGemmOptions] = None,
) -> CacheProblem:
    """Build the stable cache problem independently for DSE reuse/sweeps."""

    if options is None:
        options = GeneralGemmOptions()
    if not isinstance(options, GeneralGemmOptions):
        raise ModelingValidationError("general GEMM options have the wrong type")
    if options.cache_policy != "stable_shadow":
        raise ModelingValidationError(
            "cache-problem construction supports stable_shadow only"
        )
    schedule = extract_general_gemm_schedule(op, kernel, arch)
    return _stable_cache_problem(op, schedule, arch, options)


def build_general_gemm_cache_histogram(problem: CacheProblem) -> ReuseHistogram:
    """Build traversal reuse once; capacity-only evaluations may reuse it."""

    if not isinstance(problem, CacheProblem):
        raise ModelingValidationError("expected CacheProblem")
    return build_reuse_histogram(problem)


def evaluate_general_gemm_cache(
    problem: CacheProblem,
    histogram: Optional[ReuseHistogram] = None,
) -> CacheResult:
    """Evaluate a problem with a new or prebuilt reuse histogram."""

    if not isinstance(problem, CacheProblem):
        raise ModelingValidationError("expected CacheProblem")
    if histogram is None:
        histogram = build_reuse_histogram(problem)
    if not isinstance(histogram, ReuseHistogram):
        raise ModelingValidationError("expected ReuseHistogram")
    return evaluate_histogram(problem, histogram)


def _cache_summary(
    op: GemmOpSpec,
    schedule: GemmScheduleSpec,
    arch: Any,
    options: GeneralGemmOptions,
) -> GemmCacheSummary:
    mem_levels = schedule.mem_level_dict()
    if schedule.iterations <= options.representative_k:
        raise ModelingValidationError(
            "representative_k must be smaller than GEMM K-tile iterations"
        )
    if options.cache_policy == "legacy_rng":
        if options.cache_histogram is not None:
            raise ModelingValidationError(
                "legacy_rng cannot consume a stable cache histogram"
            )
        if options.progress_jitter.enabled:
            raise ModelingValidationError(
                "legacy_rng cache compatibility does not support progress_jitter"
            )
        # Explicit parity-only path.  Save/restore NumPy's process-global state
        # so a cost query cannot perturb the caller's later samples.
        import numpy as np
        from tilesight.modeling._cache_compat.cache_model import multi_level_hit_rate

        rng_state = np.random.get_state()
        try:
            np.random.seed(options.cache_seed)
            result = multi_level_hit_rate(
                op.m,
                op.n,
                op.k,
                schedule.tile[0],
                schedule.tile[1],
                schedule.tile[2],
                arch,
                mem_levels,
                schedule.row_panel,
                column_panel=schedule.column_panel,
                raster_axis=schedule.raster_axis,
            )
        finally:
            np.random.set_state(rng_state)
        return GemmCacheSummary(
            backend="legacy_multi_level_rng",
            problem_digest=None,
            l1_5_hit_rate=float(result["l1_5_hit_rate"]),
            l2_hit_rate=float(result["l2_hit_rate"]),
            ddr_miss_rate=float(result["ddr_miss_rate"]),
            reuse_unit_bytes=float(
                getattr(arch, "l1_5_cacheline_bytes", 128)
            ),
            reduction_fidelity="legacy_sampled_inner",
            sampling_mode="legacy_rng",
        )

    problem = _stable_cache_problem(op, schedule, arch, options)
    result = evaluate_general_gemm_cache(problem, options.cache_histogram)
    representative_ddr_read = result.traffic.ddr_read_bytes
    return GemmCacheSummary(
        backend=result.backend,
        problem_digest=result.problem_digest,
        l1_5_hit_rate=result.aggregate.l1_5_hit_rate,
        l2_hit_rate=result.aggregate.l2_hit_rate,
        ddr_miss_rate=result.aggregate.ddr_miss_rate,
        reuse_unit_bytes=problem.l2.unit_bytes,
        reduction_fidelity=result.histogram.reduction_fidelity,
        histogram_reused=options.cache_histogram is not None,
        sampling_mode=(
            "full_traversal_audit"
            if options.cache_sample_budget is None
            else "systematic_budget_%d" % options.cache_sample_budget
        ),
        whole_grid_ddr_read_bytes=(
            representative_ddr_read * schedule.iterations * op.batch
        ),
        # Stores are true one-shot epilogues for every spatial work tile.  They
        # do not repeat for the stable representative K iteration.
        whole_grid_ddr_write_bytes=result.traffic.ddr_write_bytes * op.batch,
        representative_grid_ddr_read_bytes=representative_ddr_read,
    )


def _compute_overhead(op: GemmOpSpec, schedule: GemmScheduleSpec, arch: Any) -> float:
    try:
        minimum = arch.get_tensor_core_minimum_ptx(
            bytes=op.compute_dtype.storage_bytes
        )
    except AttributeError as error:
        raise ModelingValidationError(
            "GEMM architecture needs get_tensor_core_minimum_ptx"
        ) from error
    wm, wn, wk = schedule.warp_tile
    # Preserve the calibrated model's historical cap exactly.  It normally
    # evaluates to one for legal tensor-core tiles.
    return float(
        min(
            1.0,
            math.ceil(wm) / minimum[0]
            * math.ceil(wn) / minimum[1]
            * math.ceil(wk) / minimum[2],
        )
    )


def _resource_costs(
    op: GemmOpSpec,
    schedule: GemmScheduleSpec,
    cache: GemmCacheSummary,
    tensor_throughput: ThroughputResolution,
    arch: Any,
) -> Tuple[
    GemmPerIterationCost,
    GemmEpilogueCost,
    GemmFootprint,
    GemmServiceTimes,
    float,
    TileResources,
]:
    tm, tn, tk = schedule.tile
    wm, wn, wk = schedule.warp_tile
    levels = schedule.mem_level_dict()
    a = levels["in1"]
    b = levels["in2"]
    out = levels["out1"]
    compute_overhead = _compute_overhead(op, schedule, arch)

    logical_a = float(tm * tk * op.a_storage_dtype.storage_bytes)
    logical_b = float(tk * tn * op.b_storage_dtype.storage_bytes)
    load_bytes = float(tk * (tm * a[0] * a[-1] + tn * b[0] * b[-1]))
    l1_5_io = load_bytes
    l2_load = load_bytes * (1.0 - cache.l1_5_hit_rate)
    ddr_io = l2_load * (1.0 - cache.l2_hit_rate)
    if getattr(arch, "core", None) in _LEGACY_L2_TWO_PART_CORES:
        l2_io = l2_load * cache.l2_hit_rate + 2.0 * l2_load * (
            1.0 - cache.l2_hit_rate
        )
    else:
        l2_io = l2_load
    tensor_flops = float(2 * tm * tn * tk) * compute_overhead

    active_warps = float(tm) / wm * float(tn) / wn
    if schedule.mma_type == "wgmma":
        smem_io = float(2 * tk * (tm * a[0] * a[-1] + tn * b[0] * b[-1]))
    elif getattr(arch, "core", None) == "V100":
        store_shared = tk * (tm * a[0] * a[-1] + tn * b[0] * b[-1])
        shared_load = (
            tk
            * (tm * a[-1] + tn * b[-1])
            * active_warps
            * (wm * a[1] * a[-1] + wn * b[1] * b[-1])
            / max(tm * a[-1] + tn * b[-1], 1.0)
        )
        smem_io = float(store_shared + shared_load)
    else:
        ldgsts_factor = (
            0.5
            if getattr(arch, "core", None) in ("A100", "H100")
            else 1.0
        )
        ldgsts = ldgsts_factor * tk * (
            tm * a[0] * a[-1] + tn * b[0] * b[-1]
        )
        shared_load = (
            tk
            * (tm * a[-1] + tn * b[-1])
            * active_warps
            * (wm * a[1] * a[-1] + wn * b[1] * b[-1])
            / max(tm * a[-1] + tn * b[-1], 1.0)
        )
        smem_io = float(ldgsts + shared_load)

    store_l2 = float(tm * tn * out[-1] * out[0])
    store_ddr = store_l2
    if getattr(arch, "core", None) in _LEGACY_L2_TWO_PART_CORES:
        store_l2 *= 2.0
    store_smem = float(tm * tn * out[-1] * out[1])

    smem_footprint = float(
        (
            tm * tk * (1.0 - (a[1] - a[0])) * a[-1]
            + tn * tk * (1.0 - (b[1] - b[0])) * b[-1]
            + tm * tn * (out[1] - out[0]) * out[-1]
        )
        * schedule.stages
    )
    accumulator_bytes = op.accumulation_dtype.storage_bytes
    register_footprint = (
        math.ceil(wm * wn / 32.0 / (4.0 / accumulator_bytes))
        + math.ceil(wm * wk / 32.0 / (4.0 / a[-1])) * a[1]
        + math.ceil(wn * wk / 32.0 / (4.0 / b[-1])) * b[1]
    )
    register_footprint = float(math.ceil(register_footprint * 1.1))
    warps = int(active_warps)
    occupancy = compute_occupancy(
        smem_footprint,
        register_footprint,
        warps,
        arch,
        schedule.mma_type,
    )

    per_iter = GemmPerIterationCost(
        logical_a_copy_bytes=logical_a,
        logical_b_copy_bytes=logical_b,
        l1_5_io_bytes=l1_5_io,
        l2_io_bytes=l2_io,
        ddr_io_bytes=ddr_io,
        smem_io_bytes=smem_io,
        tensor_flops=tensor_flops,
    )
    epilogue = GemmEpilogueCost(
        ddr_io_bytes=store_ddr,
        l2_io_bytes=store_l2,
        l1_5_io_bytes=0.0,
        smem_io_bytes=store_smem,
    )
    footprint = GemmFootprint(
        smem_bytes_per_cta=smem_footprint,
        legacy_registers_per_thread=register_footprint,
        warps_per_cta=warps,
        resident_ctas_per_sm=occupancy,
    )

    ddr_t, l2_t, l1_5_t, smem_t, _ignored_tensor_t = resources_to_times(
        ddr_io,
        l2_io,
        smem_io,
        0.0,
        arch,
        op.compute_dtype.storage_bytes,
        l1_5_io=l1_5_io,
    )
    sm_count = int(arch.sm_count)
    ddr_t *= sm_count
    l2_t *= sm_count
    l1_5_t *= sm_count
    smem_t *= sm_count
    tensor_t = tensor_flops / tensor_throughput.require() * sm_count
    ddr_t /= arch.ddr_max_util
    l2_t /= arch.l2_max_util
    l1_5_t /= getattr(arch, "l1_5_max_util", arch.l1_max_util)
    smem_t /= arch.l1_max_util
    tensor_t /= arch.compute_max_util
    memory_t = max(ddr_t, l2_t, l1_5_t, smem_t)
    service = GemmServiceTimes(
        ddr_s=ddr_t,
        l2_s=l2_t,
        l1_5_s=l1_5_t,
        smem_s=smem_t,
        memory_s=memory_t,
        tensor_s=tensor_t,
        resource_ii_s=max(memory_t, tensor_t),
    )

    low_per_iter = PerIterationResources(
        ddr_io=ddr_io,
        l2_io=l2_io,
        l1_5_io=l1_5_io,
        smem_io=smem_io,
        compute_flops=tensor_flops,
    )
    boundary = PrologueEpilogueResources(
        prologue_ddr_io=ddr_io * max(schedule.stages - 1, 0),
        prologue_l2_io=l2_io * max(schedule.stages - 1, 0),
        prologue_l1_5_io=l1_5_io * max(schedule.stages - 1, 0),
        prologue_smem_io=smem_io * max(schedule.stages - 1, 0),
        epilogue_compute_flops=tensor_flops * max(schedule.stages - 1, 0),
        store_ddr_io=store_ddr,
        store_l2_io=store_l2,
        store_l1_5_io=0.0,
        store_smem_io=store_smem,
    )
    tile_resources = TileResources(
        per_iter=low_per_iter,
        prologue_epilogue=boundary,
        num_iterations=schedule.iterations,
        stage_num=schedule.stages,
        smem_footprint=smem_footprint,
        reg_footprint=register_footprint,
        warps_per_block=warps,
        grids=schedule.grid,
    )
    return (
        per_iter,
        epilogue,
        footprint,
        service,
        compute_overhead,
        tile_resources,
    )


def _throughput_bound_pipeline_arch(
    arch: Any,
    compute_dtype_bytes: int,
    capacity_per_s: float,
) -> Any:
    """Transport a resolved capacity through the legacy low-level pipeline.

    The low-level helper still dispatches on byte width.  This private copy
    makes that dispatch carry the already-resolved semantic compute capacity;
    byte width never selects the capacity itself in this adapter.
    """

    bound = copy.copy(arch)
    if compute_dtype_bytes == 1:
        # The legacy helper prefers FP8 when the attribute exists, including
        # for an explicitly resolved INT8 request.  Override that transport
        # slot with the typed resolver's capacity.
        setattr(bound, "fp8_tensor_flops", capacity_per_s)
        setattr(bound, "int8_tensor_flops", capacity_per_s)
    elif compute_dtype_bytes == 2:
        setattr(bound, "fp16_tensor_flops", capacity_per_s)
    elif compute_dtype_bytes == 4:
        setattr(bound, "fp32_tensor_flops", capacity_per_s)
    else:
        raise ModelingValidationError(
            "legacy low-level pipeline cannot transport this compute dtype width"
        )
    return bound


def _replace_loop_timings(
    kernel: KernelIR,
    launch_name: str,
    loop_name: str,
    timings: Mapping[str, Timing],
) -> KernelIR:
    """Return a frozen kernel whose selected loop uses oracle-bound timings."""

    launches = []
    for launch in kernel.launches:
        if launch.name != launch_name:
            launches.append(launch)
            continue
        loops = []
        for loop in launch.periodic_loops:
            if loop.name != loop_name:
                loops.append(loop)
                continue
            phase_map = {
                phase.name: replace(phase, timing=timings[phase.name])
                for phase in loop.phases
            }
            actors = tuple(
                replace(
                    actor,
                    phases=tuple(phase_map[item.name] for item in actor.phases),
                    sequence=tuple(
                        replace(item, phase=phase_map[item.phase.name])
                        for item in actor.sequence
                    ),
                )
                for actor in loop.actors
            )
            resource_sequences = tuple(
                replace(
                    item,
                    sequence=tuple(
                        replace(
                            occurrence,
                            phase=phase_map[occurrence.phase.name],
                        )
                        for occurrence in item.sequence
                    ),
                )
                for item in loop.resource_sequences
            )
            loops.append(
                replace(
                    loop,
                    actors=actors,
                    resource_sequences=resource_sequences,
                )
            )
        launches.append(replace(launch, periodic_loops=tuple(loops)))
    return replace(kernel, launches=tuple(launches))


class BoundGemmCostOracle:
    """Architecture-bound timing oracle for one semantic GEMM schedule.

    ``resolve`` is compatible with the frontend ``CostOracle`` protocol.  The
    oracle also provides ``lower_periodic`` because static example timings
    otherwise take precedence over an oracle during generic lowering.
    """

    def __init__(
        self,
        op: GemmOpSpec,
        kernel: KernelIR,
        arch: Any,
        options: Optional[GeneralGemmOptions] = None,
    ) -> None:
        if options is None:
            options = GeneralGemmOptions()
        if not isinstance(options, GeneralGemmOptions):
            raise ModelingValidationError("general GEMM options have the wrong type")
        schedule = extract_general_gemm_schedule(op, kernel, arch)
        tensor_throughput = resolve_throughput(
            arch,
            ThroughputRequest(
                engine=op.engine,
                dtype=op.compute_dtype,
                op_class="mma",
            ),
            options.tensor_throughput_fallback,
        )
        tensor_capacity = tensor_throughput.require()
        cache = _cache_summary(op, schedule, arch, options)
        (
            per_iteration,
            epilogue,
            footprint,
            service,
            compute_overhead,
            tile_resources,
        ) = _resource_costs(
            op,
            schedule,
            cache,
            tensor_throughput,
            arch,
        )
        self.op = op
        self.kernel = kernel
        self.arch = arch
        self.options = options
        self.tile_resources = tile_resources
        self.pipeline_arch = _throughput_bound_pipeline_arch(
            arch,
            op.compute_dtype.storage_bytes,
            tensor_capacity,
        )
        self.breakdown = GemmCostBreakdown(
            schedule=schedule,
            cache=cache,
            per_iteration=per_iteration,
            epilogue=epilogue,
            footprint=footprint,
            service=service,
            tensor_throughput=tensor_throughput,
            compute_overhead=compute_overhead,
            provenance=_attrs(
                binder="BoundGemmCostOracle",
                cache_policy=options.cache_policy,
                cache_reduction=cache.reduction_fidelity,
                cache_reuse_unit_bytes=cache.reuse_unit_bytes,
                cache_sampling_mode=cache.sampling_mode,
                cache_sample_budget=options.cache_sample_budget,
                cache_histogram_reused=cache.histogram_reused,
                cache_progress_jitter_enabled=options.progress_jitter.enabled,
                cache_progress_jitter_max_k_ahead=(
                    options.progress_jitter.max_k_ahead
                ),
                cache_progress_jitter_capacity_fraction=(
                    options.progress_jitter.capacity_cap_fraction
                ),
                cache_progress_jitter_seed=options.progress_jitter.seed,
                cluster_support="cluster1_only",
                resource_scope="full_wave_per_sm",
                tensor_compute_dtype=op.compute_dtype.name,
                tensor_throughput_source=tensor_throughput.source,
                tensor_throughput_fallback=tensor_throughput.fallback,
                tensor_throughput_dtype_aliases=(
                    options.tensor_throughput_fallback.dtype_aliases
                ),
                units="SI_seconds_and_bytes",
            ),
        )
        loop = kernel.periodic_loop(schedule.loop_name)
        phases = {phase.name: phase for phase in loop.phases}
        copy_total = (
            phases[schedule.load_a_phase].work.bytes
            + phases[schedule.load_b_phase].work.bytes
        )
        if service.memory_s <= 0.0 or copy_total <= 0.0:
            raise ModelingValidationError(
                "general GEMM needs positive copy work and memory service"
            )
        timings = {}
        for phase_name in (schedule.load_a_phase, schedule.load_b_phase):
            share = phases[phase_name].work.bytes / copy_total
            duration = service.memory_s * share
            timings[phase_name] = Timing(
                latency=duration,
                resources=(
                    ResourceTiming(schedule.copy_resource, duration),
                ),
            )
        timings[schedule.mma_phase] = Timing(
            latency=service.tensor_s,
            resources=(
                ResourceTiming(schedule.tensor_resource, service.tensor_s),
            ),
        )
        if set(timings) != set(phases):
            raise ModelingValidationError(
                "general GEMM v1 cannot bind auxiliary mainloop phases"
            )
        self._timings = timings
        self._loop_owner = loop.owner
        self.bound_kernel = _replace_loop_timings(
            kernel,
            schedule.launch_name,
            schedule.loop_name,
            timings,
        )

    def resolve(self, phase: Any) -> Timing:
        try:
            if phase.owner != self._loop_owner:
                raise KeyError(phase.name)
            return self._timings[phase.name]
        except (AttributeError, KeyError) as error:
            raise ModelingValidationError(
                "phase is not part of this bound GEMM mainloop"
            ) from error

    def lower_periodic(self) -> PeriodicDAG:
        return self.bound_kernel.lower_periodic(self.breakdown.schedule.loop_name)


def bind_general_gemm(
    op: GemmOpSpec,
    kernel: KernelIR,
    arch: Any,
    options: Optional[GeneralGemmOptions] = None,
) -> BoundGemmCostOracle:
    """Bind a semantic GEMM and a schedule without choosing an executor."""

    return BoundGemmCostOracle(op, kernel, arch, options)


def _legacy_calibrated(
    oracle: BoundGemmCostOracle,
) -> LegacyCalibratedGemmResult:
    op = oracle.op
    # A private arch copy transports the semantic resolver's capacity through
    # the legacy low-level pipeline helper.  All non-throughput fields are
    # unchanged.
    arch = oracle.pipeline_arch
    breakdown = oracle.breakdown
    schedule = breakdown.schedule
    tile_resources = oracle.tile_resources
    occupancy = breakdown.footprint.resident_ctas_per_sm
    data_bytes = op.compute_dtype.storage_bytes
    full_sm_latency, full_detail = compute_pipeline_tile_latency_with_occupancy(
        tile_resources,
        occupancy,
        arch,
        data_bytes=data_bytes,
    )
    total_tiles = schedule.grid[0] * schedule.grid[1]
    total_latency, wave_info = compute_wave_adjusted_latency(
        full_sm_latency,
        occupancy,
        total_tiles,
        arch,
        pipeline_detail=full_detail,
        mma_type=schedule.mma_type,
        tile_res=tile_resources,
        data_bytes=data_bytes,
    )
    actual_active_sms = min(total_tiles, int(arch.sm_count))
    actual_tiles_per_sm = (
        1
        if total_tiles <= arch.sm_count
        else min(int(math.ceil(float(total_tiles) / arch.sm_count)), occupancy)
    )
    per_tile, detail = compute_pipeline_tile_latency_with_occupancy(
        tile_resources,
        actual_tiles_per_sm,
        arch,
        data_bytes=data_bytes,
        active_sms=actual_active_sms,
    )
    total_latency *= op.batch

    levels = schedule.mem_level_dict()
    tm, tn, _tk = schedule.tile
    total_l2_read = (
        schedule.grid[0]
        * schedule.grid[1]
        * op.k
        * (
            tm * levels["in1"][0] * levels["in1"][-1]
            + tn * levels["in2"][0] * levels["in2"][-1]
        )
    )
    total_compute = op.m * op.n * op.k * 2 * breakdown.compute_overhead * op.batch
    total_ddr = (
        total_l2_read
        * (1.0 - breakdown.cache.l2_hit_rate)
        * op.batch
    )
    ddr_roof = total_ddr / arch.ddr_bandwidth if arch.ddr_bandwidth > 0 else 0.0
    compute_roof = total_compute / breakdown.tensor_throughput.require()
    return LegacyCalibratedGemmResult(
        per_tile_latency_s=float(per_tile),
        kernel_body_s=float(total_latency),
        ddr_util=ddr_roof / total_latency if total_latency > 0.0 else 0.0,
        compute_util=compute_roof / total_latency if total_latency > 0.0 else 0.0,
        l2_hit_rate=breakdown.cache.l2_hit_rate,
        resident_ctas_per_sm=occupancy,
        waves=float(wave_info["waves_float"]),
        prologue_s=float(detail.prologue_time),
        steady_time_per_iter_s=float(detail.steady_time_per_iter),
        epilogue_s=float(detail.epilogue_time),
        mem_time_per_iter_s=float(detail.mem_time_per_iter),
        compute_time_per_iter_s=float(detail.compute_time_per_iter),
        wave_info=tuple(sorted(wave_info.items())),
        provenance=_attrs(
            monolithic_entry_point_called=False,
            occupancy_policy="tilesight.modeling._pipeline.occupancy.compute_occupancy",
            pipeline_policy=(
                "tilesight.modeling._pipeline.pipeline_overlap.compute_pipeline_tile_latency_with_occupancy"
            ),
            wave_policy="tilesight.modeling._pipeline.wave_model.compute_wave_adjusted_latency",
            parity_cache_policy=oracle.options.cache_policy,
            scope="kernel_body_excluding_launch",
        ),
    )


def _natural_boundary(
    dag: PeriodicDAG,
    selection: IIScheduleSelection,
    iterations: int,
) -> GemmNaturalBoundary:
    witness = selection.witness
    if witness is None:
        raise ModelingValidationError(
            "natural periodic boundary requires a constructive witness"
        )
    starts = []
    completions = []
    for phase in dag.phases:
        phase_start = witness.phase_starts[phase.name]
        # ``iteration_offset`` labels which logical iteration the occurrence
        # belongs to.  Clip logical iterations, then recover its base period.
        for logical_iteration in range(iterations):
            base_iteration = logical_iteration - phase.iteration_offset
            start = base_iteration * witness.ii + phase_start
            starts.append(start)
            completions.append(start + phase.latency)
    first = min(starts)
    last = max(completions)
    return GemmNaturalBoundary(
        first_start_s=first,
        last_completion_s=last,
        latency_s=last - first,
        occurrence_count=len(starts),
        iterations=iterations,
    )


def model_general_gemm(
    op: GemmOpSpec,
    kernel: KernelIR,
    arch: Any,
    options: Optional[GeneralGemmOptions] = None,
) -> GeneralGemmResult:
    """Evaluate a semantic GEMM through one explicit profile."""

    if options is None:
        options = GeneralGemmOptions()
    oracle = bind_general_gemm(op, kernel, arch, options)
    dag = oracle.lower_periodic()
    resource_ii = compute_lower_bounds(dag).resource_ii
    tolerance = 1e-15 + 1e-12 * max(
        resource_ii, oracle.breakdown.service.resource_ii_s
    )
    if abs(resource_ii - oracle.breakdown.service.resource_ii_s) > tolerance:
        raise ModelingValidationError(
            "bound phase resources disagree with typed GEMM resource II"
        )
    mode = options.selected_ii_mode
    if options.profile == "native_periodic" and mode == "resource_ii":
        raise ModelingValidationError(
            "native_periodic requires periodic_best/worst so natural boundary "
            "has a constructive witness"
        )
    if options.profile == "legacy_calibrated" and mode != "resource_ii":
        raise ModelingValidationError(
            "legacy_calibrated uses its calibrated overlap equation; select "
            "native_periodic for a periodic witness"
        )

    envelope = None  # type: Optional[ScheduleEnvelope]
    if mode == "resource_ii":
        selection = select_steady_ii(resource_ii, mode)
    else:
        envelope = schedule_periodic_dag(dag, options.search_config)
        selection = select_steady_ii(
            resource_ii,
            mode,
            solve_periodic=lambda: envelope,
        )
    natural = (
        None
        if selection.witness is None
        else _natural_boundary(dag, selection, oracle.breakdown.schedule.iterations)
    )
    calibrated = (
        _legacy_calibrated(oracle)
        if options.profile == "legacy_calibrated"
        else None
    )
    return GeneralGemmResult(
        profile=options.profile,
        breakdown=oracle.breakdown,
        dag=dag,
        ii=selection,
        envelope=envelope,
        natural_boundary=natural,
        legacy_calibrated=calibrated,
        provenance=_attrs(
            adapter="general_gemm",
            cache_policy=options.cache_policy,
            full_grid_latency=(
                "legacy_calibrated_low_level_policy"
                if calibrated is not None
                else "not_claimed_from_single_cta_periodic_witness"
            ),
            ii_scope=selection.endpoint_scope,
            profile=options.profile,
            units="SI_seconds",
        ),
    )


evaluate_general_gemm = model_general_gemm


__all__ = [
    "BoundGemmCostOracle",
    "GeneralGemmOptions",
    "GeneralGemmResult",
    "GemmCacheSummary",
    "GemmCostBreakdown",
    "GemmEpilogueCost",
    "GemmFootprint",
    "GemmNaturalBoundary",
    "GemmPerIterationCost",
    "GemmScheduleSpec",
    "GemmServiceTimes",
    "LegacyCalibratedGemmResult",
    "bind_general_gemm",
    "build_general_gemm_cache_histogram",
    "build_general_gemm_cache_problem",
    "evaluate_general_gemm",
    "evaluate_general_gemm_cache",
    "extract_general_gemm_schedule",
    "model_general_gemm",
]
