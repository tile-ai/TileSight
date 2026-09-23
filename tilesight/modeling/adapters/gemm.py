"""Full-kernel compatibility adapter for the calibrated GEMM model.

The schedule frontend deliberately does not duplicate TileSight's existing
GEMM resource, cache, occupancy, and wave equations.  This module extracts a
fully specified legacy configuration from :class:`KernelIR`, invokes
``calculate_matmul_pipeline_wave`` once, and retains all of its reporting
fields.  A common full-model result wrapper is layered on top of this bridge.

Keeping extraction separate from evaluation has two useful properties:

* malformed or underspecified frontend programs fail before the cost model;
* parity tests can compare the old and new entry points with identical
  arguments (important because the current L1.5 cache simulator is sampled).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

from ..errors import ModelingValidationError
from ..full_model import (
    FullModelError,
    FullModelOptions,
    FullModelResult,
    GridResult,
    IISummary,
    OccupancyResult,
    TimingBreakdown,
)
from ..ir import KernelIR, LaunchIR, PeriodicLoopIR
from ..liveness import analyze_fusion, analyze_liveness


_MMA_TYPES = frozenset(
    ("wmma", "wgmma", "utcmma_cta1", "utcmma_cta2")
)
_RASTER_AXES = frozenset(("legacy", "along_m", "along_n"))


def _positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ModelingValidationError("%s must be a positive integer" % label)
    return value


def _positive_int_tuple(value: Any, label: str, length: int) -> Tuple[int, ...]:
    try:
        result = tuple(value)
    except TypeError as error:
        raise ModelingValidationError("%s must be a sequence" % label) from error
    if len(result) != length:
        raise ModelingValidationError(
            "%s must contain exactly %d dimensions" % (label, length)
        )
    return tuple(_positive_int(item, "%s dimension" % label) for item in result)


def _as_mapping(value: Any, label: str) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        result = dict(value)
    else:
        try:
            result = dict(value)
        except (TypeError, ValueError) as error:
            raise ModelingValidationError("%s must be a mapping" % label) from error
    if not all(isinstance(key, str) for key in result):
        raise ModelingValidationError("%s keys must be strings" % label)
    return result


def _memory_levels(value: Any) -> Tuple[Tuple[str, Tuple[Any, ...]], ...]:
    mapping = _as_mapping(value, "gemm mem_levels")
    required = ("in1", "in2", "out1")
    missing = [name for name in required if name not in mapping]
    if missing:
        raise ModelingValidationError(
            "gemm mem_levels is missing %s" % ", ".join(missing)
        )
    normalized = []
    for name in required:
        try:
            level = tuple(mapping[name])
        except TypeError as error:
            raise ModelingValidationError(
                "gemm mem_levels[%s] must be a sequence" % name
            ) from error
        if len(level) < 4:
            raise ModelingValidationError(
                "gemm mem_levels[%s] must have at least four entries" % name
            )
        normalized.append((name, level))
    return tuple(normalized)


def _select_launch_and_loop(
    kernel: KernelIR, params: Mapping[str, Any]
) -> Tuple[LaunchIR, PeriodicLoopIR]:
    launch_name = params.get("launch", None)
    if launch_name is None:
        if len(kernel.launches) != 1:
            raise ModelingValidationError(
                "full GEMM modeling needs one launch or params['launch']"
            )
        launch = kernel.launches[0]
    else:
        launches = [item for item in kernel.launches if item.name == launch_name]
        if len(launches) != 1:
            raise ModelingValidationError(
                "params['launch'] does not select exactly one launch"
            )
        launch = launches[0]

    loop_name = params.get("mainloop", None)
    if loop_name is None:
        if len(launch.periodic_loops) != 1:
            raise ModelingValidationError(
                "full GEMM modeling needs one periodic loop or params['mainloop']"
            )
        loop = launch.periodic_loops[0]
    else:
        loops = [item for item in launch.periodic_loops if item.name == loop_name]
        if len(loops) != 1:
            raise ModelingValidationError(
                "params['mainloop'] does not select exactly one periodic loop"
            )
        loop = loops[0]
    return launch, loop


@dataclass(frozen=True)
class GemmModelSpec:
    """Concrete arguments of the existing pipeline+wave GEMM entry point."""

    op_shape: Tuple[int, int, int]
    tb_shape: Tuple[int, int, int]
    wp_shape: Tuple[int, int, int]
    stage_num: int
    mem_levels: Tuple[Tuple[str, Tuple[Any, ...]], ...]
    row_panel: int = 1
    batch: int = 1
    mma_type: str = "wmma"
    column_panel: Optional[int] = None
    raster_axis: str = "legacy"
    launch_name: str = "main"
    loop_name: str = "ko"

    def legacy_mem_levels(self) -> Dict[str, list]:
        """Return the mutable list mapping expected by the legacy function."""

        return {name: list(level) for name, level in self.mem_levels}

    def legacy_kwargs(self, arch: Any) -> Dict[str, Any]:
        return {
            "op_shape": self.op_shape,
            "tb_shape": self.tb_shape,
            "wp_shape": self.wp_shape,
            "stage_num": self.stage_num,
            "arch": arch,
            "mem_levels": self.legacy_mem_levels(),
            "row_panel": self.row_panel,
            "batch": self.batch,
            "mma_type": self.mma_type,
            "column_panel": self.column_panel,
            "raster_axis": self.raster_axis,
        }


@dataclass(frozen=True)
class GemmWaveMetadata:
    """Integer wave/tail facts omitted by legacy ``PipelineResult``.

    A logical slot is one physical SM for CTA1 and one two-SM cooperative slot
    for effective UTCMMA2. It must not be interpreted as typed CTA/SM
    occupancy in the latter case.
    """

    work_grid: Tuple[int, int]
    total_work_units: int
    effective_sm_count: int
    effective_work_units_per_logical_slot: int
    work_units_per_wave: int
    full_waves: int
    tail_work_units: int
    total_waves: int
    waves_float: float
    cluster_axis: Optional[str]
    cluster_size: int
    effective_mma_type: str


@dataclass(frozen=True)
class GemmLegacyEvaluation:
    """Lossless legacy evaluation plus derived integer wave metadata."""

    spec: GemmModelSpec
    legacy_result: Any
    wave: GemmWaveMetadata
    arch_compatibility: Tuple[Tuple[str, Any], ...] = tuple()


class _B200LegacyArchView:
    """Read-only compatibility view for a field absent from the B200 arch.

    The legacy GEMM backend queries ``get_tensor_core_minimum_ptx`` on every
    architecture, so the proxy supplies the tcgen05 minimum instruction shape
    used by the existing B200 GEMM case study.  The source object is never
    mutated.
    """

    def __init__(self, source: Any) -> None:
        self._source = source

    def __getattr__(self, name: str) -> Any:
        return getattr(self._source, name)

    def get_tensor_core_minimum_ptx(self, bytes: int = 2) -> Tuple[int, int, int]:
        if bytes not in (1, 2, 4):
            raise ModelingValidationError(
                "B200 compatibility view supports 1/2/4-byte tensor operands"
            )
        return (16, 8, 16)


def gemm_legacy_arch_view(
    arch: Any,
) -> Tuple[Any, Tuple[Tuple[str, Any], ...]]:
    """Return an immutable legacy-backend view and explicit provenance."""

    core = getattr(arch, "core", None)
    missing_ptx = not callable(getattr(arch, "get_tensor_core_minimum_ptx", None))
    if not missing_ptx:
        return arch, (("mode", "native_arch_fields"),)
    if core != "B200":
        raise ModelingValidationError(
            "legacy GEMM architecture is missing required fields %s; compatibility "
            "defaults are scoped to B200 only" % ["get_tensor_core_minimum_ptx"]
        )
    view = _B200LegacyArchView(arch)
    provenance = (
        ("mode", "b200_read_only_legacy_compat_view"),
        ("tensor_core_minimum_ptx", (16, 8, 16)),
        (
            "tensor_core_shape_source",
            "existing_B200_tcgen05_case_study_contract",
        ),
        ("source_arch_mutated", False),
    )
    return view, provenance


def extract_gemm_spec(kernel: KernelIR) -> GemmModelSpec:
    """Validate and extract the compatibility configuration from ``kernel``."""

    if not isinstance(kernel, KernelIR):
        raise ModelingValidationError("full GEMM adapter expects KernelIR")
    params = dict(kernel.params)
    launch, loop = _select_launch_and_loop(kernel, params)
    if len(kernel.launches) != 1 or len(launch.periodic_loops) != 1:
        raise ModelingValidationError(
            "GEMM parity adapter requires exactly one launch and one mainloop"
        )

    shape_value = params.get("shape", params.get("op_shape", None))
    tile_value = params.get("tile", params.get("tb_shape", None))
    warp_value = params.get("warp_tile", params.get("wp_shape", None))
    if shape_value is None or tile_value is None or warp_value is None:
        raise ModelingValidationError(
            "full GEMM params require shape, tile, and warp_tile"
        )
    if "mem_levels" not in params:
        raise ModelingValidationError("full GEMM params require mem_levels")

    op_shape = _positive_int_tuple(shape_value, "gemm shape", 3)
    tb_shape = _positive_int_tuple(tile_value, "gemm tile", 3)
    wp_shape = _positive_int_tuple(warp_value, "gemm warp_tile", 3)
    for index, axis in enumerate(("M", "N", "K")):
        if tb_shape[index] % wp_shape[index] != 0:
            raise ModelingValidationError(
                "gemm tile %s must be divisible by warp_tile %s" % (axis, axis)
            )

    if loop.stages is None:
        raise ModelingValidationError("full GEMM mainloop requires stages")
    stage_num = _positive_int(loop.stages, "gemm stages")
    expected_iterations = int(math.ceil(op_shape[2] / tb_shape[2]))
    if loop.iterations != expected_iterations:
        raise ModelingValidationError(
            "GEMM loop iterations %r do not match ceil(K/tile_K)=%d"
            % (loop.iterations, expected_iterations)
        )

    raw_grid = (
        int(math.ceil(op_shape[0] / tb_shape[0])),
        int(math.ceil(op_shape[1] / tb_shape[1])),
    )

    mma_type = params.get("mma_type", "wmma")
    if mma_type not in _MMA_TYPES:
        raise ModelingValidationError(
            "gemm mma_type must be one of %s" % sorted(_MMA_TYPES)
        )
    cluster_m = cluster_n = 1
    if mma_type == "utcmma_cta2":
        preferred = "M" if tb_shape[0] <= tb_shape[1] else "N"
        if preferred == "M" and raw_grid[0] >= 2:
            cluster_m = 2
        elif preferred == "N" and raw_grid[1] >= 2:
            cluster_n = 2
        elif raw_grid[1] >= 2:
            cluster_n = 2
        elif raw_grid[0] >= 2:
            cluster_m = 2
    expected_grid = (
        int(math.ceil(raw_grid[0] / cluster_m)),
        int(math.ceil(raw_grid[1] / cluster_n)),
        1,
    )
    if tuple(launch.work_grid) != expected_grid:
        raise ModelingValidationError(
            "GEMM work_grid %r does not match effective tile grid %r"
            % (tuple(launch.work_grid), expected_grid)
        )
    raster_axis = params.get("raster_axis", "legacy")
    if raster_axis not in _RASTER_AXES:
        raise ModelingValidationError(
            "gemm raster_axis must be one of %s" % sorted(_RASTER_AXES)
        )
    row_panel = _positive_int(params.get("row_panel", 1), "gemm row_panel")
    column_panel = params.get("column_panel", None)
    if column_panel is not None:
        column_panel = _positive_int(column_panel, "gemm column_panel")

    expected_threads = int(
        tb_shape[0] / wp_shape[0] * tb_shape[1] / wp_shape[1] * 32
    )
    if launch.threads != expected_threads:
        raise ModelingValidationError(
            "GEMM launch threads=%d does not match warp tiling (%d)"
            % (launch.threads, expected_threads)
        )
    # work_grid counts logical work units.  UTCMMA2 combines two physical CTAs
    # into one logical supertile, so its actual launch grid expands the logical
    # grid by the cluster shape.  CTA1 keeps the two grids identical.
    expected_physical_grid = (
        expected_grid[0] * cluster_m,
        expected_grid[1] * cluster_n,
        1,
    )
    if tuple(launch.physical_grid) != expected_physical_grid:
        raise ModelingValidationError(
            "GEMM physical_grid %r does not match actual CTA launch grid %r"
            % (launch.physical_grid, expected_physical_grid)
        )
    if launch.scheduler != "static" or launch.residency != "auto":
        raise ModelingValidationError(
            "GEMM parity adapter requires static launch and occupancy-derived residency"
        )
    expected_cluster = (cluster_m, cluster_n, 1)
    expected_cluster_size = cluster_m * cluster_n
    if tuple(launch.cluster) != expected_cluster:
        raise ModelingValidationError(
            "GEMM cluster %r does not match mma_type %s expected axis %r"
            % (launch.cluster, mma_type, expected_cluster)
        )

    phases = loop.phases
    if tuple(phase.name for phase in phases) != ("load_A", "load_B", "mma"):
        raise ModelingValidationError(
            "GEMM parity mainloop must contain load_A, load_B, mma exactly"
        )
    mma_phases = [phase for phase in phases if phase.work.kind == "mma"]
    expected_mma_flops = (
        2 * tb_shape[0] * tb_shape[1] * tb_shape[2] * expected_cluster_size
    )
    if len(mma_phases) != 1 or mma_phases[0].work.flops != expected_mma_flops:
        raise ModelingValidationError(
            "GEMM mma Work must match 2*tile_M*tile_N*tile_K"
        )
    shared_buffers = [buffer for buffer in loop.buffers if buffer.scope == "smem"]
    expected_shared = {
        (tb_shape[0], tb_shape[2]),
        (tb_shape[2], tb_shape[1]),
    }
    if {tuple(buffer.shape) for buffer in shared_buffers} != expected_shared:
        raise ModelingValidationError(
            "GEMM shared buffers must match A/B tile shapes"
        )
    credits = {item.buffer.name: item for item in loop.pipeline_buffers}
    for buffer in shared_buffers:
        credit = credits.get(buffer.name)
        if buffer.slots != stage_num or credit is None or credit.capacity != stage_num:
            raise ModelingValidationError(
                "GEMM shared buffers must bind stages to matching pipeline credits"
            )
    accumulator_states = [state for state in loop.states if state.storage is not None]
    if len(accumulator_states) != 1 or not any(
        carry.state == accumulator_states[0] for carry in loop.carries
    ):
        raise ModelingValidationError(
            "GEMM parity mainloop requires one storage-backed accumulator recurrence"
        )
    phase_by_name = {phase.name: phase for phase in phases}
    buffer_by_name = {buffer.name: buffer for buffer in loop.buffers}
    if set(buffer_by_name) != {"A_shared", "B_shared", "C_accumulator"}:
        raise ModelingValidationError(
            "GEMM parity buffers must be A_shared, B_shared, C_accumulator"
        )
    accumulator_buffer = buffer_by_name["C_accumulator"]
    expected_buffer_specs = {
        "A_shared": (
            "smem",
            (tb_shape[0], tb_shape[2]),
            params.get("dtype", ""),
            stage_num,
            "cta",
        ),
        "B_shared": (
            "smem",
            (tb_shape[2], tb_shape[1]),
            params.get("dtype", ""),
            stage_num,
            "cta",
        ),
        "C_accumulator": (
            "fragment",
            (tb_shape[0], tb_shape[1]),
            params.get("accumulator_dtype", ""),
            1,
            "warpgroup",
        ),
    }
    for name, expected in expected_buffer_specs.items():
        buffer = buffer_by_name[name]
        actual = (
            buffer.scope,
            tuple(buffer.shape),
            buffer.dtype,
            buffer.slots,
            buffer.execution_scope,
        )
        if (
            actual != expected
            or buffer.alias_group is not None
            or buffer.ownership is not None
        ):
            raise ModelingValidationError(
                "GEMM buffer %s does not match the canonical parity template" % name
            )
    expected_flow = {
        "load_A": (tuple(), ("A_shared",)),
        "load_B": (tuple(), ("B_shared",)),
        "mma": (("A_shared", "B_shared"), ("C_accumulator",)),
    }
    for name, (reads, writes) in expected_flow.items():
        phase = phase_by_name[name]
        if tuple(item.name for item in phase.reads) != reads or tuple(
            item.name for item in phase.writes
        ) != writes:
            raise ModelingValidationError(
                "GEMM phase %s dataflow does not match the canonical template" % name
            )
    in1_bytes = int(dict(_memory_levels(params["mem_levels"]))["in1"][-1])
    in2_bytes = int(dict(_memory_levels(params["mem_levels"]))["in2"][-1])
    expected_work = {
        "load_A": ("copy", 0.0, float(tb_shape[0] * tb_shape[2] * in1_bytes)),
        "load_B": ("copy", 0.0, float(tb_shape[2] * tb_shape[1] * in2_bytes)),
        "mma": ("mma", float(expected_mma_flops), 0.0),
    }
    for name, expected in expected_work.items():
        work = phase_by_name[name].work
        if (work.kind, work.flops, work.bytes) != expected:
            raise ModelingValidationError(
                "GEMM phase %s Work does not match the canonical template" % name
            )
    expected_resources = {
        "load_A": ("tma",),
        "load_B": ("tma",),
        "mma": ("tensor",),
    }
    for name, expected in expected_resources.items():
        timing = phase_by_name[name].timing
        if timing is None or tuple(item.resource for item in timing.resources) != expected:
            raise ModelingValidationError(
                "GEMM phase %s timing resources do not match the canonical template"
                % name
            )
    actor_by_name = {actor.name: actor for actor in loop.actors}
    if set(actor_by_name) != {"producer", "tensor"}:
        raise ModelingValidationError("GEMM parity actors must be producer and tensor")
    if any(actor.execution_scope != "unspecified" for actor in loop.actors):
        raise ModelingValidationError(
            "GEMM parity actor execution scopes must keep the canonical unspecified value"
        )
    if any(actor.execution_domain is not None for actor in loop.actors):
        raise ModelingValidationError(
            "GEMM parity actor execution domains must remain unspecified"
        )
    producer = actor_by_name["producer"]
    tensor = actor_by_name["tensor"]
    if (
        producer.order != "issue"
        or tuple((item.phase.name, item.window_offset) for item in producer.sequence)
        != (("load_A", 0), ("load_B", 0))
        or tensor.sequence
    ):
        raise ModelingValidationError(
            "GEMM actor sequences do not match the canonical parity template"
        )
    resource_sequences = {item.resource: item for item in loop.resource_sequences}
    if set(resource_sequences) != {"tma"} or tuple(
        (item.phase.name, item.window_offset)
        for item in resource_sequences["tma"].sequence
    ) != (("load_A", 0), ("load_B", 0)):
        raise ModelingValidationError(
            "GEMM TMA resource sequence does not match the canonical template"
        )
    expected_credits = {
        "A_shared": ("load_A", "start", "mma", "done", stage_num, 0.0),
        "B_shared": ("load_B", "start", "mma", "done", stage_num, 0.0),
    }
    actual_credits = {
        item.buffer.name: (
            item.acquire.phase_name,
            item.acquire.kind,
            item.release.phase_name,
            item.release.kind,
            item.capacity,
            item.minimum_residence,
        )
        for item in loop.pipeline_buffers
    }
    if actual_credits != expected_credits:
        raise ModelingValidationError(
            "GEMM pipeline credits do not match the canonical parity template"
        )
    actual_lifetimes = {
        (
            item.buffer.name,
            item.acquire.phase_name,
            item.acquire.kind,
            item.release.phase_name,
            item.release.kind,
            item.minimum_residence,
        )
        for item in loop.lifetimes
    }
    expected_lifetimes = {
        ("A_shared", "load_A", "start", "mma", "done", 0.0),
        ("B_shared", "load_B", "start", "mma", "done", 0.0),
    }
    if actual_lifetimes != expected_lifetimes:
        raise ModelingValidationError(
            "GEMM buffer lifetimes do not match the canonical parity template"
        )
    if (
        accumulator_states[0].name != "accumulator"
        or accumulator_states[0].storage != accumulator_buffer
        or len(loop.carries) != 1
    ):
        raise ModelingValidationError(
            "GEMM accumulator state does not match the canonical parity template"
        )
    carry = loop.carries[0]
    if (
        carry.state != accumulator_states[0]
        or carry.source != phase_by_name["mma"].done
        or carry.target != phase_by_name["mma"].start
        or carry.iteration_distance != 1
        or carry.lag != 0.0
    ):
        raise ModelingValidationError(
            "GEMM accumulator carry does not match the canonical parity template"
        )
    if loop.dependencies:
        raise ModelingValidationError(
            "GEMM parity mainloop does not support additional explicit dependencies"
        )

    return GemmModelSpec(
        op_shape=op_shape,
        tb_shape=tb_shape,
        wp_shape=wp_shape,
        stage_num=stage_num,
        mem_levels=_memory_levels(params["mem_levels"]),
        row_panel=row_panel,
        batch=_positive_int(params.get("batch", 1), "gemm batch"),
        mma_type=mma_type,
        column_panel=column_panel,
        raster_axis=raster_axis,
        launch_name=launch.name,
        loop_name=loop.name,
    )


def _wave_metadata(
    spec: GemmModelSpec, arch: Any, effective_work_units_per_logical_slot: int
) -> GemmWaveMetadata:
    grid_m = int(math.ceil(spec.op_shape[0] / spec.tb_shape[0]))
    grid_n = int(math.ceil(spec.op_shape[1] / spec.tb_shape[1]))

    cluster_axis = None  # type: Optional[str]
    if spec.mma_type == "utcmma_cta2":
        preferred = "M" if spec.tb_shape[0] <= spec.tb_shape[1] else "N"
        if preferred == "M" and grid_m >= 2:
            cluster_axis = "M"
        elif preferred == "N" and grid_n >= 2:
            cluster_axis = "N"
        elif preferred == "M" and grid_n >= 2:
            cluster_axis = "N"
        elif preferred == "N" and grid_m >= 2:
            cluster_axis = "M"

    cluster_m = 2 if cluster_axis == "M" else 1
    cluster_n = 2 if cluster_axis == "N" else 1
    cluster_size = cluster_m * cluster_n
    effective_mma_type = (
        "utcmma_cta1"
        if spec.mma_type == "utcmma_cta2" and cluster_size == 1
        else spec.mma_type
    )
    work_grid = (
        int(math.ceil(grid_m / cluster_m)),
        int(math.ceil(grid_n / cluster_n)),
    )
    total_work_units = work_grid[0] * work_grid[1]
    effective_sm_count = int(arch.sm_count) // cluster_size
    work_units_per_wave = max(
        effective_sm_count * effective_work_units_per_logical_slot, 1
    )
    full_waves = total_work_units // work_units_per_wave
    tail_work_units = total_work_units % work_units_per_wave
    return GemmWaveMetadata(
        work_grid=work_grid,
        total_work_units=total_work_units,
        effective_sm_count=effective_sm_count,
        effective_work_units_per_logical_slot=(
            effective_work_units_per_logical_slot
        ),
        work_units_per_wave=work_units_per_wave,
        full_waves=full_waves,
        tail_work_units=tail_work_units,
        total_waves=full_waves + (1 if tail_work_units else 0),
        waves_float=total_work_units / work_units_per_wave,
        cluster_axis=cluster_axis,
        cluster_size=cluster_size,
        effective_mma_type=effective_mma_type,
    )


def evaluate_gemm_legacy(kernel: KernelIR, arch: Any) -> GemmLegacyEvaluation:
    """Run the existing full GEMM model with arguments extracted from IR."""

    spec = extract_gemm_spec(kernel)
    from tilesight.modeling._pipeline.matmul_pipeline_wave import (
        calculate_matmul_pipeline_wave,
    )

    legacy_arch, arch_compatibility = gemm_legacy_arch_view(arch)
    legacy_result = calculate_matmul_pipeline_wave(
        **spec.legacy_kwargs(legacy_arch)
    )
    wave = _wave_metadata(spec, arch, int(legacy_result.tiles_per_sm))
    if not math.isclose(
        wave.waves_float,
        float(legacy_result.waves),
        rel_tol=1.0e-12,
        abs_tol=1.0e-15,
    ):
        raise ModelingValidationError(
            "derived GEMM wave metadata disagrees with the legacy result"
        )
    return GemmLegacyEvaluation(
        spec=spec,
        legacy_result=legacy_result,
        wave=wave,
        arch_compatibility=arch_compatibility,
    )


def _legacy_payload(result: Any) -> Dict[str, Any]:
    detail = result.pipeline_detail
    detail_payload = None
    if detail is not None:
        detail_payload = {
            "prologue_time": float(detail.prologue_time),
            "steady_time_per_iter": float(detail.steady_time_per_iter),
            "epilogue_time": float(detail.epilogue_time),
            "mem_time_per_iter": float(detail.mem_time_per_iter),
            "compute_time_per_iter": float(detail.compute_time_per_iter),
        }
    return {
        "per_tile_latency": float(result.per_tile_latency),
        "total_latency": float(result.total_latency),
        "ddr_util": float(result.ddr_util),
        "l2_util": float(result.l2_util),
        "l2_hit_rate": float(result.l2_hit_rate),
        "smem_util": float(result.smem_util),
        "compute_util": float(result.compute_util),
        "smem_footprint": float(result.smem_footprint),
        "reg_footprint": float(result.reg_footprint),
        "tiles_per_sm": int(result.tiles_per_sm),
        "waves": float(result.waves),
        "pipeline_detail": detail_payload,
    }


def _pipeline_components(result: Any) -> Tuple[float, float, float]:
    detail = result.pipeline_detail
    if detail is None:
        raise ModelingValidationError(
            "legacy GEMM result did not provide pipeline_detail"
        )
    prologue = float(detail.prologue_time)
    epilogue = float(detail.epilogue_time)
    steady = float(result.per_tile_latency) - prologue - epilogue
    tolerance = 1.0e-15 + 1.0e-10 * float(result.per_tile_latency)
    if steady < -tolerance:
        raise ModelingValidationError(
            "legacy GEMM pipeline components exceed per_tile_latency"
        )
    return prologue, max(steady, 0.0), epilogue


def model_gemm(
    kernel: KernelIR,
    arch: Any,
    options: FullModelOptions,
) -> FullModelResult:
    """Return a typed, lossless wrapper around the existing GEMM result.

    The compatibility backend intentionally supports only its original
    resource-overlap II.  ``periodic_best`` and ``periodic_worst`` remain
    available through ``kernel.lower_periodic(...)`` as schedule diagnostics,
    but substituting either into this calibrated whole-kernel equation would
    no longer reproduce the existing GEMM model.
    """

    if not isinstance(options, FullModelOptions):
        raise ModelingValidationError("full GEMM adapter expects FullModelOptions")
    if kernel.fusion_plans:
        raise FullModelError(
            "GEMM parity adapter models a standalone legacy kernel and cannot "
            "return fused latency for a nonempty FusionPlan; use analyze_fusion() "
            "for traffic/liveness until a fused executor is selected"
        )
    if options.ii_mode != "resource_ii":
        raise ModelingValidationError(
            "GEMM parity adapter supports ii_mode='resource_ii' only; use "
            "KernelIR.lower_periodic(...) for periodic best/worst diagnostics"
        )
    if options.grid_policy != "model_default":
        raise ModelingValidationError(
            "GEMM parity adapter supports grid_policy='model_default' only; "
            "set row_panel/column_panel/raster_axis in KernelIR.params"
        )
    if options.adapter_dict():
        raise ModelingValidationError(
            "GEMM parity adapter has no adapter_options; put calibrated GEMM "
            "configuration in KernelIR.params"
        )

    evaluation = evaluate_gemm_legacy(kernel, arch)
    spec = evaluation.spec
    result = evaluation.legacy_result
    wave = evaluation.wave
    launch = next(item for item in kernel.launches if item.name == spec.launch_name)
    cooperative_cta2 = (
        wave.cluster_size == 2
        and wave.cluster_axis is not None
        and wave.effective_mma_type == "utcmma_cta2"
    )
    typed_resident_ctas_per_sm = (
        None if cooperative_cta2 else float(result.tiles_per_sm)
    )

    # The old GEMM entry point returns body time only.  The unified API applies
    # the repository-wide fixed 2 us kernel-launch convention exactly once;
    # callers auditing raw legacy body parity can request kernel_launch_s=0.
    launch_s = 2.0e-6 if options.kernel_launch_s is None else options.kernel_launch_s
    dispatch_s = options.host_dispatch_s
    kernel_body_s = float(result.total_latency)
    prologue_s, steady_s, epilogue_s = _pipeline_components(result)
    detail = result.pipeline_detail
    resource_lower_bound_s = max(
        float(detail.mem_time_per_iter),
        float(detail.compute_time_per_iter),
    )

    warps_per_cta = (
        spec.tb_shape[0]
        / spec.wp_shape[0]
        * spec.tb_shape[1]
        / spec.wp_shape[1]
    )
    tail_factor = wave.tail_work_units / wave.work_units_per_wave

    resource_metrics = {
        "ddr_util": float(result.ddr_util),
        "l2_util": float(result.l2_util),
        "l2_hit_rate": float(result.l2_hit_rate),
        "smem_util": float(result.smem_util),
        "compute_util": float(result.compute_util),
        "mem_time_per_iter_s": float(detail.mem_time_per_iter),
        "compute_time_per_iter_s": float(detail.compute_time_per_iter),
        "steady_time_per_iter_s": float(detail.steady_time_per_iter),
        "full_waves": wave.full_waves,
        "tail_work_units": wave.tail_work_units,
        "work_units_per_wave": wave.work_units_per_wave,
        "effective_sm_count": wave.effective_sm_count,
    }
    if cooperative_cta2:
        resource_metrics[
            "legacy_effective_work_units_per_logical_2sm_slot"
        ] = int(result.tiles_per_sm)
    provenance = {
        "backend": "calculate_matmul_pipeline_wave",
        "legacy_module": (
            "tilesight.modeling._pipeline.matmul_pipeline_wave"
        ),
        "legacy_total_scope": "kernel_body_excluding_launch_and_dispatch",
        "pipeline_component_scope": "legacy_active_sm_work_group",
        "schedule_ir_role": "validated_configuration_and_periodic_diagnostic",
        "cache_sampling": "legacy_backend_unmodified",
        "launch_convention": (
            "fixed_2us_default"
            if options.kernel_launch_s is None
            else "explicit_option"
        ),
        "mma_type": spec.mma_type,
        "requested_mma_type": spec.mma_type,
        "effective_mma_type": wave.effective_mma_type,
        "raster_axis": spec.raster_axis,
        "row_panel": spec.row_panel,
        "column_panel": spec.column_panel,
        "batch": spec.batch,
        "cluster_axis": wave.cluster_axis,
        "cluster_semantics": (
            "cooperative_2sm_logical_supertile"
            if cooperative_cta2
            else (
                "requested_cta2_effective_cta1_fallback"
                if spec.mma_type == "utcmma_cta2"
                else "single_cta_work_tile"
            )
        ),
        "arch_compatibility": evaluation.arch_compatibility,
        "search_profile_ignored": options.search_profile,
        "units": "SI_seconds",
    }
    if cooperative_cta2:
        provenance[
            "legacy_effective_work_units_per_logical_2sm_slot"
        ] = int(result.tiles_per_sm)
        provenance["typed_cta_occupancy"] = (
            "unknown_not_inferred_from_legacy_logical_cluster_residency"
        )
        provenance["liveness_residency_binding"] = (
            "none_legacy_logical_cluster_residency_not_consumed"
        )

    return FullModelResult(
        adapter="gemm_pipeline_wave",
        timing=TimingBreakdown(
            kernel_body_s=kernel_body_s,
            launch_s=launch_s,
            host_dispatch_s=dispatch_s,
            total_s=kernel_body_s + launch_s + dispatch_s,
            prologue_s=prologue_s,
            steady_s=steady_s,
            epilogue_s=epilogue_s,
            per_tile_s=float(result.per_tile_latency),
            component_scope="legacy_active_sm_work_group",
        ),
        ii=IISummary(
            selected_s=float(detail.steady_time_per_iter),
            resource_lower_bound_s=resource_lower_bound_s,
            scope="legacy_pipeline_overlap",
            constructive=False,
            search_complete=None,
        ),
        occupancy=OccupancyResult(
            resident_ctas_per_sm=typed_resident_ctas_per_sm,
            smem_bytes_per_cta=float(result.smem_footprint),
            register_value=float(result.reg_footprint),
            register_unit="legacy_4byte_registers_per_thread",
            warps_per_cta=float(warps_per_cta),
            cluster_size=wave.cluster_size,
        ),
        grid=GridResult(
            work_grid=wave.work_grid,
            physical_grid=launch.physical_grid,
            scheduler=launch.scheduler,
            total_work_units=float(wave.total_work_units),
            waves=float(result.waves),
            tail_factor=float(tail_factor),
        ),
        resource_metrics=resource_metrics,
        liveness=analyze_liveness(
            kernel,
            arch=arch,
            resident_ctas_per_sm=typed_resident_ctas_per_sm,
        ),
        fusion=analyze_fusion(kernel),
        legacy=_legacy_payload(result),
        provenance=provenance,
    )


__all__ = [
    "GemmLegacyEvaluation",
    "GemmModelSpec",
    "GemmWaveMetadata",
    "evaluate_gemm_legacy",
    "extract_gemm_spec",
    "gemm_legacy_arch_view",
    "model_gemm",
]
