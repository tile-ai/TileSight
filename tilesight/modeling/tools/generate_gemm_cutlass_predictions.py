#!/usr/bin/env python3
"""Generate the maintained GEMM CUTLASS ground-truth prediction columns.

The generator consumes a raw unified profiler CSV, applies exactly the same
fail-closed coverage selector and CTA2 median deduplication as
``compare_gemm_cutlass.py``, and writes body and body-plus-launch predictions.

``general_shadow_v1`` is intentionally named: it is the object-IR/native
shadow used for the documented audit, not the newer ``general_gemm`` adapter.
It binds the legacy resource-cost decomposition to load/store/compute phases,
then runs ``legacy_stage`` and ``legacy_calibrated`` policies.  Cooperative
CTA2 has no native cluster-aware oracle, so its general columns remain empty
with an explicit status rather than inheriting a legacy number.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

import tilesight.modeling._pipeline.matmul_pipeline_wave as legacy_module
from tilesight.arch.a100 import A100
from tilesight.arch.b200 import B200
from tilesight.modeling._cache_compat.cache_model import multi_level_hit_rate
from tilesight.modeling import (
    Kernel,
    LoopRegion,
    NativeModelOptions,
    PeriodicAxisIR,
    PhaseRegion,
    ResourceTiming,
    SequenceRegion,
    Timing,
    model_native,
)
from tilesight.modeling.adapters.gemm import gemm_legacy_arch_view
from tilesight.modeling.cache import (
    SamplingConfig,
    gemm_cache_problem,
    model_cache,
)
from tilesight.modeling.native_policy import LegacyStageBoundary
from tilesight.modeling.tools.compare_gemm_cutlass import (
    REQUIRED,
    select_covered_rows,
)


MEM_LEVELS = {
    "in1": [1, 1, 1, 2],
    "in2": [1, 1, 1, 2],
    "out1": [1, 1, 1, 2],
}
GENERAL_MODEL_VERSION = (
    "object_ir_shadow_v1_legacy_costs_legacy_stage_legacy_calibrated_wave"
)
GENERAL_UNSUPPORTED_CTA2 = (
    "unsupported_requires_native_cooperative_cluster_cost_oracle"
)
PREDICTION_FIELDS = (
    "legacy_body_us",
    "legacy_us",
    "distinct_legacy_outer_body_us",
    "distinct_legacy_outer_us",
    "general_shadow_v1_body_us",
    "general_shadow_v1_us",
    "general_prediction_status",
    "prediction_mapping",
    "prediction_seed",
    "prediction_sample_budget",
    "prediction_reuse_unit_bytes",
    "prediction_launch_us",
    "modeled_occupancy",
    "modeled_waves",
)


@dataclass(frozen=True)
class GenerationConfig:
    arch: str
    mapping: str
    seed: int = 17
    sample_budget: int = 256
    reuse_unit_bytes: int = 8192
    launch_us: float = 2.0

    def __post_init__(self) -> None:
        supported = {("a100", "cluster1"), ("b200", "cta1"), ("b200", "cta2")}
        if (self.arch, self.mapping) not in supported:
            raise ValueError(
                "prediction generation supports a100/cluster1, b200/cta1, "
                "or b200/cta2"
            )
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("seed must be an integer")
        for name in ("sample_budget", "reuse_unit_bytes"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError("%s must be a positive integer" % name)
        if not math.isfinite(self.launch_us) or self.launch_us < 0.0:
            raise ValueError("launch_us must be finite and non-negative")


@dataclass(frozen=True)
class GenerationSummary:
    input_rows: int
    selected_before_dedupe: int
    output_rows: int
    filter_reasons: Mapping[str, int]


@dataclass(frozen=True)
class _Raster:
    row_panel: int
    column_panel: int
    axis: str


@dataclass(frozen=True)
class _ShadowCase:
    shape: Tuple[int, int, int]
    tile: Tuple[int, int, int]
    warp_tile: Tuple[int, int, int]
    stages: int


class _A100LegacyArchView:
    """Read-only field asserted by the calibrated legacy GEMM backend."""

    def __init__(self, source: Any) -> None:
        self._source = source

    def __getattr__(self, name: str) -> Any:
        return getattr(self._source, name)

    def get_tensor_core_minimum_ptx(self, bytes: int = 2) -> Tuple[int, int, int]:
        if bytes != 2:
            raise ValueError("the A100 GT shadow is calibrated only for fp16")
        return (16, 8, 16)


class _ShadowOracle:
    def __init__(self, timings: Mapping[str, Timing]) -> None:
        self.timings = dict(timings)

    def resolve(self, phase: Any) -> Timing:
        return self.timings[phase.name]


def _integer(row: Mapping[str, str], field: str) -> int:
    value = int(row[field])
    if value <= 0:
        raise ValueError("%s must be positive" % field)
    return value


def _shape(row: Mapping[str, str]) -> Tuple[int, int, int]:
    return tuple(_integer(row, field) for field in ("M", "N", "K"))


def _tile(row: Mapping[str, str]) -> Tuple[int, int, int]:
    return tuple(_integer(row, field) for field in ("tile_m", "tile_n", "tile_k"))


def _raster(row: Mapping[str, str], arch: Any) -> _Raster:
    m, n, _k = _shape(row)
    tile_m, tile_n, _tile_k = _tile(row)
    grid_m = int(math.ceil(float(m) / tile_m))
    grid_n = int(math.ceil(float(n) / tile_n))
    swizzle = _integer(row, "swizzle_size")
    if row["raster_order"] == "along_m":
        return _Raster(swizzle, min(grid_m, int(arch.sm_count)), "along_m")
    if row["raster_order"] == "along_n":
        return _Raster(min(grid_n, int(arch.sm_count)), swizzle, "along_n")
    raise ValueError("unsupported raster_order %r" % row["raster_order"])


def _cache_tile_and_raster(
    row: Mapping[str, str], arch: Any, config: GenerationConfig
) -> Tuple[Tuple[int, int, int], _Raster]:
    tile_m, tile_n, tile_k = _tile(row)
    raster = _raster(row, arch)
    if config.mapping != "cta2":
        return (tile_m, tile_n, tile_k), raster
    # The selected UTCMMA2 rows are measured 2x1 clusters.  The old cache path
    # models the M-cluster supertile and halves its panel count exactly once.
    return (
        (2 * tile_m, tile_n, tile_k),
        _Raster(
            max(raster.row_panel // 2, 1),
            max(raster.column_panel // 2, 1),
            raster.axis,
        ),
    )


def _cache_key(row: Mapping[str, str]) -> Tuple[str, ...]:
    return tuple(
        row[field]
        for field in (
            "M",
            "N",
            "K",
            "tile_m",
            "tile_n",
            "tile_k",
            "raster_order",
            "swizzle_size",
        )
    )


def _legacy_cache(
    row: Mapping[str, str], arch: Any, config: GenerationConfig
) -> Mapping[str, float]:
    tile, raster = _cache_tile_and_raster(row, arch, config)
    rng_state = np.random.get_state()
    try:
        np.random.seed(config.seed)
        return multi_level_hit_rate(
            *_shape(row),
            *tile,
            arch,
            MEM_LEVELS,
            raster.row_panel,
            column_panel=raster.column_panel,
            raster_axis=raster.axis,
        )
    finally:
        np.random.set_state(rng_state)


def _distinct_cache(
    row: Mapping[str, str], arch: Any, config: GenerationConfig
) -> Mapping[str, float]:
    tile, raster = _cache_tile_and_raster(row, arch, config)
    kwargs = {}  # type: Dict[str, Any]
    if config.arch == "b200":
        kwargs.update(
            l1_5_group_size=arch.l1_5_group_size,
            l1_5_capacity_per_group=arch.l1_5_capacity_per_group,
            l1_5_associativity=arch.l1_5_associativity,
            l1_5_cacheline_bytes=arch.l1_5_cacheline_bytes,
        )
    problem = gemm_cache_problem(
        *_shape(row),
        *tile,
        arch.l2_capacity,
        arch.sm_count,
        MEM_LEVELS,
        raster.row_panel,
        column_panel=raster.column_panel,
        raster_axis=raster.axis,
        reuse_unit_bytes=config.reuse_unit_bytes,
        sampling=SamplingConfig(
            seed=config.seed,
            sample_budget=config.sample_budget,
        ),
        **kwargs,
    )
    aggregate = model_cache(problem).aggregate
    return {
        "l1_5_hit_rate": aggregate.l1_5_hit_rate,
        "l2_hit_rate": aggregate.l2_hit_rate,
        "ddr_miss_rate": aggregate.ddr_miss_rate,
    }


def _mma_and_warp_tile(
    row: Mapping[str, str], config: GenerationConfig
) -> Tuple[str, Tuple[int, int, int]]:
    tile = _tile(row)
    if config.arch == "a100":
        # The profiler schema omits the warp tile.  This is the exact explicit
        # shadow assumption used by the documented A100 audit.
        return "wmma", (64, 64, tile[2])
    if config.mapping == "cta2":
        return "utcmma_cta2", tile
    mapping = {"wgmma": "wgmma", "utcmma1": "utcmma_cta1"}
    try:
        return mapping[row["tc_path"]], tile
    except KeyError as error:
        raise ValueError("unsupported B200 CTA1 tc_path") from error


def _legacy_prediction(
    row: Mapping[str, str],
    arch_view: Any,
    cache: Mapping[str, float],
    config: GenerationConfig,
) -> Any:
    raster = _raster(row, arch_view)
    mma_type, warp_tile = _mma_and_warp_tile(row, config)
    original = legacy_module.multi_level_hit_rate
    try:
        legacy_module.multi_level_hit_rate = lambda *_args, **_kwargs: dict(cache)
        return legacy_module.calculate_matmul_pipeline_wave(
            _shape(row),
            _tile(row),
            warp_tile,
            _integer(row, "stages"),
            arch_view,
            MEM_LEVELS,
            row_panel=raster.row_panel,
            mma_type=mma_type,
            column_panel=raster.column_panel,
            raster_axis=raster.axis,
        )
    finally:
        legacy_module.multi_level_hit_rate = original


def _timing(latency: float, resource: str) -> Timing:
    if latency <= 0.0:
        return Timing(0.0)
    return Timing(latency, (ResourceTiming(resource, latency),))


def _general_shadow(
    row: Mapping[str, str], arch: Any, calibrated: Any, config: GenerationConfig
) -> Any:
    if config.mapping == "cta2":
        return None
    shape = _shape(row)
    tile = _tile(row)
    _mma_type, warp_tile = _mma_and_warp_tile(row, config)
    case = _ShadowCase(shape, tile, warp_tile, _integer(row, "stages"))
    m, n, k = case.shape
    tile_m, tile_n, tile_k = case.tile
    warp_m, warp_n, _warp_k = case.warp_tile
    grid = (
        int(math.ceil(float(m) / tile_m)),
        int(math.ceil(float(n) / tile_n)),
        1,
    )
    threads = int(tile_m / warp_m * tile_n / warp_n * 32)

    kernel = Kernel("gemm_cutlass_general_shadow_v1")
    with kernel.launch(
        work_grid=grid,
        physical_grid=grid,
        threads=threads,
        resident_ctas=calibrated.tiles_per_sm,
        scheduler="static",
    ) as launch:
        with launch.periodic(
            "ko",
            iterations=int(math.ceil(float(k) / tile_k)),
            stages=case.stages,
        ) as loop:
            a = loop.buffer(
                "A_shared",
                scope="smem",
                shape=(tile_m, tile_k),
                dtype="fp16",
                slots=case.stages,
            )
            b = loop.buffer(
                "B_shared",
                scope="smem",
                shape=(tile_k, tile_n),
                dtype="fp16",
                slots=case.stages,
            )
            accumulator = loop.buffer(
                "C_accumulator",
                scope="fragment",
                shape=(tile_m, tile_n),
                dtype="fp16",
                slots=1,
                execution_scope="warpgroup",
            )
            with loop.actor("producer") as producer:
                load_a = producer.load(
                    "load_A",
                    bytes=tile_m * tile_k * MEM_LEVELS["in1"][-1],
                    source="ddr",
                    destination="smem",
                    engine="tma",
                    writes=(a,),
                )
                load_b = producer.load(
                    "load_B",
                    bytes=tile_k * tile_n * MEM_LEVELS["in2"][-1],
                    source="ddr",
                    destination="smem",
                    engine="tma",
                    writes=(b,),
                )
                producer.sequence(load_a.at(0), load_b.at(0))
            with loop.actor("tensor") as tensor:
                mma = tensor.compute(
                    "mma",
                    flops=2 * tile_m * tile_n * tile_k,
                    engine="tensor",
                    op="mma",
                    reads=(a, b),
                    writes=(accumulator,),
                )
            loop.pipeline_buffer(
                a, acquire=load_a.start, release=mma.done, capacity=case.stages
            )
            loop.pipeline_buffer(
                b, acquire=load_b.start, release=mma.done, capacity=case.stages
            )
            state = loop.state("accumulator", storage=accumulator)
            loop.carry(state, source=mma.done, target=mma.start, distance=1)
        with launch.periodic("epilogue", iterations=1) as epilogue:
            with epilogue.actor("store") as store_actor:
                store_actor.store(
                    "store_C",
                    bytes=tile_m * tile_n * MEM_LEVELS["out1"][-1],
                    source="register",
                    destination="ddr",
                    engine="ldst",
                )

    ir = kernel.build()
    detail = calibrated.pipeline_detail
    depth = case.stages * calibrated.tiles_per_sm - 1
    store_time = max(
        (detail.epilogue_time - depth * detail.compute_time_per_iter)
        / calibrated.tiles_per_sm,
        0.0,
    )
    a_bytes = tile_m * tile_k * MEM_LEVELS["in1"][0] * MEM_LEVELS["in1"][-1]
    b_bytes = tile_n * tile_k * MEM_LEVELS["in2"][0] * MEM_LEVELS["in2"][-1]
    total_bytes = a_bytes + b_bytes
    timings = {
        "load_A": _timing(
            detail.mem_time_per_iter * a_bytes / total_bytes, "memory"
        ),
        "load_B": _timing(
            detail.mem_time_per_iter * b_bytes / total_bytes, "memory"
        ),
        "mma": _timing(detail.compute_time_per_iter, "tensor"),
        "store_C": _timing(store_time, "store"),
    }
    oracle = _ShadowOracle(timings)
    loop_ir = ir.periodic_loop("ko")
    phases = {phase.name: phase for phase in loop_ir.phases}
    store_phase = ir.periodic_loop("epilogue").phases[0]
    region = LoopRegion(
        "ko_region",
        trip_count=loop_ir.iterations,
        body=SequenceRegion(
            "ko_body",
            (
                PhaseRegion(phases["load_A"]),
                PhaseRegion(phases["load_B"]),
                PhaseRegion(phases["mma"]),
            ),
        ),
        epilogue=PhaseRegion(store_phase),
        periodic_axis=PeriodicAxisIR(
            loop_name="ko",
            ii_mode="periodic_best",
            legacy_stage_boundary=LegacyStageBoundary(
                memory_resources=("memory",),
                compute_resources=("tensor",),
            ),
        ),
    )
    return model_native(
        ir,
        region,
        arch,
        NativeModelOptions(
            ii_mode="periodic_best",
            resident_ctas_per_sm=calibrated.tiles_per_sm,
            multi_cta_policy=(
                "reject" if calibrated.tiles_per_sm == 1 else "resource_bound"
            ),
            kernel_launch_s=0.0,
            liveness_policy="require_witness",
            boundary_policy="legacy_stage",
            wave_policy="legacy_calibrated",
        ),
        oracle=oracle,
    )


def _architecture(config: GenerationConfig) -> Tuple[Any, Any]:
    if config.arch == "a100":
        arch = A100().set_to_microbench()
        return arch, _A100LegacyArchView(arch)
    arch = B200().set_to_microbench()
    arch_view, _provenance = gemm_legacy_arch_view(arch)
    return arch, arch_view


def generate_predictions(
    rows: Sequence[Dict[str, str]],
    config: GenerationConfig,
    *,
    limit: Optional[int] = None,
) -> Tuple[List[Dict[str, str]], GenerationSummary]:
    """Select, deduplicate, and evaluate raw profiler rows deterministically."""

    selected, reasons, before_dedupe = select_covered_rows(
        rows, config.arch, config.mapping
    )
    if limit is not None:
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            raise ValueError("limit must be a positive integer")
        selected = selected[:limit]
    arch, arch_view = _architecture(config)
    legacy_cache_memo = {}  # type: Dict[Tuple[str, ...], Mapping[str, float]]
    distinct_cache_memo = {}  # type: Dict[Tuple[str, ...], Mapping[str, float]]
    output = []  # type: List[Dict[str, str]]
    for row in selected:
        key = _cache_key(row)
        if key not in legacy_cache_memo:
            legacy_cache_memo[key] = _legacy_cache(row, arch, config)
            distinct_cache_memo[key] = _distinct_cache(row, arch, config)
        legacy = _legacy_prediction(
            row, arch_view, legacy_cache_memo[key], config
        )
        distinct = _legacy_prediction(
            row, arch_view, distinct_cache_memo[key], config
        )
        general = _general_shadow(row, arch, distinct, config)
        item = dict(row)
        legacy_body_us = legacy.total_latency * 1.0e6
        distinct_body_us = distinct.total_latency * 1.0e6
        item.update(
            {
                "legacy_body_us": repr(legacy_body_us),
                "legacy_us": repr(legacy_body_us + config.launch_us),
                "distinct_legacy_outer_body_us": repr(distinct_body_us),
                "distinct_legacy_outer_us": repr(
                    distinct_body_us + config.launch_us
                ),
                "general_shadow_v1_body_us": (
                    "" if general is None else repr(general.total_s * 1.0e6)
                ),
                "general_shadow_v1_us": (
                    ""
                    if general is None
                    else repr(general.total_s * 1.0e6 + config.launch_us)
                ),
                "general_prediction_status": (
                    GENERAL_UNSUPPORTED_CTA2
                    if general is None
                    else GENERAL_MODEL_VERSION
                ),
                "prediction_mapping": "%s/%s" % (config.arch, config.mapping),
                "prediction_seed": str(config.seed),
                "prediction_sample_budget": str(config.sample_budget),
                "prediction_reuse_unit_bytes": str(config.reuse_unit_bytes),
                "prediction_launch_us": repr(config.launch_us),
                "modeled_occupancy": str(legacy.tiles_per_sm),
                "modeled_waves": repr(legacy.waves),
            }
        )
        output.append(item)
    summary = GenerationSummary(
        input_rows=len(rows),
        selected_before_dedupe=before_dedupe,
        output_rows=len(output),
        filter_reasons=dict(sorted(reasons.items())),
    )
    return output, summary


def _read_csv(path: Path) -> Tuple[List[Dict[str, str]], Tuple[str, ...]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        missing = sorted(set(REQUIRED) - set(fields))
        if missing:
            raise ValueError("CSV is missing required fields %s" % missing)
        return [dict(row) for row in reader], fields


def write_predictions(
    source: Path,
    output: Path,
    config: GenerationConfig,
    *,
    limit: Optional[int] = None,
) -> GenerationSummary:
    rows, input_fields = _read_csv(source)
    predicted, summary = generate_predictions(rows, config, limit=limit)
    if not predicted:
        raise ValueError("coverage selector produced no rows")
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(input_fields)
    for field in PREDICTION_FIELDS:
        if field not in fields:
            fields.append(field)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(predicted)
    return summary


def main(argv: Sequence[str] = ()) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--arch", required=True, choices=("a100", "b200"))
    parser.add_argument(
        "--mapping", required=True, choices=("cluster1", "cta1", "cta2")
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--sample-budget", type=int, default=256)
    parser.add_argument("--reuse-unit-bytes", type=int, default=8192)
    parser.add_argument("--launch-us", type=float, default=2.0)
    parser.add_argument(
        "--limit",
        type=int,
        help="debug/smoke limit applied after selection and CTA2 dedupe",
    )
    args = parser.parse_args(tuple(argv) if argv else None)
    try:
        config = GenerationConfig(
            arch=args.arch,
            mapping=args.mapping,
            seed=args.seed,
            sample_budget=args.sample_budget,
            reuse_unit_bytes=args.reuse_unit_bytes,
            launch_us=args.launch_us,
        )
        summary = write_predictions(args.csv, args.out, config, limit=args.limit)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(
        "input_rows=%d selected_before_dedupe=%d output_rows=%d"
        % (
            summary.input_rows,
            summary.selected_before_dedupe,
            summary.output_rows,
        )
    )
    print("filter_reasons=%s" % dict(summary.filter_reasons))
    print("prediction_csv=%s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
