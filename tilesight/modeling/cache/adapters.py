"""Compatibility and convenience adapters for the generic cache IR."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

from ..errors import ModelingValidationError
from .ir import (
    CacheAccessIR,
    CacheLevelConfig,
    CacheProblem,
    PanelTraversal,
    Projection,
    ProgressJitterConfig,
    ReductionConfig,
    RowMajorTraversal,
    SamplingConfig,
    TensorTileRegion,
    TileGrid,
    WritePolicy,
)


def _legacy_cache_config(config: Any, name: str) -> CacheLevelConfig:
    if isinstance(config, CacheLevelConfig):
        if config.name != name:
            raise ModelingValidationError("cache config level mismatch")
        return config
    try:
        capacity = config.capacity
        associativity = config.associativity
        cacheline_bytes = config.cacheline_bytes
    except AttributeError as error:
        raise ModelingValidationError(
            "legacy cache config needs capacity, associativity, and cacheline_bytes"
        ) from error
    return CacheLevelConfig(
        name=name,
        capacity_bytes=capacity,
        unit_bytes=cacheline_bytes,
        associativity=associativity,
        cacheline_bytes=cacheline_bytes,
        distance_unit="legacy_cacheline_volume",
    )


def from_legacy_tensor_accesses(
    grid_shape: Any,
    tensors: Iterable[Any],
    sm_count: int,
    l2_config: Any,
    l1_5_config: Optional[Any] = None,
    l1_5_group_size: int = 8,
    row_panel: Optional[int] = None,
    inner_iterations: int = 1,
    output_footprint_bytes: float = 0.0,
    sampling: SamplingConfig = SamplingConfig(),
) -> CacheProblem:
    """Adapt the old ``TensorAccess``/grid descriptor without importing it.

    The adapter intentionally uses structural fields, so downstream projects
    can migrate one call site at a time without making the old tile-cache
    package a frontend dependency.
    """

    grid = (
        grid_shape
        if isinstance(grid_shape, TileGrid)
        else TileGrid(tuple(int(item) for item in grid_shape))
    )
    accesses = []
    for tensor in tensors:
        try:
            name = tensor.name
            footprint = float(tensor.footprint_bytes)
            reuse_dims = tuple(int(item) for item in tensor.reuse_dims)
            enabled = bool(tensor.ddr_flag)
            frequency_weight = int(tensor.access_count)
        except AttributeError as error:
            raise ModelingValidationError(
                "legacy tensor access is missing a required field"
            ) from error
        if not enabled:
            continue
        kept_axes = tuple(axis for axis in range(len(grid.shape)) if axis not in reuse_dims)
        region = TensorTileRegion(
            value=name,
            index_map=Projection(kept_axes),
            tile_shape=(1,),
            element_bytes=footprint,
        )
        accesses.append(
            CacheAccessIR.load(
                name, region, repetitions=1, frequency_weight=frequency_weight
            )
        )

    if output_footprint_bytes > 0.0:
        output_region = TensorTileRegion(
            value="__legacy_output__",
            index_map=Projection(tuple(range(len(grid.shape)))),
            tile_shape=(1,),
            element_bytes=output_footprint_bytes,
        )
        accesses.append(
            CacheAccessIR.store(
                "__legacy_output__",
                output_region,
                write_policy=WritePolicy(propagation="write_through"),
                include_in_hit_rate=False,
                placement="legacy_wave_boundary",
            )
        )
    if not accesses:
        raise ModelingValidationError("legacy adapter produced no cache accesses")

    if len(grid.shape) == 2:
        if row_panel is None:
            row_panel = max(sm_count // grid.shape[0], 1)
        traversal = PanelTraversal(row_panel=row_panel, sm_count=sm_count)
    else:
        traversal = RowMajorTraversal(wave_size=sm_count, sm_count=sm_count)
    l1_5 = None if l1_5_config is None else _legacy_cache_config(l1_5_config, "l1_5")
    return CacheProblem(
        grid=grid,
        accesses=tuple(accesses),
        traversal=traversal,
        l2=_legacy_cache_config(l2_config, "l2"),
        l1_5=l1_5,
        l1_5_group_size=l1_5_group_size if l1_5 is not None else 0,
        inner_iterations=inner_iterations,
        sampling=sampling,
        backend="legacy_volume",
    )


def gemm_cache_problem(
    M: int,
    N: int,
    K: int,
    tb_m: int,
    tb_n: int,
    tb_k: int,
    l2_capacity_bytes: float,
    sm_count: int,
    mem_levels: Mapping[str, Sequence[float]],
    row_panel: int,
    l1_5_group_size: int = 0,
    l1_5_capacity_per_group: float = 180 * 1024,
    l1_5_associativity: int = 8,
    l1_5_cacheline_bytes: int = 128,
    column_panel: Optional[int] = None,
    raster_axis: str = "legacy",
    reuse_unit_bytes: Optional[float] = None,
    reduction_mode: str = "anonymous_inner",
    representative_k: int = 0,
    progress_jitter: ProgressJitterConfig = ProgressJitterConfig(),
    sampling: SamplingConfig = SamplingConfig(),
) -> CacheProblem:
    """Construct regular GEMM accesses with the generic object frontend."""

    for name in ("in1", "in2", "out1"):
        if name not in mem_levels or len(mem_levels[name]) < 4:
            raise ModelingValidationError("GEMM mem_levels[%s] is missing" % name)
    grid_m = int(math.ceil(float(M) / tb_m))
    grid_n = int(math.ceil(float(N) / tb_n))
    grid_k = int(math.ceil(float(K) / tb_k))
    a = TensorTileRegion(
        value="A",
        index_map=Projection((0,)),
        tile_shape=(tb_m, tb_k),
        element_bytes=mem_levels["in1"][-1],
    )
    b = TensorTileRegion(
        value="B",
        index_map=Projection((1,)),
        tile_shape=(tb_k, tb_n),
        element_bytes=mem_levels["in2"][-1],
    )
    output = TensorTileRegion(
        value="C",
        index_map=Projection((0, 1)),
        tile_shape=(tb_m, tb_n),
        element_bytes=mem_levels["out1"][-1],
    )
    accesses = []
    if mem_levels["in1"][0]:
        accesses.append(CacheAccessIR.load("A", a))
    if mem_levels["in2"][0]:
        accesses.append(CacheAccessIR.load("B", b))
    if mem_levels["out1"][0]:
        accesses.append(
            CacheAccessIR.store(
                "C",
                output,
                write_policy=WritePolicy(propagation="write_through"),
                include_in_hit_rate=False,
                placement="legacy_wave_boundary",
            )
        )
    if reuse_unit_bytes is None:
        input_units = [
            item.region.payload_bytes for item in accesses if item.mode == "read"
        ]
        if not input_units:
            raise ModelingValidationError("GEMM cache model needs a readable input")
        # Equal-shape A/B tiles each occupy one schedule-visible allocation
        # unit. Rectangular inputs use the smaller tile as a stable reference.
        reuse_unit_bytes = min(input_units)
    return CacheProblem(
        grid=TileGrid((grid_m, grid_n), axes=("m", "n")),
        accesses=tuple(accesses),
        traversal=PanelTraversal(
            row_panel=row_panel,
            column_panel=column_panel,
            sm_count=sm_count,
            raster_axis=raster_axis,
        ),
        l2=CacheLevelConfig(
            "l2",
            l2_capacity_bytes,
            reuse_unit_bytes,
            8,
            128,
            distance_unit="tile_allocation",
        ),
        l1_5=(
            CacheLevelConfig(
                "l1_5",
                l1_5_capacity_per_group,
                reuse_unit_bytes,
                l1_5_associativity,
                l1_5_cacheline_bytes,
                distance_unit="tile_allocation",
            )
            if l1_5_group_size > 0
            else None
        ),
        l1_5_group_size=l1_5_group_size,
        inner_iterations=grid_k,
        reduction=ReductionConfig(
            mode=reduction_mode,
            representative_k=representative_k,
            progress_jitter=progress_jitter,
        ),
        sampling=sampling,
    )


def eliminate_fused_accesses(problem: CacheProblem, fusion_plan: Any) -> CacheProblem:
    """Remove accesses referenced by a frontend ``FusionPlan`` before cache analysis."""

    try:
        handoffs = tuple(fusion_plan.handoffs)
    except AttributeError as error:
        raise ModelingValidationError("fusion_plan needs handoffs") from error
    names = []
    for handoff in handoffs:
        names.extend(item.name for item in handoff.baseline_writes)
        names.extend(item.name for item in handoff.baseline_reads)
    return problem.eliminate_accesses(names)
