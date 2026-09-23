"""Synthetic kernel builder for frontend/executor/adapter regression tests.

Fixed costs in this fixture are test inputs, not hardware measurements or
performance-modeling examples. Use program/examples/ and analyze for modeling.
"""

from __future__ import annotations

from dataclasses import replace

from .legacy_gemm import build_gemm


def build_b200_cta2_gemm():
    """Convert the canonical GEMM example to a 2-SM UTCMMA supertile."""

    ir = build_gemm()
    params = dict(ir.params)
    params["mma_type"] = "utcmma_cta2"
    launch = ir.launches[0]
    loop = launch.periodic_loops[0]
    tensor_actor = next(actor for actor in loop.actors if actor.name == "tensor")
    mma = tensor_actor.phases[0]
    paired_tensor = replace(
        tensor_actor,
        phases=(replace(mma, work=replace(mma.work, flops=mma.work.flops * 2)),),
    )
    paired_loop = replace(
        loop,
        actors=tuple(
            paired_tensor if actor.name == "tensor" else actor
            for actor in loop.actors
        ),
    )
    raw_grid_m, raw_grid_n, _ = launch.work_grid
    paired_grid = ((raw_grid_m + 1) // 2, raw_grid_n, 1)
    physical_cta_grid = (paired_grid[0] * 2, paired_grid[1], 1)
    paired_launch = replace(
        launch,
        work_grid=paired_grid,
        physical_grid=physical_cta_grid,
        cluster=(2, 1, 1),
        periodic_loops=(paired_loop,),
    )
    return replace(ir, params=params, launches=(paired_launch,))
