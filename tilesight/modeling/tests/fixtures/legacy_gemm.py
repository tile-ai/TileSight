"""Synthetic kernel builder for frontend/executor/adapter regression tests.

Fixed costs in this fixture are test inputs, not hardware measurements or
performance-modeling examples. Use program/examples/ and analyze for modeling.
"""

from __future__ import annotations

from tilesight.modeling import Kernel, ResourceTiming, Timing, Work, ns



def build_gemm():
    """Build a 128x128x64 tiled GEMM with a three-stage K mainloop."""

    m = n = k = 4096
    block_m = block_n = 128
    block_k = 64
    stages = 3
    dtype_bytes = 2

    kernel = Kernel(
        "gemm",
        params={
            "model_adapter": "gemm_pipeline_wave",
            "shape": (m, n, k),
            "tile": (block_m, block_n, block_k),
            # Full-kernel compatibility adapter parameters.  The frontend
            # schedule above remains architecture independent; these fields
            # select the calibrated legacy cost policy used to reproduce the
            # existing pipeline+occupancy+wave GEMM result exactly.
            "warp_tile": (64, 64, block_k),
            "mma_type": "wgmma",
            "mem_levels": {
                "in1": (1, 1, 1, 2),
                "in2": (0, 1, 1, 2),
                "out1": (0, 0, 1, 4),
            },
            "row_panel": 1,
            "column_panel": None,
            "raster_axis": "legacy",
            "batch": 1,
            "dtype": "bf16",
            "accumulator_dtype": "fp32",
        },
    )

    grid = (
        (m + block_m - 1) // block_m,
        (n + block_n - 1) // block_n,
        1,
    )
    with kernel.launch(
        work_grid=grid,
        physical_grid=grid,
        threads=128,
        residency="auto",
        scheduler="static",
    ) as launch:
        with launch.periodic(
            "ko",
            iterations=(k + block_k - 1) // block_k,
            stages=stages,
        ) as ko:
            a_shared = ko.buffer(
                "A_shared",
                scope="smem",
                shape=(block_m, block_k),
                dtype="bf16",
                slots=stages,
                execution_scope="cta",
            )
            b_shared = ko.buffer(
                "B_shared",
                scope="smem",
                shape=(block_k, block_n),
                dtype="bf16",
                slots=stages,
                execution_scope="cta",
            )
            accumulator_buffer = ko.buffer(
                "C_accumulator",
                scope="fragment",
                shape=(block_m, block_n),
                dtype="fp32",
                slots=1,
                execution_scope="warpgroup",
            )

            a_tile_bytes = block_m * block_k * dtype_bytes
            b_tile_bytes = block_k * block_n * dtype_bytes

            with ko.actor("producer") as producer:
                load_a = producer.phase(
                    "load_A",
                    work=Work.copy(
                        a_tile_bytes,
                        source="gmem",
                        target="smem",
                        dtype="bf16",
                    ),
                    timing=Timing(
                        latency=72 * ns,
                        resources=(
                            ResourceTiming("tma", service_time=8 * ns),
                        ),
                    ),
                    writes=(a_shared,),
                )
                load_b = producer.phase(
                    "load_B",
                    work=Work.copy(
                        b_tile_bytes,
                        source="gmem",
                        target="smem",
                        dtype="bf16",
                    ),
                    timing=Timing(
                        latency=72 * ns,
                        resources=(
                            ResourceTiming("tma", service_time=8 * ns),
                        ),
                    ),
                    writes=(b_shared,),
                )
                producer.sequence(
                    load_a.at(0), load_b.at(0), resource="tma"
                )

            with ko.actor("tensor") as tensor:
                mma = tensor.phase(
                    "mma",
                    work=Work.mma(
                        2 * block_m * block_n * block_k,
                        shape=(block_m, block_n, block_k),
                        input_dtype="bf16",
                        accumulator_dtype="fp32",
                    ),
                    timing=Timing(
                        latency=32 * ns,
                        resources=(
                            ResourceTiming("tensor", service_time=32 * ns),
                        ),
                    ),
                    reads=(a_shared, b_shared),
                    writes=(accumulator_buffer,),
                )

            # Buffer flow creates load_A/load_B -> mma ready edges.  These
            # declarations add only the cross-iteration slot-reuse credits.
            ko.pipeline_buffer(
                a_shared,
                acquire=load_a.start,
                release=mma.done,
                capacity=stages,
            )
            ko.pipeline_buffer(
                b_shared,
                acquire=load_b.start,
                release=mma.done,
                capacity=stages,
            )

            accumulator = ko.state(
                "accumulator", storage=accumulator_buffer
            )
            ko.carry(
                accumulator,
                source=mma.done,
                target=mma.start,
                distance=1,
            )

    return kernel.build()
