"""Synthetic kernel builder for frontend/executor/adapter regression tests.

Fixed costs in this fixture are test inputs, not hardware measurements or
performance-modeling examples. Use program/examples/ and analyze for modeling.
"""

from __future__ import annotations

from tilesight.modeling import Kernel, ResourceTiming, Timing, Work, ns



def build_fa3():
    """Build a compact FA3 forward mainloop for BF16, head dimension 128."""

    batch = 1
    heads = 32
    sequence = 2048
    head_dim = 128
    block_m = block_n = 128
    stages = 2
    dtype_bytes = 2

    kernel = Kernel(
        "fa3_forward",
        params={
            "model_adapter": "fa3",
            "model_family": "fa3",
            "shape": (batch, heads, sequence, head_dim),
            "kv_heads": heads,
            # This prototype uses the audited 128x128 causal d=128 tile.  Keep
            # the whole-kernel semantic explicit so the full-model adapter
            # does not infer it from the illustrative launch grid.
            "causal": True,
            "tile": (block_m, block_n),
            "dtype": "bf16",
        },
    )

    # The grid expresses spatially independent CTA work.  It remains separate
    # from the temporal, two-stage KV pipeline below.
    grid = ((sequence + block_m - 1) // block_m, heads, batch)
    with kernel.launch(
        work_grid=grid,
        physical_grid=("arch.sm_count", 1, 1),
        threads=256,
        residency="auto",
        scheduler="persistent",
    ) as launch:
        with launch.periodic(
            "kv",
            iterations=(sequence + block_n - 1) // block_n,
            stages=stages,
        ) as kv:
            k_shared = kv.buffer(
                "K_shared",
                scope="smem",
                shape=(block_n, head_dim),
                dtype="bf16",
                slots=stages,
                execution_scope="cta",
            )
            v_shared = kv.buffer(
                "V_shared",
                scope="smem",
                shape=(block_n, head_dim),
                dtype="bf16",
                slots=stages,
                execution_scope="cta",
            )
            scores = kv.buffer(
                "scores",
                scope="fragment",
                shape=(block_m, block_n),
                dtype="fp32",
                slots=1,
                execution_scope="warpgroup",
            )
            probabilities = kv.buffer(
                "probabilities",
                scope="fragment",
                shape=(block_m, block_n),
                dtype="bf16",
                slots=1,
                execution_scope="warpgroup",
            )
            output_accumulator_buffer = kv.buffer(
                "O_accumulator",
                scope="fragment",
                shape=(block_m, head_dim),
                dtype="fp32",
                slots=1,
                execution_scope="warpgroup",
            )
            softmax_state_buffer = kv.buffer(
                "softmax_state",
                scope="fragment",
                shape=(block_m, 2),
                dtype="fp32",
                slots=1,
                execution_scope="warpgroup",
            )

            kv_tile_bytes = block_n * head_dim * dtype_bytes
            attention_tile_flops = 2 * block_m * block_n * head_dim

            with kv.actor("producer") as producer:
                load_k = producer.phase(
                    "load_K",
                    work=Work.copy(
                        kv_tile_bytes,
                        source="gmem",
                        target="smem",
                        tensor="K",
                    ),
                    timing=Timing(
                        latency=80 * ns,
                        resources=(
                            ResourceTiming("tma", service_time=8 * ns),
                        ),
                    ),
                    writes=(k_shared,),
                )
                load_v = producer.phase(
                    "load_V",
                    work=Work.copy(
                        kv_tile_bytes,
                        source="gmem",
                        target="smem",
                        tensor="V",
                    ),
                    timing=Timing(
                        latency=80 * ns,
                        resources=(
                            ResourceTiming("tma", service_time=8 * ns),
                        ),
                    ),
                    writes=(v_shared,),
                )
                producer.sequence(
                    load_k.at(+1), load_v.at(0), resource="tma"
                )

            with kv.actor("tensor") as tensor:
                qk = tensor.phase(
                    "gemm_QK",
                    work=Work.mma(
                        attention_tile_flops,
                        operands="Q @ K.T",
                        shape=(block_m, block_n, head_dim),
                    ),
                    timing=Timing(
                        latency=42 * ns,
                        resources=(
                            ResourceTiming("tensor", service_time=42 * ns),
                        ),
                    ),
                    reads=(k_shared,),
                    writes=(scores,),
                )
                pv = tensor.phase(
                    "gemm_PV",
                    work=Work.mma(
                        attention_tile_flops,
                        operands="P @ V",
                        shape=(block_m, head_dim, block_n),
                    ),
                    timing=Timing(
                        latency=42 * ns,
                        resources=(
                            ResourceTiming("tensor", service_time=42 * ns),
                        ),
                    ),
                    reads=(probabilities, v_shared, output_accumulator_buffer),
                )
                tensor.sequence(qk.at(+1), pv.at(0), resource="tensor")

            with kv.actor("softmax") as softmax_actor:
                softmax = softmax_actor.phase(
                    "online_softmax",
                    work=Work.reduce(
                        7 * block_m * block_n,
                        operation="online_max_sum_exp2",
                    ),
                    timing=Timing(
                        latency=55 * ns,
                        resources=(
                            ResourceTiming("cuda", service_time=30 * ns),
                            ResourceTiming("sfu", service_time=25 * ns),
                        ),
                    ),
                    reads=(scores,),
                    writes=(probabilities,),
                )
                rescale_o = softmax_actor.phase(
                    "rescale_O",
                    work=Work.pointwise(
                        block_m * head_dim,
                        operation="rescale_accumulator",
                    ),
                    timing=Timing(
                        latency=12 * ns,
                        resources=(
                            ResourceTiming("cuda", service_time=12 * ns),
                        ),
                    ),
                    writes=(output_accumulator_buffer,),
                )
                softmax_actor.sequence(
                    softmax.at(+1), rescale_o.at(+1), resource="cuda"
                )

            # Same-iteration K/V/score/probability dependencies are inferred
            # from reads/writes.  These explicit edges describe source order
            # for the accumulator correction before PV.
            kv.after(softmax.done, rescale_o.start)
            kv.after(rescale_o.done, pv.start)

            kv.pipeline_buffer(
                k_shared,
                acquire=load_k.start,
                release=qk.done,
                capacity=stages,
            )
            kv.pipeline_buffer(
                v_shared,
                acquire=load_v.start,
                release=pv.done,
                capacity=stages,
            )

            online_softmax_state = kv.state(
                "online_softmax_state", storage=softmax_state_buffer
            )
            kv.carry(
                online_softmax_state,
                source=softmax.done,
                target=softmax.start,
                distance=1,
            )
            output_accumulator = kv.state(
                "O_accumulator", storage=output_accumulator_buffer
            )
            kv.carry(
                output_accumulator,
                source=pv.done,
                target=rescale_o.start,
                distance=1,
            )

    return kernel.build()
