"""Synthetic kernel builder for frontend/executor/adapter regression tests.

Fixed costs in this fixture are test inputs, not hardware measurements or
performance-modeling examples. Use program/examples/ and analyze for modeling.
"""

from __future__ import annotations

from tilesight.modeling import Kernel, ResourceTiming, Timing, Work, ns



def build_fa4():
    """Build a compact q-stage=2 FA4 forward mainloop."""

    batch = 1
    heads = 16
    sequence = 2048
    head_dim = 128
    block_m = block_n = 128
    q_stages = 2
    kv_slots = 3
    dtype_bytes = 2

    kernel = Kernel(
        "fa4_forward",
        params={
            "model_adapter": "fa4",
            "model_family": "fa4",
            "shape": (batch, heads, sequence, head_dim),
            "kv_heads": heads,
            "causal": True,
            "tile": (block_m, block_n),
            "dtype": "bf16",
            "q_stages": q_stages,
            "kv_slots": kv_slots,
        },
    )

    # These static-grid CTAs are spatially independent.  work_grid and
    # physical_grid are therefore equal; residency determines how many may
    # coexist on one SM and does not enlarge the temporal K/V ring below.
    # One FA4 CTA owns ``q_stages`` consecutive M tiles (QM in the audited
    # model), so the spatial grid is over ceil(sequence / (q_stages*block_m)).
    query_tile_m = q_stages * block_m
    grid = ((sequence + query_tile_m - 1) // query_tile_m, heads, batch)
    with kernel.launch(
        work_grid=grid,
        physical_grid=grid,
        threads=384,
        residency="auto",
        # Match FullModelOptions(grid_policy="model_default"), whose audited
        # FA4 whole-grid policy is the order-independent aggregate lower bound.
        scheduler="aggregate_lb",
    ) as launch:
        with launch.periodic(
            "kv",
            iterations=(sequence + block_n - 1) // block_n,
            stages=kv_slots,
        ) as kv:
            k_shared = kv.buffer(
                "K_shared",
                scope="smem",
                shape=(block_n, head_dim),
                dtype="bf16",
                slots=kv_slots,
            )
            v_shared = kv.buffer(
                "V_shared",
                scope="smem",
                shape=(block_n, head_dim),
                dtype="bf16",
                slots=kv_slots,
            )
            score0 = kv.buffer(
                "S0",
                scope="tmem",
                shape=(block_m, block_n),
                dtype="fp32",
                alias_group="sp_s0",
            )
            score1 = kv.buffer(
                "S1",
                scope="tmem",
                shape=(block_m, block_n),
                dtype="fp32",
                alias_group="sp_s1",
            )
            prob0 = kv.buffer(
                "P0",
                scope="tmem",
                shape=(block_m, block_n),
                dtype="bf16",
                alias_group="sp_s0",
            )
            prob1 = kv.buffer(
                "P1",
                scope="tmem",
                shape=(block_m, block_n),
                dtype="bf16",
                alias_group="sp_s1",
            )
            stats_buffers = tuple(
                kv.buffer(
                    "softmax_state_s{}".format(stage),
                    scope="fragment",
                    shape=(block_m, 2),
                    dtype="fp32",
                    execution_scope="warpgroup",
                )
                for stage in range(q_stages)
            )
            output_buffers = tuple(
                kv.buffer(
                    "O_accumulator_s{}".format(stage),
                    scope="tmem",
                    shape=(block_m, head_dim),
                    dtype="fp32",
                    execution_scope="cta",
                )
                for stage in range(q_stages)
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
                        latency=90 * ns,
                        resources=(ResourceTiming("tma", service_time=10 * ns),),
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
                        latency=90 * ns,
                        resources=(ResourceTiming("tma", service_time=10 * ns),),
                    ),
                    writes=(v_shared,),
                )
                producer.sequence(
                    load_k.at(0), load_v.at(0), resource="tma"
                )

            with kv.actor("tensor") as tensor:
                qk0 = tensor.phase(
                    "gemm_QK_s0",
                    work=Work.mma(
                        attention_tile_flops,
                        operands="Q0 @ K.T",
                        shape=(block_m, block_n, head_dim),
                    ),
                    timing=Timing(
                        latency=44 * ns,
                        resources=(ResourceTiming("tensor", service_time=44 * ns),),
                    ),
                    reads=(k_shared,),
                    writes=(score0,),
                )
                qk1 = tensor.phase(
                    "gemm_QK_s1",
                    work=Work.mma(
                        attention_tile_flops,
                        operands="Q1 @ K.T",
                        shape=(block_m, block_n, head_dim),
                    ),
                    timing=Timing(
                        latency=44 * ns,
                        resources=(ResourceTiming("tensor", service_time=44 * ns),),
                    ),
                    reads=(k_shared,),
                    writes=(score1,),
                )
                pv0 = tensor.phase(
                    "gemm_PV_s0",
                    work=Work.mma(
                        attention_tile_flops,
                        operands="P0 @ V",
                        shape=(block_m, head_dim, block_n),
                    ),
                    timing=Timing(
                        latency=44 * ns,
                        resources=(ResourceTiming("tensor", service_time=44 * ns),),
                    ),
                    reads=(prob0, v_shared),
                )
                pv1 = tensor.phase(
                    "gemm_PV_s1",
                    work=Work.mma(
                        attention_tile_flops,
                        operands="P1 @ V",
                        shape=(block_m, head_dim, block_n),
                    ),
                    timing=Timing(
                        latency=44 * ns,
                        resources=(ResourceTiming("tensor", service_time=44 * ns),),
                    ),
                    reads=(prob1, v_shared),
                )

                # The key FA4 source-window statement: PV at position 0 is
                # interleaved with QK at +1.  These are scheduling labels, not
                # buffer-flow iteration distances.
                tensor.sequence(
                    pv0.at(0),
                    qk0.at(+1),
                    pv1.at(0),
                    qk1.at(+1),
                    resource="tensor",
                )

            # The two softmax workers share CUDA/SFU capacity, but the source
            # does not impose one cyclic order between stages.  Leaving this
            # actor unordered lets the periodic scheduler search arbitration.
            with kv.actor("softmax_workers") as softmax_actor:
                softmax0 = softmax_actor.phase(
                    "online_softmax_s0",
                    work=Work.reduce(
                        7 * block_m * block_n,
                        operation="online_max_sum_exp2",
                    ),
                    timing=Timing(
                        latency=58 * ns,
                        resources=(
                            ResourceTiming("cuda", service_time=32 * ns),
                            ResourceTiming("sfu", service_time=26 * ns),
                        ),
                    ),
                    reads=(score0,),
                    writes=(prob0,),
                )
                softmax1 = softmax_actor.phase(
                    "online_softmax_s1",
                    work=Work.reduce(
                        7 * block_m * block_n,
                        operation="online_max_sum_exp2",
                    ),
                    timing=Timing(
                        latency=58 * ns,
                        resources=(
                            ResourceTiming("cuda", service_time=32 * ns),
                            ResourceTiming("sfu", service_time=26 * ns),
                        ),
                    ),
                    reads=(score1,),
                    writes=(prob1,),
                )
            # The correction worker does have source program order across the
            # two Q stages.  It shares CUDA with softmax, so this is a program
            # sequence rather than global fixed CUDA resource ownership.
            with kv.actor("correction") as correction:
                corr0 = correction.phase(
                    "corr_s0",
                    work=Work.pointwise(
                        block_m * head_dim,
                        operation="rescale_O0",
                    ),
                    timing=Timing(
                        latency=14 * ns,
                        resources=(ResourceTiming("cuda", service_time=14 * ns),),
                    ),
                )
                corr1 = correction.phase(
                    "corr_s1",
                    work=Work.pointwise(
                        block_m * head_dim,
                        operation="rescale_O1",
                    ),
                    timing=Timing(
                        latency=14 * ns,
                        resources=(ResourceTiming("cuda", service_time=14 * ns),),
                    ),
                )
                correction.sequence(corr0.at(0), corr1.at(0))

            # Buffer flow infers K->QK, QK->softmax, P->PV, and V->PV.
            # Correction is a state dependency rather than a buffer handoff.
            kv.after(softmax0.done, corr0.start, name="scale_ready_s0")
            kv.after(corr0.done, pv0.start, name="O_rescaled_s0")
            kv.after(softmax1.done, corr1.start, name="scale_ready_s1")
            kv.after(corr1.done, pv1.start, name="O_rescaled_s1")

            kv.pipeline_buffer(
                k_shared,
                acquire=load_k.start,
                release=qk1.done,
                capacity=kv_slots,
            )
            kv.pipeline_buffer(
                v_shared,
                acquire=load_v.start,
                release=pv1.done,
                capacity=kv_slots,
            )
            # S and P alias one physical TMEM region per Q stage.  Holding the
            # credit through PV prevents the following QK from overwriting P.
            kv.pipeline_buffer(
                score0,
                acquire=qk0.start,
                release=pv0.done,
                capacity=1,
            )
            kv.pipeline_buffer(
                score1,
                acquire=qk1.start,
                release=pv1.done,
                capacity=1,
            )
            kv.lifetime(
                prob0,
                acquire=softmax0.start,
                release=pv0.done,
            )
            kv.lifetime(
                prob1,
                acquire=softmax1.start,
                release=pv1.done,
            )

            for stage, softmax, pv, corr in (
                (0, softmax0, pv0, corr0),
                (1, softmax1, pv1, corr1),
            ):
                stats = kv.state(
                    "online_softmax_s{}".format(stage),
                    storage=stats_buffers[stage],
                )
                kv.carry(
                    stats,
                    source=softmax.done,
                    target=softmax.start,
                    distance=1,
                )
                output = kv.state(
                    "O_accumulator_s{}".format(stage),
                    storage=output_buffers[stage],
                )
                kv.carry(
                    output,
                    source=pv.done,
                    target=corr.start,
                    distance=1,
                )

    return kernel.build()
