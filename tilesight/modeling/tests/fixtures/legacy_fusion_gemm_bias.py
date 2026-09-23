"""Synthetic kernel builder for frontend/executor/adapter regression tests.

Fixed costs in this fixture are test inputs, not hardware measurements or
performance-modeling examples. Use program/examples/ and analyze for modeling.
"""

from __future__ import annotations

from types import SimpleNamespace

from tilesight.modeling._pipeline.periodic_schedule import (
    Dependency as ScheduleDependency,
    PeriodicDAG,
    Phase as SchedulePhase,
    ResourceUse,
    TokenBuffer,
)

from tilesight.modeling import (
    Kernel,
    LoopRegion,
    NativeModelOptions,
    ExecutionDomain,
    OwnershipMap,
    PeriodicAxisIR,
    PhaseRegion,
    ResourceTiming,
    SequenceRegion,
    Timing,
    analyze_fusion,
    analyze_liveness,
    model_native,
    ns,
)


def build_fusion_gemm_bias():
    kernel = Kernel("fusion_gemm_bias")
    with kernel.launch(work_grid=(32, 32, 1), threads=128) as launch:
        with launch.periodic("ko", iterations=64, stages=3) as loop:
            with loop.actor(
                "mainloop_warpgroup", execution_scope="warpgroup"
            ) as actor:
                load = actor.load(
                    "load_tiles",
                    bytes=2 * 128 * 64 * 2,
                    source="ddr",
                    destination="smem",
                    engine="tma",
                    timing=Timing(
                        72 * ns,
                        (ResourceTiming("tma", service_time=8 * ns),),
                    ),
                )
                gemm = actor.compute(
                    "gemm",
                    flops=2 * 128 * 128 * 64,
                    engine="tensor",
                    op="mma",
                    timing=Timing(
                        32 * ns,
                        (ResourceTiming("tensor", service_time=32 * ns),),
                    ),
                )
                actor.sequence(
                    load.at(0),
                    gemm.at(0),
                    order="completion",
                )
        with launch.periodic("epilogue", iterations=1) as epilogue:
            with epilogue.actor(
                "epilogue_warpgroup", execution_scope="warpgroup"
            ) as actor:
                bias = actor.compute(
                    "bias",
                    flops=128 * 128,
                    engine="vector",
                    op="add",
                    timing=Timing(
                        8 * ns,
                        (ResourceTiming("cuda", service_time=8 * ns),),
                    ),
                )
                store = actor.store(
                    "store_output",
                    bytes=128 * 128 * 2,
                    source="register",
                    destination="ddr",
                    engine="ldst",
                    timing=Timing(
                        40 * ns,
                        (ResourceTiming("ldst", service_time=8 * ns),),
                    ),
                )
                actor.sequence(
                    bias.at(0),
                    store.at(0),
                    order="completion",
                )

    accumulator_ownership = OwnershipMap(
        domain=ExecutionDomain(
            scope="cta",
            instances_per_cta=1,
            members_per_instance=128,
            provenance="fusion_example",
        ),
        layout="row_major",
        index_map={"logical_index": "thread_linear"},
        provenance="fusion_example",
    )
    accumulator = kernel.tensor_value(
        "gemm_accumulator",
        shape=(128, 128),
        dtype="fp32",
        bytes=128 * 128 * 4,
        layout="row_major",
        ownership=accumulator_ownership,
    )
    standalone = kernel.materialize(
        "standalone_gemm_to_bias",
        accumulator,
        producer=gemm.done,
        consumer=bias.start,
        producer_fragment="gemm",
        consumer_fragment="bias",
        storage="ddr",
        producer_source="register",
        consumer_target="register",
    )
    with kernel.fusion("gemm_bias") as fusion:
        fusion.handoff(
            "accumulator_smem",
            standalone,
            via="smem",
            execution_scope="cta",
            release=bias.done,
        )
    return kernel.build()


def _format_path(path):
    return " -> ".join(path)


def build_native_fused_region(ir):
    """Place the K pipeline in the body and bias/store in one-time drain."""

    mainloop = {phase.name: phase for phase in ir.periodic_loop("ko").phases}
    epilogue = {
        phase.name: phase for phase in ir.periodic_loop("epilogue").phases
    }
    load = mainloop["load_tiles"]
    gemm = mainloop["gemm"]
    bias = epilogue["bias"]
    store = epilogue["store_output"]
    dag = PeriodicDAG(
        phases=(
            SchedulePhase(
                load.name,
                latency=load.timing.latency,
                resources=tuple(
                    ResourceUse(item.resource, item.service_time, item.offset)
                    for item in load.timing.resources
                ),
            ),
            SchedulePhase(
                gemm.name,
                latency=gemm.timing.latency,
                resources=tuple(
                    ResourceUse(item.resource, item.service_time, item.offset)
                    for item in gemm.timing.resources
                ),
            ),
        ),
        dependencies=(
            ScheduleDependency(
                load.name,
                gemm.name,
                iteration_distance=0,
                min_delay=load.timing.latency,
                name="tile_ready",
            ),
        ),
        token_buffers=(
            TokenBuffer(
                "three_stage_tile",
                acquire=load.name,
                release=gemm.name,
                capacity=3,
                acquire_offset=0.0,
                release_offset=gemm.timing.latency,
            ),
        ),
    )
    return LoopRegion(
        "fused_gemm_bias_tile",
        trip_count=64,
        body=SequenceRegion(
            "k_body",
            (PhaseRegion(load), PhaseRegion(gemm)),
        ),
        epilogue=SequenceRegion(
            "bias_store_epilogue",
            (PhaseRegion(bias), PhaseRegion(store)),
        ),
        periodic_axis=PeriodicAxisIR(
            dag=dag,
            liveness_loop_name="ko",
            ii_mode="periodic_best",
            summary_policy="macro",
        ),
    )


def model_native_fused_static(ir):
    """Evaluate an explicitly asserted fused static timing program."""

    arch = SimpleNamespace(
        sm_count=4,
        configurable_smem_capacity=128 * 1024,
        tmem_capacity_per_sm=0,
    )
    return model_native(
        ir,
        build_native_fused_region(ir),
        arch,
        NativeModelOptions(
            ii_mode="periodic_best",
            cost_source="explicit_fused_static_timing",
            kernel_launch_s=2e-6,
        ),
    )
