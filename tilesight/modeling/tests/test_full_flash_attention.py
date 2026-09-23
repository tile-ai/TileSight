"""Exact compatibility tests for FA3/FA4 complete-kernel adapters."""

from __future__ import annotations

from dataclasses import replace

import pytest

from tilesight.modeling._attention.fa3_model import model_fa3_us
from tilesight.modeling._attention.fa4_model import model_fa4_us
from tilesight.arch.b200 import B200
from tilesight.arch.h200_sxm import H200_SXM
from tilesight.modeling.tests.fixtures.legacy_fa3 import build_fa3
from tilesight.modeling.tests.fixtures.legacy_fa4 import build_fa4
from tilesight.modeling.full_model import (
    FeasibilityStatus,
    FullModelOptions,
    model,
)
from tilesight.modeling.ir import freeze_value


def _assert_frozen_legacy_is_lossless(result, breakdown):
    frozen = dict(result.legacy)
    assert set(frozen) == set(breakdown)
    for key, value in breakdown.items():
        assert frozen[key] == freeze_value(value), key


def _replace_launch(kernel, **changes):
    return replace(
        kernel,
        launches=(replace(kernel.launches[0], **changes),),
    )


def _replace_loop(kernel, **changes):
    launch = kernel.launches[0]
    loop = replace(launch.periodic_loops[0], **changes)
    return replace(kernel, launches=(replace(launch, periodic_loops=(loop,)),))


def _mutate_full_fa_template(kernel, mutation):
    launch = kernel.launches[0]
    loop = launch.periodic_loops[0]
    if mutation == "work_grid":
        return _replace_launch(kernel, work_grid=(999,) + launch.work_grid[1:])
    if mutation == "tile":
        params = dict(kernel.params)
        params["tile"] = (params["tile"][0] + 1, params["tile"][1])
        return replace(kernel, params=params)
    if mutation == "physical_grid":
        return _replace_launch(kernel, physical_grid=(999, 1, 1))
    if mutation == "threads":
        return _replace_launch(kernel, threads=launch.threads + 32)
    if mutation == "cluster":
        return _replace_launch(kernel, cluster=(2, 1, 1))
    if mutation == "residency":
        return _replace_launch(kernel, residency=2)
    if mutation == "remove_loop":
        return _replace_launch(kernel, periodic_loops=tuple())
    if mutation == "extra_launch":
        return replace(kernel, launches=(launch, launch))
    if mutation == "iterations":
        return _replace_loop(kernel, iterations=loop.iterations + 1)
    if mutation == "loop_name":
        return _replace_loop(kernel, name="not_kv")
    if mutation == "stages":
        return _replace_loop(kernel, stages=loop.stages + 1)
    if mutation == "scheduler":
        wrong = "aggregate_lb" if kernel.name == "fa3_forward" else "sectioned_lpt"
        return _replace_launch(kernel, scheduler=wrong)
    if mutation == "remove_phase":
        actors = list(loop.actors)
        actors[0] = replace(actors[0], phases=actors[0].phases[:-1])
        return _replace_loop(kernel, actors=tuple(actors))
    if mutation == "phase_work":
        actors = list(loop.actors)
        phases = list(actors[0].phases)
        phases[0] = replace(
            phases[0], work=replace(phases[0].work, bytes=phases[0].work.bytes + 1.0)
        )
        actors[0] = replace(actors[0], phases=tuple(phases))
        return _replace_loop(kernel, actors=tuple(actors))
    if mutation == "phase_resource":
        actors = list(loop.actors)
        phases = list(actors[0].phases)
        phases[0] = replace(
            phases[0], timing=replace(phases[0].timing, resources=tuple())
        )
        actors[0] = replace(actors[0], phases=tuple(phases))
        return _replace_loop(kernel, actors=tuple(actors))
    if mutation == "phase_flow":
        actors = list(loop.actors)
        phases = list(actors[0].phases)
        phases[0] = replace(phases[0], writes=tuple())
        actors[0] = replace(actors[0], phases=tuple(phases))
        return _replace_loop(kernel, actors=tuple(actors))
    if mutation == "buffer_shape":
        buffers = list(loop.buffers)
        buffers[0] = replace(
            buffers[0], shape=(buffers[0].shape[0] + 1,) + buffers[0].shape[1:]
        )
        return _replace_loop(kernel, buffers=tuple(buffers))
    if mutation == "lifetime":
        return _replace_loop(kernel, lifetimes=loop.lifetimes[:-1])
    if mutation == "pipeline_event":
        pipeline_buffers = list(loop.pipeline_buffers)
        pipeline_buffers[0] = replace(
            pipeline_buffers[0],
            acquire=replace(pipeline_buffers[0].acquire, kind="done"),
        )
        return _replace_loop(kernel, pipeline_buffers=tuple(pipeline_buffers))
    if mutation == "pipeline_residence":
        pipeline_buffers = list(loop.pipeline_buffers)
        pipeline_buffers[0] = replace(
            pipeline_buffers[0], minimum_residence=1.0e-9
        )
        return _replace_loop(kernel, pipeline_buffers=tuple(pipeline_buffers))
    if mutation == "state_storage":
        states = list(loop.states)
        states[0] = replace(states[0], storage=None)
        return _replace_loop(kernel, states=tuple(states))
    if mutation == "resource_sequence":
        sequences = list(loop.resource_sequences)
        index = next(
            index
            for index, item in enumerate(sequences)
            if item.resource == "tensor"
        )
        sequences[index] = replace(
            sequences[index], sequence=tuple(reversed(sequences[index].sequence))
        )
        return _replace_loop(kernel, resource_sequences=tuple(sequences))
    raise AssertionError("unknown test mutation %r" % mutation)


def test_fa3_default_has_exact_legacy_total_and_all_components():
    arch = H200_SXM().set_to_microbench()
    kernel = build_fa3()
    legacy_us, legacy = model_fa3_us(
        1,
        32,
        2048,
        128,
        kv_h=32,
        causal=True,
        arch=arch,
        dtype_bytes=2,
        accum_bytes=4,
        max_util=0.85,
        host_dispatch_overhead_us=0.0,
        scheduler_model="source",
        inner_schedule_model="resource_ii",
    )

    result = model(kernel, arch)

    hash(result)
    assert result.total_s == legacy_us * 1.0e-6
    assert result.timing.kernel_body_s == legacy["gpu_body_us"] * 1.0e-6
    assert result.timing.launch_s == legacy["kernel_launch_us"] * 1.0e-6
    assert result.timing.host_dispatch_s == legacy["host_dispatch_us"] * 1.0e-6
    assert result.timing.prologue_s == legacy["t_prologue_us"] * 1.0e-6
    assert result.timing.steady_s == legacy["t_steady_us"] * 1.0e-6
    assert result.timing.epilogue_s == legacy["t_epilogue_us"] * 1.0e-6
    assert result.grid.total_work_units == legacy["grid_work_tiles"]
    assert result.grid.waves == legacy["waves"]
    assert result.occupancy.resident_ctas_per_sm == 1.0
    assert result.liveness.status is FeasibilityStatus.UNKNOWN
    assert result.fusion.status is FeasibilityStatus.NOT_REQUESTED
    _assert_frozen_legacy_is_lossless(result, legacy)


def test_fa4_default_has_exact_legacy_total_and_split_tile_overhead():
    arch = B200().set_to_microbench()
    kernel = build_fa4()
    legacy_us, legacy = model_fa4_us(
        1,
        16,
        2048,
        128,
        kv_h=16,
        causal=True,
        arch=arch,
        dtype=2,
        accum=4,
        max_util=0.85,
        correction_freq=1.0,
        dispatch_overhead_us=0.0,
        kernel_launch_overhead_us=2.0,
        scheduler_model="aggregate_lb",
        inner_schedule_model="resource_ii",
        periodic_search_profile="reference",
    )

    result = model(kernel, arch)

    assert result.total_s == legacy_us * 1.0e-6
    assert result.timing.kernel_body_s == legacy["kernel_body_us"] * 1.0e-6
    assert result.timing.launch_s == legacy["kernel_launch_us"] * 1.0e-6
    assert result.timing.host_dispatch_s == legacy["dispatch_us"] * 1.0e-6
    assert result.timing.steady_s == legacy["t_steady_us"] * 1.0e-6
    split_tile_overhead_s = result.timing.prologue_s + result.timing.epilogue_s
    assert split_tile_overhead_s == pytest.approx(
        legacy["t_tile_oh_us"] * 1.0e-6, rel=0.0, abs=1.0e-18
    )
    assert result.grid.total_work_units == legacy["grid"]
    assert result.grid.waves == legacy["waves"]
    _assert_frozen_legacy_is_lossless(result, legacy)


def test_unified_options_preserve_fa4_periodic_search_and_dispatch_metadata():
    arch = B200().set_to_microbench()
    options = FullModelOptions(
        ii_mode="periodic_best",
        grid_policy="sectioned_lpt",
        search_profile="fast",
        kernel_launch_s=3.0e-6,
        host_dispatch_s=7.0e-6,
    )
    legacy_us, legacy = model_fa4_us(
        1,
        16,
        2048,
        128,
        kv_h=16,
        causal=True,
        arch=arch,
        dtype=2,
        accum=4,
        max_util=0.85,
        correction_freq=1.0,
        dispatch_overhead_us=7.0,
        kernel_launch_overhead_us=3.0,
        scheduler_model="sectioned_lpt",
        inner_schedule_model="periodic_best",
        periodic_search_profile="fast",
    )

    kernel = build_fa4()
    kernel = replace(
        kernel,
        launches=(replace(kernel.launches[0], scheduler="sectioned_lpt"),),
    )
    result = model(kernel, arch, options)

    assert result.total_s == legacy_us * 1.0e-6
    assert result.ii.selected_s == legacy["ii_selected_ns"] * 1.0e-9
    assert result.ii.best_s == legacy["periodic_best_ii_ns"] * 1.0e-9
    assert result.ii.worst_s == legacy["periodic_worst_ii_ns"] * 1.0e-9
    assert result.ii.search_complete == legacy["ii_search_complete"]
    resources = dict(result.resource_metrics)
    search = dict(resources["search"])
    assert search["periodic_search_profile"] == "fast"
    assert search["periodic_orderings_explored"] == legacy[
        "periodic_orderings_explored"
    ]
    _assert_frozen_legacy_is_lossless(result, legacy)


def test_fa3_periodic_envelope_is_retained_in_typed_and_legacy_metadata():
    arch = H200_SXM().set_to_microbench()
    result = model(
        build_fa3(), arch, FullModelOptions(ii_mode="periodic_worst")
    )
    legacy = dict(result.legacy)

    assert result.ii.constructive is True
    assert result.ii.search_complete is True
    assert result.ii.scope == "model_exhaustive"
    assert result.ii.best_s == dict(legacy["ii_metadata"])[
        "periodic_best_ii_ns"
    ] * 1.0e-9
    assert result.ii.worst_s == dict(legacy["ii_metadata"])[
        "periodic_worst_ii_ns"
    ] * 1.0e-9
    search = dict(dict(result.resource_metrics)["search"])
    assert search["periodic_search_complete"] is True
    assert dict(search["periodic_best_resource_orders"])["tensor"] == (
        "gemm_QK",
        "gemm_PV",
    )


def test_fa3_dispatch_is_separate_and_fixed_launch_cannot_be_changed():
    arch = H200_SXM().set_to_microbench()
    result = model(
        build_fa3(),
        arch,
        FullModelOptions(host_dispatch_s=7.0e-6),
    )
    assert result.timing.launch_s == 2.0e-6
    assert result.timing.host_dispatch_s == 7.0e-6
    assert result.total_s == pytest.approx(
        result.timing.kernel_body_s
        + result.timing.launch_s
        + result.timing.host_dispatch_s,
        rel=0.0,
        abs=1.0e-18,
    )

    with pytest.raises(ValueError, match="fixes kernel launch overhead"):
        model(
            build_fa3(),
            arch,
            FullModelOptions(kernel_launch_s=3.0e-6),
        )


@pytest.mark.parametrize(
    "kernel,arch,policy",
    (
        (build_fa3, lambda: H200_SXM().set_to_microbench(), "sectioned_lpt"),
        (build_fa4, lambda: B200().set_to_microbench(), "source"),
    ),
)
def test_family_specific_grid_policy_fails_loudly(kernel, arch, policy):
    with pytest.raises(ValueError, match="grid_policy"):
        model(kernel(), arch(), FullModelOptions(grid_policy=policy))


@pytest.mark.parametrize(
    "build,arch",
    (
        (build_fa3, lambda: H200_SXM().set_to_microbench()),
        (build_fa4, lambda: B200().set_to_microbench()),
    ),
    ids=("fa3", "fa4"),
)
@pytest.mark.parametrize(
    "mutation",
    (
        "work_grid",
        "tile",
        "physical_grid",
        "threads",
        "cluster",
        "residency",
        "remove_loop",
        "extra_launch",
        "iterations",
        "loop_name",
        "stages",
        "scheduler",
        "remove_phase",
        "phase_work",
        "phase_resource",
        "phase_flow",
        "buffer_shape",
        "lifetime",
        "pipeline_event",
        "pipeline_residence",
        "state_storage",
        "resource_sequence",
    ),
)
def test_full_fa_adapter_rejects_mutated_frontend_template(
    build, arch, mutation
):
    kernel = _mutate_full_fa_template(build(), mutation)
    with pytest.raises(ValueError, match="frontend template"):
        model(kernel, arch())
