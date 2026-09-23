"""Whole-kernel parity tests for the GEMM frontend compatibility adapter."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from tilesight.arch.h100_sxm import H100_SXM
from tilesight.arch.b200 import B200
from tilesight.modeling._pipeline.matmul_pipeline_wave import (
    calculate_matmul_pipeline_wave,
)
from tilesight.modeling.adapters.gemm import (
    evaluate_gemm_legacy,
    extract_gemm_spec,
    gemm_legacy_arch_view,
)
from tilesight.modeling.errors import ModelingValidationError
from tilesight.modeling.tests.fixtures.legacy_gemm import build_gemm
from tilesight.modeling.tests.fixtures.legacy_gemm_b200_cta2 import (
    build_b200_cta2_gemm,
)
from tilesight.modeling.full_model import (
    FeasibilityStatus,
    FullModelOptions,
    model,
)
from tilesight.modeling.ir import Dependency


def _legacy_detail(result):
    return {
        "prologue_time": result.pipeline_detail.prologue_time,
        "steady_time_per_iter": result.pipeline_detail.steady_time_per_iter,
        "epilogue_time": result.pipeline_detail.epilogue_time,
        "mem_time_per_iter": result.pipeline_detail.mem_time_per_iter,
        "compute_time_per_iter": result.pipeline_detail.compute_time_per_iter,
    }


def test_gemm_full_model_is_field_for_field_legacy_parity():
    ir = build_gemm()
    spec = extract_gemm_spec(ir)
    arch = H100_SXM().set_to_microbench()

    # The legacy L1.5 cache simulator samples a permutation.  Resetting the
    # same seed before each public entry point verifies the adapter itself is
    # numerically transparent without changing legacy RNG behavior.
    rng_state = np.random.get_state()
    try:
        np.random.seed(20260817)
        legacy = calculate_matmul_pipeline_wave(**spec.legacy_kwargs(arch))
        np.random.seed(20260817)
        result = model(
            ir,
            arch,
            FullModelOptions(kernel_launch_s=2.0e-6, host_dispatch_s=3.0e-6),
        )
    finally:
        np.random.set_state(rng_state)

    assert result.adapter == "gemm_pipeline_wave"
    hash(result)
    assert result.timing.kernel_body_s == legacy.total_latency
    assert result.timing.per_tile_s == legacy.per_tile_latency
    assert result.timing.launch_s == 2.0e-6
    assert result.timing.host_dispatch_s == 3.0e-6
    assert result.total_s == pytest.approx(
        legacy.total_latency + 5.0e-6, rel=0.0, abs=1.0e-18
    )

    assert result.timing.prologue_s == legacy.pipeline_detail.prologue_time
    assert result.timing.epilogue_s == legacy.pipeline_detail.epilogue_time
    assert (
        result.timing.prologue_s
        + result.timing.steady_s
        + result.timing.epilogue_s
    ) == pytest.approx(legacy.per_tile_latency, rel=0.0, abs=1.0e-18)
    assert result.ii.selected_s == legacy.pipeline_detail.steady_time_per_iter
    assert result.ii.resource_lower_bound_s == max(
        legacy.pipeline_detail.mem_time_per_iter,
        legacy.pipeline_detail.compute_time_per_iter,
    )
    assert result.ii.constructive is False

    assert result.occupancy.resident_ctas_per_sm == legacy.tiles_per_sm
    assert result.occupancy.smem_bytes_per_cta == legacy.smem_footprint
    assert result.occupancy.register_value == legacy.reg_footprint
    assert result.occupancy.warps_per_cta == 4
    assert result.grid.work_grid == (32, 32)
    assert result.grid.physical_grid == (32, 32, 1)
    assert result.grid.total_work_units == 1024
    assert result.grid.waves == legacy.waves
    assert result.grid.tail_factor == pytest.approx(100 / 132)

    legacy_payload = dict(result.legacy)
    for field in (
        "per_tile_latency",
        "total_latency",
        "ddr_util",
        "l2_util",
        "l2_hit_rate",
        "smem_util",
        "compute_util",
        "smem_footprint",
        "reg_footprint",
        "tiles_per_sm",
        "waves",
    ):
        assert legacy_payload[field] == getattr(legacy, field)
    assert dict(legacy_payload["pipeline_detail"]) == _legacy_detail(legacy)

    assert result.liveness.status is FeasibilityStatus.UNKNOWN
    assert result.liveness.peaks
    assert result.fusion.status is FeasibilityStatus.NOT_REQUESTED
    assert any("not requested" in item for item in result.fusion.diagnostics)


def test_gemm_default_adds_fixed_launch_once_and_zero_preserves_body_parity(
    monkeypatch,
):
    ir = build_gemm()
    arch = H100_SXM().set_to_microbench()

    # Reuse a real-shaped legacy result while avoiding a second sampled cache
    # simulation in the adapter invocation.
    spec = extract_gemm_spec(ir)
    np.random.seed(11)
    legacy = calculate_matmul_pipeline_wave(**spec.legacy_kwargs(arch))
    monkeypatch.setattr(
        "tilesight.modeling._pipeline.matmul_pipeline_wave."
        "calculate_matmul_pipeline_wave",
        lambda **_kwargs: legacy,
    )
    result = model(ir, arch)
    body_only = model(ir, arch, FullModelOptions(kernel_launch_s=0.0))

    assert result.timing.kernel_body_s == legacy.total_latency
    assert result.timing.launch_s == 2.0e-6
    assert result.timing.host_dispatch_s == 0.0
    assert result.total_s == legacy.total_latency + 2.0e-6
    assert dict(result.provenance)["launch_convention"] == "fixed_2us_default"
    assert body_only.timing.launch_s == 0.0
    assert body_only.total_s == legacy.total_latency
    assert dict(body_only.provenance)["launch_convention"] == "explicit_option"


def test_gemm_preserves_multi_cta_residency_and_integer_wave_tail():
    ir = build_gemm()
    params = dict(ir.params)
    params["warp_tile"] = (128, 64, 64)
    launch = replace(ir.launches[0], threads=64)
    two_warp_ir = replace(ir, params=params, launches=(launch,))

    rng_state = np.random.get_state()
    try:
        np.random.seed(91)
        result = model(
            two_warp_ir,
            H100_SXM().set_to_microbench(),
            FullModelOptions(kernel_launch_s=0.0),
        )
    finally:
        np.random.set_state(rng_state)

    assert result.occupancy.resident_ctas_per_sm == 2
    assert result.occupancy.warps_per_cta == 2
    assert result.grid.total_work_units == 1024
    assert result.grid.waves == 1024 / (132 * 2)
    metrics = dict(result.resource_metrics)
    assert metrics["work_units_per_wave"] == 264
    assert metrics["full_waves"] == 3
    assert metrics["tail_work_units"] == 232
    assert result.grid.tail_factor == 232 / 264


@pytest.mark.parametrize("mode", ["periodic_best", "periodic_worst"])
def test_gemm_parity_adapter_does_not_silently_substitute_periodic_ii(mode):
    with pytest.raises(ModelingValidationError, match="lower_periodic"):
        model(
            build_gemm(),
            H100_SXM().set_to_microbench(),
            FullModelOptions(ii_mode=mode),
        )


def test_gemm_full_model_rejects_missing_calibrated_backend_configuration():
    ir = build_gemm()
    params = dict(ir.params)
    params.pop("mem_levels")
    incomplete = replace(ir, params=params)
    with pytest.raises(ModelingValidationError, match="require mem_levels"):
        extract_gemm_spec(incomplete)


def test_gemm_full_model_rejects_frontend_schedule_mismatch():
    ir = build_gemm()
    launch = ir.launches[0]

    wrong_threads = replace(ir, launches=(replace(launch, threads=64),))
    with pytest.raises(ModelingValidationError, match="threads"):
        extract_gemm_spec(wrong_threads)

    wrong_grid = replace(
        ir,
        launches=(replace(launch, physical_grid=(1, 1, 1)),),
    )
    with pytest.raises(ModelingValidationError, match="physical_grid"):
        extract_gemm_spec(wrong_grid)

    loop = launch.periodic_loops[0]
    tensor_actor = next(actor for actor in loop.actors if actor.name == "tensor")
    mma = tensor_actor.phases[0]
    bad_actor = replace(
        tensor_actor,
        phases=(replace(mma, work=replace(mma.work, flops=1.0)),),
    )
    bad_loop = replace(
        loop,
        actors=tuple(
            bad_actor if actor.name == "tensor" else actor for actor in loop.actors
        ),
    )
    bad_ir = replace(
        ir,
        launches=(replace(launch, periodic_loops=(bad_loop,)),),
    )
    with pytest.raises(ModelingValidationError, match="mma Work"):
        extract_gemm_spec(bad_ir)

    producer_actor = next(actor for actor in loop.actors if actor.name == "producer")
    load_a, load_b = producer_actor.phases
    bad_producer = replace(
        producer_actor,
        phases=(replace(load_a, writes=tuple()), load_b),
    )
    bad_flow_loop = replace(
        loop,
        actors=tuple(
            bad_producer if actor.name == "producer" else actor
            for actor in loop.actors
        ),
    )
    bad_flow_ir = replace(
        ir,
        launches=(replace(launch, periodic_loops=(bad_flow_loop,)),),
    )
    with pytest.raises(ModelingValidationError, match="dataflow"):
        extract_gemm_spec(bad_flow_ir)

    bad_carry_loop = replace(
        loop,
        carries=(replace(loop.carries[0], iteration_distance=99),),
    )
    bad_carry_ir = replace(
        ir,
        launches=(replace(launch, periodic_loops=(bad_carry_loop,)),),
    )
    with pytest.raises(ModelingValidationError, match="accumulator carry"):
        extract_gemm_spec(bad_carry_ir)

    bad_dependency_loop = replace(
        loop,
        dependencies=(
            Dependency(
                source=mma.done,
                target=load_a.start,
                iteration_distance=1,
            ),
        ),
    )
    bad_dependency_ir = replace(
        ir,
        launches=(replace(launch, periodic_loops=(bad_dependency_loop,)),),
    )
    with pytest.raises(ModelingValidationError, match="explicit dependencies"):
        extract_gemm_spec(bad_dependency_ir)

    a_buffer = next(buffer for buffer in loop.buffers if buffer.name == "A_shared")
    bad_dtype_loop = replace(
        loop,
        buffers=tuple(
            replace(a_buffer, dtype="fp64") if buffer.name == "A_shared" else buffer
            for buffer in loop.buffers
        ),
    )
    bad_dtype_ir = replace(
        ir,
        launches=(replace(launch, periodic_loops=(bad_dtype_loop,)),),
    )
    with pytest.raises(ModelingValidationError, match="buffer A_shared"):
        extract_gemm_spec(bad_dtype_ir)


def test_gemm_cta2_cluster_axis_must_match_legacy_selection():
    ir = build_gemm()
    params = dict(ir.params)
    params["mma_type"] = "utcmma_cta2"
    launch = ir.launches[0]
    loop = launch.periodic_loops[0]
    tensor_actor = next(actor for actor in loop.actors if actor.name == "tensor")
    mma = tensor_actor.phases[0]
    doubled_tensor = replace(
        tensor_actor,
        phases=(replace(mma, work=replace(mma.work, flops=mma.work.flops * 2)),),
    )
    cta2_loop = replace(
        loop,
        actors=tuple(
            doubled_tensor if actor.name == "tensor" else actor
            for actor in loop.actors
        ),
    )
    valid = replace(
        ir,
        params=params,
        launches=(
            replace(
                launch,
                work_grid=(16, 32, 1),
                physical_grid=(32, 32, 1),
                cluster=(2, 1, 1),
                periodic_loops=(cta2_loop,),
            ),
        ),
    )
    assert extract_gemm_spec(valid).mma_type == "utcmma_cta2"

    wrong_axis = replace(
        valid,
        launches=(replace(valid.launches[0], cluster=(1, 2, 1)),),
    )
    with pytest.raises(ModelingValidationError, match="expected axis"):
        extract_gemm_spec(wrong_axis)

    logical_grid_as_physical = replace(
        valid,
        launches=(
            replace(valid.launches[0], physical_grid=valid.launches[0].work_grid),
        ),
    )
    with pytest.raises(ModelingValidationError, match="actual CTA launch grid"):
        extract_gemm_spec(logical_grid_as_physical)


def test_b200_cta2_read_only_arch_view_runs_and_preserves_legacy_parity():
    ir = build_b200_cta2_gemm()
    spec = extract_gemm_spec(ir)
    arch = B200().set_to_microbench()
    assert not hasattr(arch, "get_tensor_core_minimum_ptx")
    legacy_arch, compatibility = gemm_legacy_arch_view(arch)

    rng_state = np.random.get_state()
    try:
        np.random.seed(20260818)
        direct = calculate_matmul_pipeline_wave(
            **spec.legacy_kwargs(legacy_arch)
        )
        np.random.seed(20260818)
        evaluation = evaluate_gemm_legacy(ir, arch)
        np.random.seed(20260818)
        typed = model(ir, arch, FullModelOptions(kernel_launch_s=0.0))
    finally:
        np.random.set_state(rng_state)

    assert evaluation.legacy_result.total_latency == direct.total_latency
    assert evaluation.legacy_result.per_tile_latency == direct.per_tile_latency
    assert evaluation.legacy_result.tiles_per_sm == direct.tiles_per_sm
    assert evaluation.legacy_result.waves == direct.waves
    assert evaluation.wave.cluster_size == 2
    assert evaluation.wave.cluster_axis == "M"
    assert evaluation.wave.effective_mma_type == "utcmma_cta2"
    assert evaluation.wave.effective_work_units_per_logical_slot == (
        direct.tiles_per_sm
    )
    assert evaluation.arch_compatibility == compatibility
    assert dict(compatibility)["mode"] == "b200_read_only_legacy_compat_view"
    assert not hasattr(arch, "get_tensor_core_minimum_ptx")

    assert typed.timing.kernel_body_s == direct.total_latency
    assert typed.grid.work_grid == (16, 32)
    assert typed.grid.physical_grid == (32, 32, 1)
    assert typed.grid.total_work_units == 16 * 32
    assert typed.occupancy.cluster_size == 2
    assert typed.occupancy.resident_ctas_per_sm is None
    metric = dict(typed.resource_metrics)
    provenance = dict(typed.provenance)
    key = "legacy_effective_work_units_per_logical_2sm_slot"
    assert metric[key] == direct.tiles_per_sm
    assert provenance[key] == direct.tiles_per_sm
    assert provenance["typed_cta_occupancy"].startswith("unknown")
    assert provenance["liveness_residency_binding"].startswith("none")
    assert all(
        capacity.resident_ctas_per_sm == 1.0
        for capacity in typed.liveness.capacity
    )


def test_requested_cta2_single_tile_grid_falls_back_to_typed_cta1_semantics():
    ir = build_gemm()
    params = dict(ir.params)
    params["shape"] = (128, 128, params["shape"][2])
    params["mma_type"] = "utcmma_cta2"
    launch = replace(
        ir.launches[0],
        work_grid=(1, 1, 1),
        physical_grid=(1, 1, 1),
        cluster=(1, 1, 1),
    )
    fallback_ir = replace(ir, params=params, launches=(launch,))
    arch = B200().set_to_microbench()

    rng_state = np.random.get_state()
    try:
        np.random.seed(37)
        evaluation = evaluate_gemm_legacy(fallback_ir, arch)
        np.random.seed(37)
        typed = model(
            fallback_ir,
            arch,
            FullModelOptions(kernel_launch_s=0.0),
        )
    finally:
        np.random.set_state(rng_state)

    assert evaluation.spec.mma_type == "utcmma_cta2"
    assert evaluation.wave.cluster_axis is None
    assert evaluation.wave.cluster_size == 1
    assert evaluation.wave.effective_mma_type == "utcmma_cta1"
    assert typed.grid.work_grid == (1, 1)
    assert typed.grid.physical_grid == (1, 1, 1)
    assert typed.occupancy.cluster_size == 1
    assert typed.occupancy.resident_ctas_per_sm == (
        evaluation.legacy_result.tiles_per_sm
    )
    key = "legacy_effective_work_units_per_logical_2sm_slot"
    assert key not in dict(typed.resource_metrics)
    provenance = dict(typed.provenance)
    assert provenance["requested_mma_type"] == "utcmma_cta2"
    assert provenance["effective_mma_type"] == "utcmma_cta1"
    assert provenance["cluster_semantics"] == (
        "requested_cta2_effective_cta1_fallback"
    )
    assert all(
        capacity.resident_ctas_per_sm
        == evaluation.legacy_result.tiles_per_sm
        for capacity in typed.liveness.capacity
    )
