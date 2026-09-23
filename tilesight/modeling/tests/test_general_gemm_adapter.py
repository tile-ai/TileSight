"""Semantic GEMM binding, calibrated parity, and periodic diagnostics."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from tilesight.arch.h100_sxm import H100_SXM
from tilesight.modeling._pipeline.matmul_pipeline_wave import (
    calculate_matmul_pipeline_wave,
)
from tilesight.modeling import Kernel, ResourceTiming, Timing, Work
from tilesight.modeling.adapters.general_gemm import (
    BoundGemmCostOracle,
    GeneralGemmOptions,
    build_general_gemm_cache_histogram,
    build_general_gemm_cache_problem,
    evaluate_general_gemm_cache,
    model_general_gemm,
)
from tilesight.modeling.cache import ProgressJitterConfig
from tilesight.modeling.errors import ModelingValidationError
from tilesight.modeling.ops import (
    ThroughputFallbackPolicy,
    UnsupportedThroughputError,
    make_gemm,
)


MEM_LEVELS = {
    "in1": [1, 1, 1, 2],
    "in2": [0, 1, 1, 2],
    "out1": [0, 0, 1, 4],
}


CASES = (
    ("A", (1024, 1024, 1024), (128, 128, 64), (64, 64, 64), 1, 397.7351526793783),
    ("B", (4096, 4096, 4096), (128, 128, 64), (64, 64, 64), 1, 397.7351526793783),
    ("C", (4096, 4096, 4096), (128, 128, 64), (128, 64, 64), 2, 397.7351526793783),
    ("D", (16384, 16384, 2048), (128, 256, 64), (64, 128, 64), 1, 795.4703053587566),
)


BF16_TENSOR_ALIAS = ThroughputFallbackPolicy(
    dtype_aliases=(("bf16", "fp16"),)
)


def _options(**kwargs):
    kwargs.setdefault("tensor_throughput_fallback", BF16_TENSOR_ALIAS)
    return GeneralGemmOptions(**kwargs)


def _program(
    shape,
    tile,
    warp,
    *,
    a_storage_dtype="bf16",
    b_storage_dtype="bf16",
    result_dtype="fp32",
    compute_dtype="bf16",
    accumulation_dtype="fp32",
    accumulator_buffer_dtype=None,
    a_work_dtype=None,
    b_work_dtype=None,
    mem_levels=None,
    batch=1,
    object_primitives=False,
    include_mma_attrs=True,
):
    m, n, k = shape
    tm, tn, tk = tile
    wm, wn, wk = warp
    stages = 3
    dtype_bytes = {
        "fp8_e4m3": 1,
        "fp8_e5m2": 1,
        "bf16": 2,
        "fp16": 2,
        "fp32": 4,
    }
    if a_work_dtype is None:
        a_work_dtype = a_storage_dtype
    if b_work_dtype is None:
        b_work_dtype = b_storage_dtype
    if mem_levels is None:
        mem_levels = {
            "in1": [1, 1, 1, dtype_bytes[a_storage_dtype]],
            "in2": [0, 1, 1, dtype_bytes[b_storage_dtype]],
            "out1": [0, 0, 1, dtype_bytes[result_dtype]],
        }
    if accumulator_buffer_dtype is None:
        accumulator_buffer_dtype = accumulation_dtype
    kernel = Kernel(
        "semantic_gemm",
        params={
            "shape": shape,
            "tile": tile,
            "warp_tile": warp,
            "mma_type": "wgmma",
            "mem_levels": mem_levels,
            "row_panel": 1,
            "batch": batch,
            "dtype": compute_dtype,
            "accumulator_dtype": accumulation_dtype,
        },
    )
    grid = ((m + tm - 1) // tm, (n + tn - 1) // tn, 1)
    with kernel.launch(
        work_grid=grid,
        physical_grid=grid,
        threads=int(tm / wm * tn / wn * 32),
        residency="auto",
        scheduler="static",
    ) as launch:
        with launch.periodic(
            "ko", iterations=(k + tk - 1) // tk, stages=stages
        ) as ko:
            a_shared = ko.buffer(
                "A_shared", scope="smem", shape=(tm, tk),
                dtype=a_storage_dtype, slots=stages,
                execution_scope="cta",
            )
            b_shared = ko.buffer(
                "B_shared", scope="smem", shape=(tk, tn),
                dtype=b_storage_dtype, slots=stages,
                execution_scope="cta",
            )
            accumulator = ko.buffer(
                "C_accumulator", scope="fragment", shape=(tm, tn),
                dtype=accumulator_buffer_dtype, slots=1,
                execution_scope="warpgroup",
            )
            with ko.actor("producer") as producer:
                if object_primitives:
                    load_a = producer.load(
                        "load_A", bytes=tm * tk * dtype_bytes[a_storage_dtype],
                        source="ddr",
                        destination="smem", engine="tma",
                        attrs={"dtype": a_work_dtype},
                        timing=Timing(1e-9, (ResourceTiming("tma", 1e-9),)),
                        writes=(a_shared,),
                    )
                    load_b = producer.load(
                        "load_B", bytes=tk * tn * dtype_bytes[b_storage_dtype],
                        source="ddr",
                        destination="smem", engine="tma",
                        attrs={"dtype": b_work_dtype},
                        timing=Timing(1e-9, (ResourceTiming("tma", 1e-9),)),
                        writes=(b_shared,),
                    )
                else:
                    load_a = producer.phase(
                        "load_A",
                        work=Work.copy(
                            tm * tk * dtype_bytes[a_storage_dtype],
                            source="gmem",
                            target="smem",
                            dtype=a_work_dtype,
                        ),
                        timing=Timing(1e-9, (ResourceTiming("tma", 1e-9),)),
                        writes=(a_shared,),
                    )
                    load_b = producer.phase(
                        "load_B",
                        work=Work.copy(
                            tk * tn * dtype_bytes[b_storage_dtype],
                            source="gmem",
                            target="smem",
                            dtype=b_work_dtype,
                        ),
                        timing=Timing(1e-9, (ResourceTiming("tma", 1e-9),)),
                        writes=(b_shared,),
                    )
                producer.sequence(load_a.at(0), load_b.at(0), resource="tma")
            with ko.actor("tensor") as tensor:
                mma_attrs = (
                    {
                        "shape": tile,
                        "input_dtype": compute_dtype,
                        "accumulator_dtype": accumulation_dtype,
                    }
                    if include_mma_attrs
                    else {}
                )
                if object_primitives:
                    mma = tensor.compute(
                        "mma", flops=2 * tm * tn * tk, engine="tensor",
                        op="mma", attrs=mma_attrs,
                        timing=Timing(1e-9, (ResourceTiming("tensor", 1e-9),)),
                        reads=(a_shared, b_shared), writes=(accumulator,),
                    )
                else:
                    mma = tensor.phase(
                        "mma",
                        work=Work.mma(2 * tm * tn * tk, **mma_attrs),
                        timing=Timing(1e-9, (ResourceTiming("tensor", 1e-9),)),
                        reads=(a_shared, b_shared),
                        writes=(accumulator,),
                    )
            ko.pipeline_buffer(
                a_shared, acquire=load_a.start, release=mma.done, capacity=stages
            )
            ko.pipeline_buffer(
                b_shared, acquire=load_b.start, release=mma.done, capacity=stages
            )
            state = ko.state("accumulator", storage=accumulator)
            ko.carry(state, source=mma.done, target=mma.start, distance=1)
    return kernel.build()


def _op(
    shape,
    *,
    a_storage_dtype="bf16",
    b_storage_dtype="bf16",
    result_dtype="fp32",
    compute_dtype="bf16",
    accumulation_dtype="fp32",
    batch=1,
):
    return make_gemm(
        m=shape[0],
        n=shape[1],
        k=shape[2],
        a_storage_dtype=a_storage_dtype,
        b_storage_dtype=b_storage_dtype,
        result_storage_dtype=result_dtype,
        compute_dtype=compute_dtype,
        accumulation_dtype=accumulation_dtype,
        batch=batch,
    )


@pytest.mark.parametrize("_name,shape,tile,warp,occupancy,expected_ii_ns", CASES)
def test_h100_shadow_cases_have_exact_resource_ii_and_occupancy(
    _name, shape, tile, warp, occupancy, expected_ii_ns
):
    result = model_general_gemm(
        _op(shape),
        _program(shape, tile, warp),
        H100_SXM().set_to_microbench(),
        _options(),
    )
    assert result.breakdown.cache.reduction_fidelity == "stable_shadow_cohort"
    assert result.breakdown.cache.reuse_unit_bytes == 8 * 1024
    assert result.breakdown.footprint.resident_ctas_per_sm == occupancy
    assert result.ii.ii * 1e9 == pytest.approx(expected_ii_ns, abs=1e-11)
    assert result.breakdown.service.tensor_s * 1e9 == pytest.approx(
        expected_ii_ns, abs=1e-11
    )
    assert result.breakdown.service.memory_s < result.ii.ii
    assert result.legacy_calibrated is not None
    assert result.kernel_body_s > 0.0
    assert dict(result.legacy_calibrated.provenance)[
        "monolithic_entry_point_called"
    ] is False


@pytest.mark.parametrize("name,shape,tile,warp,_occupancy,_expected_ii_ns", CASES)
def test_legacy_rng_profile_is_field_parity_without_calling_monolith(
    name, shape, tile, warp, _occupancy, _expected_ii_ns
):
    seed = 100 + ord(name)
    arch = H100_SXM().set_to_microbench()
    result = model_general_gemm(
        _op(shape),
        _program(shape, tile, warp),
        arch,
        _options(cache_policy="legacy_rng", cache_seed=seed),
    ).legacy_calibrated
    assert result is not None

    np.random.seed(seed)
    old = calculate_matmul_pipeline_wave(
        shape,
        tile,
        warp,
        3,
        arch,
        MEM_LEVELS,
        row_panel=1,
        mma_type="wgmma",
    )
    assert result.per_tile_latency_s == old.per_tile_latency
    assert result.kernel_body_s == old.total_latency
    assert result.ddr_util == old.ddr_util
    assert result.compute_util == old.compute_util
    assert result.l2_hit_rate == old.l2_hit_rate
    assert result.resident_ctas_per_sm == old.tiles_per_sm
    assert result.waves == old.waves
    assert result.prologue_s == old.pipeline_detail.prologue_time
    assert result.steady_time_per_iter_s == old.pipeline_detail.steady_time_per_iter
    assert result.epilogue_s == old.pipeline_detail.epilogue_time
    assert result.mem_time_per_iter_s == old.pipeline_detail.mem_time_per_iter
    assert result.compute_time_per_iter_s == old.pipeline_detail.compute_time_per_iter


def test_native_periodic_reports_constructive_steady_and_natural_boundary_only():
    shape, tile, warp = CASES[1][1:4]
    result = model_general_gemm(
        _op(shape),
        _program(shape, tile, warp),
        H100_SXM().set_to_microbench(),
        _options(profile="native_periodic"),
    )
    assert result.ii.is_constructive
    assert result.ii.ii * 1e9 == pytest.approx(397.7351526793783, abs=1e-11)
    assert result.envelope is not None
    assert result.natural_boundary is not None
    assert result.natural_boundary.iterations == 64
    assert result.natural_boundary.occurrence_count == 3 * 64
    assert result.natural_boundary.latency_s > 63 * result.ii.ii
    assert result.kernel_body_s is None
    assert result.legacy_calibrated is None
    assert dict(result.provenance)["full_grid_latency"] == (
        "not_claimed_from_single_cta_periodic_witness"
    )


def test_oracle_overrides_example_static_timing_without_mutating_source_ir():
    shape, tile, warp = CASES[0][1:4]
    kernel = _program(shape, tile, warp)
    original = {
        phase.name: phase.timing for phase in kernel.periodic_loop("ko").phases
    }
    oracle = BoundGemmCostOracle(
        _op(shape), kernel, H100_SXM().set_to_microbench(), _options()
    )
    dag = oracle.lower_periodic()
    assert {phase.name for phase in dag.phases} == {"load_A", "load_B", "mma"}
    assert next(phase for phase in dag.phases if phase.name == "mma").latency > 1e-9
    assert {
        phase.name: phase.timing for phase in kernel.periodic_loop("ko").phases
    } == original


def test_bf16_tensor_capacity_is_fail_closed_without_an_explicit_alias():
    shape, tile, warp = CASES[0][1:4]
    with pytest.raises(UnsupportedThroughputError, match="dtype=bf16"):
        model_general_gemm(
            _op(shape),
            _program(shape, tile, warp),
            H100_SXM().set_to_microbench(),
            GeneralGemmOptions(),
        )


def test_explicit_bf16_fp16_capacity_alias_is_recorded_in_provenance():
    shape, tile, warp = CASES[0][1:4]
    result = model_general_gemm(
        _op(shape),
        _program(shape, tile, warp),
        H100_SXM().set_to_microbench(),
        _options(),
    )
    resolution = result.breakdown.tensor_throughput
    provenance = dict(result.breakdown.provenance)
    assert resolution.request.dtype.name == "bf16"
    assert resolution.fallback == "dtype_alias:bf16->fp16"
    assert resolution.source == "arch.fp16_tensor_flops"
    assert provenance["tensor_throughput_fallback"] == (
        "dtype_alias:bf16->fp16"
    )
    assert provenance["tensor_throughput_dtype_aliases"] == (
        ("bf16", "fp16"),
    )


def test_register_footprint_uses_accumulation_not_result_storage_dtype():
    shape, tile, warp = CASES[0][1:4]
    result = model_general_gemm(
        _op(shape, result_dtype="bf16", accumulation_dtype="fp32"),
        _program(
            shape,
            tile,
            warp,
            result_dtype="bf16",
            accumulation_dtype="fp32",
        ),
        H100_SXM().set_to_microbench(),
        _options(),
    )
    footprint = result.breakdown.footprint
    assert footprint.legacy_registers_per_thread == 282.0
    assert footprint.resident_ctas_per_sm == 1


@pytest.mark.parametrize(
    "program_kwargs,error",
    (
        ({"include_mma_attrs": False}, "explicitly provide"),
        (
            {"accumulation_dtype": "fp32", "accumulator_buffer_dtype": "bf16"},
            "accumulator buffer dtype",
        ),
        ({"compute_dtype": "fp16"}, "compute_dtype"),
    ),
)
def test_schedule_dtype_mutations_fail_closed(program_kwargs, error):
    shape, tile, warp = CASES[0][1:4]
    with pytest.raises(ModelingValidationError, match=error):
        BoundGemmCostOracle(
            _op(shape),
            _program(shape, tile, warp, **program_kwargs),
            H100_SXM().set_to_microbench(),
            _options(),
        )


def test_input_load_buffer_and_work_dtype_mutations_fail_closed():
    shape, tile, warp = CASES[0][1:4]
    arch = H100_SXM().set_to_microbench()

    # FP16 and BF16 have the same byte width.  The validation must compare
    # dtype identity rather than accepting equal Work.bytes/mem-level widths.
    with pytest.raises(ModelingValidationError, match="A load buffer dtype"):
        BoundGemmCostOracle(
            _op(shape, a_storage_dtype="fp16"),
            _program(shape, tile, warp),
            arch,
            _options(),
        )
    with pytest.raises(ModelingValidationError, match="A load Work dtype"):
        BoundGemmCostOracle(
            _op(shape),
            _program(shape, tile, warp, a_work_dtype="fp16"),
            arch,
            _options(),
        )


def test_storage_to_compute_dtype_change_requires_an_explicit_cast_phase():
    shape, tile, warp = CASES[0][1:4]
    with pytest.raises(ModelingValidationError, match="explicit phase"):
        BoundGemmCostOracle(
            _op(shape, a_storage_dtype="fp16", compute_dtype="bf16"),
            _program(shape, tile, warp, a_storage_dtype="fp16"),
            H100_SXM().set_to_microbench(),
            _options(),
        )


def test_object_load_compute_primitives_bind_to_the_same_gemm_dag():
    shape, tile, warp = CASES[0][1:4]
    result = model_general_gemm(
        _op(shape),
        _program(shape, tile, warp, object_primitives=True),
        H100_SXM().set_to_microbench(),
        _options(),
    )
    assert {phase.name for phase in result.dag.phases} == {
        "load_A",
        "load_B",
        "mma",
    }
    assert result.breakdown.schedule.copy_resource == "tma"
    assert result.breakdown.schedule.tensor_resource == "tensor"


def test_stable_cache_whole_grid_traffic_obeys_k_and_batch_conservation():
    shape = (256, 256, 128)
    tile = (128, 128, 64)
    warp = (64, 64, 64)
    batch = 3
    mem_levels = {
        "in1": [1, 1, 1, 2],
        "in2": [1, 1, 1, 2],
        "out1": [1, 0, 1, 2],
    }
    op = _op(shape, result_dtype="bf16", batch=batch)
    kernel = _program(
        shape,
        tile,
        warp,
        result_dtype="bf16",
        mem_levels=mem_levels,
        batch=batch,
    )
    options = _options(cache_sample_budget=None)
    problem = build_general_gemm_cache_problem(op, kernel, H100_SXM(), options)
    raw = evaluate_general_gemm_cache(problem)
    modeled = model_general_gemm(
        op, kernel, H100_SXM().set_to_microbench(), options
    ).breakdown.cache

    iterations = shape[2] // tile[2]
    assert modeled.representative_grid_ddr_read_bytes == pytest.approx(
        raw.traffic.ddr_read_bytes
    )
    assert modeled.whole_grid_ddr_read_bytes == pytest.approx(
        raw.traffic.ddr_read_bytes * iterations * batch
    )
    assert modeled.whole_grid_ddr_write_bytes == pytest.approx(
        raw.traffic.ddr_write_bytes * batch
    )
    # The write-through epilogue is one result materialization per batch; it
    # must not be multiplied by the K-loop trip count.
    assert modeled.whole_grid_ddr_write_bytes == pytest.approx(
        shape[0] * shape[1] * 2 * batch
    )


def test_progress_jitter_is_disabled_by_default_and_explicit_in_provenance():
    shape, tile, warp = CASES[0][1:4]
    op = _op(shape)
    kernel = _program(shape, tile, warp)
    arch = H100_SXM().set_to_microbench()
    disabled = _options()
    enabled = _options(
        progress_jitter=ProgressJitterConfig(
            max_k_ahead=2.0,
            capacity_cap_fraction=0.25,
            seed=17,
        )
    )
    disabled_problem = build_general_gemm_cache_problem(
        op, kernel, arch, disabled
    )
    enabled_problem = build_general_gemm_cache_problem(op, kernel, arch, enabled)
    assert not disabled_problem.reduction.progress_jitter.enabled
    assert enabled_problem.reduction.progress_jitter.enabled
    assert disabled_problem.reuse_digest != enabled_problem.reuse_digest

    result = model_general_gemm(op, kernel, arch, enabled)
    provenance = dict(result.breakdown.provenance)
    assert provenance["cache_progress_jitter_enabled"] is True
    assert provenance["cache_progress_jitter_max_k_ahead"] == 2.0
    assert provenance["cache_progress_jitter_capacity_fraction"] == 0.25
    assert provenance["cache_progress_jitter_seed"] == 17


def test_default_fast_cache_budget_and_reusable_histogram_capacity_sweep():
    shape, tile, warp = CASES[0][1:4]
    op = _op(shape)
    kernel = _program(shape, tile, warp)
    arch = H100_SXM().set_to_microbench()
    options = _options()
    assert options.cache_sample_budget == 512

    problem = build_general_gemm_cache_problem(op, kernel, arch, options)
    histogram = build_general_gemm_cache_histogram(problem)
    first = evaluate_general_gemm_cache(problem, histogram)
    smaller_l2 = replace(
        problem,
        l2=replace(problem.l2, capacity_bytes=problem.l2.capacity_bytes / 2.0),
    )
    second = evaluate_general_gemm_cache(smaller_l2, histogram)
    assert problem.reuse_digest == smaller_l2.reuse_digest
    assert problem.digest != smaller_l2.digest
    assert first.histogram.reuse_digest == second.histogram.reuse_digest
    assert first.problem_digest != second.problem_digest

    reused = model_general_gemm(
        op,
        kernel,
        arch,
        replace(options, cache_histogram=histogram),
    )
    cache = reused.breakdown.cache
    assert cache.histogram_reused is True
    assert cache.problem_digest == first.problem_digest
    assert cache.sampling_mode == "systematic_budget_512"
    assert dict(reused.breakdown.provenance)["cache_sample_budget"] == 512
