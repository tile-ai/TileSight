"""Tests for the opt-in recursive native region/wave executor."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from tilesight.modeling import (
    Kernel,
    LoopRegion,
    NativeModelError,
    NativeModelOptions,
    PeriodicAxisIR,
    Phase,
    PhaseRegion,
    ResourceTiming,
    SequenceRegion,
    Timing,
    Work,
    evaluate_region,
    model_native,
    ns,
)
from tilesight.modeling.tests.fixtures.legacy_fusion_gemm_bias import (
    build_fusion_gemm_bias,
    build_native_fused_region,
    model_native_fused_static,
)


def _standalone_phase(name, latency_ns, resource=None):
    resources = ()
    if resource is not None:
        resources = (ResourceTiming(resource, latency_ns * ns),)
    return Phase(
        name=name,
        actor="serial",
        owner="native/regions",
        work=Work.opaque(name),
        timing=Timing(latency_ns * ns, resources),
    )


def _kernel_and_inner_phases():
    kernel = Kernel("native_nested")
    with kernel.launch(
        work_grid=(8,),
        physical_grid=(8,),
        threads=128,
        resident_ctas=2,
    ) as launch:
        with launch.periodic("inner_pipeline", iterations=64, stages=2) as loop:
            tile = loop.buffer(
                "tile",
                scope="smem",
                shape=(32,),
                dtype="fp32",
                slots=2,
            )
            with loop.actor("producer", execution_scope="warpgroup") as producer:
                load = producer.load(
                    "load",
                    bytes=128,
                    source="ddr",
                    destination="smem",
                    engine="tma",
                    timing=Timing(
                        10 * ns,
                        (ResourceTiming("tma", 4 * ns),),
                    ),
                    writes=(tile,),
                )
            with loop.actor("tensor", execution_scope="warpgroup") as tensor:
                mma = tensor.compute(
                    "mma",
                    flops=4096,
                    engine="tensor",
                    op="mma",
                    timing=Timing(
                        6 * ns,
                        (ResourceTiming("tensor", 6 * ns),),
                    ),
                    reads=(tile,),
                )
            loop.pipeline_buffer(tile, acquire=load.start, release=mma.done)
    return kernel.build(), load, mma


def _inner_loop(load, mma, trip_count=64, policy="auto"):
    return LoopRegion(
        "inner",
        trip_count=trip_count,
        prologue=PhaseRegion(_standalone_phase("inner_prologue", 2)),
        body=SequenceRegion(
            "inner_body",
            (PhaseRegion(load), PhaseRegion(mma)),
        ),
        epilogue=PhaseRegion(_standalone_phase("inner_epilogue", 3)),
        periodic_axis=PeriodicAxisIR(
            loop_name="inner_pipeline",
            ii_mode="periodic_best",
            summary_policy=policy,
        ),
    )


@pytest.mark.parametrize(
    "trip_count,expected_body_repetitions,strategy",
    [(0, 0, "zero_trip"), (1, 1, "inline_serial"), (3, 3, "inline_serial")],
)
def test_zero_one_and_short_serial_loops_inline_without_double_count(
    trip_count, expected_body_repetitions, strategy
):
    kernel, load, mma = _kernel_and_inner_phases()
    region = replace(
        _inner_loop(load, mma, trip_count=trip_count),
        periodic_axis=None,
    )
    result = evaluate_region(
        kernel,
        region,
        NativeModelOptions(ii_mode="periodic_best", unroll_threshold=4),
    )
    expected = (2 + expected_body_repetitions * (10 + 6) + 3) * ns
    assert result.total_s == pytest.approx(expected, rel=0, abs=1e-20)
    assert result.first_s + result.steady_s + result.drain_s == pytest.approx(
        result.total_s, rel=0, abs=1e-20
    )
    assert result.strategy == strategy
    if trip_count == 0:
        assert not any(item.phase in ("load", "mma") for item in result.resource_trace)


@pytest.mark.parametrize("trip_count", [1, 3])
def test_short_periodic_loop_uses_labelled_boundary_approx_not_serial_fallback(
    trip_count,
):
    kernel, load, mma = _kernel_and_inner_phases()
    region = _inner_loop(load, mma, trip_count=trip_count)
    result = evaluate_region(
        kernel,
        region,
        NativeModelOptions(ii_mode="periodic_best", unroll_threshold=4),
    )
    assert result.strategy == "periodic_macro_boundary_approx"
    assert result.estimate_scope.endswith("_boundary_approx")
    assert result.children[0].total_s == 2 * ns
    assert result.children[2].total_s == 3 * ns
    assert result.first_s + result.steady_s + result.drain_s == pytest.approx(
        result.total_s
    )
    assert any("finite periodic unroll is not implemented" in x for x in result.diagnostics)


def test_periodic_inline_policy_rejects_until_finite_scheduler_exists():
    kernel, load, mma = _kernel_and_inner_phases()
    region = _inner_loop(load, mma, trip_count=3, policy="inline")
    with pytest.raises(NativeModelError, match="finite periodic scheduler"):
        evaluate_region(kernel, region, NativeModelOptions())


def test_long_periodic_loop_uses_constructive_macro_and_compressed_trace():
    kernel, load, mma = _kernel_and_inner_phases()
    region = _inner_loop(load, mma, trip_count=64)
    hash(region)
    result = evaluate_region(
        kernel,
        region,
        NativeModelOptions(ii_mode="periodic_best", unroll_threshold=4),
    )
    assert result.strategy == "periodic_macro"
    hash(result)
    assert result.estimate_scope == "constructive_periodic_schedule"
    assert result.selected_ii_s is not None
    assert result.first_s + result.steady_s + result.drain_s == pytest.approx(
        result.total_s
    )
    periodic_entries = [item for item in result.resource_trace if item.phase in ("load", "mma")]
    assert periodic_entries
    assert all(item.repeats[-1].count == 64 for item in periodic_entries)
    assert all(item.repeats[-1].stride_s == result.selected_ii_s for item in periodic_entries)
    assert dict(result.resource_service_s)["tensor"] == pytest.approx(64 * 6 * ns)


def test_resource_ii_macro_is_explicitly_a_bound_without_witness():
    kernel, load, mma = _kernel_and_inner_phases()
    region = _inner_loop(load, mma, trip_count=64)
    result = evaluate_region(
        kernel,
        region,
        NativeModelOptions(ii_mode="resource_ii", unroll_threshold=4),
    )
    # The axis-level explicit periodic_best overrides the global default.
    assert result.estimate_scope == "constructive_periodic_schedule"

    inherited = LoopRegion(
        "inherited",
        trip_count=64,
        body=region.body,
        periodic_axis=PeriodicAxisIR(
            loop_name="inner_pipeline",
            ii_mode="inherit",
            summary_policy="macro",
        ),
    )
    lower_bound = evaluate_region(
        kernel,
        inherited,
        NativeModelOptions(ii_mode="resource_ii"),
    )
    assert lower_bound.estimate_scope == (
        "necessary_steady_bound_with_serial_fill_drain"
    )
    assert lower_bound.ii_scope == "resource_ii_lower_bound"
    assert all(
        item.offset_s is None for item in lower_bound.resource_trace
    )
    arch = SimpleNamespace(
        sm_count=2,
        configurable_smem_capacity=128 * 1024,
        tmem_capacity_per_sm=0,
    )
    with pytest.raises(NativeModelError, match="requires constructive witnesses"):
        model_native(
            kernel,
            inherited,
            arch,
            NativeModelOptions(
                ii_mode="resource_ii",
                resident_ctas_per_sm=1,
            ),
        )


def test_serial_outer_loop_can_reuse_inner_macro_without_double_count():
    kernel, load, mma = _kernel_and_inner_phases()
    inner = _inner_loop(load, mma, trip_count=64)
    outer = LoopRegion(
        "outer",
        trip_count=10,
        prologue=PhaseRegion(_standalone_phase("outer_prologue", 1)),
        body=inner,
        epilogue=PhaseRegion(_standalone_phase("outer_epilogue", 1)),
    )
    options = NativeModelOptions(ii_mode="periodic_best", unroll_threshold=4)
    inner_result = evaluate_region(kernel, inner, options)
    result = evaluate_region(kernel, outer, options)
    assert result.total_s == pytest.approx(2 * ns + 10 * inner_result.total_s)
    assert result.children[1].strategy == "periodic_macro"
    assert result.strategy == "serial_repeat_macro"


def test_periodic_outer_loop_rejects_an_unflattened_inner_loop():
    kernel, load, mma = _kernel_and_inner_phases()
    inner = _inner_loop(load, mma, trip_count=64)
    dag = kernel.lower_periodic("inner_pipeline")
    outer = LoopRegion(
        "bad_outer",
        trip_count=64,
        body=inner,
        periodic_axis=PeriodicAxisIR(
            dag=dag,
            summary_policy="macro",
        ),
    )
    with pytest.raises(NativeModelError, match="contains an inner loop"):
        evaluate_region(kernel, outer, NativeModelOptions(ii_mode="periodic_best"))


def test_static_liveness_fallback_must_be_explicit_and_is_in_provenance():
    kernel, load, mma = _kernel_and_inner_phases()
    dag = kernel.lower_periodic("inner_pipeline")
    region = LoopRegion(
        "explicit_dag_without_liveness_binding",
        trip_count=64,
        body=SequenceRegion("body", (PhaseRegion(load), PhaseRegion(mma))),
        periodic_axis=PeriodicAxisIR(
            dag=dag,
            liveness_loop_name="inner_pipeline",
            ii_mode="resource_ii",
        ),
    )
    arch = SimpleNamespace(
        sm_count=2,
        configurable_smem_capacity=128 * 1024,
        tmem_capacity_per_sm=0,
    )
    with pytest.raises(NativeModelError, match="requires constructive witnesses"):
        model_native(
            kernel,
            region,
            arch,
            NativeModelOptions(resident_ctas_per_sm=1),
        )
    result = model_native(
        kernel,
        region,
        arch,
        NativeModelOptions(
            resident_ctas_per_sm=1,
            liveness_policy="report_static",
        ),
    )
    assert dict(result.provenance)["liveness_static_fallback_regions"] == (
        region.name,
    )


def test_native_full_executor_applies_grid_waves_and_launch_exactly_once():
    kernel, load, mma = _kernel_and_inner_phases()
    region = _inner_loop(load, mma, trip_count=64)
    arch = SimpleNamespace(
        sm_count=2,
        configurable_smem_capacity=128 * 1024,
        tmem_capacity_per_sm=0,
    )
    options = NativeModelOptions(
        ii_mode="periodic_best",
        multi_cta_policy="resource_bound",
        kernel_launch_s=2e-6,
        host_dispatch_s=1e-6,
    )
    result = model_native(kernel, region, arch, options)
    hash(result)
    assert result.resident_ctas_per_sm == 2
    assert result.effective_resident_ctas_per_sm == 2
    assert result.grid.active_work_units == 4
    assert result.grid.active_physical_ctas == 4
    assert result.grid.waves == 2
    expected_group = max(
        result.per_work_unit_s,
        2 * max(dict(result.region.resource_service_s).values()),
    )
    assert result.resident_group_s == pytest.approx(expected_group)
    assert result.kernel_body_s == pytest.approx(2 * expected_group)
    assert result.total_s == pytest.approx(result.kernel_body_s + 3e-6)
    assert result.spatial_scope == "resident_group_resource_bound_nonconstructive"
    assert dict(result.provenance)["launch_accounting"] == "exactly_once"
    assert result.liveness.witness_ii_s == result.region.selected_ii_s
    assert dict(result.provenance)["liveness_witness_loops"] == (
        "inner_pipeline",
    )
    assert dict(result.provenance)["liveness_static_fallback_regions"] == ()
    decisions = dict(result.provenance)["region_decisions"]
    assert any(dict(item)["strategy"] == "periodic_macro" for item in decisions)
    coverage = tuple(
        dict(item) for item in dict(result.provenance)["region_loop_coverage"]
    )
    assert coverage == (
        {
            "binding": "inner",
            "kernel_iterations": 64,
            "loop": "inner_pipeline",
            "phase_count": 2,
            "region_trip_count": 64,
        },
    )
    assert set(dict(result.provenance)["region_standalone_phases"]) == {
        "native/regions/inner_epilogue",
        "native/regions/inner_prologue",
    }


def _coverage_arch():
    return SimpleNamespace(
        sm_count=2,
        configurable_smem_capacity=128 * 1024,
        tmem_capacity_per_sm=0,
    )


def _coverage_options():
    return NativeModelOptions(
        ii_mode="periodic_best",
        resident_ctas_per_sm=1,
    )


def test_full_native_rejects_partial_phase_region_that_bypasses_loop_witness():
    kernel, load, _mma = _kernel_and_inner_phases()
    with pytest.raises(NativeModelError, match="cover every selected-launch phase"):
        model_native(
            kernel,
            PhaseRegion(load),
            _coverage_arch(),
            _coverage_options(),
        )


def test_full_native_rejects_duplicate_and_mutated_kernel_owned_phases():
    kernel, load, mma = _kernel_and_inner_phases()
    duplicate = SequenceRegion(
        "duplicate",
        (PhaseRegion(load), PhaseRegion(load), PhaseRegion(mma)),
    )
    with pytest.raises(NativeModelError, match="duplicate_or_repeated"):
        model_native(kernel, duplicate, _coverage_arch(), _coverage_options())

    mutated_load = replace(load, timing=Timing(11 * ns))
    mutated = SequenceRegion(
        "mutated",
        (PhaseRegion(mutated_load), PhaseRegion(mma)),
    )
    with pytest.raises(NativeModelError, match="mutates"):
        model_native(kernel, mutated, _coverage_arch(), _coverage_options())

    invented = SequenceRegion(
        "invented",
        (PhaseRegion(load), PhaseRegion(mma), PhaseRegion(replace(load, name="fake"))),
    )
    with pytest.raises(NativeModelError, match="invents unknown phase"):
        model_native(kernel, invented, _coverage_arch(), _coverage_options())


def test_full_native_rejects_unbound_or_trip_count_drifted_kernel_loop():
    kernel, load, mma = _kernel_and_inner_phases()
    unbound = SequenceRegion("unbound", (PhaseRegion(load), PhaseRegion(mma)))
    with pytest.raises(NativeModelError, match="no PeriodicAxisIR binding"):
        model_native(kernel, unbound, _coverage_arch(), _coverage_options())

    drifted = _inner_loop(load, mma, trip_count=1)
    with pytest.raises(NativeModelError, match="trip_count=1 disagrees"):
        model_native(kernel, drifted, _coverage_arch(), _coverage_options())


def test_native_multi_cta_is_rejected_without_an_explicit_bound_policy():
    kernel, load, mma = _kernel_and_inner_phases()
    region = _inner_loop(load, mma, trip_count=64)
    arch = SimpleNamespace(
        sm_count=2,
        configurable_smem_capacity=128 * 1024,
        tmem_capacity_per_sm=0,
    )
    with pytest.raises(NativeModelError, match="cannot constructively arbitrate"):
        model_native(kernel, region, arch, NativeModelOptions(ii_mode="periodic_best"))


def test_small_grid_uses_effective_not_configured_residency():
    kernel, load, mma = _kernel_and_inner_phases()
    launch = replace(kernel.launches[0], work_grid=(2,), physical_grid=(2,))
    kernel = replace(kernel, launches=(launch,))
    region = _inner_loop(load, mma, trip_count=64)
    arch = SimpleNamespace(
        sm_count=4,
        configurable_smem_capacity=128 * 1024,
        tmem_capacity_per_sm=0,
    )
    # LaunchIR still permits two CTAs/SM, but only one CTA lands on each of the
    # two active SMs, so the default constructive single-CTA path is valid.
    result = model_native(kernel, region, arch, NativeModelOptions())
    assert result.resident_ctas_per_sm == 2
    assert result.effective_resident_ctas_per_sm == 1
    assert result.resident_group_s == result.per_work_unit_s
    assert result.grid.waves == 1


def test_native_tail_wave_uses_only_tail_residency_resource_bound():
    kernel, load, mma = _kernel_and_inner_phases()
    launch = replace(kernel.launches[0], work_grid=(9,), physical_grid=(9,))
    kernel = replace(kernel, launches=(launch,))
    region = _inner_loop(load, mma, trip_count=64)
    arch = SimpleNamespace(
        sm_count=2,
        configurable_smem_capacity=128 * 1024,
        tmem_capacity_per_sm=0,
    )
    result = model_native(
        kernel,
        region,
        arch,
        NativeModelOptions(
            ii_mode="periodic_best",
            multi_cta_policy="resource_bound",
        ),
    )
    assert result.grid.full_waves == 2
    assert result.grid.tail_work_units == 1
    assert result.grid.waves == 3
    assert result.tail_wave_s == result.per_work_unit_s
    assert result.kernel_body_s == pytest.approx(
        2 * result.resident_group_s + result.tail_wave_s
    )


def test_fused_native_model_requires_explicitly_fused_cost_source():
    kernel = build_fusion_gemm_bias()
    phase = kernel.launches[0].periodic_loops[0].phases[0]
    region = PhaseRegion(phase)
    with pytest.raises(NativeModelError, match="explicit_fused_static_timing"):
        evaluate_region(kernel, region, NativeModelOptions())

    result = evaluate_region(
        kernel,
        region,
        NativeModelOptions(cost_source="explicit_fused_static_timing"),
    )
    assert result.total_s == phase.timing.latency


def test_fused_gemm_bias_runs_full_recursive_static_bound_not_single_phase():
    kernel = build_fusion_gemm_bias()
    region = build_native_fused_region(kernel)
    result = model_native_fused_static(kernel)
    assert result.region.name == region.name
    assert result.region.strategy == "periodic_macro"
    assert result.region.trip_count == 64
    assert result.region.selected_ii_s is not None
    # bias (8 ns) and store (40 ns) occur once in the loop epilogue.
    assert result.region.children[2].total_s == pytest.approx(48 * ns)
    assert result.region.total_s > result.region.children[2].total_s
    assert result.fusion.status.value == "proven_feasible"
    assert result.liveness.witness_ii_s == result.region.selected_ii_s
    assert result.launch_s == 2e-6
    assert result.total_s == result.kernel_body_s + result.launch_s
    assert dict(result.provenance)["cost_source"] == "explicit_fused_static_timing"
    assert dict(result.provenance)["fusion_traffic_binding"] == (
        "user_asserted_consumed_by_static_timing"
    )
    assert dict(result.provenance)["cache_traffic_binding"] == (
        "user_asserted_consumed_by_static_timing"
    )
    fused_coverage = {
        dict(item)["loop"]: dict(item)
        for item in dict(result.provenance)["region_loop_coverage"]
    }
    assert fused_coverage["ko"]["region_trip_count"] == 64
    assert fused_coverage["epilogue"]["binding"] == "one_shot_structural"


def test_fusion_aware_oracle_must_declare_and_consume_fusion_traffic():
    kernel = build_fusion_gemm_bias()
    phase = kernel.launches[0].periodic_loops[0].phases[0]
    region = PhaseRegion(phase)

    class BadOracle:
        consumes_fusion_traffic = False
        consumes_cache_traffic = False

        def resolve_fused(self, phase, kernel, fusion):
            return Timing(123 * ns)

    with pytest.raises(NativeModelError, match="consumes_fusion_traffic=True"):
        evaluate_region(
            kernel,
            region,
            NativeModelOptions(cost_source="fusion_aware_oracle"),
            oracle=BadOracle(),
        )

    class GoodOracle(BadOracle):
        consumes_fusion_traffic = True
        consumes_cache_traffic = True

    result = evaluate_region(
        kernel,
        region,
        NativeModelOptions(cost_source="fusion_aware_oracle"),
        oracle=GoodOracle(),
    )
    assert result.total_s == 123 * ns
