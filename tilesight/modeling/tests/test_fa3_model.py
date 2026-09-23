"""Focused structural tests for the source-scoped FA3 model."""

import pytest
from tilesight.modeling._attention import fa3_periodic

from tilesight.arch.h200_sxm import H200_SXM

from tilesight.modeling._attention.fa3_model import (
    KERNEL_LAUNCH_OVERHEAD_US,
    _build_iteration_body,
    _kv_iters_by_qblock,
    _should_pack_gqa,
    _tile_config,
    model_fa3_us,
)
from tilesight.modeling._attention.fa3_periodic import (
    analyze_fa3_periodic,
    build_fa3_periodic_dag,
    fa3_periodic_best,
    fa3_periodic_worst,
)


def test_audited_hopper_tiles():
    assert _tile_config(64, causal=False) == (192, 192, False, True)
    assert _tile_config(64, causal=True) == (192, 128, True, True)
    assert _tile_config(128, causal=False) == (128, 176, True, True)
    assert _tile_config(128, causal=True) == (128, 128, True, True)
    with pytest.raises(NotImplementedError):
        _tile_config(256, causal=False)


def test_causal_kv_triangle_matches_blockmn_bounds():
    assert _kv_iters_by_qblock(512, 128, 128, 1, False, True) == [1, 2, 3, 4]
    assert _kv_iters_by_qblock(512, 128, 128, 1, False, False) == [4, 4, 4, 4]


def test_packgqa_uses_flattened_query_coordinate():
    assert _should_pack_gqa(64, 4, 128)
    assert _kv_iters_by_qblock(64, 128, 128, 4, True, True) == [1, 1]


def test_gqa_grid_matches_source_pack_heuristic():
    arch = H200_SXM().set_to_microbench()
    _, measured_shape = model_fa3_us(1, 32, 2048, 128, kv_h=8, arch=arch)
    assert not measured_shape["pack_gqa"]
    assert measured_shape["grid_work_tiles"] == 32 * 16

    _, short_shape = model_fa3_us(1, 32, 64, 128, kv_h=8, arch=arch)
    assert short_shape["pack_gqa"]
    assert short_shape["grid_work_tiles"] == 8 * 2


def test_launch_and_host_dispatch_are_distinct_and_added_once():
    arch = H200_SXM().set_to_microbench()
    total_us, breakdown = model_fa3_us(
        1, 32, 1024, 128, arch=arch, host_dispatch_overhead_us=7.0
    )
    assert breakdown["kernel_launch_us"] == KERNEL_LAUNCH_OVERHEAD_US == 2.0
    assert breakdown["kernel_us"] == pytest.approx(
        breakdown["gpu_body_us"] + KERNEL_LAUNCH_OVERHEAD_US
    )
    assert total_us == pytest.approx(breakdown["kernel_us"] + 7.0)


def test_periodic_hook_cannot_beat_resource_lower_bound():
    arch = H200_SXM().set_to_microbench()

    def infeasible_lower_ii(**kwargs):
        return kwargs["resource_ii_s"] * 0.5

    with pytest.raises(ValueError, match="below the resource lower bound"):
        model_fa3_us(
            1,
            32,
            1024,
            128,
            arch=arch,
            steady_ii_scheduler=infeasible_lower_ii,
        )


def test_default_resource_ii_is_lazy_and_has_common_selection_semantics():
    arch = H200_SXM().set_to_microbench()
    fa3_periodic._schedule_cached.cache_clear()

    _, breakdown = model_fa3_us(1, 32, 1024, 128, arch=arch)

    assert fa3_periodic._schedule_cached.cache_info().misses == 0
    assert breakdown["inner_schedule_model"] == "resource_ii"
    assert breakdown["ii_selection_label"] == "resource_ii_lower_bound"
    assert breakdown["ii_selection_scope"] == "lower_bound"
    assert not breakdown["ii_is_constructive"]
    assert breakdown["ii_search_complete"] is None
    assert breakdown["ii_selected_ns"] == pytest.approx(
        breakdown["resource_ii_lb_ns"]
    )


def _periodic_body(d=128, causal=False):
    arch = H200_SXM().set_to_microbench()
    block_m, block_n, mma_pv_is_rs, _ = _tile_config(d, causal)
    return _build_iteration_body(
        arch,
        block_m,
        block_n,
        d,
        dtype_bytes=2,
        accum_bytes=4,
        kv_miss_fraction=1.0,
        mma_pv_is_rs=mma_pv_is_rs,
        max_util=0.85,
    )


def test_periodic_dag_has_stage_tokens_recurrences_and_source_orders():
    dag = build_fa3_periodic_dag(_periodic_body())

    assert {token.name: token.capacity for token in dag.token_buffers} == {
        "K_pipeline": 2,
        "V_pipeline": 2,
    }
    deps = {
        (dependency.source, dependency.target, dependency.iteration_distance)
        for dependency in dag.dependencies
    }
    assert ("online_softmax", "online_softmax", 1) in deps
    assert ("gemm_PV", "rescale_O", 1) in deps
    assert ("load_K", "load_V", -1) in deps
    offsets = {phase.name: phase.iteration_offset for phase in dag.phases}
    assert offsets["load_K"] == offsets["gemm_QK"] == 1
    assert offsets["load_V"] == offsets["gemm_PV"] == 0
    fixed = {order.resource: order.phases for order in dag.fixed_resource_orders}
    assert fixed["tensor"] == ("gemm_QK", "gemm_PV")
    assert fixed["ddr"] == ("load_K", "load_V")
    assert fixed["l2"] == ("load_K", "load_V")


def test_periodic_envelope_is_constructive_and_complete():
    envelope = analyze_fa3_periodic(_periodic_body())

    assert envelope.search_complete
    assert envelope.best.ii == pytest.approx(
        envelope.lower_bounds.overall, abs=2e-12
    )
    assert envelope.worst.ii >= envelope.best.ii
    assert set(envelope.best.phase_starts) == {
        "load_K",
        "load_V",
        "gemm_QK",
        "online_softmax",
        "rescale_O",
        "gemm_PV",
    }


def test_periodic_modes_are_opt_in_and_bound_model_latency():
    arch = H200_SXM().set_to_microbench()
    resource_us, resource = model_fa3_us(1, 32, 1024, 128, arch=arch)
    best_us, best = model_fa3_us(
        1, 32, 1024, 128, arch=arch, inner_schedule_model="periodic_best"
    )
    worst_us, worst = model_fa3_us(
        1, 32, 1024, 128, arch=arch, inner_schedule_model="periodic_worst"
    )

    assert resource["ii_model"] == "resource_ii"
    assert best["ii_model"] == "periodic:fa3_periodic_best"
    assert worst["ii_model"] == "periodic:fa3_periodic_worst"
    assert resource_us <= best_us <= worst_us
    assert best["ii_metadata"]["periodic_search_complete"]
    assert best["ii_selection_scope"] == "model_exhaustive"
    assert best["ii_is_constructive"]
    assert best["ii_search_complete"] is True
    assert worst["ii_selected_ns"] == pytest.approx(worst["steady_ii_ns"])

    # The pre-selector callback surface remains a compatibility path.
    legacy_best_us, legacy_best = model_fa3_us(
        1, 32, 1024, 128, arch=arch, steady_ii_scheduler=fa3_periodic_best
    )
    legacy_worst_us, legacy_worst = model_fa3_us(
        1, 32, 1024, 128, arch=arch, steady_ii_scheduler=fa3_periodic_worst
    )
    assert legacy_best_us == pytest.approx(best_us)
    assert legacy_worst_us == pytest.approx(worst_us)
    assert legacy_best["ii_selection_scope"] == "legacy_hook"
    assert legacy_worst["ii_selection_scope"] == "legacy_hook"
