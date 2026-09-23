"""Hand-checkable traces for the paper distinct-allocation backend."""

from __future__ import annotations

from dataclasses import replace

import pytest

from tilesight.modeling._cache_compat.reuse_distance_multilevel import CacheConfig, TensorAccess
from tilesight.modeling.cache import (
    CacheAccessIR,
    CacheLevelConfig,
    CacheProblem,
    Projection,
    RowMajorTraversal,
    SamplingConfig,
    TensorTileRegion,
    TileGrid,
    build_reuse_histogram,
    evaluate_histogram,
    from_legacy_tensor_accesses,
    gemm_cache_problem,
    model_cache,
)
from tilesight.modeling.errors import ModelingValidationError


def _one_bin(result, access_name):
    bins = [item for item in result.histogram.bins if item.access_name == access_name]
    assert len(bins) == 1
    return bins[0]


def test_repeated_interference_allocation_counts_once_in_x_y_y_x_trace():
    x = TensorTileRegion("X", Projection(tuple()), (64,), 1)
    y = TensorTileRegion("Y", Projection(tuple()), (64,), 1)
    problem = CacheProblem(
        grid=TileGrid((1,)),
        accesses=(
            CacheAccessIR.load("x_first", x),
            CacheAccessIR.load("y_first", y),
            CacheAccessIR.load("y_second", y),
            CacheAccessIR.load("x_second", x),
        ),
        traversal=RowMajorTraversal(1, 1),
        l2=CacheLevelConfig("l2", 4096, 64),
    )
    result = model_cache(problem)

    assert result.histogram.distance_semantics == "distinct_tile_allocation"
    assert result.histogram.exact
    assert _one_bin(result, "y_second").signature.l2_distance_units == 0
    # Y appears twice, but only its latest stack entry is live between X uses.
    assert _one_bin(result, "x_second").signature.l2_distance_units == 1


def test_weighted_distinct_trace_sums_live_allocation_units():
    x = TensorTileRegion("X", Projection(tuple()), (64,), 1)
    y = TensorTileRegion("Y", Projection(tuple()), (128,), 1)
    z = TensorTileRegion("Z", Projection(tuple()), (32,), 1)
    problem = CacheProblem(
        grid=TileGrid((1,)),
        accesses=(
            CacheAccessIR.load("x0", x),
            CacheAccessIR.load("y0", y),
            CacheAccessIR.load("z0", z),
            CacheAccessIR.load("y1", y),
            CacheAccessIR.load("x1", x),
        ),
        traversal=RowMajorTraversal(1, 1),
        l2=CacheLevelConfig("l2", 4096, 64),
    )
    result = model_cache(problem)
    assert _one_bin(result, "y1").signature.l2_distance_units == 0.5
    assert _one_bin(result, "x1").signature.l2_distance_units == 2.5


def test_distinct_distance_is_shared_across_access_names_for_same_value():
    x = TensorTileRegion("X", Projection(tuple()), (128,), 1)
    problem = CacheProblem(
        grid=TileGrid((1,)),
        accesses=(
            CacheAccessIR.load("producer_view", x),
            CacheAccessIR.load("consumer_view", x),
        ),
        traversal=RowMajorTraversal(1, 1),
        l2=CacheLevelConfig("l2", 4096, 128),
    )
    result = model_cache(problem)
    consumer = _one_bin(result, "consumer_view").signature
    assert not consumer.l2_cold
    assert consumer.l2_distance_units == 0


def test_l1_5_uses_ceil_tail_group_and_preserves_joint_distance_signature():
    shared = TensorTileRegion("X", Projection(tuple()), (64,), 1)
    problem = CacheProblem(
        grid=TileGrid((10,)),
        accesses=(CacheAccessIR.load("X", shared),),
        traversal=RowMajorTraversal(wave_size=10, sm_count=10),
        l2=CacheLevelConfig("l2", 4096, 64),
        l1_5=CacheLevelConfig("l1_5", 2048, 64),
        l1_5_group_size=8,
        sampling=SamplingConfig(seed=3),
    )
    result = model_cache(problem)
    bins = [item for item in result.histogram.bins if item.access_name == "X"]
    l1_5_cold = sum(item.frequency_weight for item in bins if item.signature.l1_5_cold)
    l2_cold = sum(item.frequency_weight for item in bins if item.signature.l2_cold)

    # SMs [0..7] and tail SMs [8..9] are two physical L1.5 groups.
    assert l1_5_cold == 2
    assert l2_cold == 1
    # The first tail-group access is L1.5-cold after L2 was globally warmed.
    assert any(
        item.signature.l1_5_cold and not item.signature.l2_cold for item in bins
    )


def test_legacy_volume_keeps_floor_tail_group_quirk_explicitly_labeled():
    legacy = from_legacy_tensor_accesses(
        TileGrid((1, 10)),
        (TensorAccess("X", 64, [0, 1]),),
        sm_count=10,
        l2_config=CacheConfig(4096, 2, 128),
        l1_5_config=CacheConfig(2048, 2, 128),
        l1_5_group_size=8,
        row_panel=10,
        sampling=SamplingConfig(seed=3),
    )
    result = model_cache(legacy)
    assert result.histogram.distance_semantics == "scalar_access_volume"
    assert any("floor L1.5 group" in item for item in result.diagnostics)


def test_legacy_boundary_only_is_trivially_exact_under_finite_budget():
    legacy = from_legacy_tensor_accesses(
        TileGrid((2, 2)),
        tuple(),
        sm_count=4,
        l2_config=CacheConfig(4096, 2, 128),
        row_panel=2,
        output_footprint_bytes=64,
        sampling=SamplingConfig(seed=1, sample_budget=1),
    )
    result = model_cache(legacy)
    assert result.histogram.exact
    assert result.histogram.represented_requests == 0
    assert result.traffic.ddr_write_bytes == 4 * 64


def test_inner_reduction_sampling_is_not_claimed_as_exact_distinct_trace():
    x = TensorTileRegion("X", Projection((0,)), (64,), 1)
    problem = CacheProblem(
        grid=TileGrid((8,)),
        accesses=(CacheAccessIR.load("X", x),),
        traversal=RowMajorTraversal(4, 4),
        l2=CacheLevelConfig("l2", 4096, 64),
        inner_iterations=16,
    )
    result = model_cache(problem)
    assert not result.histogram.exact
    assert result.histogram.reduction_fidelity == "sampled_inner_compat"
    assert any("anonymous distinct-allocation" in item for item in result.diagnostics)


def test_finite_budget_uses_range_distinct_and_stays_within_one_pp():
    mem_levels = {
        "in1": [1, 1, 1, 2],
        "in2": [1, 1, 1, 2],
        "out1": [1, 1, 1, 2],
    }

    def problem(budget):
        return gemm_cache_problem(
            128 * 128,
            128 * 128,
            64 * 32,
            128,
            128,
            32,
            60 * 1024 * 1024,
            120,
            mem_levels,
            8,
            l1_5_group_size=8,
            sampling=SamplingConfig(seed=17, sample_budget=budget),
        )

    exact = model_cache(problem(None))
    sampled = model_cache(problem(512))
    assert sampled.backend == "tile-reuse-distinct-range-sampled-v1"
    assert not sampled.histogram.exact
    errors_pp = (
        abs(exact.aggregate.l1_5_hit_rate - sampled.aggregate.l1_5_hit_rate) * 100,
        abs(exact.aggregate.l2_hit_rate - sampled.aggregate.l2_hit_rate) * 100,
        abs(exact.aggregate.ddr_miss_rate - sampled.aggregate.ddr_miss_rate) * 100,
    )
    assert max(errors_pp) < 1.0


def test_range_sample_signatures_are_exact_subsets_and_dense_budget_uses_fenwick():
    a = TensorTileRegion("A", Projection((0,)), (64,), 1)
    b = TensorTileRegion("B", Projection(tuple()), (64,), 1)
    common = dict(
        grid=TileGrid((64,)),
        accesses=(CacheAccessIR.load("A", a), CacheAccessIR.load("B", b)),
        traversal=RowMajorTraversal(8, 8),
        l2=CacheLevelConfig("l2", 4096, 64),
        sampling=SamplingConfig(seed=11),
    )
    exact = model_cache(CacheProblem(**common))
    sparse = model_cache(
        CacheProblem(**dict(common, sampling=SamplingConfig(seed=11, sample_budget=16)))
    )
    dense = model_cache(
        CacheProblem(**dict(common, sampling=SamplingConfig(seed=11, sample_budget=96)))
    )
    exact_signatures = {
        (item.access_name, item.signature) for item in exact.histogram.bins
    }
    sparse_signatures = {
        (item.access_name, item.signature) for item in sparse.histogram.bins
    }
    assert sparse_signatures <= exact_signatures
    assert sparse.backend == "tile-reuse-distinct-range-sampled-v1"
    assert dense.backend == "tile-reuse-distinct-fenwick-sampled-v1"


def test_evaluator_fails_closed_on_forged_histogram_contract():
    x = TensorTileRegion("X", Projection(tuple()), (64,), 1)
    problem = CacheProblem(
        grid=TileGrid((2,)),
        accesses=(CacheAccessIR.load("X", x),),
        traversal=RowMajorTraversal(2, 2),
        l2=CacheLevelConfig("l2", 4096, 64),
    )
    histogram = build_reuse_histogram(problem)
    first = histogram.bins[0]
    mutations = (
        replace(histogram, bins=tuple()),
        replace(histogram, represented_requests=histogram.represented_requests + 1),
        replace(histogram, sampled_requests=histogram.sampled_requests + 1),
        replace(histogram, distance_semantics="scalar_access_volume"),
        replace(histogram, backend="legacy-volume-scalar-clock-v3"),
        replace(histogram, reduction_fidelity="sampled_inner_compat"),
        replace(histogram, exact=False),
        replace(
            histogram,
            bins=(replace(first, access_name="unknown"),) + histogram.bins[1:],
        ),
        replace(
            histogram,
            bins=(replace(first, mode="write"),) + histogram.bins[1:],
        ),
        replace(
            histogram,
            bins=(
                replace(first, frequency_weight=first.frequency_weight * 0.5),
            )
            + histogram.bins[1:],
        ),
        replace(
            histogram,
            bins=(
                replace(
                    first,
                    payload_bytes_weight=first.payload_bytes_weight * 0.5,
                ),
            )
            + histogram.bins[1:],
        ),
        replace(
            histogram,
            bins=(
                replace(
                    first,
                    transaction_bytes_weight=first.transaction_bytes_weight * 0.5,
                ),
            )
            + histogram.bins[1:],
        ),
        replace(
            histogram,
            bins=(
                replace(
                    first,
                    signature=replace(first.signature, has_l1_5=True),
                ),
            )
            + histogram.bins[1:],
        ),
    )
    for forged in mutations:
        with pytest.raises(ModelingValidationError):
            evaluate_histogram(problem, forged)
