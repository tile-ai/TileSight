"""Phase-1 tests for the generic sampled tile-reuse cache backend."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from tilesight.modeling._cache_compat.reuse_distance_multilevel import (
    CacheConfig as LegacyCacheConfig,
    TensorAccess,
    multilevel_hit_rate_general,
)
from tilesight.modeling.cache import (
    AffineIndexMap,
    CacheAccessIR,
    CacheLevelConfig,
    CacheProblem,
    PanelTraversal,
    Projection,
    RowMajorTraversal,
    SamplingConfig,
    TensorTileRegion,
    TileGrid,
    WritePolicy,
    build_reuse_histogram,
    eliminate_fused_accesses,
    evaluate_histogram,
    from_legacy_tensor_accesses,
    gemm_cache_problem,
    model_cache,
)
from tilesight.modeling.errors import ModelingValidationError


MEM_LEVELS = {
    "in1": [1, 1, 1, 2],
    "in2": [1, 1, 1, 2],
    "out1": [1, 1, 1, 2],
}


def _legacy_case(seed=17, side=32, access_count=1):
    tensors = [
        TensorAccess("A", 8192, [1], access_count=access_count),
        TensorAccess("B", 8192, [0]),
    ]
    l2 = LegacyCacheConfig(60 * 1024 * 1024, 8, 128)
    l1_5 = LegacyCacheConfig(180 * 1024, 8, 128)
    np.random.seed(seed)
    old = multilevel_hit_rate_general(
        (side, side),
        tensors,
        120,
        l2,
        l1_5,
        l1_5_group_size=8,
        row_panel=8,
        k_iterations=64,
        output_footprint_bytes=32768,
    )
    problem = from_legacy_tensor_accesses(
        (side, side),
        tensors,
        120,
        l2,
        l1_5,
        l1_5_group_size=8,
        row_panel=8,
        inner_iterations=64,
        output_footprint_bytes=32768,
        sampling=SamplingConfig(seed=seed),
    )
    return old, problem, model_cache(problem)


@pytest.mark.parametrize("seed", [0, 7, 41])
def test_explicit_legacy_volume_adapter_has_exact_old_model_parity(seed):
    old, problem, result = _legacy_case(seed=seed)
    new = result.to_legacy_dict()

    assert problem.backend == "legacy_volume"
    assert result.histogram.distance_unit == "legacy_cacheline_volume"
    for name in ("l1_5_hit_rate", "l2_hit_rate", "ddr_miss_rate"):
        assert new[name] == pytest.approx(old[name], rel=0.0, abs=2.0e-14)
    for tensor in ("A", "B"):
        for name in ("l1_5_hit", "l2_hit", "ddr_miss"):
            assert new["per_tensor"][tensor][name] == pytest.approx(
                old["per_tensor"][tensor][name], rel=0.0, abs=2.0e-14
            )


def test_legacy_access_count_and_owned_rng_match_without_mutating_global_rng():
    old, problem, result = _legacy_case(seed=23, access_count=3)
    assert result.to_legacy_dict()["l2_hit_rate"] == pytest.approx(
        old["l2_hit_rate"], rel=0.0, abs=2.0e-14
    )

    np.random.seed(101)
    expected = np.random.random(4)
    np.random.seed(101)
    model_cache(problem)
    actual = np.random.random(4)
    assert actual == pytest.approx(expected, rel=0.0, abs=0.0)


def test_legacy_tensor_access_adapter_accepts_typed_tile_grid():
    tensors = (TensorAccess("A", 1024, [1]),)
    l2 = LegacyCacheConfig(8 * 1024 * 1024, 8, 128)
    problem = from_legacy_tensor_accesses(
        TileGrid((4, 8), ("m", "n")), tensors, 16, l2, row_panel=4
    )
    assert problem.grid.axes == ("m", "n")


def test_object_gemm_defaults_to_paper_tile_allocation_units():
    problem = gemm_cache_problem(
        4096,
        4096,
        4096,
        128,
        128,
        32,
        60 * 1024 * 1024,
        120,
        MEM_LEVELS,
        8,
        l1_5_group_size=8,
    )
    result = model_cache(problem)

    assert problem.backend == "tile_reuse_distance"
    assert problem.l2.distance_unit == "tile_allocation"
    assert problem.l2.unit_bytes == 128 * 32 * 2
    assert problem.l2.capacity_units == (60 * 1024 * 1024) / (128 * 32 * 2)
    assert result.histogram.distance_unit == "tile_allocation"
    assert all(
        item.signature.distance_unit == "tile_allocation"
        for item in result.histogram.bins
    )


def test_paper_distinct_tile_gemm_stays_within_one_pp_of_legacy_calibration():
    side = 64
    old, _legacy_problem, _legacy_result = _legacy_case(seed=17, side=side)
    paper = model_cache(
        gemm_cache_problem(
            side * 128,
            side * 128,
            64 * 32,
            128,
            128,
            32,
            60 * 1024 * 1024,
            120,
            MEM_LEVELS,
            8,
            l1_5_group_size=8,
            sampling=SamplingConfig(seed=17),
        )
    ).aggregate
    for legacy, current in (
        (old["l1_5_hit_rate"], paper.l1_5_hit_rate),
        (old["l2_hit_rate"], paper.l2_hit_rate),
        (old["ddr_miss_rate"], paper.ddr_miss_rate),
    ):
        assert abs(legacy - current) * 100.0 < 1.0


def test_cacheline_size_does_not_change_tile_reuse_distance_or_capacity_units():
    base = gemm_cache_problem(
        1024,
        1024,
        1024,
        128,
        128,
        32,
        8 * 1024 * 1024,
        120,
        MEM_LEVELS,
        8,
    )
    changed = replace(base, l2=replace(base.l2, cacheline_bytes=256))
    base_histogram = model_cache(base).histogram
    changed_histogram = model_cache(changed).histogram

    assert base.l2.capacity_units == changed.l2.capacity_units
    assert tuple(item.signature for item in base_histogram.bins) == tuple(
        item.signature for item in changed_histogram.bins
    )


def test_write_through_is_a_first_class_polluting_event_and_conserves_traffic():
    read = CacheAccessIR.load(
        "read_A",
        TensorTileRegion("A", Projection((0,)), (64,), 1),
    )
    store = CacheAccessIR.store(
        "store_C",
        TensorTileRegion("C", Projection((0, 1)), (64,), 1),
        transaction_bytes=128,
        write_policy=WritePolicy(propagation="write_through"),
    )
    problem = CacheProblem(
        grid=TileGrid((1, 16), ("m", "n")),
        accesses=(read, store),
        traversal=RowMajorTraversal(wave_size=4, sm_count=4),
        l2=CacheLevelConfig("l2", 256, 64, associativity=1, cacheline_bytes=128),
        sampling=SamplingConfig(seed=9),
    )
    with_store = model_cache(problem)
    without_store = model_cache(problem.eliminate_accesses(("store_C",)))

    writes = with_store.by_name("store_C")
    assert writes.traffic.payload_write_bytes == 16 * 64
    assert writes.traffic.ddr_write_bytes == 16 * 128
    assert writes.traffic.transaction_amplification_bytes == 16 * 64
    assert without_store.traffic.ddr_write_bytes == 0
    # Removing a fused-away store happens before clock construction, so it no
    # longer ages the repeated A allocation.
    assert without_store.by_name("read_A").ddr_miss_rate < with_store.by_name(
        "read_A"
    ).ddr_miss_rate


def test_write_allocate_store_warms_later_read_of_same_value():
    region = TensorTileRegion("X", Projection((0,)), (64,), 1)
    allocating_store = CacheAccessIR.store(
        "store_X", region, write_policy=WritePolicy(allocate_on_miss=True)
    )
    read = CacheAccessIR.load("read_X", region)
    base = CacheProblem(
        grid=TileGrid((1,)),
        accesses=(allocating_store, read),
        traversal=RowMajorTraversal(1, 1),
        l2=CacheLevelConfig("l2", 4096, 64),
    )
    warm = model_cache(base)
    cold = model_cache(
        replace(
            base,
            accesses=(
                replace(
                    allocating_store,
                    write_policy=WritePolicy(allocate_on_miss=False),
                ),
                read,
            ),
        )
    )

    assert warm.by_name("read_X").ddr_miss_rate == 0
    assert cold.by_name("read_X").ddr_miss_rate == 1


def test_multiple_accesses_to_same_value_match_sequential_repetitions():
    region = TensorTileRegion("X", Projection((0,)), (64,), 1)
    common = dict(
        grid=TileGrid((8,)),
        traversal=RowMajorTraversal(4, 4),
        l2=CacheLevelConfig("l2", 4096, 64),
        inner_iterations=3,
        sampling=SamplingConfig(seed=13),
    )
    split = CacheProblem(
        accesses=(
            CacheAccessIR.load("first", region),
            CacheAccessIR.load("second", region),
        ),
        **common
    )
    repeated = CacheProblem(
        accesses=(CacheAccessIR.load("combined", region, repetitions=2),),
        **common
    )

    split_result = model_cache(split)
    repeated_result = model_cache(repeated)
    assert split_result.aggregate == repeated_result.aggregate
    assert split_result.traffic == repeated_result.traffic
    assert split_result.histogram.represented_requests == 16


def test_same_value_requires_one_allocation_identity_and_footprint():
    first = TensorTileRegion("X", Projection((0,)), (64,), 1)
    different_footprint = TensorTileRegion(
        "X", Projection((0,)), (32,), 1, allocation_bytes=128
    )
    with pytest.raises(ModelingValidationError, match="disagree"):
        CacheProblem(
            grid=TileGrid((1,)),
            accesses=(
                CacheAccessIR.load("first", first),
                CacheAccessIR.load("second", different_footprint),
            ),
            traversal=RowMajorTraversal(1, 1),
            l2=CacheLevelConfig("l2", 4096, 64),
        )


def test_transaction_amplification_does_not_change_allocation_reuse_distance():
    region = TensorTileRegion(
        "X", Projection(tuple()), (64,), 1, allocation_bytes=64
    )
    base = CacheProblem(
        grid=TileGrid((8,)),
        accesses=(CacheAccessIR.load("X", region, transaction_bytes=64),),
        traversal=RowMajorTraversal(4, 4),
        l2=CacheLevelConfig("l2", 4096, 64),
    )
    amplified = replace(
        base,
        accesses=(CacheAccessIR.load("X", region, transaction_bytes=128),),
    )
    base_result = model_cache(base)
    amplified_result = model_cache(amplified)
    assert tuple(item.signature for item in base_result.histogram.bins) == tuple(
        item.signature for item in amplified_result.histogram.bins
    )
    assert amplified_result.traffic.transaction_amplification_bytes > 0


def test_fusion_plan_accesses_are_removed_before_histogram_even_if_all_disappear():
    read = CacheAccessIR.load(
        "baseline_read",
        TensorTileRegion("X", Projection((0,)), (16,), 4),
    )
    write = CacheAccessIR.store(
        "baseline_write",
        TensorTileRegion("X", Projection((0,)), (16,), 4),
        write_policy=WritePolicy(),
    )
    problem = CacheProblem(
        grid=TileGrid((4,)),
        accesses=(write, read),
        traversal=RowMajorTraversal(4, 4),
        l2=CacheLevelConfig("l2", 4096, 64),
    )
    handoff = SimpleNamespace(
        baseline_writes=(SimpleNamespace(name="baseline_write"),),
        baseline_reads=(SimpleNamespace(name="baseline_read"),),
    )
    fused = eliminate_fused_accesses(
        problem, SimpleNamespace(handoffs=(handoff,))
    )
    result = model_cache(fused)

    assert fused.accesses == tuple()
    assert result.histogram.bins == tuple()
    assert result.traffic == type(result.traffic)()


@pytest.mark.parametrize("flush", [False, True])
def test_write_back_dirty_conservation_is_explicit(flush):
    store = CacheAccessIR.store(
        "store",
        TensorTileRegion("C", Projection((0,)), (32,), 2),
        write_policy=WritePolicy(propagation="write_back", flush_at_end=flush),
    )
    problem = CacheProblem(
        grid=TileGrid((8,), ("x",)),
        accesses=(store,),
        traversal=RowMajorTraversal(wave_size=4, sm_count=4),
        l2=CacheLevelConfig("l2", 4096, 64),
    )
    traffic = model_cache(problem).traffic
    assert traffic.dirty_created_bytes == pytest.approx(
        traffic.dirty_writeback_bytes + traffic.dirty_resident_bytes
    )
    assert traffic.dirty_eviction_writeback_bytes == 0
    assert (traffic.dirty_terminal_flush_bytes > 0) is flush
    assert (traffic.dirty_writeback_bytes > 0) is flush
    assert (traffic.dirty_resident_bytes == 0) is flush


def test_sampling_is_deterministic_weighted_and_part_of_digest():
    exact = gemm_cache_problem(
        8192,
        8192,
        4096,
        128,
        128,
        32,
        60 * 1024 * 1024,
        120,
        MEM_LEVELS,
        8,
        sampling=SamplingConfig(seed=5),
    )
    sampled = replace(exact, sampling=SamplingConfig(seed=5, sample_budget=128))
    another_seed = replace(sampled, sampling=SamplingConfig(seed=6, sample_budget=128))
    first = model_cache(sampled)
    second = model_cache(sampled)

    assert first == second
    assert not first.histogram.exact
    assert first.histogram.sampled_requests <= 128
    assert sum(item.request_count for item in first.per_access if item.mode == "read") == pytest.approx(
        exact.grid.total_tiles * 2
    )
    assert exact.digest != sampled.digest != another_seed.digest
    with pytest.raises(ModelingValidationError, match="at least one request"):
        model_cache(replace(exact, sampling=SamplingConfig(seed=5, sample_budget=1)))


def test_repetitions_are_real_recency_events_while_frequency_weight_is_not():
    access = CacheAccessIR.load(
        "A",
        TensorTileRegion("A", Projection(tuple()), (64,), 1),
        repetitions=2,
        frequency_weight=3,
    )
    problem = CacheProblem(
        grid=TileGrid((1,)),
        accesses=(access,),
        traversal=RowMajorTraversal(1, 1),
        l2=CacheLevelConfig("l2", 4096, 64),
    )
    result = model_cache(problem)

    assert result.histogram.represented_requests == 2
    assert result.histogram.sampled_requests == 2
    assert result.by_name("A").request_count == 6
    # First sequential request is cold; the second sees the just-refreshed
    # allocation. A frequency weight scales both, but creates no third state
    # transition.
    assert result.by_name("A").ddr_miss_rate == pytest.approx(0.5)
    assert result.traffic.payload_read_bytes == 6 * 64


def test_affine_index_map_and_validation():
    mapping = AffineIndexMap(((1, 0), (0, 1)), offset=(1, -1), modulus=(None, 4))
    assert mapping.key((3, 6)) == (4, 1)
    with pytest.raises(ModelingValidationError, match="rank"):
        mapping.key((3,))
    with pytest.raises(ModelingValidationError, match="same reuse-distance unit"):
        CacheProblem(
            grid=TileGrid((1,)),
            accesses=(
                CacheAccessIR.load(
                    "A", TensorTileRegion("A", Projection((0,)), (1,), 1)
                ),
            ),
            traversal=RowMajorTraversal(1, 1),
            l2=CacheLevelConfig("l2", 1024, 64),
            l1_5=CacheLevelConfig(
                "l1_5",
                2048,
                128,
                distance_unit="legacy_cacheline_volume",
            ),
            l1_5_group_size=1,
        )


def test_digest_tracks_access_order_traversal_policy_and_unit_version():
    base = gemm_cache_problem(
        1024,
        1024,
        1024,
        128,
        128,
        32,
        8 * 1024 * 1024,
        32,
        MEM_LEVELS,
        4,
    )
    assert base.digest == gemm_cache_problem(
        1024,
        1024,
        1024,
        128,
        128,
        32,
        8 * 1024 * 1024,
        32,
        MEM_LEVELS,
        4,
    ).digest
    assert base.digest != replace(base, accesses=tuple(reversed(base.accesses))).digest
    assert base.digest != replace(
        base, traversal=replace(base.traversal, raster_axis="along_m")
    ).digest
    store_index = next(i for i, item in enumerate(base.accesses) if item.mode == "write")
    changed_accesses = list(base.accesses)
    changed_accesses[store_index] = replace(
        changed_accesses[store_index],
        write_policy=WritePolicy(propagation="write_back"),
    )
    assert base.digest != replace(base, accesses=tuple(changed_accesses)).digest


def test_reuse_digest_allows_capacity_sweep_but_rejects_trace_mismatch():
    base = gemm_cache_problem(
        2048,
        2048,
        1024,
        128,
        128,
        32,
        8 * 1024 * 1024,
        32,
        MEM_LEVELS,
        4,
    )
    histogram = build_reuse_histogram(base)
    larger = replace(
        base,
        l2=replace(base.l2, capacity_bytes=16 * 1024 * 1024, associativity=16),
    )
    assert base.digest != larger.digest
    assert base.reuse_digest == larger.reuse_digest
    assert evaluate_histogram(larger, histogram).problem_digest == larger.digest

    reordered = replace(base, accesses=tuple(reversed(base.accesses)))
    with pytest.raises(ModelingValidationError, match="histogram does not match"):
        evaluate_histogram(reordered, histogram)
