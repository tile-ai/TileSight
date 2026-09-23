"""Regression tests for the stable representative-K shadow-cohort mode."""

from __future__ import annotations

from dataclasses import replace

import pytest

from tilesight.modeling.cache import (
    ProgressJitterConfig,
    SamplingConfig,
    gemm_cache_problem,
    evaluate_histogram,
    model_cache,
)
from tilesight.modeling.errors import ModelingValidationError
from tilesight.modeling.cache.traversal import iter_waves
from tilesight.modeling.cache import tile_reuse_distance as trd
from tilesight.util.sdcm import sdcm


MEM_LEVELS = {
    "in1": [1, 1, 1, 2],
    "in2": [1, 1, 1, 2],
    "out1": [1, 1, 1, 2],
}


class _NaiveWeightedLRU:
    def __init__(self):
        self.order = []
        self.weights = {}

    def access(self, identity, weight):
        if identity in self.weights:
            index = self.order.index(identity)
            distance = sum(self.weights[item] for item in self.order[index + 1 :])
            self.order.pop(index)
        else:
            distance = 1.0e9
        self.order.append(identity)
        self.weights[identity] = weight
        return distance


def _problem(
    inner=8,
    representative=0,
    jitter=ProgressJitterConfig(),
    l1_5=False,
):
    return gemm_cache_problem(
        8,
        8,
        inner,
        2,
        2,
        1,
        4096,
        4,
        MEM_LEVELS,
        2,
        l1_5_group_size=1 if l1_5 else 0,
        l1_5_capacity_per_group=1024,
        reduction_mode="stable_shadow_cohort",
        representative_k=representative,
        progress_jitter=jitter,
        sampling=SamplingConfig(seed=7),
    )


def _full_trace_representative_l2(problem):
    """Full lockstep K trace, observing only the configured representative."""

    accesses = {item.name: item for item in problem.accesses}
    a, b, c = accesses["A"], accesses["B"], accesses["C"]
    unit = problem.l2.unit_bytes
    stack = _NaiveWeightedLRU()
    hit_bytes = 0.0
    total_bytes = 0.0
    import numpy as np

    rng = np.random.RandomState(problem.sampling.seed)
    for wave in iter_waves(problem.grid, problem.traversal):
        order = list(rng.permutation(len(wave)))
        scheduled = [wave[int(item)] for item in order]
        for k in range(problem.inner_iterations):
            for m, n in scheduled:
                for access, identity in (
                    (a, ("A", (m, k))),
                    (b, ("B", (k, n))),
                ):
                    distance = stack.access(
                        identity, access.region.allocation_bytes / unit
                    )
                    if k == problem.reduction.representative_k:
                        probability = min(
                            max(
                                float(
                                    sdcm(
                                        distance,
                                        problem.l2.associativity,
                                        problem.l2.capacity_units,
                                    )
                                ),
                                0.0,
                            ),
                            1.0,
                        )
                        hit_bytes += probability * access.region.payload_bytes
                        total_bytes += access.region.payload_bytes
        for m, n in scheduled:
            stack.access(("C", (m, n)), c.region.allocation_bytes / unit)
    return hit_bytes / total_bytes


@pytest.mark.parametrize("representative", [0, 3, 7])
def test_stable_shadow_matches_small_full_lockstep_trace(representative):
    problem = _problem(representative=representative)
    result = model_cache(problem)
    reference = _full_trace_representative_l2(problem)
    assert result.backend == "tile-reuse-stable-shadow-fenwick-v1"
    assert result.histogram.reduction_fidelity == "stable_shadow_cohort"
    assert result.aggregate.l2_hit_rate == pytest.approx(reference, abs=2.0e-12)


def test_shadow_identities_repeat_across_waves_and_tail_stores_are_real(monkeypatch):
    problem = _problem(inner=16)
    original = trd._WeightedLRUStack.access
    shadow = []
    output = []

    def wrapped(self, identity, weight, refresh):
        if identity[0].startswith("__shadow_"):
            shadow.append(identity)
        if identity[0] == "C":
            output.append(identity)
        return original(self, identity, weight, refresh)

    monkeypatch.setattr(trd._WeightedLRUStack, "access", wrapped)
    model_cache(problem)
    assert len(shadow) > len(set(shadow))
    # One L2 and one L1.5 state are absent here, so every real C tile is touched once.
    assert len(output) == problem.grid.total_tiles


def test_progress_jitter_is_deterministic_antithetic_and_capacity_capped():
    base = _problem(inner=8, l1_5=True)
    jittered = _problem(
        inner=8,
        l1_5=True,
        jitter=ProgressJitterConfig(
            max_k_ahead=1000,
            capacity_cap_fraction=1.0e-5,
            seed=19,
        ),
    )
    baseline = model_cache(base)
    first = model_cache(jittered)
    second = model_cache(jittered)
    assert first == second

    for name in ("A", "B"):
        base_bins = [
            item for item in baseline.histogram.bins
            if item.access_name == name and not item.signature.l2_cold
        ]
        jitter_bins = [
            item for item in first.histogram.bins
            if item.access_name == name and not item.signature.l2_cold
        ]
        # Every baseline occurrence is split into an equal-frequency +/- pair.
        assert sum(item.frequency_weight for item in jitter_bins) == pytest.approx(
            sum(item.frequency_weight for item in base_bins)
        )
        base_mean_l2 = sum(
            item.frequency_weight * item.signature.l2_distance_units
            for item in base_bins
        ) / sum(item.frequency_weight for item in base_bins)
        jitter_mean_l2 = sum(
            item.frequency_weight * item.signature.l2_distance_units
            for item in jitter_bins
        ) / sum(item.frequency_weight for item in jitter_bins)
        assert jitter_mean_l2 == pytest.approx(base_mean_l2, abs=2.0e-12)
        base_l2 = [item.signature.l2_distance_units for item in base_bins]
        l2_cap = jittered.reduction.progress_jitter.capacity_cap_fraction * (
            jittered.l2.capacity_units
        )
        assert all(
            min(abs(item.signature.l2_distance_units - value) for value in base_l2)
            <= l2_cap + 1.0e-12
            for item in jitter_bins
        )
        base_mean_l1 = sum(
            item.frequency_weight * item.signature.l1_5_distance_units
            for item in base_bins
        ) / sum(item.frequency_weight for item in base_bins)
        jitter_mean_l1 = sum(
            item.frequency_weight * item.signature.l1_5_distance_units
            for item in jitter_bins
        ) / sum(item.frequency_weight for item in jitter_bins)
        assert jitter_mean_l1 == pytest.approx(base_mean_l1, abs=2.0e-12)
        base_l1 = [item.signature.l1_5_distance_units for item in base_bins]
        l1_cap = jittered.reduction.progress_jitter.capacity_cap_fraction * (
            jittered.l1_5.capacity_units
        )
        assert all(
            min(abs(item.signature.l1_5_distance_units - value) for value in base_l1)
            <= l1_cap + 1.0e-12
            for item in jitter_bins
        )


def test_stable_shadow_state_event_count_is_independent_of_k_extent(monkeypatch):
    original = trd._WeightedLRUStack.access

    def count_for(inner):
        calls = [0]

        def wrapped(self, identity, weight, refresh):
            calls[0] += 1
            return original(self, identity, weight, refresh)

        with monkeypatch.context() as context:
            context.setattr(trd._WeightedLRUStack, "access", wrapped)
            model_cache(_problem(inner=inner))
        return calls[0]

    short = count_for(4)
    long = count_for(128)
    assert short == long
    assert long < _problem(inner=128).grid.total_tiles * 128 * 2


def test_anonymous_mode_remains_the_adapter_default():
    problem = gemm_cache_problem(
        8, 8, 8, 2, 2, 1, 4096, 4, MEM_LEVELS, 2
    )
    result = model_cache(problem)
    assert problem.reduction.mode == "anonymous_inner"
    assert result.histogram.reduction_fidelity == "sampled_inner_compat"


def test_stable_histogram_cannot_be_relabelled_as_anonymous():
    problem = _problem()
    result = model_cache(problem)
    forged = replace(
        result.histogram,
        backend="tile-reuse-distinct-fenwick-v1",
        reduction_fidelity="sampled_inner_compat",
    )
    with pytest.raises(ModelingValidationError):
        evaluate_histogram(problem, forged)
