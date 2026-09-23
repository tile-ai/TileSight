"""Representative parity and speed benchmark for both cache backends.

Run from the repository root:

    python -m tilesight.modeling.cache.benchmark
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import replace

import numpy as np

from tilesight.modeling._cache_compat.reuse_distance_multilevel import (
    CacheConfig as LegacyCacheConfig,
    TensorAccess,
    multilevel_hit_rate_general,
)

from .adapters import from_legacy_tensor_accesses, gemm_cache_problem
from .api import evaluate_histogram, model_cache
from .ir import SamplingConfig


def _timed(callable_, repeats):
    samples = []
    value = None
    for _ in range(repeats):
        start = time.perf_counter()
        value = callable_()
        samples.append(time.perf_counter() - start)
    return value, statistics.median(samples), tuple(samples)


def run(side=256, repeats=5, seed=17):
    tensors = (
        TensorAccess("A", footprint_bytes=8192, reuse_dims=[1]),
        TensorAccess("B", footprint_bytes=8192, reuse_dims=[0]),
    )
    l2 = LegacyCacheConfig(60 * 1024 * 1024, 8, 128)
    l1_5 = LegacyCacheConfig(180 * 1024, 8, 128)

    def old_call():
        np.random.seed(seed)
        return multilevel_hit_rate_general(
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
    mem_levels = {
        "in1": [1, 1, 1, 2],
        "in2": [1, 1, 1, 2],
        "out1": [1, 1, 1, 2],
    }
    distinct_problem = gemm_cache_problem(
        side * 128,
        side * 128,
        64 * 32,
        128,
        128,
        32,
        60 * 1024 * 1024,
        120,
        mem_levels,
        8,
        l1_5_group_size=8,
        sampling=SamplingConfig(seed=seed),
    )
    sampled_problem = replace(
        distinct_problem,
        sampling=SamplingConfig(seed=seed, sample_budget=512),
    )
    stable_problem = gemm_cache_problem(
        side * 128,
        side * 128,
        64 * 32,
        128,
        128,
        32,
        60 * 1024 * 1024,
        120,
        mem_levels,
        8,
        l1_5_group_size=8,
        reduction_mode="stable_shadow_cohort",
        representative_k=0,
        sampling=SamplingConfig(seed=seed),
    )
    stable_sampled_problem = replace(
        stable_problem,
        sampling=SamplingConfig(seed=seed, sample_budget=512),
    )

    # Warm imports, dataclass code paths, and allocator state before timing.
    old_call()
    model_cache(problem)
    model_cache(distinct_problem)
    model_cache(sampled_problem)
    model_cache(stable_problem)
    model_cache(stable_sampled_problem)
    old, old_seconds, old_samples = _timed(old_call, repeats)
    new_result, new_seconds, new_samples = _timed(lambda: model_cache(problem), repeats)
    distinct_result, distinct_seconds, distinct_samples = _timed(
        lambda: model_cache(distinct_problem), repeats
    )
    sampled_result, sampled_seconds, sampled_distinct_samples = _timed(
        lambda: model_cache(sampled_problem), repeats
    )
    stable_result, stable_seconds, stable_samples = _timed(
        lambda: model_cache(stable_problem), repeats
    )
    stable_sampled_result, stable_sampled_seconds, stable_sampled_samples = _timed(
        lambda: model_cache(stable_sampled_problem), repeats
    )
    capacity_problem = replace(
        distinct_problem,
        l2=replace(distinct_problem.l2, capacity_bytes=80 * 1024 * 1024),
    )
    _capacity_result, capacity_seconds, capacity_samples = _timed(
        lambda: evaluate_histogram(capacity_problem, distinct_result.histogram),
        repeats,
    )
    new = new_result.to_legacy_dict()
    sampled_errors_pp = (
        abs(
            distinct_result.aggregate.l1_5_hit_rate
            - sampled_result.aggregate.l1_5_hit_rate
        )
        * 100.0,
        abs(
            distinct_result.aggregate.l2_hit_rate
            - sampled_result.aggregate.l2_hit_rate
        )
        * 100.0,
        abs(
            distinct_result.aggregate.ddr_miss_rate
            - sampled_result.aggregate.ddr_miss_rate
        )
        * 100.0,
    )
    stable_delta_pp = max(
        abs(
            distinct_result.aggregate.l1_5_hit_rate
            - stable_result.aggregate.l1_5_hit_rate
        ),
        abs(
            distinct_result.aggregate.l2_hit_rate
            - stable_result.aggregate.l2_hit_rate
        ),
        abs(
            distinct_result.aggregate.ddr_miss_rate
            - stable_result.aggregate.ddr_miss_rate
        ),
    ) * 100.0
    errors_pp = {
        name: abs(float(old[name]) - float(new[name])) * 100.0
        for name in ("l1_5_hit_rate", "l2_hit_rate", "ddr_miss_rate")
    }
    return {
        "grid": (side, side),
        "old_seconds": old_seconds,
        "new_seconds": new_seconds,
        "speedup": old_seconds / new_seconds,
        "distinct_seconds": distinct_seconds,
        "distinct_vs_old": old_seconds / distinct_seconds,
        "cached_capacity_seconds": capacity_seconds,
        "sampled_distinct_seconds": sampled_seconds,
        "sampled_distinct_error_pp": max(sampled_errors_pp),
        "stable_seconds": stable_seconds,
        "stable_sampled_seconds": stable_sampled_seconds,
        "stable_delta_pp": stable_delta_pp,
        "max_error_pp": max(errors_pp.values()),
        "errors_pp": errors_pp,
        "old_samples": old_samples,
        "new_samples": new_samples,
        "distinct_samples": distinct_samples,
        "capacity_samples": capacity_samples,
        "sampled_distinct_samples": sampled_distinct_samples,
        "stable_samples": stable_samples,
        "stable_sampled_samples": stable_sampled_samples,
        "histogram_bins": len(new_result.histogram.bins),
        "problem_digest": problem.digest,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    result = run(args.side, args.repeats, args.seed)
    print("grid: %dx%d" % result["grid"])
    print("legacy median: %.6f s" % result["old_seconds"])
    print("scalar-clock median: %.6f s" % result["new_seconds"])
    print("speedup: %.2fx" % result["speedup"])
    print("distinct Fenwick median: %.6f s" % result["distinct_seconds"])
    print("distinct / legacy-old speed ratio: %.2fx" % result["distinct_vs_old"])
    print(
        "sampled range-distinct (budget=512): %.6f s, max error %.4f pp"
        % (
            result["sampled_distinct_seconds"],
            result["sampled_distinct_error_pp"],
        )
    )
    print(
        "stable shadow: %.6f s; sampled %.6f s; delta vs anonymous %.4f pp"
        % (
            result["stable_seconds"],
            result["stable_sampled_seconds"],
            result["stable_delta_pp"],
        )
    )
    print("cached capacity-only evaluation: %.6f s" % result["cached_capacity_seconds"])
    print("maximum hit-rate error: %.12g pp" % result["max_error_pp"])
    print("sparse joint bins: %d" % result["histogram_bins"])
    print("problem digest: %s" % result["problem_digest"])


if __name__ == "__main__":
    main()
