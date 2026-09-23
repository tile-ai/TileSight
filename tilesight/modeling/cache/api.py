"""Stable cache-model dispatch between paper and compatibility backends."""

from __future__ import annotations

from .ir import CacheProblem
from .legacy_volume import (
    build_reuse_histogram as build_legacy_histogram,
    evaluate_histogram,
)
from .result import CacheResult, ReuseHistogram
from .tile_reuse_distance import build_reuse_histogram as build_distinct_histogram


def build_reuse_histogram(problem: CacheProblem) -> ReuseHistogram:
    if problem.backend == "tile_reuse_distance":
        return build_distinct_histogram(problem)
    if problem.backend == "legacy_volume":
        return build_legacy_histogram(problem)
    raise ValueError("unsupported cache backend %r" % problem.backend)


def model_cache(problem: CacheProblem) -> CacheResult:
    return evaluate_histogram(problem, build_reuse_histogram(problem))


__all__ = ["build_reuse_histogram", "evaluate_histogram", "model_cache"]

