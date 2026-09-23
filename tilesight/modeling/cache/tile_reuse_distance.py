"""Streaming weighted distinct-allocation reuse distance.

For an allocation's previous access at position ``p``, its weighted LRU stack
distance is the sum of allocation weights whose *latest* position is newer
than ``p``.  A Fenwick tree stores one weight at the latest position of every
resident logical allocation. The Fenwick coordinate is event position, so
query/update is O(log E), not O(log U), for E refresh events.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from ..errors import ModelingValidationError
from .ir import CacheAccessIR, CacheProblem
from .legacy_volume import _sampling_plan
from .result import DistanceSignature, HistogramBin, ReuseHistogram
from .traversal import iter_waves, traversal_sm_count, traversal_wave_size


BACKEND_VERSION = "tile-reuse-distinct-fenwick-v1"
SAMPLED_BACKEND_VERSION = "tile-reuse-distinct-range-sampled-v1"
FENWICK_SAMPLED_BACKEND_VERSION = "tile-reuse-distinct-fenwick-sampled-v1"
STABLE_BACKEND_VERSION = "tile-reuse-stable-shadow-fenwick-v1"
STABLE_SAMPLED_BACKEND_VERSION = "tile-reuse-stable-shadow-range-sampled-v1"
STABLE_FENWICK_SAMPLED_BACKEND_VERSION = (
    "tile-reuse-stable-shadow-fenwick-sampled-v1"
)
_COLD_DISTANCE = 1.0e9


class _Fenwick:
    def __init__(self, size: int) -> None:
        self._tree = [0.0] * (size + 1)
        self.total = 0.0

    def add(self, index: int, delta: float) -> None:
        if index <= 0 or index >= len(self._tree):
            raise ModelingValidationError(
                "Fenwick event bound was underestimated for this traversal"
            )
        self.total += delta
        tree = self._tree
        size = len(tree)
        while index < size:
            tree[index] += delta
            index += index & -index

    def prefix(self, index: int) -> float:
        result = 0.0
        tree = self._tree
        while index > 0:
            result += tree[index]
            index -= index & -index
        return result


class _WeightedLRUStack:
    def __init__(self, max_events: int) -> None:
        self._fenwick = _Fenwick(max(max_events, 1))
        self._last: Dict[Tuple[str, Tuple[int, ...]], Tuple[int, float]] = {}
        self._position = 0

    def access(
        self,
        identity: Tuple[str, Tuple[int, ...]],
        weight: float,
        refresh: bool,
    ) -> Tuple[float, bool]:
        previous = self._last.get(identity)
        cold = previous is None
        distance = (
            _COLD_DISTANCE
            if cold
            else max(self._fenwick.total - self._fenwick.prefix(previous[0]), 0.0)
        )
        if refresh:
            if previous is not None:
                self._fenwick.add(previous[0], -previous[1])
            self._position += 1
            self._fenwick.add(self._position, weight)
            self._last[identity] = (self._position, weight)
        return distance, cold

    def insert_anonymous(self, weight: float) -> None:
        """Insert sampled distinct blocks that are never queried again."""

        if weight <= 0.0:
            return
        self._position += 1
        self._fenwick.add(self._position, weight)


@dataclass(frozen=True)
class _RangeQuery:
    previous: int
    end: int


class _RangeDistinctStack:
    """O(E) trace builder with exact range-distinct queries for Q samples."""

    def __init__(self, _max_events: int) -> None:
        self._sequence: List[int] = []
        self._last: Dict[Tuple[str, Tuple[int, ...]], int] = {}
        self._identity_ids: Dict[Tuple[str, Tuple[int, ...]], int] = {}
        self._weights: List[float] = []

    def _id(self, identity: Tuple[str, Tuple[int, ...]], weight: float) -> int:
        item = self._identity_ids.get(identity)
        if item is None:
            item = len(self._weights)
            self._identity_ids[identity] = item
            self._weights.append(weight)
        return item

    def access(
        self,
        identity: Tuple[str, Tuple[int, ...]],
        weight: float,
        refresh: bool,
    ) -> Tuple[_RangeQuery, bool]:
        previous = self._last.get(identity, -1)
        query = _RangeQuery(previous, len(self._sequence))
        if refresh:
            identity_id = self._id(identity, weight)
            self._sequence.append(identity_id)
            self._last[identity] = len(self._sequence) - 1
        return query, previous < 0

    def insert_anonymous(self, weight: float) -> None:
        if weight <= 0.0:
            return
        identity_id = len(self._weights)
        self._weights.append(weight)
        self._sequence.append(identity_id)

    def resolve(self, query: _RangeQuery) -> float:
        if query.previous < 0:
            return _COLD_DISTANCE
        identities = set(self._sequence[query.previous + 1 : query.end])
        return sum(self._weights[item] for item in identities)


@dataclass
class _MutableBin:
    frequency: float = 0.0
    payload_bytes: float = 0.0
    transaction_bytes: float = 0.0


@dataclass
class _PendingSample:
    access: CacheAccessIR
    frequency: float
    l2_stack: _RangeDistinctStack
    l2_query: _RangeQuery
    l2_cold: bool
    l1_5_stack: Optional[_RangeDistinctStack]
    l1_5_query: Optional[_RangeQuery]
    l1_5_cold: bool
    l2_delta: float = 0.0
    l1_5_delta: float = 0.0


def _refreshes_stack(access: CacheAccessIR) -> bool:
    return access.mode == "read" or access.write_policy.allocate_on_miss


def build_reuse_histogram(problem: CacheProblem) -> ReuseHistogram:
    if not isinstance(problem, CacheProblem):
        raise ModelingValidationError("build_reuse_histogram expects CacheProblem")
    if problem.backend != "tile_reuse_distance":
        raise ModelingValidationError("distinct backend needs tile_reuse_distance problem")

    tile_accesses = tuple(
        item for item in problem.accesses if item.enabled and item.placement == "tile"
    )
    boundary_accesses = tuple(
        item
        for item in problem.accesses
        if item.enabled and item.placement == "legacy_wave_boundary"
    )
    event_accesses = tile_accesses + boundary_accesses
    reduction_fidelity = (
        "exact_no_inner_expansion"
        if problem.inner_iterations == 1
        else (
            "stable_shadow_cohort"
            if problem.reduction.mode == "stable_shadow_cohort"
            else "sampled_inner_compat"
        )
    )
    stable_shadow = (
        problem.reduction.mode == "stable_shadow_cohort"
        and problem.inner_iterations > 1
    )
    if not event_accesses:
        return ReuseHistogram(
            bins=tuple(),
            distance_unit=problem.l2.distance_unit,
            reuse_digest=problem.reuse_digest,
            backend=(STABLE_BACKEND_VERSION if stable_shadow else BACKEND_VERSION),
            distance_semantics="distinct_tile_allocation",
            reduction_fidelity=reduction_fidelity,
            exact=problem.inner_iterations == 1,
            sampled_requests=0,
            represented_requests=0,
        )

    has_l1_5 = problem.l1_5 is not None
    sm_count = traversal_sm_count(problem.traversal)
    group_count = (
        max(
            (sm_count + problem.l1_5_group_size - 1)
            // problem.l1_5_group_size,
            1,
        )
        if has_l1_5
        else 1
    )
    represented = problem.grid.total_tiles * sum(
        item.repetitions for item in event_accesses
    )
    # One anonymous aggregate is inserted per wave globally and per active
    # L1.5 group when reduction-axis expansion is sampled. L1.5 Fenwick arrays
    # are bounded by that group's share rather than allocating global E slots
    # G times.
    wave_size = traversal_wave_size(problem.traversal)
    wave_count = int(math.ceil(float(problem.grid.total_tiles) / wave_size))
    shadow_multiplier = 3 if stable_shadow else 1
    max_events = represented * shadow_multiplier + wave_count + 8
    repetitions_per_coordinate = sum(
        item.repetitions for item in event_accesses
    )
    group_coordinates_per_wave = (
        int(math.ceil(float(wave_size) / sm_count)) * problem.l1_5_group_size
        if has_l1_5
        else wave_size
    )
    max_group_events = (
        min(
            problem.grid.total_tiles,
            wave_count * group_coordinates_per_wave,
        )
        * repetitions_per_coordinate
        * shadow_multiplier
        + wave_count
        + 8
    )
    use_sampled_range = (
        problem.sampling.sample_budget is not None
        and problem.sampling.sample_budget < represented
        # Q range-set queries become more expensive than streaming Fenwick
        # when most events are observed. Keep the offline path for sparse Q.
        and problem.sampling.sample_budget <= max(represented // 4, 1)
    )
    stack_type = _RangeDistinctStack if use_sampled_range else _WeightedLRUStack
    l2_stack = stack_type(max_events)
    l1_5_stacks = [
        stack_type(max_group_events) for _ in range(group_count)
    ]
    sample_plan = _sampling_plan(problem, event_accesses)
    occurrence = {item.name: 0 for item in event_accesses}
    mutable_bins: Dict[
        Tuple[str, str, float, float, bool, bool], _MutableBin
    ] = {}
    pending_samples: List[_PendingSample] = []
    sampled_requests = 0
    rng = np.random.RandomState(problem.sampling.seed)
    l2_unit = problem.l2.unit_bytes
    l1_5_unit = problem.l1_5.unit_bytes if has_l1_5 else l2_unit
    jitter = problem.reduction.progress_jitter
    jitter_enabled = stable_shadow and jitter.enabled
    per_k_l2_units = sum(
        item.repetitions * item.region.allocation_bytes / l2_unit
        for item in tile_accesses
        if _refreshes_stack(item)
    )
    per_k_l1_5_units = sum(
        item.repetitions * item.region.allocation_bytes / l1_5_unit
        for item in tile_accesses
        if _refreshes_stack(item)
    )
    max_l2_delta = (
        min(
            jitter.max_k_ahead * per_k_l2_units,
            jitter.capacity_cap_fraction * problem.l2.capacity_units,
        )
        if jitter_enabled
        else 0.0
    )
    max_l1_5_delta = (
        min(
            jitter.max_k_ahead * per_k_l1_5_units,
            jitter.capacity_cap_fraction * problem.l1_5.capacity_units,
        )
        if jitter_enabled and has_l1_5
        else 0.0
    )

    def accumulate(
        access: CacheAccessIR,
        frequency: float,
        l1_5_distance: float,
        l2_distance: float,
        l1_5_cold: bool,
        l2_cold: bool,
    ) -> None:
        bin_key = (
            access.name,
            access.mode,
            l1_5_distance,
            l2_distance,
            l1_5_cold,
            l2_cold,
        )
        value = mutable_bins.get(bin_key)
        if value is None:
            value = _MutableBin()
            mutable_bins[bin_key] = value
        value.frequency += frequency
        value.payload_bytes += frequency * access.region.payload_bytes
        value.transaction_bytes += frequency * access.transaction_bytes

    def record_measurement(
        access: CacheAccessIR,
        frequency: float,
        l1_5_distance: float,
        l2_distance: float,
        l1_5_cold: bool,
        l2_cold: bool,
        l1_5_delta: float,
        l2_delta: float,
    ) -> None:
        if l1_5_cold:
            l1_5_delta = 0.0
        if l2_cold:
            l2_delta = 0.0
        if l1_5_delta == 0.0 and l2_delta == 0.0:
            accumulate(
                access,
                frequency,
                l1_5_distance,
                l2_distance,
                l1_5_cold,
                l2_cold,
            )
            return
        half = 0.5 * frequency
        for sign in (-1.0, 1.0):
            accumulate(
                access,
                half,
                max(l1_5_distance + sign * l1_5_delta, 0.0),
                max(l2_distance + sign * l2_delta, 0.0),
                l1_5_cold,
                l2_cold,
            )

    def jitter_deltas(
        wave_index: int,
        coordinate: Tuple[int, ...],
        access: CacheAccessIR,
        repetition: int,
    ) -> Tuple[float, float]:
        if not jitter_enabled:
            return 0.0, 0.0
        token = "%d:%d:%d:%r:%s:%d" % (
            problem.sampling.seed,
            jitter.seed,
            wave_index,
            coordinate,
            access.name,
            repetition,
        )
        fraction = int(
            hashlib.sha256(token.encode("utf-8")).hexdigest()[:16], 16
        ) / float(0xFFFFFFFFFFFFFFFF)
        return fraction * max_l1_5_delta, fraction * max_l2_delta

    def process_access(
        access: CacheAccessIR,
        coordinate: Tuple[int, ...],
        group: int,
        unique_global: Set[Tuple[str, Tuple[int, ...]]],
        unique_local: List[Set[Tuple[str, Tuple[int, ...]]]],
        wave_index: int,
        is_inner: bool = True,
    ) -> None:
        nonlocal sampled_requests
        key = access.region.index_map.key(coordinate)
        base_identity = (access.region.value, key)
        identity = (
            (access.region.value, key + (problem.reduction.representative_k,))
            if stable_shadow and is_inner
            else base_identity
        )
        refresh = _refreshes_stack(access)
        for repetition in range(access.repetitions):
            l2_measure, l2_cold = l2_stack.access(
                identity,
                access.region.allocation_bytes / l2_unit,
                refresh,
            )
            if has_l1_5:
                l1_5_measure, l1_5_cold = l1_5_stacks[group].access(
                    identity,
                    access.region.allocation_bytes / l1_5_unit,
                    refresh,
                )
            else:
                l1_5_measure = 0.0
                l1_5_cold = False

            index = occurrence[access.name]
            period, offset, sample_weight, _sample_count = sample_plan[access.name]
            if index % period == offset:
                frequency = access.frequency_weight * sample_weight
                l1_5_delta, l2_delta = (
                    jitter_deltas(wave_index, coordinate, access, repetition)
                    if is_inner
                    else (0.0, 0.0)
                )
                if use_sampled_range:
                    pending_samples.append(
                        _PendingSample(
                            access=access,
                            frequency=frequency,
                            l2_stack=l2_stack,
                            l2_query=l2_measure,
                            l2_cold=l2_cold,
                            l1_5_stack=(l1_5_stacks[group] if has_l1_5 else None),
                            l1_5_query=(l1_5_measure if has_l1_5 else None),
                            l1_5_cold=l1_5_cold,
                            l2_delta=l2_delta,
                            l1_5_delta=l1_5_delta,
                        )
                    )
                else:
                    record_measurement(
                        access,
                        frequency,
                        l1_5_measure,
                        l2_measure,
                        l1_5_cold,
                        l2_cold,
                        l1_5_delta,
                        l2_delta,
                    )
                sampled_requests += 1
            occurrence[access.name] = index + 1
        if refresh:
            unique_global.add(base_identity)
            if has_l1_5:
                unique_local[group].add(base_identity)

    allocation_bytes = {
        item.region.value: item.region.allocation_bytes for item in tile_accesses
    }

    def touch_shadow_set(
        stack: Any,
        identities: Set[Tuple[str, Tuple[int, ...]]],
        count: int,
        unit_bytes: float,
        position: str,
    ) -> None:
        if count <= 0:
            return
        for value, key in sorted(identities, key=repr):
            stack.access(
                ("__shadow_%s__:%s" % (position, value), key),
                count * allocation_bytes[value] / unit_bytes,
                True,
            )

    for wave_index, wave in enumerate(iter_waves(problem.grid, problem.traversal)):
        unique_global: Set[Tuple[str, Tuple[int, ...]]] = set()
        unique_local: List[Set[Tuple[str, Tuple[int, ...]]]] = [
            set() for _ in range(group_count)
        ]
        scheduled = []
        for scheduled_index in rng.permutation(len(wave)):
            coordinate = wave[int(scheduled_index)]
            sm_index = int(scheduled_index) % sm_count
            group = (
                min(sm_index // problem.l1_5_group_size, group_count - 1)
                if has_l1_5
                else 0
            )
            scheduled.append((coordinate, group))

        if stable_shadow:
            for coordinate, group in scheduled:
                for access in tile_accesses:
                    if not _refreshes_stack(access):
                        continue
                    key = access.region.index_map.key(coordinate)
                    identity = (access.region.value, key)
                    unique_global.add(identity)
                    if has_l1_5:
                        unique_local[group].add(identity)
            prefix = problem.reduction.representative_k
            touch_shadow_set(
                l2_stack, unique_global, prefix, l2_unit, "prefix"
            )
            if has_l1_5:
                for group, identities in enumerate(unique_local):
                    touch_shadow_set(
                        l1_5_stacks[group],
                        identities,
                        prefix,
                        l1_5_unit,
                        "prefix",
                    )

        for coordinate, group in scheduled:
            for access in tile_accesses:
                process_access(
                    access,
                    coordinate,
                    group,
                    unique_global,
                    unique_local,
                    wave_index,
                )

        if problem.inner_iterations > 1 and not stable_shadow:
            l2_weight = sum(
                allocation_bytes[value] / l2_unit
                for value, _key in unique_global
            ) * (problem.inner_iterations - 1)
            l2_stack.insert_anonymous(l2_weight)
            if has_l1_5:
                for group, identities in enumerate(unique_local):
                    local_weight = sum(
                        allocation_bytes[value] / l1_5_unit
                        for value, _key in identities
                    ) * (problem.inner_iterations - 1)
                    l1_5_stacks[group].insert_anonymous(local_weight)
        elif stable_shadow:
            suffix = (
                problem.inner_iterations
                - problem.reduction.representative_k
                - 1
            )
            touch_shadow_set(
                l2_stack, unique_global, suffix, l2_unit, "suffix"
            )
            if has_l1_5:
                for group, identities in enumerate(unique_local):
                    touch_shadow_set(
                        l1_5_stacks[group],
                        identities,
                        suffix,
                        l1_5_unit,
                        "suffix",
                    )

        # Compatibility placement means the store happens after the sampled
        # inner loop, but unlike legacy scalar aging these are real allocation
        # identities and tail waves contain only their actual coordinates.
        for coordinate, group in scheduled:
            for access in boundary_accesses:
                process_access(
                    access,
                    coordinate,
                    group,
                    unique_global,
                    unique_local,
                    wave_index,
                    is_inner=False,
                )

    if use_sampled_range:
        for sample in pending_samples:
            l2_distance = sample.l2_stack.resolve(sample.l2_query)
            l1_5_distance = (
                sample.l1_5_stack.resolve(sample.l1_5_query)
                if sample.l1_5_stack is not None
                else 0.0
            )
            record_measurement(
                sample.access,
                sample.frequency,
                l1_5_distance,
                l2_distance,
                sample.l1_5_cold,
                sample.l2_cold,
                sample.l1_5_delta,
                sample.l2_delta,
            )

    bins = tuple(
        HistogramBin(
            access_name=access_name,
            mode=mode,
            signature=DistanceSignature(
                l1_5_distance_units=l1_5_distance,
                l2_distance_units=l2_distance,
                l1_5_cold=l1_5_cold,
                l2_cold=l2_cold,
                has_l1_5=has_l1_5,
                distance_unit=problem.l2.distance_unit,
            ),
            frequency_weight=value.frequency,
            payload_bytes_weight=value.payload_bytes,
            transaction_bytes_weight=value.transaction_bytes,
        )
        for (
            access_name,
            mode,
            l1_5_distance,
            l2_distance,
            l1_5_cold,
            l2_cold,
        ), value in sorted(
            mutable_bins.items(),
            key=lambda item: (
                item[0][0],
                item[0][1],
                item[0][2],
                item[0][3],
                item[0][4],
                item[0][5],
            ),
        )
    )
    fully_observed = (
        problem.sampling.sample_budget is None
        or problem.sampling.sample_budget >= represented
    )
    return ReuseHistogram(
        bins=bins,
        distance_unit=problem.l2.distance_unit,
        reuse_digest=problem.reuse_digest,
        backend=(
            (
                STABLE_SAMPLED_BACKEND_VERSION
                if use_sampled_range
                else (
                    STABLE_FENWICK_SAMPLED_BACKEND_VERSION
                    if not fully_observed
                    else STABLE_BACKEND_VERSION
                )
            )
            if stable_shadow
            else (
                SAMPLED_BACKEND_VERSION
                if use_sampled_range
                else (
                    FENWICK_SAMPLED_BACKEND_VERSION
                    if not fully_observed
                    else BACKEND_VERSION
                )
            )
        ),
        distance_semantics="distinct_tile_allocation",
        reduction_fidelity=reduction_fidelity,
        exact=fully_observed and problem.inner_iterations == 1,
        sampled_requests=sampled_requests,
        represented_requests=represented,
    )
