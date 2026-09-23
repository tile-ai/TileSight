"""Fast scalar-clock implementation of TileSight's reuse-volume SDCM model."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Set, Tuple

import numpy as np

from tilesight.util.sdcm import sdcm

from ..errors import ModelingValidationError
from .ir import CacheAccessIR, CacheProblem, WritePolicy
from .result import (
    CacheAccessResult,
    CacheAggregateResult,
    CacheResult,
    CacheTraffic,
    DistanceSignature,
    HistogramBin,
    ReuseHistogram,
)
from .traversal import iter_waves, traversal_sm_count, traversal_wave_size


BACKEND_VERSION = "legacy-volume-scalar-clock-v3"
_COLD_DISTANCE = 1.0e9


@dataclass
class _MutableBin:
    frequency: float = 0.0
    payload_bytes: float = 0.0
    transaction_bytes: float = 0.0


@dataclass
class _AccessStats:
    requests: float = 0.0
    l1_5_hits: float = 0.0
    l2_hits: float = 0.0
    ddr_misses: float = 0.0
    payload_bytes: float = 0.0
    l1_5_payload_bytes: float = 0.0
    l2_payload_bytes: float = 0.0
    ddr_payload_bytes: float = 0.0
    traffic: CacheTraffic = CacheTraffic()


def _sampling_plan(problem: CacheProblem, tile_accesses: Tuple[CacheAccessIR, ...]):
    """Return per-access ``(period, offset, weight, sample_count)`` tuples."""

    tiles = problem.grid.total_tiles
    total_requests = tiles * sum(item.repetitions for item in tile_accesses)
    budget = problem.sampling.sample_budget
    if budget is None or budget >= total_requests:
        return {
            item.name: (1, 0, 1.0, tiles * item.repetitions)
            for item in tile_accesses
        }

    if budget < len(tile_accesses):
        raise ModelingValidationError(
            "sample_budget must cover at least one request per tile access"
        )
    repetition_sum = sum(item.repetitions for item in tile_accesses)
    remaining = budget - len(tile_accesses)
    quotas = {
        item.name: 1 + (remaining * item.repetitions) // repetition_sum
        for item in tile_accesses
    }
    assigned = sum(quotas.values())
    if assigned < budget:
        priorities = sorted(
            tile_accesses,
            key=lambda item: (
                -((remaining * item.repetitions) % repetition_sum), item.name
            ),
        )
        for item in priorities[: budget - assigned]:
            quotas[item.name] += 1

    result = {}
    for access in tile_accesses:
        access_requests = tiles * access.repetitions
        share = quotas[access.name]
        period = max(int(math.ceil(float(access_requests) / share)), 1)
        payload = ("%d:%s" % (problem.sampling.seed, access.name)).encode("utf-8")
        offset = int(hashlib.sha256(payload).hexdigest()[:16], 16) % period
        sample_count = (
            0
            if offset >= access_requests
            else 1 + (access_requests - 1 - offset) // period
        )
        if sample_count <= 0:
            raise ModelingValidationError("systematic cache sampling selected no samples")
        result[access.name] = (
            period,
            offset,
            float(access_requests) / sample_count,
            sample_count,
        )
    return result


def build_reuse_histogram(problem: CacheProblem) -> ReuseHistogram:
    """Build a sparse joint L1.5/L2 reuse-distance histogram.

    The historical implementation stores an RD vector per tensor and adds a
    scalar to the entire vector after every tile.  Algebraically, each entry is
    ``clock - last_timestamp``.  This implementation advances one scalar clock
    and changes only the accessed timestamp, preserving the exact distance.
    """

    if not isinstance(problem, CacheProblem):
        raise ModelingValidationError("build_reuse_histogram expects CacheProblem")
    tile_accesses = tuple(
        item for item in problem.accesses if item.enabled and item.placement == "tile"
    )
    boundary_accesses = tuple(
        item
        for item in problem.accesses
        if item.enabled and item.placement == "legacy_wave_boundary"
    )
    if not tile_accesses:
        return ReuseHistogram(
            bins=tuple(),
            distance_unit=problem.l2.distance_unit,
            reuse_digest=problem.reuse_digest,
            backend=BACKEND_VERSION,
            distance_semantics="scalar_access_volume",
            reduction_fidelity=(
                "exact_no_inner_expansion"
                if problem.inner_iterations == 1
                else "sampled_inner_compat"
            ),
            # No reuse-distance probes exist, so any observation budget is
            # trivially fully observed. Boundary-only traffic is synthesized
            # separately by the compatibility evaluator.
            exact=True,
            sampled_requests=0,
            represented_requests=0,
        )

    has_l1_5 = problem.l1_5 is not None
    sm_count = traversal_sm_count(problem.traversal)
    group_count = (
        # Preserve the calibrated old model's floor-group quirk. The default
        # distinct backend uses ceil and models the physical tail group.
        max(sm_count // problem.l1_5_group_size, 1)
        if has_l1_5
        else 1
    )
    sample_plan = _sampling_plan(problem, tile_accesses)
    occurrence = {item.name: 0 for item in tile_accesses}

    # Reuse distance and capacity use schedule-visible tile/allocation units.
    # Cache lines remain solely a physical transaction-accounting property.
    l2_unit = float(problem.l2.unit_bytes)
    l1_5_unit = float(problem.l1_5.unit_bytes) if has_l1_5 else l2_unit
    footprint_l2 = {
        item.name: item.region.allocation_bytes / l2_unit for item in tile_accesses
    }
    footprint_l1_5 = {
        item.name: item.region.allocation_bytes / l1_5_unit for item in tile_accesses
    }
    prefix_l2 = {}
    prefix_l1_5 = {}
    running_l2 = 0.0
    running_l1_5 = 0.0
    for access in tile_accesses:
        prefix_l2[access.name] = running_l2
        prefix_l1_5[access.name] = running_l1_5
        running_l2 += access.repetitions * footprint_l2[access.name]
        running_l1_5 += access.repetitions * footprint_l1_5[access.name]
    total_l2 = running_l2
    total_l1_5 = running_l1_5

    clock_l2 = 0.0
    clock_l1_5 = [0.0] * group_count
    timestamp_l2: Dict[Tuple[str, Tuple[int, ...]], float] = {}
    timestamp_l1_5: List[Dict[Tuple[str, Tuple[int, ...]], float]] = [
        {} for _ in range(group_count)
    ]
    # Hot-loop keys stay primitive. Constructing/validating frozen result
    # dataclasses for every sampled access costs more than the timestamp update
    # itself; typed signatures are materialized once per merged bin below.
    mutable_bins: Dict[
        Tuple[str, str, float, float, bool, bool], _MutableBin
    ] = {}
    sampled_requests = 0
    rng = np.random.RandomState(problem.sampling.seed)

    for wave in iter_waves(problem.grid, problem.traversal):
        unique_allocations: Set[Tuple[str, Tuple[int, ...]]] = set()
        permutation = rng.permutation(len(wave))
        for scheduled_index in permutation:
            coordinate = wave[int(scheduled_index)]
            sm_index = int(scheduled_index) % sm_count
            group = min(sm_index // problem.l1_5_group_size, group_count - 1) if has_l1_5 else 0

            for access in tile_accesses:
                key = access.region.index_map.key(coordinate)
                allocation = (access.region.value, key)
                for repetition in range(access.repetitions):
                    previous_l2 = timestamp_l2.get(allocation)
                    l2_cold = previous_l2 is None
                    distance_l2 = (
                        _COLD_DISTANCE
                        if l2_cold
                        else max(clock_l2 - previous_l2, 0.0)
                    )
                    if has_l1_5:
                        previous_l1_5 = timestamp_l1_5[group].get(allocation)
                        l1_5_cold = previous_l1_5 is None
                        distance_l1_5 = (
                            _COLD_DISTANCE
                            if l1_5_cold
                            else max(clock_l1_5[group] - previous_l1_5, 0.0)
                        )
                    else:
                        l1_5_cold = False
                        distance_l1_5 = 0.0

                    index = occurrence[access.name]
                    period, offset, sample_weight, _sample_count = sample_plan[
                        access.name
                    ]
                    if index % period == offset:
                        frequency = access.frequency_weight * sample_weight
                        bin_key = (
                            access.name,
                            access.mode,
                            distance_l1_5,
                            distance_l2,
                            l1_5_cold,
                            l2_cold,
                        )
                        bin_value = mutable_bins.get(bin_key)
                        if bin_value is None:
                            bin_value = _MutableBin()
                            mutable_bins[bin_key] = bin_value
                        bin_value.frequency += frequency
                        bin_value.payload_bytes += (
                            frequency * access.region.payload_bytes
                        )
                        bin_value.transaction_bytes += (
                            frequency * access.transaction_bytes
                        )
                        sampled_requests += 1
                    occurrence[access.name] = index + 1

                    # A no-write-allocate store is conservatively treated as
                    # pressure without creating a reusable timestamp. Modeling
                    # probabilistic hit-dependent refresh requires state split.
                    refresh = (
                        access.mode == "read" or access.write_policy.allocate_on_miss
                    )
                    if refresh:
                        timestamp_l2[allocation] = (
                            clock_l2
                            + prefix_l2[access.name]
                            + (repetition + 0.5) * footprint_l2[access.name]
                        )
                        if has_l1_5:
                            timestamp_l1_5[group][allocation] = (
                                clock_l1_5[group]
                                + prefix_l1_5[access.name]
                                + (repetition + 0.5) * footprint_l1_5[access.name]
                            )
                unique_allocations.add(allocation)

            clock_l2 += total_l2
            if has_l1_5:
                clock_l1_5[group] += total_l1_5

        allocation_bytes = {
            item.region.value: item.region.allocation_bytes for item in tile_accesses
        }
        unique_l2 = sum(
            allocation_bytes[value] / l2_unit
            for value, _key in unique_allocations
        )
        unique_l1_5 = sum(
            allocation_bytes[value] / l1_5_unit
            for value, _key in unique_allocations
        )
        boundary_l2 = traversal_wave_size(problem.traversal) * sum(
            item.repetitions * item.region.allocation_bytes / l2_unit
            for item in boundary_accesses
        )
        boundary_l1_5 = traversal_wave_size(problem.traversal) * sum(
            item.repetitions * item.region.allocation_bytes / l1_5_unit
            for item in boundary_accesses
        )
        clock_l2 += boundary_l2 + (problem.inner_iterations - 1) * unique_l2
        if has_l1_5:
            aging = boundary_l1_5 + (problem.inner_iterations - 1) * unique_l1_5
            for group in range(group_count):
                clock_l1_5[group] += aging

    bins = tuple(
        HistogramBin(
            access_name=access_name,
            mode=mode,
            signature=DistanceSignature(
                l1_5_distance_units=distance_l1_5,
                l2_distance_units=distance_l2,
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
            distance_l1_5,
            distance_l2,
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
    represented = problem.grid.total_tiles * sum(
        item.repetitions for item in tile_accesses
    )
    return ReuseHistogram(
        bins=bins,
        distance_unit=problem.l2.distance_unit,
        reuse_digest=problem.reuse_digest,
        backend=BACKEND_VERSION,
        distance_semantics="scalar_access_volume",
        reduction_fidelity=(
            "exact_no_inner_expansion"
            if problem.inner_iterations == 1
            else "sampled_inner_compat"
        ),
        exact=problem.sampling.sample_budget is None
        or problem.sampling.sample_budget >= represented,
        sampled_requests=sampled_requests,
        represented_requests=represented,
    )


def _traffic_for_bin(
    access: CacheAccessIR,
    bin_item: HistogramBin,
    l1_5_probability: float,
    l2_probability: float,
) -> CacheTraffic:
    payload = bin_item.payload_bytes_weight
    transaction = bin_item.transaction_bytes_weight
    amplification = max(transaction - payload, 0.0)
    l1_5_requests = transaction if bin_item.signature.has_l1_5 else 0.0
    l2_miss_fraction = (1.0 - l1_5_probability) * (1.0 - l2_probability)
    if access.mode == "read":
        return CacheTraffic(
            payload_read_bytes=payload,
            l1_5_request_bytes=l1_5_requests,
            l2_request_bytes=transaction * (1.0 - l1_5_probability),
            ddr_read_bytes=transaction * l2_miss_fraction,
            transaction_amplification_bytes=amplification,
        )

    policy = access.write_policy
    rfo = transaction * l2_miss_fraction if policy.partial_write_rfo else 0.0
    if policy.propagation == "write_through":
        return CacheTraffic(
            payload_write_bytes=payload,
            l1_5_request_bytes=l1_5_requests,
            l2_request_bytes=transaction,
            ddr_read_bytes=rfo,
            ddr_write_bytes=transaction,
            rfo_bytes=rfo,
            transaction_amplification_bytes=amplification,
        )

    dirty = transaction
    writeback = dirty if policy.flush_at_end else 0.0
    resident = dirty - writeback
    return CacheTraffic(
        payload_write_bytes=payload,
        l1_5_request_bytes=l1_5_requests,
        l2_request_bytes=transaction,
        ddr_read_bytes=rfo,
        ddr_write_bytes=writeback,
        rfo_bytes=rfo,
        dirty_created_bytes=dirty,
        dirty_terminal_flush_bytes=writeback,
        dirty_resident_bytes=resident,
        transaction_amplification_bytes=amplification,
    )


def _boundary_traffic(problem: CacheProblem, access: CacheAccessIR) -> Tuple[float, CacheTraffic]:
    requests = float(
        problem.grid.total_tiles * access.repetitions * access.frequency_weight
    )
    signature = DistanceSignature(
        0.0,
        _COLD_DISTANCE,
        False,
        True,
        False,
        distance_unit=problem.l2.distance_unit,
    )
    item = HistogramBin(
        access_name=access.name,
        mode=access.mode,
        signature=signature,
        frequency_weight=requests,
        payload_bytes_weight=requests * access.region.payload_bytes,
        transaction_bytes_weight=requests * access.transaction_bytes,
    )
    return requests, _traffic_for_bin(access, item, 0.0, 0.0)


def evaluate_histogram(problem: CacheProblem, histogram: ReuseHistogram) -> CacheResult:
    """Evaluate one reusable histogram against the problem's cache capacities."""

    if not isinstance(problem, CacheProblem) or not isinstance(histogram, ReuseHistogram):
        raise ModelingValidationError(
            "evaluate_histogram expects CacheProblem and ReuseHistogram"
        )
    if histogram.distance_unit != problem.l2.distance_unit:
        raise ModelingValidationError(
            "histogram distance unit does not match cache capacity unit"
        )
    if histogram.reuse_digest != problem.reuse_digest:
        raise ModelingValidationError(
            "histogram does not match accesses, traversal, units, seed, or budget"
        )
    expected_reduction = (
        "exact_no_inner_expansion"
        if problem.inner_iterations == 1
        else (
            "stable_shadow_cohort"
            if problem.reduction.mode == "stable_shadow_cohort"
            else "sampled_inner_compat"
        )
    )
    if histogram.reduction_fidelity != expected_reduction:
        raise ModelingValidationError(
            "histogram reduction fidelity does not match inner_iterations"
        )
    if histogram.sampled_requests > histogram.represented_requests:
        raise ModelingValidationError(
            "histogram samples more requests than it represents"
        )
    fully_observed = (
        problem.sampling.sample_budget is None
        or problem.sampling.sample_budget >= histogram.represented_requests
    )
    expected_exact = fully_observed and (
        problem.backend == "legacy_volume" or problem.inner_iterations == 1
    )
    if histogram.exact != expected_exact:
        raise ModelingValidationError(
            "histogram exact marker disagrees with backend, sampling, or reduction fidelity"
        )
    if problem.backend == "legacy_volume":
        if (
            histogram.distance_semantics != "scalar_access_volume"
            or histogram.backend != BACKEND_VERSION
        ):
            raise ModelingValidationError(
                "legacy_volume accepts only its scalar compatibility histogram"
            )
    else:
        distinct_backends = (
            {
                "tile-reuse-stable-shadow-fenwick-v1",
                "tile-reuse-stable-shadow-range-sampled-v1",
                "tile-reuse-stable-shadow-fenwick-sampled-v1",
            }
            if problem.reduction.mode == "stable_shadow_cohort"
            and problem.inner_iterations > 1
            else {
                "tile-reuse-distinct-fenwick-v1",
                "tile-reuse-distinct-range-sampled-v1",
                "tile-reuse-distinct-fenwick-sampled-v1",
            }
        )
        if (
            histogram.distance_semantics != "distinct_tile_allocation"
            or histogram.backend not in distinct_backends
        ):
            raise ModelingValidationError(
                "tile_reuse_distance accepts only a known distinct-allocation histogram"
            )
        base_backends = {
            "tile-reuse-distinct-fenwick-v1",
            "tile-reuse-stable-shadow-fenwick-v1",
        }
        sampled_backend = histogram.backend not in base_backends
        if sampled_backend == fully_observed:
            raise ModelingValidationError(
                "distinct histogram backend label disagrees with observation sampling"
            )
    accesses = {item.name: item for item in problem.accesses if item.enabled}
    has_l1_5 = problem.l1_5 is not None
    observed_accesses = tuple(
        item
        for item in problem.accesses
        if item.enabled
        and not (
            problem.backend == "legacy_volume"
            and item.placement == "legacy_wave_boundary"
        )
    )
    expected_represented = problem.grid.total_tiles * sum(
        item.repetitions for item in observed_accesses
    )
    if histogram.represented_requests != expected_represented:
        raise ModelingValidationError(
            "histogram represented_requests disagrees with active accesses"
        )
    if observed_accesses:
        plan = _sampling_plan(problem, observed_accesses)
        expected_sampled = sum(item[3] for item in plan.values())
    else:
        expected_sampled = 0
    if histogram.sampled_requests != expected_sampled:
        raise ModelingValidationError(
            "histogram sampled_requests disagrees with the sampling plan"
        )
    bins_by_access = {name: [] for name in accesses}
    for item in histogram.bins:
        access = accesses.get(item.access_name)
        if access is None:
            raise ModelingValidationError(
                "histogram references unknown or disabled access %s" % item.access_name
            )
        if item.mode != access.mode:
            raise ModelingValidationError(
                "histogram mode disagrees with access %s" % item.access_name
            )
        if item.signature.has_l1_5 != has_l1_5:
            raise ModelingValidationError(
                "histogram L1.5 signature disagrees with cache hierarchy"
            )
        bins_by_access[item.access_name].append(item)
    for access in observed_accesses:
        items = bins_by_access[access.name]
        frequency = sum(item.frequency_weight for item in items)
        expected_frequency = (
            problem.grid.total_tiles
            * access.repetitions
            * access.frequency_weight
        )
        tolerance = 1.0e-9 * max(expected_frequency, 1.0)
        if abs(frequency - expected_frequency) > tolerance:
            raise ModelingValidationError(
                "histogram frequency does not conserve access %s" % access.name
            )
        payload = sum(item.payload_bytes_weight for item in items)
        transaction = sum(item.transaction_bytes_weight for item in items)
        if abs(payload - expected_frequency * access.region.payload_bytes) > (
            1.0e-9 * max(expected_frequency * access.region.payload_bytes, 1.0)
        ):
            raise ModelingValidationError(
                "histogram payload bytes do not conserve access %s" % access.name
            )
        if abs(transaction - expected_frequency * access.transaction_bytes) > (
            1.0e-9 * max(expected_frequency * access.transaction_bytes, 1.0)
        ):
            raise ModelingValidationError(
                "histogram transaction bytes do not conserve access %s" % access.name
            )
    if problem.backend == "legacy_volume":
        for access in problem.accesses:
            if (
                access.enabled
                and access.placement == "legacy_wave_boundary"
                and bins_by_access[access.name]
            ):
                raise ModelingValidationError(
                    "legacy boundary-only access must not appear in reuse histogram"
                )
    stats = {name: _AccessStats() for name in accesses}
    for item in histogram.bins:
        access = accesses[item.access_name]
        if item.signature.has_l1_5:
            l1_5_probability = min(
                max(
                    float(
                        sdcm(
                            item.signature.l1_5_distance_units,
                            problem.l1_5.associativity,
                            problem.l1_5.capacity_units,
                        )
                    ),
                    0.0,
                ),
                1.0,
            )
        else:
            l1_5_probability = 0.0
        l2_probability = min(
            max(
                float(
                    sdcm(
                        item.signature.l2_distance_units,
                        problem.l2.associativity,
                        problem.l2.capacity_units,
                    )
                ),
                0.0,
            ),
            1.0,
        )
        l2_served = (1.0 - l1_5_probability) * l2_probability
        ddr_miss = (1.0 - l1_5_probability) * (1.0 - l2_probability)
        target = stats[item.access_name]
        target.requests += item.frequency_weight
        target.l1_5_hits += item.frequency_weight * l1_5_probability
        target.l2_hits += item.frequency_weight * l2_served
        target.ddr_misses += item.frequency_weight * ddr_miss
        target.payload_bytes += item.payload_bytes_weight
        target.l1_5_payload_bytes += item.payload_bytes_weight * l1_5_probability
        target.l2_payload_bytes += item.payload_bytes_weight * l2_served
        target.ddr_payload_bytes += item.payload_bytes_weight * ddr_miss
        target.traffic = target.traffic.add(
            _traffic_for_bin(access, item, l1_5_probability, l2_probability)
        )

    for access in accesses.values():
        if (
            histogram.distance_semantics != "scalar_access_volume"
            or access.placement != "legacy_wave_boundary"
        ):
            continue
        requests, traffic = _boundary_traffic(problem, access)
        target = stats[access.name]
        target.requests = requests
        target.ddr_misses = requests
        target.payload_bytes = requests * access.region.payload_bytes
        target.ddr_payload_bytes = target.payload_bytes
        target.traffic = traffic

    per_access = []
    aggregate_l1_5 = 0.0
    aggregate_l2 = 0.0
    aggregate_ddr = 0.0
    total_traffic = CacheTraffic()
    diagnostics = [
        "distance backend: %s" % histogram.backend,
        "distance semantics: %s" % histogram.distance_semantics,
        "reduction fidelity: %s" % histogram.reduction_fidelity,
        "L1.5/L2 distances remain joint until SDCM evaluation",
    ]
    if (
        histogram.distance_semantics == "scalar_access_volume"
        and problem.l1_5 is not None
        and traversal_sm_count(problem.traversal) % problem.l1_5_group_size != 0
    ):
        diagnostics.append(
            "legacy compatibility quirk: floor L1.5 group count merges tail SMs"
        )
    if histogram.sampled_requests < histogram.represented_requests:
        diagnostics.append("hit rates use deterministic weighted systematic samples")
    if histogram.reduction_fidelity == "sampled_inner_compat":
        diagnostics.append(
            "inner reduction iterations use sampled anonymous distinct-allocation "
            "interference rather than an explicit reduction-key trace"
        )
    elif histogram.reduction_fidelity == "stable_shadow_cohort":
        diagnostics.append(
            "inner reduction uses a fixed representative K plus stable prefix/suffix "
            "shadow cohorts under a wave-lockstep progress assumption"
        )
        if problem.reduction.progress_jitter.enabled:
            diagnostics.append(
                "CTA progress skew is query-only deterministic antithetic jitter; "
                "cache state remains unchanged"
            )

    for access in problem.accesses:
        if not access.enabled:
            continue
        item = stats[access.name]
        requests = item.requests
        if requests > 0.0:
            l1_5_rate = item.l1_5_hits / requests
            l2_served_rate = item.l2_hits / requests
            ddr_rate = item.ddr_misses / requests
            l1_5_misses = requests - item.l1_5_hits
            l2_conditional = item.l2_hits / l1_5_misses if l1_5_misses > 0.0 else 0.0
        else:
            l1_5_rate = l2_served_rate = 0.0
            l2_conditional = 0.0
            ddr_rate = 1.0
        per_access.append(
            CacheAccessResult(
                name=access.name,
                mode=access.mode,
                request_count=requests,
                l1_5_hit_rate=l1_5_rate,
                l2_served_rate=l2_served_rate,
                l2_hit_rate_of_l1_5_misses=l2_conditional,
                ddr_miss_rate=ddr_rate,
                traffic=item.traffic,
                include_in_hit_rate=access.include_in_hit_rate,
            )
        )
        total_traffic = total_traffic.add(item.traffic)
        if access.include_in_hit_rate:
            aggregate_l1_5 += item.l1_5_payload_bytes
            aggregate_l2 += item.l2_payload_bytes
            aggregate_ddr += item.ddr_payload_bytes
        if access.mode != "read" and not access.write_policy.allocate_on_miss:
            if histogram.distance_semantics == "distinct_tile_allocation":
                diagnostics.append(
                    "%s: capacity-independent conservative NWA never inserts or "
                    "refreshes the distinct-allocation stack" % access.name
                )
            else:
                diagnostics.append(
                    "%s: legacy NWA is conservatively pressure-only on misses"
                    % access.name
                )
        if access.mode != "read" and access.write_policy.propagation == "write_back":
            diagnostics.append(
                "%s: dirty eviction writeback is not estimated; the model retains "
                "dirty bytes unless terminal flush_at_end is explicit"
                % access.name
            )

    aggregate_total = aggregate_l1_5 + aggregate_l2 + aggregate_ddr
    aggregate_l1_5_rate = aggregate_l1_5 / aggregate_total if aggregate_total else 0.0
    aggregate_l2_rate = (
        aggregate_l2 / (aggregate_l2 + aggregate_ddr)
        if aggregate_l2 + aggregate_ddr > 0.0
        else 0.0
    )
    aggregate_ddr_rate = aggregate_ddr / aggregate_total if aggregate_total else 1.0
    return CacheResult(
        problem_digest=problem.digest,
        backend=histogram.backend,
        histogram=histogram,
        per_access=tuple(per_access),
        aggregate=CacheAggregateResult(
            l1_5_hit_rate=aggregate_l1_5_rate,
            l2_hit_rate=aggregate_l2_rate,
            ddr_miss_rate=aggregate_ddr_rate,
        ),
        traffic=total_traffic,
        diagnostics=tuple(diagnostics),
    )


def model_cache(problem: CacheProblem) -> CacheResult:
    """Internal explicit compatibility entry point; public API dispatches."""

    if problem.backend != "legacy_volume":
        raise ModelingValidationError(
            "legacy_volume.model_cache cannot evaluate the default distinct backend"
        )
    return evaluate_histogram(problem, build_reuse_histogram(problem))
