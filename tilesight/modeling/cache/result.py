"""Typed reuse histogram and cache-model results."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Tuple

from ..errors import ModelingValidationError


def _nonnegative(value: float, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ModelingValidationError("%s must be finite and non-negative" % label)
    return result


def _rate(value: float, label: str) -> float:
    result = _nonnegative(value, label)
    if result > 1.0 + 1.0e-12:
        raise ModelingValidationError("%s must not exceed one" % label)
    return min(result, 1.0)


@dataclass(frozen=True)
class DistanceSignature:
    """Joint reuse distance observed by every passive cache level.

    L2 distance is global.  L1.5 distance is local to the physical SM group;
    ``None`` means the modeled architecture has no L1.5.  Coldness is retained
    separately rather than inferred from a magic distance threshold.
    """

    l1_5_distance_units: float
    l2_distance_units: float
    l1_5_cold: bool
    l2_cold: bool
    has_l1_5: bool
    distance_unit: str = "tile_allocation"

    def __post_init__(self) -> None:
        if self.has_l1_5:
            _nonnegative(self.l1_5_distance_units, "L1.5 reuse distance")
        elif self.l1_5_distance_units != 0.0 or self.l1_5_cold:
            raise ModelingValidationError(
                "L1.5-free signature must use zero distance and non-cold marker"
            )
        _nonnegative(self.l2_distance_units, "L2 reuse distance")
        if not isinstance(self.l1_5_cold, bool) or not isinstance(self.l2_cold, bool):
            raise ModelingValidationError("distance cold markers must be boolean")
        if self.distance_unit not in ("tile_allocation", "legacy_cacheline_volume"):
            raise ModelingValidationError("distance signature has an unknown unit")


@dataclass(frozen=True)
class WeightedAccessSample:
    """One sampled signature representing one or more structural accesses."""

    access_name: str
    mode: str
    signature: DistanceSignature
    frequency_weight: float
    payload_bytes_weight: float
    transaction_bytes_weight: float

    def __post_init__(self) -> None:
        if not self.access_name:
            raise ModelingValidationError("weighted sample needs an access name")
        if self.mode not in ("read", "write", "atomic"):
            raise ModelingValidationError("weighted sample has invalid access mode")
        if not isinstance(self.signature, DistanceSignature):
            raise ModelingValidationError("weighted sample needs a distance signature")
        if _nonnegative(self.frequency_weight, "frequency_weight") <= 0.0:
            raise ModelingValidationError("frequency_weight must be positive")
        if _nonnegative(self.payload_bytes_weight, "payload_bytes_weight") <= 0.0:
            raise ModelingValidationError("payload_bytes_weight must be positive")
        if _nonnegative(self.transaction_bytes_weight, "transaction_bytes_weight") <= 0.0:
            raise ModelingValidationError("transaction_bytes_weight must be positive")


@dataclass(frozen=True)
class HistogramBin(WeightedAccessSample):
    """Merged samples sharing access identity and a joint distance signature."""


@dataclass(frozen=True)
class ReuseHistogram:
    bins: Tuple[HistogramBin, ...]
    distance_unit: str
    reuse_digest: str
    backend: str
    distance_semantics: str
    reduction_fidelity: str
    exact: bool
    sampled_requests: int
    represented_requests: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "bins", tuple(self.bins))
        if self.distance_unit not in ("tile_allocation", "legacy_cacheline_volume"):
            raise ModelingValidationError("histogram has an unknown distance unit")
        if any(item.signature.distance_unit != self.distance_unit for item in self.bins):
            raise ModelingValidationError("histogram mixes reuse-distance units")
        if len(self.reuse_digest) != 64:
            raise ModelingValidationError("histogram needs a SHA-256 reuse digest")
        if self.distance_semantics not in (
            "distinct_tile_allocation",
            "scalar_access_volume",
        ):
            raise ModelingValidationError("histogram has unknown distance semantics")
        if self.reduction_fidelity not in (
            "exact_no_inner_expansion",
            "sampled_inner_compat",
            "stable_shadow_cohort",
        ):
            raise ModelingValidationError("histogram has unknown reduction fidelity")
        if not isinstance(self.backend, str) or not self.backend:
            raise ModelingValidationError("histogram needs backend provenance")
        if not isinstance(self.exact, bool):
            raise ModelingValidationError("histogram exact marker must be boolean")
        for name in ("sampled_requests", "represented_requests"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ModelingValidationError("%s must be a non-negative integer" % name)


@dataclass(frozen=True)
class CacheTraffic:
    """Expected physical traffic in bytes over the modeled logical grid."""

    payload_read_bytes: float = 0.0
    payload_write_bytes: float = 0.0
    l1_5_request_bytes: float = 0.0
    l2_request_bytes: float = 0.0
    ddr_read_bytes: float = 0.0
    ddr_write_bytes: float = 0.0
    rfo_bytes: float = 0.0
    dirty_created_bytes: float = 0.0
    dirty_eviction_writeback_bytes: float = 0.0
    dirty_terminal_flush_bytes: float = 0.0
    dirty_resident_bytes: float = 0.0
    transaction_amplification_bytes: float = 0.0

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name))
        dirty_error = abs(
            self.dirty_created_bytes
            - self.dirty_eviction_writeback_bytes
            - self.dirty_terminal_flush_bytes
            - self.dirty_resident_bytes
        )
        tolerance = 1.0e-9 * max(self.dirty_created_bytes, 1.0)
        if dirty_error > tolerance:
            raise ModelingValidationError(
                "dirty traffic violates created = writeback + resident conservation"
            )

    @property
    def dirty_writeback_bytes(self) -> float:
        """Compatibility total; components distinguish eviction from flush."""

        return self.dirty_eviction_writeback_bytes + self.dirty_terminal_flush_bytes

    def add(self, other: "CacheTraffic") -> "CacheTraffic":
        values = {
            name: getattr(self, name) + getattr(other, name)
            for name in self.__dataclass_fields__
        }
        return CacheTraffic(**values)


@dataclass(frozen=True)
class CacheAccessResult:
    name: str
    mode: str
    request_count: float
    l1_5_hit_rate: float
    l2_served_rate: float
    l2_hit_rate_of_l1_5_misses: float
    ddr_miss_rate: float
    traffic: CacheTraffic
    include_in_hit_rate: bool

    def __post_init__(self) -> None:
        _nonnegative(self.request_count, "request_count")
        for name in (
            "l1_5_hit_rate",
            "l2_served_rate",
            "l2_hit_rate_of_l1_5_misses",
            "ddr_miss_rate",
        ):
            object.__setattr__(self, name, _rate(getattr(self, name), name))
        if abs(
            self.l1_5_hit_rate + self.l2_served_rate + self.ddr_miss_rate - 1.0
        ) > 1.0e-9 and self.request_count > 0:
            raise ModelingValidationError("cache service rates do not sum to one")


@dataclass(frozen=True)
class CacheAggregateResult:
    l1_5_hit_rate: float
    l2_hit_rate: float
    ddr_miss_rate: float

    def __post_init__(self) -> None:
        for name in ("l1_5_hit_rate", "l2_hit_rate", "ddr_miss_rate"):
            object.__setattr__(self, name, _rate(getattr(self, name), name))


@dataclass(frozen=True)
class CacheResult:
    problem_digest: str
    backend: str
    histogram: ReuseHistogram
    per_access: Tuple[CacheAccessResult, ...]
    aggregate: CacheAggregateResult
    traffic: CacheTraffic
    diagnostics: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if len(self.problem_digest) != 64:
            raise ModelingValidationError("cache result needs a SHA-256 problem digest")
        object.__setattr__(self, "per_access", tuple(self.per_access))
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))

    def by_name(self, name: str) -> CacheAccessResult:
        matches = [item for item in self.per_access if item.name == name]
        if len(matches) != 1:
            raise KeyError(name)
        return matches[0]

    def to_legacy_dict(self) -> Dict[str, object]:
        """Return the published multilevel model's historical result shape."""

        per_tensor = {}
        for item in self.per_access:
            if not item.include_in_hit_rate:
                continue
            per_tensor[item.name] = {
                "l1_5_hit": item.l1_5_hit_rate,
                # Historical per-tensor L2 hit is unconditional.
                "l2_hit": item.l2_served_rate,
                "ddr_miss": item.ddr_miss_rate,
            }
        return {
            "per_tensor": per_tensor,
            "l1_5_hit_rate": self.aggregate.l1_5_hit_rate,
            "l2_hit_rate": self.aggregate.l2_hit_rate,
            "ddr_miss_rate": self.aggregate.ddr_miss_rate,
        }
