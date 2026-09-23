"""Frozen, architecture-neutral IR for sampled cache reuse modeling.

The frontend records *logical* global accesses.  L1.5 and L2 are deliberately
not programmable endpoints: they are introduced by cache lowering.  The IR is
small enough to construct in a DSE inner loop and has a canonical SHA-256
digest; it never relies on Python's process-randomized ``hash()``.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, replace
from functools import reduce
from operator import mul
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

from ..errors import ModelingValidationError


CACHE_IR_VERSION = "cache-ir-v2-tile-units"


def _identifier(value: str, label: str) -> str:
    if not isinstance(value, str) or not value or "/" in value:
        raise ModelingValidationError("%s must be a non-empty identifier" % label)
    return value


def _positive_float(value: float, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ModelingValidationError("%s must be numeric" % label) from error
    if not math.isfinite(result) or result <= 0.0:
        raise ModelingValidationError("%s must be finite and positive" % label)
    return result


def _positive_int(value: int, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ModelingValidationError("%s must be a positive integer" % label)
    return value


def _integer_tuple(value: Sequence[int], label: str) -> Tuple[int, ...]:
    try:
        result = tuple(value)
    except TypeError as error:
        raise ModelingValidationError("%s must be a sequence" % label) from error
    if not result:
        raise ModelingValidationError("%s must not be empty" % label)
    if any(not isinstance(item, int) or isinstance(item, bool) for item in result):
        raise ModelingValidationError("%s must contain integers" % label)
    return result


class IndexMap:
    """Map a logical work-tile coordinate to a cache reuse identity."""

    def key(self, coordinate: Sequence[int]) -> Tuple[int, ...]:
        raise NotImplementedError

    def canonical(self) -> Dict[str, Any]:
        raise NotImplementedError


@dataclass(frozen=True)
class Projection(IndexMap):
    """Keep selected work-grid axes; projected-away axes reuse the tile."""

    axes: Tuple[int, ...]

    def __post_init__(self) -> None:
        axes = tuple(self.axes)
        if any(not isinstance(axis, int) or isinstance(axis, bool) or axis < 0 for axis in axes):
            raise ModelingValidationError("projection axes must be non-negative integers")
        if len(axes) != len(set(axes)):
            raise ModelingValidationError("projection axes must be unique")
        object.__setattr__(self, "axes", axes)

    def key(self, coordinate: Sequence[int]) -> Tuple[int, ...]:
        if self.axes and max(self.axes) >= len(coordinate):
            raise ModelingValidationError("projection axis exceeds work-grid rank")
        return tuple(int(coordinate[axis]) for axis in self.axes)

    def canonical(self) -> Dict[str, Any]:
        return {"kind": "projection", "axes": list(self.axes)}


@dataclass(frozen=True)
class AffineIndexMap(IndexMap):
    """Small integer affine map ``key = matrix @ coordinate + offset``.

    Optional positive moduli are useful for ring/paged layouts.  This is a
    tile-identity map, not an element-level address expression.
    """

    matrix: Tuple[Tuple[int, ...], ...]
    offset: Tuple[int, ...] = field(default_factory=tuple)
    modulus: Tuple[Optional[int], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        matrix = tuple(tuple(row) for row in self.matrix)
        if not matrix or not matrix[0]:
            raise ModelingValidationError("affine matrix must not be empty")
        width = len(matrix[0])
        if any(len(row) != width for row in matrix):
            raise ModelingValidationError("affine matrix rows must have equal width")
        if any(
            not isinstance(item, int) or isinstance(item, bool)
            for row in matrix
            for item in row
        ):
            raise ModelingValidationError("affine matrix must contain integers")
        offset = tuple(self.offset) if self.offset else (0,) * len(matrix)
        if len(offset) != len(matrix) or any(
            not isinstance(item, int) or isinstance(item, bool) for item in offset
        ):
            raise ModelingValidationError("affine offset must match matrix rows")
        modulus = tuple(self.modulus) if self.modulus else (None,) * len(matrix)
        if len(modulus) != len(matrix) or any(
            item is not None
            and (not isinstance(item, int) or isinstance(item, bool) or item <= 0)
            for item in modulus
        ):
            raise ModelingValidationError(
                "affine modulus must contain one positive integer or None per row"
            )
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "offset", offset)
        object.__setattr__(self, "modulus", modulus)

    def key(self, coordinate: Sequence[int]) -> Tuple[int, ...]:
        if len(coordinate) != len(self.matrix[0]):
            raise ModelingValidationError("affine map rank differs from work-grid rank")
        result = []
        for row, offset, modulus in zip(self.matrix, self.offset, self.modulus):
            item = offset + sum(coefficient * int(axis) for coefficient, axis in zip(row, coordinate))
            result.append(item if modulus is None else item % modulus)
        return tuple(result)

    def canonical(self) -> Dict[str, Any]:
        return {
            "kind": "affine",
            "matrix": [list(row) for row in self.matrix],
            "offset": list(self.offset),
            "modulus": list(self.modulus),
        }


@dataclass(frozen=True)
class TensorTileRegion:
    """One tensor tile touched by every matching logical work item."""

    value: str
    index_map: IndexMap
    tile_shape: Tuple[int, ...]
    element_bytes: float
    payload_bytes: Optional[float] = None
    allocation_bytes: Optional[float] = None

    def __post_init__(self) -> None:
        _identifier(self.value, "tensor value")
        if not isinstance(self.index_map, IndexMap):
            raise ModelingValidationError("tensor tile index_map must be IndexMap")
        shape = _integer_tuple(self.tile_shape, "tensor tile shape")
        if any(item <= 0 for item in shape):
            raise ModelingValidationError("tensor tile dimensions must be positive")
        element_bytes = _positive_float(self.element_bytes, "element_bytes")
        payload = (
            reduce(mul, shape, 1) * element_bytes
            if self.payload_bytes is None
            else _positive_float(self.payload_bytes, "payload_bytes")
        )
        allocation = (
            payload
            if self.allocation_bytes is None
            else _positive_float(self.allocation_bytes, "allocation_bytes")
        )
        if payload > allocation + 1.0e-12:
            raise ModelingValidationError(
                "payload_bytes cannot exceed its cache allocation_bytes"
            )
        object.__setattr__(self, "tile_shape", shape)
        object.__setattr__(self, "element_bytes", element_bytes)
        object.__setattr__(self, "payload_bytes", float(payload))
        object.__setattr__(self, "allocation_bytes", float(allocation))

    def canonical(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "index_map": self.index_map.canonical(),
            "tile_shape": list(self.tile_shape),
            "element_bytes": self.element_bytes,
            "payload_bytes": self.payload_bytes,
            "allocation_bytes": self.allocation_bytes,
        }


@dataclass(frozen=True)
class WritePolicy:
    """Minimum write behavior needed for traffic and pollution accounting."""

    propagation: str = "write_through"
    allocate_on_miss: bool = True
    partial_write_rfo: bool = False
    flush_at_end: bool = False

    def __post_init__(self) -> None:
        if self.propagation not in ("write_through", "write_back"):
            raise ModelingValidationError(
                "write propagation must be 'write_through' or 'write_back'"
            )
        for name in ("allocate_on_miss", "partial_write_rfo", "flush_at_end"):
            if not isinstance(getattr(self, name), bool):
                raise ModelingValidationError("%s must be boolean" % name)

    def canonical(self) -> Dict[str, Any]:
        return {
            "propagation": self.propagation,
            "allocate_on_miss": self.allocate_on_miss,
            "partial_write_rfo": self.partial_write_rfo,
            "flush_at_end": self.flush_at_end,
        }


@dataclass(frozen=True)
class CacheAccessIR:
    """A logical global read or write before passive-cache lowering.

    ``placement='tile'`` is a normal ordered access.  The
    ``legacy_wave_boundary`` placement is an explicit compatibility hook for
    the published GEMM reuse-volume model, where output stores age all input
    reuse distances between scheduling waves but are not cache-hit probes.

    ``repetitions`` means sequential real requests: each request probes and
    updates recency/pollution. ``frequency_weight`` is only a statistical
    weight attached to every such request and never creates another recency
    event. Ordinary object programs leave it at one; the legacy adapter uses
    it to preserve the old ``TensorAccess.access_count`` estimator semantics.
    """

    name: str
    region: TensorTileRegion
    mode: str
    repetitions: int = 1
    frequency_weight: float = 1.0
    transaction_bytes: Optional[float] = None
    include_in_hit_rate: bool = True
    placement: str = "tile"
    write_policy: Optional[WritePolicy] = None
    enabled: bool = True

    def __post_init__(self) -> None:
        _identifier(self.name, "cache access name")
        if not isinstance(self.region, TensorTileRegion):
            raise ModelingValidationError("cache access region must be TensorTileRegion")
        if self.mode not in ("read", "write", "atomic"):
            raise ModelingValidationError("cache access mode must be read, write, or atomic")
        _positive_int(self.repetitions, "cache access repetitions")
        object.__setattr__(
            self,
            "frequency_weight",
            _positive_float(self.frequency_weight, "cache access frequency_weight"),
        )
        transaction = (
            self.region.payload_bytes
            if self.transaction_bytes is None
            else _positive_float(self.transaction_bytes, "transaction_bytes")
        )
        if transaction + 1.0e-12 < self.region.payload_bytes:
            raise ModelingValidationError(
                "transaction_bytes cannot be smaller than logical payload_bytes"
            )
        if not isinstance(self.include_in_hit_rate, bool) or not isinstance(self.enabled, bool):
            raise ModelingValidationError("cache access flags must be boolean")
        if self.placement not in ("tile", "legacy_wave_boundary"):
            raise ModelingValidationError("unsupported cache access placement")
        if self.placement == "legacy_wave_boundary" and self.mode != "write":
            raise ModelingValidationError("legacy wave-boundary access must be a write")
        if self.mode == "read":
            if self.write_policy is not None:
                raise ModelingValidationError("read access cannot carry WritePolicy")
        elif not isinstance(self.write_policy, WritePolicy):
            raise ModelingValidationError("write/atomic access requires WritePolicy")
        object.__setattr__(self, "transaction_bytes", float(transaction))

    @classmethod
    def load(
        cls,
        name: str,
        region: TensorTileRegion,
        repetitions: int = 1,
        frequency_weight: float = 1.0,
        transaction_bytes: Optional[float] = None,
    ) -> "CacheAccessIR":
        return cls(
            name=name,
            region=region,
            mode="read",
            repetitions=repetitions,
            frequency_weight=frequency_weight,
            transaction_bytes=transaction_bytes,
        )

    @classmethod
    def store(
        cls,
        name: str,
        region: TensorTileRegion,
        repetitions: int = 1,
        frequency_weight: float = 1.0,
        transaction_bytes: Optional[float] = None,
        write_policy: WritePolicy = WritePolicy(),
        include_in_hit_rate: bool = False,
        placement: str = "tile",
    ) -> "CacheAccessIR":
        return cls(
            name=name,
            region=region,
            mode="write",
            repetitions=repetitions,
            frequency_weight=frequency_weight,
            transaction_bytes=transaction_bytes,
            include_in_hit_rate=include_in_hit_rate,
            placement=placement,
            write_policy=write_policy,
        )

    def canonical(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "region": self.region.canonical(),
            "mode": self.mode,
            "repetitions": self.repetitions,
            "frequency_weight": self.frequency_weight,
            "transaction_bytes": self.transaction_bytes,
            "include_in_hit_rate": self.include_in_hit_rate,
            "placement": self.placement,
            "write_policy": None
            if self.write_policy is None
            else self.write_policy.canonical(),
            "enabled": self.enabled,
        }


@dataclass(frozen=True)
class TileGrid:
    shape: Tuple[int, ...]
    axes: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        shape = _integer_tuple(self.shape, "tile-grid shape")
        if any(item <= 0 for item in shape):
            raise ModelingValidationError("tile-grid dimensions must be positive")
        axes = tuple(self.axes) if self.axes else tuple("axis%d" % i for i in range(len(shape)))
        if len(axes) != len(shape):
            raise ModelingValidationError("tile-grid axes must match shape rank")
        for axis in axes:
            _identifier(axis, "tile-grid axis")
        if len(axes) != len(set(axes)):
            raise ModelingValidationError("tile-grid axes must be unique")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "axes", axes)

    @property
    def total_tiles(self) -> int:
        return reduce(mul, self.shape, 1)

    def canonical(self) -> Dict[str, Any]:
        return {"shape": list(self.shape), "axes": list(self.axes)}


class TraversalIR:
    def canonical(self) -> Dict[str, Any]:
        raise NotImplementedError


@dataclass(frozen=True)
class PanelTraversal(TraversalIR):
    """2-D panel traversal compatible with TileSight's GEMM scheduler."""

    row_panel: int
    sm_count: int
    column_panel: Optional[int] = None
    raster_axis: str = "legacy"

    def __post_init__(self) -> None:
        _positive_int(self.row_panel, "row_panel")
        _positive_int(self.sm_count, "sm_count")
        if self.column_panel is not None:
            _positive_int(self.column_panel, "column_panel")
        if self.raster_axis not in ("legacy", "along_m", "along_n"):
            raise ModelingValidationError("unsupported panel raster_axis")

    @property
    def stride_m(self) -> int:
        return (
            self.column_panel
            if self.column_panel is not None
            else max(self.sm_count // self.row_panel, 1)
        )

    @property
    def wave_size(self) -> int:
        return self.stride_m * self.row_panel

    def canonical(self) -> Dict[str, Any]:
        return {
            "kind": "panel",
            "row_panel": self.row_panel,
            "sm_count": self.sm_count,
            "column_panel": self.column_panel,
            "raster_axis": self.raster_axis,
        }


@dataclass(frozen=True)
class RowMajorTraversal(TraversalIR):
    wave_size: int
    sm_count: int

    def __post_init__(self) -> None:
        _positive_int(self.wave_size, "wave_size")
        _positive_int(self.sm_count, "sm_count")

    def canonical(self) -> Dict[str, Any]:
        return {
            "kind": "row_major",
            "wave_size": self.wave_size,
            "sm_count": self.sm_count,
        }


@dataclass(frozen=True)
class ExplicitTraversal(TraversalIR):
    """A materialised launch order (any block mapper) cut into waves of ``wave_size``."""

    coordinates: Tuple[Tuple[int, ...], ...]
    wave_size: int
    sm_count: int
    label: str = "explicit"

    def __post_init__(self) -> None:
        _positive_int(self.wave_size, "wave_size")
        _positive_int(self.sm_count, "sm_count")
        coordinates = tuple(tuple(int(v) for v in item) for item in self.coordinates)
        if not coordinates:
            raise ModelingValidationError("explicit traversal needs at least one coordinate")
        object.__setattr__(self, "coordinates", coordinates)

    def canonical(self) -> Dict[str, Any]:
        payload = json.dumps(self.coordinates, separators=(",", ":")).encode("utf-8")
        return {
            "kind": "explicit",
            "label": self.label,
            "wave_size": self.wave_size,
            "sm_count": self.sm_count,
            "count": len(self.coordinates),
            "order_sha256": hashlib.sha256(payload).hexdigest(),
        }


@dataclass(frozen=True)
class CacheLevelConfig:
    name: str
    capacity_bytes: float
    unit_bytes: float
    associativity: int = 8
    cacheline_bytes: int = 128
    distance_unit: str = "tile_allocation"

    def __post_init__(self) -> None:
        if self.name not in ("l1_5", "l2"):
            raise ModelingValidationError("cache level name must be l1_5 or l2")
        object.__setattr__(
            self, "capacity_bytes", _positive_float(self.capacity_bytes, "cache capacity")
        )
        object.__setattr__(
            self, "unit_bytes", _positive_float(self.unit_bytes, "cache reuse unit_bytes")
        )
        _positive_int(self.associativity, "cache associativity")
        _positive_int(self.cacheline_bytes, "cacheline_bytes")
        if self.distance_unit not in ("tile_allocation", "legacy_cacheline_volume"):
            raise ModelingValidationError(
                "distance_unit must be tile_allocation or legacy_cacheline_volume"
            )
        if (
            self.distance_unit == "legacy_cacheline_volume"
            and abs(self.unit_bytes - self.cacheline_bytes) > 1.0e-12
        ):
            raise ModelingValidationError(
                "legacy_cacheline_volume requires unit_bytes == cacheline_bytes"
            )
        if self.capacity_units <= self.associativity:
            raise ModelingValidationError(
                "cache capacity_units must exceed associativity for SDCM"
            )

    @property
    def capacity_units(self) -> float:
        """Capacity in the same tile/allocation units as reuse distance."""

        return self.capacity_bytes / self.unit_bytes

    def canonical(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "capacity_bytes": self.capacity_bytes,
            "unit_bytes": self.unit_bytes,
            "associativity": self.associativity,
            "cacheline_bytes": self.cacheline_bytes,
            "distance_unit": self.distance_unit,
        }


@dataclass(frozen=True)
class SamplingConfig:
    """Deterministic traversal randomization and optional systematic sampling."""

    seed: int = 0
    sample_budget: Optional[int] = None

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ModelingValidationError("sampling seed must be an integer")
        if self.seed < 0 or self.seed >= 2 ** 32:
            raise ModelingValidationError("sampling seed must fit NumPy's uint32 range")
        if self.sample_budget is not None:
            _positive_int(self.sample_budget, "sample_budget")

    def canonical(self) -> Dict[str, Any]:
        return {"seed": self.seed, "sample_budget": self.sample_budget}


@dataclass(frozen=True)
class ProgressJitterConfig:
    """Query-only antithetic perturbation for bounded CTA K-progress skew.

    The perturbation never mutates cache state.  Its maximum distance at each
    level is the smaller of ``max_k_ahead`` per-K access footprints and
    ``capacity_cap_fraction`` of that level's capacity.
    """

    max_k_ahead: float = 0.0
    capacity_cap_fraction: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        max_k_ahead = float(self.max_k_ahead)
        cap = float(self.capacity_cap_fraction)
        if not math.isfinite(max_k_ahead) or max_k_ahead < 0.0:
            raise ModelingValidationError("max_k_ahead must be finite and non-negative")
        if not math.isfinite(cap) or cap < 0.0 or cap > 1.0:
            raise ModelingValidationError(
                "capacity_cap_fraction must be between zero and one"
            )
        if (max_k_ahead == 0.0) != (cap == 0.0):
            raise ModelingValidationError(
                "progress jitter needs both max_k_ahead and capacity cap, or neither"
            )
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ModelingValidationError("progress jitter seed must be an integer")
        if self.seed < 0 or self.seed >= 2 ** 32:
            raise ModelingValidationError("progress jitter seed must fit uint32")
        object.__setattr__(self, "max_k_ahead", max_k_ahead)
        object.__setattr__(self, "capacity_cap_fraction", cap)

    @property
    def enabled(self) -> bool:
        return self.max_k_ahead > 0.0

    def canonical(self) -> Dict[str, Any]:
        return {
            "max_k_ahead": self.max_k_ahead,
            "capacity_cap_fraction": self.capacity_cap_fraction,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class ReductionConfig:
    """Reduction-axis approximation selected by the cache frontend."""

    mode: str = "anonymous_inner"
    representative_k: int = 0
    progress_jitter: ProgressJitterConfig = ProgressJitterConfig()

    def __post_init__(self) -> None:
        if self.mode not in ("anonymous_inner", "stable_shadow_cohort"):
            raise ModelingValidationError(
                "reduction mode must be anonymous_inner or stable_shadow_cohort"
            )
        if not isinstance(self.representative_k, int) or isinstance(
            self.representative_k, bool
        ) or self.representative_k < 0:
            raise ModelingValidationError("representative_k must be non-negative integer")
        if not isinstance(self.progress_jitter, ProgressJitterConfig):
            raise ModelingValidationError(
                "progress_jitter must be ProgressJitterConfig"
            )
        if self.mode == "anonymous_inner" and (
            self.representative_k != 0 or self.progress_jitter.enabled
        ):
            raise ModelingValidationError(
                "anonymous_inner does not accept representative K or progress jitter"
            )

    def canonical(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "representative_k": self.representative_k,
            "progress_jitter": self.progress_jitter.canonical(),
        }


@dataclass(frozen=True)
class CacheProblem:
    """Complete input to a cache reuse backend."""

    grid: TileGrid
    accesses: Tuple[CacheAccessIR, ...]
    traversal: TraversalIR
    l2: CacheLevelConfig
    l1_5: Optional[CacheLevelConfig] = None
    l1_5_group_size: int = 0
    inner_iterations: int = 1
    reduction: ReductionConfig = ReductionConfig()
    sampling: SamplingConfig = SamplingConfig()
    backend: str = "tile_reuse_distance"

    def __post_init__(self) -> None:
        if not isinstance(self.grid, TileGrid):
            raise ModelingValidationError("cache problem grid must be TileGrid")
        accesses = tuple(self.accesses)
        if not all(isinstance(item, CacheAccessIR) for item in accesses):
            raise ModelingValidationError("cache problem accesses must be CacheAccessIR")
        names = [item.name for item in accesses]
        if len(names) != len(set(names)):
            raise ModelingValidationError("cache access names must be unique")
        # All accesses to one logical value must agree on allocation identity
        # and footprint. Payload/transaction bytes may differ (partial access
        # and transaction amplification), but cache recency is shared.
        value_specs = {}
        for access in accesses:
            value = access.region.value
            spec = (
                access.region.index_map.canonical(),
                access.region.allocation_bytes,
            )
            previous = value_specs.get(value)
            if previous is not None and previous != spec:
                raise ModelingValidationError(
                    "cache accesses to value %s disagree on index map or allocation_bytes"
                    % value
                )
            value_specs[value] = spec
        if not isinstance(self.traversal, TraversalIR):
            raise ModelingValidationError("cache problem traversal must be TraversalIR")
        if not isinstance(self.l2, CacheLevelConfig) or self.l2.name != "l2":
            raise ModelingValidationError("cache problem needs an L2 config")
        if self.l1_5 is not None and (
            not isinstance(self.l1_5, CacheLevelConfig) or self.l1_5.name != "l1_5"
        ):
            raise ModelingValidationError("l1_5 must be an L1.5 config or None")
        if self.l1_5 is None:
            if self.l1_5_group_size != 0:
                raise ModelingValidationError("l1_5_group_size must be zero without L1.5")
        else:
            _positive_int(self.l1_5_group_size, "l1_5_group_size")
        _positive_int(self.inner_iterations, "inner_iterations")
        if not isinstance(self.reduction, ReductionConfig):
            raise ModelingValidationError("cache problem reduction must be ReductionConfig")
        if self.reduction.representative_k >= self.inner_iterations:
            raise ModelingValidationError(
                "representative_k must be smaller than inner_iterations"
            )
        if self.reduction.progress_jitter.enabled and self.inner_iterations <= 1:
            raise ModelingValidationError(
                "progress jitter requires more than one inner iteration"
            )
        if not isinstance(self.sampling, SamplingConfig):
            raise ModelingValidationError("cache problem sampling must be SamplingConfig")
        if self.backend not in ("tile_reuse_distance", "legacy_volume"):
            raise ModelingValidationError(
                "cache API supports tile_reuse_distance or explicit legacy_volume"
            )
        expected_unit = (
            "tile_allocation"
            if self.backend == "tile_reuse_distance"
            else "legacy_cacheline_volume"
        )
        if self.l2.distance_unit != expected_unit:
            raise ModelingValidationError(
                "backend %s requires distance_unit=%s"
                % (self.backend, expected_unit)
            )
        if self.backend == "legacy_volume" and self.reduction.mode != "anonymous_inner":
            raise ModelingValidationError(
                "legacy_volume supports only its anonymous_inner compatibility mode"
            )
        if self.l1_5 is not None and self.l1_5.distance_unit != self.l2.distance_unit:
            raise ModelingValidationError(
                "all cache levels must use the same reuse-distance unit kind"
            )
        rank = len(self.grid.shape)
        for access in accesses:
            # Fail early instead of waiting for a sampled coordinate.
            access.region.index_map.key((0,) * rank)
        if isinstance(self.traversal, ExplicitTraversal):
            expected = reduce(mul, self.grid.shape, 1)
            seen = set(self.traversal.coordinates)
            if (len(self.traversal.coordinates) != expected or len(seen) != expected
                    or any(len(c) != rank or any(not 0 <= v < e for v, e in zip(c, self.grid.shape)) for c in seen)):
                raise ModelingValidationError("explicit traversal must visit every grid tile exactly once")
        if isinstance(self.traversal, PanelTraversal) and rank != 2:
            raise ModelingValidationError("panel traversal requires a rank-2 tile grid")
        object.__setattr__(self, "accesses", accesses)

    def eliminate_accesses(self, names: Iterable[str]) -> "CacheProblem":
        """Return a transformed problem with exact logical accesses removed.

        Fusion must call this transformation *before* reuse analysis so removed
        materializations neither generate traffic nor pollute cache clocks.
        """

        requested = tuple(names)
        if len(requested) != len(set(requested)):
            raise ModelingValidationError("fusion elimination contains duplicate access names")
        available = {item.name for item in self.accesses}
        missing = sorted(set(requested) - available)
        if missing:
            raise ModelingValidationError(
                "fusion elimination references unknown cache accesses: %s"
                % ", ".join(missing)
            )
        removed = set(requested)
        remaining = tuple(item for item in self.accesses if item.name not in removed)
        return replace(self, accesses=remaining)

    def canonical(self) -> Dict[str, Any]:
        return {
            "version": CACHE_IR_VERSION,
            "backend": self.backend,
            "grid": self.grid.canonical(),
            "accesses": [item.canonical() for item in self.accesses],
            "traversal": self.traversal.canonical(),
            "l2": self.l2.canonical(),
            "l1_5": None if self.l1_5 is None else self.l1_5.canonical(),
            "l1_5_group_size": self.l1_5_group_size,
            "inner_iterations": self.inner_iterations,
            "reduction": self.reduction.canonical(),
            "sampling": self.sampling.canonical(),
        }

    def reuse_canonical(self) -> Dict[str, Any]:
        """Canonical key for capacity-independent histogram construction."""

        value = self.canonical()
        value["l2"] = {
            "name": self.l2.name,
            "unit_bytes": self.l2.unit_bytes,
            "distance_unit": self.l2.distance_unit,
        }
        value["l1_5"] = (
            None
            if self.l1_5 is None
            else {
                "name": self.l1_5.name,
                "unit_bytes": self.l1_5.unit_bytes,
                "distance_unit": self.l1_5.distance_unit,
            }
        )
        return value

    @property
    def digest(self) -> str:
        payload = json.dumps(
            self.canonical(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @property
    def reuse_digest(self) -> str:
        """Stable key for reusing one histogram across cache-capacity sweeps."""

        payload = json.dumps(
            self.reuse_canonical(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
