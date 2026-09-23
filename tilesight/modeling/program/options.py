"""Typed analysis options, calibration context, and latency tables."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

from tilesight.modeling._pipeline.periodic_schedule import SearchConfig

from ..errors import ModelingValidationError
from .traffic import Traffic
from ..ops.throughput import ThroughputFallbackPolicy
from .contract import ContractError, _ident, _nonneg_float, _pairs, _pos_float, _pos_int

II_MODE_CHOICES = ("resource_ii", "periodic_best", "periodic_worst")
CACHE_MODES = ("none", "fast", "trace")
CACHE_BOUND_POLICIES = ("conservative", "scenarios")
RESIDENT_SHARING = ("equal_share", "schedule")
BOUNDARY_POLICIES = ("periodic_witness", "finite_witness")
TAIL_POLICIES = ("full_wave_costs", "tail_active_sms")
MISSING_TRAFFIC_POLICIES = ("error", "assume_miss")
CACHE_STATES = ("cold",)
LATENCY_UNITS = ("s", "ns", "us", "cycle")
CLOCK_DOMAINS = ("sm", "l2", "ddr")
MEASURED_QUANTITIES = ("extra_latency", "completion_latency")


@dataclass(frozen=True)
class Options:
    """Modeling policy for one ``analyze`` call.

    ``ii_mode`` selects the steady-state statement; ``cache`` selects how
    global-access traffic is produced; ``cache_bound_policy`` picks how an
    interval-valued traffic becomes the scalar used for pricing.
    """

    ii_mode: str = "resource_ii"
    cache: str = "fast"
    cache_bound_policy: str = "conservative"
    resident_sharing: str = "equal_share"
    boundary_policy: str = "periodic_witness"
    tail_policy: str = "full_wave_costs"
    missing_traffic_policy: str = "error"
    unroll_threshold: int = 4
    max_finite_iterations: int = 4096
    kernel_launch_s: float = 2.0e-6
    host_dispatch_s: float = 0.0
    cache_sample_budget: Optional[int] = 512
    cache_seed: int = 0
    cache_representative_k: int = 0
    cache_reuse_unit_bytes: Optional[float] = None
    search_config: SearchConfig = field(default_factory=SearchConfig)
    capacity_policy: str = "error"      # declared SMEM/TMEM above the per-SM capacity: "error" or "report"
    periodic_topology_trials: int = 0   # extra seeded topological-order proposals for periodic_best/worst
    periodic_topology_seed: int = 0
    throughput_fallback: ThroughputFallbackPolicy = field(
        default_factory=lambda: ThroughputFallbackPolicy(
            dtype_aliases=(("bf16", "fp16"),), allow_dtype_agnostic_sfu=True,
        )
    )

    def __post_init__(self) -> None:
        path = "Options"
        if not isinstance(self.throughput_fallback, ThroughputFallbackPolicy):
            raise ContractError("%s.throughput_fallback: must be ThroughputFallbackPolicy" % path)
        if self.ii_mode not in II_MODE_CHOICES:
            raise ContractError("%s.ii_mode=%r: must be one of %s"
                                % (path, self.ii_mode, ", ".join(II_MODE_CHOICES)))
        if self.cache not in CACHE_MODES:
            raise ContractError("%s.cache=%r: must be one of %s"
                                % (path, self.cache, ", ".join(CACHE_MODES)))
        if self.cache_bound_policy not in CACHE_BOUND_POLICIES:
            raise ContractError("%s.cache_bound_policy=%r: must be one of %s"
                                % (path, self.cache_bound_policy, ", ".join(CACHE_BOUND_POLICIES)))
        if self.resident_sharing not in RESIDENT_SHARING:
            raise ContractError("%s.resident_sharing=%r: must be one of %s"
                                % (path, self.resident_sharing, ", ".join(RESIDENT_SHARING)))
        if self.boundary_policy not in BOUNDARY_POLICIES:
            raise ContractError("%s.boundary_policy=%r: must be one of %s"
                                % (path, self.boundary_policy, ", ".join(BOUNDARY_POLICIES)))
        if self.tail_policy not in TAIL_POLICIES:
            raise ContractError("%s.tail_policy=%r: must be one of %s"
                                % (path, self.tail_policy, ", ".join(TAIL_POLICIES)))
        if self.missing_traffic_policy not in MISSING_TRAFFIC_POLICIES:
            raise ContractError("%s.missing_traffic_policy=%r: must be one of %s"
                                % (path, self.missing_traffic_policy,
                                   ", ".join(MISSING_TRAFFIC_POLICIES)))
        _pos_int(self.unroll_threshold + 1, path, "unroll_threshold")
        _pos_int(self.max_finite_iterations, path, "max_finite_iterations")
        object.__setattr__(self, "kernel_launch_s", _nonneg_float(self.kernel_launch_s, path, "kernel_launch_s"))
        object.__setattr__(self, "host_dispatch_s", _nonneg_float(self.host_dispatch_s, path, "host_dispatch_s"))
        if self.cache_sample_budget is not None:
            _pos_int(self.cache_sample_budget, path, "cache_sample_budget")
        if not isinstance(self.cache_seed, int) or self.cache_seed < 0:
            raise ContractError("%s.cache_seed=%r: must be a non-negative integer" % (path, self.cache_seed))
        _pos_int(self.cache_representative_k + 1, path, "cache_representative_k")
        if self.cache_reuse_unit_bytes is not None:
            object.__setattr__(self, "cache_reuse_unit_bytes",
                               _pos_float(self.cache_reuse_unit_bytes, path, "cache_reuse_unit_bytes"))
        if self.capacity_policy not in ("error", "report"):
            raise ContractError("%s.capacity_policy=%r: must be 'error' or 'report'" % (path, self.capacity_policy))
        for name in ("periodic_topology_trials", "periodic_topology_seed"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ContractError("%s.%s=%r: must be a non-negative int" % (path, name, value))
        if not isinstance(self.search_config, SearchConfig):
            raise ContractError("%s.search_config: must be SearchConfig" % path)

    def canonical(self) -> Dict[str, Any]:
        return {
            "ii_mode": self.ii_mode,
            "cache": self.cache,
            "cache_bound_policy": self.cache_bound_policy,
            "resident_sharing": self.resident_sharing,
            "boundary_policy": self.boundary_policy,
            "tail_policy": self.tail_policy,
            "capacity_policy": self.capacity_policy,
            "periodic_topology_trials": self.periodic_topology_trials,
            "periodic_topology_seed": self.periodic_topology_seed,
            "missing_traffic_policy": self.missing_traffic_policy,
            "unroll_threshold": self.unroll_threshold,
            "max_finite_iterations": self.max_finite_iterations,
            "kernel_launch_s": self.kernel_launch_s,
            "host_dispatch_s": self.host_dispatch_s,
            "cache_sample_budget": self.cache_sample_budget,
            "cache_seed": self.cache_seed,
            "cache_representative_k": self.cache_representative_k,
            "cache_reuse_unit_bytes": self.cache_reuse_unit_bytes,
            "throughput_fallback": {
                "dtype_aliases": list(self.throughput_fallback.dtype_aliases),
                "explicit_capacities": list(self.throughput_fallback.explicit_capacities),
                "allow_dtype_agnostic_sfu": self.throughput_fallback.allow_dtype_agnostic_sfu,
            },
        }


@dataclass(frozen=True)
class LatencyEntry:
    """One architecture latency table row.

    Wildcard ``"*"`` matches any value.  ``latency`` is expressed in ``unit``;
    ``cycle`` values are converted with the clock of ``clock_domain``.
    """

    arch: str
    kind: str  # load/store/compute/sync
    engine: str = "*"
    source: str = "*"
    destination: str = "*"
    op_class: str = "*"
    dtype: str = "*"
    size_class: str = "*"
    latency: float = 0.0
    unit: str = "s"
    clock_domain: str = "sm"
    measured_quantity: str = "extra_latency"
    condition: str = ""
    source_file: str = ""

    def __post_init__(self) -> None:
        path = "LatencyEntry"
        if self.unit not in LATENCY_UNITS:
            raise ContractError("%s.unit=%r: must be one of %s" % (path, self.unit, ", ".join(LATENCY_UNITS)))
        if self.clock_domain not in CLOCK_DOMAINS:
            raise ContractError("%s.clock_domain=%r: must be one of %s" % (path, self.clock_domain, ", ".join(CLOCK_DOMAINS)))
        object.__setattr__(self, "latency", _nonneg_float(self.latency, path, "latency"))
        if self.measured_quantity not in MEASURED_QUANTITIES:
            raise ContractError(
                "%s.measured_quantity=%r: must be one of %s (a full completion latency is "
                "converted to extra latency by subtracting bound service)"
                % (path, self.measured_quantity, ", ".join(MEASURED_QUANTITIES))
            )
        for name in ("arch", "kind", "engine", "source", "destination", "op_class", "dtype", "size_class"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ContractError("%s.%s=%r: must be a non-empty string" % (path, name, value))

    def matches(self, key: Mapping[str, str]) -> int:
        """Return specificity (matched non-wildcard fields) or -1."""

        score = 0
        for name in ("arch", "kind", "engine", "source", "destination", "op_class", "dtype", "size_class"):
            mine = getattr(self, name)
            theirs = key.get(name, "*")
            if mine == "*":
                continue
            if mine != theirs:
                return -1
            score += 1
        return score

    def seconds(self, clocks_hz: Mapping[str, float]) -> float:
        if self.unit == "s":
            return self.latency
        if self.unit == "ns":
            return self.latency * 1e-9
        if self.unit == "us":
            return self.latency * 1e-6
        clock = clocks_hz.get(self.clock_domain)
        if clock is None or clock <= 0.0:
            raise ContractError(
                "LatencyEntry: cycle latency needs clock domain %r frequency" % self.clock_domain
            )
        return self.latency / clock

    def canonical(self) -> Dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class LatencyTable:
    entries: Tuple[LatencyEntry, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        entries = tuple(self.entries)
        if not all(isinstance(item, LatencyEntry) for item in entries):
            raise ContractError("LatencyTable.entries: must contain LatencyEntry")
        object.__setattr__(self, "entries", entries)

    @classmethod
    def from_csv(cls, path: str) -> "LatencyTable":
        entries = []
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                fields = {}
                for name in LatencyEntry.__dataclass_fields__:
                    if name in row and row[name] != "":
                        fields[name] = row[name]
                fields["latency"] = float(fields.get("latency", 0.0))
                fields.setdefault("source_file", path)
                entries.append(LatencyEntry(**fields))
        return cls(tuple(entries))

    def lookup(self, key: Mapping[str, str], clocks_hz: Mapping[str, float]) -> Tuple[float, Optional[LatencyEntry]]:
        best = None
        best_score = -1
        for entry in self.entries:
            score = entry.matches(key)
            if score > best_score:
                best, best_score = entry, score
        if best is None:
            return 0.0, None
        return best.seconds(clocks_hz), best

    @property
    def digest(self) -> str:
        payload = json.dumps([e.canonical() for e in self.entries], sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class TrafficAssumption:
    """Caller-declared per-execution traffic for one access or op path."""

    traffic: Traffic
    source: str
    hit_rate_l2: Optional[float] = None

    def __post_init__(self) -> None:
        if not isinstance(self.traffic, Traffic):
            raise ContractError("TrafficAssumption.traffic: must be program.Traffic (read/write split)")
        if not isinstance(self.source, str) or not self.source:
            raise ContractError("TrafficAssumption.source: must name the assumption's origin")
        if self.hit_rate_l2 is not None:
            value = float(self.hit_rate_l2)
            if not (0.0 <= value <= 1.0):
                raise ContractError("TrafficAssumption.hit_rate_l2=%r: must be in [0, 1]" % value)
            object.__setattr__(self, "hit_rate_l2", value)


@dataclass(frozen=True)
class Context:
    """Calibration inputs that override or extend the architecture.

    Keys of ``traffic_assumptions`` are access ids or op paths.  ``max_util``
    maps resource names to utilisation caps in ``[0, 1]``.  ``clock_overrides``
    maps clock domains (``sm``/``l2``/``ddr``) to Hz.  ``residency_overrides``
    maps launch names to resident CTAs per SM.
    """

    latency_table: LatencyTable = field(default_factory=LatencyTable)
    traffic_assumptions: Tuple[Tuple[str, TrafficAssumption], ...] = field(default_factory=tuple)
    max_util: Tuple[Tuple[str, float], ...] = field(default_factory=tuple)
    clock_overrides: Tuple[Tuple[str, float], ...] = field(default_factory=tuple)
    engine_throughput_bytes_per_s: Tuple[Tuple[str, float], ...] = field(default_factory=tuple)
    residency_overrides: Tuple[Tuple[str, int], ...] = field(default_factory=tuple)
    cache_state_between_launches: str = "cold"
    label: str = "default"

    def __post_init__(self) -> None:
        path = "Context"
        if not isinstance(self.latency_table, LatencyTable):
            raise ContractError("%s.latency_table: must be LatencyTable" % path)
        assumptions = _pairs(self.traffic_assumptions, path, "traffic_assumptions")
        for key, value in assumptions:
            if not isinstance(value, TrafficAssumption):
                raise ContractError("%s.traffic_assumptions[%s]: must be TrafficAssumption" % (path, key))
        object.__setattr__(self, "traffic_assumptions", assumptions)
        caps = []
        for key, value in _pairs(self.max_util, path, "max_util"):
            value = float(value)
            if not (0.0 < value <= 1.0):
                raise ContractError("%s.max_util[%s]=%r: must be in (0, 1]" % (path, key, value))
            caps.append((key, value))
        object.__setattr__(self, "max_util", tuple(caps))
        clocks = []
        for key, value in _pairs(self.clock_overrides, path, "clock_overrides"):
            if key not in CLOCK_DOMAINS:
                raise ContractError("%s.clock_overrides[%s]: unknown clock domain" % (path, key))
            clocks.append((key, _pos_float(value, path, "clock_overrides[%s]" % key)))
        object.__setattr__(self, "clock_overrides", tuple(clocks))
        engines = []
        for key, value in _pairs(self.engine_throughput_bytes_per_s, path, "engine_throughput_bytes_per_s"):
            engines.append((key, _pos_float(value, path, "engine_throughput_bytes_per_s[%s]" % key)))
        object.__setattr__(self, "engine_throughput_bytes_per_s", tuple(engines))
        residency = []
        for key, value in _pairs(self.residency_overrides, path, "residency_overrides"):
            residency.append((key, _pos_int(value, path, "residency_overrides[%s]" % key)))
        object.__setattr__(self, "residency_overrides", tuple(residency))
        if self.cache_state_between_launches not in CACHE_STATES:
            raise ContractError(
                "%s.cache_state_between_launches=%r: this version supports %s; "
                "warm_assumed/carry are reported unsupported"
                % (path, self.cache_state_between_launches, ", ".join(CACHE_STATES))
            )
        _ident(self.label, path, "label")

    ARCH_UTILIZATION_FIELDS = (
        ("ddr_max_util", ("ddr",)), ("l2_max_util", ("l2",)), ("l1_5_max_util", ("l1_5",)),
        ("l1_max_util", ("smem", "register")), ("tmem_max_util", ("tmem",)),
        ("compute_max_util", ("tensor", "cuda", "sfu")),
    )

    @classmethod
    def from_arch_utilization(cls, arch: Any, *, label: str = "arch_utilization", **kwargs: Any) -> "Context":
        """Context whose ``max_util`` are the architecture object's own per-channel caps.

        The caps stay an explicit, labelled calibration input (they appear in the
        context canonical form and in every cost group's provenance); nothing is
        applied unless the caller builds the context this way.
        """

        caps = dict(kwargs.pop("max_util", {}) or {})
        for attribute, resources in cls.ARCH_UTILIZATION_FIELDS:
            value = getattr(arch, attribute, None)
            if isinstance(value, (int, float)) and 0.0 < float(value) <= 1.0:
                for resource in resources:
                    caps.setdefault(resource, float(value))
        return cls(max_util=caps, label=label, **kwargs)

    def traffic_assumption(self, *keys: str) -> Optional[Tuple[str, TrafficAssumption]]:
        table = dict(self.traffic_assumptions)
        for key in keys:
            if key in table:
                return key, table[key]
        return None

    def util(self, resource: str) -> float:
        return dict(self.max_util).get(resource, 1.0)

    def clock(self, domain: str) -> Optional[float]:
        return dict(self.clock_overrides).get(domain)

    def engine_throughput(self, engine: str) -> Optional[float]:
        return dict(self.engine_throughput_bytes_per_s).get(engine)

    def residency(self, launch: str) -> Optional[int]:
        return dict(self.residency_overrides).get(launch)

    def canonical(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "latency_table_digest": self.latency_table.digest,
            "traffic_assumptions": [
                {"key": key, "source": value.source, "hit_rate_l2": value.hit_rate_l2,
                 "traffic": value.traffic.to_dict()}
                for key, value in self.traffic_assumptions
            ],
            "max_util": list(self.max_util),
            "clock_overrides": list(self.clock_overrides),
            "engine_throughput_bytes_per_s": list(self.engine_throughput_bytes_per_s),
            "residency_overrides": list(self.residency_overrides),
            "cache_state_between_launches": self.cache_state_between_launches,
        }


__all__ = [
    "BOUNDARY_POLICIES", "CACHE_BOUND_POLICIES", "CACHE_MODES", "Context",
    "II_MODE_CHOICES", "LatencyEntry", "LatencyTable", "Options",
    "RESIDENT_SHARING", "TrafficAssumption",
]
