"""Generic cost binding: ``GenericOracle.resolve(op, arch, context, cache, wave)``.

Service times are bound per resource and may overlap; completion is
``max(offset + service) + latency_extra``.  Extra latency defaults to zero
and is looked up from ``Context.latency_table`` when present.  Device-wide
bandwidths (DDR/L2/L1.5) are shared by the wave context under
``resident_sharing='equal_share'``; per-SM engines (tensor/CUDA/SFU/SMEM/
register/TMEM) are divided only by residency.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

from .traffic import ON_CHIP_LEVELS, Traffic
from ..errors import ModelingError
from ..ir import Phase, ResourceTiming, Timing
from ..ops.throughput import UnsupportedThroughputError, bind_work_throughputs
from .contract import (
    Compute, Load, OpSite, Store, Sync, move_executed_bytes, move_payload_bytes,
    touches_global,
)
from .options import Context, Options


CUDA_FLOPS_PER_INSTRUCTION = 2.0


class UnsupportedCostError(ModelingError):
    """Raised when no honest cost can be bound for an op."""


@dataclass(frozen=True)
class WaveView:
    """Spatial context used to share device-wide resources."""

    wave_kind: str
    sm_count: int
    active_sms: int
    residency: int
    work_units: int

    def __post_init__(self) -> None:
        if self.wave_kind not in ("full", "tail", "not_applicable"):
            raise ModelingError("WaveView.wave_kind=%r is invalid" % self.wave_kind)
        for name in ("sm_count", "active_sms", "residency", "work_units"):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ModelingError("WaveView.%s must be a positive integer" % name)
        if self.active_sms > self.sm_count:
            raise ModelingError("WaveView.active_sms exceeds sm_count")

    def canonical(self) -> Dict[str, Any]:
        return {
            "wave_kind": self.wave_kind, "sm_count": self.sm_count,
            "active_sms": self.active_sms, "residency": self.residency,
            "work_units": self.work_units,
        }


@dataclass(frozen=True)
class BindingContext:
    """Everything that makes two op executions cost-equivalent."""

    launch: str
    group_id: str
    trip_counts: Tuple[Tuple[str, int], ...]
    effective_fractions: Tuple[Tuple[str, float], ...]
    wave: WaveView
    cache_scenario: str
    arch_id: str
    options_digest: str
    context_digest: str

    @property
    def binding_context_id(self) -> str:
        payload = json.dumps(
            {
                "launch": self.launch, "group": self.group_id,
                "trip_counts": list(self.trip_counts),
                "effective_fractions": list(self.effective_fractions),
                "wave": self.wave.canonical(), "cache_scenario": self.cache_scenario,
                "arch": self.arch_id, "options": self.options_digest,
                "context": self.context_digest,
            },
            sort_keys=True,
        ).encode()
        return hashlib.sha256(payload).hexdigest()[:16]


@dataclass(frozen=True)
class BoundCost:
    """The bound cost of one execution of one op."""

    op_id: str
    timing: Timing
    service_s: Tuple[Tuple[str, float], ...]
    completion_latency_s: float
    latency_extra_s: float
    traffic_per_execution: Traffic
    traffic_source: str
    provenance: Tuple[Tuple[str, Any], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ArchView:
    """Validated architecture numbers with clock-domain overrides applied."""

    arch_id: str
    sm_count: int
    sm_clock_hz: Optional[float]
    ddr_bandwidth: Optional[float]
    l2_bandwidth: Optional[float]
    l1_5_bandwidth: Optional[float]
    l1_5_group_size: int
    smem_bandwidth: Optional[float]
    register_bandwidth: Optional[float]
    tmem_bandwidth: Optional[float]
    compute_scale: float
    overrides: Tuple[str, ...]
    clocks_hz: Tuple[Tuple[str, float], ...]
    arch_digest: str = ""

    @staticmethod
    def digest_of(arch: Any) -> str:
        """SHA-256 over the architecture's scalar attributes (numbers, strings, bools)."""

        items = {}
        for name, value in sorted(vars(arch).items()) if hasattr(arch, "__dict__") else ():
            if name.startswith("_"):
                continue
            if isinstance(value, (int, float, str, bool)) or value is None:
                items[name] = value
            elif isinstance(value, (tuple, list)) and all(isinstance(v, (int, float, str, bool)) for v in value):
                items[name] = list(value)
        payload = json.dumps({"type": type(arch).__name__, "attrs": items}, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def from_arch(cls, arch: Any, context: Context) -> "ArchView":
        sm_count = getattr(arch, "sm_count", None)
        if not isinstance(sm_count, int) or isinstance(sm_count, bool) or sm_count <= 0:
            raise UnsupportedCostError("architecture must provide a positive integer sm_count")

        def positive(name: str) -> Optional[float]:
            value = getattr(arch, name, None)
            if isinstance(value, (int, float)) and math.isfinite(float(value)) and value > 0:
                return float(value)
            return None

        base_sm_clock = positive("max_freq") or positive("freq")
        overrides = []
        compute_scale = 1.0
        sm_clock = base_sm_clock
        override = context.clock("sm")
        if override is not None:
            if base_sm_clock is None:
                raise UnsupportedCostError(
                    "clock override for 'sm' needs arch.max_freq to scale from"
                )
            compute_scale = override / base_sm_clock
            sm_clock = override
            overrides.append("clock:sm=%g (base %g, compute scale %.4f)"
                             % (override, base_sm_clock, compute_scale))

        def scaled_bw(name: str, domain: str) -> Optional[float]:
            value = positive(name)
            override = context.clock(domain)
            if override is None or value is None:
                return value
            base = positive("%s_freq" % domain)
            if base is None:
                raise UnsupportedCostError(
                    "clock override for %r needs arch.%s_freq to scale %s" % (domain, domain, name)
                )
            overrides.append("clock:%s=%g scales %s by %.4f" % (domain, override, name, override / base))
            return value * override / base

        clocks = {}
        if sm_clock is not None:
            clocks["sm"] = sm_clock
        for domain in ("l2", "ddr"):
            value = context.clock(domain) or positive("%s_freq" % domain)
            if value is not None:
                clocks[domain] = value
        return cls(
            arch_id="%s:%s" % (type(arch).__name__, getattr(arch, "core", "") or ""),
            sm_count=sm_count,
            sm_clock_hz=sm_clock,
            ddr_bandwidth=scaled_bw("ddr_bandwidth", "ddr"),
            l2_bandwidth=scaled_bw("l2_bandwidth", "l2"),
            l1_5_bandwidth=positive("l1_5_bandwidth"),
            l1_5_group_size=int(getattr(arch, "l1_5_group_size", 0) or 0),
            smem_bandwidth=(positive("smem_bandwidth") or 0.0) * compute_scale or None,
            register_bandwidth=(positive("register_bandwidth") or 0.0) * compute_scale or None,
            tmem_bandwidth=(positive("tmem_bandwidth") or 0.0) * compute_scale or None,
            compute_scale=compute_scale,
            overrides=tuple(overrides),
            clocks_hz=tuple(sorted(clocks.items())),
            arch_digest=cls.digest_of(arch),
        )


def _size_class(nbytes: float) -> str:
    if nbytes <= 4096:
        return "le4k"
    if nbytes <= 65536:
        return "le64k"
    return "gt64k"


class GenericOracle:
    """Architecture-neutral cost binder for the four primitives."""

    def __init__(self, arch: Any, options: Options, context: Context) -> None:
        self.arch = arch
        self.options = options
        self.context = context
        self.view = ArchView.from_arch(arch, context)

    # -- public --------------------------------------------------------------

    def bind(
        self,
        sites: Mapping[str, OpSite],
        binding: BindingContext,
        cache_view: Any,
    ) -> "BoundOracle":
        return BoundOracle(self, sites, binding, cache_view)

    def resolve(
        self,
        site: OpSite,
        binding: BindingContext,
        cache_view: Any,
    ) -> BoundCost:
        op = site.op
        if op.timing is not None:
            timing = op.timing
            return BoundCost(
                op_id=op.op_id,
                timing=timing,
                service_s=tuple((r.resource, r.service_time) for r in timing.resources),
                completion_latency_s=timing.latency,
                latency_extra_s=0.0,
                traffic_per_execution=self._explicit_or_zero_traffic(site, cache_view)[0],
                traffic_source="explicit_timing_override",
                provenance=(("cost_source", "explicit_timing_override"),),
            )
        if isinstance(op, Compute):
            return self._bind_compute(site, binding)
        if isinstance(op, (Load, Store)):
            return self._bind_move(site, binding, cache_view)
        if isinstance(op, Sync):
            extra, extra_source = self._latency_extra(op, {"kind": "sync"})
            return BoundCost(
                op_id=op.op_id,
                timing=Timing(extra),
                service_s=tuple(),
                completion_latency_s=extra,
                latency_extra_s=extra,
                traffic_per_execution=Traffic(),
                traffic_source="not_applicable",
                provenance=(("cost_source", "sync_zero_service"), ("latency_extra_source", extra_source)),
            )
        raise UnsupportedCostError("%s: unknown op kind" % site.path)

    # -- helpers -------------------------------------------------------------

    def _latency_extra(self, op: Any, key: Mapping[str, str], service_max_s: float = 0.0) -> Tuple[float, str]:
        """Extra latency beyond resource service, in seconds.

        A table row measured as ``completion_latency`` is converted with
        ``extra = max(measured - bound service, 0)`` and the conversion is
        recorded in the returned source label.
        """

        if op.latency_extra_s is not None:
            return op.latency_extra_s, "op_explicit"
        full_key = dict(key)
        full_key.setdefault("arch", self.view.arch_id)
        value, entry = self.context.latency_table.lookup(full_key, dict(self.view.clocks_hz))
        if entry is None:
            # Also allow arch wildcard rows.
            full_key["arch"] = "*"
            value, entry = self.context.latency_table.lookup(full_key, dict(self.view.clocks_hz))
        if entry is None:
            return 0.0, "default_zero"
        label = "latency_table:%s" % (entry.source_file or "inline")
        if entry.measured_quantity == "completion_latency":
            converted = max(value - service_max_s, 0.0)
            return converted, "%s:completion_latency-service(%.3g-%.3g)" % (label, value, service_max_s)
        return value, label

    def _util(self, resource: str) -> float:
        return self.context.util(resource)

    def _bind_compute(self, site: OpSite, binding: BindingContext) -> BoundCost:
        op = site.op  # type: Compute
        try:
            bindings = bind_work_throughputs(op.spec, self.arch, self.options.throughput_fallback)
        except UnsupportedThroughputError as error:
            raise UnsupportedCostError("%s: %s" % (site.path, error)) from error
        per_resource = {}  # type: Dict[str, float]
        sources = []
        demand = []
        for item in bindings:
            resource = item.work.engine.name
            if op.engine is not None and resource != op.engine:
                raise UnsupportedCostError(
                    "%s: op engine %r disagrees with spec work engine %r"
                    % (site.path, op.engine, resource)
                )
            if not item.resolution.supported:
                raise UnsupportedCostError(
                    "%s: %s throughput unsupported (%s)"
                    % (site.path, resource, item.resolution.source)
                )
            # Whole-device capacity -> per-SM share -> residency share.
            capacity = item.resolution.capacity_per_s * self.view.compute_scale
            unit = dict(item.work.attrs).get("issue_unit")
            if resource == "cuda" and unit == "cuda_instruction":
                # Design 2.2.1: the demand is a count of CUDA instruction units
                # (add, mul and a fused FMA are one unit each).  The architecture
                # parameter is an FMA-peak FLOP/s figure (cores x 2 FLOPs per
                # cycle), so the same-granularity instruction rate is half of it.
                capacity = capacity / CUDA_FLOPS_PER_INSTRUCTION
                amount = item.work.count * (item.work.issue_equivalent_flops_per_op or 1.0)
                unit_note = "cuda_instruction/s = %s / %g (FMA-peak FLOP/s convention)" % (
                    item.resolution.source, CUDA_FLOPS_PER_INSTRUCTION)
            else:
                amount = item.work.issue_equivalent_flops or item.work.count
                unit_note = "%s in %s" % (item.resolution.source, item.resolution.unit)
            per_sm = capacity / self.view.sm_count
            share = per_sm * self._util(resource) / binding.wave.residency
            per_resource[resource] = per_resource.get(resource, 0.0) + amount / share
            demand.append((item.work.op_class, resource, unit or item.resolution.unit, amount, item.work.semantic_flops))
            sources.append("%s:%s[fallback=%s]" % (resource, unit_note, item.resolution.fallback))
        resources = tuple(
            ResourceTiming(name, value) for name, value in sorted(per_resource.items()) if value > 0.0
        )
        service_max = max((r.service_time for r in resources), default=0.0)
        extra, extra_source = self._latency_extra(
            op,
            {
                "kind": "compute",
                "engine": op.engine or (bindings[0].work.engine.name if bindings else "*"),
                "op_class": bindings[0].work.op_class if bindings else "*",
                "dtype": bindings[0].work.dtype.name if bindings else "*",
            },
            service_max,
        )
        completion = service_max + extra
        timing = Timing(completion, resources)
        return BoundCost(
            op_id=op.op_id,
            timing=timing,
            service_s=tuple((r.resource, r.service_time) for r in resources),
            completion_latency_s=completion,
            latency_extra_s=extra,
            traffic_per_execution=Traffic.compute_internal(),
            traffic_source="compute_internal_unknown",
            provenance=(
                ("cost_source", "generic_oracle_compute"),
                ("throughput_sources", tuple(sources)),
                ("issue_demand", tuple(demand)),  # (op_class, resource, unit, issue amount, semantic FLOPs)
                ("latency_extra_source", extra_source),
                ("resident_sharing", self.options.resident_sharing),
                ("residency_divisor", binding.wave.residency),
                ("max_util", tuple((r.resource, self._util(r.resource)) for r in resources)),
            ),
        )

    def _explicit_or_zero_traffic(self, site: OpSite, cache_view: Any) -> Tuple[Traffic, str]:
        entry = None if cache_view is None else cache_view.traffic_for(site.op.op_id)
        if entry is None:
            if isinstance(site.op, Compute):
                return Traffic.compute_internal(), "compute_internal_unknown"
            if isinstance(site.op, (Load, Store)):
                return Traffic.on_chip_move(site.op.source, site.op.destination, move_executed_bytes(site.op)), "on_chip_endpoints"
            return Traffic(), "none"
        return entry.per_execution, entry.source

    def _bind_move(self, site: OpSite, binding: BindingContext, cache_view: Any) -> BoundCost:
        """Price one move from its read/write-split traffic (modeling is the only source)."""

        op = site.op
        payload = move_payload_bytes(op)
        executed = move_executed_bytes(op)
        per_resource = {}  # type: Dict[str, float]
        provenance = [("cost_source", "generic_oracle_move")]
        wave = binding.wave
        entry = None if cache_view is None else cache_view.traffic_for(op.op_id)
        if entry is None:
            if touches_global(op):
                raise UnsupportedCostError(
                    "%s: global access has no traffic; provide a TileAccess with "
                    "cache='fast'/'trace', a Context.traffic_assumptions entry, or "
                    "Options(missing_traffic_policy='assume_miss')" % site.path
                )
            traffic = Traffic.on_chip_move(op.source, op.destination, executed)
            traffic_source = "on_chip_endpoints"
        else:
            traffic = entry.per_execution
            traffic_source = entry.source
        # Device-wide passive levels: read + write requests share one declared peak.
        if touches_global(op):
            active = wave.active_sms
            required = ["ddr_read_bytes", "ddr_write_bytes", "l2_read_request_bytes", "l2_write_request_bytes"]
            if self.view.l1_5_bandwidth is not None:
                required += ["l1_5_read_request_bytes", "l1_5_write_request_bytes"]
            missing = traffic.unknown_among(required)
            if missing:
                raise UnsupportedCostError(
                    "%s: cannot price a global move with unknown traffic %s (%s)"
                    % (site.path, list(missing), "; ".join(
                        "%s: %s" % (name, traffic.unknown_reason(name)) for name in missing))
                )
            ddr_bytes = traffic.ddr_read_bytes + traffic.ddr_write_bytes
            l2_bytes = traffic.l2_read_request_bytes + traffic.l2_write_request_bytes
            l1_5_bytes = traffic.l1_5_read_request_bytes + traffic.l1_5_write_request_bytes
            if ddr_bytes > 0.0:
                if self.view.ddr_bandwidth is None:
                    raise UnsupportedCostError("%s: arch lacks ddr_bandwidth" % site.path)
                share = self.view.ddr_bandwidth * self._util("ddr") / active / wave.residency
                per_resource["ddr"] = ddr_bytes / share
            if l2_bytes > 0.0:
                if self.view.l2_bandwidth is None:
                    raise UnsupportedCostError("%s: arch lacks l2_bandwidth" % site.path)
                share = self.view.l2_bandwidth * self._util("l2") / active / wave.residency
                per_resource["l2"] = l2_bytes / share
            if l1_5_bytes > 0.0 and self.view.l1_5_bandwidth is not None:
                # L1.5 is a per-SM-group cache: the whole-chip figure is the sum over
                # groups, so an SM never gets more than peak / sm_count of it, however
                # few SMs are active (unlike the pooled DDR/L2 links).
                share = self.view.l1_5_bandwidth / self.view.sm_count * self._util("l1_5") / wave.residency
                per_resource["l1_5"] = l1_5_bytes / share
            provenance.append(("device_bandwidth_share",
                               "DDR/L2: device_peak x max_util / active_sms / residency; "
                               "L1.5: device_peak / sm_count x max_util / residency (per-group)"))
            provenance.append(("active_sms", active))
            provenance.append(("read_write_sharing", "read + write requests share one declared peak per level"))
        # On-chip endpoints: read + write bytes of each port from the same traffic.
        port_service = 0.0
        on_chip = []
        for storage in ON_CHIP_LEVELS:
            level_bytes = traffic.level_bytes(storage)
            if level_bytes is None:
                if storage in (op.source, op.destination):
                    raise UnsupportedCostError(
                        "%s: %s endpoint traffic is unknown; cannot price the move" % (site.path, storage)
                    )
                continue
            if level_bytes <= 0.0:
                continue
            on_chip.append(storage)
            bandwidth = {
                "smem": self.view.smem_bandwidth,
                "register": self.view.register_bandwidth,
                "tmem": self.view.tmem_bandwidth,
            }[storage]
            if bandwidth is None:
                raise UnsupportedCostError(
                    "%s: arch provides no %s bandwidth; cannot bind on-chip move service"
                    % (site.path, storage)
                )
            share = bandwidth / self.view.sm_count * self._util(storage) / wave.residency
            service = level_bytes / share
            per_resource[storage] = per_resource.get(storage, 0.0) + service
            port_service = max(port_service, service)
        provenance.append(("on_chip_port_service", "per_sm_bandwidth x max_util / residency over read+write bytes"))
        engine_rate = self.context.engine_throughput(op.engine)
        if op.engine == "unbound":
            provenance.append(("engine_service", "omitted:engine_unbound"))
        elif engine_rate is not None:
            share = engine_rate * self._util(op.engine) / wave.residency
            per_resource[op.engine] = executed / share
            provenance.append(("engine_service", "context.engine_throughput_bytes_per_s"))
        else:
            per_resource[op.engine] = per_resource.get(op.engine, 0.0) + port_service
            provenance.append((
                "engine_service",
                "proxy:%s_port_bandwidth (no engine throughput for %s)"
                % ("/".join(on_chip) or "none", op.engine),
            ))
        resources = tuple(
            ResourceTiming(name, value) for name, value in sorted(per_resource.items()) if value > 0.0
        )
        service_max = max((r.service_time for r in resources), default=0.0)
        extra, extra_source = self._latency_extra(
            op,
            {
                "kind": op.kind, "engine": op.engine, "source": op.source,
                "destination": op.destination, "size_class": _size_class(executed),
            },
            service_max,
        )
        completion = service_max + extra
        timing = Timing(completion, resources)
        provenance.extend((
            ("latency_extra_source", extra_source),
            ("payload_bytes", payload),
            ("executed_bytes", executed),
            ("resident_sharing", self.options.resident_sharing),
            ("max_util", tuple((r.resource, self._util(r.resource)) for r in resources)),
            ("traffic_per_execution", traffic.to_dict()),
        ))
        return BoundCost(
            op_id=op.op_id,
            timing=timing,
            service_s=tuple((r.resource, r.service_time) for r in resources),
            completion_latency_s=completion,
            latency_extra_s=extra,
            traffic_per_execution=traffic,
            traffic_source=traffic_source,
            provenance=tuple(provenance),
        )


class BoundOracle:
    """``CostOracle``-compatible resolver bound to one context and cache view."""

    def __init__(
        self,
        oracle: GenericOracle,
        sites: Mapping[str, OpSite],
        binding: BindingContext,
        cache_view: Any,
    ) -> None:
        self.oracle = oracle
        self.sites = dict(sites)
        self.binding = binding
        self.cache_view = cache_view
        self.costs = {}  # type: Dict[str, BoundCost]

    def cost(self, op_id: str) -> BoundCost:
        found = self.costs.get(op_id)
        if found is None:
            site = self.sites.get(op_id)
            if site is None:
                raise UnsupportedCostError("oracle has no op for phase %r" % op_id)
            found = self.oracle.resolve(site, self.binding, self.cache_view)
            self.costs[op_id] = found
        return found

    def resolve(self, phase: Phase) -> Timing:
        return self.cost(phase.name).timing


__all__ = [
    "ArchView", "BindingContext", "BoundCost", "BoundOracle", "GenericOracle",
    "UnsupportedCostError", "WaveView",
]
