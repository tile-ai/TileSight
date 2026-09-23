"""Dtype-aware architecture throughput lookup with explicit fallbacks."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from ..errors import ModelingError, ModelingValidationError
from ..ir import freeze_attrs, validate_identifier
from .schema import DTypeSpec, EngineSpec, ScalarWorkSpec, as_dtype, as_engine


class UnsupportedThroughputError(ModelingError):
    pass


_ARCH_ATTR = {
    ("cuda", "fp16"): "fp16_cuda_core_flops",
    ("cuda", "fp32"): "fp32_cuda_core_flops",
    ("cuda", "fp64"): "fp64_cuda_core_flops",
    ("cuda", "int32"): "int32_cuda_core_flops",
    ("tensor", "fp16"): "fp16_tensor_flops",
    ("tensor", "fp8_e4m3"): "fp8_tensor_flops",
    ("tensor", "fp8_e5m2"): "fp8_tensor_flops",
    ("tensor", "int8"): "int8_tensor_flops",
    ("tensor", "fp32"): "fp32_tensor_flops",
    ("tensor", "tf32"): "tf32_tensor_flops",
}


@dataclass(frozen=True)
class ThroughputRequest:
    engine: EngineSpec = field(default_factory=EngineSpec)
    dtype: DTypeSpec = field(default_factory=DTypeSpec)
    op_class: str = "generic"
    vector_width: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "engine", as_engine(self.engine))
        object.__setattr__(self, "dtype", as_dtype(self.dtype))
        validate_identifier(self.op_class, "throughput op_class")
        if (
            not isinstance(self.vector_width, int)
            or isinstance(self.vector_width, bool)
            or self.vector_width <= 0
        ):
            raise ModelingValidationError("vector_width must be positive")


@dataclass(frozen=True)
class ThroughputFallbackPolicy:
    """Only caller-declared aliases/capacities are permitted."""

    dtype_aliases: Tuple[Tuple[str, str], ...] = field(default_factory=tuple)
    explicit_capacities: Tuple[Tuple[str, float], ...] = field(default_factory=tuple)
    allow_dtype_agnostic_sfu: bool = False

    def __post_init__(self) -> None:
        aliases = tuple(
            (as_dtype(source).name, as_dtype(target).name)
            for source, target in self.dtype_aliases
        )
        keys = [source for source, _target in aliases]
        if len(keys) != len(set(keys)):
            raise ModelingValidationError("fallback dtype aliases must be unique")
        capacities = []
        for key, value in self.explicit_capacities:
            validate_identifier(key, "explicit throughput key")
            value = float(value)
            if not math.isfinite(value) or value <= 0.0:
                raise ModelingValidationError("explicit throughput must be positive")
            capacities.append((key, value))
        if len(capacities) != len({key for key, _value in capacities}):
            raise ModelingValidationError("explicit throughput keys must be unique")
        if not isinstance(self.allow_dtype_agnostic_sfu, bool):
            raise ModelingValidationError("allow_dtype_agnostic_sfu must be bool")
        object.__setattr__(self, "dtype_aliases", aliases)
        object.__setattr__(self, "explicit_capacities", tuple(capacities))


@dataclass(frozen=True)
class ThroughputResolution:
    request: ThroughputRequest
    supported: bool
    capacity_per_s: Optional[float]
    unit: str
    source: str
    fallback: str = "none"
    provenance: Tuple[Tuple[str, Any], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.request, ThroughputRequest):
            raise ModelingValidationError("resolution request must be ThroughputRequest")
        if not isinstance(self.supported, bool):
            raise ModelingValidationError("resolution supported must be bool")
        if self.capacity_per_s is not None:
            value = float(self.capacity_per_s)
            if not math.isfinite(value) or value <= 0.0:
                raise ModelingValidationError("throughput capacity must be positive")
            object.__setattr__(self, "capacity_per_s", value)
        if self.supported != (self.capacity_per_s is not None):
            raise ModelingValidationError("supported and capacity_per_s disagree")
        validate_identifier(self.unit, "throughput unit")
        if not self.source:
            raise ModelingValidationError("throughput source must not be empty")
        if not self.fallback:
            raise ModelingValidationError("throughput fallback must not be empty")
        object.__setattr__(self, "provenance", freeze_attrs(self.provenance))

    def require(self) -> float:
        if not self.supported:
            raise UnsupportedThroughputError(
                "unsupported throughput: engine=%s dtype=%s op=%s; %s"
                % (
                    self.request.engine.name,
                    self.request.dtype.name,
                    self.request.op_class,
                    self.source,
                )
            )
        return self.capacity_per_s


def _explicit_key(request: ThroughputRequest) -> str:
    return "%s.%s.%s" % (
        request.engine.name,
        request.dtype.name,
        request.op_class,
    )


def resolve_throughput(
    arch: Any,
    request: ThroughputRequest,
    fallback_policy: Optional[ThroughputFallbackPolicy] = None,
) -> ThroughputResolution:
    """Resolve without ever treating byte width as dtype identity."""

    if not isinstance(request, ThroughputRequest):
        raise ModelingValidationError("resolve_throughput expects ThroughputRequest")
    policy = fallback_policy or ThroughputFallbackPolicy()
    explicit = dict(policy.explicit_capacities)
    key = _explicit_key(request)
    if key in explicit:
        return ThroughputResolution(
            request,
            True,
            explicit[key],
            "flop_equivalent_per_s",
            source="explicit_capacity:%s" % key,
            fallback="explicit_capacity",
            provenance={"arch": type(arch).__name__},
        )

    if request.engine.name == "sfu":
        value = getattr(arch, "sfu_flops", None)
        if policy.allow_dtype_agnostic_sfu and isinstance(value, (int, float)) and value > 0:
            return ThroughputResolution(
                request,
                True,
                float(value),
                "op_equivalent_per_s",
                source="arch.sfu_flops",
                fallback="dtype_agnostic_sfu",
                provenance={"confidence": "legacy_dtype_agnostic"},
            )
        return ThroughputResolution(
            request,
            False,
            None,
            "op_equivalent_per_s",
            source="dtype-specific SFU throughput is unavailable",
            fallback="unsupported",
        )

    attr = _ARCH_ATTR.get((request.engine.name, request.dtype.name))
    value = None if attr is None else getattr(arch, attr, None)
    if isinstance(value, (int, float)) and math.isfinite(float(value)) and value > 0:
        return ThroughputResolution(
            request,
            True,
            float(value),
            "flop_equivalent_per_s",
            source="arch.%s" % attr,
            provenance={
                "requested_engine": request.engine.requested,
                "canonical_engine": request.engine.name,
            },
        )

    aliases = dict(policy.dtype_aliases)
    target = aliases.get(request.dtype.name)
    if target is not None:
        target_request = ThroughputRequest(
            request.engine,
            DTypeSpec(target),
            request.op_class,
            request.vector_width,
        )
        resolved = resolve_throughput(
            arch,
            target_request,
            ThroughputFallbackPolicy(
                dtype_aliases=tuple(),
                explicit_capacities=policy.explicit_capacities,
                allow_dtype_agnostic_sfu=policy.allow_dtype_agnostic_sfu,
            ),
        )
        if resolved.supported:
            return ThroughputResolution(
                request,
                True,
                resolved.capacity_per_s,
                resolved.unit,
                source=resolved.source,
                fallback="dtype_alias:%s->%s" % (request.dtype.name, target),
                provenance={
                    "fallback_dtype": target,
                    "underlying_source": resolved.source,
                },
            )

    missing = "no architecture throughput mapping"
    if attr is not None:
        missing = "architecture is missing %s" % attr
    return ThroughputResolution(
        request,
        False,
        None,
        "flop_equivalent_per_s",
        source=missing,
        fallback="unsupported",
        provenance={"arch": type(arch).__name__},
    )


@dataclass(frozen=True)
class WorkThroughputBinding:
    work: ScalarWorkSpec
    resolution: ThroughputResolution
    service_time_s: Optional[float]

    def __post_init__(self) -> None:
        if not isinstance(self.work, ScalarWorkSpec):
            raise ModelingValidationError("binding work must be ScalarWorkSpec")
        if not isinstance(self.resolution, ThroughputResolution):
            raise ModelingValidationError("binding resolution must be typed")
        if self.service_time_s is not None:
            value = float(self.service_time_s)
            if not math.isfinite(value) or value < 0.0:
                raise ModelingValidationError("service_time_s must be non-negative")
            object.__setattr__(self, "service_time_s", value)


def bind_work_throughputs(
    op: Any,
    arch: Any,
    fallback_policy: Optional[ThroughputFallbackPolicy] = None,
) -> Tuple[WorkThroughputBinding, ...]:
    """Bind each work item separately; engines may later overlap in scheduling."""

    try:
        work_items = tuple(op.work_items)
    except (AttributeError, TypeError) as error:
        raise ModelingValidationError("semantic op must expose work_items") from error
    result = []
    for work in work_items:
        request = ThroughputRequest(work.engine, work.dtype, work.op_class)
        resolution = resolve_throughput(arch, request, fallback_policy)
        amount = work.issue_equivalent_flops
        if amount <= 0.0:
            amount = work.count
        service = None if not resolution.supported else amount / resolution.capacity_per_s
        result.append(WorkThroughputBinding(work, resolution, service))
    return tuple(result)


__all__ = [
    "ThroughputFallbackPolicy",
    "ThroughputRequest",
    "ThroughputResolution",
    "UnsupportedThroughputError",
    "WorkThroughputBinding",
    "bind_work_throughputs",
    "resolve_throughput",
]
