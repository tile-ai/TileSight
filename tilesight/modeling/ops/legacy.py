"""Opt-in compatibility formulas recovered from legacy TileSight op models."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Tuple

from ..errors import ModelingValidationError
from ..ir import freeze_attrs


@dataclass(frozen=True)
class LegacyCompatibilityPolicy:
    """Keep empirical v0 behavior explicit and out of semantic work counts."""

    profile: str = "legacy_tilesight_v0"
    register_spill_factor: float = 1.1
    scalar_issue_equivalent_flops: float = 2.0
    ddr_nonideal_elementwise: float = 1.0
    ddr_nonideal_reduce: float = 1.1
    force_elementwise_l2_hit_zero: bool = True
    use_thread_bucket_overhead: bool = True
    attrs: Tuple[Tuple[str, Any], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.profile:
            raise ModelingValidationError("legacy profile must not be empty")
        for name in (
            "register_spill_factor",
            "scalar_issue_equivalent_flops",
            "ddr_nonideal_elementwise",
            "ddr_nonideal_reduce",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ModelingValidationError("legacy %s must be positive" % name)
            object.__setattr__(self, name, value)
        for name in ("force_elementwise_l2_hit_zero", "use_thread_bucket_overhead"):
            if not isinstance(getattr(self, name), bool):
                raise ModelingValidationError("legacy %s must be bool" % name)
        object.__setattr__(self, "attrs", freeze_attrs(self.attrs))

    def thread_overhead(self, threads_per_cta: int) -> float:
        if (
            not isinstance(threads_per_cta, int)
            or isinstance(threads_per_cta, bool)
            or threads_per_cta <= 0
        ):
            raise ModelingValidationError("threads_per_cta must be positive")
        if not self.use_thread_bucket_overhead:
            return 1.0
        if threads_per_cta <= 32:
            return 32.0 / threads_per_cta
        if threads_per_cta <= 128:
            return 128.0 / threads_per_cta
        if threads_per_cta <= 256:
            return 256.0 / threads_per_cta
        if threads_per_cta <= 384:
            return 384.0 / threads_per_cta
        return 1.0

    def elementwise_issue_equivalent_flops(
        self,
        output_elements: int,
        num_ops: int,
        threads_per_cta: int,
    ) -> float:
        if output_elements < 0 or num_ops < 0:
            raise ModelingValidationError("element and op counts must be non-negative")
        return (
            self.scalar_issue_equivalent_flops
            * output_elements
            * num_ops
            * self.thread_overhead(threads_per_cta)
        )

    def reduce_issue_equivalent_flops(
        self,
        output_tile_elements: int,
        reduction_step_elements: int,
        threads_per_cta: int,
    ) -> float:
        if output_tile_elements < 0 or reduction_step_elements < 0:
            raise ModelingValidationError("reduce element counts must be non-negative")
        return (
            self.scalar_issue_equivalent_flops
            * output_tile_elements
            * reduction_step_elements
            * self.thread_overhead(threads_per_cta)
        )

    def collective_issue_equivalent_flops(
        self,
        output_tile_elements: int,
        reduce_threads: int,
        threads_per_cta: int,
    ) -> float:
        if reduce_threads <= 0:
            raise ModelingValidationError("reduce_threads must be positive")
        return (
            self.scalar_issue_equivalent_flops
            * output_tile_elements
            * math.ceil(math.log2(reduce_threads))
            * self.thread_overhead(threads_per_cta)
        )

    def provenance(self) -> Tuple[Tuple[str, Any], ...]:
        return freeze_attrs(
            {
                "profile": self.profile,
                "register_spill_factor": self.register_spill_factor,
                "scalar_issue_equivalent_flops": self.scalar_issue_equivalent_flops,
                "thread_bucket_overhead": self.use_thread_bucket_overhead,
                "semantic_default": False,
            }
        )


__all__ = ["LegacyCompatibilityPolicy"]
