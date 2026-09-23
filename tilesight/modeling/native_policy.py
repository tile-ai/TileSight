"""Typed policies for the opt-in native whole-kernel executor.

The objects in this module are deliberately descriptive.  They separate
temporal boundary calibration, spatial launch topology, and small-grid
latency assumptions so that none of those choices can be inferred from an
unrelated integer such as ``cluster_size`` or ``resident_ctas_per_sm``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Tuple

from .errors import ModelingValidationError
from .ir import freeze_attrs, validate_identifier


BOUNDARY_POLICIES = ("periodic_witness", "finite_witness", "legacy_stage")
WAVE_POLICIES = ("native", "legacy_calibrated")
SPATIAL_CONCURRENCY_KINDS = (
    "single",
    "resident",
    "ordinary_cluster",
    "cooperative_cluster",
)


def _identifiers(values: Tuple[str, ...], label: str) -> Tuple[str, ...]:
    result = tuple(values)
    if len(result) != len(set(result)):
        raise ModelingValidationError("%s must not contain duplicates" % label)
    for value in result:
        validate_identifier(value, label)
    return result


@dataclass(frozen=True)
class LegacyStageBoundary:
    """Explicit resource partition for the calibrated ``(stages-1)`` edge.

    The old pipeline equation treats the largest per-iteration service time
    in ``memory_resources`` as the fill cost and the largest service time in
    ``compute_resources`` as the drain cost.  Requiring both sets prevents a
    generic DAG from silently guessing which resources belong on either side
    of the software pipeline.
    """

    memory_resources: Tuple[str, ...]
    compute_resources: Tuple[str, ...]

    def __post_init__(self) -> None:
        memory = _identifiers(tuple(self.memory_resources), "legacy memory resource")
        compute = _identifiers(tuple(self.compute_resources), "legacy compute resource")
        if not memory or not compute:
            raise ModelingValidationError(
                "legacy stage boundary needs nonempty memory and compute resources"
            )
        overlap = sorted(set(memory).intersection(compute))
        if overlap:
            raise ModelingValidationError(
                "legacy memory/compute resource sets must be disjoint; overlap=%s"
                % overlap
            )
        object.__setattr__(self, "memory_resources", memory)
        object.__setattr__(self, "compute_resources", compute)


@dataclass(frozen=True)
class SmallTileLatencyConfig:
    """Explicit latency/credit and underfilled-grid sensitivity knobs.

    ``fixed_load_latency_s`` raises the selected phases' completion latency.
    ``outstanding_credits`` caps their existing frontend ``PipelineBuffer``
    tokens; it never creates a token or a dependency.  Consequently both
    quantities enter the normal PeriodicDAG recurrence/credit calculation.

    The admission and efficiency terms are disabled by default and only apply
    to an underfilled spatial wave when the user opts in.
    """

    load_phases: Tuple[str, ...] = field(default_factory=tuple)
    fixed_load_latency_s: float = 0.0
    outstanding_credits: int = 0
    underfilled_admission_s: float = 0.0
    underfilled_efficiency: float = 1.0

    def __post_init__(self) -> None:
        phases = _identifiers(tuple(self.load_phases), "small-tile load phase")
        object.__setattr__(self, "load_phases", phases)
        latency = float(self.fixed_load_latency_s)
        admission = float(self.underfilled_admission_s)
        efficiency = float(self.underfilled_efficiency)
        if not math.isfinite(latency) or latency < 0.0:
            raise ModelingValidationError(
                "fixed_load_latency_s must be finite and non-negative"
            )
        if not math.isfinite(admission) or admission < 0.0:
            raise ModelingValidationError(
                "underfilled_admission_s must be finite and non-negative"
            )
        if not math.isfinite(efficiency) or not (0.0 < efficiency <= 1.0):
            raise ModelingValidationError(
                "underfilled_efficiency must be in the interval (0, 1]"
            )
        if (
            not isinstance(self.outstanding_credits, int)
            or isinstance(self.outstanding_credits, bool)
            or self.outstanding_credits < 0
        ):
            raise ModelingValidationError(
                "outstanding_credits must be a non-negative integer"
            )
        if (latency > 0.0 or self.outstanding_credits > 0) and not phases:
            raise ModelingValidationError(
                "load_phases are required for fixed latency or credit caps"
            )
        object.__setattr__(self, "fixed_load_latency_s", latency)
        object.__setattr__(self, "underfilled_admission_s", admission)
        object.__setattr__(self, "underfilled_efficiency", efficiency)

    @property
    def adjusts_underfilled_wave(self) -> bool:
        return (
            self.underfilled_admission_s > 0.0
            or self.underfilled_efficiency < 1.0
        )


@dataclass(frozen=True)
class LaunchTopology:
    """Typed meaning of launch-level spatial concurrency.

    ``ordinary_cluster`` means each CTA remains an independent work tile; the
    cluster only constrains placement/locality.  ``cooperative_cluster`` means
    all CTAs jointly implement one logical work unit.  Neither interpretation
    creates a generic DSM bandwidth resource.  Peer-SMEM traffic is named
    explicitly and must be consumed by a cluster-aware cost oracle.
    """

    kind: str
    cluster_size: int = 1
    tma_multicast: bool = False
    operand_plan: Tuple[Tuple[str, str], ...] = field(default_factory=tuple)
    peer_smem_resources: Tuple[str, ...] = field(default_factory=tuple)
    cluster_barrier_s: float = 0.0
    cluster_issue_s: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in SPATIAL_CONCURRENCY_KINDS:
            raise ModelingValidationError(
                "launch topology kind must be one of %s"
                % ", ".join(SPATIAL_CONCURRENCY_KINDS)
            )
        if (
            not isinstance(self.cluster_size, int)
            or isinstance(self.cluster_size, bool)
            or self.cluster_size <= 0
        ):
            raise ModelingValidationError("topology cluster_size must be positive")
        if self.kind in ("single", "resident") and self.cluster_size != 1:
            raise ModelingValidationError(
                "%s topology requires cluster_size=1" % self.kind
            )
        if self.kind in ("ordinary_cluster", "cooperative_cluster") and (
            self.cluster_size <= 1
        ):
            raise ModelingValidationError(
                "%s topology requires cluster_size>1" % self.kind
            )
        if not isinstance(self.tma_multicast, bool):
            raise ModelingValidationError("tma_multicast must be bool")
        plan = freeze_attrs(self.operand_plan)
        # This tuple is a compact oracle-contract summary, not the canonical
        # transfer IR.  ``peer_smem`` is kept distinct so a cooperative oracle
        # cannot silently price a directional peer access as duplication.
        allowed = ("partition", "duplicate", "multicast", "peer_smem")
        for operand, mode in plan:
            validate_identifier(operand, "cluster operand")
            if mode not in allowed:
                raise ModelingValidationError(
                    "cluster operand mode must be one of %s" % ", ".join(allowed)
                )
        object.__setattr__(self, "operand_plan", plan)
        peers = _identifiers(
            tuple(self.peer_smem_resources), "peer-SMEM resource"
        )
        object.__setattr__(self, "peer_smem_resources", peers)
        for name in ("cluster_barrier_s", "cluster_issue_s"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ModelingValidationError("%s must be non-negative" % name)
            object.__setattr__(self, name, value)
        if self.kind == "cooperative_cluster" and not plan:
            raise ModelingValidationError(
                "cooperative_cluster needs a source-derived operand_plan"
            )
        if self.kind != "cooperative_cluster" and plan:
            raise ModelingValidationError(
                "operand_plan is only valid for cooperative_cluster"
            )

    @classmethod
    def single(cls) -> "LaunchTopology":
        return cls("single")

    @classmethod
    def resident(cls) -> "LaunchTopology":
        return cls("resident")

    @classmethod
    def ordinary_cluster(
        cls,
        cluster_size: int,
        *,
        tma_multicast: bool = False,
        peer_smem_resources: Tuple[str, ...] = tuple(),
    ) -> "LaunchTopology":
        return cls(
            "ordinary_cluster",
            cluster_size=cluster_size,
            tma_multicast=tma_multicast,
            peer_smem_resources=peer_smem_resources,
        )

    @classmethod
    def cooperative_cluster(
        cls,
        cluster_size: int,
        operand_plan: Tuple[Tuple[str, str], ...],
        *,
        tma_multicast: bool = False,
        peer_smem_resources: Tuple[str, ...] = tuple(),
        cluster_barrier_s: float = 0.0,
        cluster_issue_s: float = 0.0,
    ) -> "LaunchTopology":
        return cls(
            "cooperative_cluster",
            cluster_size=cluster_size,
            tma_multicast=tma_multicast,
            operand_plan=operand_plan,
            peer_smem_resources=peer_smem_resources,
            cluster_barrier_s=cluster_barrier_s,
            cluster_issue_s=cluster_issue_s,
        )


@dataclass(frozen=True)
class WaveCostContext:
    """Spatial context passed to an optional context-bound cost oracle."""

    wave_kind: str
    physical_sm_count: int
    logical_slot_count: int
    active_physical_sms: int
    active_logical_slots: int
    resident_work_units_per_slot: int
    work_units: int
    topology: LaunchTopology

    def __post_init__(self) -> None:
        if self.wave_kind not in ("full", "tail", "underfilled"):
            raise ModelingValidationError("unsupported wave_kind")
        for name in (
            "physical_sm_count",
            "logical_slot_count",
            "active_physical_sms",
            "active_logical_slots",
            "resident_work_units_per_slot",
            "work_units",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
            ):
                raise ModelingValidationError("%s must be positive" % name)
        if not isinstance(self.topology, LaunchTopology):
            raise ModelingValidationError("wave topology must be LaunchTopology")
        if self.active_physical_sms > self.physical_sm_count:
            raise ModelingValidationError("active physical SM count exceeds device")
        if self.active_logical_slots > self.logical_slot_count:
            raise ModelingValidationError("active logical slots exceed device")


__all__ = [
    "BOUNDARY_POLICIES",
    "LegacyStageBoundary",
    "LaunchTopology",
    "SPATIAL_CONCURRENCY_KINDS",
    "SmallTileLatencyConfig",
    "WAVE_POLICIES",
    "WaveCostContext",
]
