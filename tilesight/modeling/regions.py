"""Frozen recursive region IR for the opt-in native full-kernel executor."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from tilesight.modeling._pipeline.periodic_schedule import (
    IIScheduleMode,
    PeriodicDAG,
    SearchConfig,
)

from .errors import ModelingValidationError
from .ir import Phase, validate_identifier
from .native_policy import LegacyStageBoundary


class RegionIR:
    """Marker base for frozen recursive region nodes."""


@dataclass(frozen=True)
class PeriodicAxisIR:
    """How one long ``LoopRegion`` obtains its steady-state II.

    Exactly one of ``loop_name`` and ``dag`` is present. ``loop_name`` reuses
    the frontend-to-PeriodicDAG lowering; ``dag`` supports a native region
    whose periodic schedule has already been constructed explicitly.
    """

    loop_name: Optional[str] = None
    dag: Optional[PeriodicDAG] = None
    liveness_loop_name: Optional[str] = None
    ii_mode: str = "inherit"
    summary_policy: str = "auto"
    boundary_anchor_phase: Optional[str] = None
    legacy_stage_boundary: Optional[LegacyStageBoundary] = None
    search_config: SearchConfig = field(default_factory=SearchConfig)

    def __post_init__(self) -> None:
        if (self.loop_name is None) == (self.dag is None):
            raise ModelingValidationError(
                "PeriodicAxisIR needs exactly one of loop_name or dag"
            )
        if self.loop_name is not None:
            validate_identifier(self.loop_name, "periodic axis loop_name")
            if self.liveness_loop_name is None:
                object.__setattr__(self, "liveness_loop_name", self.loop_name)
            elif self.liveness_loop_name != self.loop_name:
                raise ModelingValidationError(
                    "loop_name and liveness_loop_name must identify the same loop"
                )
        if self.liveness_loop_name is not None:
            validate_identifier(
                self.liveness_loop_name, "periodic axis liveness_loop_name"
            )
        if self.dag is not None and not isinstance(self.dag, PeriodicDAG):
            raise ModelingValidationError("periodic axis dag must be PeriodicDAG")
        allowed_modes = ("inherit",) + tuple(item.value for item in IIScheduleMode)
        if self.ii_mode not in allowed_modes:
            raise ModelingValidationError(
                "periodic axis ii_mode must be one of %s"
                % ", ".join(allowed_modes)
            )
        if self.summary_policy not in ("auto", "inline", "macro"):
            raise ModelingValidationError(
                "summary_policy must be 'auto', 'inline', or 'macro'"
            )
        if self.boundary_anchor_phase is not None:
            validate_identifier(
                self.boundary_anchor_phase, "periodic boundary anchor phase"
            )
        if self.legacy_stage_boundary is not None and not isinstance(
            self.legacy_stage_boundary, LegacyStageBoundary
        ):
            raise ModelingValidationError(
                "legacy_stage_boundary must be LegacyStageBoundary or None"
            )
        if not isinstance(self.search_config, SearchConfig):
            raise ModelingValidationError("search_config must be SearchConfig")


@dataclass(frozen=True)
class PhaseRegion(RegionIR):
    """A leaf whose work/timing is carried by an existing frontend Phase."""

    phase: Phase
    label: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.phase, Phase):
            raise ModelingValidationError("PhaseRegion.phase must be Phase")
        if self.label:
            validate_identifier(self.label, "phase region label")

    @property
    def name(self) -> str:
        return self.label or self.phase.name

    def summary(self) -> Dict[str, Any]:
        return {
            "kind": "phase",
            "name": self.name,
            "phase": self.phase.name,
            "work": self.phase.work.kind,
        }


@dataclass(frozen=True)
class SequenceRegion(RegionIR):
    """Children that execute in explicit serial program order."""

    name: str
    children: Tuple[Any, ...]

    def __post_init__(self) -> None:
        validate_identifier(self.name, "sequence region name")
        children = tuple(self.children)
        if not children:
            raise ModelingValidationError("SequenceRegion needs at least one child")
        if not all(is_region(item) for item in children):
            raise ModelingValidationError("SequenceRegion contains a non-region child")
        object.__setattr__(self, "children", children)

    def summary(self) -> Dict[str, Any]:
        return {
            "kind": "sequence",
            "name": self.name,
            "children": tuple(region_summary(item) for item in self.children),
        }


@dataclass(frozen=True)
class LoopRegion(RegionIR):
    """A possibly nested loop with one-time prologue and epilogue regions."""

    name: str
    trip_count: int
    body: Any
    prologue: Optional[Any] = None
    epilogue: Optional[Any] = None
    periodic_axis: Optional[PeriodicAxisIR] = None

    def __post_init__(self) -> None:
        validate_identifier(self.name, "loop region name")
        if (
            not isinstance(self.trip_count, int)
            or isinstance(self.trip_count, bool)
            or self.trip_count < 0
        ):
            raise ModelingValidationError("loop trip_count must be a non-negative integer")
        if not is_region(self.body):
            raise ModelingValidationError("LoopRegion.body must be a region")
        for label in ("prologue", "epilogue"):
            value = getattr(self, label)
            if value is not None and not is_region(value):
                raise ModelingValidationError("LoopRegion.%s must be a region or None" % label)
        if self.periodic_axis is not None and not isinstance(
            self.periodic_axis, PeriodicAxisIR
        ):
            raise ModelingValidationError(
                "LoopRegion.periodic_axis must be PeriodicAxisIR or None"
            )

    def summary(self) -> Dict[str, Any]:
        return {
            "kind": "loop",
            "name": self.name,
            "trip_count": self.trip_count,
            "prologue": None
            if self.prologue is None
            else region_summary(self.prologue),
            "body": region_summary(self.body),
            "epilogue": None
            if self.epilogue is None
            else region_summary(self.epilogue),
            "periodic_axis": None
            if self.periodic_axis is None
            else {
                "loop_name": self.periodic_axis.loop_name,
                "has_explicit_dag": self.periodic_axis.dag is not None,
                "liveness_loop_name": self.periodic_axis.liveness_loop_name,
                "ii_mode": self.periodic_axis.ii_mode,
                "summary_policy": self.periodic_axis.summary_policy,
                "legacy_stage_boundary": None
                if self.periodic_axis.legacy_stage_boundary is None
                else {
                    "memory_resources": (
                        self.periodic_axis.legacy_stage_boundary.memory_resources
                    ),
                    "compute_resources": (
                        self.periodic_axis.legacy_stage_boundary.compute_resources
                    ),
                },
            },
        }


def is_region(value: Any) -> bool:
    return isinstance(value, RegionIR)


def region_summary(region: Any) -> Dict[str, Any]:
    if not is_region(region):
        raise ModelingValidationError("expected a recursive region")
    return region.summary()


def phase_regions(region: Any) -> Tuple[PhaseRegion, ...]:
    """Return leaves in explicit tree order, retaining repeated references."""

    if isinstance(region, PhaseRegion):
        return (region,)
    if isinstance(region, SequenceRegion):
        return tuple(leaf for child in region.children for leaf in phase_regions(child))
    if isinstance(region, LoopRegion):
        values = []
        if region.prologue is not None:
            values.extend(phase_regions(region.prologue))
        values.extend(phase_regions(region.body))
        if region.epilogue is not None:
            values.extend(phase_regions(region.epilogue))
        return tuple(values)
    raise ModelingValidationError("expected a recursive region")


def contains_loop(region: Any) -> bool:
    if isinstance(region, PhaseRegion):
        return False
    if isinstance(region, SequenceRegion):
        return any(contains_loop(child) for child in region.children)
    if isinstance(region, LoopRegion):
        return True
    raise ModelingValidationError("expected a recursive region")


__all__ = [
    "LoopRegion",
    "PeriodicAxisIR",
    "PhaseRegion",
    "RegionIR",
    "SequenceRegion",
    "contains_loop",
    "is_region",
    "phase_regions",
    "region_summary",
]
