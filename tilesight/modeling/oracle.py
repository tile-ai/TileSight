"""Timing-oracle protocol for the experimental modeling frontend."""

from __future__ import annotations

from typing import Protocol

from .ir import Phase, Timing


class CostOracle(Protocol):
    """Future architecture/model-bank timing provider.

    v0 examples use static ``Phase.timing`` overrides.  A later model-bank
    adapter can implement this protocol without changing frontend IR.
    """

    def resolve(self, phase: Phase) -> Timing:
        """Return completion latency and resource service for ``phase``."""


class StaticTimingOracle:
    """Resolve only explicit per-phase timing overrides."""

    def resolve(self, phase: Phase) -> Timing:
        from .errors import TimingResolutionError

        if phase.timing is None:
            raise TimingResolutionError(
                "phase %s has no static Timing override" % phase.name
            )
        return phase.timing
