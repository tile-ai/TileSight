"""Resident CTAs per SM, decided by TileSight's own storage accounting.

The residency is the largest one, at most the architecture's ``max_blocks_per_sm``,
at which TileSight finds that the storage one CTA declares fits the SM:

* the program layer's capacity check: shared and tensor memory of one work unit
  (``Launch.storage_bytes``) against the architecture's capacity divided by the
  residency, the comparison ``analyze`` itself makes (and repeats, with
  ``capacity_policy="error"``, when the kernel is analyzed);
* TileSight's liveness analysis (``tilesight.modeling.liveness``) of every work
  group, which also holds register fragments against ``register_capacity_per_sm``.

No compiler runs. What TileSight does not count does not limit the residency:
registers the compiler allocates beyond the declared buffers, register buffers
that the liveness analysis does not see (it covers pipelines), and per-SM thread
limits, which TileSight's architecture does not describe. When even one CTA
exceeds a capacity, the residency is one and the excess is recorded in the notes,
as TileSight's liveness report records it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Set, Tuple

from ..modeling.liveness import analyze_liveness
from ..modeling.program.lowering import lower_launch
from .errors import Refusal

_CAPACITY = {"smem": "configurable_smem_capacity", "tmem": "tmem_capacity_per_sm", "register": "register_capacity_per_sm"}


@dataclass(frozen=True)
class Residency:
    resident_ctas_per_sm: int
    max_blocks_per_sm: int
    binding: Tuple[str, ...]                          # what keeps one more CTA out
    storage_per_cta: Tuple[Tuple[str, float], ...]    # bytes per CTA that TileSight checks, by storage
    capacity_per_sm: Tuple[Tuple[str, float], ...]
    unchecked_registers: Tuple[str, ...] = ()         # declared register buffers the liveness analysis does not see
    notes: Tuple[str, ...] = field(default=())


def _kib(value: float) -> str:
    return "%.1f KiB" % (value / 1024.0)


def solve(program: Any, arch: Any, options: Any) -> Residency:
    launch = program.launches[0]
    ceiling = getattr(arch, "max_blocks_per_sm", None)
    if not isinstance(ceiling, int) or isinstance(ceiling, bool) or ceiling < 1:
        raise Refusal("ARCH_RESIDENCY_UNKNOWN", "the TileSight architecture has no max_blocks_per_sm")
    capacity = {s: float(getattr(arch, a, 0) or 0) for s, a in _CAPACITY.items()}
    declared = launch.storage_bytes()
    kernels = [lower_launch(launch, group_id, group, options).kernel
               for group_id, group in launch.effective_work_units().groups]

    def exceeded(residency: int) -> Set[str]:
        over = {s for s, amount in declared.items() if capacity.get(s) and amount > capacity[s] / residency + 1e-9}
        for kernel in kernels:
            over.update(item.storage_scope for item in analyze_liveness(kernel, arch, resident_ctas_per_sm=residency).capacity
                        if item.exceeds_capacity is True)
        return over

    per_cta: Dict[str, float] = dict(declared)
    for kernel in kernels:
        seen: Dict[str, float] = {}
        for item in analyze_liveness(kernel, arch, resident_ctas_per_sm=1).capacity:
            scope = "register" if item.storage_scope == "fragment" else item.storage_scope
            seen[scope] = seen.get(scope, 0.0) + item.aggregate_peak_bytes
        for scope, amount in seen.items():
            per_cta[scope] = max(per_cta.get(scope, 0.0), amount)
    buffers, _groups = launch.allocations()
    registers = sorted(name for name, item in buffers.items() if item.storage in ("register", "fragment"))
    unchecked = tuple(registers) if registers and not per_cta.get("register") else ()

    stored = ", ".join("%s %s" % (s, _kib(v)) for s, v in sorted(per_cta.items()) if v) or "none"
    over = exceeded(1)
    if over:
        # Recorded, not refused, as TileSight's liveness report does; the program layer's own shared-memory check
        # still refuses the analysis when shared memory is what does not fit.
        residency, binding = 1, tuple(sorted(over))
        notes = ["one CTA already exceeds the SM's %s (%s); TileSight records it and the analysis uses one CTA per SM"
                 % (" and ".join(binding), ", ".join("%s %s of %s" % (s, _kib(per_cta.get(s, 0.0)), _kib(capacity.get(s, 0.0)))
                                                     for s in binding))]
    else:
        residency = next(r for r in range(ceiling, 0, -1) if not exceeded(r))
        if residency == ceiling:
            binding = ("max_blocks_per_sm",)
            notes = ["residency %d is TileSight's max_blocks_per_sm: no declared storage limits it (per CTA: %s)"
                     % (residency, stored)]
        else:
            binding = tuple(sorted(exceeded(residency + 1)))
            notes = ["residency %d: one more CTA would exceed the SM's %s (per CTA: %s)"
                     % (residency, " and ".join(binding), stored)]
    if unchecked:
        notes.append("register buffers %s are outside TileSight's liveness analysis (it covers pipelines) and do not "
                     "limit the residency" % ", ".join(unchecked))
    if residency > 1:
        notes.append("registers the compiler allocates beyond the declared buffers and the per-SM thread limit are not "
                     "in TileSight's model, so the residency can exceed what the hardware allows")
    return Residency(residency, ceiling, binding, tuple(sorted(per_cta.items())),
                     tuple(sorted((s, v) for s, v in capacity.items() if v)), unchecked, tuple(notes))
