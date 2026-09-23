"""Experimental periodic DAG scheduler.

This module deliberately keeps *phase latency* separate from *resource service
time*.  The former appears on data-dependency edges; the latter creates
non-preemptive reservations on exclusive resources.  Consequently, a long
asynchronous operation may have multiple iterations in flight without
artificially occupying its issue resource for its full latency.

The scheduler uses periodic difference constraints::

    start[v] >= start[u] + weight - iteration_distance * II

Dependencies, FIFO token reuse, and a selected cyclic order for every resource
are all expressed in this form.  For a small graph we enumerate the resource
orders exactly.  For a larger graph ``SearchConfig(strategy="auto")`` falls
back to a deterministic beam and marks the returned envelope as incomplete.

The default TileSight overlap path still uses its inexpensive resource-II
bound.  Kernel adapters can opt into this module through ``select_steady_ii``;
the selector keeps that lower-bound path lazy and exposes constructive periodic
best/worst witnesses with explicit search scope.

The current minimum-II solver covers the causally monotone domain in which
every directed constraint cycle has non-negative total iteration distance.
Some combinations of delayed ``ResourceUse.offset`` reservations and mixed
iteration labels can instead create a finite *upper* bound on II.  Such
orderings are counted as unsupported and force ``search_complete=False``;
if no supported witness remains, the solver raises explicitly.
"""

from __future__ import annotations

import itertools
import math
import struct
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union


_ABS_TOL = 1.0e-15
_REL_TOL = 1.0e-12

# ``_minimum_ii_difference`` performs roughly this many Python edge visits
# during its fixed-step feasibility search.  Above this point the compiled
# HiGHS LP backend is normally faster even after its per-call setup cost.  Once
# selected, a zero/high-II prefilter rejects obviously infeasible orderings
# before the LP is constructed.
_HIGHS_AUTO_WORK_THRESHOLD = 20_000

# Tiny scheduling LPs are faster as dense arrays, but resource-order searches
# can also be used on much larger DAGs.  Beyond this many matrix elements use
# an O(E)-storage COO/CSR matrix: every constraint row has at most three terms.
_HIGHS_DENSE_MATRIX_ELEMENT_LIMIT = 100_000


def _numeric_tolerance(*values: float) -> float:
    """Unit-agnostic tolerance for seconds, cycles, or normalized time."""

    return _ABS_TOL + _REL_TOL * max((abs(value) for value in values), default=0.0)


def _next_float_up(value: float) -> float:
    """Python-3.8-compatible equivalent of ``math.nextafter(value, +inf)``."""

    if math.isnan(value) or value == math.inf:
        return value
    if value == 0.0:
        return float.fromhex("0x0.0000000000001p-1022")
    bits = struct.unpack(">Q", struct.pack(">d", value))[0]
    bits = bits + 1 if value > 0.0 else bits - 1
    return struct.unpack(">d", struct.pack(">Q", bits))[0]


class PeriodicScheduleError(ValueError):
    """Base class for invalid or infeasible periodic scheduling problems."""


class NoFeasibleScheduleError(PeriodicScheduleError):
    """Raised when no searched resource ordering has a periodic schedule."""


class UnsupportedConstraintDomainError(PeriodicScheduleError):
    """Raised for an II feasibility interval outside the monotone solver."""


@dataclass(frozen=True)
class ResourceUse:
    """One non-preemptive use of an exclusive resource.

    ``service_time`` is how long the resource is occupied.  ``offset`` is the
    reservation start relative to the phase start.  A phase can contain several
    uses with the same offset to describe coupled resources (for example, a
    phase that needs both DDR and L2 for the same interval).
    """

    resource: str
    service_time: float
    offset: float = 0.0

    def __post_init__(self) -> None:
        if not self.resource:
            raise PeriodicScheduleError("resource name must not be empty")
        if not math.isfinite(self.service_time) or self.service_time <= 0.0:
            raise PeriodicScheduleError("resource service_time must be finite and positive")
        if not math.isfinite(self.offset) or self.offset < 0.0:
            raise PeriodicScheduleError("resource offset must be finite and non-negative")


@dataclass(frozen=True)
class Phase:
    """A logical phase with completion latency and resource reservations.

    ``iteration_offset`` identifies which logical iteration this phase occupies
    in a source-faithful unrolled scheduling window.  It is normally zero.  A
    mixed window such as ``PV_i, QK_i+1, PV_i`` can use offsets 0/1/0; resource
    serialization arcs then acquire signed iteration distances automatically.
    """

    name: str
    latency: float
    resources: Tuple[ResourceUse, ...] = field(default_factory=tuple)
    iteration_offset: int = 0

    def __post_init__(self) -> None:
        if not self.name:
            raise PeriodicScheduleError("phase name must not be empty")
        if not math.isfinite(self.latency) or self.latency < 0.0:
            raise PeriodicScheduleError("phase latency must be finite and non-negative")
        if not isinstance(self.iteration_offset, int):
            raise PeriodicScheduleError("phase iteration_offset must be an integer")
        object.__setattr__(self, "resources", tuple(self.resources))

        names = [use.resource for use in self.resources]
        if len(names) != len(set(names)):
            raise PeriodicScheduleError(
                "a phase may use an exclusive resource at most once; "
                "split repeated reservations into separate phases"
            )


@dataclass(frozen=True)
class Dependency:
    """A start-to-start dependency, possibly crossing loop iterations.

    By default the target waits for the source phase's full ``latency``.
    ``min_delay`` can instead specify an explicit ready offset.  The constraint
    is ``target(i + iteration_distance) >= source(i) + min_delay``.  Signed
    distances are supported because source-faithful software pipelines can
    interleave phases labelled with adjacent logical iterations.  The complete
    constraint graph must still be causal: every directed cycle must have a
    non-negative total iteration distance.
    """

    source: str
    target: str
    iteration_distance: int = 0
    min_delay: Optional[float] = None
    name: str = ""

    def __post_init__(self) -> None:
        if not self.source or not self.target:
            raise PeriodicScheduleError("dependency endpoints must not be empty")
        if not isinstance(self.iteration_distance, int):
            raise PeriodicScheduleError("dependency iteration_distance must be an integer")
        if self.min_delay is not None and (
            not math.isfinite(self.min_delay) or self.min_delay < 0.0
        ):
            raise PeriodicScheduleError("dependency min_delay must be finite and non-negative")


@dataclass(frozen=True)
class TokenBuffer:
    """A FIFO token/buffer pool with fixed capacity.

    One token is acquired at ``acquire`` and released at ``release`` for every
    logical iteration.  The pool adds the reuse constraint

    ``acquire(i + capacity) >= release(i)``

    at the selected event offsets.  This directly models double/triple
    buffering without approximating a long latency as resource occupancy.
    ``release_offset=None`` means the release phase's completion latency.
    """

    name: str
    acquire: str
    release: str
    capacity: int
    acquire_offset: float = 0.0
    release_offset: Optional[float] = None
    minimum_residence: float = 0.0

    def __post_init__(self) -> None:
        if not self.name:
            raise PeriodicScheduleError("token-buffer name must not be empty")
        if not self.acquire or not self.release:
            raise PeriodicScheduleError("token-buffer endpoints must not be empty")
        if not isinstance(self.capacity, int) or isinstance(self.capacity, bool):
            raise PeriodicScheduleError("token-buffer capacity must be an integer")
        if self.capacity <= 0:
            raise PeriodicScheduleError("token-buffer capacity must be positive")
        for value, label in (
            (self.acquire_offset, "acquire_offset"),
            (self.minimum_residence, "minimum_residence"),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise PeriodicScheduleError("%s must be finite and non-negative" % label)
        if self.release_offset is not None and (
            not math.isfinite(self.release_offset) or self.release_offset < 0.0
        ):
            raise PeriodicScheduleError("release_offset must be finite and non-negative")


@dataclass(frozen=True)
class FixedResourceOrder:
    """A source/barrier-defined cyclic order that the scheduler must preserve."""

    resource: str
    phases: Tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.resource:
            raise PeriodicScheduleError("fixed-order resource must not be empty")
        object.__setattr__(self, "phases", tuple(self.phases))
        if not self.phases:
            raise PeriodicScheduleError("fixed resource order must contain a phase")
        if len(self.phases) != len(set(self.phases)):
            raise PeriodicScheduleError("fixed resource order contains duplicate phases")


@dataclass(frozen=True)
class PeriodicDAG:
    """A loop body whose phases repeat once per initiation interval."""

    phases: Tuple[Phase, ...]
    dependencies: Tuple[Dependency, ...] = field(default_factory=tuple)
    token_buffers: Tuple[TokenBuffer, ...] = field(default_factory=tuple)
    fixed_resource_orders: Tuple[FixedResourceOrder, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "phases", tuple(self.phases))
        object.__setattr__(self, "dependencies", tuple(self.dependencies))
        object.__setattr__(self, "token_buffers", tuple(self.token_buffers))
        object.__setattr__(self, "fixed_resource_orders", tuple(self.fixed_resource_orders))
        if not self.phases:
            raise PeriodicScheduleError("a periodic DAG needs at least one phase")

        names = [phase.name for phase in self.phases]
        if len(names) != len(set(names)):
            raise PeriodicScheduleError("phase names must be unique")
        known = set(names)
        for dependency in self.dependencies:
            if dependency.source not in known or dependency.target not in known:
                raise PeriodicScheduleError(
                    "unknown dependency endpoint: %s -> %s"
                    % (dependency.source, dependency.target)
                )
        token_names = set()
        for token in self.token_buffers:
            if token.name in token_names:
                raise PeriodicScheduleError("token-buffer names must be unique")
            token_names.add(token.name)
            if token.acquire not in known or token.release not in known:
                raise PeriodicScheduleError(
                    "unknown token-buffer endpoint: %s -> %s"
                    % (token.acquire, token.release)
                )

        users_by_resource: Dict[str, List[str]] = {}
        for phase in self.phases:
            for use in phase.resources:
                users_by_resource.setdefault(use.resource, []).append(phase.name)
        fixed_resources = set()
        for fixed_order in self.fixed_resource_orders:
            if fixed_order.resource in fixed_resources:
                raise PeriodicScheduleError("a resource can have only one fixed order")
            fixed_resources.add(fixed_order.resource)
            actual_users = users_by_resource.get(fixed_order.resource)
            if actual_users is None:
                raise PeriodicScheduleError(
                    "fixed order names unused resource %s" % fixed_order.resource
                )
            if set(fixed_order.phases) != set(actual_users) or len(
                fixed_order.phases
            ) != len(actual_users):
                raise PeriodicScheduleError(
                    "fixed order for %s must list every resource user exactly once"
                    % fixed_order.resource
                )


@dataclass(frozen=True)
class SearchConfig:
    """Control resource-order search and its fixed-order minimum-II backend.

    ``fixed_order_solver="auto"`` retains cheap difference-constraint
    feasibility filters and uses the scaled SciPy HiGHS LP only when its
    compiled solve is expected to amortize setup cost.
    """

    strategy: str = "auto"  # "auto", "exact", or "beam"
    max_exact_orderings: int = 50_000
    beam_width: int = 64
    max_orders_per_resource: int = 120
    binary_search_steps: int = 70
    fixed_order_solver: str = "auto"  # "auto", "difference", or "highs"

    def __post_init__(self) -> None:
        if self.strategy not in ("auto", "exact", "beam"):
            raise PeriodicScheduleError("strategy must be 'auto', 'exact', or 'beam'")
        if self.max_exact_orderings <= 0:
            raise PeriodicScheduleError("max_exact_orderings must be positive")
        if self.beam_width <= 0:
            raise PeriodicScheduleError("beam_width must be positive")
        if self.max_orders_per_resource <= 0:
            raise PeriodicScheduleError("max_orders_per_resource must be positive")
        if self.binary_search_steps <= 0:
            raise PeriodicScheduleError("binary_search_steps must be positive")
        if self.fixed_order_solver not in ("auto", "difference", "highs"):
            raise PeriodicScheduleError(
                "fixed_order_solver must be 'auto', 'difference', or 'highs'"
            )


@dataclass(frozen=True)
class IILowerBounds:
    """Cheap necessary bounds; they are not a constructive schedule.

    ``credit_ii`` is recurrence-aware: it is the minimum II of the dependency
    graph after FIFO token-reuse edges are added.  It can therefore equal or
    include ``recurrence_ii`` rather than being an orthogonal additive term.
    Keeping it inclusive is important for paths whose signed iteration
    distances cancel before a buffer is released.
    """

    resource_ii: float
    recurrence_ii: float
    credit_ii: float

    @property
    def overall(self) -> float:
        return max(self.resource_ii, self.recurrence_ii, self.credit_ii)


@dataclass(frozen=True)
class ScheduleWitness:
    """A constructive periodic schedule for one set of resource orders."""

    ii: float
    phase_starts: Mapping[str, float]
    resource_orders: Mapping[str, Tuple[str, ...]]

    def to_dict(self) -> Dict[str, object]:
        """Return a serialization-friendly copy of the immutable witness."""

        return {
            "ii": self.ii,
            "phase_starts": dict(self.phase_starts),
            "resource_orders": dict(self.resource_orders),
        }


@dataclass(frozen=True)
class ScheduleEnvelope:
    """Best/worst work-conserving schedules among the searched orderings.

    They are global endpoints only when ``search_complete`` is true.  During a
    beam search, ``orderings_explored`` counts partial search states as well as
    completed orderings.  ``unsupported_orderings`` records candidates whose
    feasibility has a finite II upper bound outside this solver's monotone
    domain.  Either condition makes the endpoints searched-best/worst only.
    """

    lower_bounds: IILowerBounds
    best: ScheduleWitness
    worst: ScheduleWitness
    orderings_explored: int
    infeasible_orderings: int
    search_complete: bool
    unsupported_orderings: int = 0

    @property
    def overlap_sensitivity(self) -> float:
        """Absolute II spread among the searched legal serialization choices."""

        return self.worst.ii - self.best.ii


class IIScheduleMode(str, Enum):
    """Stable selector modes shared by kernel-specific DAG adapters."""

    RESOURCE_II = "resource_ii"
    PERIODIC_BEST = "periodic_best"
    PERIODIC_WORST = "periodic_worst"


class IIResultScope(str, Enum):
    """What kind of statement the selected II represents."""

    NECESSARY_BOUND = "necessary_bound"
    SEARCHED_ORDERINGS = "searched_orderings"
    MODEL_EXHAUSTIVE = "model_exhaustive"


@dataclass(frozen=True)
class IIScheduleSelection:
    """One consistently labelled steady-state II selection.

    ``resource_ii`` mode is a necessary utilization bound and deliberately has
    no witness.  Periodic modes always carry a constructive witness.  When the
    underlying search is incomplete, best/worst mean the endpoints among the
    searched work-conserving resource orders, not global extrema or hardware
    latency bounds.
    """

    mode: IIScheduleMode
    ii: float
    resource_ii: float
    scope: IIResultScope
    witness: Optional[ScheduleWitness]
    envelope: Optional[ScheduleEnvelope]
    search_complete: Optional[bool]

    @property
    def is_constructive(self) -> bool:
        return self.witness is not None

    @property
    def is_lower_bound(self) -> bool:
        return self.scope == IIResultScope.NECESSARY_BOUND

    @property
    def endpoint_scope(self) -> str:
        if self.scope == IIResultScope.NECESSARY_BOUND:
            return "lower_bound"
        if self.scope == IIResultScope.MODEL_EXHAUSTIVE:
            return "model_exhaustive"
        return "searched"

    @property
    def best_ii(self) -> Optional[float]:
        return None if self.envelope is None else self.envelope.best.ii

    @property
    def worst_ii(self) -> Optional[float]:
        return None if self.envelope is None else self.envelope.worst.ii

    @property
    def periodic_lower_bounds(self) -> Optional[IILowerBounds]:
        return None if self.envelope is None else self.envelope.lower_bounds

    @property
    def label(self) -> str:
        if self.mode == IIScheduleMode.RESOURCE_II:
            return "resource_ii_lower_bound"
        endpoint = (
            "best"
            if self.mode == IIScheduleMode.PERIODIC_BEST
            else "worst"
        )
        return "periodic_%s_%s" % (self.endpoint_scope, endpoint)


@dataclass(frozen=True)
class ResourceOrderCandidate:
    """Externally proposed cyclic orders for every non-fixed resource.

    Candidates may come from source templates or projected topological orders.
    They still receive the full dependency/token/resource feasibility check;
    proposing an order never makes it a valid periodic schedule by itself.
    """

    resource_orders: Mapping[str, Sequence[str]]
    source: str = "external"
    label: str = ""


@dataclass(frozen=True)
class _Constraint:
    source: int
    target: int
    weight: float
    iteration_distance: int
    label: str


@dataclass(frozen=True)
class _UseRef:
    phase_index: int
    use: ResourceUse


def _dependency_constraints(
    dag: PeriodicDAG, phase_index: Mapping[str, int]
) -> List[_Constraint]:
    result = []
    phases = {phase.name: phase for phase in dag.phases}
    for dependency in dag.dependencies:
        delay = dependency.min_delay
        if delay is None:
            delay = phases[dependency.source].latency
        result.append(
            _Constraint(
                source=phase_index[dependency.source],
                target=phase_index[dependency.target],
                weight=delay,
                iteration_distance=dependency.iteration_distance,
                label=dependency.name
                or "%s->%s" % (dependency.source, dependency.target),
            )
        )
    return result


def _token_constraints(
    dag: PeriodicDAG, phase_index: Mapping[str, int]
) -> Tuple[List[_Constraint], Dict[str, Tuple[_Constraint, _Constraint]]]:
    phases = {phase.name: phase for phase in dag.phases}
    result = []
    by_name = {}
    for token in dag.token_buffers:
        release_offset = token.release_offset
        if release_offset is None:
            release_offset = phases[token.release].latency

        # Ensure that a declared lifetime cannot release before it acquires,
        # even when the user did not also supply a data-dependency path.
        forward = _Constraint(
            source=phase_index[token.acquire],
            target=phase_index[token.release],
            weight=token.acquire_offset + token.minimum_residence - release_offset,
            iteration_distance=0,
            label="token:%s:lifetime" % token.name,
        )
        reuse = _Constraint(
            source=phase_index[token.release],
            target=phase_index[token.acquire],
            weight=release_offset - token.acquire_offset,
            iteration_distance=token.capacity,
            label="token:%s:reuse" % token.name,
        )
        result.extend((forward, reuse))
        by_name[token.name] = (forward, reuse)
    return result, by_name


def _resource_uses(dag: PeriodicDAG) -> Dict[str, List[_UseRef]]:
    result: Dict[str, List[_UseRef]] = {}
    for phase_index, phase in enumerate(dag.phases):
        for use in phase.resources:
            result.setdefault(use.resource, []).append(_UseRef(phase_index, use))
    return result


def _resource_constraints(
    resource: str, order: Sequence[_UseRef], phases: Sequence[Phase]
) -> List[_Constraint]:
    """Convert one cyclic resource order into periodic separation edges."""

    result = []
    count = len(order)
    if count == 0:
        return result
    for position, current in enumerate(order):
        following = order[(position + 1) % count]
        wraps = 1 if position + 1 == count else 0
        iteration_distance = (
            wraps
            + phases[following.phase_index].iteration_offset
            - phases[current.phase_index].iteration_offset
        )
        result.append(
            _Constraint(
                source=current.phase_index,
                target=following.phase_index,
                weight=(
                    current.use.offset
                    + current.use.service_time
                    - following.use.offset
                ),
                iteration_distance=iteration_distance,
                label="resource:%s:%s->%s"
                % (
                    resource,
                    phases[current.phase_index].name,
                    phases[following.phase_index].name,
                ),
            )
        )
    return result


def _relax_constraints(
    constraints: Sequence[_Constraint], phase_count: int, ii: float
) -> Tuple[bool, Tuple[float, ...]]:
    """Check difference constraints and return earliest non-negative starts."""

    starts = [0.0] * phase_count
    # This is the scheduler's dominant hot loop: FA4 evaluates hundreds of
    # resource orders, each through dozens of feasibility checks.  Spell out
    # ``_numeric_tolerance(candidate, target)`` here to avoid constructing a
    # generator and calling variadic ``max`` for every relaxed edge.  The
    # formula and operation order remain identical.
    absolute_tolerance = _ABS_TOL
    relative_tolerance = _REL_TOL
    for _ in range(max(phase_count - 1, 0)):
        changed = False
        for edge in constraints:
            candidate = (
                starts[edge.source]
                + edge.weight
                - edge.iteration_distance * ii
            )
            target = starts[edge.target]
            magnitude = abs(candidate)
            target_magnitude = abs(target)
            if target_magnitude > magnitude:
                magnitude = target_magnitude
            tolerance = absolute_tolerance + relative_tolerance * magnitude
            if candidate > target + tolerance:
                starts[edge.target] = candidate
                changed = True
        if not changed:
            break

    # A further improvement proves a positive cycle, hence infeasibility.
    for edge in constraints:
        candidate = (
            starts[edge.source] + edge.weight - edge.iteration_distance * ii
        )
        target = starts[edge.target]
        magnitude = abs(candidate)
        target_magnitude = abs(target)
        if target_magnitude > magnitude:
            magnitude = target_magnitude
        tolerance = absolute_tolerance + relative_tolerance * magnitude
        if candidate > target + tolerance:
            return False, tuple()

    minimum = min(starts) if starts else 0.0
    return True, tuple(value - minimum for value in starts)


def _has_negative_distance_cycle(
    constraints: Sequence[_Constraint], phase_count: int
) -> bool:
    """Return whether a constraint cycle travels backwards in logical time."""

    # All vertices are connected to an implicit source, so initializing every
    # distance to zero is the standard Bellman-Ford negative-cycle check.
    distance = [0] * phase_count
    for iteration in range(phase_count):
        changed = False
        for edge in constraints:
            candidate = distance[edge.source] + edge.iteration_distance
            if candidate < distance[edge.target]:
                distance[edge.target] = candidate
                changed = True
        if not changed:
            return False
        if iteration + 1 == phase_count:
            return True
    return False


def _minimum_ii_difference(
    constraints: Sequence[_Constraint], phase_count: int, binary_search_steps: int
) -> Tuple[float, Tuple[float, ...]]:
    """Find minimum II with the dependency-free difference-constraint solver."""

    if _has_negative_distance_cycle(constraints, phase_count):
        raise UnsupportedConstraintDomainError(
            "negative-total-distance cycle creates an II upper bound; this "
            "causally monotone minimum-II solver does not support bounded-II "
            "feasibility intervals"
        )

    feasible, starts = _relax_constraints(constraints, phase_count, 0.0)
    if feasible:
        return 0.0, starts

    max_positive_weight = max(
        (max(edge.weight, 0.0) for edge in constraints), default=0.0
    )
    # Every finite critical ratio has a simple-cycle numerator bounded by the
    # sum of at most ``phase_count`` positive edge weights and denominator >= 1.
    high = max(1.0, phase_count * max_positive_weight + 1.0)
    feasible, high_starts = _relax_constraints(constraints, phase_count, high)
    if not feasible:
        # Increasing II cannot break a positive cycle made entirely of
        # distance-zero constraints.
        raise NoFeasibleScheduleError(
            "positive zero-distance cycle makes this ordering infeasible"
        )

    low = 0.0
    for _ in range(binary_search_steps):
        middle = (low + high) / 2.0
        feasible, middle_starts = _relax_constraints(constraints, phase_count, middle)
        if feasible:
            high = middle
            high_starts = middle_starts
        else:
            low = middle

    # Re-evaluate at the retained feasible side to provide its witness.
    feasible, high_starts = _relax_constraints(constraints, phase_count, high)
    if not feasible:  # pragma: no cover - defensive against extreme FP inputs
        high = _next_float_up(high)
        feasible, high_starts = _relax_constraints(constraints, phase_count, high)
    if not feasible:  # pragma: no cover
        raise NoFeasibleScheduleError("failed to recover a numerical schedule witness")
    return high, high_starts


def _constraints_hold(
    constraints: Sequence[_Constraint], ii: float, starts: Sequence[float]
) -> bool:
    """Recheck a published witness after its translation to a zero origin."""

    for edge in constraints:
        candidate = (
            starts[edge.source]
            + edge.weight
            - edge.iteration_distance * ii
        )
        target = starts[edge.target]
        if candidate > target + _numeric_tolerance(candidate, target):
            return False
    return True


def _difference_with_lower_bound(
    constraints: Sequence[_Constraint],
    phase_count: int,
    binary_search_steps: int,
    minimum_ii: float,
) -> Tuple[float, Tuple[float, ...]]:
    """Run the legacy solver and canonicalize to an independent II bound."""

    ii, starts = _minimum_ii_difference(
        constraints, phase_count, binary_search_steps
    )
    if ii >= minimum_ii:
        return ii, starts

    feasible, starts = _relax_constraints(constraints, phase_count, minimum_ii)
    if not feasible:  # pragma: no cover - defensive numerical fallback
        minimum_ii = _next_float_up(minimum_ii)
        feasible, starts = _relax_constraints(
            constraints, phase_count, minimum_ii
        )
    if not feasible:  # pragma: no cover
        raise NoFeasibleScheduleError(
            "failed to recover a witness at the necessary II bound"
        )
    return minimum_ii, starts


def _build_highs_constraint_matrix(
    constraints: Sequence[_Constraint],
    phase_count: int,
    time_scale: float,
    np,
):
    """Build the scaled LP inequalities with bounded memory growth."""

    row_count = len(constraints)
    variable_count = phase_count + 1
    ii_index = phase_count
    upper_bounds = np.empty(row_count, dtype=np.float64)

    if row_count * variable_count <= _HIGHS_DENSE_MATRIX_ELEMENT_LIMIT:
        matrix = np.zeros(
            (row_count, variable_count), dtype=np.float64
        )
        for row, edge in enumerate(constraints):
            # ``+=``/``-=`` intentionally handle self edges by cancellation.
            matrix[row, edge.source] += 1.0
            matrix[row, edge.target] -= 1.0
            matrix[row, ii_index] -= float(edge.iteration_distance)
            upper_bounds[row] = -edge.weight / time_scale
        return matrix, upper_bounds

    # Import the sparse helper only for large problems.  COO deliberately
    # receives duplicate source/target coordinates for self edges; conversion
    # to CSR sums them before explicit zeros are removed.
    from scipy.sparse import coo_matrix

    entries = row_count * 3
    rows = np.empty(entries, dtype=np.int64)
    columns = np.empty(entries, dtype=np.int64)
    values = np.empty(entries, dtype=np.float64)
    for row, edge in enumerate(constraints):
        offset = row * 3
        rows[offset : offset + 3] = row
        columns[offset] = edge.source
        columns[offset + 1] = edge.target
        columns[offset + 2] = ii_index
        values[offset] = 1.0
        values[offset + 1] = -1.0
        values[offset + 2] = -float(edge.iteration_distance)
        upper_bounds[row] = -edge.weight / time_scale
    matrix = coo_matrix(
        (values, (rows, columns)),
        shape=(row_count, variable_count),
        dtype=np.float64,
    ).tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    return matrix, upper_bounds


def _minimum_ii_highs(
    constraints: Sequence[_Constraint],
    phase_count: int,
    minimum_ii: float,
) -> Optional[Tuple[float, Tuple[float, ...]]]:
    """Try a scaled HiGHS LP, returning ``None`` for a safe legacy fallback.

    Every periodic edge is the linear inequality

    ``start[source] - start[target] - distance * II <= -weight``.

    Phase starts are translation invariant, so one is anchored at zero and
    all others must be explicitly free (``linprog`` otherwise defaults them to
    non-negative).  Only the optimized II is trusted: phase starts are rebuilt
    and checked with TileSight's existing difference-constraint semantics.
    """

    try:
        # Keep the inexpensive resource-II path free of NumPy/SciPy imports.
        import numpy as np
        from scipy.optimize import linprog
    except (ImportError, ModuleNotFoundError):  # pragma: no cover - dependencies
        return None

    try:
        time_scale = max(
            abs(minimum_ii),
            max((abs(edge.weight) for edge in constraints), default=0.0),
        )
        if not math.isfinite(time_scale):
            return None
        if time_scale == 0.0:
            time_scale = 1.0

        variable_count = phase_count + 1
        ii_index = phase_count
        matrix, upper_bounds = _build_highs_constraint_matrix(
            constraints, phase_count, time_scale, np
        )
        objective = np.zeros(variable_count, dtype=np.float64)
        objective[ii_index] = 1.0
        bounds = [(None, None)] * phase_count
        if phase_count:
            bounds[0] = (0.0, 0.0)
        bounds.append((minimum_ii / time_scale, None))

        result = linprog(
            objective,
            A_ub=matrix,
            b_ub=upper_bounds,
            bounds=bounds,
            method="highs",
            options={
                "presolve": True,
                # HiGHS' smallest accepted primal/dual tolerance.  Scaling the
                # time domain first makes this meaningful for ns/us models.
                "primal_feasibility_tolerance": 1.0e-10,
                "dual_feasibility_tolerance": 1.0e-10,
                "ipm_optimality_tolerance": 1.0e-12,
            },
        )
    except Exception:  # pragma: no cover - solver/library-specific failures
        return None

    if not getattr(result, "success", False) or getattr(result, "status", None) != 0:
        return None
    try:
        ii = max(minimum_ii, float(result.fun) * time_scale)
    except (AttributeError, TypeError, ValueError):
        return None
    if not math.isfinite(ii):
        return None

    try:
        feasible, starts = _relax_constraints(constraints, phase_count, ii)
        certified = feasible and _constraints_hold(constraints, ii, starts)
    except Exception:  # pragma: no cover - defensive reconstruction fallback
        return None
    if not certified:
        return None
    return ii, starts


def _minimum_ii(
    constraints: Sequence[_Constraint],
    phase_count: int,
    binary_search_steps: int,
    *,
    fixed_order_solver: str = "difference",
    minimum_ii: float = 0.0,
) -> Tuple[float, Tuple[float, ...]]:
    """Find fixed-order minimum II with a selectable, safe solver backend."""

    if fixed_order_solver not in ("auto", "difference", "highs"):
        raise PeriodicScheduleError(
            "fixed_order_solver must be 'auto', 'difference', or 'highs'"
        )
    if not math.isfinite(minimum_ii) or minimum_ii < 0.0:
        raise PeriodicScheduleError("minimum_ii must be finite and non-negative")
    if fixed_order_solver == "difference":
        return _difference_with_lower_bound(
            constraints, phase_count, binary_search_steps, minimum_ii
        )

    work = phase_count * len(constraints) * binary_search_steps
    if (
        fixed_order_solver == "auto"
        and work < _HIGHS_AUTO_WORK_THRESHOLD
    ):
        return _difference_with_lower_bound(
            constraints, phase_count, binary_search_steps, minimum_ii
        )

    # Preserve the existing constraint-domain and feasibility classifications
    # before invoking an external numerical solver.  The high-II test cheaply
    # rejects positive zero-distance cycles, which dominate invalid beam states.
    if _has_negative_distance_cycle(constraints, phase_count):
        raise UnsupportedConstraintDomainError(
            "negative-total-distance cycle creates an II upper bound; this "
            "causally monotone minimum-II solver does not support bounded-II "
            "feasibility intervals"
        )
    feasible, starts = _relax_constraints(
        constraints, phase_count, minimum_ii
    )
    if feasible:
        return minimum_ii, starts

    max_positive_weight = max(
        (max(edge.weight, 0.0) for edge in constraints), default=0.0
    )
    high = max(
        minimum_ii,
        1.0,
        phase_count * max_positive_weight + 1.0,
    )
    high_feasible, _ = _relax_constraints(constraints, phase_count, high)
    if not high_feasible:
        raise NoFeasibleScheduleError(
            "positive zero-distance cycle makes this ordering infeasible"
        )

    result = _minimum_ii_highs(constraints, phase_count, minimum_ii)
    if result is not None:
        return result
    return _difference_with_lower_bound(
        constraints, phase_count, binary_search_steps, minimum_ii
    )


def compute_lower_bounds(dag: PeriodicDAG) -> IILowerBounds:
    """Compute resource, recurrence, and token-credit II lower bounds."""

    phase_index = {phase.name: index for index, phase in enumerate(dag.phases)}
    dependency_edges = _dependency_constraints(dag, phase_index)
    token_edges, _token_edges_by_name = _token_constraints(dag, phase_index)
    resources = _resource_uses(dag)

    resource_ii = max(
        (sum(ref.use.service_time for ref in refs) for refs in resources.values()),
        default=0.0,
    )
    recurrence_ii, _ = _minimum_ii(
        dependency_edges,
        len(dag.phases),
        70,
        fixed_order_solver="difference",
    )

    # Do not inspect only distance-zero *edges*: a legal acquire-to-release path
    # can contain +d and -d arcs whose net distance is zero.  Solving the full
    # dependency + token graph captures that path and its reuse cycle exactly.
    # This is an inclusive dependency/credit bound, not an additive component.
    if token_edges:
        credit_ii, _ = _minimum_ii(
            dependency_edges + token_edges,
            len(dag.phases),
            70,
            fixed_order_solver="difference",
        )
    else:
        credit_ii = 0.0
    return IILowerBounds(resource_ii, recurrence_ii, credit_ii)


def _full_permutations(refs: Sequence[_UseRef]) -> List[Tuple[_UseRef, ...]]:
    return list(itertools.permutations(refs))


def _limited_permutations(
    refs: Sequence[_UseRef], dag: PeriodicDAG, limit: int
) -> List[Tuple[_UseRef, ...]]:
    """Deterministic source/criticality templates for the DSE fast path."""

    source_order = tuple(refs)
    candidates: List[Tuple[_UseRef, ...]] = []
    seen = set()

    def add(order: Iterable[_UseRef]) -> None:
        value = tuple(order)
        key = tuple(ref.phase_index for ref in value)
        if key not in seen and len(candidates) < limit:
            seen.add(key)
            candidates.append(value)

    add(source_order)
    add(reversed(source_order))
    add(sorted(refs, key=lambda ref: (dag.phases[ref.phase_index].latency, ref.phase_index)))
    add(
        sorted(
            refs,
            key=lambda ref: (-dag.phases[ref.phase_index].latency, ref.phase_index),
        )
    )
    add(sorted(refs, key=lambda ref: (ref.use.service_time, ref.phase_index)))
    add(sorted(refs, key=lambda ref: (-ref.use.service_time, ref.phase_index)))

    # The cyclic cut can interact with loop-carried dependencies, so retain all
    # rotations of the source and reverse templates before local perturbations.
    for base in (source_order, tuple(reversed(source_order))):
        for shift in range(len(base)):
            add(base[shift:] + base[:shift])
    for index in range(max(len(source_order) - 1, 0)):
        swapped = list(source_order)
        swapped[index], swapped[index + 1] = swapped[index + 1], swapped[index]
        add(swapped)
    return candidates


def _candidate_orders(
    refs: Sequence[_UseRef], dag: PeriodicDAG, exhaustive: bool, config: SearchConfig
) -> List[Tuple[_UseRef, ...]]:
    if exhaustive:
        return _full_permutations(refs)
    factorial = math.factorial(len(refs))
    if factorial <= config.max_orders_per_resource:
        return _full_permutations(refs)
    return _limited_permutations(refs, dag, config.max_orders_per_resource)


def _make_witness(
    dag: PeriodicDAG,
    ii: float,
    starts: Sequence[float],
    orders: Mapping[str, Tuple[_UseRef, ...]],
) -> ScheduleWitness:
    phase_names = tuple(phase.name for phase in dag.phases)
    return ScheduleWitness(
        ii=ii,
        phase_starts=MappingProxyType(
            {
                name: starts[index]
                for index, name in enumerate(phase_names)
            }
        ),
        resource_orders=MappingProxyType(
            {
                resource: tuple(
                    phase_names[ref.phase_index] for ref in order
                )
                for resource, order in orders.items()
            }
        ),
    )


def _evaluate_ordering(
    dag: PeriodicDAG,
    base_constraints: Sequence[_Constraint],
    orders: Mapping[str, Tuple[_UseRef, ...]],
    config: SearchConfig,
    minimum_ii: float = 0.0,
) -> ScheduleWitness:
    constraints = list(base_constraints)
    for resource, order in orders.items():
        constraints.extend(_resource_constraints(resource, order, dag.phases))
    ii, starts = _minimum_ii(
        constraints,
        len(dag.phases),
        config.binary_search_steps,
        fixed_order_solver=config.fixed_order_solver,
        minimum_ii=minimum_ii,
    )
    return _make_witness(dag, ii, starts, orders)


def _resolve_named_resource_orders(
    dag: PeriodicDAG,
    resources: Mapping[str, Sequence[_UseRef]],
    resource_orders: Mapping[str, Sequence[str]],
) -> Dict[str, Tuple[_UseRef, ...]]:
    """Validate public name-based orders and resolve them to use references."""

    fixed = {
        order.resource: tuple(order.phases)
        for order in dag.fixed_resource_orders
    }
    unknown = set(resource_orders) - set(resources)
    if unknown:
        raise PeriodicScheduleError(
            "resource-order candidate names unknown resources %s"
            % sorted(unknown)
        )

    dynamic = set(resources) - set(fixed)
    missing = dynamic - set(resource_orders)
    if missing:
        raise PeriodicScheduleError(
            "resource-order candidate omits non-fixed resources %s"
            % sorted(missing)
        )

    resolved: Dict[str, Tuple[_UseRef, ...]] = {}
    for resource, references in resources.items():
        if resource in fixed:
            supplied = resource_orders.get(resource)
            if supplied is not None and tuple(supplied) != fixed[resource]:
                raise PeriodicScheduleError(
                    "candidate conflicts with fixed order for %s" % resource
                )
            names = fixed[resource]
        else:
            names = tuple(resource_orders[resource])

        reference_by_name = {
            dag.phases[reference.phase_index].name: reference
            for reference in references
        }
        if len(names) != len(reference_by_name) or set(names) != set(
            reference_by_name
        ):
            raise PeriodicScheduleError(
                "candidate order for %s must list every resource user "
                "exactly once" % resource
            )
        resolved[resource] = tuple(reference_by_name[name] for name in names)
    return resolved


def evaluate_resource_ordering(
    dag: PeriodicDAG,
    resource_orders: Mapping[str, Sequence[str]],
    *,
    binary_search_steps: int = 70,
    fixed_order_solver: str = "auto",
) -> ScheduleWitness:
    """Return the minimum-II witness for one named cyclic resource ordering.

    Fixed resource orders declared by the DAG are automatically merged.  The
    caller must name every non-fixed resource and every phase using it exactly
    once.  Full signed dependencies and token reuse are always rechecked.
    ``fixed_order_solver`` has the same ``auto``/``difference``/``highs``
    contract as :class:`SearchConfig`.
    """

    if binary_search_steps <= 0:
        raise PeriodicScheduleError("binary_search_steps must be positive")
    phase_index = {phase.name: index for index, phase in enumerate(dag.phases)}
    dependency_edges = _dependency_constraints(dag, phase_index)
    token_edges, _ = _token_constraints(dag, phase_index)
    resources = _resource_uses(dag)
    orders = _resolve_named_resource_orders(dag, resources, resource_orders)
    minimum_ii = compute_lower_bounds(dag).overall
    config = SearchConfig(
        strategy="beam",
        binary_search_steps=binary_search_steps,
        fixed_order_solver=fixed_order_solver,
    )
    return _evaluate_ordering(
        dag, dependency_edges + token_edges, orders, config, minimum_ii
    )


def _retain_beam(
    states: Sequence[Tuple[float, Dict[str, Tuple[_UseRef, ...]]]], width: int
) -> List[Tuple[float, Dict[str, Tuple[_UseRef, ...]]]]:
    """Keep both fast and slow envelopes instead of optimizing only the best."""

    if len(states) <= width:
        return list(states)
    low_count = (width + 1) // 2
    high_count = width - low_count
    ordered = sorted(states, key=lambda item: (item[0], repr(item[1])))
    retained = ordered[:low_count]
    if high_count:
        retained.extend(ordered[-high_count:])
    return retained


def schedule_periodic_dag(
    dag: PeriodicDAG,
    config: Optional[SearchConfig] = None,
    *,
    extra_orderings: Iterable[ResourceOrderCandidate] = (),
) -> ScheduleEnvelope:
    """Search resource orders and return best/worst feasible steady-state II.

    The best and worst values are *minimum feasible* IIs for their respective
    cyclic resource orders; arbitrary idle time is never counted.  Therefore
    ``worst`` is a finite overlap-sensitivity endpoint, not an unbounded delay.
    """

    if config is None:
        config = SearchConfig()

    phase_index = {phase.name: index for index, phase in enumerate(dag.phases)}
    dependency_edges = _dependency_constraints(dag, phase_index)
    token_edges, _ = _token_constraints(dag, phase_index)
    base_constraints = dependency_edges + token_edges
    lower_bounds = compute_lower_bounds(dag)
    minimum_ii = lower_bounds.overall
    resources = _resource_uses(dag)
    resource_names = sorted(resources)
    fixed_orders = {order.resource: order for order in dag.fixed_resource_orders}

    total_orderings = 1
    for resource, refs in resources.items():
        if resource not in fixed_orders:
            total_orderings *= math.factorial(len(refs))
    exact = config.strategy == "exact" or (
        config.strategy == "auto" and total_orderings <= config.max_exact_orderings
    )
    if config.strategy == "exact" and total_orderings > config.max_exact_orderings:
        raise PeriodicScheduleError(
            "exact search needs %d orderings, above max_exact_orderings=%d"
            % (total_orderings, config.max_exact_orderings)
        )

    candidate_orders = {}
    for resource in resource_names:
        if resource in fixed_orders:
            ref_by_phase = {
                dag.phases[ref.phase_index].name: ref for ref in resources[resource]
            }
            candidate_orders[resource] = [
                tuple(ref_by_phase[name] for name in fixed_orders[resource].phases)
            ]
        else:
            candidate_orders[resource] = _candidate_orders(
                resources[resource], dag, exact, config
            )

    explored = 0
    infeasible = 0
    unsupported = 0
    witnesses: List[ScheduleWitness] = []

    if exact:
        products = itertools.product(*(candidate_orders[name] for name in resource_names))
        # With no resources, product over zero inputs intentionally yields one
        # empty ordering and still schedules dependencies/tokens.
        for product in products:
            orders = dict(zip(resource_names, product))
            explored += 1
            try:
                witnesses.append(
                    _evaluate_ordering(
                        dag, base_constraints, orders, config, minimum_ii
                    )
                )
            except UnsupportedConstraintDomainError:
                unsupported += 1
            except NoFeasibleScheduleError:
                infeasible += 1
    else:
        # Incremental deterministic beam.  Partial graphs are scored by their
        # current minimum II, retaining both ends for a best/worst envelope.
        states: List[Tuple[float, Dict[str, Tuple[_UseRef, ...]]]] = [(0.0, {})]
        for resource in resource_names:
            expanded: List[Tuple[float, Dict[str, Tuple[_UseRef, ...]]]] = []
            for _score, previous_orders in states:
                for order in candidate_orders[resource]:
                    orders = dict(previous_orders)
                    orders[resource] = order
                    explored += 1
                    try:
                        witness = _evaluate_ordering(
                            dag, base_constraints, orders, config
                        )
                    except UnsupportedConstraintDomainError:
                        unsupported += 1
                        continue
                    except NoFeasibleScheduleError:
                        infeasible += 1
                        continue
                    expanded.append((witness.ii, orders))
            states = _retain_beam(expanded, config.beam_width)
            if not states:
                break

        for _score, orders in states:
            try:
                witnesses.append(
                    _evaluate_ordering(
                        dag, base_constraints, orders, config, minimum_ii
                    )
                )
            except UnsupportedConstraintDomainError:
                unsupported += 1
            except NoFeasibleScheduleError:  # pragma: no cover - already scored
                infeasible += 1

    # Kernel adapters may propose source-derived or projected topological
    # orders.  They are only candidates: resolve names, merge fixed orders,
    # deduplicate, and run the same complete periodic feasibility check.
    seen_orderings = {
        tuple(
            (resource, tuple(order))
            for resource, order in sorted(witness.resource_orders.items())
        )
        for witness in witnesses
    }
    for candidate in extra_orderings:
        if not isinstance(candidate, ResourceOrderCandidate):
            raise PeriodicScheduleError(
                "extra_orderings entries must be ResourceOrderCandidate"
            )
        orders = _resolve_named_resource_orders(
            dag, resources, candidate.resource_orders
        )
        key = tuple(
            (
                resource,
                tuple(dag.phases[ref.phase_index].name for ref in order),
            )
            for resource, order in sorted(orders.items())
        )
        if key in seen_orderings:
            continue
        seen_orderings.add(key)
        explored += 1
        try:
            witnesses.append(
                _evaluate_ordering(
                    dag, base_constraints, orders, config, minimum_ii
                )
            )
        except UnsupportedConstraintDomainError:
            unsupported += 1
        except NoFeasibleScheduleError:
            infeasible += 1

    if not witnesses:
        if unsupported:
            raise UnsupportedConstraintDomainError(
                "all searched resource orderings require bounded-II "
                "feasibility intervals unsupported by this solver"
            )
        raise NoFeasibleScheduleError("no feasible periodic resource ordering found")

    def witness_key(witness: ScheduleWitness):
        canonical_orders = tuple(
            (resource, tuple(order))
            for resource, order in sorted(witness.resource_orders.items())
        )
        return witness.ii, canonical_orders

    best = min(witnesses, key=witness_key)
    worst = max(witnesses, key=witness_key)
    return ScheduleEnvelope(
        lower_bounds=lower_bounds,
        best=best,
        worst=worst,
        orderings_explored=explored,
        infeasible_orderings=infeasible,
        search_complete=(exact or total_orderings == 1) and unsupported == 0,
        unsupported_orderings=unsupported,
    )


def select_steady_ii(
    resource_ii: float,
    mode: Union[IIScheduleMode, str] = IIScheduleMode.RESOURCE_II,
    *,
    solve_periodic: Optional[Callable[[], ScheduleEnvelope]] = None,
) -> IIScheduleSelection:
    """Select one steady-state II through the common lazy public interface.

    Parameters
    ----------
    resource_ii:
        Utilization-only necessary lower bound.  It need not be attainable.
    mode:
        ``resource_ii`` (default), ``periodic_best``, or ``periodic_worst``.
    solve_periodic:
        Zero-argument kernel-adapter callback returning a ScheduleEnvelope.
        It is required only for periodic modes and is never called in the
        default mode, preserving the fast DSE path.

    FA3, FA4, and GEMM share this selector and result contract, while their
    adapters remain responsible for constructing truthful dependencies,
    buffer capacities, and source-fixed resource orders.
    """

    if not math.isfinite(resource_ii) or resource_ii < 0.0:
        raise PeriodicScheduleError(
            "resource_ii must be finite and non-negative"
        )
    try:
        parsed_mode = mode if isinstance(mode, IIScheduleMode) else IIScheduleMode(mode)
    except ValueError as error:
        choices = ", ".join(item.value for item in IIScheduleMode)
        raise PeriodicScheduleError(
            "mode must be one of %s; got %r" % (choices, mode)
        ) from error

    if parsed_mode == IIScheduleMode.RESOURCE_II:
        return IIScheduleSelection(
            mode=parsed_mode,
            ii=resource_ii,
            resource_ii=resource_ii,
            scope=IIResultScope.NECESSARY_BOUND,
            witness=None,
            envelope=None,
            search_complete=None,
        )

    if solve_periodic is None:
        raise PeriodicScheduleError(
            "periodic modes require a solve_periodic envelope factory"
        )
    envelope = solve_periodic()
    if not isinstance(envelope, ScheduleEnvelope):
        raise PeriodicScheduleError(
            "solve_periodic must return a ScheduleEnvelope"
        )
    lower_bound_values = (
        envelope.lower_bounds.resource_ii,
        envelope.lower_bounds.recurrence_ii,
        envelope.lower_bounds.credit_ii,
    )
    if any(
        not math.isfinite(value) or value < 0.0
        for value in lower_bound_values
    ):
        raise PeriodicScheduleError(
            "periodic lower bounds must be finite and non-negative"
        )
    if not math.isfinite(envelope.best.ii) or not math.isfinite(
        envelope.worst.ii
    ):
        raise PeriodicScheduleError("periodic best/worst II must be finite")
    necessary_bound = max(resource_ii, envelope.lower_bounds.overall)
    if envelope.best.ii < necessary_bound:
        raise PeriodicScheduleError(
            "periodic best II %.17e is below necessary bound %.17e; "
            "the periodic solver or adapter must canonicalize numerical "
            "tolerance before selection"
            % (envelope.best.ii, necessary_bound)
        )
    if envelope.worst.ii < envelope.best.ii:
        raise PeriodicScheduleError(
            "periodic envelope is reversed: best II %.17e exceeds worst "
            "II %.17e" % (envelope.best.ii, envelope.worst.ii)
        )
    witness = (
        envelope.best
        if parsed_mode == IIScheduleMode.PERIODIC_BEST
        else envelope.worst
    )
    scope = (
        IIResultScope.MODEL_EXHAUSTIVE
        if envelope.search_complete
        else IIResultScope.SEARCHED_ORDERINGS
    )
    return IIScheduleSelection(
        mode=parsed_mode,
        ii=witness.ii,
        resource_ii=resource_ii,
        scope=scope,
        witness=witness,
        envelope=envelope,
        search_complete=envelope.search_complete,
    )


__all__ = [
    "Dependency",
    "FixedResourceOrder",
    "IIResultScope",
    "IIScheduleMode",
    "IIScheduleSelection",
    "IILowerBounds",
    "NoFeasibleScheduleError",
    "PeriodicDAG",
    "PeriodicScheduleError",
    "Phase",
    "ResourceUse",
    "ResourceOrderCandidate",
    "ScheduleEnvelope",
    "ScheduleWitness",
    "SearchConfig",
    "TokenBuffer",
    "UnsupportedConstraintDomainError",
    "compute_lower_bounds",
    "evaluate_resource_ordering",
    "schedule_periodic_dag",
    "select_steady_ii",
]
