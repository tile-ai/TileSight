"""Opt-in constructive scheduling for independent resident CTA streams.

The existing :class:`PeriodicDAG` describes one CTA stream.  This module
copies that stream ``R`` times and constructs a periodic schedule for one
resident group.  A group period advances every CTA stream by one logical
iteration, so aggregate work is produced every ``group_period / R``.
That quotient is an aggregate work-throughput interval, not the II of one CTA:
each individual CTA stream advances once per full ``group_period``.

This is intentionally not a cluster scheduler.  Cooperative/cluster CTAs can
share storage and work and require explicit inter-SM communication; passing
them off as independent resident CTAs is rejected by the configuration.
Nothing imports or calls this module from the default model path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

from tilesight.modeling._pipeline.periodic_schedule import (
    Dependency,
    IILowerBounds,
    NoFeasibleScheduleError,
    PeriodicDAG,
    PeriodicScheduleError,
    Phase,
    ResourceUse,
    ScheduleWitness,
    SearchConfig,
    TokenBuffer,
    UnsupportedConstraintDomainError,
    compute_lower_bounds,
    evaluate_resource_ordering,
    schedule_periodic_dag,
)


PhaseRef = Tuple[int, str]
_ABS_TOL = 1.0e-15
_REL_TOL = 1.0e-12
_INDEPENDENT_RELATIONSHIP = "independent_resident"
_ORDER_POLICIES = ("fixed", "round_robin", "cta_major")


class ResidentScheduleError(PeriodicScheduleError):
    """Invalid resident-CTA configuration or an invalid returned witness."""


class ResourceScope(str, Enum):
    """How a one-CTA resource is lifted into a resident CTA group.

    ``CTA_LOCAL`` creates one exclusive resource instance per CTA.
    ``SM_SHARED`` creates one capacity-one resource shared by every CTA.
    ``EXTERNAL_PRICED`` retains CTA-local ordering, but deliberately does not
    model cross-CTA contention: a cache/bandwidth oracle outside this scheduler
    must already have priced that contention.
    """

    CTA_LOCAL = "cta_local"
    SM_SHARED = "sm_shared"
    EXTERNAL_PRICED = "external_priced"


@dataclass(frozen=True, order=True)
class ResidentResourceRef:
    """Stable public identity for a lifted resource instance."""

    resource: str
    scope: ResourceScope
    cta: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.resource:
            raise ResidentScheduleError("resource name must not be empty")
        try:
            scope = ResourceScope(self.scope)
        except ValueError as error:
            raise ResidentScheduleError("invalid resident resource scope") from error
        object.__setattr__(self, "scope", scope)
        if scope is ResourceScope.SM_SHARED:
            if self.cta is not None:
                raise ResidentScheduleError("an sm_shared resource has no CTA owner")
        elif self.cta is None or self.cta < 0:
            raise ResidentScheduleError("a non-shared resource needs a CTA owner")


@dataclass(frozen=True)
class ResidentScheduleConfig:
    """Configuration for a bounded, explicit resident-CTA search.

    ``resource_scopes`` may omit resources; omitted resources conservatively
    use ``default_resource_scope`` (SM-shared by default).  ``fixed_orders``
    can override the deterministic ``fixed`` template for an SM-shared
    resource with a complete sequence of ``(cta, phase)`` pairs.

    ``cta_relationship`` accepts only ``"independent_resident"``.  The field
    exists to make accidental cluster-CTA use fail explicitly at the API
    boundary rather than silently changing the modeled work unit.

    For ``R>1``, only ``binary_search_steps`` and ``fixed_order_solver`` from
    ``search_config`` apply because this layer searches the bounded resident
    templates, not all expanded-DAG permutations.  For ``R=1`` the complete
    ``SearchConfig`` is forwarded unchanged to the existing scheduler.
    """

    resident_ctas: int
    resource_scopes: Mapping[str, Union[ResourceScope, str]] = field(
        default_factory=dict
    )
    default_resource_scope: Union[ResourceScope, str] = ResourceScope.SM_SHARED
    order_policies: Tuple[str, ...] = _ORDER_POLICIES
    fixed_orders: Mapping[str, Sequence[PhaseRef]] = field(default_factory=dict)
    search_config: SearchConfig = field(default_factory=SearchConfig)
    cta_relationship: str = _INDEPENDENT_RELATIONSHIP

    def __post_init__(self) -> None:
        if (
            not isinstance(self.resident_ctas, int)
            or isinstance(self.resident_ctas, bool)
            or not 1 <= self.resident_ctas <= 4
        ):
            raise ResidentScheduleError("resident_ctas must be an integer in [1, 4]")
        if self.cta_relationship != _INDEPENDENT_RELATIONSHIP:
            raise ResidentScheduleError(
                "cluster/cooperative CTAs are not independent resident CTA "
                "streams; use a cluster-specific work and communication model"
            )
        try:
            default_scope = ResourceScope(self.default_resource_scope)
        except ValueError as error:
            raise ResidentScheduleError(
                "default_resource_scope must be cta_local, sm_shared, or "
                "external_priced"
            ) from error

        scopes: Dict[str, ResourceScope] = {}
        for resource, raw_scope in self.resource_scopes.items():
            if not isinstance(resource, str) or not resource:
                raise ResidentScheduleError("resource_scopes keys must be names")
            try:
                scopes[resource] = ResourceScope(raw_scope)
            except ValueError as error:
                raise ResidentScheduleError(
                    "scope for %s must be cta_local, sm_shared, or "
                    "external_priced" % resource
                ) from error

        policies = tuple(self.order_policies)
        if not policies:
            raise ResidentScheduleError("order_policies must not be empty")
        if len(policies) != len(set(policies)):
            raise ResidentScheduleError("order_policies must not contain duplicates")
        unknown_policies = set(policies) - set(_ORDER_POLICIES)
        if unknown_policies:
            raise ResidentScheduleError(
                "unknown resident order policies %s" % sorted(unknown_policies)
            )

        fixed: Dict[str, Tuple[PhaseRef, ...]] = {}
        for resource, raw_order in self.fixed_orders.items():
            if not isinstance(resource, str) or not resource:
                raise ResidentScheduleError("fixed_orders keys must be resource names")
            order: List[PhaseRef] = []
            for item in raw_order:
                if (
                    not isinstance(item, (tuple, list))
                    or len(item) != 2
                    or not isinstance(item[0], int)
                    or isinstance(item[0], bool)
                    or not isinstance(item[1], str)
                    or not item[1]
                ):
                    raise ResidentScheduleError(
                        "fixed order entries must be (cta, phase) pairs"
                    )
                if not 0 <= item[0] < self.resident_ctas:
                    raise ResidentScheduleError(
                        "fixed order CTA index %d is outside resident group" % item[0]
                    )
                order.append((item[0], item[1]))
            if not order:
                raise ResidentScheduleError("a fixed resident order must not be empty")
            if len(order) != len(set(order)):
                raise ResidentScheduleError("a fixed resident order has duplicates")
            fixed[resource] = tuple(order)

        if not isinstance(self.search_config, SearchConfig):
            raise ResidentScheduleError("search_config must be a SearchConfig")
        object.__setattr__(self, "default_resource_scope", default_scope)
        object.__setattr__(self, "resource_scopes", MappingProxyType(scopes))
        object.__setattr__(self, "order_policies", policies)
        object.__setattr__(self, "fixed_orders", MappingProxyType(fixed))


@dataclass(frozen=True)
class ResidentScheduleWitness:
    """Constructive group schedule and aggregate work-throughput interval.

    ``effective_work_interval`` is ``group_period / resident_ctas``.  It is
    not a per-CTA II: each CTA stream repeats after ``group_period``.
    """

    candidate: str
    resident_ctas: int
    group_period: float
    effective_work_interval: float
    phase_starts: Mapping[PhaseRef, float]
    resource_orders: Mapping[ResidentResourceRef, Tuple[PhaseRef, ...]]

    def __post_init__(self) -> None:
        if not self.candidate:
            raise ResidentScheduleError("witness candidate label must not be empty")
        if self.resident_ctas <= 0:
            raise ResidentScheduleError("witness resident_ctas must be positive")
        if not math.isfinite(self.group_period) or self.group_period < 0.0:
            raise ResidentScheduleError("group_period must be finite and non-negative")
        if (
            not math.isfinite(self.effective_work_interval)
            or self.effective_work_interval < 0.0
        ):
            raise ResidentScheduleError(
                "effective_work_interval must be finite and non-negative"
            )
        expected = self.group_period / float(self.resident_ctas)
        if not _close(self.effective_work_interval, expected):
            raise ResidentScheduleError(
                "effective_work_interval must equal group_period / resident_ctas"
            )
        object.__setattr__(
            self, "phase_starts", MappingProxyType(dict(self.phase_starts))
        )
        object.__setattr__(
            self,
            "resource_orders",
            MappingProxyType(
                {
                    resource: tuple(order)
                    for resource, order in self.resource_orders.items()
                }
            ),
        )


@dataclass(frozen=True)
class ResidentScheduleResult:
    """Best/worst endpoints among the explicitly searched CTA templates."""

    resident_ctas: int
    lower_bounds: IILowerBounds
    best: ResidentScheduleWitness
    worst: ResidentScheduleWitness
    candidates_explored: int
    infeasible_candidates: int
    unsupported_candidates: int
    search_scope: str
    search_complete: bool
    resource_scopes: Mapping[str, ResourceScope]
    external_priced_resources: Tuple[str, ...]
    shared_resource_capacity: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "resource_scopes", MappingProxyType(dict(self.resource_scopes))
        )
        object.__setattr__(
            self, "external_priced_resources", tuple(self.external_priced_resources)
        )
        if self.shared_resource_capacity != 1:
            raise ResidentScheduleError("resident scheduler supports shared capacity 1")

    @property
    def overlap_sensitivity(self) -> float:
        """Searched worst-minus-best aggregate work interval."""

        return (
            self.worst.effective_work_interval
            - self.best.effective_work_interval
        )


@dataclass(frozen=True)
class _ExpandedDAG:
    dag: PeriodicDAG
    phase_to_internal: Mapping[PhaseRef, str]
    internal_to_phase: Mapping[str, PhaseRef]
    resource_to_internal: Mapping[ResidentResourceRef, str]
    internal_to_resource: Mapping[str, ResidentResourceRef]
    resource_scopes: Mapping[str, ResourceScope]
    source_orders: Mapping[str, Tuple[str, ...]]


def _close(left: float, right: float) -> bool:
    tolerance = _ABS_TOL + _REL_TOL * max(abs(left), abs(right))
    return abs(left - right) <= tolerance


def _require_ge(left: float, right: float, label: str) -> None:
    # The difference solver itself relaxes every edge with the same tolerance
    # over many iterations.  A published endpoint can accumulate a few ulps
    # beyond one local edge tolerance, so verification uses a small multiple
    # while remaining many orders below modeled ns/us quantities.
    tolerance = 8.0 * (
        _ABS_TOL + _REL_TOL * max(abs(left), abs(right))
    )
    if left + tolerance < right:
        raise ResidentScheduleError(
            "resident witness violates %s: %.17e < %.17e"
            % (label, left, right)
        )


def _resources_and_source_orders(
    dag: PeriodicDAG,
) -> Tuple[Tuple[str, ...], Dict[str, Tuple[str, ...]]]:
    users: Dict[str, List[str]] = {}
    resource_names: List[str] = []
    for phase in dag.phases:
        for use in phase.resources:
            if use.resource not in users:
                users[use.resource] = []
                resource_names.append(use.resource)
            users[use.resource].append(phase.name)
    fixed = {
        order.resource: tuple(order.phases) for order in dag.fixed_resource_orders
    }
    source_orders = {
        resource: fixed.get(resource, tuple(users[resource]))
        for resource in resource_names
    }
    return tuple(resource_names), source_orders


def _resolve_scopes(
    dag: PeriodicDAG, config: ResidentScheduleConfig
) -> Tuple[Dict[str, ResourceScope], Dict[str, Tuple[str, ...]]]:
    resources, source_orders = _resources_and_source_orders(dag)
    unknown = set(config.resource_scopes) - set(resources)
    if unknown:
        raise ResidentScheduleError(
            "resource_scopes names unknown DAG resources %s" % sorted(unknown)
        )
    unknown_fixed = set(config.fixed_orders) - set(resources)
    if unknown_fixed:
        raise ResidentScheduleError(
            "fixed_orders names unknown DAG resources %s" % sorted(unknown_fixed)
        )
    scopes = {
        resource: config.resource_scopes.get(
            resource, config.default_resource_scope
        )
        for resource in resources
    }
    for resource in config.fixed_orders:
        if scopes[resource] is not ResourceScope.SM_SHARED:
            raise ResidentScheduleError(
                "fixed resident order for %s requires sm_shared scope" % resource
            )
    return scopes, source_orders


def _expand_dag(dag: PeriodicDAG, config: ResidentScheduleConfig) -> _ExpandedDAG:
    if not isinstance(dag, PeriodicDAG):
        raise ResidentScheduleError("schedule_resident_ctas expects a PeriodicDAG")
    scopes, source_orders = _resolve_scopes(dag, config)
    phase_to_internal: Dict[PhaseRef, str] = {}
    internal_to_phase: Dict[str, PhaseRef] = {}
    for cta in range(config.resident_ctas):
        for phase_index, phase in enumerate(dag.phases):
            internal = "resident_phase_%d_%d" % (cta, phase_index)
            reference = (cta, phase.name)
            phase_to_internal[reference] = internal
            internal_to_phase[internal] = reference

    resource_to_internal: Dict[ResidentResourceRef, str] = {}
    internal_to_resource: Dict[str, ResidentResourceRef] = {}
    resource_ids: Dict[Tuple[str, Optional[int]], str] = {}
    next_resource_id = 0
    for original, scope in scopes.items():
        owners = (None,) if scope is ResourceScope.SM_SHARED else tuple(
            range(config.resident_ctas)
        )
        for owner in owners:
            internal = "resident_resource_%d" % next_resource_id
            next_resource_id += 1
            public = ResidentResourceRef(original, scope, owner)
            resource_ids[(original, owner)] = internal
            resource_to_internal[public] = internal
            internal_to_resource[internal] = public

    phases: List[Phase] = []
    for cta in range(config.resident_ctas):
        for phase in dag.phases:
            uses = []
            for use in phase.resources:
                scope = scopes[use.resource]
                owner = None if scope is ResourceScope.SM_SHARED else cta
                uses.append(
                    ResourceUse(
                        resource_ids[(use.resource, owner)],
                        use.service_time,
                        use.offset,
                    )
                )
            phases.append(
                Phase(
                    phase_to_internal[(cta, phase.name)],
                    phase.latency,
                    tuple(uses),
                    phase.iteration_offset,
                )
            )

    dependencies = tuple(
        Dependency(
            phase_to_internal[(cta, dependency.source)],
            phase_to_internal[(cta, dependency.target)],
            dependency.iteration_distance,
            dependency.min_delay,
            "cta%d:%s" % (
                cta,
                dependency.name
                or "%s->%s" % (dependency.source, dependency.target),
            ),
        )
        for cta in range(config.resident_ctas)
        for dependency in dag.dependencies
    )
    tokens = tuple(
        TokenBuffer(
            "cta%d:%s" % (cta, token.name),
            phase_to_internal[(cta, token.acquire)],
            phase_to_internal[(cta, token.release)],
            token.capacity,
            token.acquire_offset,
            token.release_offset,
            token.minimum_residence,
        )
        for cta in range(config.resident_ctas)
        for token in dag.token_buffers
    )
    expanded = PeriodicDAG(
        phases=tuple(phases),
        dependencies=dependencies,
        token_buffers=tokens,
        # Source-fixed orders become per-resource base sequences below.  A
        # shared resource needs a *complete* cross-CTA cyclic order, so copying
        # the one-CTA FixedResourceOrder here would be incomplete.
        fixed_resource_orders=tuple(),
    )
    return _ExpandedDAG(
        dag=expanded,
        phase_to_internal=MappingProxyType(phase_to_internal),
        internal_to_phase=MappingProxyType(internal_to_phase),
        resource_to_internal=MappingProxyType(resource_to_internal),
        internal_to_resource=MappingProxyType(internal_to_resource),
        resource_scopes=MappingProxyType(scopes),
        source_orders=MappingProxyType(source_orders),
    )


def _shared_order(
    resource: str,
    source_order: Sequence[str],
    config: ResidentScheduleConfig,
    policy: str,
) -> Tuple[PhaseRef, ...]:
    if policy == "fixed" and resource in config.fixed_orders:
        return tuple(config.fixed_orders[resource])
    if policy == "cta_major" or policy == "fixed":
        # ``fixed`` without a caller override preserves the source cyclic
        # sequence and rotates CTA priority at each phase.  This is a stable,
        # fair merge distinct from the two simpler templates.
        if policy == "fixed":
            return tuple(
                ((cta + position) % config.resident_ctas, phase)
                for position, phase in enumerate(source_order)
                for cta in range(config.resident_ctas)
            )
        return tuple(
            (cta, phase)
            for cta in range(config.resident_ctas)
            for phase in source_order
        )
    if policy == "round_robin":
        return tuple(
            (cta, phase)
            for phase in source_order
            for cta in range(config.resident_ctas)
        )
    raise ResidentScheduleError("unknown resident order policy %r" % policy)


def _validate_fixed_orders(
    expanded: _ExpandedDAG, config: ResidentScheduleConfig
) -> None:
    for resource, supplied in config.fixed_orders.items():
        expected = {
            (cta, phase)
            for cta in range(config.resident_ctas)
            for phase in expanded.source_orders[resource]
        }
        if len(supplied) != len(expected) or set(supplied) != expected:
            raise ResidentScheduleError(
                "fixed order for %s must list every resident CTA resource "
                "user exactly once" % resource
            )


def _candidate_orders(
    expanded: _ExpandedDAG,
    config: ResidentScheduleConfig,
    policy: str,
) -> Dict[str, Tuple[str, ...]]:
    result: Dict[str, Tuple[str, ...]] = {}
    for public, internal_resource in expanded.resource_to_internal.items():
        source_order = expanded.source_orders[public.resource]
        if public.scope is ResourceScope.SM_SHARED:
            references = _shared_order(
                public.resource, source_order, config, policy
            )
        else:
            assert public.cta is not None
            references = tuple((public.cta, phase) for phase in source_order)
        result[internal_resource] = tuple(
            expanded.phase_to_internal[reference] for reference in references
        )
    return result


def _verify_internal_witness(dag: PeriodicDAG, witness: ScheduleWitness) -> None:
    phase_by_name = {phase.name: phase for phase in dag.phases}
    expected_phases = set(phase_by_name)
    if set(witness.phase_starts) != expected_phases:
        raise ResidentScheduleError(
            "resident witness must contain every expanded phase exactly once"
        )
    for name, start in witness.phase_starts.items():
        if not math.isfinite(start):
            raise ResidentScheduleError("non-finite phase start for %s" % name)

    for dependency in dag.dependencies:
        delay = dependency.min_delay
        if delay is None:
            delay = phase_by_name[dependency.source].latency
        ready = (
            witness.phase_starts[dependency.target]
            + dependency.iteration_distance * witness.ii
        )
        required = witness.phase_starts[dependency.source] + delay
        _require_ge(ready, required, "dependency %s" % (
            dependency.name
            or "%s->%s" % (dependency.source, dependency.target)
        ))

    for token in dag.token_buffers:
        release_offset = token.release_offset
        if release_offset is None:
            release_offset = phase_by_name[token.release].latency
        acquire_event = witness.phase_starts[token.acquire] + token.acquire_offset
        release_event = witness.phase_starts[token.release] + release_offset
        _require_ge(
            release_event,
            acquire_event + token.minimum_residence,
            "token %s lifetime" % token.name,
        )
        _require_ge(
            acquire_event + token.capacity * witness.ii,
            release_event,
            "token %s reuse" % token.name,
        )

    uses: Dict[str, Dict[str, ResourceUse]] = {}
    for phase in dag.phases:
        for use in phase.resources:
            uses.setdefault(use.resource, {})[phase.name] = use
    if set(witness.resource_orders) != set(uses):
        raise ResidentScheduleError(
            "resident witness must contain every expanded resource order"
        )
    for resource, use_by_phase in uses.items():
        order = tuple(witness.resource_orders[resource])
        if len(order) != len(use_by_phase) or set(order) != set(use_by_phase):
            raise ResidentScheduleError(
                "resource order for %s must list every user exactly once" % resource
            )
        for position, current_name in enumerate(order):
            following_name = order[(position + 1) % len(order)]
            wraps = 1 if position + 1 == len(order) else 0
            current_phase = phase_by_name[current_name]
            following_phase = phase_by_name[following_name]
            current_use = use_by_phase[current_name]
            following_use = use_by_phase[following_name]
            distance = (
                wraps
                + following_phase.iteration_offset
                - current_phase.iteration_offset
            )
            following_reservation = (
                witness.phase_starts[following_name]
                + distance * witness.ii
                + following_use.offset
            )
            current_completion = (
                witness.phase_starts[current_name]
                + current_use.offset
                + current_use.service_time
            )
            _require_ge(
                following_reservation,
                current_completion,
                "resource %s capacity-one order" % resource,
            )


def _to_public_witness(
    expanded: _ExpandedDAG,
    witness: ScheduleWitness,
    resident_ctas: int,
    candidate: str,
) -> ResidentScheduleWitness:
    return ResidentScheduleWitness(
        candidate=candidate,
        resident_ctas=resident_ctas,
        group_period=witness.ii,
        effective_work_interval=witness.ii / float(resident_ctas),
        phase_starts={
            expanded.internal_to_phase[name]: start
            for name, start in witness.phase_starts.items()
        },
        resource_orders={
            expanded.internal_to_resource[resource]: tuple(
                expanded.internal_to_phase[name] for name in order
            )
            for resource, order in witness.resource_orders.items()
        },
    )


def _from_public_witness(
    expanded: _ExpandedDAG, witness: ResidentScheduleWitness
) -> ScheduleWitness:
    expected_phase_refs = set(expanded.phase_to_internal)
    if set(witness.phase_starts) != expected_phase_refs:
        raise ResidentScheduleError(
            "resident witness must contain every (cta, phase) start exactly once"
        )
    expected_resources = set(expanded.resource_to_internal)
    if set(witness.resource_orders) != expected_resources:
        raise ResidentScheduleError(
            "resident witness must contain every lifted resource order"
        )
    return ScheduleWitness(
        ii=witness.group_period,
        phase_starts=MappingProxyType(
            {
                expanded.phase_to_internal[reference]: start
                for reference, start in witness.phase_starts.items()
            }
        ),
        resource_orders=MappingProxyType(
            {
                expanded.resource_to_internal[resource]: tuple(
                    expanded.phase_to_internal[reference] for reference in order
                )
                for resource, order in witness.resource_orders.items()
            }
        ),
    )


def verify_resident_witness(
    dag: PeriodicDAG,
    config: ResidentScheduleConfig,
    witness: ResidentScheduleWitness,
) -> None:
    """Raise if a published witness violates dependency/token/resource rules."""

    if witness.resident_ctas != config.resident_ctas:
        raise ResidentScheduleError(
            "witness resident_ctas does not match the resident configuration"
        )
    expanded = _expand_dag(dag, config)
    _validate_fixed_orders(expanded, config)
    internal = _from_public_witness(expanded, witness)
    _verify_internal_witness(expanded.dag, internal)


def _lift_single_cta_witness(
    expanded: _ExpandedDAG,
    witness: ScheduleWitness,
) -> ScheduleWitness:
    phase_starts = {
        expanded.phase_to_internal[(0, name)]: start
        for name, start in witness.phase_starts.items()
    }
    resource_orders: Dict[str, Tuple[str, ...]] = {}
    for public, internal_resource in expanded.resource_to_internal.items():
        names = witness.resource_orders[public.resource]
        resource_orders[internal_resource] = tuple(
            expanded.phase_to_internal[(0, name)] for name in names
        )
    return ScheduleWitness(
        witness.ii,
        MappingProxyType(phase_starts),
        MappingProxyType(resource_orders),
    )


def schedule_resident_ctas(
    dag: PeriodicDAG, config: ResidentScheduleConfig
) -> ResidentScheduleResult:
    """Construct best/worst resident schedules among bounded order templates.

    For ``R=1`` this delegates to :func:`schedule_periodic_dag` and therefore
    preserves its best/worst envelope exactly.  For ``R>1`` only the requested
    ``fixed``/``round_robin``/``cta_major`` templates are evaluated, using the
    existing fixed-order difference/HiGHS solver and its full feasibility
    checks.  The returned endpoints are searched results, never a claim that
    every legal cross-CTA order was enumerated.
    """

    if not isinstance(config, ResidentScheduleConfig):
        raise ResidentScheduleError("config must be a ResidentScheduleConfig")
    expanded = _expand_dag(dag, config)
    _validate_fixed_orders(expanded, config)
    external = tuple(
        sorted(
            resource
            for resource, scope in expanded.resource_scopes.items()
            if scope is ResourceScope.EXTERNAL_PRICED
        )
    )

    if config.resident_ctas == 1:
        if config.fixed_orders:
            raise ResidentScheduleError(
                "fixed_orders are cross-CTA merge templates; for R=1 put a "
                "FixedResourceOrder on the source PeriodicDAG"
            )
        envelope = schedule_periodic_dag(dag, config.search_config)
        best_internal = _lift_single_cta_witness(expanded, envelope.best)
        worst_internal = _lift_single_cta_witness(expanded, envelope.worst)
        _verify_internal_witness(expanded.dag, best_internal)
        _verify_internal_witness(expanded.dag, worst_internal)
        return ResidentScheduleResult(
            resident_ctas=1,
            lower_bounds=envelope.lower_bounds,
            best=_to_public_witness(
                expanded, best_internal, 1, "single_cta_best"
            ),
            worst=_to_public_witness(
                expanded, worst_internal, 1, "single_cta_worst"
            ),
            candidates_explored=envelope.orderings_explored,
            infeasible_candidates=envelope.infeasible_orderings,
            unsupported_candidates=envelope.unsupported_orderings,
            search_scope=(
                "single_cta_model_exhaustive"
                if envelope.search_complete
                else "single_cta_searched_orderings"
            ),
            search_complete=envelope.search_complete,
            resource_scopes=expanded.resource_scopes,
            external_priced_resources=external,
        )

    seen = set()
    witnesses: List[ResidentScheduleWitness] = []
    explored = 0
    infeasible = 0
    unsupported = 0
    for policy in config.order_policies:
        orders = _candidate_orders(expanded, config, policy)
        canonical = tuple(
            (resource, tuple(order)) for resource, order in sorted(orders.items())
        )
        if canonical in seen:
            continue
        seen.add(canonical)
        explored += 1
        try:
            internal = evaluate_resource_ordering(
                expanded.dag,
                orders,
                binary_search_steps=config.search_config.binary_search_steps,
                fixed_order_solver=config.search_config.fixed_order_solver,
            )
            _verify_internal_witness(expanded.dag, internal)
            witnesses.append(
                _to_public_witness(
                    expanded, internal, config.resident_ctas, policy
                )
            )
        except UnsupportedConstraintDomainError:
            unsupported += 1
        except NoFeasibleScheduleError:
            infeasible += 1

    if not witnesses:
        if unsupported:
            raise UnsupportedConstraintDomainError(
                "all resident CTA order templates are outside the monotone "
                "minimum-II solver domain"
            )
        raise NoFeasibleScheduleError(
            "no feasible resident CTA order template was found"
        )

    def witness_key(item: ResidentScheduleWitness):
        orders = tuple(
            (
                resource,
                tuple(order),
            )
            for resource, order in sorted(item.resource_orders.items())
        )
        return item.group_period, item.candidate, orders

    best = min(witnesses, key=witness_key)
    worst = max(witnesses, key=witness_key)
    return ResidentScheduleResult(
        resident_ctas=config.resident_ctas,
        lower_bounds=compute_lower_bounds(expanded.dag),
        best=best,
        worst=worst,
        candidates_explored=explored,
        infeasible_candidates=infeasible,
        unsupported_candidates=unsupported,
        search_scope="resident_template_search",
        search_complete=False,
        resource_scopes=expanded.resource_scopes,
        external_priced_resources=external,
    )


__all__ = [
    "PhaseRef",
    "ResidentResourceRef",
    "ResidentScheduleConfig",
    "ResidentScheduleError",
    "ResidentScheduleResult",
    "ResidentScheduleWitness",
    "ResourceScope",
    "schedule_resident_ctas",
    "verify_resident_witness",
]
