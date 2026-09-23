"""Experimental periodic-DAG adapter for the six-phase FA3 SM90 model.

The default FA3 path remains the cheap resource-II lower bound in
``fa3_model.py``.  This sidecar converts the existing analytical phase costs to
the constructive periodic scheduler without refitting any cost or utilization
parameter.  It adds only source-structural constraints:

* independent two-stage K and V producer/consumer pipelines;
* online-softmax and O-accumulator loop-carried recurrences;
* source steady tensor order QK(i+1) -> PV(i); and
* producer issue order K(i+1) -> V(i).

The six-phase abstraction cannot represent the instruction-level SM90
``IntraWGOverlap`` interleave (QK of the next block can be issued before PV of
the current block), partial GMMA completion, or exact warp-scheduler barriers.
The best/worst result is therefore an explicit scheduling sensitivity, not a
replacement for source-level simulation.
"""

from __future__ import annotations

from functools import lru_cache

from tilesight.modeling._pipeline.periodic_schedule import (
    Dependency,
    FixedResourceOrder,
    PeriodicDAG,
    Phase,
    ResourceUse,
    ScheduleEnvelope,
    SearchConfig,
    TokenBuffer,
    schedule_periodic_dag,
    select_steady_ii,
)


HOPPER_KV_PIPELINE_STAGES = 2

_EXPECTED_PHASES = {
    "load_K",
    "load_V",
    "gemm_QK",
    "online_softmax",
    "rescale_O",
    "gemm_PV",
}

_USAGE_TO_RESOURCE = (
    ("ddr_time", "ddr"),
    ("l2_time", "l2"),
    ("l1_5_time", "l1_5"),
    ("smem_time", "smem"),
    ("tmem_time", "tmem"),
    ("tensor_time", "tensor"),
    ("cuda_time", "cuda"),
    ("sfu_time", "sfu"),
    ("network_time", "network"),
)


def _phase_from_group(group) -> Phase:
    """Preserve latency and service demand as separate quantities."""

    uses = tuple(
        ResourceUse(resource, getattr(group.usage, field))
        for field, resource in _USAGE_TO_RESOURCE
        if getattr(group.usage, field) > 0.0
    )
    # The IntraWGOverlap source window issues next-iteration K/QK before
    # current-iteration V/PV.  Softmax and O correction follow the QK whose
    # scores they consume, hence they use the same +1 label in this window.
    iteration_offset = (
        1
        if group.name
        in {"load_K", "gemm_QK", "online_softmax", "rescale_O"}
        else 0
    )
    return Phase(
        group.name,
        latency=group.latency,
        resources=uses,
        iteration_offset=iteration_offset,
    )


def build_fa3_periodic_dag(
    groups, *, pipeline_stages: int = HOPPER_KV_PIPELINE_STAGES
) -> PeriodicDAG:
    """Convert the existing six phase body to a source-constrained DAG."""

    groups = tuple(groups)
    if pipeline_stages != HOPPER_KV_PIPELINE_STAGES:
        raise ValueError(
            "the audited SM90 FA3 path has exactly two K/V stages; "
            f"got {pipeline_stages}"
        )
    by_name = {group.name: group for group in groups}
    if set(by_name) != _EXPECTED_PHASES or len(groups) != len(_EXPECTED_PHASES):
        raise ValueError(
            "unexpected FA3 body: "
            f"missing={sorted(_EXPECTED_PHASES - set(by_name))}, "
            f"extra={sorted(set(by_name) - _EXPECTED_PHASES)}"
        )

    # Retain the caller's source order in the phase tuple.  ``groups`` is a
    # tuple at the model hook, but normalizing here also makes the public helper
    # deterministic for list callers.
    phases = tuple(_phase_from_group(group) for group in groups)
    dependencies = [
        Dependency(predecessor.name, group.name)
        for group in groups
        for predecessor in group.depends_on
    ]
    dependencies.extend(
        (
            # In the IntraWGOverlap producer loop each step issues K for the
            # next logical iteration before V for the current iteration.  V
            # does not wait for K's full data latency; exclusive reservations
            # provide the actual channel separation.
            Dependency(
                "load_K",
                "load_V",
                iteration_distance=-1,
                min_delay=0.0,
                name="producer_K_next_before_V_current",
            ),
            Dependency(
                "online_softmax",
                "online_softmax",
                iteration_distance=1,
                name="online_softmax_state",
            ),
            Dependency(
                "gemm_PV",
                "rescale_O",
                iteration_distance=1,
                name="O_accumulator_state",
            ),
        )
    )

    tokens = (
        TokenBuffer(
            "K_pipeline",
            acquire="load_K",
            release="gemm_QK",
            capacity=pipeline_stages,
        ),
        TokenBuffer(
            "V_pipeline",
            acquire="load_V",
            release="gemm_PV",
            capacity=pipeline_stages,
        ),
    )

    fixed_orders = [
        FixedResourceOrder("tensor", ("gemm_QK", "gemm_PV")),
    ]
    # DDR and L2 are used only by the producer phases in this six-phase body,
    # so their cyclic order and phase iteration offsets directly express
    # K(i+1) -> V(i) -> K(i+2).
    # SMEM can additionally be used by the SS-PV softmax path; leave that
    # relative placement searchable while the zero-delay K->V edge preserves
    # producer order.
    for resource in ("ddr", "l2", "smem"):
        users = tuple(
            phase.name
            for phase in phases
            if any(use.resource == resource for use in phase.resources)
        )
        if set(users) == {"load_K", "load_V"} and len(users) == 2:
            fixed_orders.append(
                FixedResourceOrder(resource, ("load_K", "load_V"))
            )

    return PeriodicDAG(
        phases=phases,
        dependencies=tuple(dependencies),
        token_buffers=tokens,
        fixed_resource_orders=tuple(fixed_orders),
    )


@lru_cache(maxsize=128)
def _schedule_cached(dag: PeriodicDAG) -> ScheduleEnvelope:
    # At this granularity the only unresolved cyclic orders are CUDA and, for
    # the SS-PV d=64 non-causal path, SMEM.  Exhaustive search is tiny and keeps
    # both endpoints exact relative to this abstraction.
    return schedule_periodic_dag(
        dag,
        SearchConfig(
            strategy="exact",
            max_exact_orderings=50_000,
            binary_search_steps=60,
        ),
    )


def analyze_fa3_periodic(
    groups, *, pipeline_stages: int = HOPPER_KV_PIPELINE_STAGES
) -> ScheduleEnvelope:
    """Return the complete best/worst periodic-II envelope for one body."""

    dag = build_fa3_periodic_dag(groups, pipeline_stages=pipeline_stages)
    return _schedule_cached(dag)


def _hook_result(selection: str, **kwargs):
    groups = tuple(kwargs["groups"])
    pipeline_stages = int(kwargs["pipeline_stages"])
    resource_ii_s = float(kwargs["resource_ii_s"])
    envelope = analyze_fa3_periodic(groups, pipeline_stages=pipeline_stages)
    if not envelope.search_complete:
        raise RuntimeError("FA3 periodic envelope unexpectedly used an incomplete search")
    mode = "periodic_best" if selection == "best" else "periodic_worst"
    selected = select_steady_ii(
        resource_ii_s,
        mode,
        solve_periodic=lambda: envelope,
    )
    best_ii_s = envelope.best.ii
    worst_ii_s = envelope.worst.ii
    metadata = {
        "periodic_selection": selection,
        "periodic_best_ii_ns": best_ii_s * 1e9,
        "periodic_worst_ii_ns": worst_ii_s * 1e9,
        "periodic_overlap_spread_ns": (worst_ii_s - best_ii_s) * 1e9,
        "periodic_resource_lb_ns": envelope.lower_bounds.resource_ii * 1e9,
        "periodic_recurrence_lb_ns": envelope.lower_bounds.recurrence_ii * 1e9,
        "periodic_credit_lb_ns": envelope.lower_bounds.credit_ii * 1e9,
        "periodic_orderings_explored": envelope.orderings_explored,
        "periodic_infeasible_orderings": envelope.infeasible_orderings,
        "periodic_unsupported_orderings": envelope.unsupported_orderings,
        "periodic_search_complete": envelope.search_complete,
        "periodic_best_phase_starts_ns": {
            name: value * 1e9 for name, value in envelope.best.phase_starts.items()
        },
        "periodic_worst_phase_starts_ns": {
            name: value * 1e9 for name, value in envelope.worst.phase_starts.items()
        },
        "periodic_best_resource_orders": dict(envelope.best.resource_orders),
        "periodic_worst_resource_orders": dict(envelope.worst.resource_orders),
    }
    return selected.ii, metadata


def fa3_periodic_best(**kwargs):
    """FA3 model hook selecting the fastest constructive periodic witness."""

    return _hook_result("best", **kwargs)


def fa3_periodic_worst(**kwargs):
    """FA3 model hook selecting the slowest work-conserving periodic witness."""

    return _hook_result("worst", **kwargs)


__all__ = [
    "HOPPER_KV_PIPELINE_STAGES",
    "analyze_fa3_periodic",
    "build_fa3_periodic_dag",
    "fa3_periodic_best",
    "fa3_periodic_worst",
]
