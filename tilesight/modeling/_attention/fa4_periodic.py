"""Source-detailed periodic-DAG adapter for the FA4 SM100 model.

The default FA4/DSE path remains the inexpensive resource-II lower bound in
``overlap_analysis.py``.  This experimental adapter refines one KV-loop body
at the synchronization points visible in the pinned SM100 kernel:

* online softmax is split into score/row-max, the first P split, the P tail,
  and row-sum update;
* PV is split at the kernel's P-ready barrier, so its first 96/128 columns can
  overlap production of the final 32 columns;
* O and online-softmax state are explicit recurrences;
* the single physical K/V SMEM ring is released after the last QK/PV subtile;
* Tensor and K/V producer issue orders match the source program.

The ordinary, zero-distance dependency DAG may be checked with the existing
topological-sort utilities (including their optional NetworkX cross-check),
but a legal single-iteration topological order is *not* a proof of a feasible
steady-state II.  Signed loop distances, buffer reuse, and cyclic exclusive-
resource orders are solved by ``periodic_schedule.py``.

Times in this file are seconds, matching ``HardwareUsage`` and the periodic
scheduler.  Conversion to ns/us happens only in ``fa4_model.py``.
"""

from __future__ import annotations

import math
import operator
import random
from functools import lru_cache
from typing import Mapping

from tilesight.modeling._pipeline.periodic_schedule import (
    Dependency,
    FixedResourceOrder,
    PeriodicDAG,
    Phase,
    ResourceOrderCandidate,
    ResourceUse,
    ScheduleEnvelope,
    SearchConfig,
    schedule_periodic_dag,
)


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
_RESOURCE_TO_USAGE = {resource: field for field, resource in _USAGE_TO_RESOURCE}

FA4_REFERENCE_TOPOLOGY_TRIALS = 20_000

# Regression seeds recovered from the original 20k projected-topology audit.
# They are not asserted to be global extrema; retaining them guarantees that a
# later bounded-search heuristic cannot silently discard already-known legal
# witnesses.  All four are tried for every compatible q_stage=2 DAG.
_KNOWN_DYNAMIC_RESOURCE_ORDERS = (
    {
        "tmem": (
            "softmax_s0_max", "softmax_s0_p75", "softmax_s0_p25",
            "softmax_s1_max", "softmax_s1_p75", "corr_s0",
            "softmax_s1_p25", "corr_s1",
        ),
        "cuda": (
            "softmax_s0_max", "softmax_s0_p75", "softmax_s0_p25",
            "softmax_s1_max", "softmax_s1_p75", "corr_s0",
            "softmax_s0_sum", "softmax_s1_p25", "corr_s1",
            "softmax_s1_sum",
        ),
        "sfu": (
            "softmax_s0_max", "softmax_s0_p75", "softmax_s0_p25",
            "softmax_s1_max", "softmax_s1_p75", "softmax_s1_p25",
        ),
    },
    {
        "tmem": (
            "softmax_s1_max", "softmax_s0_max", "corr_s0", "corr_s1",
            "softmax_s1_p75", "softmax_s0_p75", "softmax_s1_p25",
            "softmax_s0_p25",
        ),
        "cuda": (
            "softmax_s1_max", "softmax_s0_max", "corr_s0", "corr_s1",
            "softmax_s1_p75", "softmax_s0_p75", "softmax_s1_p25",
            "softmax_s1_sum", "softmax_s0_p25", "softmax_s0_sum",
        ),
        "sfu": (
            "softmax_s1_max", "softmax_s0_max", "softmax_s1_p75",
            "softmax_s0_p75", "softmax_s1_p25", "softmax_s0_p25",
        ),
    },
    {
        "tmem": (
            "softmax_s0_max", "softmax_s0_p75", "corr_s0",
            "softmax_s0_p25", "softmax_s1_max", "softmax_s1_p75",
            "corr_s1", "softmax_s1_p25",
        ),
        "cuda": (
            "softmax_s0_max", "softmax_s0_p75", "corr_s0",
            "softmax_s0_p25", "softmax_s1_max", "softmax_s1_p75",
            "softmax_s0_sum", "corr_s1", "softmax_s1_p25",
            "softmax_s1_sum",
        ),
        "sfu": (
            "softmax_s0_max", "softmax_s0_p75", "softmax_s0_p25",
            "softmax_s1_max", "softmax_s1_p75", "softmax_s1_p25",
        ),
    },
    {
        "tmem": (
            "softmax_s1_max", "softmax_s0_max", "corr_s0", "corr_s1",
            "softmax_s1_p75", "softmax_s1_p25", "softmax_s0_p75",
            "softmax_s0_p25",
        ),
        "cuda": (
            "softmax_s1_max", "softmax_s0_max", "corr_s0", "corr_s1",
            "softmax_s1_p75", "softmax_s1_p25", "softmax_s1_sum",
            "softmax_s0_p75", "softmax_s0_p25", "softmax_s0_sum",
        ),
        "sfu": (
            "softmax_s1_max", "softmax_s0_max", "softmax_s1_p75",
            "softmax_s1_p25", "softmax_s0_p75", "softmax_s0_p25",
        ),
    },
)


def sm100_kv_stage(
    tile_m: int,
    tile_n: int,
    d: int,
    dv: int,
    q_stage: int,
    dtype_bytes: int,
) -> int:
    """Mirror the pinned single-CTA SM100 forward kernel's K/V-stage budget.

    The kernel pads D/Dv to 16 elements.  On its head-dim-192 non-persistent
    path, Q and O alias SMEM, so the budget contains their maximum rather than
    their sum.  The 192/128 uneven-layout exception then promotes two slots to
    three exactly as the source does.
    """

    if min(tile_m, tile_n, d, dv, q_stage, dtype_bytes) <= 0:
        raise ValueError("FA4 tile dimensions, stages, and dtype size must be positive")
    d_padded = math.ceil(d / 16) * 16
    dv_padded = math.ceil(dv / 16) * 16
    smem_q = q_stage * tile_m * d_padded * dtype_bytes
    smem_o = q_stage * tile_m * dv_padded * dtype_bytes
    overlap_s_o_s_q = d_padded == 192 and dv_padded >= 64
    smem_q_o = max(smem_q, smem_o) if overlap_s_o_s_q else smem_q + smem_o
    smem_kv_slot = max(
        tile_n * d_padded * dtype_bytes,
        tile_n * dv_padded * dtype_bytes,
    )
    stages = min((224 * 1024 - smem_q_o) // smem_kv_slot, 32)
    if d_padded == 192 and dv_padded == 128 and stages == 2:
        stages = 3
    if stages <= 0:
        raise ValueError(
            "FA4 SMEM budget leaves no K/V pipeline slot for "
            f"M={tile_m}, N={tile_n}, D={d}, Dv={dv}, q_stage={q_stage}"
        )
    return stages


def _phase_from_group(group, *, iteration_offset: int = 0) -> Phase:
    uses = tuple(
        ResourceUse(resource, getattr(group.usage, field))
        for field, resource in _USAGE_TO_RESOURCE
        if getattr(group.usage, field) > 0.0
    )
    return Phase(
        name=group.name,
        latency=group.latency,
        resources=uses,
        iteration_offset=iteration_offset,
    )


def _fractional_phase(
    group,
    name: str,
    fractions: Mapping[str, float],
    *,
    iteration_offset: int = 0,
) -> Phase:
    """Split one analytical op group while preserving every resource total."""

    unknown = set(fractions) - set(_RESOURCE_TO_USAGE)
    if unknown:
        raise ValueError(f"unknown resources in {name}: {sorted(unknown)}")
    nonzero_resources = {
        resource
        for field, resource in _USAGE_TO_RESOURCE
        if getattr(group.usage, field) > 0.0
    }
    missing = nonzero_resources - set(fractions)
    if missing:
        raise ValueError(
            f"split phase {name} omits nonzero resources {sorted(missing)}"
        )

    uses = []
    for resource, fraction in fractions.items():
        if not math.isfinite(fraction) or fraction < 0.0:
            raise ValueError(f"invalid {resource} fraction {fraction!r} in {name}")
        service = getattr(group.usage, _RESOURCE_TO_USAGE[resource]) * fraction
        if service > 0.0:
            uses.append(ResourceUse(resource, service))
    latency = max((use.service_time for use in uses), default=0.0)
    return Phase(
        name=name,
        latency=latency,
        resources=tuple(uses),
        iteration_offset=iteration_offset,
    )


def _softmax_phases(group, stage: int, tile_n: int, dtype_bytes: int) -> tuple[Phase, ...]:
    """Split softmax at the source's correction and P-ready barriers.

    The existing analytical group charges FP32 S reads plus P writes to TMEM,
    three equal CUDA heuristic components (row-max, P transform, row-sum), and
    M row-scale plus M*N element exponentials to SFU.  These fractions preserve
    those totals exactly while exposing the synchronization points.
    """

    expected = {"tmem", "cuda", "sfu"}
    actual = {
        resource
        for field, resource in _USAGE_TO_RESOURCE
        if getattr(group.usage, field) > 0.0
    }
    if actual != expected:
        raise ValueError(
            f"unexpected resources for {group.name}: {sorted(actual)}"
        )

    split_columns = (tile_n * 3 // 4) // 32 * 32
    if not 0 < split_columns < tile_n:
        raise ValueError(
            f"FA4 split-P barrier requires 0 < split < tile_n, got {split_columns}/{tile_n}"
        )
    first = split_columns / tile_n
    tail = 1.0 - first

    # QK accumulators are FP32 in the pinned kernel; P uses the input dtype.
    score_tmem = 4.0 / (4.0 + dtype_bytes)
    p_tmem = 1.0 - score_tmem
    row_scale_sfu = 1.0 / (tile_n + 1.0)
    p_sfu = 1.0 - row_scale_sfu
    prefix = f"softmax_s{stage}"
    return (
        _fractional_phase(
            group,
            f"{prefix}_max",
            {
                "tmem": score_tmem,
                "cuda": 1.0 / 3.0,
                "sfu": row_scale_sfu,
            },
        ),
        _fractional_phase(
            group,
            f"{prefix}_p75",
            {
                "tmem": p_tmem * first,
                "cuda": first / 3.0,
                "sfu": p_sfu * first,
            },
        ),
        _fractional_phase(
            group,
            f"{prefix}_p25",
            {
                "tmem": p_tmem * tail,
                "cuda": tail / 3.0,
                "sfu": p_sfu * tail,
            },
        ),
        _fractional_phase(
            group,
            f"{prefix}_sum",
            {"tmem": 0.0, "cuda": 1.0 / 3.0, "sfu": 0.0},
        ),
    )


def _pv_phases(group, stage: int, tile_n: int) -> tuple[Phase, Phase]:
    """Split PV where the first source-level P-ready barrier fires."""

    split_columns = (tile_n * 3 // 4) // 32 * 32
    first = split_columns / tile_n
    return (
        _fractional_phase(
            group,
            f"gemm_PV75_s{stage}",
            {"tensor": first},
        ),
        _fractional_phase(
            group,
            f"gemm_PV25_s{stage}",
            {"tensor": 1.0 - first},
        ),
    )


def build_fa4_periodic_dag(
    body,
    *,
    tile_m: int,
    tile_n: int,
    d: int,
    dv: int,
    q_stage: int,
    dtype_bytes: int,
) -> tuple[PeriodicDAG, int]:
    """Build the source-constrained repeating FA4 K/V-loop DAG.

    This adapter covers the q_stage=2 path used by the bundled 24-shape B200
    sweep.  In the natural steady window, the Tensor stream is

    ``PV0_i -> QK0_i+1 -> PV1_i -> QK1_i+1``.

    Each PV is represented by contiguous 75/25 Tensor reservations, with the P
    tail allowed to execute alongside PV75.  K/V occupy one aliased SMEM ring
    and are issued ``K_i, V_i, K_i+1, V_i+1``; parity therefore changes which
    future load reuses a released physical slot.
    """

    if q_stage != 2:
        raise NotImplementedError(
            "experimental FA4 periodic DAG currently covers q_stage=2 only"
        )
    expected = {
        "load_K",
        "load_V",
        "gemm_QK_s0",
        "softmax_s0",
        "corr_s0",
        "gemm_PV_s0",
        "gemm_QK_s1",
        "softmax_s1",
        "corr_s1",
        "gemm_PV_s1",
    }
    by_name = {group.name: group for group in body}
    if set(by_name) != expected:
        raise ValueError(
            "unexpected FA4 q_stage=2 body: "
            f"missing={sorted(expected - set(by_name))}, "
            f"extra={sorted(set(by_name) - expected)}"
        )

    phases = [
        _phase_from_group(by_name["load_K"]),
        _phase_from_group(by_name["load_V"]),
    ]
    for stage in range(q_stage):
        sm_max, sm_p75, sm_p25, sm_sum = _softmax_phases(
            by_name[f"softmax_s{stage}"], stage, tile_n, dtype_bytes
        )
        pv75, pv25 = _pv_phases(by_name[f"gemm_PV_s{stage}"], stage, tile_n)
        # This ordering is only a deterministic search seed.  Dependencies and
        # the periodic resource solver, not tuple position, define feasibility.
        phases.extend(
            (
                _phase_from_group(
                    by_name[f"gemm_QK_s{stage}"], iteration_offset=1
                ),
                sm_max,
                _phase_from_group(by_name[f"corr_s{stage}"]),
                sm_p75,
                sm_p25,
                sm_sum,
                pv75,
                pv25,
            )
        )

    dependencies = []
    for stage in range(q_stage):
        qk = f"gemm_QK_s{stage}"
        sm_max = f"softmax_s{stage}_max"
        sm_p75 = f"softmax_s{stage}_p75"
        sm_p25 = f"softmax_s{stage}_p25"
        sm_sum = f"softmax_s{stage}_sum"
        corr = f"corr_s{stage}"
        pv75 = f"gemm_PV75_s{stage}"
        pv25 = f"gemm_PV25_s{stage}"
        dependencies.extend(
            (
                Dependency("load_K", qk, name=f"K_ready_s{stage}"),
                Dependency(qk, sm_max, name=f"S_ready_s{stage}"),
                Dependency(sm_max, sm_p75, name=f"P_begin_s{stage}"),
                Dependency(sm_p75, sm_p25, name=f"P_tail_s{stage}"),
                Dependency(sm_max, corr, name=f"scale_ready_s{stage}"),
                Dependency(sm_p25, sm_sum, name=f"P_complete_s{stage}"),
                Dependency(corr, sm_sum, name=f"stats_release_s{stage}"),
                Dependency("load_V", pv75, name=f"V_ready_s{stage}"),
                Dependency(sm_p75, pv75, name=f"P75_ready_s{stage}"),
                Dependency(corr, pv75, name=f"O_rescaled_s{stage}"),
                Dependency(pv75, pv25, name=f"PV_split_s{stage}"),
                Dependency(sm_p25, pv25, name=f"P25_ready_s{stage}"),
                # Per-softmax-WG online max/sum state.
                Dependency(
                    sm_sum,
                    sm_max,
                    iteration_distance=1,
                    name=f"softmax_state_s{stage}",
                ),
                # O from iteration i is the input corrected for iteration i+1.
                Dependency(
                    pv25,
                    corr,
                    iteration_distance=1,
                    name=f"O_state_s{stage}",
                ),
                # The following QK overwrites the same S/P TMEM slot.
                Dependency(
                    pv25,
                    qk,
                    iteration_distance=1,
                    name=f"SP_slot_s{stage}",
                ),
            )
        )

    # One correction WG processes stage 0 then stage 1 on every KV block.
    dependencies.extend(
        (
            Dependency("corr_s0", "corr_s1", name="correction_stage_order"),
            Dependency(
                "corr_s1",
                "corr_s0",
                iteration_distance=1,
                name="correction_iteration_order",
            ),
        )
    )

    kv_stage = sm100_kv_stage(tile_m, tile_n, d, dv, q_stage, dtype_bytes)
    half = kv_stage // 2
    if kv_stage % 2 == 0:
        dependencies.extend(
            (
                Dependency(
                    "gemm_QK_s1", "load_K", half, name="K_ring_reuse"
                ),
                Dependency(
                    "gemm_PV25_s1", "load_V", half, name="V_ring_reuse"
                ),
            )
        )
    else:
        dependencies.extend(
            (
                Dependency(
                    "gemm_QK_s1",
                    "load_V",
                    half,
                    name="K_to_V_ring_reuse",
                ),
                Dependency(
                    "gemm_PV25_s1",
                    "load_K",
                    half + 1,
                    name="V_to_K_ring_reuse",
                ),
            )
        )

    fixed_orders = [
        FixedResourceOrder(
            "tensor",
            (
                "gemm_PV75_s0",
                "gemm_PV25_s0",
                "gemm_QK_s0",
                "gemm_PV75_s1",
                "gemm_PV25_s1",
                "gemm_QK_s1",
            ),
        )
    ]
    # The producer issues K then V.  Preserve that order on every modeled
    # memory channel used by both load phases.
    for resource in ("ddr", "l2", "l1_5", "smem"):
        users = [
            phase.name
            for phase in phases
            if any(use.resource == resource for use in phase.resources)
        ]
        if users:
            if set(users) != {"load_K", "load_V"}:
                raise ValueError(
                    f"cannot apply FA4 fixed {resource} order to users {users}"
                )
            fixed_orders.append(FixedResourceOrder(resource, ("load_K", "load_V")))

    return (
        PeriodicDAG(
            phases=tuple(phases),
            dependencies=tuple(dependencies),
            fixed_resource_orders=tuple(fixed_orders),
        ),
        kv_stage,
    )


def _resource_users(dag: PeriodicDAG) -> dict[str, tuple[str, ...]]:
    """Return source-ordered phase names for every modeled resource."""

    users = {}
    for phase in dag.phases:
        for use in phase.resources:
            users.setdefault(use.resource, []).append(phase.name)
    return {resource: tuple(names) for resource, names in users.items()}


def _extra_ordering_candidates(
    dag: PeriodicDAG,
    sample_seed: int,
    topology_trials: int,
):
    """Yield deduplicated, source-derived public resource-order candidates.

    A zero-distance topological order is only a proposal.  The public periodic
    engine subsequently merges fixed orders and checks the complete signed
    dependency, buffer-credit, and resource constraint system.
    """

    phase_index = {phase.name: index for index, phase in enumerate(dag.phases)}
    users = _resource_users(dag)
    fixed_resources = {order.resource for order in dag.fixed_resource_orders}
    dynamic_resources = tuple(
        resource for resource in sorted(users) if resource not in fixed_resources
    )

    zero_successors = {phase.name: [] for phase in dag.phases}
    zero_indegree = {phase.name: 0 for phase in dag.phases}
    for dependency in dag.dependencies:
        if dependency.iteration_distance == 0:
            zero_successors[dependency.source].append(dependency.target)
            zero_indegree[dependency.target] += 1

    generator = random.Random(sample_seed)

    def topological_order(trial):
        indegree = dict(zero_indegree)
        ready = [name for name, degree in indegree.items() if degree == 0]
        result = []
        while ready:
            if trial == 0:
                ready.sort(key=lambda name: phase_index[name])
                position = 0
            else:
                position = generator.randrange(len(ready))
            name = ready.pop(position)
            result.append(name)
            for successor in zero_successors[name]:
                indegree[successor] -= 1
                if indegree[successor] == 0:
                    ready.append(successor)
        if len(result) != len(dag.phases):
            raise ValueError("zero-distance FA4 dependency graph contains a cycle")
        return tuple(result)

    def project(global_order):
        rank = {name: index for index, name in enumerate(global_order)}
        return {
            resource: tuple(sorted(users[resource], key=rank.__getitem__))
            for resource in dynamic_resources
        }

    seen = set()

    def candidate(orders, source, label):
        key = tuple(
            (resource, tuple(orders[resource])) for resource in dynamic_resources
        )
        if key in seen:
            return None
        seen.add(key)
        return ResourceOrderCandidate(
            resource_orders={
                resource: tuple(orders[resource])
                for resource in dynamic_resources
            },
            source=source,
            label=label,
        )

    source_orders = project(topological_order(0))
    proposed = candidate(source_orders, "zero_distance_topology", "source_order")
    if proposed is not None:
        yield proposed

    for index, known in enumerate(_KNOWN_DYNAMIC_RESOURCE_ORDERS):
        orders = dict(source_orders)
        orders.update(
            (resource, order)
            for resource, order in known.items()
            if resource in dynamic_resources
        )
        proposed = candidate(orders, "regression_witness", f"known_{index}")
        if proposed is not None:
            yield proposed

    for trial in range(topology_trials):
        proposed = candidate(
            project(topological_order(trial)),
            "sampled_zero_distance_topology",
            f"trial_{trial}",
        )
        if proposed is not None:
            yield proposed


@lru_cache(maxsize=128)
def _schedule_cached(
    dag: PeriodicDAG,
    sample_seed: int,
    topology_trials: int,
) -> ScheduleEnvelope:
    """Return a bounded, reproducible envelope containing known witnesses.

    Detailed splitting creates millions of possible global CUDA/TMEM/SFU
    cyclic orders.  We union three bounded candidate families through the
    public ``extra_orderings`` interface:

    1. the generic scheduler's deterministic beam;
    2. a deterministic source projection plus previously found regression orders;
    3. ``topology_trials`` fixed-seed random zero-distance topological orders,
       projected onto each resource and checked by the periodic engine.

    The projection uses a single global topological order only to propose
    mutually consistent resource orders.  It does not decide II feasibility;
    every projection is evaluated with the full signed-distance periodic
    constraints.  This is sampled/bounded, so ``search_complete`` is always
    false and best/worst mean extrema among the evaluated legal orders.
    """

    return schedule_periodic_dag(
        dag,
        SearchConfig(
            strategy="auto",
            max_exact_orderings=50_000,
            beam_width=64,
            max_orders_per_resource=96,
            binary_search_steps=60,
        ),
        extra_orderings=_extra_ordering_candidates(
            dag, sample_seed, topology_trials
        ),
    )


def analyze_fa4_periodic(
    body,
    *,
    tile_m: int,
    tile_n: int,
    d: int,
    dv: int,
    q_stage: int,
    dtype_bytes: int,
    topology_trials: int = FA4_REFERENCE_TOPOLOGY_TRIALS,
) -> tuple[ScheduleEnvelope, int]:
    """Analyze one FA4 body with an explicit randomized-proposal budget.

    ``topology_trials=0`` still evaluates the generic beam, the source-order
    projection, and the retained regression witnesses; it only disables new
    random topological proposals.  The normalized integer budget is always an
    explicit part of the schedule-cache key.
    """

    if isinstance(topology_trials, bool):
        raise ValueError("topology_trials must be a non-negative integer")
    try:
        topology_trials = int(operator.index(topology_trials))
    except TypeError:
        raise ValueError(
            "topology_trials must be a non-negative integer"
        ) from None
    if topology_trials < 0:
        raise ValueError("topology_trials must be a non-negative integer")

    dag, kv_stage = build_fa4_periodic_dag(
        body,
        tile_m=tile_m,
        tile_n=tile_n,
        d=d,
        dv=dv,
        q_stage=q_stage,
        dtype_bytes=dtype_bytes,
    )
    return _schedule_cached(dag, 1000 + d, topology_trials), kv_stage


__all__ = [
    "FA4_REFERENCE_TOPOLOGY_TRIALS",
    "analyze_fa4_periodic",
    "build_fa4_periodic_dag",
    "sm100_kv_stage",
]
