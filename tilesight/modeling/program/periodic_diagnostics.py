"""Inspectable facts about one periodic schedule: DAG capture, bounds, critical cycle.

``critical_cycle`` answers *why* a witness has its II: for fixed resource
orders the minimum II is the maximum over constraint cycles of
``sum(weight) / sum(iteration_distance)``; the function returns the edges of a
cycle that attains it (data dependencies, carried state, buffer credit/token
reuse and resource-order separations keep their own labels).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from tilesight.modeling._pipeline import periodic_schedule as ps

from .. import native_executor


@contextmanager
def capture_periodic_dags() -> Iterator[List[Tuple[Any, Any]]]:
    """Collect every ``(PeriodicDAG, ScheduleEnvelope)`` the analysis solves inside the block."""

    captured = []  # type: List[Tuple[Any, Any]]
    original = native_executor._schedule_periodic_dag_cached

    def recording(dag: Any, *args: Any, **kwargs: Any) -> Any:
        envelope = original(dag, *args, **kwargs)
        captured.append((dag, envelope))          # (dag, envelope); native_executor.periodic_dag_digest(dag) is its identity
        return envelope

    native_executor._schedule_periodic_dag_cached = recording
    try:
        yield captured
    finally:
        native_executor._schedule_periodic_dag_cached = original


def resource_demand(dag: Any) -> Dict[str, float]:
    """Per-iteration service demand of every resource (the resource bound is their maximum)."""

    totals = {}  # type: Dict[str, float]
    for phase in dag.phases:
        for use in phase.resources:
            totals[use.resource] = totals.get(use.resource, 0.0) + use.service_time
    return dict(sorted(totals.items(), key=lambda item: -item[1]))


def _constraints(dag: Any, resource_orders: Mapping[str, Sequence[str]]) -> List[Any]:
    phase_index = {phase.name: index for index, phase in enumerate(dag.phases)}
    edges = list(ps._dependency_constraints(dag, phase_index))
    edges += ps._token_constraints(dag, phase_index)[0]
    resources = ps._resource_uses(dag)
    orders = ps._resolve_named_resource_orders(dag, resources, resource_orders)
    for resource, order in orders.items():
        edges += ps._resource_constraints(resource, order, dag.phases)
    return edges


def critical_cycle(dag: Any, resource_orders: Mapping[str, Sequence[str]], ii: float) -> Dict[str, Any]:
    """A constraint cycle whose ``sum(weight) / sum(distance)`` equals the schedule's minimum II."""

    edges = _constraints(dag, resource_orders)
    names = [phase.name for phase in dag.phases]
    probe = ii * (1.0 - 1e-6)
    distance = [0.0] * len(names)
    parent = [None] * len(names)  # type: List[Optional[Any]]
    touched = None
    for _round in range(len(names) + 1):
        touched = None
        for edge in edges:
            value = distance[edge.source] + edge.weight - probe * edge.iteration_distance
            if value > distance[edge.target] + 1e-18:
                distance[edge.target] = value
                parent[edge.target] = edge
                touched = edge.target
        if touched is None:
            break
    if touched is None:
        return {"found": False, "note": "no positive cycle just below the reported II (II is not cycle-bound here)"}
    node = touched
    for _ in range(len(names)):
        node = parent[node].source
    cycle = []
    start = node
    while True:
        edge = parent[node]
        cycle.append(edge)
        node = edge.source
        if node == start:
            break
    cycle.reverse()
    weight = sum(edge.weight for edge in cycle)
    hops = sum(edge.iteration_distance for edge in cycle)
    return {
        "found": True, "sum_weight_s": weight, "sum_iteration_distance": hops,
        "cycle_ii_s": weight / hops if hops else None,
        "edges": [{"from": names[edge.source], "to": names[edge.target], "weight_s": edge.weight,
                   "iteration_distance": edge.iteration_distance, "label": edge.label,
                   "kind": ("resource_order" if edge.label.startswith("resource:") else
                            "buffer_credit" if edge.label.startswith("token:") else "dependency")}
                  for edge in cycle],
    }


def check_witness(dag: Any, starts: Mapping[str, float], ii: float) -> Dict[str, Any]:
    """Check dependencies, buffer credits and resource occupancy at fixed start times and II."""

    phase_index = {phase.name: index for index, phase in enumerate(dag.phases)}
    names = [phase.name for phase in dag.phases]
    missing = [name for name in names if name not in starts]
    edges = list(ps._dependency_constraints(dag, phase_index)) + ps._token_constraints(dag, phase_index)[0]
    violations = []
    tolerance = 1e-12
    for edge in edges:
        source, target = names[edge.source], names[edge.target]
        if source in missing or target in missing:
            continue
        slack = starts[target] + ii * edge.iteration_distance - starts[source] - edge.weight
        if slack < -tolerance:
            violations.append({"kind": "buffer_credit" if edge.label.startswith("token:") else "dependency",
                               "constraint": edge.label, "from": source, "to": target,
                               "iteration_distance": edge.iteration_distance, "violated_by_ns": -slack * 1e9})
    overlaps = []
    by_resource = {}  # type: Dict[str, List[Tuple[str, float, float]]]
    for phase in dag.phases:
        if phase.name in missing:
            continue
        for use in phase.resources:
            begin = (starts[phase.name] + use.offset) % ii
            by_resource.setdefault(use.resource, []).append((phase.name, begin, use.service_time))
    for resource, items in by_resource.items():
        if sum(service for _n, _b, service in items) > ii + tolerance:
            overlaps.append({"kind": "resource_capacity", "resource": resource,
                             "demand_ns": sum(s for _n, _b, s in items) * 1e9, "ii_ns": ii * 1e9})
        for a in range(len(items)):
            for b in range(a + 1, len(items)):
                (name_a, begin_a, len_a), (name_b, begin_b, len_b) = items[a], items[b]
                amount = 0.0
                for shift in (-ii, 0.0, ii):
                    amount = max(amount, min(begin_a + len_a, begin_b + shift + len_b) - max(begin_a, begin_b + shift))
                if amount > 1e-12:
                    overlaps.append({"kind": "resource_overlap", "resource": resource, "ops": [name_a, name_b], "overlap_ns": amount * 1e9})
    violations.sort(key=lambda item: -item["violated_by_ns"])
    overlaps.sort(key=lambda item: -(item.get("overlap_ns") or item.get("demand_ns") or 0.0))
    return {"ii_ns": ii * 1e9, "feasible": not violations and not overlaps and not missing, "ops_without_start": missing,
            "violated_constraints": violations, "resource_conflicts": overlaps}



def witness_report(dag: Any, envelope: Any, which: str = "best") -> Dict[str, Any]:
    """Facts about one endpoint of the envelope: ``which`` = ``"best"`` or ``"worst"``."""

    if which not in ("best", "worst"):
        raise ValueError("which must be 'best' or 'worst'")
    bounds = envelope.lower_bounds
    chosen = envelope.best if which == "best" else envelope.worst
    return {
        "endpoint": which,
        "bounds_s": {"resource": bounds.resource_ii, "recurrence": bounds.recurrence_ii, "credit": bounds.credit_ii,
                     "overall": bounds.overall},
        "ii_s": chosen.ii, "best_ii_s": envelope.best.ii, "worst_ii_s": envelope.worst.ii,
        "search_complete": envelope.search_complete,
        "orderings_explored": envelope.orderings_explored, "infeasible_orderings": envelope.infeasible_orderings,
        "unsupported_orderings": envelope.unsupported_orderings,
        "resource_demand_s": resource_demand(dag),
        "phase_starts_s": dict(sorted(chosen.phase_starts.items(), key=lambda item: item[1])),
        "phase_latency_s": {phase.name: phase.latency for phase in dag.phases},
        "resource_orders": {resource: list(order) for resource, order in chosen.resource_orders.items()},
        "fixed_resource_orders": {order.resource: list(order.phases) for order in dag.fixed_resource_orders},
        "critical_cycle": critical_cycle(dag, chosen.resource_orders, chosen.ii),
    }


def explain_loop(program: Any, arch: Any, *, options: Any = None, context: Any = None,
                 loop: Optional[str] = None, group: Optional[str] = None) -> Dict[str, Any]:
    """Explain the schedule that ``analyze`` selected for one (loop, work group).

    The group result records the identity of the PeriodicDAG it was solved with
    (``RegionGroupResult.periodic_dag_digest``) and the II mode that governed the
    loop (``ii.mode``: a loop-local ``ii_mode`` overrides ``Options.ii_mode``); the
    report is built from exactly that DAG and that endpoint (``worst`` under
    ``periodic_worst``), never by matching start times.  Default: the (loop, group)
    with the largest selected II; ``loop`` / ``group`` pick another.
    """

    from .analysis import analyze
    from .options import Options

    options = options or Options(cache="fast", ii_mode="periodic_best")
    with capture_periodic_dags() as captured:
        result = analyze(program, arch, options=options, context=context)
    by_digest = {native_executor.periodic_dag_digest(dag): (dag, envelope) for dag, envelope in captured}
    candidates = []
    for path, region in result.regions.items():
        if region.kind != "loop":          # the launch body only summarises its loops
            continue
        for item in region.groups:
            ii = item.ii
            if ii is None or ii.witness is None or ii.selected_ii_s is None:
                continue
            if (loop is not None and path != loop) or (group is not None and item.work_group_id != group):
                continue
            candidates.append((ii.selected_ii_s, path, item))
    if not candidates:
        return {"result": result, "witness": None,
                "note": "no constructive periodic witness for loop=%r group=%r (ii_mode=%s)" % (loop, group, options.ii_mode)}
    _selected, path, item = max(candidates, key=lambda entry: entry[0])
    found = by_digest.get(item.periodic_dag_digest or "")
    if found is None:
        return {"result": result, "witness": None,
                "note": "the PeriodicDAG solved for %s / %s (digest %s) was not captured" % (path, item.work_group_id, item.periodic_dag_digest)}
    dag, envelope = found
    which = "worst" if item.ii.mode == "periodic_worst" else "best"
    report = witness_report(dag, envelope, which)
    report.update(loop=path, work_group=item.work_group_id, ii_mode=item.ii.mode, selected_ii_s=item.ii.selected_ii_s,
                  periodic_dag_digest=item.periodic_dag_digest)
    cycle = report["critical_cycle"].get("edges", [])
    report["declared_constraints_on_critical_cycle"] = [
        edge["label"] for edge in cycle
        if edge["label"].startswith(("hint:", "actor:", "ring:", "alias:")) or edge["kind"] == "buffer_credit"]
    return {"result": result, "witness": report}


__all__ = ["capture_periodic_dags", "check_witness", "critical_cycle", "explain_loop", "resource_demand", "witness_report"]
