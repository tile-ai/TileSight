"""Generic resource-order proposals for the periodic scheduler.

The scheduler's own beam explores cyclic resource orders greedily.  A global
zero-distance topological order of the PeriodicDAG, projected onto every
resource without a fixed order, gives a family of *mutually consistent* orders
that the beam can miss.  Proposals never decide feasibility: the periodic
engine checks each one against the full dependency, token/credit and resource
constraint system.  The search stays bounded, so ``search_complete`` is false.
"""

from __future__ import annotations

import random
from typing import Dict, Iterator, List, Tuple

from tilesight.modeling._pipeline.periodic_schedule import PeriodicDAG, ResourceOrderCandidate


def topology_order_proposals(dag: PeriodicDAG, seed: int, trials: int) -> Iterator[ResourceOrderCandidate]:
    """Source order plus ``trials`` seeded random zero-distance topological orders, deduplicated."""

    phase_index = {phase.name: index for index, phase in enumerate(dag.phases)}
    users = {}  # type: Dict[str, List[str]]
    for phase in dag.phases:
        for use in phase.resources:
            users.setdefault(use.resource, []).append(phase.name)
    fixed = {order.resource for order in dag.fixed_resource_orders}
    dynamic = tuple(resource for resource in sorted(users) if resource not in fixed)
    if not any(len(users[resource]) > 1 for resource in dynamic):
        return  # nothing to order: the scheduler's own search is already complete
    successors = {phase.name: [] for phase in dag.phases}  # type: Dict[str, List[str]]
    indegree = {phase.name: 0 for phase in dag.phases}
    for dependency in dag.dependencies:
        if dependency.iteration_distance == 0:
            successors[dependency.source].append(dependency.target)
            indegree[dependency.target] += 1
    generator = random.Random(seed)

    def order(trial: int) -> Tuple[str, ...]:
        degree = dict(indegree)
        ready = [name for name, value in degree.items() if value == 0]
        result = []
        while ready:
            if trial == 0:
                ready.sort(key=phase_index.__getitem__)
                position = 0
            else:
                position = generator.randrange(len(ready))
            name = ready.pop(position)
            result.append(name)
            for successor in successors[name]:
                degree[successor] -= 1
                if degree[successor] == 0:
                    ready.append(successor)
        if len(result) != len(dag.phases):
            raise ValueError("zero-distance dependency graph contains a cycle")
        return tuple(result)

    seen = set()
    for trial in range(trials + 1):
        rank = {name: index for index, name in enumerate(order(trial))}
        orders = {resource: tuple(sorted(users[resource], key=rank.__getitem__)) for resource in dynamic}
        key = tuple(sorted(orders.items()))
        if key in seen:
            continue
        seen.add(key)
        yield ResourceOrderCandidate(
            resource_orders=orders,
            source="zero_distance_topology" if trial == 0 else "sampled_zero_distance_topology",
            label="source_order" if trial == 0 else "trial_%d" % trial,
        )


__all__ = ["topology_order_proposals"]
