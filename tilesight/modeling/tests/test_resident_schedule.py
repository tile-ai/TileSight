"""Tests for the opt-in constructive resident-CTA scheduler."""

from dataclasses import replace

import pytest

from tilesight.modeling._pipeline.periodic_schedule import (
    Dependency,
    PeriodicDAG,
    Phase,
    ResourceUse,
    SearchConfig,
    TokenBuffer,
    schedule_periodic_dag,
)
from tilesight.modeling.resident_schedule import (
    ResidentResourceRef,
    ResidentScheduleConfig,
    ResidentScheduleError,
    ResourceScope,
    schedule_resident_ctas,
    verify_resident_witness,
)


def _single_cta_envelope_dag():
    return PeriodicDAG(
        phases=(
            Phase(
                "a",
                latency=3.0,
                resources=(ResourceUse("r0", 1.0), ResourceUse("r1", 1.0)),
            ),
            Phase("b", latency=3.0, resources=(ResourceUse("r1", 1.0),)),
            Phase("c", latency=3.0, resources=(ResourceUse("r0", 1.0),)),
            Phase("d", latency=1.0, resources=(ResourceUse("r1", 2.0),)),
        ),
        dependencies=(Dependency("c", "d", iteration_distance=1),),
    )


def test_one_resident_cta_preserves_existing_periodic_envelope_exactly():
    dag = _single_cta_envelope_dag()
    search = SearchConfig(strategy="exact", fixed_order_solver="difference")
    expected = schedule_periodic_dag(dag, search)
    result = schedule_resident_ctas(
        dag, ResidentScheduleConfig(1, search_config=search)
    )

    assert result.best.group_period == pytest.approx(expected.best.ii)
    assert result.worst.group_period == pytest.approx(expected.worst.ii)
    assert result.best.effective_work_interval == pytest.approx(expected.best.ii)
    assert result.worst.effective_work_interval == pytest.approx(expected.worst.ii)
    assert result.lower_bounds == expected.lower_bounds
    assert result.search_complete == expected.search_complete
    assert result.candidates_explored == expected.orderings_explored
    assert result.best.phase_starts == {
        (0, phase): start for phase, start in expected.best.phase_starts.items()
    }
    verify_resident_witness(
        dag, ResidentScheduleConfig(1, search_config=search), result.best
    )


def test_dependencies_carries_and_token_pools_are_independent_per_cta():
    dag = PeriodicDAG(
        phases=(Phase("acquire", 2.0), Phase("release", 4.0)),
        dependencies=(
            Dependency("acquire", "release"),
            Dependency("release", "acquire", iteration_distance=1),
        ),
        token_buffers=(
            TokenBuffer("ring", "acquire", "release", capacity=1),
        ),
    )
    config = ResidentScheduleConfig(
        2,
        order_policies=("round_robin",),
        search_config=SearchConfig(fixed_order_solver="difference"),
    )
    result = schedule_resident_ctas(dag, config)

    # Each CTA has a six-unit recurrence/token lifetime.  Independent pools
    # allow both streams to advance in one six-unit group period.
    assert result.best.group_period == pytest.approx(6.0)
    assert result.best.effective_work_interval == pytest.approx(3.0)
    assert set(result.best.phase_starts) == {
        (0, "acquire"),
        (0, "release"),
        (1, "acquire"),
        (1, "release"),
    }
    verify_resident_witness(dag, config, result.best)


def test_sm_shared_is_capacity_one_while_local_and_external_are_not_cross_cta():
    dag = PeriodicDAG(
        phases=(Phase("issue", 0.0, (ResourceUse("engine", 4.0),)),)
    )
    shared = schedule_resident_ctas(
        dag,
        ResidentScheduleConfig(
            2,
            resource_scopes={"engine": "sm_shared"},
            order_policies=("round_robin",),
        ),
    )
    local = schedule_resident_ctas(
        dag,
        ResidentScheduleConfig(
            2,
            resource_scopes={"engine": "cta_local"},
            order_policies=("round_robin",),
        ),
    )
    external = schedule_resident_ctas(
        dag,
        ResidentScheduleConfig(
            2,
            resource_scopes={"engine": "external_priced"},
            order_policies=("round_robin",),
        ),
    )

    assert shared.shared_resource_capacity == 1
    assert shared.best.group_period == pytest.approx(8.0)
    assert shared.best.effective_work_interval == pytest.approx(4.0)
    assert local.best.group_period == pytest.approx(4.0)
    assert local.best.effective_work_interval == pytest.approx(2.0)
    assert external.best.group_period == pytest.approx(4.0)
    assert external.external_priced_resources == ("engine",)
    assert len(local.best.resource_orders) == 2
    assert len(external.best.resource_orders) == 2


def test_round_robin_and_caller_fixed_orders_are_concrete_resource_witnesses():
    dag = PeriodicDAG(
        phases=(
            Phase("load", 1.0, (ResourceUse("pipe", 1.0),)),
            Phase("use", 1.0, (ResourceUse("pipe", 1.0),)),
        )
    )
    round_robin = schedule_resident_ctas(
        dag,
        ResidentScheduleConfig(2, order_policies=("round_robin",)),
    )
    shared_resource = ResidentResourceRef("pipe", ResourceScope.SM_SHARED)
    assert round_robin.best.resource_orders[shared_resource] == (
        (0, "load"),
        (1, "load"),
        (0, "use"),
        (1, "use"),
    )

    fixed_order = (
        (1, "use"),
        (0, "load"),
        (0, "use"),
        (1, "load"),
    )
    fixed = schedule_resident_ctas(
        dag,
        ResidentScheduleConfig(
            2,
            order_policies=("fixed",),
            fixed_orders={"pipe": fixed_order},
        ),
    )
    assert fixed.best.candidate == "fixed"
    assert fixed.best.resource_orders[shared_resource] == fixed_order
    verify_resident_witness(
        dag,
        ResidentScheduleConfig(
            2,
            order_policies=("fixed",),
            fixed_orders={"pipe": fixed_order},
        ),
        fixed.best,
    )


def test_published_witness_verifier_rejects_dependency_token_and_resource_overlap():
    dependency_dag = PeriodicDAG(
        phases=(Phase("a", 3.0), Phase("b", 1.0)),
        dependencies=(Dependency("a", "b"),),
    )
    dependency_config = ResidentScheduleConfig(
        2, order_policies=("round_robin",)
    )
    dependency_result = schedule_resident_ctas(dependency_dag, dependency_config)
    starts = dict(dependency_result.best.phase_starts)
    starts[(0, "b")] = starts[(0, "a")]
    with pytest.raises(ResidentScheduleError, match="dependency"):
        verify_resident_witness(
            dependency_dag,
            dependency_config,
            replace(dependency_result.best, phase_starts=starts),
        )

    token_dag = PeriodicDAG(
        phases=(Phase("get", 0.0), Phase("put", 0.0)),
        token_buffers=(
            TokenBuffer(
                "slot",
                "get",
                "put",
                capacity=1,
                release_offset=0.0,
                minimum_residence=2.0,
            ),
        ),
    )
    token_config = ResidentScheduleConfig(2, order_policies=("round_robin",))
    token_result = schedule_resident_ctas(token_dag, token_config)
    starts = dict(token_result.best.phase_starts)
    starts[(0, "put")] = starts[(0, "get")]
    with pytest.raises(ResidentScheduleError, match="token"):
        verify_resident_witness(
            token_dag,
            token_config,
            replace(token_result.best, phase_starts=starts),
        )

    resource_dag = PeriodicDAG(
        phases=(Phase("issue", 0.0, (ResourceUse("r", 2.0),)),)
    )
    resource_config = ResidentScheduleConfig(2, order_policies=("round_robin",))
    resource_result = schedule_resident_ctas(resource_dag, resource_config)
    starts = dict(resource_result.best.phase_starts)
    starts[(1, "issue")] = starts[(0, "issue")]
    with pytest.raises(ResidentScheduleError, match="resource"):
        verify_resident_witness(
            resource_dag,
            resource_config,
            replace(resource_result.best, phase_starts=starts),
        )


def test_cluster_ctas_and_unbounded_resident_groups_are_rejected():
    with pytest.raises(ResidentScheduleError, match="cluster/cooperative"):
        ResidentScheduleConfig(2, cta_relationship="cluster")
    with pytest.raises(ResidentScheduleError, match=r"\[1, 4\]"):
        ResidentScheduleConfig(5)

