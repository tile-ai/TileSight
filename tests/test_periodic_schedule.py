from dataclasses import replace

import pytest

from tilesight.modeling._pipeline import periodic_schedule

from tilesight.modeling._pipeline.periodic_schedule import (
    Dependency,
    FixedResourceOrder,
    IIResultScope,
    IIScheduleMode,
    NoFeasibleScheduleError,
    PeriodicDAG,
    PeriodicScheduleError,
    Phase,
    ResourceOrderCandidate,
    ResourceUse,
    SearchConfig,
    TokenBuffer,
    UnsupportedConstraintDomainError,
    compute_lower_bounds,
    evaluate_resource_ordering,
    schedule_periodic_dag,
    select_steady_ii,
)


def _two_stage_pipeline(load_latency):
    return PeriodicDAG(
        phases=(
            Phase("load", latency=load_latency, resources=(ResourceUse("ddr", 10.0),)),
            Phase("mma", latency=10.0, resources=(ResourceUse("tensor", 10.0),)),
        ),
        dependencies=(Dependency("load", "mma"),),
        token_buffers=(TokenBuffer("smem_ring", "load", "mma", capacity=2),),
    )


def _coupled_resource_dag(scale=1.0, extra_recurrences=0):
    return PeriodicDAG(
        phases=(
            Phase(
                "ab",
                latency=1.0 * scale,
                resources=(
                    ResourceUse("a", 1.0 * scale),
                    ResourceUse("b", 1.0 * scale),
                ),
            ),
            Phase(
                "bc",
                latency=1.0 * scale,
                resources=(
                    ResourceUse("b", 1.0 * scale),
                    ResourceUse("c", 1.0 * scale),
                ),
            ),
            Phase(
                "ca",
                latency=1.0 * scale,
                resources=(
                    ResourceUse("c", 1.0 * scale),
                    ResourceUse("a", 1.0 * scale),
                ),
            ),
        ),
        dependencies=tuple(
            Dependency(
                "ab",
                "ab",
                iteration_distance=1,
                min_delay=0.0,
                name="padding_%d" % index,
            )
            for index in range(extra_recurrences)
        ),
    )


_COUPLED_ORDERS = {
    "a": ("ab", "ca"),
    "b": ("ab", "bc"),
    "c": ("bc", "ca"),
}


def test_two_stage_pipeline_hides_twenty_unit_end_to_end_latency():
    result = schedule_periodic_dag(
        _two_stage_pipeline(load_latency=10.0), SearchConfig(strategy="exact")
    )

    assert result.lower_bounds.resource_ii == pytest.approx(10.0)
    assert result.lower_bounds.recurrence_ii == pytest.approx(0.0)
    assert result.lower_bounds.credit_ii == pytest.approx(10.0)
    assert result.best.ii == pytest.approx(10.0)
    assert result.worst.ii == pytest.approx(10.0)
    assert (
        result.best.phase_starts["mma"] - result.best.phase_starts["load"]
        == pytest.approx(10.0)
    )


def test_double_buffer_cannot_hide_thirty_unit_residence_time():
    # The DDR issue engine is occupied for only 10, but its asynchronous result
    # becomes ready after 20.  MMA then holds the slot for another 10.
    result = schedule_periodic_dag(
        _two_stage_pipeline(load_latency=20.0), SearchConfig(strategy="exact")
    )

    assert result.lower_bounds.resource_ii == pytest.approx(10.0)
    assert result.lower_bounds.credit_ii == pytest.approx(15.0)
    assert result.best.ii == pytest.approx(15.0)
    assert result.worst.ii == pytest.approx(15.0)


def test_credit_bound_follows_signed_path_with_zero_net_distance():
    dag = PeriodicDAG(
        phases=(
            Phase("acquire", latency=0.0),
            Phase("middle", latency=0.0),
            Phase("release", latency=0.0),
        ),
        dependencies=(
            Dependency(
                "acquire", "middle", iteration_distance=1, min_delay=10.0
            ),
            Dependency(
                "middle", "release", iteration_distance=-1, min_delay=10.0
            ),
        ),
        token_buffers=(
            TokenBuffer("ring", "acquire", "release", capacity=2),
        ),
    )

    result = schedule_periodic_dag(dag, SearchConfig(strategy="exact"))

    assert result.lower_bounds.recurrence_ii == pytest.approx(0.0)
    assert result.lower_bounds.credit_ii == pytest.approx(10.0)
    assert result.best.ii == pytest.approx(10.0)


def test_token_capacity_is_a_positive_integer_iteration_count():
    with pytest.raises(PeriodicScheduleError, match="must be an integer"):
        TokenBuffer("ring", "a", "b", capacity=2.5)
    with pytest.raises(PeriodicScheduleError, match="must be positive"):
        TokenBuffer("ring", "a", "b", capacity=0)


def test_coupled_resources_need_constructive_schedule_not_just_resource_lb():
    # Each resource has demand 2, so a fluid utilization bound says II >= 2.
    # But every pair of phases conflicts on one resource; the three unit phases
    # therefore require a length-3 cyclic packing.
    dag = PeriodicDAG(
        phases=(
            Phase(
                "ab",
                latency=1.0,
                resources=(ResourceUse("a", 1.0), ResourceUse("b", 1.0)),
            ),
            Phase(
                "bc",
                latency=1.0,
                resources=(ResourceUse("b", 1.0), ResourceUse("c", 1.0)),
            ),
            Phase(
                "ca",
                latency=1.0,
                resources=(ResourceUse("c", 1.0), ResourceUse("a", 1.0)),
            ),
        )
    )

    result = schedule_periodic_dag(dag, SearchConfig(strategy="exact"))

    assert result.search_complete
    assert result.lower_bounds.resource_ii == pytest.approx(2.0)
    assert result.best.ii == pytest.approx(3.0)
    assert result.worst.ii == pytest.approx(3.0)
    assert result.best.ii > result.lower_bounds.overall


def test_iteration_distance_dependency_contributes_recurrence_bound():
    dag = PeriodicDAG(
        phases=(Phase("state", latency=12.0),),
        dependencies=(Dependency("state", "state", iteration_distance=2),),
    )

    result = schedule_periodic_dag(dag, SearchConfig(strategy="exact"))

    assert result.lower_bounds.recurrence_ii == pytest.approx(6.0)
    assert result.best.ii == pytest.approx(6.0)


def test_signed_distances_represent_source_faithful_interleaving():
    # Source order around a steady-state boundary can be
    # PV0_i -> QK0_i+1 -> PV1_i -> QK1_i+1 -> PV0_i+1.  The -1 edge is valid:
    # the complete recurrence advances by one iteration and remains causal.
    dag = PeriodicDAG(
        phases=tuple(
            Phase(
                name,
                latency=1.0,
                resources=(ResourceUse("tensor", 1.0),),
                iteration_offset=iteration_offset,
            )
            for name, iteration_offset in (
                ("pv0", 0),
                ("qk0", 1),
                ("pv1", 0),
                ("qk1", 1),
            )
        ),
        dependencies=(
            Dependency("pv0", "qk0", iteration_distance=1),
            Dependency("qk0", "pv1", iteration_distance=-1),
            Dependency("pv1", "qk1", iteration_distance=1),
            Dependency("qk1", "pv0", iteration_distance=0),
        ),
        fixed_resource_orders=(
            FixedResourceOrder("tensor", ("pv0", "qk0", "pv1", "qk1")),
        ),
    )

    result = schedule_periodic_dag(dag, SearchConfig(strategy="exact"))

    assert result.lower_bounds.resource_ii == pytest.approx(4.0)
    assert result.lower_bounds.recurrence_ii == pytest.approx(4.0)
    assert result.best.ii == pytest.approx(4.0)
    assert result.orderings_explored == 1
    assert result.best.resource_orders["tensor"] == ("pv0", "qk0", "pv1", "qk1")


def test_best_and_worst_are_finite_work_conserving_order_envelope():
    dag = PeriodicDAG(
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

    result = schedule_periodic_dag(dag, SearchConfig(strategy="exact"))

    assert result.lower_bounds.resource_ii == pytest.approx(4.0)
    assert result.best.ii == pytest.approx(4.0)
    assert result.worst.ii == pytest.approx(7.0)
    assert result.overlap_sensitivity == pytest.approx(3.0)


def test_common_selector_keeps_resource_lower_bound_lazy_and_nonconstructive():
    called = False

    def must_not_run():
        nonlocal called
        called = True
        raise AssertionError("resource-II mode must remain lazy")

    result = select_steady_ii(
        10.0, mode="resource_ii", solve_periodic=must_not_run
    )

    assert not called
    assert result.mode == IIScheduleMode.RESOURCE_II
    assert result.ii == pytest.approx(10.0)
    assert not result.is_constructive
    assert result.witness is None
    assert result.envelope is None
    assert result.search_complete is None
    assert result.scope == IIResultScope.NECESSARY_BOUND
    assert result.endpoint_scope == "lower_bound"


def test_common_selector_labels_complete_periodic_endpoints():
    envelope = schedule_periodic_dag(
        _two_stage_pipeline(load_latency=20.0),
        SearchConfig(strategy="exact"),
    )
    best = select_steady_ii(
        10.0, "periodic_best", solve_periodic=lambda: envelope
    )
    worst = select_steady_ii(
        10.0, "periodic_worst", solve_periodic=lambda: envelope
    )

    assert best.is_constructive and worst.is_constructive
    assert best.ii == pytest.approx(15.0)
    assert worst.ii == pytest.approx(15.0)
    assert best.scope == IIResultScope.MODEL_EXHAUSTIVE
    assert best.endpoint_scope == "model_exhaustive"
    assert best.best_ii == pytest.approx(15.0)
    assert best.worst_ii == pytest.approx(15.0)
    assert best.periodic_lower_bounds is envelope.lower_bounds
    assert best.witness.to_dict()["phase_starts"] == dict(
        best.witness.phase_starts
    )
    with pytest.raises(TypeError):
        best.witness.phase_starts["load"] = 0.0


def test_periodic_witness_is_canonicalized_to_microsecond_resource_bound():
    # A former unit-insensitive 1 ps feasibility tolerance could report a
    # periodic witness infinitesimally below this independently summed bound.
    service_times = (1.0167818916575659e-6, 0.9139179832712471e-6)
    dag = PeriodicDAG(
        phases=tuple(
            Phase(
                "p%d" % index,
                latency=service_time,
                resources=(ResourceUse("tensor", service_time),),
            )
            for index, service_time in enumerate(service_times)
        )
    )
    resource_ii = sum(service_times)
    envelope = schedule_periodic_dag(dag, SearchConfig(strategy="exact"))
    selection = select_steady_ii(
        resource_ii,
        "periodic_best",
        solve_periodic=lambda: envelope,
    )

    assert envelope.best.ii >= resource_ii
    assert selection.ii >= resource_ii
    assert selection.ii <= envelope.worst.ii


def test_common_selector_rejects_invalid_modes_and_missing_solver():
    with pytest.raises(PeriodicScheduleError, match="mode must be one of"):
        select_steady_ii(1.0, "fastest")
    with pytest.raises(PeriodicScheduleError, match="require a solve_periodic"):
        select_steady_ii(1.0, "periodic_best")


def test_common_selector_never_returns_periodic_ii_below_resource_bound():
    envelope = schedule_periodic_dag(
        _two_stage_pipeline(load_latency=10.0),
        SearchConfig(strategy="exact"),
    )
    invalid = replace(
        envelope,
        best=replace(envelope.best, ii=envelope.lower_bounds.resource_ii - 1e-12),
    )

    with pytest.raises(PeriodicScheduleError, match="below necessary bound"):
        select_steady_ii(
            envelope.lower_bounds.resource_ii,
            "periodic_best",
            solve_periodic=lambda: invalid,
        )


def test_common_selector_rejects_nonfinite_or_reversed_envelope():
    envelope = schedule_periodic_dag(
        _two_stage_pipeline(load_latency=10.0),
        SearchConfig(strategy="exact"),
    )
    reversed_envelope = replace(
        envelope,
        best=replace(envelope.best, ii=11.0),
        worst=replace(envelope.worst, ii=10.0),
    )
    with pytest.raises(PeriodicScheduleError, match="envelope is reversed"):
        select_steady_ii(
            10.0,
            "periodic_best",
            solve_periodic=lambda: reversed_envelope,
        )

    nan_envelope = replace(
        envelope,
        best=replace(envelope.best, ii=float("nan")),
    )
    with pytest.raises(PeriodicScheduleError, match="must be finite"):
        select_steady_ii(
            10.0,
            "periodic_best",
            solve_periodic=lambda: nan_envelope,
        )


def test_unsupported_bounded_ii_orderings_force_incomplete_envelope():
    dag = PeriodicDAG(
        phases=(
            Phase(
                "a",
                1.0,
                (ResourceUse("r", 1.0, offset=2.0),),
                iteration_offset=1,
            ),
            Phase(
                "b",
                1.0,
                (ResourceUse("r", 1.0),),
                iteration_offset=1,
            ),
            Phase("c", 1.0, (ResourceUse("r", 1.0),)),
        ),
        dependencies=(
            Dependency("a", "c", iteration_distance=-2),
        ),
    )

    envelope = schedule_periodic_dag(
        dag, SearchConfig(strategy="exact", max_exact_orderings=10)
    )

    assert envelope.unsupported_orderings > 0
    assert not envelope.search_complete
    assert envelope.best.ii >= envelope.lower_bounds.resource_ii
    with pytest.raises(UnsupportedConstraintDomainError):
        evaluate_resource_ordering(dag, {"r": ("b", "c", "a")})


def test_public_named_resource_order_candidate_is_fully_validated():
    dag = PeriodicDAG(
        phases=(
            Phase("a", 1.0, (ResourceUse("r", 1.0),)),
            Phase("b", 1.0, (ResourceUse("r", 1.0),)),
        )
    )
    witness = evaluate_resource_ordering(dag, {"r": ("b", "a")})
    assert witness.ii == pytest.approx(2.0)
    assert witness.resource_orders["r"] == ("b", "a")

    result = schedule_periodic_dag(
        dag,
        SearchConfig(strategy="beam", beam_width=1),
        extra_orderings=(
            ResourceOrderCandidate({"r": ("b", "a")}, source="test"),
        ),
    )
    assert result.best.ii == pytest.approx(2.0)
    with pytest.raises(PeriodicScheduleError, match="omits non-fixed"):
        evaluate_resource_ordering(dag, {})


@pytest.mark.parametrize("scale", (1.0, 1.0e-9))
def test_fixed_order_difference_and_highs_backends_agree_across_scales(
    scale, monkeypatch
):
    pytest.importorskip("scipy.optimize")
    dag = _coupled_resource_dag(scale)
    original = periodic_schedule._minimum_ii_highs
    highs_results = []

    def counted_highs(*args, **kwargs):
        result = original(*args, **kwargs)
        highs_results.append(result)
        return result

    monkeypatch.setattr(
        periodic_schedule, "_minimum_ii_highs", counted_highs
    )

    difference = evaluate_resource_ordering(
        dag, _COUPLED_ORDERS, fixed_order_solver="difference"
    )
    highs = evaluate_resource_ordering(
        dag, _COUPLED_ORDERS, fixed_order_solver="highs"
    )

    assert len(highs_results) == 1
    assert highs_results[0] is not None
    assert difference.ii == pytest.approx(3.0 * scale)
    assert highs.ii == pytest.approx(3.0 * scale)
    assert highs.ii == pytest.approx(
        difference.ii,
        rel=0.0,
        abs=max(2.0e-15, 2.0e-11 * scale),
    )
    assert highs.resource_orders == difference.resource_orders


def test_fixed_order_solver_is_validated_at_public_interfaces():
    with pytest.raises(PeriodicScheduleError, match="fixed_order_solver"):
        SearchConfig(fixed_order_solver="fastest")

    dag = PeriodicDAG(
        phases=(Phase("only", 1.0, (ResourceUse("r", 1.0),)),)
    )
    with pytest.raises(PeriodicScheduleError, match="fixed_order_solver"):
        evaluate_resource_ordering(
            dag, {"r": ("only",)}, fixed_order_solver="fastest"
        )


def test_difference_and_lower_bound_paths_do_not_call_highs(monkeypatch):
    def must_not_run(*args, **kwargs):
        raise AssertionError("difference/lower-bound path must remain lazy")

    monkeypatch.setattr(periodic_schedule, "_minimum_ii_highs", must_not_run)
    dag = _two_stage_pipeline(load_latency=20.0)

    bounds = compute_lower_bounds(dag)
    result = schedule_periodic_dag(
        dag,
        SearchConfig(strategy="exact", fixed_order_solver="difference"),
    )

    assert bounds.overall == pytest.approx(15.0)
    assert result.best.ii == pytest.approx(15.0)


def test_highs_preserves_unsupported_and_infeasible_classification():
    unsupported = PeriodicDAG(
        phases=(
            Phase(
                "a",
                1.0,
                (ResourceUse("r", 1.0, offset=2.0),),
                iteration_offset=1,
            ),
            Phase(
                "b",
                1.0,
                (ResourceUse("r", 1.0),),
                iteration_offset=1,
            ),
            Phase("c", 1.0, (ResourceUse("r", 1.0),)),
        ),
        dependencies=(Dependency("a", "c", iteration_distance=-2),),
    )
    with pytest.raises(UnsupportedConstraintDomainError):
        evaluate_resource_ordering(
            unsupported,
            {"r": ("b", "c", "a")},
            fixed_order_solver="highs",
        )

    infeasible = PeriodicDAG(
        phases=(
            Phase("a", 1.0, (ResourceUse("r", 1.0),)),
            Phase("b", 1.0, (ResourceUse("r", 1.0),)),
        ),
        dependencies=(Dependency("a", "b"),),
    )
    with pytest.raises(NoFeasibleScheduleError):
        evaluate_resource_ordering(
            infeasible,
            {"r": ("b", "a")},
            fixed_order_solver="highs",
        )


def test_highs_failure_falls_back_to_difference_solver(monkeypatch):
    calls = 0

    def fail_highs(*args, **kwargs):
        nonlocal calls
        calls += 1
        return None

    monkeypatch.setattr(
        periodic_schedule,
        "_minimum_ii_highs",
        fail_highs,
    )
    dag = _coupled_resource_dag()

    fallback = evaluate_resource_ordering(
        dag, _COUPLED_ORDERS, fixed_order_solver="highs"
    )
    expected = evaluate_resource_ordering(
        dag, _COUPLED_ORDERS, fixed_order_solver="difference"
    )

    assert calls == 1
    assert fallback.ii == expected.ii
    assert fallback.phase_starts == expected.phase_starts


def test_auto_solver_routes_low_and_high_workloads(monkeypatch):
    pytest.importorskip("scipy.optimize")
    original = periodic_schedule._minimum_ii_highs
    calls = 0

    def counted_highs(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        periodic_schedule, "_minimum_ii_highs", counted_highs
    )

    low = evaluate_resource_ordering(
        _coupled_resource_dag(),
        _COUPLED_ORDERS,
        fixed_order_solver="auto",
    )
    assert low.ii == pytest.approx(3.0)
    assert calls == 0

    high = evaluate_resource_ordering(
        _coupled_resource_dag(extra_recurrences=100),
        _COUPLED_ORDERS,
        fixed_order_solver="auto",
    )
    assert high.ii == pytest.approx(3.0)
    assert calls == 1


def test_highs_dense_and_sparse_matrices_handle_self_parallel_signed_edges(
    monkeypatch,
):
    np = pytest.importorskip("numpy")
    sparse = pytest.importorskip("scipy.sparse")
    constraints = (
        periodic_schedule._Constraint(0, 0, 3.0, 1, "self"),
        periodic_schedule._Constraint(0, 1, 4.0, -1, "signed"),
        periodic_schedule._Constraint(0, 1, 3.0, -1, "parallel"),
        periodic_schedule._Constraint(1, 0, 2.0, 2, "return"),
    )

    monkeypatch.setattr(
        periodic_schedule,
        "_HIGHS_DENSE_MATRIX_ELEMENT_LIMIT",
        10_000,
    )
    dense_matrix, dense_bounds = periodic_schedule._build_highs_constraint_matrix(
        constraints, 2, 4.0, np
    )
    assert isinstance(dense_matrix, np.ndarray)

    monkeypatch.setattr(
        periodic_schedule,
        "_HIGHS_DENSE_MATRIX_ELEMENT_LIMIT",
        0,
    )
    sparse_matrix, sparse_bounds = periodic_schedule._build_highs_constraint_matrix(
        constraints, 2, 4.0, np
    )
    assert sparse.issparse(sparse_matrix)
    np.testing.assert_array_equal(sparse_matrix.toarray(), dense_matrix)
    np.testing.assert_array_equal(sparse_bounds, dense_bounds)
    # The first row is a self edge: start coefficients cancel exactly.
    np.testing.assert_array_equal(dense_matrix[0], (0.0, 0.0, -1.0))

    result = periodic_schedule._minimum_ii_highs(constraints, 2, 0.0)
    assert result is not None
    ii, starts = result
    assert ii == pytest.approx(6.0)
    assert periodic_schedule._constraints_hold(constraints, ii, starts)


def test_highs_matrix_memory_error_falls_back_to_difference(monkeypatch):
    calls = 0

    def fail_matrix_build(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise MemoryError("synthetic allocation failure")

    monkeypatch.setattr(
        periodic_schedule,
        "_build_highs_constraint_matrix",
        fail_matrix_build,
    )
    fallback = evaluate_resource_ordering(
        _coupled_resource_dag(),
        _COUPLED_ORDERS,
        fixed_order_solver="highs",
    )

    assert calls == 1
    assert fallback.ii == pytest.approx(3.0)
