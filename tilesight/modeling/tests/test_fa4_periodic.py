import pytest

from tilesight.arch.b200 import B200

from tilesight.modeling._attention.fa4_model import _build_body, model_fa4_us
from tilesight.modeling._attention import fa4_periodic
from tilesight.modeling._attention.fa4_periodic import (
    analyze_fa4_periodic,
    build_fa4_periodic_dag,
    sm100_kv_stage,
)


_USAGE_FIELDS = {
    "ddr": "ddr_time",
    "l2": "l2_time",
    "l1_5": "l1_5_time",
    "smem": "smem_time",
    "tmem": "tmem_time",
    "tensor": "tensor_time",
    "cuda": "cuda_time",
    "sfu": "sfu_time",
    "network": "network_time",
}


@pytest.fixture(scope="module")
def arch():
    return B200().set_to_microbench()


def _body(arch, d, *, l2_hit=0.5):
    return _build_body(
        arch,
        128,
        128,
        d,
        d,
        2,
        l2_hit,
        2,
        4,
        0.85,
        1.0,
    )


def _build(arch, d, *, l2_hit=0.5):
    body = _body(arch, d, l2_hit=l2_hit)
    dag, kv_stage = build_fa4_periodic_dag(
        body,
        tile_m=128,
        tile_n=128,
        d=d,
        dv=d,
        q_stage=2,
        dtype_bytes=2,
    )
    return body, dag, kv_stage


def _resource_totals_from_body(body):
    return {
        resource: sum(getattr(group.usage, field) for group in body)
        for resource, field in _USAGE_FIELDS.items()
    }


def _resource_totals_from_dag(dag):
    return {
        resource: sum(
            use.service_time
            for phase in dag.phases
            for use in phase.resources
            if use.resource == resource
        )
        for resource in _USAGE_FIELDS
    }


def _dependency_by_name(dag):
    return {dependency.name: dependency for dependency in dag.dependencies}


def test_kv_stage_matches_padded_source_budget_and_alias_exception():
    assert sm100_kv_stage(128, 128, 64, 64, 2, 2) == 10
    assert sm100_kv_stage(128, 128, 96, 96, 2, 2) == 5
    assert sm100_kv_stage(128, 128, 128, 128, 2, 2) == 3
    # Irregular dimensions are padded to the kernel's 16-element granularity.
    assert sm100_kv_stage(128, 128, 65, 65, 2, 2) == 7
    # D=192 aliases Q/O SMEM; the uneven 192/128 layout promotes 2 -> 3.
    assert sm100_kv_stage(128, 128, 192, 128, 2, 2) == 3
    assert sm100_kv_stage(128, 128, 192, 192, 2, 2) == 2


def test_detailed_split_preserves_every_analytical_resource_total(arch):
    body, dag, kv_stage = _build(arch, 64)

    assert kv_stage == 10
    assert len(dag.phases) == 18
    assert _resource_totals_from_dag(dag) == pytest.approx(
        _resource_totals_from_body(body)
    )

    phase_by_name = {phase.name: phase for phase in dag.phases}
    pv_group = next(group for group in body if group.name == "gemm_PV_s0")
    pv75 = next(
        use.service_time
        for use in phase_by_name["gemm_PV75_s0"].resources
        if use.resource == "tensor"
    )
    pv25 = next(
        use.service_time
        for use in phase_by_name["gemm_PV25_s0"].resources
        if use.resource == "tensor"
    )
    assert pv75 == pytest.approx(pv_group.usage.tensor_time * 0.75)
    assert pv25 == pytest.approx(pv_group.usage.tensor_time * 0.25)


def test_recurrences_split_barriers_and_fixed_tensor_order_are_source_faithful(arch):
    _, dag, _ = _build(arch, 64)
    edges = _dependency_by_name(dag)

    for stage in range(2):
        assert (
            edges[f"softmax_state_s{stage}"].source,
            edges[f"softmax_state_s{stage}"].target,
            edges[f"softmax_state_s{stage}"].iteration_distance,
        ) == (f"softmax_s{stage}_sum", f"softmax_s{stage}_max", 1)
        assert (
            edges[f"O_state_s{stage}"].source,
            edges[f"O_state_s{stage}"].target,
            edges[f"O_state_s{stage}"].iteration_distance,
        ) == (f"gemm_PV25_s{stage}", f"corr_s{stage}", 1)
        assert edges[f"P75_ready_s{stage}"].target == f"gemm_PV75_s{stage}"
        assert edges[f"P25_ready_s{stage}"].target == f"gemm_PV25_s{stage}"

    assert edges["correction_stage_order"].iteration_distance == 0
    assert edges["correction_iteration_order"].iteration_distance == 1
    tensor_order = next(
        order for order in dag.fixed_resource_orders if order.resource == "tensor"
    )
    assert tensor_order.phases == (
        "gemm_PV75_s0",
        "gemm_PV25_s0",
        "gemm_QK_s0",
        "gemm_PV75_s1",
        "gemm_PV25_s1",
        "gemm_QK_s1",
    )


def test_even_and_odd_physical_kv_ring_release_the_last_consumer(arch):
    _, even_dag, even_kv = _build(arch, 64)
    _, odd_dag, odd_kv = _build(arch, 128)
    even = _dependency_by_name(even_dag)
    odd = _dependency_by_name(odd_dag)

    assert even_kv == 10
    assert (
        even["K_ring_reuse"].source,
        even["K_ring_reuse"].target,
        even["K_ring_reuse"].iteration_distance,
    ) == ("gemm_QK_s1", "load_K", 5)
    assert (
        even["V_ring_reuse"].source,
        even["V_ring_reuse"].target,
        even["V_ring_reuse"].iteration_distance,
    ) == ("gemm_PV25_s1", "load_V", 5)

    assert odd_kv == 3
    assert (
        odd["K_to_V_ring_reuse"].source,
        odd["K_to_V_ring_reuse"].target,
        odd["K_to_V_ring_reuse"].iteration_distance,
    ) == ("gemm_QK_s1", "load_V", 1)
    assert (
        odd["V_to_K_ring_reuse"].source,
        odd["V_to_K_ring_reuse"].target,
        odd["V_to_K_ring_reuse"].iteration_distance,
    ) == ("gemm_PV25_s1", "load_K", 2)


def test_zero_distance_topology_is_acyclic_but_not_used_as_an_ii_proof(arch):
    nx = pytest.importorskip("networkx")
    _, dag, _ = _build(arch, 64)
    graph = nx.DiGraph()
    graph.add_nodes_from(phase.name for phase in dag.phases)
    graph.add_edges_from(
        (dependency.source, dependency.target)
        for dependency in dag.dependencies
        if dependency.iteration_distance == 0
    )

    assert nx.is_directed_acyclic_graph(graph)
    # This checks the same single-iteration property as all_topological_sorts;
    # periodic feasibility is tested separately through analyze_fa4_periodic.
    assert next(nx.all_topological_sorts(graph))


def test_default_resource_ii_is_lazy_and_has_common_selection_semantics(arch):
    fa4_periodic._schedule_cached.cache_clear()

    _, breakdown = model_fa4_us(1, 32, 128, 128, arch=arch, causal=True)

    assert fa4_periodic._schedule_cached.cache_info().misses == 0
    assert breakdown["q_stage"] == 1
    assert breakdown["inner_schedule_model"] == "resource_ii"
    assert breakdown["ii_selection_label"] == "resource_ii_lower_bound"
    assert breakdown["ii_selection_scope"] == "lower_bound"
    assert not breakdown["ii_is_constructive"]
    assert breakdown["ii_search_complete"] is None
    assert breakdown["periodic_search_profile"] == "reference"
    assert breakdown["periodic_topology_trials"] is None
    assert breakdown["ii_selected_ns"] == pytest.approx(
        breakdown["resource_ii_lb_ns"]
    )
    with pytest.raises(NotImplementedError, match="q_stage=2"):
        model_fa4_us(
            1,
            32,
            128,
            128,
            arch=arch,
            causal=True,
            inner_schedule_model="periodic_best",
        )

    with pytest.raises(ValueError, match="periodic_search_profile"):
        model_fa4_us(
            1,
            32,
            128,
            128,
            arch=arch,
            periodic_search_profile="turbo",
        )


def test_single_kv_iteration_is_fully_charged_to_tile_overhead(arch):
    total_us, breakdown = model_fa4_us(
        1,
        1,
        128,
        64,
        arch=arch,
        causal=True,
        scheduler_model="aggregate_lb",
    )

    assert breakdown["max_tile_iters"] == 1
    assert breakdown["busy_kv_iters"] == pytest.approx(1.0)
    assert breakdown["tiles_busy"] == pytest.approx(1.0)
    assert breakdown["busy_steady_iters"] == pytest.approx(0.0)
    assert breakdown["t_steady_us"] == pytest.approx(0.0)
    assert breakdown["kernel_body_us"] == pytest.approx(
        breakdown["t_tile_oh_us"]
    )
    assert total_us == pytest.approx(
        breakdown["kernel_body_us"] + breakdown["kernel_launch_us"]
    )
    assert breakdown["kernel_launch_us"] == pytest.approx(2.0)


def test_multi_kv_iteration_charges_ii_only_after_serial_first_iter(arch):
    total_us, breakdown = model_fa4_us(
        1,
        1,
        512,
        128,
        arch=arch,
        causal=True,
        scheduler_model="aggregate_lb",
    )

    # With fewer tiles than SMs, the longest causal tile is the critical tile.
    assert breakdown["max_tile_iters"] == 4
    assert breakdown["busy_kv_iters"] == pytest.approx(4.0)
    assert breakdown["tiles_busy"] == pytest.approx(1.0)
    assert breakdown["busy_steady_iters"] == pytest.approx(3.0)
    assert breakdown["t_steady_us"] == pytest.approx(
        3.0 * breakdown["iter_lat_pipe_ns"] / 1.0e3
    )
    assert breakdown["kernel_body_us"] == pytest.approx(
        breakdown["t_tile_oh_us"] + breakdown["t_steady_us"]
    )
    assert total_us == pytest.approx(
        breakdown["kernel_body_us"] + breakdown["kernel_launch_us"]
    )
    assert breakdown["kernel_launch_us"] == pytest.approx(2.0)


def test_periodic_smoke_is_constructive_bounded_and_cache_keyed_by_dag(arch):
    fa4_periodic._schedule_cached.cache_clear()
    body = _body(arch, 128, l2_hit=0.5)
    kwargs = dict(
        tile_m=128,
        tile_n=128,
        d=128,
        dv=128,
        q_stage=2,
        dtype_bytes=2,
    )
    envelope, kv_stage = analyze_fa4_periodic(
        body, **kwargs, topology_trials=0
    )
    cached_envelope, _ = analyze_fa4_periodic(
        body, **kwargs, topology_trials=0
    )

    assert kv_stage == 3
    assert envelope is cached_envelope
    assert fa4_periodic._schedule_cached.cache_info().hits == 1
    assert envelope.best.ii >= envelope.lower_bounds.overall - 1.0e-12
    assert envelope.worst.ii >= envelope.best.ii
    assert envelope.orderings_explored > 0
    assert envelope.unsupported_orderings >= 0
    # The detailed q_stage=2 resource-order product exceeds the exact cap.
    assert not envelope.search_complete
    # All HardwareUsage and scheduler quantities remain seconds.
    assert 1.0e-9 < envelope.best.ii < 1.0e-3

    different_body = _body(arch, 128, l2_hit=0.75)
    different_envelope, _ = analyze_fa4_periodic(
        different_body, **kwargs, topology_trials=0
    )
    assert different_envelope is not envelope
    assert fa4_periodic._schedule_cached.cache_info().misses == 2

    different_budget, _ = analyze_fa4_periodic(
        body, **kwargs, topology_trials=1
    )
    assert different_budget is not envelope
    assert fa4_periodic._schedule_cached.cache_info().misses == 3

    with pytest.raises(ValueError, match="non-negative integer"):
        analyze_fa4_periodic(body, **kwargs, topology_trials=-1)


def test_model_periodic_modes_share_cached_envelope_and_common_semantics(arch):
    fa4_periodic._schedule_cached.cache_clear()
    kwargs = dict(
        b=1,
        h=32,
        s=512,
        d=128,
        arch=arch,
        causal=True,
    )

    best_us, best = model_fa4_us(
        **kwargs,
        inner_schedule_model="periodic_best",
        periodic_search_profile="fast",
    )
    worst_us, worst = model_fa4_us(
        **kwargs,
        inner_schedule_model="periodic_worst",
        periodic_search_profile="fast",
    )

    cache = fa4_periodic._schedule_cached.cache_info()
    assert cache.misses == 1
    assert cache.hits == 1
    assert best_us <= worst_us
    assert best["ii_is_constructive"] and worst["ii_is_constructive"]
    assert best["ii_selection_scope"] == "searched"
    assert best["ii_search_complete"] is False
    assert best["ii_selected_ns"] == pytest.approx(best["periodic_best_ii_ns"])
    assert worst["ii_selected_ns"] == pytest.approx(worst["periodic_worst_ii_ns"])
    assert best["periodic_unsupported_orderings"] >= 0
    assert best["periodic_search_profile"] == "fast"
    assert worst["periodic_search_profile"] == "fast"
    assert best["periodic_topology_trials"] == 0
    assert worst["periodic_topology_trials"] == 0
    # Existing per-model diagnostics remain available alongside the common fields.
    assert best["periodic_endpoint_scope"] == "searched"
    assert best["ii_model"] == "periodic_searched_best"
