"""Source-shaped cooperative CTA2 DAG tests."""

from __future__ import annotations

import numpy as np
import pytest

from tilesight.arch.b200 import B200
from tilesight.modeling import (
    B200Cta2ExecutionPlan,
    B200Cta2NativeOptions,
    Cta2OperandRoute,
    Cta2PhysicalTransfer,
    Cta2ProtocolCosts,
    ModelingValidationError,
    model_b200_cta2_native,
    ns,
)
from tilesight.modeling.tests.fixtures.legacy_gemm_b200_cta2 import (
    build_b200_cta2_gemm,
)


def _model(options=None):
    state = np.random.get_state()
    try:
        np.random.seed(20260818)
        return model_b200_cta2_native(
            build_b200_cta2_gemm(),
            B200().set_to_microbench(),
            options,
        )
    finally:
        np.random.set_state(state)


def _trace_resources(region):
    values = [item.resource for item in region.resource_trace]
    for child in region.children:
        values.extend(_trace_resources(child))
    return values


def test_source_plan_keeps_two_participants_without_generic_dsm_payload():
    result = _model()
    plan = result.plan
    assert plan.cluster_axis == "M"
    assert plan.topology_operand_plan == (("A", "partition"), ("B", "partition"))
    assert plan.payload_transfer_bytes_per_iteration == 0.0
    assert not plan.has_peer_smem
    assert not plan.has_tma_multicast
    assert {
        (item.operand, item.shard, item.issuer_rank, item.destination_ranks, item.path)
        for item in plan.physical_transfers
    } == {
        ("A", "rank0_partition", 0, (0,), "tma_local"),
        ("A", "rank1_partition", 1, (1,), "tma_local"),
        ("B", "rank0_partition", 0, (0,), "tma_local"),
        ("B", "rank1_partition", 1, (1,), "tma_local"),
    }
    assert plan.global_request_bytes_per_iteration == sum(
        sum(route.global_load_bytes) for route in plan.routes
    )
    assert plan.delivered_smem_bytes_per_iteration == sum(
        sum(route.local_bytes) for route in plan.routes
    )

    routes = plan.route_map
    # Canonical example: CTA tile 128x128x64, bf16. Pair M is 256;
    # each CTA owns 128x64 of A and 64x64 of B.
    assert routes["A"].local_bytes == (128 * 64 * 2, 128 * 64 * 2)
    assert routes["B"].local_bytes == (64 * 64 * 2, 64 * 64 * 2)
    assert result.kernel.launches[0].work_grid == (16, 32, 1)
    assert result.kernel.launches[0].physical_grid == (32, 32, 1)

    names = {phase.name for phase in result.kernel.periodic_loop("ko").phases}
    assert names == {
        "load_A_rank0",
        "load_B_rank0",
        "load_A_rank1",
        "load_B_rank1",
        "cluster_ready",
        "pair_issue",
        "pair_inputs_consumed",
        "pair_mma",
    }
    assert result.native.spatial_scope == "cooperative_cluster_cluster_aware_oracle"
    assert result.native.grid.active_physical_ctas == (
        2 * result.native.grid.active_work_units
    )
    assert tuple(
        (item.rank, item.smem_all_stages_bytes, item.tmem_bytes)
        for item in result.participant_footprints
    ) == (
        (0, 3 * (128 * 64 * 2 + 64 * 64 * 2), 128 * 128 * 4),
        (1, 3 * (128 * 64 * 2 + 64 * 64 * 2), 128 * 128 * 4),
    )
    assert {peak.execution_scope for peak in result.native.liveness.peaks} == {
        "physical_cta",
        "physical_sm",
    }
    assert result.native.liveness.guard_eligible is False
    provenance = dict(result.provenance)
    assert provenance["generic_dsm_bandwidth"] == "not_created"
    assert provenance["independent_hardware_calibration"] is False
    assert 0.5 < result.body_ratio < 2.0


def test_peer_smem_route_is_an_explicit_payload_and_link_reservation():
    folded = _model(
        B200Cta2NativeOptions(
            plan_kind="legacy_broadcast_peer_smem",
            plan_source="test_peer_plan",
        )
    )
    explicit = _model(
        B200Cta2NativeOptions(
            plan_kind="legacy_broadcast_peer_smem",
            plan_source="test_peer_plan",
            protocol_costs=Cta2ProtocolCosts(
                transfer_pricing="explicit_link",
                transfer_bandwidth_bytes_per_s=1.0e12,
                transfer_fixed_latency_s=5 * ns,
                cluster_barrier_s=3 * ns,
                cluster_issue_s=2 * ns,
            ),
        )
    )
    assert explicit.plan.has_peer_smem
    assert explicit.plan.payload_transfer_bytes_per_iteration > 0.0
    peer = [
        item for item in explicit.plan.physical_transfers if item.path == "peer_smem"
    ]
    assert len(peer) == 1
    assert peer[0].issuer_rank == 0
    assert peer[0].destination_ranks == (1,)
    assert "transfer_B" in {
        phase.name for phase in explicit.kernel.periodic_loop("ko").phases
    }
    assert "cluster_link" in _trace_resources(explicit.native.region)
    assert explicit.native.kernel_body_s > folded.native.kernel_body_s
    contract = dict(dict(explicit.native.provenance)["cluster_cost_contract"])
    assert contract["peer_smem_resources"] == ("cluster_link",)
    assert contract["generic_dsm_bandwidth"] == "not_created"


def test_tma_multicast_is_one_source_load_with_two_smem_destinations():
    result = _model(
        B200Cta2NativeOptions(
            plan_kind="legacy_broadcast_multicast",
            plan_source="test_multicast_plan",
        )
    )
    route = result.plan.route_map["B"]
    assert route.mode == "multicast"
    assert result.plan.has_tma_multicast
    assert not result.plan.has_peer_smem
    phases = {phase.name: phase for phase in result.kernel.periodic_loop("ko").phases}
    assert "transfer_B" not in phases
    assert {buffer.name for buffer in phases["load_B_rank0"].writes} == {
        "B_rank0",
        "B_rank1",
    }
    assert result.plan.global_request_bytes_per_iteration < (
        result.plan.delivered_smem_bytes_per_iteration
    )


def test_explicit_link_pricing_cannot_be_unpriced():
    with pytest.raises(ModelingValidationError, match="bandwidth or fixed latency"):
        Cta2ProtocolCosts(transfer_pricing="explicit_link")


def test_equal_byte_totals_cannot_change_multicast_into_peer_smem():
    a = Cta2OperandRoute(
        "A", "partition", local_bytes=(10, 10), global_load_bytes=(10, 10)
    )
    b = Cta2OperandRoute(
        "B",
        "multicast",
        local_bytes=(10, 10),
        global_load_bytes=(10, 0),
        transfer_kind="tma_multicast",
        transfer_bytes=10,
    )
    transfers = (
        Cta2PhysicalTransfer("a0", "A", "rank0", 0, (0,), "tma_local", 10),
        Cta2PhysicalTransfer("a1", "A", "rank1", 1, (1,), "tma_local", 10),
        Cta2PhysicalTransfer("b0", "B", "common", 0, (0,), "tma_local", 10),
        Cta2PhysicalTransfer(
            "b_peer", "B", "common", 0, (1,), "peer_smem", 10,
            source_level="smem",
        ),
    )
    with pytest.raises(ModelingValidationError, match="paths.*multicast"):
        B200Cta2ExecutionPlan("M", (a, b), transfers)


def test_physical_transfers_preserve_per_rank_bytes_and_memory_levels():
    a = Cta2OperandRoute(
        "A", "partition", local_bytes=(8, 12), global_load_bytes=(8, 12)
    )
    b = Cta2OperandRoute(
        "B", "partition", local_bytes=(5, 5), global_load_bytes=(5, 5)
    )
    wrong_rank_split = (
        Cta2PhysicalTransfer("a0", "A", "rank0", 0, (0,), "tma_local", 10),
        Cta2PhysicalTransfer("a1", "A", "rank1", 1, (1,), "tma_local", 10),
        Cta2PhysicalTransfer("b0", "B", "rank0", 0, (0,), "tma_local", 5),
        Cta2PhysicalTransfer("b1", "B", "rank1", 1, (1,), "tma_local", 5),
    )
    with pytest.raises(ModelingValidationError, match="per-rank bytes"):
        B200Cta2ExecutionPlan("M", (a, b), wrong_rank_split)
    with pytest.raises(ModelingValidationError, match="ddr to smem"):
        Cta2PhysicalTransfer(
            "reversed",
            "A",
            "rank0",
            0,
            (0,),
            "tma_local",
            8,
            source_level="smem",
            target_level="ddr",
        )


def test_input_consumed_milestone_can_release_stages_before_pair_completion():
    result = _model(
        B200Cta2NativeOptions(
            protocol_costs=Cta2ProtocolCosts(input_consumed_latency_s=10 * ns)
        )
    )
    loop = result.kernel.periodic_loop("ko")
    assert {item.release.phase_name for item in loop.pipeline_buffers} == {
        "pair_inputs_consumed"
    }
    assert result.native.region.children[0].selected_ii_s > 0.0


def test_cta2_uses_finite_source_boundary_and_separates_tma_latency():
    baseline = _model(B200Cta2NativeOptions(boundary_policy="finite_witness"))
    delayed = _model(
        B200Cta2NativeOptions(
            protocol_costs=Cta2ProtocolCosts(tma_fixed_latency_s=100 * ns),
            boundary_policy="finite_witness",
        )
    )
    baseline_loop = baseline.native.region.children[0]
    delayed_loop = delayed.native.region.children[0]
    assert baseline_loop.strategy == "periodic_finite_witness"
    assert baseline_loop.first_s > 0.0
    assert baseline_loop.drain_s > 0.0
    # Fixed TMA completion latency delays first data readiness but does not
    # reserve the TMA throughput resource for the added latency.
    assert delayed_loop.first_s == pytest.approx(baseline_loop.first_s + 100 * ns)
    assert delayed_loop.selected_ii_s == pytest.approx(baseline_loop.selected_ii_s)
    assert delayed_loop.drain_s == pytest.approx(baseline_loop.drain_s)
    assert delayed.native.kernel_body_s > baseline.native.kernel_body_s
    provenance = dict(delayed.provenance)
    assert provenance["boundary_policy"] == "finite_witness"
    assert provenance["tma_fixed_latency_s"] == pytest.approx(100 * ns)


def test_cta2_default_keeps_fast_periodic_boundary():
    result = _model()
    mainloop = result.native.region.children[0]
    assert mainloop.strategy == "periodic_macro"
    assert dict(result.provenance)["boundary_policy"] == "periodic_witness"
