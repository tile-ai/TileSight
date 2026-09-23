"""Policy-level tests for native boundaries, waves, and topology semantics."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from tilesight.modeling import (
    Kernel,
    LoopRegion,
    NativeModelError,
    NativeModelOptions,
    PeriodicAxisIR,
    PhaseRegion,
    ResourceTiming,
    SequenceRegion,
    Timing,
    Work,
    evaluate_region,
    model_native,
    ns,
)
from tilesight.modeling.native_policy import (
    LaunchTopology,
    LegacyStageBoundary,
    SmallTileLatencyConfig,
)
from tilesight.modeling.resident_schedule import (
    ResidentScheduleConfig,
    ResourceScope,
)


def _policy_kernel(
    work_grid=(5,),
    cluster=(1,),
    resident_ctas=1,
    physical_grid=None,
    static_timings=False,
):
    if physical_grid is None:
        physical_grid = work_grid
    kernel = Kernel("native_policy")
    with kernel.launch(
        work_grid=work_grid,
        physical_grid=physical_grid,
        threads=128,
        resident_ctas=resident_ctas,
        cluster=cluster,
    ) as launch:
        with launch.periodic("ko", iterations=8, stages=3) as loop:
            tile = loop.buffer(
                "tile", scope="smem", shape=(32,), dtype="fp32", slots=3
            )
            with loop.actor("producer") as producer:
                load = producer.phase(
                    "load",
                    work=Work.copy(128, source="gmem", target="smem"),
                    timing=(
                        Timing(4 * ns, (ResourceTiming("memory", 4 * ns),))
                        if static_timings
                        else None
                    ),
                    writes=(tile,),
                )
            with loop.actor("tensor") as tensor:
                compute = tensor.phase(
                    "compute",
                    work=Work.mma(4096),
                    timing=(
                        Timing(2 * ns, (ResourceTiming("tensor", 2 * ns),))
                        if static_timings
                        else None
                    ),
                    reads=(tile,),
                )
            loop.pipeline_buffer(
                tile,
                acquire=load.start,
                release=compute.done,
                capacity=3,
            )
    ir = kernel.build()
    phases = {phase.name: phase for phase in ir.periodic_loop("ko").phases}
    region = LoopRegion(
        "ko_region",
        trip_count=8,
        body=SequenceRegion(
            "body",
            (PhaseRegion(phases["load"]), PhaseRegion(phases["compute"])),
        ),
        periodic_axis=PeriodicAxisIR(
            loop_name="ko",
            ii_mode="periodic_best",
            legacy_stage_boundary=LegacyStageBoundary(
                memory_resources=("memory",),
                compute_resources=("tensor",),
            ),
        ),
    )
    return ir, region


class _ContextOracle:
    def __init__(self, context=None):
        self.context = context
        self.bound_contexts = []

    def bind_context(self, context):
        self.bound_contexts.append(context)
        return _ContextOracle(context)

    def resolve(self, phase):
        active = 4 if self.context is None else self.context.active_physical_sms
        if phase.name == "load":
            latency = active * ns
            return Timing(latency, (ResourceTiming("memory", latency),))
        return Timing(2 * ns, (ResourceTiming("tensor", 2 * ns),))


class _CooperativeClusterOracle(_ContextOracle):
    supports_cooperative_cluster = True
    cooperative_cluster_size = 2
    consumes_cluster_operand_plan = True
    consumes_pair_tensor_throughput = True
    consumes_per_sm_local_smem = True
    consumes_cluster_issue_latency = True

    def bind_context(self, context):
        self.bound_contexts.append(context)
        return _CooperativeClusterOracle(context)

    def resolve(self, phase):
        timing = super().resolve(phase)
        if self.context is not None and phase.name == "compute":
            extra = (
                self.context.topology.cluster_barrier_s
                + self.context.topology.cluster_issue_s
            )
            return replace(timing, latency=timing.latency + extra)
        return timing


class _ClaimsOnlyClusterOracle:
    supports_cooperative_cluster = True
    cooperative_cluster_size = 2
    consumes_cluster_operand_plan = True
    consumes_pair_tensor_throughput = True
    consumes_per_sm_local_smem = True
    consumes_cluster_issue_latency = True

    def resolve(self, phase):
        if phase.name == "load":
            return Timing(4 * ns, (ResourceTiming("memory", 4 * ns),))
        return Timing(2 * ns, (ResourceTiming("tensor", 2 * ns),))


class _BindWithoutResolveClusterOracle(_ClaimsOnlyClusterOracle):
    def bind_context(self, _context):
        return SimpleNamespace(resolve_fused=lambda *_args: None)


class _ExternalPricedOracle(_ContextOracle):
    consumes_external_priced_resources = True
    external_priced_resources = ("memory",)


def _arch(sm_count=4):
    return SimpleNamespace(
        sm_count=sm_count,
        configurable_smem_capacity=128 * 1024,
        tmem_capacity_per_sm=0,
    )


def test_legacy_stage_boundary_reproduces_explicit_fill_steady_drain():
    ir, region = _policy_kernel()
    result = evaluate_region(
        ir,
        region,
        NativeModelOptions(boundary_policy="legacy_stage"),
        oracle=_ContextOracle(),
    )
    # stages=3 => depth=2; memory=4ns, compute=2ns, trip_count=8.
    assert result.first_s == pytest.approx(8 * ns)
    assert result.steady_s == pytest.approx(6 * 4 * ns)
    assert result.drain_s == pytest.approx(4 * ns)
    assert result.total_s == pytest.approx(36 * ns)
    assert result.strategy == "periodic_macro_legacy_stage"
    assert result.estimate_scope == "calibrated_legacy_stage_boundary"


def test_legacy_stage_fails_closed_without_explicit_resource_partition():
    ir, region = _policy_kernel()
    region = replace(
        region,
        periodic_axis=replace(region.periodic_axis, legacy_stage_boundary=None),
    )
    with pytest.raises(NativeModelError, match="explicit.*resource partition"):
        evaluate_region(
            ir,
            region,
            NativeModelOptions(boundary_policy="legacy_stage"),
            oracle=_ContextOracle(),
        )


def test_finite_witness_expands_empty_start_and_final_drain():
    ir, region = _policy_kernel()
    region = replace(
        region,
        periodic_axis=replace(
            region.periodic_axis,
            boundary_anchor_phase="compute",
        ),
    )
    latency = SmallTileLatencyConfig(
        load_phases=("load",),
        fixed_load_latency_s=10 * ns,
        outstanding_credits=2,
    )
    periodic = evaluate_region(
        ir,
        region,
        NativeModelOptions(
            boundary_policy="periodic_witness",
            small_tile=latency,
        ),
        oracle=_ContextOracle(),
    )
    finite = evaluate_region(
        ir,
        region,
        NativeModelOptions(
            boundary_policy="finite_witness",
            small_tile=latency,
        ),
        oracle=_ContextOracle(),
    )
    # The producer can fill two initially empty slots at 0ns and 4ns.  The
    # first compute begins when the first asynchronous load completes at 10ns;
    # the final compute drains for 2ns.  A cyclic affine witness includes a
    # predecessor outside the finite window and is therefore 2ns longer.
    assert finite.first_s == pytest.approx(10 * ns)
    assert finite.steady_s == pytest.approx(40 * ns)
    assert finite.drain_s == pytest.approx(2 * ns)
    assert finite.total_s == pytest.approx(52 * ns)
    assert finite.total_s < periodic.total_s
    assert finite.strategy == "periodic_finite_witness"
    assert finite.estimate_scope == "constructive_finite_source_schedule"


def test_finite_witness_requires_explicit_boundary_anchor_and_limit():
    ir, region = _policy_kernel()
    with pytest.raises(NativeModelError, match="boundary_anchor_phase"):
        evaluate_region(
            ir,
            region,
            NativeModelOptions(boundary_policy="finite_witness"),
            oracle=_ContextOracle(),
        )
    region = replace(
        region,
        periodic_axis=replace(
            region.periodic_axis,
            boundary_anchor_phase="compute",
        ),
    )
    with pytest.raises(NativeModelError, match="max_finite_iterations"):
        evaluate_region(
            ir,
            region,
            NativeModelOptions(
                boundary_policy="finite_witness",
                max_finite_iterations=7,
            ),
            oracle=_ContextOracle(),
        )


def test_small_tile_latency_and_credit_use_existing_periodic_tokens():
    ir, region = _policy_kernel()
    baseline = evaluate_region(ir, region, oracle=_ContextOracle())
    configured = evaluate_region(
        ir,
        region,
        NativeModelOptions(
            small_tile=SmallTileLatencyConfig(
                load_phases=("load",),
                fixed_load_latency_s=20 * ns,
                outstanding_credits=1,
            )
        ),
        oracle=_ContextOracle(),
    )
    assert configured.credit_ii_s > baseline.credit_ii_s
    assert configured.selected_ii_s >= configured.credit_ii_s

    with pytest.raises(NativeModelError, match="resolve uniquely"):
        evaluate_region(
            ir,
            region,
            NativeModelOptions(
                small_tile=SmallTileLatencyConfig(
                    load_phases=("typo",), fixed_load_latency_s=1 * ns
                )
            ),
            oracle=_ContextOracle(),
        )


def test_legacy_wave_rebinds_underfilled_tail_and_preserves_head_penalty():
    ir, region = _policy_kernel(work_grid=(5,))
    native_oracle = _ContextOracle()
    native = model_native(
        ir,
        region,
        _arch(),
        NativeModelOptions(kernel_launch_s=0.0),
        oracle=native_oracle,
    )
    calibrated_oracle = _ContextOracle()
    calibrated = model_native(
        ir,
        region,
        _arch(),
        NativeModelOptions(
            boundary_policy="legacy_stage",
            wave_policy="legacy_calibrated",
            kernel_launch_s=0.0,
        ),
        oracle=calibrated_oracle,
    )
    assert native.grid.full_waves == 1
    assert native.grid.tail_work_units == 1
    assert native.tail_wave_s < native.resident_group_s
    provenance = dict(calibrated.provenance)
    assert provenance["primary_context_binding"] == "bound"
    assert provenance["tail_context_binding"] == "bound"
    assert provenance["head_penalty"] > 1.0
    assert calibrated.kernel_body_s > (
        calibrated.resident_group_s + calibrated.tail_wave_s
    )


def test_underfilled_efficiency_is_explicit_and_disabled_by_default():
    ir, region = _policy_kernel(work_grid=(2,))
    baseline = model_native(
        ir,
        region,
        _arch(),
        NativeModelOptions(kernel_launch_s=0.0),
        oracle=_ContextOracle(),
    )
    adjusted = model_native(
        ir,
        region,
        _arch(),
        NativeModelOptions(
            kernel_launch_s=0.0,
            small_tile=SmallTileLatencyConfig(
                underfilled_admission_s=3 * ns,
                underfilled_efficiency=0.5,
            ),
        ),
        oracle=_ContextOracle(),
    )
    assert baseline.grid.full_waves == 0
    assert baseline.grid.tail_work_units == 2
    assert adjusted.kernel_body_s == pytest.approx(
        baseline.kernel_body_s / 0.5 + 3 * ns
    )
    assert dict(baseline.provenance)["underfilled_adjustment"] == "disabled"
    assert dict(adjusted.provenance)["underfilled_adjustment"] == (
        "explicit_small_tile_policy"
    )


def test_cluster_topology_cannot_alias_resident_or_cooperative_work():
    ir, region = _policy_kernel(cluster=(2,))
    with pytest.raises(NativeModelError, match="explicit LaunchTopology"):
        model_native(ir, region, _arch(), oracle=_ContextOracle())
    with pytest.raises(NativeModelError, match="independent CTA work tiles"):
        model_native(
            ir,
            region,
            _arch(),
            NativeModelOptions(
                launch_topology=LaunchTopology.ordinary_cluster(
                    2, tma_multicast=True
                )
            ),
            oracle=_ContextOracle(),
        )
    cooperative = LaunchTopology.cooperative_cluster(
        2,
        operand_plan=(("A", "duplicate"), ("B", "partition")),
    )
    cooperative_ir, cooperative_region = _policy_kernel(
        cluster=(2,), physical_grid=(10,)
    )
    with pytest.raises(NativeModelError, match="cluster-aware cost oracle"):
        model_native(
            cooperative_ir,
            cooperative_region,
            _arch(),
            NativeModelOptions(launch_topology=cooperative),
            oracle=_ContextOracle(),
        )


def test_static_topology_grids_distinguish_ctas_from_logical_supertile_work():
    wrong_single, wrong_single_region = _policy_kernel(physical_grid=(6,))
    with pytest.raises(NativeModelError, match="single topology requires physical_grid"):
        model_native(
            wrong_single,
            wrong_single_region,
            _arch(),
            oracle=_ContextOracle(),
        )

    wrong_resident, wrong_resident_region = _policy_kernel(
        work_grid=(8,), resident_ctas=2, physical_grid=(9,)
    )
    with pytest.raises(
        NativeModelError, match="resident topology requires physical_grid"
    ):
        model_native(
            wrong_resident,
            wrong_resident_region,
            _arch(),
            NativeModelOptions(
                launch_topology=LaunchTopology.resident(),
                multi_cta_policy="resource_bound",
            ),
            oracle=_ContextOracle(),
        )

    wrong_ordinary, wrong_ordinary_region = _policy_kernel(
        cluster=(2,), physical_grid=(10,)
    )
    with pytest.raises(
        NativeModelError, match="ordinary_cluster topology requires physical_grid"
    ):
        model_native(
            wrong_ordinary,
            wrong_ordinary_region,
            _arch(),
            NativeModelOptions(
                launch_topology=LaunchTopology.ordinary_cluster(2)
            ),
            oracle=_ContextOracle(),
        )

    wrong_cooperative, wrong_cooperative_region = _policy_kernel(cluster=(2,))
    cooperative = LaunchTopology.cooperative_cluster(
        2,
        operand_plan=(("A", "duplicate"), ("B", "partition")),
    )
    with pytest.raises(
        NativeModelError, match="cooperative_cluster topology requires physical_grid"
    ):
        model_native(
            wrong_cooperative,
            wrong_cooperative_region,
            _arch(),
            NativeModelOptions(launch_topology=cooperative),
            oracle=_CooperativeClusterOracle(),
        )

    wrong_axis, wrong_axis_region = _policy_kernel(
        work_grid=(5, 7), cluster=(2, 1), physical_grid=(5, 14)
    )
    with pytest.raises(
        NativeModelError, match="cooperative_cluster topology requires physical_grid"
    ):
        model_native(
            wrong_axis,
            wrong_axis_region,
            _arch(),
            NativeModelOptions(launch_topology=cooperative),
            oracle=_CooperativeClusterOracle(),
        )


def test_cluster_aware_cooperative_oracle_uses_logical_supertile_slots():
    # Rank-one work/physical grids and a rank-three cluster are normalized by
    # padding trailing dimensions with one before the dimension-wise check.
    ir, region = _policy_kernel(cluster=(2, 1, 1), physical_grid=(10,))
    topology = LaunchTopology.cooperative_cluster(
        2,
        operand_plan=(("A", "duplicate"), ("B", "partition")),
        tma_multicast=True,
        cluster_barrier_s=1 * ns,
        cluster_issue_s=2 * ns,
    )
    result = model_native(
        ir,
        region,
        _arch(),
        NativeModelOptions(
            launch_topology=topology,
            kernel_launch_s=0.0,
        ),
        oracle=_CooperativeClusterOracle(),
    )
    provenance = dict(result.provenance)
    assert provenance["logical_slot_count"] == 2
    assert dict(provenance["launch_topology"])["kind"] == (
        "cooperative_cluster"
    )
    contract = dict(provenance["cluster_cost_contract"])
    assert contract["generic_dsm_bandwidth"] == "not_created"
    assert contract["per_sm_local_smem"] is True
    assert provenance["spatial_grid_semantics"] == (
        "logical_supertile_work_grid_cluster_expanded_cta_grid"
    )
    assert result.grid.work_units == 5
    assert result.grid.physical_programs == 10
    assert result.spatial_scope == "cooperative_cluster_cluster_aware_oracle"
    assert result.grid.active_work_units == 2
    assert result.grid.active_physical_ctas == 4


def test_cooperative_cluster_requires_bound_dynamic_costs_not_claims_or_static():
    topology = LaunchTopology.cooperative_cluster(
        2,
        operand_plan=(("A", "duplicate"), ("B", "partition")),
        cluster_barrier_s=1.0,
        cluster_issue_s=1.0,
    )
    dynamic_ir, dynamic_region = _policy_kernel(
        cluster=(2,), physical_grid=(10,)
    )
    with pytest.raises(NativeModelError, match="callable bind_context"):
        model_native(
            dynamic_ir,
            dynamic_region,
            _arch(),
            NativeModelOptions(launch_topology=topology),
            oracle=_ClaimsOnlyClusterOracle(),
        )
    with pytest.raises(NativeModelError, match="must return an oracle with resolve"):
        model_native(
            dynamic_ir,
            dynamic_region,
            _arch(),
            NativeModelOptions(launch_topology=topology),
            oracle=_BindWithoutResolveClusterOracle(),
        )

    # Previously this returned the same few-nanosecond total even when the
    # topology declared a one-second barrier/issue cost, because static phase
    # timings bypassed the oracle. It must now fail rather than ignore 2 s.
    static_ir, static_region = _policy_kernel(
        cluster=(2,), physical_grid=(10,), static_timings=True
    )
    with pytest.raises(NativeModelError, match="static Phase.timing"):
        model_native(
            static_ir,
            static_region,
            _arch(),
            NativeModelOptions(launch_topology=topology),
            oracle=_CooperativeClusterOracle(),
        )


def test_constructive_resident_best_worst_integrate_product_dag_witness():
    ir, region = _policy_kernel(work_grid=(8,), resident_ctas=2)
    config = ResidentScheduleConfig(
        resident_ctas=2,
        resource_scopes={
            "memory": ResourceScope.SM_SHARED,
            "tensor": ResourceScope.SM_SHARED,
        },
    )
    best = model_native(
        ir,
        region,
        _arch(sm_count=2),
        NativeModelOptions(
            multi_cta_policy="constructive_best",
            resident_schedule=config,
            kernel_launch_s=0.0,
        ),
        oracle=_ContextOracle(),
    )
    worst = model_native(
        ir,
        region,
        _arch(sm_count=2),
        NativeModelOptions(
            multi_cta_policy="constructive_worst",
            resident_schedule=config,
            kernel_launch_s=0.0,
        ),
        oracle=_ContextOracle(),
    )
    assert best.spatial_scope == "resident_group_constructive_best_product_dag"
    assert worst.spatial_scope == "resident_group_constructive_worst_product_dag"
    assert best.resident_group_s <= worst.resident_group_s
    records = dict(best.provenance)["resident_schedule_records"]
    assert records
    primary = dict(dict(records)["primary"])
    assert primary["group_period_s"] > 0.0
    assert primary["phase_starts"]


def test_constructive_resident_external_resource_requires_priced_oracle():
    ir, region = _policy_kernel(work_grid=(8,), resident_ctas=2)
    config = ResidentScheduleConfig(
        resident_ctas=2,
        resource_scopes={
            "memory": ResourceScope.EXTERNAL_PRICED,
            "tensor": ResourceScope.SM_SHARED,
        },
    )
    options = NativeModelOptions(
        multi_cta_policy="constructive_best",
        resident_schedule=config,
        kernel_launch_s=0.0,
    )
    with pytest.raises(NativeModelError, match="context-bound oracle"):
        model_native(
            ir,
            region,
            _arch(sm_count=2),
            options,
            oracle=_ContextOracle(),
        )
    result = model_native(
        ir,
        region,
        _arch(sm_count=2),
        options,
        oracle=_ExternalPricedOracle(),
    )
    records = dict(result.provenance)["resident_schedule_records"]
    primary = dict(dict(records)["primary"])
    assert primary["external_priced_resources"] == ("memory",)
