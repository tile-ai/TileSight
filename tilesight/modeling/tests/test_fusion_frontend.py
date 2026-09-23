"""Tests for explicit fragment handoffs and mechanical traffic accounting."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from tilesight.modeling import (
    ExecutionDomain,
    FeasibilityStatus,
    FusionPlan,
    FullModelError,
    FullModelOptions,
    Kernel,
    LayoutMapping,
    ModelingValidationError,
    OwnershipMap,
    Timing,
    Work,
    analyze_fusion,
    analyze_liveness,
    legacy_level_path,
    model,
)


def _gemm_bias(
    *, model_adapter=None, via="register", size=4096, prove_register=False
):
    params = {} if model_adapter is None else {"model_adapter": model_adapter}
    kernel = Kernel("gemm_bias", params=params)
    domain = None
    ownership = None
    if via == "register" and prove_register:
        domain = ExecutionDomain(
            "thread", instances_per_cta=128, members_per_instance=1
        )
        ownership = OwnershipMap(
            domain,
            layout="row_major",
            index_map={"element": "threadIdx.x"},
        )
    with kernel.launch(work_grid=(1,), threads=128) as launch:
        with launch.periodic("tiles", iterations=1) as loop:
            with loop.actor(
                "epilogue_warpgroup",
                execution_scope="thread" if via == "register" else "unspecified",
                execution_domain=domain,
            ) as actor:
                gemm = actor.phase(
                    "gemm",
                    work=Work.mma(1024),
                    timing=Timing(1.0),
                )
                bias = actor.phase(
                    "bias",
                    work=Work.pointwise(256),
                    timing=Timing(1.0),
                )
    accumulator = kernel.tensor_value(
        "accumulator",
        shape=(size // 4,),
        dtype="fp32",
        bytes=size,
        layout="row_major",
        ownership=ownership,
    )
    baseline = kernel.materialize(
        "gemm_bias_baseline",
        accumulator,
        producer=gemm.done,
        consumer=bias.start,
        producer_fragment="gemm",
        consumer_fragment="bias",
        storage="ddr",
        producer_source="register",
        consumer_target="register",
    )
    with kernel.fusion("gemm_bias_fusion") as fusion:
        fusion.handoff(
            "accumulator_onchip",
            baseline,
            via=via,
            execution_scope="thread" if via == "register" else "cta",
            release=bias.done,
        )
    return kernel.build()


def _fa_qk_softmax_pv():
    kernel = Kernel("fa_qk_softmax_pv")
    with kernel.launch(work_grid=(1,), threads=256) as launch:
        with launch.periodic("kv", iterations=32, stages=2) as loop:
            with loop.actor("tensor") as tensor:
                qk = tensor.phase("qk", work=Work.mma(4096), timing=Timing(1.0))
                pv = tensor.phase("pv", work=Work.mma(4096), timing=Timing(1.0))
            with loop.actor("softmax") as vector:
                softmax = vector.phase(
                    "softmax",
                    work=Work.reduce(1024),
                    timing=Timing(1.0),
                )
    scores = kernel.tensor_value(
        "scores",
        shape=(128, 64),
        dtype="fp32",
        bytes=128 * 64 * 4,
        layout="row_major",
    )
    probabilities = kernel.tensor_value(
        "probabilities",
        shape=(128, 64),
        dtype="bf16",
        bytes=128 * 64 * 2,
        layout="row_major",
    )
    score_baseline = kernel.materialize(
        "qk_softmax_baseline",
        scores,
        producer=qk.done,
        consumer=softmax.start,
        producer_fragment="qk",
        consumer_fragment="softmax",
        storage="ddr",
    )
    probability_baseline = kernel.materialize(
        "softmax_pv_baseline",
        probabilities,
        producer=softmax.done,
        consumer=pv.start,
        producer_fragment="softmax",
        consumer_fragment="pv",
        storage="ddr",
    )
    with kernel.fusion("attention_fusion") as fusion:
        fusion.handoff(
            "scores_tmem",
            score_baseline,
            via="tmem",
            execution_scope="cta",
            slots=2,
            alias_group="attention_intermediate",
        )
        fusion.handoff(
            "probabilities_tmem",
            probability_baseline,
            via="tmem",
            execution_scope="cta",
            slots=2,
            alias_group="attention_intermediate",
        )
    return kernel.build()


def _traffic(report, level):
    return next(item for item in report.traffic if item.memory_level == level)


def test_gemm_bias_eliminates_ddr_l2_and_introduces_register_handoff():
    ir = _gemm_bias(prove_register=True)
    report = analyze_fusion(ir)
    dag = ir.lower_periodic("tiles")

    assert report.status is FeasibilityStatus.PROVEN_FEASIBLE
    assert any(
        "compatible explicit OwnershipMap" in item
        for item in report.diagnostics
    )
    assert report.handoff_count == 1
    assert report.eliminated_global_bytes == 8192
    for level in ("ddr", "l2"):
        item = _traffic(report, level)
        assert item.baseline_read_bytes == 4096
        assert item.baseline_write_bytes == 4096
        assert item.eliminated_read_bytes == 4096
        assert item.eliminated_write_bytes == 4096
        assert item.fused_read_bytes == 0
        assert item.fused_write_bytes == 0
    registers = _traffic(report, "register")
    assert registers.introduced_write_bytes == 4096
    assert registers.introduced_read_bytes == 4096
    assert registers.fused_write_bytes == 4096
    assert registers.fused_read_bytes == 4096
    path = report.value_paths[0]
    assert path.baseline_store_path == ("register", "ddr")
    assert path.baseline_load_paths == (("ddr", "register"),)
    assert path.fused_handoff_paths == (("register",),)
    assert any("generic equal-byte" in item for item in report.diagnostics)
    dependency = next(
        item for item in dag.dependencies if item.name.startswith("handoff:")
    )
    assert (dependency.source, dependency.target) == ("gemm", "bias")
    assert dependency.min_delay == 1.0
    token = next(
        item for item in dag.token_buffers if item.name.startswith("__fusion__/")
    )
    assert (token.acquire, token.release, token.capacity) == ("gemm", "bias", 1)
    assert token.acquire_offset == 1.0
    assert token.release_offset == 1.0


def test_register_handoff_without_ownership_map_is_unknown_but_traffic_closed():
    report = analyze_fusion(_gemm_bias())

    assert report.status is FeasibilityStatus.UNKNOWN
    assert report.eliminated_global_bytes == 8192
    registers = _traffic(report, "register")
    assert registers.introduced_write_bytes == 4096
    assert registers.introduced_read_bytes == 4096
    assert any("ownership=user_asserted" in item for item in report.diagnostics)
    assert any("traffic accounting remains complete" in item for item in report.diagnostics)


def test_fa_qk_softmax_pv_reports_two_handoffs_and_alias_liveness():
    ir = _fa_qk_softmax_pv()
    fusion = analyze_fusion(ir)
    liveness = analyze_liveness(ir)

    assert fusion.status is FeasibilityStatus.PROVEN_FEASIBLE
    assert fusion.handoff_count == 2
    assert {item.value for item in fusion.value_paths} == {
        "scores",
        "probabilities",
    }
    tmem = _traffic(fusion, "tmem")
    expected = 128 * 64 * (4 + 2)
    assert tmem.introduced_read_bytes == expected
    assert tmem.introduced_write_bytes == expected
    peak = next(
        item
        for item in liveness.peaks
        if item.storage_scope == "tmem" and item.execution_scope == "cta"
    )
    assert peak.selected_bytes == 2 * 128 * 64 * 4
    assert peak.upper_bytes == 2 * expected
    assert any("explicit handoff storage" in item for item in liveness.diagnostics)


def test_materialization_supports_multiple_consumers_with_one_write_many_reads():
    kernel = Kernel("multi_consumer")
    with kernel.launch(work_grid=(1,), threads=32) as launch:
        with launch.periodic("loop", iterations=1) as loop:
            with loop.actor("actor") as actor:
                producer = actor.phase("produce", work=Work.opaque(), timing=Timing(1))
                first = actor.phase("first", work=Work.opaque(), timing=Timing(1))
                second = actor.phase("second", work=Work.opaque(), timing=Timing(1))
    value = kernel.tensor_value(
        "value", shape=(16,), dtype="fp32", bytes=64, layout="linear"
    )
    baseline = kernel.materialize(
        "baseline",
        value,
        producer=producer.done,
        consumer=(first.start, second.start),
        producer_fragment="producer",
        consumer_fragment=("first", "second"),
        consumer_target=("register", "smem"),
    )
    with kernel.fusion("fused") as fusion:
        fusion.handoff(
            "shared_value",
            baseline,
            via="smem",
            execution_scope="cta",
            release=(first.done, second.done),
        )
    ir = kernel.build()
    report = analyze_fusion(ir)
    dag = ir.lower_periodic("loop")
    smem = _traffic(report, "smem")
    assert smem.introduced_write_bytes == 64
    assert smem.introduced_read_bytes == 128
    assert report.value_paths[0].consumer_fragments == ("first", "second")
    assert report.value_paths[0].fused_handoff_paths == (
        ("register", "smem", "register"),
        ("register", "smem"),
    )
    assert len(
        [x for x in dag.token_buffers if x.name.startswith("__fusion__/")]
    ) == 2
    assert {
        (x.source, x.target)
        for x in dag.dependencies
        if x.name.startswith("handoff:")
    } == {("produce", "first"), ("produce", "second")}


def test_direct_memory_access_is_exact_and_global_alias_canonicalizes_to_ddr():
    kernel = Kernel("exact_access")
    with kernel.launch(work_grid=(1,), threads=32) as launch:
        with launch.periodic("loop", iterations=1) as loop:
            with loop.actor("actor") as actor:
                phase = actor.phase("phase", work=Work.opaque(), timing=Timing(1))
    value = kernel.tensor_value(
        "value", shape=(8,), dtype="fp32", bytes=32, layout="linear"
    )
    access = kernel.memory_access(
        "write",
        value,
        mode="WRITE",
        memory_level="global",
        bytes=16,
        event=phase.done,
        fragment="producer",
    )
    ir = kernel.build()
    assert access.accounting == "exact"
    assert access.mode == "write"
    assert access.event == phase.done
    assert ir.memory_accesses[0].memory_level == "ddr"


def test_exact_materialization_allows_asymmetric_cache_level_traffic():
    kernel = Kernel("asymmetric_cache")
    with kernel.launch(work_grid=(1,), threads=32) as launch:
        with launch.periodic("loop", iterations=1) as loop:
            with loop.actor("actor") as actor:
                producer = actor.phase(
                    "producer", work=Work.opaque(), timing=Timing(1)
                )
                consumer = actor.phase(
                    "consumer", work=Work.opaque(), timing=Timing(1)
                )
    value = kernel.tensor_value(
        "value", shape=(16,), dtype="fp32", bytes=64, layout="linear"
    )
    ddr_write = kernel.memory_access(
        "ddr_write",
        value,
        mode="write",
        memory_level="ddr",
        bytes=64,
        event=producer.done,
        fragment="producer",
    )
    l2_write = kernel.memory_access(
        "l2_write",
        value,
        mode="write",
        memory_level="l2",
        bytes=64,
        event=producer.done,
        fragment="producer",
    )
    l2_read = kernel.memory_access(
        "l2_read",
        value,
        mode="read",
        memory_level="l2",
        bytes=64,
        event=consumer.start,
        fragment="consumer",
    )
    baseline = kernel.exact_materialization(
        "baseline",
        value,
        writes=(ddr_write, l2_write),
        reads=(l2_read,),
        storage="ddr",
        producer_source="register",
        consumer_target="register",
    )
    with kernel.fusion("fusion") as fusion:
        fusion.handoff(
            "handoff", baseline, via="smem", execution_scope="cta"
        )

    report = analyze_fusion(kernel.build())
    assert report.status is FeasibilityStatus.PROVEN_FEASIBLE
    ddr = _traffic(report, "ddr")
    assert ddr.eliminated_write_bytes == 64
    assert ddr.eliminated_read_bytes == 0
    l2 = _traffic(report, "l2")
    assert l2.eliminated_write_bytes == 64
    assert l2.eliminated_read_bytes == 64


def test_exact_per_level_traffic_can_differ_from_logical_handoff_payload():
    kernel = Kernel("exact_traffic")
    with kernel.launch(work_grid=(1,), threads=32) as launch:
        with launch.periodic("loop", iterations=1) as loop:
            with loop.actor("actor") as actor:
                producer = actor.phase("produce", work=Work.opaque(), timing=Timing(1))
                consumer = actor.phase("consume", work=Work.opaque(), timing=Timing(1))
    value = kernel.tensor_value(
        "value", shape=(16,), dtype="fp32", bytes=64, layout="linear"
    )
    writes = tuple(
        kernel.memory_access(
            "write_%s" % level,
            value,
            mode="write",
            memory_level=level,
            bytes=traffic,
            event=producer.done,
            fragment="producer",
        )
        for level, traffic in (("ddr", 80), ("l2", 128))
    )
    reads = tuple(
        kernel.memory_access(
            "read_%s" % level,
            value,
            mode="read",
            memory_level=level,
            bytes=traffic,
            event=consumer.start,
            fragment="consumer",
        )
        for level, traffic in (("ddr", 96), ("l2", 160))
    )
    baseline = kernel.exact_materialization(
        "baseline",
        value,
        writes=writes,
        reads=reads,
        storage="ddr",
        producer_source="register",
        consumer_target="register",
        transfer_bytes=64,
    )
    with kernel.fusion("fusion") as fusion:
        fusion.handoff(
            "handoff",
            baseline,
            via="smem",
            execution_scope="cta",
        )
    report = analyze_fusion(kernel.build())
    assert _traffic(report, "ddr").eliminated_write_bytes == 80
    assert _traffic(report, "ddr").eliminated_read_bytes == 96
    assert _traffic(report, "l2").eliminated_write_bytes == 128
    assert _traffic(report, "l2").eliminated_read_bytes == 160
    assert _traffic(report, "smem").introduced_write_bytes == 64
    assert _traffic(report, "smem").introduced_read_bytes == 64
    assert not any("generic equal-byte" in item for item in report.diagnostics)


def test_invalid_layout_mapping_is_rejected_before_analysis():
    with pytest.raises(ModelingValidationError, match="bijective"):
        LayoutMapping(
            "row_major",
            "swizzled",
            "row_major",
            kind="custom",
            bijective=False,
        )
    with pytest.raises(ModelingValidationError, match="identity"):
        LayoutMapping("row_major", "swizzled", "row_major")


def test_handoff_storage_capacity_is_visible_to_liveness():
    ir = _gemm_bias(via="smem", size=4096)
    arch = SimpleNamespace(configurable_smem_capacity=1024, tmem_capacity_per_sm=0)
    report = analyze_liveness(ir, arch=arch, resident_ctas_per_sm=1)
    assert report.status is FeasibilityStatus.PROVEN_INFEASIBLE
    assert any("exceeds" in item for item in report.diagnostics)


def test_liveness_does_not_false_reject_unknown_cta_group_replication():
    ir = _gemm_bias(via="smem", size=4096)
    plan = ir.fusion_plans[0]
    grouped = replace(
        plan.handoffs[0], via="tmem", execution_scope="cta_group"
    )
    grouped_ir = replace(
        ir,
        fusion_plans=(replace(plan, handoffs=(grouped,)),),
    )
    arch = SimpleNamespace(configurable_smem_capacity=0, tmem_capacity_per_sm=5000)
    report = analyze_liveness(grouped_ir, arch=arch, resident_ctas_per_sm=2)
    assert report.status is FeasibilityStatus.UNKNOWN
    assert any("non-CTA replica scopes" in item for item in report.diagnostics)


def test_documented_legacy_memory_paths_are_directional_and_l2_is_not_a_slot():
    assert legacy_level_path([1, 1, 1, 2], role="read") == (
        "ddr",
        "smem",
        "register",
    )
    assert legacy_level_path([1, 1, 1, 2], role="write") == (
        "register",
        "smem",
        "ddr",
    )
    assert legacy_level_path([0, 1, 1, 2], role="read") == (
        "smem",
        "register",
    )
    assert legacy_level_path([0, 0, 1, 4], role="read") == ("register",)
    with pytest.raises(ModelingValidationError, match="named memory path"):
        legacy_level_path([1, 0, 0, 4], role="read")


def test_load_store_compute_wrappers_record_programmed_endpoints_and_engines():
    kernel = Kernel("primitive_wrappers")
    with kernel.launch(work_grid=(1,), threads=32) as launch:
        with launch.periodic("loop", iterations=1) as loop:
            with loop.actor("actor") as actor:
                load = actor.load(
                    "load",
                    bytes=64,
                    source="ddr",
                    destination="smem",
                    engine="tma",
                    timing=Timing(1),
                )
                compute = actor.compute(
                    "compute",
                    flops=128,
                    engine="tensor",
                    op="mma",
                    timing=Timing(1),
                )
                store = actor.store(
                    "store",
                    bytes=64,
                    source="register",
                    destination="ddr",
                    engine="ldst",
                    timing=Timing(1),
                )
    ir = kernel.build()
    assert load.work.kind == "load"
    assert dict(load.work.attrs) == {
        "destination": "smem",
        "engine": "tma",
        "source": "ddr",
    }
    assert compute.work.kind == "compute"
    assert dict(compute.work.attrs)["engine"] == "tensor"
    assert store.work.kind == "store"
    with pytest.raises(ModelingValidationError, match="caches are not programmed"):
        actor.load(
            "invalid",
            bytes=1,
            source="l2",
            destination="register",
            engine="ldst",
        )


def test_logical_materialization_rejects_l2_but_direct_traffic_accepts_it():
    kernel = Kernel("passive_l2")
    with kernel.launch(work_grid=(1,), threads=32) as launch:
        with launch.periodic("loop", iterations=1) as loop:
            with loop.actor("actor") as actor:
                producer = actor.phase("producer", work=Work.opaque(), timing=Timing(1))
                consumer = actor.phase("consumer", work=Work.opaque(), timing=Timing(1))
    value = kernel.tensor_value(
        "value", shape=(1,), dtype="fp32", bytes=4, layout="linear"
    )
    with pytest.raises(ModelingValidationError, match="passive cache"):
        kernel.materialize(
            "invalid",
            value,
            producer=producer,
            consumer=consumer,
            producer_fragment="producer",
            consumer_fragment="consumer",
            storage="l2",
        )
    access = kernel.memory_access(
        "derived_l2_read",
        value,
        mode="read",
        memory_level="l2",
        bytes=8,
        event=consumer.start,
        fragment="consumer",
    )
    assert access.memory_level == "l2"


def test_register_handoff_rejects_cross_actor_ownership():
    kernel = Kernel("cross_actor")
    with kernel.launch(work_grid=(1,), threads=32) as launch:
        with launch.periodic("loop", iterations=1) as loop:
            with loop.actor("producer_actor") as actor:
                producer = actor.phase("producer", work=Work.opaque(), timing=Timing(1))
            with loop.actor("consumer_actor") as actor:
                consumer = actor.phase("consumer", work=Work.opaque(), timing=Timing(1))
    value = kernel.tensor_value(
        "value", shape=(1,), dtype="fp32", bytes=4, layout="linear"
    )
    baseline = kernel.materialize(
        "baseline",
        value,
        producer=producer,
        consumer=consumer,
        producer_fragment="producer",
        consumer_fragment="consumer",
    )
    with kernel.fusion("fusion") as fusion:
        fusion.handoff(
            "register_handoff",
            baseline,
            via="register",
            execution_scope="warp",
        )
    with pytest.raises(ModelingValidationError, match="declared actor ownership"):
        kernel.build()


def test_register_handoff_requires_explicit_actor_execution_scope():
    ir = _gemm_bias(via="register")
    launch = ir.launches[0]
    loop = launch.periodic_loops[0]
    actor = loop.actors[0]
    unspecified_loop = replace(
        loop,
        actors=(replace(actor, execution_scope="unspecified"),),
    )
    with pytest.raises(ModelingValidationError, match="explicit execution_scope"):
        replace(
            ir,
            launches=(replace(launch, periodic_loops=(unspecified_loop,)),),
        )


def test_cross_iteration_handoff_is_rejected_until_tokens_are_distance_aware():
    ir = _gemm_bias()
    handoff = ir.fusion_plans[0].handoffs[0]
    with pytest.raises(ModelingValidationError, match="same-iteration"):
        replace(handoff, iteration_distance=1)


def test_handoff_names_are_globally_unique_across_fusion_plans():
    ir = _fa_qk_softmax_pv()
    first, second = ir.fusion_plans[0].handoffs
    duplicate = replace(second, name=first.name)
    plans = (
        FusionPlan("first_plan", ir.name, (first,)),
        FusionPlan("second_plan", ir.name, (duplicate,)),
    )
    with pytest.raises(ModelingValidationError, match="globally unique"):
        replace(ir, fusion_plans=plans)


def test_internal_handoff_tokens_cannot_collide_with_user_buffer_tokens():
    kernel = Kernel("token_namespace")
    with kernel.launch(work_grid=(1,), threads=32) as launch:
        with launch.periodic("loop", iterations=1) as loop:
            collision = loop.buffer(
                "handoff:h:consumer0", scope="smem", shape=(1,), dtype="fp32"
            )
            with loop.actor("actor") as actor:
                producer = actor.phase(
                    "producer",
                    work=Work.opaque(),
                    timing=Timing(1),
                    writes=(collision,),
                )
                consumer = actor.phase(
                    "consumer",
                    work=Work.opaque(),
                    timing=Timing(1),
                    reads=(collision,),
                )
            loop.pipeline_buffer(
                collision, acquire=producer.start, release=consumer.done
            )
    value = kernel.tensor_value(
        "value", shape=(1,), dtype="fp32", bytes=4, layout="linear"
    )
    baseline = kernel.materialize(
        "baseline",
        value,
        producer=producer,
        consumer=consumer,
        producer_fragment="producer",
        consumer_fragment="consumer",
    )
    with kernel.fusion("fusion") as fusion:
        fusion.handoff(
            "h", baseline, via="smem", execution_scope="cta"
        )

    dag = kernel.lower_periodic("loop")
    names = [token.name for token in dag.token_buffers]
    assert len(names) == len(set(names))
    assert "handoff:h:consumer0" in names
    assert "__fusion__/h/consumer0" in names


@pytest.mark.parametrize("adapter", ["gemm_pipeline_wave", "fa3", "fa4"])
def test_legacy_parity_adapters_reject_nonempty_fusion_plan(adapter):
    ir = _gemm_bias(model_adapter=adapter)
    with pytest.raises(FullModelError, match="cannot return fused latency"):
        model(ir, object(), FullModelOptions())
