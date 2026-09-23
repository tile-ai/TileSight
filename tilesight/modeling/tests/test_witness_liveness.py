"""Witness-aware lifetime, ownership, alias, and replica tests."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from tilesight.modeling._pipeline.periodic_schedule import ScheduleWitness
from tilesight.modeling import (
    ExecutionDomain,
    FeasibilityStatus,
    Kernel,
    ModelingValidationError,
    OwnershipMap,
    Timing,
    Work,
    analyze_liveness,
    ns,
)
from tilesight.modeling.adapters.gemm import extract_gemm_spec
from tilesight.modeling.tests.fixtures.legacy_gemm import build_gemm


def _witness(ii_ns, **phase_starts_ns):
    return ScheduleWitness(
        ii=ii_ns * ns,
        phase_starts={
            name: value * ns for name, value in phase_starts_ns.items()
        },
        resource_orders={},
    )


def _capacity(report, storage_scope, execution_scope):
    return next(
        item
        for item in report.capacity
        if item.storage_scope == storage_scope
        and item.execution_scope == execution_scope
    )


def _peak(report, storage_scope, execution_scope):
    return next(
        item
        for item in report.peaks
        if item.storage_scope == storage_scope
        and item.execution_scope == execution_scope
    )


def test_double_buffer_uses_witness_overlap_but_smem_stays_static():
    kernel = Kernel("double_buffer")
    with kernel.launch(work_grid=(1,), threads=32) as launch:
        with launch.periodic("k", iterations=8, stages=2) as loop:
            dynamic = loop.buffer(
                "dynamic",
                scope="tmem",
                shape=(4,),
                dtype="fp32",
                slots=2,
                execution_scope="cta",
            )
            static = loop.buffer(
                "static",
                scope="smem",
                shape=(4,),
                dtype="fp32",
                slots=2,
                execution_scope="cta",
            )
            with loop.actor("actor", execution_scope="cta") as actor:
                produce = actor.phase(
                    "produce",
                    work=Work.opaque(),
                    timing=Timing(1 * ns),
                    writes=(dynamic, static),
                )
                consume = actor.phase(
                    "consume",
                    work=Work.opaque(),
                    timing=Timing(1 * ns),
                    reads=(dynamic, static),
                )
            loop.pipeline_buffer(
                dynamic, acquire=produce.start, release=consume.done
            )
            loop.pipeline_buffer(
                static, acquire=produce.start, release=consume.done
            )

    ir = kernel.build()
    report = analyze_liveness(
        ir,
        witness=_witness(10, produce=0, consume=15),
    )

    assert _peak(report, "tmem", "cta").selected_bytes == 32
    assert _capacity(report, "tmem", "cta").allocation_kind == "witness_dynamic"
    assert _peak(report, "smem", "cta").selected_bytes == 32
    assert _capacity(report, "smem", "cta").allocation_kind == "static_allocation"
    dynamic_intervals = [
        item for item in report.intervals if item.storage_scope == "tmem"
    ]
    assert {item.iteration for item in dynamic_intervals} == {-1, 0}
    assert report.guard_eligible is False


def test_cross_iteration_register_state_uses_explicit_thread_domain_replicas():
    domain = ExecutionDomain(
        "thread",
        instances_per_cta=32,
        members_per_instance=1,
    )
    ownership = OwnershipMap(
        domain,
        layout="thread_scalar",
        index_map={"element": "threadIdx.x"},
    )
    kernel = Kernel("register_state")
    with kernel.launch(work_grid=(1,), threads=32) as launch:
        with launch.periodic("k", iterations=8, stages=2) as loop:
            registers = loop.buffer(
                "registers",
                scope="register",
                shape=(4,),
                dtype="fp32",
                slots=2,
                execution_scope="thread",
                ownership=ownership,
            )
            with loop.actor("thread_actor", execution_domain=domain) as actor:
                update = actor.phase(
                    "update",
                    work=Work.opaque(),
                    timing=Timing(2 * ns),
                    writes=(registers,),
                )
            state = loop.state("accumulator", storage=registers)
            loop.carry(
                state,
                source=update.done,
                target=update.start,
                distance=2,
            )

    ir = kernel.build()
    report = analyze_liveness(
        ir,
        witness=_witness(10, update=0),
        resident_ctas_per_sm=2,
    )

    peak = _peak(report, "register", "thread")
    capacity = _capacity(report, "register", "thread")
    assert peak.selected_bytes == 32
    assert capacity.instances_per_cta == 32
    assert capacity.members_per_instance == 1
    assert capacity.replicas_per_sm == 64
    assert capacity.aggregate_peak_bytes == 2048
    assert capacity.ownership_provenance == "explicit"
    assert {item.source_kind for item in report.intervals} == {"state"}
    loop_summary = ir.summary()["launches"][0]["periodic_loops"][0]
    actor_summary = loop_summary["actor_specs"][0]
    assert actor_summary["execution_domain"]["instances_per_cta"] == 32
    assert loop_summary["buffer_specs"][0]["ownership"]["layout"] == "thread_scalar"


def test_overlapping_register_handoff_alias_reports_interference_and_proof():
    domain = ExecutionDomain(
        "warp",
        instances_per_cta=2,
        members_per_instance=32,
    )
    ownership = OwnershipMap(
        domain,
        layout="lane4",
        index_map={"element": "lane_id"},
    )
    kernel = Kernel("handoff_alias")
    with kernel.launch(work_grid=(1,), threads=64) as launch:
        with launch.periodic("k", iterations=8) as loop:
            temporary = loop.buffer(
                "temporary",
                scope="register",
                shape=(4,),
                dtype="fp32",
                execution_scope="warp",
                alias_group="reuse",
                ownership=ownership,
            )
            with loop.actor("warp_actor", execution_domain=domain) as actor:
                producer = actor.phase(
                    "producer",
                    work=Work.opaque(),
                    timing=Timing(1 * ns),
                )
                temporary_write = actor.phase(
                    "temporary_write",
                    work=Work.opaque(),
                    timing=Timing(1 * ns),
                    writes=(temporary,),
                )
                temporary_read = actor.phase(
                    "temporary_read",
                    work=Work.opaque(),
                    timing=Timing(1 * ns),
                    reads=(temporary,),
                )
                consumer = actor.phase(
                    "consumer",
                    work=Work.opaque(),
                    timing=Timing(1 * ns),
                )
            loop.lifetime(
                temporary,
                acquire=temporary_write.start,
                release=temporary_read.done,
            )

    value = kernel.tensor_value(
        "value",
        shape=(4,),
        dtype="fp32",
        bytes=16,
        layout="lane4",
        ownership=ownership,
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
            "value",
            baseline,
            via="register",
            execution_scope="warp",
            alias_group="reuse",
        )

    report = analyze_liveness(
        kernel.build(),
        witness=_witness(
            10,
            producer=0,
            temporary_write=2,
            temporary_read=6,
            consumer=8,
        ),
    )

    peak = _peak(report, "register", "warp")
    assert peak.selected_bytes == 16
    assert peak.upper_bytes == 32
    assert len(report.alias_interference) == 1
    interference = report.alias_interference[0]
    assert interference.alias_group == "reuse"
    assert len(interference.members) == 2
    assert report.ownership[0].status == "proven_compatible"


def test_cross_actor_register_handoff_is_proven_only_with_shared_domain_map():
    domain = ExecutionDomain("warp", instances_per_cta=1, members_per_instance=32)
    ownership = OwnershipMap(
        domain,
        layout="lane_scalar",
        index_map={"element": "lane_id"},
    )
    kernel = Kernel("cross_actor_proof")
    with kernel.launch(work_grid=(1,), threads=32) as launch:
        with launch.periodic("k", iterations=8) as loop:
            with loop.actor("producer_actor", execution_domain=domain) as actor:
                producer = actor.phase(
                    "producer", work=Work.opaque(), timing=Timing(1 * ns)
                )
            with loop.actor("consumer_actor", execution_domain=domain) as actor:
                consumer = actor.phase(
                    "consumer", work=Work.opaque(), timing=Timing(1 * ns)
                )
    value = kernel.tensor_value(
        "value",
        shape=(1,),
        dtype="fp32",
        bytes=4,
        layout="lane_scalar",
        ownership=ownership,
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
            "value",
            baseline,
            via="register",
            execution_scope="warp",
        )

    report = analyze_liveness(
        kernel.build(),
        witness=_witness(10, producer=0, consumer=5),
    )
    assert report.ownership[0].status == "proven_compatible"


def test_cta_group_replicas_divide_residency_by_cluster_size_report_only():
    domain = ExecutionDomain(
        "cta_group",
        instances_per_cta=1,
        members_per_instance=2,
    )
    ownership = OwnershipMap(
        domain,
        layout="cluster_linear",
        index_map={"owner": "cluster_id"},
    )
    kernel = Kernel("cta_group")
    with kernel.launch(
        work_grid=(2,),
        threads=128,
        cluster=(2, 1, 1),
    ) as launch:
        with launch.periodic("k", iterations=8) as loop:
            grouped = loop.buffer(
                "grouped",
                scope="tmem",
                shape=(64,),
                dtype="fp32",
                execution_scope="cta_group",
                ownership=ownership,
            )
            with loop.actor("group_actor", execution_domain=domain) as actor:
                produce = actor.phase(
                    "produce",
                    work=Work.opaque(),
                    timing=Timing(1 * ns),
                    writes=(grouped,),
                )
                consume = actor.phase(
                    "consume",
                    work=Work.opaque(),
                    timing=Timing(1 * ns),
                    reads=(grouped,),
                )
            loop.lifetime(grouped, acquire=produce.start, release=consume.done)

    arch = SimpleNamespace(tmem_capacity_per_sm=400)
    report = analyze_liveness(
        kernel.build(),
        arch=arch,
        resident_ctas_per_sm=4,
        witness=_witness(10, produce=0, consume=5),
    )

    capacity = _capacity(report, "tmem", "cta_group")
    assert capacity.cluster_size == 2
    assert capacity.replicas_per_sm == 2
    assert capacity.per_replica_peak_bytes == 256
    assert capacity.aggregate_peak_bytes == 512
    assert capacity.exceeds_capacity is True
    assert report.status is FeasibilityStatus.UNKNOWN
    assert report.guard_eligible is False


def test_legacy_execution_scope_is_one_low_confidence_instance_not_inferred_width():
    kernel = Kernel("legacy_scope")
    with kernel.launch(work_grid=(1,), threads=128) as launch:
        with launch.periodic("k", iterations=8) as loop:
            storage = loop.buffer(
                "storage",
                scope="register",
                shape=(1,),
                dtype="fp32",
                execution_scope="warp",
            )
            with loop.actor("actor", execution_scope="warp") as actor:
                produce = actor.phase(
                    "produce",
                    work=Work.opaque(),
                    timing=Timing(1 * ns),
                    writes=(storage,),
                )
                consume = actor.phase(
                    "consume",
                    work=Work.opaque(),
                    timing=Timing(1 * ns),
                    reads=(storage,),
                )
            loop.lifetime(storage, acquire=produce.start, release=consume.done)

    report = analyze_liveness(
        kernel.build(),
        resident_ctas_per_sm=2,
        witness=_witness(10, produce=0, consume=5),
    )
    capacity = _capacity(report, "register", "warp")
    assert capacity.instances_per_cta == 1
    assert capacity.replicas_per_sm == 2
    assert capacity.ownership_provenance == "legacy_unspecified"


def test_strict_gemm_template_rejects_new_ownership_annotation():
    ir = build_gemm()
    launch = ir.launches[0]
    loop = launch.periodic_loops[0]
    a_shared = next(buffer for buffer in loop.buffers if buffer.name == "A_shared")
    ownership = OwnershipMap(
        ExecutionDomain("cta"),
        layout="tile",
        index_map={"owner": "blockIdx"},
    )
    modified = replace(
        ir,
        launches=(
            replace(
                launch,
                periodic_loops=(
                    replace(
                        loop,
                        buffers=tuple(
                            replace(a_shared, ownership=ownership)
                            if buffer is a_shared
                            else buffer
                            for buffer in loop.buffers
                        ),
                    ),
                ),
            ),
        ),
    )

    with pytest.raises(ModelingValidationError, match="canonical parity template"):
        extract_gemm_spec(modified)
