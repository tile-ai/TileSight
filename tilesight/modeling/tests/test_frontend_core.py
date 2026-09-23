"""Unit tests for frozen IR, builder semantics, and periodic lowering."""

from dataclasses import FrozenInstanceError

import pytest

from tilesight.modeling import (
    Kernel,
    ModelingValidationError,
    ResourceTiming,
    Timing,
    TimingResolutionError,
    Work,
    ns,
)


def _base_kernel():
    kernel = Kernel("pipeline", params={"nested": {"tile": [64, 64]}})
    launch = kernel.launch(
        work_grid=(8, 4, 1),
        physical_grid=("arch.sm_count", 1, 1),
        threads=128,
        cluster=(2, 1, 1),
        residency=2,
        scheduler="persistent",
        swizzle={"axis": "n", "panel": 4},
    )
    return kernel, launch.periodic("ko", iterations=16, stages=2)


def test_ir_is_frozen_hashable_and_summary_keeps_both_concurrency_scopes():
    kernel, loop = _base_kernel()
    buffer = loop.buffer(
        "tile",
        scope="tmem",
        shape=(64, 64),
        dtype="bf16",
        slots=2,
        execution_scope="warpgroup",
        alias_group="score_or_probability",
    )
    with loop.actor("tensor") as actor:
        phase = actor.phase(
            "mma",
            work=Work.mma(2 * 64**3, instruction="wgmma"),
            timing=Timing(12 * ns, (ResourceTiming("tensor", 8 * ns),)),
            writes=(buffer,),
        )
        actor.sequence(phase.at(+1), resource="tensor")

    ir = kernel.build()
    hash(ir)
    with pytest.raises(FrozenInstanceError):
        ir.name = "changed"

    launch_summary = ir.summary()["launches"][0]
    assert launch_summary["work_grid"] == (8, 4, 1)
    assert launch_summary["physical_grid"] == ("arch.sm_count", 1, 1)
    assert launch_summary["residency"] == 2
    loop_summary = launch_summary["periodic_loops"][0]
    assert loop_summary["actor_sequences"][0]["cyclic"] is True
    assert loop_summary["actor_sequences"][0]["order"] == "issue"
    assert loop_summary["actor_sequences"][0]["occurrences"] == (
        {"phase": "mma", "window_offset": 1},
    )
    occurrence = ir.periodic_loop("ko").actors[0].sequence[0]
    assert occurrence.window_offset == 1
    assert occurrence.iteration_offset == 1
    assert not hasattr(occurrence, "start")
    assert not hasattr(occurrence, "done")
    assert loop_summary["buffer_specs"][0]["alias_group"] == "score_or_probability"
    assert loop_summary["phase_specs"][0]["work"]["kind"] == "mma"
    assert loop_summary["resource_sequences"][0]["resource"] == "tensor"
    hash(Timing(2 * ns, [ResourceTiming("cuda", 1 * ns)]))


def test_issue_sequence_and_resource_order_are_explicit_but_can_be_declared_together():
    kernel, loop = _base_kernel()
    with loop.actor("tensor") as actor:
        pv = actor.phase(
            "pv",
            work=Work.mma(1),
            timing=Timing(20 * ns, (ResourceTiming("tensor", 5 * ns),)),
        )
        qk = actor.phase(
            "qk",
            work=Work.mma(1),
            timing=Timing(30 * ns, (ResourceTiming("tensor", 7 * ns),)),
        )
        actor.sequence(pv.at(0), qk.at(+1), resource="tensor")

    dag = kernel.lower_periodic("ko")
    assert {phase.name: phase.iteration_offset for phase in dag.phases} == {
        "pv": 0,
        "qk": 1,
    }
    assert dag.fixed_resource_orders[0].phases == ("pv", "qk")
    edges = [edge for edge in dag.dependencies if edge.name.startswith("actor:")]
    assert [edge.iteration_distance for edge in edges] == [1, 0]
    assert [edge.min_delay for edge in edges] == [0.0, 0.0]


def test_completion_order_is_explicit_and_does_not_fix_resource_arbitration():
    kernel, loop = _base_kernel()
    with loop.actor("control") as actor:
        first = actor.phase("first", work=Work.pointwise(1), timing=Timing(3 * ns))
        second = actor.phase("second", work=Work.pointwise(1), timing=Timing(5 * ns))
        actor.sequence(first.at(0), second.at(0), order="completion")

    dag = kernel.lower_periodic("ko")
    edges = [edge for edge in dag.dependencies if edge.name.startswith("actor:")]
    assert [(edge.iteration_distance, edge.min_delay) for edge in edges] == [
        (0, 3 * ns),
        (1, 5 * ns),
    ]
    assert dag.fixed_resource_orders == tuple()


def test_default_issue_order_without_resource_leaves_arbitration_searchable():
    kernel, loop = _base_kernel()
    with loop.actor("tensor") as actor:
        first = actor.phase(
            "first",
            work=Work.mma(1),
            timing=Timing(5 * ns, (ResourceTiming("tensor", 2 * ns),)),
        )
        second = actor.phase(
            "second",
            work=Work.mma(1),
            timing=Timing(5 * ns, (ResourceTiming("tensor", 2 * ns),)),
        )
        actor.sequence(first.at(0), second.at(0))

    ir = kernel.build()
    dag = ir.lower_periodic("ko")
    assert ir.periodic_loop("ko").actors[0].order == "issue"
    assert dag.fixed_resource_orders == tuple()
    edges = [edge for edge in dag.dependencies if edge.name.startswith("actor:")]
    assert [edge.min_delay for edge in edges] == [0.0, 0.0]


def test_buffer_flow_pipeline_credit_and_state_carry_lower_semantically():
    kernel, loop = _base_kernel()
    buffer = loop.buffer("smem", scope="smem", slots=2)
    accumulator_buffer = loop.buffer(
        "accumulator", scope="register", execution_scope="thread"
    )
    with loop.actor("producer") as producer:
        load = producer.phase(
            "load",
            work=Work.copy(128),
            timing=Timing(10 * ns, (ResourceTiming("tma", 2 * ns),)),
            writes=(buffer,),
        )
        producer.sequence(load.at(0), resource="tma")
    with loop.actor("compute") as compute:
        mma = compute.phase(
            "mma", work=Work.mma(256), timing=Timing(4 * ns), reads=(buffer,)
        )
    loop.pipeline_buffer(buffer, acquire=load.start, release=mma.done)
    accumulator = loop.state("accumulator", storage=accumulator_buffer)
    loop.carry(accumulator, source=mma.done, target=mma.start, distance=1)

    dag = kernel.lower_periodic("ko")
    dependencies = {
        (edge.source, edge.target, edge.iteration_distance, edge.min_delay)
        for edge in dag.dependencies
    }
    assert ("load", "mma", 0, 10 * ns) in dependencies
    assert ("mma", "mma", 1, 4 * ns) in dependencies
    assert dag.token_buffers[0].capacity == 2
    assert dag.token_buffers[0].release_offset == 4 * ns
    assert len(kernel.build().periodic_loop("ko").lifetimes) == 1
    state = kernel.build().periodic_loop("ko").states[0]
    assert state.storage == accumulator_buffer
    assert kernel.summary()["launches"][0]["periodic_loops"][0]["states"] == (
        {"name": "accumulator", "storage": "accumulator"},
    )


def test_state_storage_must_belong_to_the_same_periodic_loop():
    kernel, loop = _base_kernel()
    other_kernel = Kernel("other")
    other_launch = other_kernel.launch(work_grid=(1,), threads=32)
    other_loop = other_launch.periodic("other_loop")
    foreign = other_loop.buffer("foreign", scope="register")
    with other_loop.actor("foreign_actor") as foreign_actor:
        foreign_phase = foreign_actor.phase(
            "foreign_phase",
            work=Work.pointwise(1),
            timing=Timing(1 * ns, (ResourceTiming("cuda", 1 * ns),)),
        )

    with pytest.raises(ModelingValidationError, match="another periodic loop"):
        loop.state("state", storage=foreign)
    with pytest.raises(ModelingValidationError, match="another periodic loop"):
        loop.resource_sequence("cuda", foreign_phase.at(0))


def test_state_storage_binding_does_not_relax_single_writer_buffer_flow():
    kernel, loop = _base_kernel()
    storage = loop.buffer("state_storage", scope="register")
    loop.state("state", storage=storage)
    with loop.actor("writers") as actor:
        actor.phase(
            "writer_a",
            work=Work.pointwise(1),
            timing=Timing(1 * ns),
            writes=(storage,),
        )
        actor.phase(
            "writer_b",
            work=Work.pointwise(1),
            timing=Timing(1 * ns),
            writes=(storage,),
        )

    with pytest.raises(ModelingValidationError, match="multiple writers"):
        kernel.lower_periodic("ko")


def test_fixed_resource_sequence_requires_complete_coverage():
    kernel, loop = _base_kernel()
    with loop.actor("tensor") as actor:
        first = actor.phase(
            "first",
            work=Work.mma(1),
            timing=Timing(1 * ns, (ResourceTiming("tensor", 1 * ns),)),
        )
        actor.phase(
            "omitted",
            work=Work.mma(1),
            timing=Timing(1 * ns, (ResourceTiming("tensor", 1 * ns),)),
        )
        actor.sequence(first.at(0), resource="tensor")

    with pytest.raises(ModelingValidationError, match="does not cover every tensor user"):
        kernel.lower_periodic("ko")


def test_serial_sequence_rejects_occurrence_without_declared_resource_use():
    kernel, loop = _base_kernel()
    with loop.actor("mixed") as actor:
        cuda = actor.phase(
            "cuda",
            work=Work.pointwise(1),
            timing=Timing(1 * ns, (ResourceTiming("cuda", 1 * ns),)),
        )
        sfu = actor.phase(
            "sfu",
            work=Work.pointwise(1),
            timing=Timing(1 * ns, (ResourceTiming("sfu", 1 * ns),)),
        )
        actor.sequence(cuda.at(0), sfu.at(0), resource="cuda")

    with pytest.raises(ModelingValidationError, match="resource sequence cuda must use"):
        kernel.lower_periodic("ko")


def test_expert_resource_sequence_can_span_actors_without_changing_actor_order():
    kernel, loop = _base_kernel()
    with loop.actor("producer") as producer:
        first = producer.phase(
            "first",
            work=Work.pointwise(1),
            timing=Timing(2 * ns, (ResourceTiming("cuda", 1 * ns),)),
        )
        producer.sequence(first.at(0))
    with loop.actor("consumer") as consumer:
        second = consumer.phase(
            "second",
            work=Work.pointwise(1),
            timing=Timing(2 * ns, (ResourceTiming("cuda", 1 * ns),)),
        )
        consumer.sequence(second.at(0))
    loop.resource_sequence("cuda", first.at(0), second.at(0))
    with pytest.raises(ModelingValidationError, match="already has"):
        loop.resource_sequence("cuda", second.at(0), first.at(0))

    ir = kernel.build()
    dag = ir.lower_periodic("ko")
    assert dag.fixed_resource_orders[0].phases == ("first", "second")
    assert all(actor.order == "issue" for actor in ir.periodic_loop("ko").actors)
    assert all(edge.min_delay == 0.0 for edge in dag.dependencies)


def test_serial_resource_compatibility_alias_is_exact_or_rejected():
    kernel, loop = _base_kernel()
    with loop.actor("compat", serial_resource="cuda") as actor:
        phase = actor.phase(
            "phase",
            work=Work.pointwise(1),
            timing=Timing(2 * ns, (ResourceTiming("cuda", 1 * ns),)),
        )
        actor.sequence(phase.at(0), order="completion")
    ir = kernel.build()
    resource_sequence = ir.periodic_loop("ko").resource_sequences[0]
    assert resource_sequence.resource == "cuda"
    assert resource_sequence.sequence == ir.periodic_loop("ko").actors[0].sequence
    assert ir.periodic_loop("ko").actors[0].order == "completion"

    other_kernel, other_loop = _base_kernel()
    with other_loop.actor("conflict", serial_resource="cuda") as conflict:
        other_phase = conflict.phase(
            "other_phase",
            work=Work.pointwise(1),
            timing=Timing(2 * ns, (ResourceTiming("cuda", 1 * ns),)),
        )
        with pytest.raises(ModelingValidationError, match="conflicts"):
            conflict.sequence(other_phase.at(0), resource="sfu")

    missing_kernel, missing_loop = _base_kernel()
    with missing_loop.actor("missing", serial_resource="cuda") as missing:
        missing.phase(
            "missing_phase",
            work=Work.pointwise(1),
            timing=Timing(2 * ns, (ResourceTiming("cuda", 1 * ns),)),
        )
    with pytest.raises(ModelingValidationError, match="requires actor.sequence"):
        missing_kernel.build()


def test_conflicting_actor_and_resource_window_offsets_fail():
    kernel, loop = _base_kernel()
    with loop.actor("actor") as actor:
        phase = actor.phase(
            "phase",
            work=Work.pointwise(1),
            timing=Timing(2 * ns, (ResourceTiming("cuda", 1 * ns),)),
        )
        actor.sequence(phase.at(0))
    loop.resource_sequence("cuda", phase.at(+1))

    with pytest.raises(ModelingValidationError, match="conflicting window offsets"):
        kernel.lower_periodic("ko")


def test_frontend_lag_is_relative_to_selected_source_event():
    kernel, loop = _base_kernel()
    with loop.actor("actor") as actor:
        source = actor.phase(
            "source", work=Work.pointwise(1), timing=Timing(10 * ns)
        )
        from_start = actor.phase(
            "from_start", work=Work.pointwise(1), timing=Timing(1 * ns)
        )
        from_done = actor.phase(
            "from_done", work=Work.pointwise(1), timing=Timing(1 * ns)
        )
    loop.after(
        source.start,
        from_start.start,
        lag=3 * ns,
        name="after_start",
    )
    loop.after(
        source.done,
        from_done.start,
        lag=3 * ns,
        name="after_done",
    )
    state_from_start = loop.state("state_from_start")
    loop.carry(
        state_from_start,
        source=source.start,
        target=from_start.start,
        distance=1,
        lag=2 * ns,
    )
    state_from_done = loop.state("state_from_done")
    loop.carry(
        state_from_done,
        source=source.done,
        target=from_done.start,
        distance=1,
        lag=2 * ns,
    )

    edges = {edge.name: edge for edge in kernel.lower_periodic("ko").dependencies}
    assert edges["after_start"].min_delay == pytest.approx(3 * ns)
    assert edges["after_done"].min_delay == pytest.approx(13 * ns)
    assert edges["state:state_from_start"].min_delay == pytest.approx(2 * ns)
    assert edges["state:state_from_done"].min_delay == pytest.approx(12 * ns)


def test_capacity_carry_timing_and_identifier_invariants_fail_early():
    with pytest.raises(ModelingValidationError, match="must be a string"):
        Kernel(7)
    with pytest.raises(ModelingValidationError, match="must not contain"):
        Kernel("bad/name")
    with pytest.raises(ModelingValidationError, match="attribute key must be a string"):
        Work("custom", attrs=((7, "value"),))
    valid = Kernel("valid")
    with pytest.raises(ModelingValidationError, match="must not contain"):
        valid.launch(work_grid=(1,), threads=32, name="bad/launch")
    valid_launch = valid.launch(work_grid=(1,), threads=32)
    with pytest.raises(ModelingValidationError, match="must not contain"):
        valid_launch.periodic("bad/loop")
    with pytest.raises(ModelingValidationError, match="ends after phase latency"):
        Timing(
            5 * ns,
            (ResourceTiming("tensor", service_time=4 * ns, offset=2 * ns),),
        )

    kernel, loop = _base_kernel()
    buffer = loop.buffer("ring", scope="smem", slots=2)
    with loop.actor("actor") as actor:
        first = actor.phase("first", work=Work.copy(1), timing=Timing(1 * ns))
        last = actor.phase("last", work=Work.pointwise(1), timing=Timing(1 * ns))
    with pytest.raises(ModelingValidationError, match="must not exceed buffer.slots"):
        loop.pipeline_buffer(
            buffer,
            acquire=first.start,
            release=last.done,
            capacity=3,
        )
    state = loop.state("state", storage=buffer)
    with pytest.raises(ModelingValidationError, match="carry distance must be positive"):
        loop.carry(state, source=last.done, target=first.start, distance=0)


def test_ordinary_lifetime_and_stage_metadata_do_not_create_pipeline_credit():
    kernel, loop = _base_kernel()
    buffer = loop.buffer("ordinary", scope="smem", slots=2)
    with loop.actor("actor") as actor:
        first = actor.phase("first", work=Work.copy(1), timing=Timing(1 * ns))
        last = actor.phase("last", work=Work.pointwise(1), timing=Timing(1 * ns))
    loop.lifetime(buffer, acquire=first.start, release=last.done)

    ir = kernel.build()
    assert ir.periodic_loop("ko").stages == 2
    assert ir.lower_periodic("ko").token_buffers == tuple()


def test_cost_oracle_fills_only_missing_static_timing():
    kernel, loop = _base_kernel()
    with loop.actor("compute") as actor:
        actor.phase("unknown", work=Work.opaque("custom"))

    with pytest.raises(TimingResolutionError, match="no static Timing"):
        kernel.lower_periodic("ko")

    class Oracle:
        def resolve(self, phase):
            return Timing(9 * ns)

    assert kernel.lower_periodic("ko", oracle=Oracle()).phases[0].latency == pytest.approx(
        9 * ns
    )


def test_compute_vector_alias_canonicalizes_to_cuda_with_provenance():
    kernel = Kernel("vector_alias")
    with kernel.launch(work_grid=(1,), threads=32) as launch:
        with launch.periodic("loop", iterations=1) as loop:
            with loop.actor("vector_actor") as actor:
                phase = actor.compute(
                    "add",
                    flops=32,
                    engine="vector",
                    op="add",
                    timing=Timing(
                        1 * ns,
                        (ResourceTiming("cuda", 1 * ns),),
                    ),
                )
    attrs = dict(phase.work.attrs)
    assert attrs["engine"] == "cuda"
    assert attrs["requested_engine"] == "vector"
