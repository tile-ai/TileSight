"""Step-1 contract: construction, validation, and derived views."""

import pytest

from tilesight.modeling import program as sight
from tilesight.modeling.ops.schema import GemmOpSpec
from tilesight.modeling.program import examples as ex


def _mma(op_id="mma", actor="tensor", **kw):
    return sight.Compute(op_id, GemmOpSpec(m=64, n=64, k=32, compute_dtype="fp16"), engine="tensor", actor=actor, **kw)


def _pipeline():
    load = sight.Load("load", "global", "smem", "tma", bytes=4096, actor="producer", writes=("buf",))
    mma = _mma(reads=("buf",))
    return sight.Pipeline(
        actors=(sight.Actor("producer", (load,)), sight.Actor("tensor", (mma,))),
        buffers=(sight.Buffer("buf", "smem", (64, 32), "fp16", slots=2),),
        buffer_slots=(sight.BufferSlots("buf", sight.start(load), sight.done(mma), 2),),
    )


def test_examples_construct_and_expose_paths():
    program = ex.gemm_program()
    assert program.op_paths() == ("main/k/load_A", "main/k/load_B", "main/k/mma", "main/k/cast_C", "main/k/store_C")
    assert program.region_paths() == ("main", "main/k")
    sites = {site.path: site for site in program.op_sites()}
    assert sites["main/k/store_C"].region_site == "k:epilogue"
    assert sites["main/k/mma"].region_site == "k:body"
    assert sites["main/k/mma"].in_pipeline


def test_pipeline_only_as_loop_body():
    with pytest.raises(sight.ContractError, match="direct body of a Loop"):
        sight.Sequence((_pipeline(),))
    with pytest.raises(sight.ContractError, match="direct body of a Loop"):
        sight.Loop("k", 4, body=_mma(), prologue=_pipeline())
    with pytest.raises(sight.ContractError, match="wrap a pipeline"):
        sight.Launch("main", _pipeline(), (sight.WorkAxis("m", 2),))
    loop = sight.Loop("k", 1, body=_pipeline())
    assert loop.is_periodic


def test_op_ids_and_loop_ids_unique_within_launch():
    body = sight.Sequence((_mma("a"), _mma("a")))
    with pytest.raises(sight.ContractError, match="op ids must be unique"):
        sight.Launch("main", body, (sight.WorkAxis("m", 1),))
    nested = sight.Loop("k", 2, body=sight.Loop("k", 2, body=_mma()))
    with pytest.raises(sight.ContractError, match="loop ids must be unique"):
        sight.Launch("main", nested, (sight.WorkAxis("m", 1),))


def test_from_work_group_binding_rules():
    loop = sight.Loop("kv", sight.FromWorkGroup(), body=_mma())
    with pytest.raises(sight.ContractError, match="FromWorkGroup"):
        sight.Launch("main", loop, (sight.WorkAxis("q", 2),))
    units = sight.WorkUnits(
        groups=(("short", sight.WorkGroup({"kv": 4})), ("long", sight.WorkGroup({"kv": 8}))),
        order=(sight.WorkRun("short", 3), sight.WorkRun("long", 2)),
    )
    launch = sight.Launch("main", loop, (sight.WorkAxis("q", 5),), work_units=units)
    assert launch.bound_trip_count(loop, units.group("short")) == 4
    assert launch.bound_trip_count(loop, units.group("long")) == 8
    assert units.group_counts() == {"short": 3, "long": 2}
    assert units.instance_group_ids() == ["short"] * 3 + ["long"] * 2
    fixed = sight.Loop("kv", 6, body=_mma())
    with pytest.raises(sight.ContractError, match="conflicts with work-group override"):
        sight.Launch("main", fixed, (sight.WorkAxis("q", 5),), work_units=units)


def test_work_unit_order_must_cover_work_axes():
    units = sight.WorkUnits(groups=(("g", sight.WorkGroup()),), order=(sight.WorkRun("g", 3),))
    with pytest.raises(sight.ContractError, match="must sum to the work-axis product"):
        sight.Launch("main", _mma(), (sight.WorkAxis("m", 2), sight.WorkAxis("n", 2)), work_units=units)
    launch = sight.Launch("main", _mma(), (sight.WorkAxis("m", 2), sight.WorkAxis("n", 2)))
    default = launch.effective_work_units()
    assert default.total_count == 4 and default.groups[0][0] == "default"


def test_persistent_partitions_cover_exactly_once():
    with pytest.raises(sight.ContractError, match="covered exactly once"):
        sight.Launch("main", _mma(), (sight.WorkAxis("m", 4),),
                  scheduler=sight.Persistent((((0, 2),), ((1, 4),))))
    launch = sight.Launch("main", _mma(), (sight.WorkAxis("m", 4),),
                       scheduler=sight.Persistent((((0, 2),), ((2, 4),))))
    assert launch.scheduler.cta_count == 2


def test_tile_access_payload_and_bounds():
    access = sight.TileAccess("A", sight.Projection(("m",)), (64, 32), 2, valid_shape=(60, 32), tensor_shape=(60, 32))
    assert access.payload_bytes == 60 * 32 * 2
    assert access.executed_bytes == 64 * 32 * 2
    with pytest.raises(sight.ContractError, match="exceeds tensor_shape"):
        sight.TileAccess("A", sight.Projection(("m",)), (64, 32), 2, tensor_shape=(60, 32))
    with pytest.raises(sight.ContractError, match="fit inside tile_shape"):
        sight.TileAccess("A", sight.Projection(("m",)), (64, 32), 2, valid_shape=(65, 32))
    with pytest.raises(sight.ContractError, match="cannot be smaller than the logical payload"):
        sight.TileAccess("A", sight.Projection(("m",)), (64, 32), 2, transaction_bytes=10)


def test_access_axes_must_name_work_axes_or_enclosing_loops():
    bad = sight.Load("l", "global", "smem", access=sight.TileAccess("A", sight.Projection(("zz",)), (8,), 2))
    with pytest.raises(sight.ContractError, match="must name a launch work axis"):
        sight.Launch("main", bad, (sight.WorkAxis("m", 1),))
    ok = sight.Load("l", "global", "smem", access=sight.TileAccess("A", sight.Projection(("m", "k")), (8,), 2))
    sight.Launch("main", sight.Loop("k", 2, body=ok), (sight.WorkAxis("m", 1),))
    with pytest.raises(sight.ContractError, match="must name a launch work axis"):
        sight.Launch("main", ok, (sight.WorkAxis("m", 1),))


def test_same_tensor_needs_one_allocation_size():
    a = sight.Load("a", "global", "smem", access=sight.TileAccess("X", sight.Projection(("m",)), (8,), 2))
    b = sight.Load("b", "global", "smem", access=sight.TileAccess("X", sight.Projection(("m",)), (16,), 2))
    with pytest.raises(sight.ContractError, match="already declared allocation"):
        sight.Launch("main", sight.Sequence((a, b)), (sight.WorkAxis("m", 1),))


def test_move_validation():
    with pytest.raises(sight.ContractError, match="needs a TileAccess or explicit bytes"):
        sight.Load("l", "global", "smem")
    with pytest.raises(sight.ContractError, match="a load cannot target global"):
        sight.Load("l", "smem", "global", bytes=1)
    with pytest.raises(sight.ContractError, match="must be one of"):
        sight.Load("l", "global", "smem", engine="dma", bytes=1)
    with pytest.raises(sight.ContractError, match="must differ from source"):
        sight.Store("s", "smem", "smem", bytes=1)


def test_pipeline_references_are_checked():
    load = sight.Load("load", "global", "smem", bytes=1, actor="p", writes=("missing",))
    with pytest.raises(sight.ContractError, match="undeclared buffer"):
        sight.Pipeline(actors=(sight.Actor("p", (load,)),))
    load = sight.Load("load", "global", "smem", bytes=1, actor="p")
    with pytest.raises(sight.ContractError, match="references unknown op"):
        sight.Pipeline(actors=(sight.Actor("p", (load,)),), dependencies=(sight.Dependency(sight.done("x"), sight.start("load")),))
    with pytest.raises(sight.ContractError, match="op.actor"):
        sight.Actor("q", (load,))
    with pytest.raises(sight.ContractError, match="must target a start event"):
        sight.Dependency(sight.done("a"), sight.done("b"))


def test_program_edges_follow_launch_order():
    a = sight.Launch("a", _mma(), (sight.WorkAxis("m", 1),))
    b = sight.Launch("b", _mma(), (sight.WorkAxis("m", 1),))
    sight.Program((a, b), edges=(sight.LaunchEdge("a", "b", gap_s=1e-6),))
    with pytest.raises(sight.ContractError, match="declared sequential launch order"):
        sight.Program((a, b), edges=(sight.LaunchEdge("b", "a"),))
    with pytest.raises(sight.ContractError, match="unknown launch"):
        sight.Program((a,), edges=(sight.LaunchEdge("a", "c"),))


def test_compute_work_accounting():
    op = _mma(effective_fraction=0.5)
    assert op.executed_flops == 2 * 64 * 64 * 32
    assert op.useful_flops == op.executed_flops * 0.5
    with pytest.raises(sight.ContractError, match="effective_fraction"):
        _mma(effective_fraction=1.5)


def test_loop_boundary_anchor_needs_pipeline_op():
    with pytest.raises(sight.ContractError, match="only meaningful for a Pipeline body"):
        sight.Loop("k", 2, body=_mma(), boundary_anchor_op="mma")
    with pytest.raises(sight.ContractError, match="is not an op of the pipeline body"):
        sight.Loop("k", 2, body=_pipeline(), boundary_anchor_op="nope")
    sight.Loop("k", 2, body=_pipeline(), boundary_anchor_op="load")
