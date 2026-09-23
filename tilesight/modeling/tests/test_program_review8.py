"""Review round 8: CUTLASS mapping cross-check, one launch order for both cache paths,
mapper validation/snapshots, per-role dtype identity, tail-wave pricing and attention pipeline constraints."""

import copy

import pytest

from tilesight.modeling import program as sight
from tilesight.modeling.program import examples as ex
from tilesight.modeling.program import validation as v

from .test_program_analyze import ARCH


# ---------------------------------------------------------------------------
# CUTLASS SM90 block id -> tile coordinate (transliteration of the C++ sources)
# ---------------------------------------------------------------------------


class _Params:
    """PersistentTileSchedulerSm90Params::initialize (tile_scheduler_params.h)."""

    def __init__(self, tiles_m, tiles_n, cluster_m, cluster_n, max_swizzle, raster_option):
        smallest = min(tiles_m, tiles_n)
        if max_swizzle >= 8 and smallest >= 6:
            self.log_swizzle = 3
        elif max_swizzle >= 4 and smallest >= 3:
            self.log_swizzle = 2
        elif max_swizzle >= 2 and smallest >= 2:
            self.log_swizzle = 1
        else:
            self.log_swizzle = 0
        round_up = lambda value, multiple: (value + multiple - 1) // multiple * multiple
        self.blocks_m = round_up(tiles_m, (1 << self.log_swizzle) * cluster_m)
        self.blocks_n = round_up(tiles_n, (1 << self.log_swizzle) * cluster_n)
        if raster_option == "heuristic":
            self.along_n = not (self.blocks_n > self.blocks_m)
        else:
            self.along_n = raster_option == "along_n"
        if self.along_n:
            self.shape_major, self.shape_minor = cluster_n, cluster_m
            self.blk_major = self.blocks_n // cluster_n
        else:
            self.shape_major, self.shape_minor = cluster_m, cluster_n
            self.blk_major = self.blocks_m // cluster_m


def _cpp_work_idx(params, linear_idx):
    """StaticPersistentTileScheduler::get_current_work_for_linear_idx + Sm90::get_work_idx_m_and_n."""

    _work_l, remainder = divmod(linear_idx, params.blocks_m * params.blocks_n)
    blk_per_grid_dim = remainder // params.shape_minor
    cluster_minor_offset = remainder % params.shape_minor  # blockIdx along the cluster's minor axis
    cluster_id, cluster_major_offset = divmod(blk_per_grid_dim, params.shape_major)
    offset = cluster_id & ((1 << params.log_swizzle) - 1)
    extra = cluster_id >> params.log_swizzle
    cluster_idx_minor_div_swizzle, cluster_idx_major = divmod(extra, params.blk_major)
    cluster_idx_minor = cluster_idx_minor_div_swizzle * (1 << params.log_swizzle) + offset
    minor_work_idx = cluster_idx_minor * params.shape_minor + cluster_minor_offset
    major_work_idx = cluster_idx_major * params.shape_major + cluster_major_offset
    return (minor_work_idx, major_work_idx) if params.along_n else (major_work_idx, minor_work_idx)


def _cpp_linear_idx(params, tile_m, tile_n):
    """StaticPersistentTileScheduler::get_linear_idx_from_m_and_n (the C++ inverse mapping)."""

    if params.along_n:
        minor_work_idx, major_work_idx = tile_m, tile_n
    else:
        major_work_idx, minor_work_idx = tile_m, tile_n
    cluster_minor_offset = minor_work_idx - (minor_work_idx // params.shape_minor) * params.shape_minor
    cluster_idx_minor = (minor_work_idx - cluster_minor_offset) // params.shape_minor
    cluster_idx_major, cluster_major_offset = divmod(major_work_idx, params.shape_major)
    cluster_idx_minor_div_swizzle = cluster_idx_minor >> params.log_swizzle
    offset = cluster_idx_minor & ((1 << params.log_swizzle) - 1)
    extra = cluster_idx_minor_div_swizzle * params.blk_major + cluster_idx_major
    cluster_id = (extra << params.log_swizzle) | offset
    return (cluster_id * params.shape_major + cluster_major_offset) * params.shape_minor + cluster_minor_offset


@pytest.mark.parametrize("cluster", [(1, 1), (2, 1), (1, 2), (2, 2)])
@pytest.mark.parametrize("max_swizzle", [1, 2, 4, 8])
@pytest.mark.parametrize("raster", ["along_m", "along_n", "heuristic"])
@pytest.mark.parametrize("grid", [(8, 8), (5, 7), (1, 9), (3, 3), (2, 13), (16, 6)])
def test_cutlass_sm90_mapper_matches_the_cpp_mapping_and_its_inverse(cluster, max_swizzle, raster, grid):
    mapper = sight.CutlassSm90(cluster_m=cluster[0], cluster_n=cluster[1], max_swizzle_size=max_swizzle, raster_order=raster)
    params = _Params(grid[0], grid[1], cluster[0], cluster[1], max_swizzle, raster)
    info = mapper.resolve(grid)
    assert (info["log_swizzle_size"], info["padded_m"], info["padded_n"]) == (params.log_swizzle, params.blocks_m, params.blocks_n)
    assert info["raster_order"] == ("along_n" if params.along_n else "along_m")
    expected = []
    for linear_idx in range(params.blocks_m * params.blocks_n):
        m, n = _cpp_work_idx(params, linear_idx)
        assert mapper.work_index(linear_idx, grid) == (0, m, n)
        assert _cpp_linear_idx(params, m, n) == linear_idx        # CUTLASS' own inverse agrees
        if m < grid[0] and n < grid[1]:                           # padding blocks carry no work
            expected.append((m, n))
    assert mapper.coordinates(grid) == expected
    assert sorted(expected) == sorted(sight.LinearBlockId().coordinates(grid))
    assert info["padding_blocks"] == params.blocks_m * params.blocks_n - grid[0] * grid[1]


def test_cutlass_sm90_reviewer_example_and_batch_axis():
    mapper = sight.CutlassSm90(cluster_m=2, cluster_n=1, max_swizzle_size=4, raster_order="along_m")
    assert mapper.coordinates((8, 8))[:10] == [
        (0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 0), (3, 0)]
    # the batch (l) axis is outermost, as divmod_batch_ makes it
    batched = mapper.coordinates((2, 4, 4))
    assert batched[:16] == [(0,) + c for c in mapper.coordinates((4, 4))]
    assert batched[16:] == [(1,) + c for c in mapper.coordinates((4, 4))]
    with pytest.raises(sight.ContractError, match="power of two"):
        sight.CutlassSm90(cluster_m=3)
    program = ex.gemm_program(m=512, n=512, k=256, mapper=mapper)
    assert sight.program_from_snapshot(sight.program_to_snapshot(program)) == program
    assert program.launches[0].instance_coordinates()[:4] == [(0, 0), (1, 0), (0, 1), (1, 1)]


# ---------------------------------------------------------------------------
# One launch order for fast and trace
# ---------------------------------------------------------------------------


def test_fast_and_trace_consume_the_same_mapper_order():
    single = copy.copy(ARCH)
    single.sm_count = 1
    shape = dict(m=1024, n=8192, k=8192, block_m=128, block_n=128, block_k=64)
    panel = sight.PanelSwizzle(row_panel=4, column_panel=2, raster_axis="along_m")
    same = sight.ExplicitOrder(tuple(panel.coordinates((8, 64))), "same_as_panel")

    def ddr(mapper, cache, arch=single):
        result = sight.analyze(ex.gemm_program(mapper=mapper, **shape), arch, options=sight.Options(cache=cache))
        return result.launches["main"].traffic_total.ddr_read_bytes, result.cache[0]

    panel_fast, report = ddr(panel, "fast")
    same_fast, _ = ddr(same, "fast")
    panel_trace, trace_report = ddr(panel, "trace")
    same_trace, _ = ddr(same, "trace")
    linear_fast, _ = ddr("linear_block_id", "fast")
    # the same order gives the same answer however the mapper is written down ...
    assert panel_fast == same_fast and panel_trace == same_trace
    # ... the order matters (so a path ignoring it would show) ...
    assert linear_fast > 4 * panel_fast
    # ... and the fast estimate stays close to the exact trace of that order.
    assert panel_fast == pytest.approx(panel_trace, rel=0.05)
    assert trace_report.exact and not report.exact       # fast = representative-iteration estimate
    assert any("waves of 1 concurrent work units" in a for a in report.assumptions)
    # waves are `slots` wide, never the panel size
    wide = sight.analyze(ex.gemm_program(mapper=panel, **shape), ARCH, options=sight.Options(cache="fast")).cache[0]
    assert any("waves of %d concurrent work units" % min(ARCH.sm_count, 8 * 64) in a for a in wide.assumptions)
    assert any("seeded random order" in a for a in wide.assumptions) and not wide.exact


# ---------------------------------------------------------------------------
# Mapper validation and snapshots
# ---------------------------------------------------------------------------


class _Bad(sight.BlockMapper):
    kind = "bad"

    def __init__(self, order):
        self._order = order

    def coordinates(self, extents):
        return list(self._order)

    def canonical(self):
        return {"kind": self.kind}


class _Reverse(sight.BlockMapper):
    kind = "reverse"

    def coordinates(self, extents):
        return list(reversed(sight.LinearBlockId().coordinates(extents)))

    def canonical(self):
        return {"kind": self.kind}


@pytest.mark.parametrize("order, message", [
    ([(0,), (42,)], "outside"),
    ([(0,), (0,)], "exactly once"),
    ([(0,)], "exactly once"),
    ([(0, 0), (1, 0)], "rank-1"),
    ([(0,), (1.0,)], "non-integer"),
    ([(0,), (True,)], "non-integer"),
    ([0, 1], "rank-1"),
])
def test_block_mapper_output_is_validated(order, message):
    program = ex.elementwise_add_program()
    launch = program.launches[0]
    axis = sight.WorkAxis(launch.work_axes[0].name, 2)
    with pytest.raises(sight.ContractError, match=message):
        sight.Launch(launch.name, launch.body, (axis,), dispatch_order=_Bad(order))


def test_custom_mapper_survives_a_snapshot():
    program = ex.gemm_program(m=512, n=512, k=256, mapper=_Reverse())
    launch = program.launches[0]
    # an unregistered custom mapper is frozen into the order it produced
    assert launch.mapper.kind == "explicit_order" and launch.mapper.label == "reverse"
    assert launch.instance_coordinates()[0] == (3, 3)
    restored = sight.program_from_snapshot(sight.program_to_snapshot(program))
    assert restored == program and restored.launches[0].instance_coordinates() == launch.instance_coordinates()
    sight.analyze(restored, ARCH, options=sight.Options(cache="fast"))
    # a registered mapper keeps its own kind and is rebuilt by its factory
    sight.register_block_mapper("reverse", lambda data: _Reverse())
    try:
        registered = ex.gemm_program(m=512, n=512, k=256, mapper=_Reverse())
        assert registered.launches[0].mapper.kind == "reverse"
        again = sight.program_from_snapshot(sight.program_to_snapshot(registered))
        assert again.launches[0].instance_coordinates() == launch.instance_coordinates()
    finally:
        from tilesight.modeling.program import contract
        contract._MAPPER_FACTORIES.pop("reverse", None)
    with pytest.raises(sight.ContractError, match="built in"):
        sight.register_block_mapper("panel_swizzle", lambda data: None)


# ---------------------------------------------------------------------------
# dtype identity per role
# ---------------------------------------------------------------------------


def test_dtype_identity_is_per_role(tmp_path):
    program = ex.gemm_program(m=128, n=128, k=64, dtype="bf16")     # BF16 tensors, FP32 accumulator
    roles = v.program_facts(program)["dtype_roles"]
    assert roles["input"] == ("bf16",) and roles["output"] == ("bf16",) and roles["accumulation"] == ("fp32",)
    base = {"arch": "h100"}
    assert v.bound_identity_differences(dict(base, dtype="bf16"), ARCH, program) == ()
    wrong = v.bound_identity_differences(dict(base, dtype="fp32"), ARCH, program)
    assert any("dtype='fp32' is not the program's input dtype ['bf16']" in r for r in wrong)
    assert any("output dtype" in r for r in wrong)
    assert v.bound_identity_differences(dict(base, dtype="bf16", accumulation_dtype="fp32"), ARCH, program) == ()
    assert any("accumulation_dtype='bf16'" in r for r in
               v.bound_identity_differences(dict(base, dtype="bf16", accumulation_dtype="bf16"), ARCH, program))
    # an FP32 measurement never becomes comparable with the BF16 program
    csv = tmp_path / "gemm.csv"
    csv.write_text("kernel,shape_id,precision,arch,measured_us\ngemm,M128_N128_K64,fp32,h100,10.0\n")
    identity = {"kernel": "gemm", "implementation": "i", "source_version": "s", "variant": "v", "arch": "h100",
                "dtype": "fp32", "causal": "n/a", "gqa": "n/a", "tile": "t", "scheduler": "static",
                "cache_state": "cold", "timing_scope": "kernel_elapsed"}
    declared = {k: val for k, val in identity.items() if k not in ("kernel", "arch", "dtype")}
    measured = sight.load_measurements(str(csv), identity=declared)
    report = sight.validate(measured, lambda p, _r: ex.gemm_program(m=p["m"], n=p["n"], k=p["k"], dtype="bf16"), ARCH,
                         name="x", model_identity=identity, options=sight.Options(cache="fast"),
                         shape_check=lambda p, prog: [])
    assert report.cases[0].status == "unverified" and report.comparable_stats["n"] == 0
    # role fields declared on one side only are a difference
    assert any("accumulation_dtype is declared on only one side" in r
               for r in v.identity_differences(dict(identity, accumulation_dtype="fp32"), identity))


# ---------------------------------------------------------------------------
# GEMM join ambiguity
# ---------------------------------------------------------------------------


def _record(**overrides):
    base = {"raster_order": "along_m", "swizzle_size": "4", "cluster_m": "1", "cluster_n": "2", "kernel_name": "k"}
    base.update(overrides)
    return base


def test_gemm_join_marks_ambiguous_scheduler_records():
    with pytest.raises(ValueError, match="no profiler record"):
        v.cutlass_sm90_mapper_from_records([], 4, 4)
    # two records that CUTLASS resolves to the same launch order are accepted (M has 2 tiles -> swizzle 2) ...
    mapper = v.cutlass_sm90_mapper_from_records([_record(swizzle_size="4"), _record(swizzle_size="2")], 2, 48)
    assert mapper.resolve((2, 48))["swizzle_size"] == 2
    # ... but not when the orders differ
    with pytest.raises(ValueError, match="ambiguous scheduler configuration: 2 profiler records"):
        v.cutlass_sm90_mapper_from_records([_record(swizzle_size="4"), _record(swizzle_size="2")], 8, 48)
    with pytest.raises(ValueError, match="no raster_order"):
        v.cutlass_sm90_mapper_from_records([_record(raster_order="")], 8, 48)
    assert v.cutlass_sm90_mapper_from_records([_record(raster_order="heuristic", swizzle_size="1")], 8, 48).raster_order == "heuristic"


# ---------------------------------------------------------------------------
# Tail-wave resource sharing
# ---------------------------------------------------------------------------


def test_tail_wave_is_priced_with_its_own_active_sms():
    slots = ARCH.sm_count
    tiles = slots + 4                                         # one full wave + a 4-unit tail
    program = ex.gemm_program(m=128 * tiles, n=128, k=256, block_m=128, block_n=128, block_k=64)
    full = sight.analyze(program, ARCH, options=sight.Options(cache="fast"))
    tail = sight.analyze(program, ARCH, options=sight.Options(cache="fast", tail_policy="tail_active_sms"))
    launch = tail.launches["main"]
    context = dict(launch.provenance)["tail_wave_context"]
    assert context["wave_kind"] == "tail" and context["active_sms"] == 4 and launch.tail_work_units == 4
    load = tail.ops["main/k/load_A"]
    kinds = {g.key.wave_kind: g for g in load.cost_groups if g.count}
    assert set(kinds) == {"full", "tail"}
    assert kinds["full"].count + kinds["tail"].count == load.count == full.ops["main/k/load_A"].count
    assert kinds["tail"].count == 4 * 4                       # 4 tail units x 4 k-iterations
    ddr = lambda group: dict(group.service_s)["ddr"]
    assert ddr(kinds["tail"]) == pytest.approx(ddr(kinds["full"]) * 4 / slots)
    assert dict(kinds["tail"].service_s)["l1_5"] == pytest.approx(dict(kinds["full"].service_s)["l1_5"])
    # traffic is conserved; the tail wave starts after the full wave and is never slower per unit
    assert tail.launches["main"].traffic_total.close_to(full.launches["main"].traffic_total)
    per_unit_full = launch.groups[0].per_unit_s
    per_unit_tail = dict(dict(launch.provenance)["tail_per_unit_s"])[launch.groups[0].group_id]
    assert per_unit_tail <= per_unit_full
    assert launch.kernel_body_s == pytest.approx(per_unit_full + per_unit_tail)
    assert launch.kernel_body_s <= full.launches["main"].kernel_body_s
    assert any("tail_policy='tail_active_sms'" in a for a in tail.diagnostics.approximations)
    # no partial wave -> nothing changes
    even = ex.gemm_program(m=128 * slots, n=128, k=256)
    assert (sight.analyze(even, ARCH, options=sight.Options(cache="fast", tail_policy="tail_active_sms")).launches["main"].kernel_body_s
            == sight.analyze(even, ARCH, options=sight.Options(cache="fast")).launches["main"].kernel_body_s)


def test_arch_utilisation_caps_are_an_explicit_context():
    context = sight.Context.from_arch_utilization(ARCH)
    caps = dict(context.max_util)
    for attribute, resources in sight.Context.ARCH_UTILIZATION_FIELDS:
        value = getattr(ARCH, attribute, None)
        if isinstance(value, (int, float)) and 0 < value <= 1:
            assert all(caps[r] == value for r in resources)
    program = ex.gemm_program(m=512, n=512, k=256)
    plain = sight.analyze(program, ARCH, options=sight.Options(cache="fast"))
    capped = sight.analyze(program, ARCH, options=sight.Options(cache="fast"), context=context)
    assert capped.launches["main"].kernel_body_s >= plain.launches["main"].kernel_body_s
    ddr_cap = caps.get("ddr", 1.0)
    service = lambda r: dict(r.ops["main/k/load_A"].cost_groups[0].service_s)["ddr"]
    assert service(capped) == pytest.approx(service(plain) / ddr_cap)
    # an explicit cap wins over the arch value
    override = sight.Context.from_arch_utilization(ARCH, max_util={"ddr": 0.5})
    assert dict(override.max_util)["ddr"] == 0.5


# ---------------------------------------------------------------------------
# FA3 / FA4 shape and mapper behavior
# ---------------------------------------------------------------------------


def test_fa3_noncausal_accepts_a_partial_q_tile():
    from tilesight.arch.h200_sxm import H200_SXM

    program = ex.fa3_program(batch=1, heads=4, seq_len=1024, head_dim=64, block_m=192, block_n=192)   # non-divisible Q tile
    launch = program.launches[0]
    assert dict((a.name, a.extent) for a in launch.work_axes)["q"] == 6                               # ceil(1024 / 192)
    groups = dict(launch.work_units.groups)
    assert set(groups) == {"full", "tail_q"}
    tail = dict(groups["tail_q"].effective_fractions)
    assert tail["load_Q"] == pytest.approx(64 / 192) and tail["store_O"] == pytest.approx(64 / 192)
    assert tail["load_K"] == pytest.approx(1024 / (6 * 192))                                          # averaged KV tail
    result = sight.analyze(program, H200_SXM().set_to_microbench(), options=sight.Options(cache="fast"))
    assert result.ops["main/kv/load_Q"].count == 24
    with pytest.raises(ValueError, match="partial causal q tiles"):
        ex.fa3_program(seq_len=1024, head_dim=64, block_m=192, block_n=192, causal=True)


def test_fa4_causal_groups_follow_the_block_mapper():
    from tilesight.arch.b200 import B200

    arch = B200().set_to_microbench()
    order = ex.sectioned_lpt_order((1, 4, 1, 4), 2)
    assert order.coordinates((1, 4, 1, 4))[:6] == [(0, 0, 0, 3), (0, 1, 0, 3), (0, 0, 0, 2), (0, 1, 0, 2), (0, 0, 0, 1), (0, 1, 0, 1)]
    kwargs = dict(batch=1, heads=4, seq_len=1024, head_dim=64, block_m=256, causal=True, stages=2)
    lpt = ex.fa4_program(mapper=lambda extents: ex.sectioned_lpt_order(extents, 2), **kwargs)
    linear = ex.fa4_program(**kwargs)
    runs = [(run.group_id, run.count) for run in lpt.launches[0].work_units.order]
    assert runs[:4] == [("q3_kv8", 2), ("q2_kv6", 2), ("q1_kv4", 2), ("q0_kv2", 2)]
    assert [run.group_id for run in linear.launches[0].work_units.order][:4] == ["q0_kv2", "q1_kv4", "q2_kv6", "q3_kv8"]
    options = sight.Options(cache="fast")
    a, b = sight.analyze(lpt, arch, options=options), sight.analyze(linear, arch, options=options)
    assert a.ops["main/kv/gemm_QK"].count == b.ops["main/kv/gemm_QK"].count == 4 * (2 + 4 + 6 + 8)
    assert sight.program_from_snapshot(sight.program_to_snapshot(lpt)) == lpt
    # The selected resource bound also includes recurrence and buffer-credit constraints.
    region = a.regions["main/kv"]
    ii = region.groups[0].ii
    assert ii.selected_ii_s == pytest.approx(max(ii.resource_ii_s, ii.recurrence_ii_s, ii.credit_ii_s))


# ---------------------------------------------------------------------------
# Review round 9: utilization 1.0, real periodic schedules, two-Q-stage FA4
# ---------------------------------------------------------------------------


def test_actor_order_sets_the_issue_order_without_touching_data_flow():
    kb = sight.KernelBuilder("main", grid={"i": 4})
    X = kb.tensor("X", (4 * 64, 64), "fp16")
    a = kb.shared("a", (64, 64), "fp16", stages=2)
    t0, t1 = kb.fragment("t0", (64, 64), "fp16"), kb.fragment("t1", (64, 64), "fp16")
    with kb.loop("k", 8, stages=2) as loop:
        kb.copy(X[tiles_i(), :], a, name="load")
        kb.elementwise("first", [a], t0, (("add", 1.0, "cuda"),), actor="worker")
        kb.elementwise("second", [t0], t1, (("add", 1.0, "cuda"),), actor="worker")
        loop.actor_order("worker", [("second", 0), ("first", 1)])
        with pytest.raises(sight.ContractError, match="must list exactly"):
            bad = sight.KernelBuilder("bad", grid={"i": 1})
            b = bad.fragment("b", (4,), "fp16")
            with bad.loop("k", 2, stages=1) as inner:
                bad.elementwise("only", [b], bad.fragment("c", (4,), "fp16"), (("add", 1.0, "cuda"),), actor="w")
                inner.actor_order("w", [("missing", 0)])
            bad.program()
    program = kb.program()
    worker = [actor for actor in program.launches[0].loop("k").body.actors if actor.name == "worker"][0]
    assert tuple(worker.sequence) == (("second", 0), ("first", 1))


def tiles_i():
    from tilesight.modeling.program import tiles
    return tiles("i")


def test_fa4_wasp_two_q_stage_structure_and_periodic_schedule():
    from tilesight.arch.b200 import B200
    from tilesight.modeling.program.periodic_diagnostics import capture_periodic_dags, witness_report

    arch = B200().set_to_microbench()
    program = ex.fa4_wasp_program(batch=1, heads=4, seq_len=1024, head_dim=64, causal=True, q_stages=2, kv_stages=10)
    launch = program.launches[0]
    pipeline = launch.loop("kv").body
    actors = {actor.name: actor for actor in pipeline.actors}
    assert tuple(actors["bmm"].sequence) == (("gemm_PV_s0", 0), ("gemm_QK_s0", 1), ("gemm_PV_s1", 0), ("gemm_QK_s1", 1))
    assert {"softmax_s0", "softmax_s1", "correction", "producer"} <= set(actors)
    assert [op_id for op_id, _o in actors["correction"].sequence] == [
        "tmem_ld_O_s0", "rescale_O_s0", "tmem_st_O_s0", "tmem_ld_O_s1", "rescale_O_s1", "tmem_st_O_s1"]
    carried = {carry.storage for carry in pipeline.carries}
    assert {"O_tmem_s0", "O_tmem_s1", "softmax_state_s0", "softmax_state_s1"} <= carried
    assert dict((a.name, a.extent) for a in launch.work_axes)["q"] == 1024 // 256
    with pytest.raises(ValueError):
        ex.fa4_wasp_program(seq_len=1024 + 128, q_stages=2, causal=True)
    unit = sight.Context(max_util={name: 1.0 for name in ("ddr", "l2", "l1_5", "smem", "register", "tmem", "tensor", "cuda", "sfu", "tma", "ldst")})
    base = sight.Options(cache="fast", ii_mode="periodic_best")
    with capture_periodic_dags() as captured:
        beam = sight.analyze(program, arch, options=base, context=unit)
    proposals = sight.analyze(program, arch, options=sight.Options(cache="fast", ii_mode="periodic_best", periodic_topology_trials=300),
                           context=unit)
    default_context = sight.analyze(program, arch, options=base)
    assert default_context.launches["main"].kernel_body_s == beam.launches["main"].kernel_body_s   # util defaults to 1.0
    group = lambda result: result.regions["main/kv"].groups[-1].ii
    assert group(proposals).selected_ii_s <= group(beam).selected_ii_s          # proposals only add candidates
    ii = group(proposals)
    assert ii.selected_ii_s >= max(ii.resource_ii_s, ii.recurrence_ii_s, ii.credit_ii_s) - 1e-15
    assert ii.witness and ii.search_complete is False
    # every bound cost group reports utilization 1.0
    utils = {value for op in proposals.ops.values() for g in op.cost_groups if g.count
             for key, entries in g.provenance if key == "max_util" for _r, value in entries}
    assert utils == {1.0}
    dag, envelope = max(captured, key=lambda item: item[1].best.ii)
    report = witness_report(dag, envelope)
    cycle = report["critical_cycle"]
    assert cycle["found"] and cycle["cycle_ii_s"] == pytest.approx(envelope.best.ii, rel=1e-5)
    assert max(report["resource_demand_s"].values()) == pytest.approx(report["bounds_s"]["resource"])
    assert sight.program_from_snapshot(sight.program_to_snapshot(program)) == program


def test_fa3_ss_pv_stages_p_through_smem():
    from tilesight.arch.h200_sxm import H200_SXM

    arch = H200_SXM().set_to_microbench()
    options = sight.Options(cache="fast", ii_mode="periodic_best")
    kwargs = dict(batch=1, heads=4, seq_len=1536, head_dim=64, block_m=192, block_n=192)
    register = sight.analyze(ex.fa3_program(**kwargs), arch, options=options)
    staged = sight.analyze(ex.fa3_program(pv_via_smem=True, **kwargs), arch, options=options)
    assert "main/kv/store_P" in staged.ops and "main/kv/store_P" not in register.ops
    actors = {a.name: [op_id for op_id, _o in a.sequence] for a in ex.fa3_program(pv_via_smem=True, **kwargs).launches[0].loop("kv").body.actors}
    assert actors["p_stage"] == ["store_P"]
    assert staged.ops["main/kv/store_P"].traffic_total.smem_write_bytes == pytest.approx(192 * 192 * 2 * staged.ops["main/kv/store_P"].count)
    ii = lambda result: result.regions["main/kv"].groups[0].ii
    assert ii(staged).selected_ii_s > ii(register).selected_ii_s and ii(register).search_complete




# ---------------------------------------------------------------------------
# Review round 10: shared storage, source order, per-stage masks, witness checks
# ---------------------------------------------------------------------------


def _two_stage(d, kv_stages, **kwargs):
    return ex.fa4_wasp_program(batch=1, heads=4, seq_len=1024, head_dim=d, causal=True, q_stages=2, kv_stages=kv_stages, **kwargs)


@pytest.mark.parametrize("d, kv_stages, expected", [
    (64, 10, {("K_shared", "K_shared"): 5, ("V_shared", "V_shared"): 5}),      # even ring: half-capacity reuse distance
    (128, 3, {("K_shared", "V_shared"): 1, ("V_shared", "K_shared"): 2}),      # odd ring: K->V at h, V->K at h+1
])
def test_shared_kv_ring_and_sp_alias_are_explicit_and_fit_the_sm(d, kv_stages, expected):
    from tilesight.arch.b200 import B200

    arch = B200().set_to_microbench()
    program = _two_stage(d, kv_stages)
    pipeline = program.launches[0].loop("kv").body
    ring = {tuple(dep.name.split(":")[2].split("->")): dep for dep in pipeline.dependencies if dep.name.startswith("ring:KV_ring:")}
    assert {key: dep.distance for key, dep in ring.items()} == expected
    # a ring slot is released by the buffer's last reader (both Q stages read K and V)
    assert all(dep.source.op_id.startswith("release_") and dep.source.kind == "done" for dep in ring.values())
    assert {dep.target.op_id for dep in ring.values()} == {"load_K", "load_V"}
    groups = {group.name: group for group in pipeline.storage_groups}
    assert groups["KV_ring"].kind == "ring" and groups["KV_ring"].slots == kv_stages
    assert groups["SP_tmem_s0"].kind == "alias" and groups["SP_tmem_s0"].members == ("S_tmem_s0", "P_tmem_s0")
    slots = {item.buffer: item for item in pipeline.buffer_slots}
    assert "K_shared" not in slots and "P_tmem_s0" not in slots                  # no independent lifetimes
    alias = slots["S_tmem_s0"]
    assert (alias.acquire.op_id, alias.acquire.kind, alias.release.op_id, alias.release.kind, alias.capacity) == (
        "gemm_QK_s0", "start", "gemm_PV_s0", "done", 1)                          # next QK waits for this PV (old SP_slot)
    storage = pipeline.storage_bytes()
    assert storage["smem"] == 224 * 1024                                         # Q + O staging + shared ring
    assert storage["tmem"] == (192 if d == 64 else 256) * 1024                   # (S|P) + O per stage
    assert storage["smem"] <= arch.configurable_smem_capacity and storage["tmem"] <= arch.tmem_capacity_per_sm
    result = sight.analyze(program, arch, options=sight.Options(cache="fast"))
    assert any("declared tmem" in note for note in result.diagnostics.notes)
    assert sight.program_from_snapshot(sight.program_to_snapshot(program)) == program


def test_declared_storage_above_the_sm_capacity_is_rejected():
    from tilesight.arch.b200 import B200

    arch = B200().set_to_microbench()

    def build(separate):
        kb = sight.KernelBuilder("main", grid={"i": 4}, requires_arch=("tmem", "tcgen05"))
        X = kb.tensor("X", (4 * 512, 128), "bf16")
        W = kb.tensor("W", (128, 128), "bf16")
        x = kb.shared("x", (512, 128), "bf16")
        w = kb.shared("w", (128, 128), "bf16")
        a = kb.tmem("a", (512, 128), "bf16")           # 128 KiB
        b = kb.tmem("b", (512, 128), "bf16")           # 128 KiB
        c = kb.tmem("c", (512, 128), "bf16")           # 128 KiB -> 384 KiB unless a and b alias
        r0, r1 = kb.fragment("r0", (512, 128), "bf16"), kb.fragment("r1", (512, 128), "bf16")
        if not separate:
            kb.alias("ab", [a, b])
        with kb.loop("k", 4, stages=1):
            kb.copy(X[sight.tiles("i"), :], x, name="load")
            kb.copy(W[:, :], w, name="load_w")
            kb.gemm(x, w, a, name="g1")
            kb.copy(a, r0, name="ld_a")
            kb.copy(r0, b, name="st_b")
            kb.gemm(b, w, c, name="g2")
            kb.copy(c, r1, name="ld_c")
        return kb.program()

    with pytest.raises(sight.UnsupportedError, match=r"declares 393216 bytes \(384.0 KiB\) of tmem but an SM provides 262144"):
        sight.analyze(build(True), arch, options=sight.Options(cache="fast"))
    reported = sight.analyze(build(True), arch, options=sight.Options(cache="fast", capacity_policy="report"))
    assert any("384.0 KiB" in text for text in reported.diagnostics.unsupported)
    assert sight.analyze(build(False), arch, options=sight.Options(cache="fast")).diagnostics.unsupported == ()


def test_wasp_single_q_stage_follows_the_source_barriers():
    from tilesight.arch.b200 import B200

    program = ex.fa4_wasp_program(batch=1, heads=4, seq_len=1024, head_dim=128, causal=True, q_stages=1)
    pipeline = program.launches[0].loop("kv").body
    actors = {actor.name: [op_id for op_id, _o in actor.sequence] for actor in pipeline.actors}
    assert actors["softmax"] == ["tmem_ld_O", "tmem_ld_S", "online_softmax", "rescale_O", "tmem_st_P", "tmem_st_O"]
    assert actors["bmm"] == ["gemm_QK", "gemm_PV"] and "correction" not in actors
    assert any(dep.source.op_id == "gemm_QK" and dep.source.kind == "done" and dep.target.op_id == "tmem_ld_O"
               for dep in pipeline.dependencies)
    assert not pipeline.storage_groups                       # separate K/V and S/P allocations, as in the source
    result = sight.analyze(program, B200().set_to_microbench(), options=sight.Options(cache="fast", ii_mode="periodic_best"))
    for group in result.regions["main/kv"].groups:
        starts = dict(group.ii.witness)
        service = dict(result.ops["main/kv/gemm_QK"].cost_groups[0].service_s)["tensor"]
        assert starts["tmem_ld_O"] >= starts["gemm_QK"] + service - 1e-15      # O is read only after the QK barrier
        assert starts["tmem_ld_S"] >= starts["tmem_ld_O"] - 1e-15


def test_two_q_stage_causal_work_is_per_stage():
    from tilesight.arch.b200 import B200

    program = ex.fa4_wasp_program(batch=1, heads=1, seq_len=256, head_dim=64, q_stages=2, causal=True, kv_stages=10)
    result = sight.analyze(program, B200().set_to_microbench(), options=sight.Options(cache="fast"))
    assert result.ops["main/kv/gemm_QK_s0"].useful_flops == pytest.approx(1056768)     # rows 0..127
    assert result.ops["main/kv/gemm_QK_s1"].useful_flops == pytest.approx(3153920)     # rows 128..255
    assert result.ops["main/kv/gemm_QK_s0"].executed_flops == result.ops["main/kv/gemm_QK_s1"].executed_flops == 2 * 2 * 128 * 128 * 64
    assert ex.causal_valid_fraction_rows(0, 256, 128, 2) == pytest.approx(ex.causal_valid_fraction(0, 256, 128, 2))


def test_check_witness_reports_violations_at_a_fixed_ii():
    from tilesight.arch.b200 import B200
    from tilesight.modeling.program.periodic_diagnostics import check_witness
    from tilesight.modeling.program.periodic_diagnostics import capture_periodic_dags

    program = _two_stage(64, 10)
    with capture_periodic_dags() as captured:
        sight.analyze(program, B200().set_to_microbench(), options=sight.Options(cache="fast", ii_mode="periodic_best", periodic_topology_trials=200))
    dag, envelope = captured[-1]
    own = check_witness(dag, envelope.best.phase_starts, envelope.best.ii)
    assert own["feasible"] and not own["violated_constraints"] and not own["resource_conflicts"]
    squeezed = check_witness(dag, envelope.best.phase_starts, envelope.best.ii * 0.8)      # same starts, shorter period
    assert not squeezed["feasible"] and (squeezed["violated_constraints"] or squeezed["resource_conflicts"])
    shifted = dict(envelope.best.phase_starts)
    shifted["gemm_PV_s0"] = shifted["tmem_st_P_s0"]                                        # PV before P is stored
    broken = check_witness(dag, shifted, envelope.best.ii)
    assert any("P_tmem_s0" in item["constraint"] or "S_tmem_s0" in item["constraint"] for item in broken["violated_constraints"])
    partial = check_witness(dag, {"load_K": 0.0}, envelope.best.ii)
    assert not partial["feasible"] and "gemm_QK_s0" in partial["ops_without_start"]


# ---------------------------------------------------------------------------
# The examples folder: one runnable module per kernel, builders importable as ``ex.<name>``
# ---------------------------------------------------------------------------


def test_examples_package_exports_and_runnable_modules(capsys):
    import importlib
    import pkgutil

    from tilesight.modeling.program import examples as package

    modules = sorted(info.name for info in pkgutil.iter_modules(package.__path__) if not info.name.startswith("_"))
    assert modules == ["elementwise_add", "fa3", "fa4_ts", "fa4_wasp", "flashmla_decode", "gemm", "reductions"]
    for name in modules:
        module = importlib.import_module(package.__name__ + "." + name)
        assert callable(module.main) and "python -m tilesight.modeling.program.examples." + name in module.__doc__
    for name in package.__all__:                      # lazily resolved builders and helpers
        assert getattr(package, name) is not None and name in dir(package)
    with pytest.raises(AttributeError):
        package.not_an_example
    assert package.gemm_program is importlib.import_module(package.__name__ + ".gemm").gemm_program
    for name in ("elementwise_add", "reductions", "flashmla_decode"):     # the quick ones run end to end
        importlib.import_module(package.__name__ + "." + name).main()
    printed = capsys.readouterr().out
    assert "tail_mn" in printed and "launch combine" in printed and "rms_norm" in printed and "unsupported" not in printed


# ---------------------------------------------------------------------------
# Front-end review: alias safety, launch-wide capacity, strict shapes, stages, report, arithmetic, state, scheduling layer
# ---------------------------------------------------------------------------


def _alias_program(safe):
    kb = sight.KernelBuilder("main", grid={"i": 2})
    A = kb.tensor("A", (2 * 128, 64), "bf16")
    B = kb.tensor("B", (2 * 128, 64), "bf16")
    a = kb.shared("a", (128, 64), "bf16", stages=1)
    b = kb.shared("b", (128, 64), "bf16", stages=1)
    ra, rb, out = (kb.fragment(n, (128, 64), "bf16") for n in ("ra", "rb", "out"))
    kb.alias("ab", [a, b])
    with kb.loop("k", 8, stages=1):
        kb.copy(A[sight.tiles("i"), :], a, name="load_A")
        kb.copy(a, ra, name="read_a", actor="reader")
        kb.copy(B[sight.tiles("i"), :], b, name="load_B")
        kb.copy(b, rb, name="read_b", actor="reader")
        kb.add(ra, rb, out, name="sum", actor="reader")
    return kb.program()


def test_alias_member_is_overwritten_only_after_its_readers():
    program = _alias_program(True)
    pipeline = program.launches[0].loop("k").body
    guard = [dep for dep in pipeline.dependencies if dep.name == "alias:ab:a->b"]
    assert len(guard) == 1 and (guard[0].source.op_id, guard[0].source.kind, guard[0].target.op_id, guard[0].target.kind) == (
        "read_a", "done", "load_B", "start")
    result = sight.analyze(program, ARCH, options=sight.Options(cache="fast", ii_mode="periodic_best"))
    starts = dict(result.regions["main/k"].groups[0].ii.witness)
    read_a_done = starts["read_a"] + max(dict(result.ops["main/k/read_a"].cost_groups[0].service_s).values())
    assert starts["load_B"] >= read_a_done - 1e-15
    # an order that can never be safe (b is produced before a's reader can run) is rejected, not modelled
    kb = sight.KernelBuilder("bad", grid={"i": 2})
    A = kb.tensor("A", (2 * 128, 64), "bf16")
    a, b = kb.shared("a", (128, 64), "bf16", stages=1), kb.shared("b", (128, 64), "bf16", stages=1)
    ra, rb, rc = (kb.fragment(n, (128, 64), "bf16") for n in ("ra", "rb", "rc"))
    kb.alias("ab", [a, b])
    with kb.loop("k", 4, stages=1):
        kb.copy(A[sight.tiles("i"), :], a, name="load_A")
        kb.copy(a, ra, name="read_a")
        kb.copy(ra, b, name="fill_b")
        kb.copy(b, rb, name="read_b")
        kb.copy(a, rc, name="read_a_again")                      # reads a after b has overwritten the slot
    with pytest.raises(sight.UnsupportedError, match="cycle|infeasible"):
        sight.analyze(kb.program(), ARCH, options=sight.Options(cache="fast", ii_mode="periodic_best"))


def test_capacity_counts_each_allocation_once_across_the_launch():
    from tilesight.arch.h200_sxm import H200_SXM

    arch = H200_SXM().set_to_microbench()
    kb = sight.KernelBuilder("main", grid={"i": 2})
    X = kb.tensor("X", (2 * 512, 128), "bf16")
    x = kb.shared("x", (512, 128), "bf16")                         # 128 KiB, reused by two serial loops
    r = kb.fragment("r", (512, 128), "bf16")
    for loop_id in ("first", "second"):
        with kb.loop(loop_id, 2):
            kb.copy(X[sight.tiles("i"), :], x, name="load_" + loop_id)
            kb.copy(x, r, name="ld_" + loop_id)
    program = kb.program()
    assert program.launches[0].storage_bytes() == {"smem": 128 * 1024}
    result = sight.analyze(program, arch, options=sight.Options(cache="fast"))
    assert any("declared smem 128.0 KiB of 228.0 KiB" in note for note in result.diagnostics.notes)
    # an allocation outside every pipeline is part of the check
    kb = sight.KernelBuilder("main", grid={"i": 2})
    X = kb.tensor("X", (2 * 1024, 128), "bf16")
    x = kb.shared("x", (1024, 128), "bf16")                        # 256 KiB > 228 KiB
    r = kb.fragment("r", (1024, 128), "bf16")
    kb.copy(X[sight.tiles("i"), :], x, name="load")
    kb.copy(x, r, name="ld")
    outside = kb.program()
    assert [b.name for b in outside.launches[0].buffers] == ["x", "r"]
    with pytest.raises(sight.UnsupportedError, match=r"256.0 KiB\) of smem but an SM provides"):
        sight.analyze(outside, arch, options=sight.Options(cache="fast"))
    assert sight.program_from_snapshot(sight.program_to_snapshot(outside)) == outside
    # the same name may not describe two different allocations
    launch = program.launches[0]
    with pytest.raises(sight.ContractError, match="declared twice"):
        sight.Launch(launch.name, launch.body, launch.work_axes,
                  buffers=(sight.Buffer("x", "smem", (4, 4), "bf16"), sight.Buffer("x", "smem", (8, 8), "bf16")))


def test_elementwise_shapes_are_strict_and_mappings_explicit():
    kb = sight.KernelBuilder("main", grid={"i": 1})
    a, b, out = kb.fragment("a", (4,), "fp16"), kb.fragment("b", (8,), "fp16"), kb.fragment("out", (8,), "fp16")
    with pytest.raises(sight.ContractError, match=r"input a\(4,\) does not broadcast to output out\(8,\).*row_state.*mapped"):
        kb.add(a, b, out)
    row = kb.fragment("row", (8, 1), "fp16")
    wide = kb.fragment("wide", (8, 16), "fp16")
    state = kb.fragment("state", (8, 2), "fp32")
    other = kb.fragment("other", (4, 2), "fp32")
    op = kb.mul(wide, row, wide, name="scaled")                              # NumPy-style broadcast is fine
    assert [o.access for o in op.spec.operands] == ["identity", "broadcast"]
    op = kb.mul(wide, kb.row_state(state), wide, name="per_row")
    assert op.spec.operands[1].access == "affine" and dict(op.spec.operands[1].index_map)["kind"] == "row_state"
    with pytest.raises(sight.ContractError, match="same leading extent"):
        kb.mul(wide, kb.row_state(other), wide, name="bad_rows")
    op = kb.elementwise("gather", [kb.mapped(other, {"kind": "gather", "index": "row // 2"})], wide, (("add", 1.0, "cuda"),))
    assert dict(op.spec.operands[0].index_map)["index"] == "row // 2"
    with pytest.raises(sight.ContractError, match="non-empty index_map"):
        kb.mapped(other, {})


def test_arithmetic_helpers_derive_their_work():
    kb = sight.KernelBuilder("main", grid={"i": 1})
    x, y, z = (kb.fragment(n, (16, 8), "fp16") for n in ("x", "y", "z"))
    s = kb.fragment("s", (16, 1), "fp32")
    work = lambda op: [(w.op_class, w.engine.name, w.count) for w in op.spec.work_items]
    assert work(kb.add(x, y, z)) == [("add", "cuda", 128)]
    assert work(kb.fma(x, y, z, z, name="fma")) == [("fma", "cuda", 128)]
    assert work(kb.exp2(x, z, name="e")) == [("exp2", "sfu", 128)]
    assert work(kb.unary("rsqrt", x, z, name="r")) == [("rsqrt", "sfu", 128)]
    assert work(kb.reduce_sum(x, s, axis=1, name="rs")) == [("add", "cuda", 128)]
    assert work(kb.reduce_max(x, s, axis=1, name="rm")) == [("max", "cuda", 128)]
    with pytest.raises(sight.ContractError, match="supported functions"):
        kb.unary("erf", x, z)


def test_unspecified_shared_stages_follow_the_loop_and_are_reported():
    def build(buffer_stages, loop_stages):
        kb = sight.KernelBuilder("main", grid={"i": 4})
        X = kb.tensor("X", (4 * 64, 64), "bf16")
        x = kb.shared("x", (64, 64), "bf16", stages=buffer_stages)
        r = kb.fragment("r", (64, 64), "bf16")
        q = kb.shared("q", (64, 64), "bf16")
        with kb.loop("k", 8, stages=loop_stages) as loop:
            with loop.prologue():
                kb.copy(X[sight.tiles("i"), :], q, name="load_q")           # written outside the body: one slot
            kb.copy(X[sight.tiles("i"), :], x, name="load")
            kb.copy(x, r, name="ld")
        return kb.program()

    slots = lambda program: {item.buffer: item.capacity for item in program.launches[0].loop("k").body.buffer_slots}
    assert slots(build(None, 3)) == {"x": 3}                   # inherits the pipeline depth
    assert slots(build(1, 3)) == {"x": 1}                      # an explicit physical slot count wins
    assert slots(build(None, 1)) == {"x": 1}
    program = build(None, 3)
    assert {b.name: b.slots for b in program.launches[0].buffers}["q"] == 1
    result = sight.analyze(program, ARCH, options=sight.Options(cache="fast"))
    assert any("effective on-chip slots: " in note and "x=3" in note and "q=1" in note and "pipeline depth 3" in note
               for note in result.diagnostics.notes)


def test_read_only_carried_state_does_not_serialise_its_readers():
    kb = sight.KernelBuilder("main", grid={"i": 1})
    state = kb.fragment("state", (8, 2), "fp32", carried=True)
    x, a, b, c = (kb.fragment(n, (8, 4), "fp32") for n in ("x", "a", "b", "c"))
    X = kb.tensor("X", (8, 4), "fp32")
    with kb.loop("k", 4, stages=1):
        kb.copy(X[:, :], x, name="load")
        kb.elementwise("update", [x, kb.row_state(state)], a, (("add", 1.0, "cuda"),), updates=[state], actor="u")
        kb.mul(x, kb.row_state(state), b, name="reader_two", actor="r2")
        kb.mul(a, kb.row_state(state), c, name="reader_three", actor="r3")
    names = {dep.name for dep in kb.program().launches[0].loop("k").body.dependencies if dep.name.startswith("inplace:state")}
    assert "inplace:state:update->reader_two" in names and "inplace:state:update->reader_three" in names
    assert not any("reader_two->reader_three" in name or "reader_three->reader_two" in name for name in names)
    assert {"inplace:state:reader_two->update", "inplace:state:reader_three->update"} <= names       # next update waits for both
    # a carried buffer that nothing updates is a declaration error
    kb = sight.KernelBuilder("main", grid={"i": 1})
    state = kb.fragment("state", (8, 2), "fp32", carried=True)
    x, y = kb.fragment("x", (8, 4), "fp32"), kb.fragment("y", (8, 4), "fp32")
    X = kb.tensor("X", (8, 4), "fp32")
    with kb.loop("k", 4, stages=1):
        kb.copy(X[:, :], x, name="load")
        kb.mul(x, kb.row_state(state), y, name="only_reader")
    with pytest.raises(sight.ContractError, match="never updated in the loop body"):
        kb.program()


def test_schedule_layer_is_explicit_and_explainable(capsys):
    from tilesight.arch.b200 import B200
    from tilesight.modeling.program.periodic_diagnostics import explain_loop

    program = ex.fa4_wasp_program(batch=1, heads=4, seq_len=1024, head_dim=128, causal=True, q_stages=1)
    out = explain_loop(program, B200().set_to_microbench())
    assert any("declared schedule constraints: barriers [gemm_QK->tmem_ld_O]" in note for note in out["result"].diagnostics.notes)
    assert "hint:gemm_QK->tmem_ld_O" in out["witness"]["declared_constraints_on_critical_cycle"]
    kb = sight.KernelBuilder("main", grid={"i": 2})
    X = kb.tensor("X", (2 * 64, 64), "bf16")
    x, r = kb.shared("x", (64, 64), "bf16", stages=2), kb.fragment("r", (64, 64), "bf16")
    with kb.loop("k", 4, stages=2) as loop:
        kb.copy(X[sight.tiles("i"), :], x, name="load")
        kb.copy(x, r, name="ld")
        loop.schedule.window("load", 1)
        loop.schedule.window("nope", 1)
    with pytest.raises(sight.ContractError, match="outside the loop body"):
        kb.program()
    # the example reporter keeps unknowns and assumptions visible
    ex.report(out["result"], loop="main/kv", op="main/kv/tmem_ld_S")
    printed = capsys.readouterr().out
    assert "unknown (" in printed and "assumptions and approximations" in printed and "effective on-chip slots" in printed


# ---------------------------------------------------------------------------
# Combination review: repeatable builds, broadcast tails, explain the selected witness, serial-loop schedule
# ---------------------------------------------------------------------------


def test_slot_resolution_is_launch_wide_and_build_is_repeatable():
    def builder(second_depth, explicit=None):
        kb = sight.KernelBuilder("main", grid={"i": 2})
        X = kb.tensor("X", (2 * 64, 64), "bf16")
        x = kb.shared("x", (64, 64), "bf16", stages=explicit)
        r = kb.fragment("r", (64, 64), "bf16")
        with kb.loop("first", 4, stages=2) as loop:
            with loop.prologue():
                kb.copy(X[sight.tiles("i"), :], x, name="pro_load")        # written in a prologue, read in the body
            kb.copy(x, r, name="ld1")
        with kb.loop("second", 4, stages=second_depth):
            kb.copy(X[sight.tiles("i"), :], x, name="load2")
            kb.copy(x, r, name="ld2")
        return kb

    kb = builder(3)
    first, second = kb.program(), kb.program()                          # the first build works and equals the second
    assert first == second
    launch = first.launches[0]
    assert {b.name: b.slots for b in launch.buffers}["x"] == 3
    for loop_id in ("first", "second"):                                 # one definition in every pipeline
        assert {b.name: b.slots for b in launch.loop(loop_id).body.buffers}["x"] == 3
    assert {s.buffer: s.capacity for s in launch.loop("second").body.buffer_slots} == {"x": 3}
    sight.analyze(first, ARCH, options=sight.Options(cache="fast"))
    # two pipelined bodies that write it with different depths are ambiguous
    kb = sight.KernelBuilder("main", grid={"i": 2})
    X = kb.tensor("X", (2 * 64, 64), "bf16")
    x, r = kb.shared("x", (64, 64), "bf16"), kb.fragment("r", (64, 64), "bf16")
    for loop_id, depth in (("a", 2), ("b", 3)):
        with kb.loop(loop_id, 4, stages=depth):
            kb.copy(X[sight.tiles("i"), :], x, name="load_" + loop_id)
            kb.copy(x, r, name="ld_" + loop_id)
    for _attempt in range(2):                                           # the same error every time
        with pytest.raises(sight.ContractError, match=r"different depth \(a: stages=2, b: stages=3\).*explicit stages"):
            kb.program()
    assert {b.name: b.slots for b in builder(3, explicit=1).program().launches[0].buffers}["x"] == 1


def test_valid_range_follows_broadcast_axes():
    kb = sight.KernelBuilder("main", grid={"j": 1})
    X, Y, C = kb.tensor("X", (1, 5), "fp16"), kb.tensor("Y", (4, 5), "fp16"), kb.tensor("C", (1,), "fp16")
    x, c, y = kb.fragment("x", (1, 8), "fp16"), kb.fragment("c", (1,), "fp16"), kb.fragment("y", (4, 8), "fp16")
    kb.copy(X[:, sight.tiles("j")], x, name="load_X")                      # 5 of 8 columns are valid
    kb.copy(C[:], c, name="load_c")
    kb.add(x, c, y, name="add")                                         # x broadcasts over rows, c over everything
    kb.copy(y, Y[:, sight.tiles("j")], name="store_Y")
    result = sight.analyze(kb.program(), ARCH, options=sight.Options(cache="fast"))
    add = result.ops["main/add"]
    assert add.executed_flops == 32 and add.useful_flops == pytest.approx(20)
    assert result.ops["main/store_Y"].useful_bytes == pytest.approx(20 * 2)
    # a broadcast axis itself constrains nothing: a full (4, 8) operand keeps the output full
    kb = sight.KernelBuilder("main", grid={"j": 1})
    X, Y = kb.tensor("X", (4, 8), "fp16"), kb.tensor("Y", (4, 8), "fp16")
    full, row, y = kb.fragment("full", (4, 8), "fp16"), kb.fragment("row", (4, 1), "fp16"), kb.fragment("y", (4, 8), "fp16")
    R = kb.tensor("R", (4, 1), "fp16")
    kb.copy(X[:, sight.tiles("j")], full, name="load_X")
    kb.copy(R[:, :], row, name="load_R")
    kb.mul(full, row, y, name="scale")
    kb.copy(y, Y[:, sight.tiles("j")], name="store_Y")
    result = sight.analyze(kb.program(), ARCH, options=sight.Options(cache="fast"))
    assert result.ops["main/scale"].useful_flops == result.ops["main/scale"].executed_flops == 32


def test_explain_loop_explains_the_selected_endpoint():
    from tilesight.arch.h200_sxm import H200_SXM
    from tilesight.modeling.program.periodic_diagnostics import explain_loop

    arch = H200_SXM().set_to_microbench()
    program = ex.fa3_program(batch=1, heads=4, seq_len=768, head_dim=64, block_m=192, block_n=192, pv_via_smem=True)
    explained = {}
    for mode in ("periodic_best", "periodic_worst"):
        out = explain_loop(program, arch, options=sight.Options(cache="fast", ii_mode=mode))
        witness = out["witness"]
        selected = out["result"].regions[witness["loop"]].groups[0].ii
        assert witness["loop"] == "main/kv" and witness["work_group"] == out["result"].regions["main/kv"].groups[0].work_group_id
        assert witness["endpoint"] == mode.split("_")[1] and witness["ii_mode"] == mode
        assert witness["ii_s"] == pytest.approx(selected.selected_ii_s, rel=1e-12)
        assert witness["phase_starts_s"] == pytest.approx(dict(selected.witness))
        assert witness["critical_cycle"]["cycle_ii_s"] == pytest.approx(selected.selected_ii_s, rel=1e-5)
        explained[mode] = witness["ii_s"]
    assert explained["periodic_worst"] > explained["periodic_best"]              # two different witnesses, each explained
    none = explain_loop(program, arch, options=sight.Options(cache="fast", ii_mode="resource_ii"))
    assert none["witness"] is None and "no constructive periodic witness" in none["note"]
    assert explain_loop(program, arch, loop="main/nope")["witness"] is None


@pytest.mark.parametrize("declare, message", [
    (lambda loop: loop.schedule.window("missing_op", 1), r"serial \(stages=None\): window\(\)"),
    (lambda loop: loop.schedule.actor_order("missing_actor", [("load", 0)]), r"serial \(stages=None\): actor_order\(\)"),
    (lambda loop: loop.schedule.anchor("load"), r"serial \(stages=None\): anchor\(\)"),
    (lambda loop: loop.schedule.after("load", "ld"), r"serial \(stages=None\): after\(\)"),
])
def test_schedule_declarations_on_a_serial_loop_are_rejected(declare, message):
    kb = sight.KernelBuilder("main", grid={"i": 2})
    X = kb.tensor("X", (2 * 64, 64), "bf16")
    x, r = kb.shared("x", (64, 64), "bf16"), kb.fragment("r", (64, 64), "bf16")
    with kb.loop("k", 4) as loop:                                        # stages=None -> serial
        kb.copy(X[sight.tiles("i"), :], x, name="load")
        kb.copy(x, r, name="ld")
        declare(loop)
    with pytest.raises(sight.ContractError, match=message):
        kb.program()


@pytest.mark.parametrize("declare, message", [
    (lambda loop: loop.schedule.actor_order("missing_actor", [("load", 0)]), "unknown actors"),
    (lambda loop: loop.schedule.after("load", "missing_op"), r"after\(\) names ops outside"),
    (lambda loop: loop.schedule.anchor("missing_op"), r"anchor\(\) names an op outside"),
    (lambda loop: loop.schedule.window("missing_op", 1), r"window\(\) names ops outside"),
])
def test_schedule_declarations_must_reference_body_objects(declare, message):
    kb = sight.KernelBuilder("main", grid={"i": 2})
    X = kb.tensor("X", (2 * 64, 64), "bf16")
    x, r = kb.shared("x", (64, 64), "bf16", stages=2), kb.fragment("r", (64, 64), "bf16")
    with kb.loop("k", 4, stages=2) as loop:
        kb.copy(X[sight.tiles("i"), :], x, name="load")
        kb.copy(x, r, name="ld")
        declare(loop)
    with pytest.raises(sight.ContractError, match=message):
        kb.program()


# ---------------------------------------------------------------------------
# Valid ranges of in-place / mapped inputs; explanations bound to identity and the effective II mode
# ---------------------------------------------------------------------------


def test_in_place_output_keeps_the_old_valid_range():
    def build(in_place):
        kb = sight.KernelBuilder("main", grid={"j": 1})
        X, Y = kb.tensor("X", (5,), "fp16"), kb.tensor("Y", (5,), "fp16")
        x, y = kb.fragment("x", (8,), "fp16"), kb.fragment("y", (8,), "fp16")
        kb.copy(X[sight.tiles("j")], x, name="load")                       # 5 of 8 elements valid
        kb.exp2(x, x if in_place else y, name="e")
        kb.copy(x if in_place else y, Y[sight.tiles("j")], name="store")
        return sight.analyze(kb.program(), ARCH, options=sight.Options(cache="fast"))

    for in_place in (False, True):
        result = build(in_place)
        assert result.ops["main/e"].executed_flops == 8 and result.ops["main/e"].useful_flops == pytest.approx(5)
        assert result.ops["main/store"].useful_bytes == pytest.approx(5 * 2)


def test_mapped_inputs_do_not_imply_a_valid_range():
    kb = sight.KernelBuilder("main", grid={"j": 1})
    X, Y = kb.tensor("X", (5,), "fp16"), kb.tensor("Y", (8,), "fp16")
    x, y = kb.fragment("x", (8,), "fp16"), kb.fragment("y", (8,), "fp16")
    kb.copy(X[sight.tiles("j")], x, name="load")
    kb.elementwise("gather", [kb.mapped(x, {"kind": "gather", "index": "i % 5"})], y, (("add", 1.0, "cuda"),))
    kb.copy(y, Y[sight.tiles("j")], name="store")
    result = sight.analyze(kb.program(), ARCH, options=sight.Options(cache="fast"))
    assert result.ops["main/gather"].executed_flops == result.ops["main/gather"].useful_flops == 8   # all 8 indices hit valid input
    assert result.ops["main/store"].useful_bytes == pytest.approx(8 * 2)
    assert dict(result.ops["main/gather"].cost_groups[0].provenance) is not None


def test_explain_loop_uses_the_recorded_dag_identity():
    from tilesight.arch.h200_sxm import H200_SXM
    from tilesight.modeling.program.periodic_diagnostics import explain_loop

    arch = H200_SXM().set_to_microbench()
    kb = sight.KernelBuilder("main", grid={"i": 2})
    X = kb.tensor("X", (64, 64), "bf16")
    x, r = kb.shared("x", (64, 64), "bf16", stages=2), kb.fragment("r", (64, 64), "bf16")
    with kb.loop("k", 4, stages=2):
        kb.copy(X[:, :], x, name="load")            # both units read the same tile: the second one hits in L2
        kb.copy(x, r, name="ld")
    units = sight.WorkUnits((("cold", sight.WorkGroup()), ("warm", sight.WorkGroup())), (sight.WorkRun("cold", 1), sight.WorkRun("warm", 1)))
    program = kb.program(work_units=units, tails="declared")
    options = sight.Options(cache="trace", ii_mode="periodic_best")
    result = sight.analyze(program, arch, options=options)
    groups = {g.work_group_id: g for g in result.regions["main/k"].groups}
    assert groups["cold"].ii.selected_ii_s == groups["warm"].ii.selected_ii_s            # same period ...
    assert dict(groups["cold"].ii.witness) == dict(groups["warm"].ii.witness)             # ... and the same start times
    assert groups["cold"].periodic_dag_digest != groups["warm"].periodic_dag_digest       # but different cost bindings
    for name in ("cold", "warm"):
        witness = explain_loop(program, arch, group=name, options=options)["witness"]
        assert witness["work_group"] == name and witness["periodic_dag_digest"] == groups[name].periodic_dag_digest
    cold, warm = (explain_loop(program, arch, group=n, options=options)["witness"]["resource_demand_s"] for n in ("cold", "warm"))
    assert cold.get("ddr", 0.0) > 0.0 and warm.get("ddr") is None                        # the warm group has no DDR service
    snapshot = dict((path, region) for path, region in result.to_snapshot()["regions"])
    assert snapshot["main/k"]["groups"][0]["periodic_dag_digest"] == groups["cold"].periodic_dag_digest


def test_loop_local_ii_mode_is_reported_and_explained():
    from dataclasses import replace

    from tilesight.arch.h200_sxm import H200_SXM
    from tilesight.modeling.program.periodic_diagnostics import explain_loop

    arch = H200_SXM().set_to_microbench()
    launch = ex.fa3_program(batch=1, heads=4, seq_len=768, head_dim=64, block_m=192, block_n=192, pv_via_smem=True).launches[0]
    local = sight.Program((replace(launch, body=replace(launch.body, ii_mode="periodic_worst")),), name="local")
    options = sight.Options(cache="fast", ii_mode="periodic_best")
    result = sight.analyze(local, arch, options=options)
    region = result.regions["main/kv"]
    assert region.ii.mode == region.groups[0].ii.mode == "periodic_worst"
    assert region.groups[0].ii.selected_ii_s == pytest.approx(region.groups[0].ii.worst_ii_s)
    baseline = sight.analyze(sight.Program((launch,), name="global"), arch, options=options).regions["main/kv"].groups[0].ii
    assert baseline.mode == "periodic_best" and baseline.selected_ii_s < region.groups[0].ii.selected_ii_s
    witness = explain_loop(local, arch, options=options)["witness"]
    assert witness is not None and witness["endpoint"] == "worst" and witness["ii_mode"] == "periodic_worst"
    assert witness["ii_s"] == pytest.approx(region.groups[0].ii.selected_ii_s, rel=1e-12)


@pytest.mark.parametrize("mapped_first", [False, True])
def test_plain_read_of_a_mapped_buffer_still_bounds_the_output(mapped_first):
    kb = sight.KernelBuilder("main", grid={"j": 1})
    X, Y = kb.tensor("X", (5,), "fp16"), kb.tensor("Y", (5,), "fp16")
    x, y = kb.fragment("x", (8,), "fp16"), kb.fragment("y", (8,), "fp16")
    kb.copy(X[sight.tiles("j")], x, name="load")                                   # x[0:5] valid
    gathered = kb.mapped(x, {"kind": "gather", "index": "i % 5"})
    operands = (gathered, x) if mapped_first else (x, gathered)
    op = kb.add(operands[0], operands[1], y, name="add")                         # x[i] + x[i % 5]
    kb.copy(y, Y[sight.tiles("j")], name="store")
    assert [o.access for o in op.spec.operands] == (["affine", "identity"] if mapped_first else ["identity", "affine"])
    result = sight.analyze(kb.program(), ARCH, options=sight.Options(cache="fast"))
    add = result.ops["main/add"]
    assert add.executed_flops == 8 and add.useful_flops == pytest.approx(5)       # the plain read x[i] limits useful work
    assert result.ops["main/store"].useful_bytes == pytest.approx(5 * 2)          # store validity follows the destination slice


@pytest.mark.parametrize("target_length, store_elements", [(5, 5), (8, 8)])
def test_store_usefulness_follows_the_destination_slice_not_the_compute(target_length, store_elements):
    kb = sight.KernelBuilder("main", grid={"j": 1})
    X, Y = kb.tensor("X", (5,), "fp16"), kb.tensor("Y", (target_length,), "fp16")
    x, y = kb.fragment("x", (8,), "fp16"), kb.fragment("y", (8,), "fp16")
    kb.copy(X[sight.tiles("j")], x, name="load")
    kb.add(x, kb.mapped(x, {"kind": "gather", "index": "i % 5"}), y, name="add")
    kb.copy(y, Y[sight.tiles("j")], name="store")
    result = sight.analyze(kb.program(), ARCH, options=sight.Options(cache="fast"))
    assert result.ops["main/add"].useful_flops == pytest.approx(5)                      # compute: 5 of 8 outputs useful
    store = result.ops["main/store"]
    assert store.executed_bytes == 8 * 2 and store.useful_bytes == pytest.approx(store_elements * 2)
