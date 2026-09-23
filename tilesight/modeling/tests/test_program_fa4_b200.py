"""FA4 forward on B200 (SM100 ``ts`` variant) through the unified Program -> analyze path."""

import copy
import json

import pytest

from tilesight.arch.b200 import B200
from tilesight.arch.h100_sxm import H100_SXM
from tilesight.modeling import program as sight
from tilesight.modeling.program import examples as ex

B200_ARCH = B200()
OPTIONS = sight.Options(cache="fast", ii_mode="periodic_best", cache_sample_budget=128)
BLOCK_M = BLOCK_N = HEAD_DIM = 128
TMEM_OPS = ("tmem_ld_S", "tmem_st_P", "tmem_ld_D")


def _analyze(program, arch=B200_ARCH, options=OPTIONS):
    return sight.analyze(program, arch, options=options)


def test_fa4_ts_structure_matches_source_variant():
    program = ex.fa4_program(seq_len=1024, heads=4)
    assert sight.program_from_snapshot(json.loads(json.dumps(sight.program_to_snapshot(program)))) == program
    launch = program.launches[0]
    assert launch.requires_arch == ("tmem", "tcgen05")
    assert dict(launch.metadata)["variant"] == "ts"
    assert launch.work_axis_names == ("batch", "kv_head", "hg", "q")  # T.Kernel(q, head, batch): q fastest
    ops = {op.op_id: op for op in launch.ops()}
    assert (ops["tmem_ld_S"].source, ops["tmem_ld_S"].destination) == ("tmem", "register")
    assert (ops["tmem_st_P"].source, ops["tmem_st_P"].destination) == ("register", "tmem")
    assert (ops["tmem_ld_D"].source, ops["tmem_ld_D"].destination) == ("tmem", "register")
    assert "tmem_ld_O" not in ops and "tmem_st_O" not in ops  # ts keeps O in registers
    pipeline = launch.loop("kv").body
    buffers = {b.name: b for b in pipeline.buffers}
    assert {buffers[n].storage for n in ("S_tmem", "P_tmem", "D_tmem")} == {"tmem"}
    assert buffers["O_reg"].storage == "register"
    assert pipeline.stages == 1
    assert "S_tmem" in ops["gemm_QK"].writes and "S_reg" in ops["tmem_ld_S"].writes
    assert "P_cast" in ops["online_softmax"].writes and "P_cast" in ops["tmem_st_P"].reads
    assert "P_tmem" in ops["gemm_PV"].reads and "D_tmem" in ops["gemm_PV"].writes
    slots = {s.buffer: s for s in pipeline.buffer_slots}
    assert slots["S_tmem"].acquire.op_id == "gemm_QK" and slots["S_tmem"].release.op_id == "tmem_ld_S"
    assert slots["P_tmem"].acquire.op_id == "tmem_st_P" and slots["P_tmem"].release.op_id == "gemm_PV"
    assert slots["D_tmem"].acquire.op_id == "gemm_PV" and slots["D_tmem"].release.op_id == "tmem_ld_D"
    carries = {c.state: c for c in pipeline.carries}
    assert carries["carry_O_reg"].source.op_id == "accumulate_O" and carries["carry_O_reg"].target.op_id == "rescale_O"
    assert all(op.timing is None for op in launch.ops())


def _witness(result):
    group = result.regions["main/kv"].groups[0]
    starts = dict(group.ii.witness)
    latency = {c.op_id: c.completion_latency_s for c in group.completion_trace}
    return starts, latency


def test_fa4_witness_orders_tmem_load_compute_store():
    result = _analyze(ex.fa4_program(seq_len=1024, heads=2))
    starts, latency = _witness(result)
    tolerance = 1e-15

    def after(later, earlier):
        assert starts[later] + tolerance >= starts[earlier] + latency[earlier], (later, earlier)

    after("tmem_ld_S", "gemm_QK")          # S must be in TMEM before tcgen05.ld
    after("online_softmax", "tmem_ld_S")   # TMEM load.done -> compute.start
    after("tmem_st_P", "online_softmax")   # compute.done -> TMEM store.start
    after("gemm_PV", "tmem_st_P")          # P in TMEM before the MMA reads it
    after("tmem_ld_D", "gemm_PV")
    after("accumulate_O", "tmem_ld_D")
    after("accumulate_O", "rescale_O")     # in-place O_reg chain
    assert starts["gemm_QK"] + tolerance >= starts["load_K"] + latency["load_K"]
    assert starts["gemm_PV"] + tolerance >= starts["load_V"] + latency["load_V"]


def test_fa4_tmem_traffic_and_service_on_b200():
    program = ex.fa4_program(seq_len=1024, heads=4)
    result = _analyze(program)
    assert not result.diagnostics.unsupported
    kv_tiles = 1024 // BLOCK_N
    units = (1024 // BLOCK_M) * 4
    ld_s = result.ops["main/kv/tmem_ld_S"]
    st_p = result.ops["main/kv/tmem_st_P"]
    ld_d = result.ops["main/kv/tmem_ld_D"]
    assert ld_s.count == st_p.count == ld_d.count == units * kv_tiles
    assert ld_s.traffic_total.tmem_read_bytes == pytest.approx(units * kv_tiles * BLOCK_M * BLOCK_N * 4)
    assert ld_s.traffic_total.tmem_write_bytes == 0.0 and ld_s.traffic_total.register_write_bytes == ld_s.traffic_total.tmem_read_bytes
    assert st_p.traffic_total.tmem_write_bytes == pytest.approx(units * kv_tiles * BLOCK_M * BLOCK_N * 2)
    assert st_p.traffic_total.register_read_bytes == st_p.traffic_total.tmem_write_bytes
    assert ld_d.traffic_total.tmem_read_bytes == pytest.approx(units * kv_tiles * BLOCK_M * HEAD_DIM * 4)
    # The MMA accumulator writes are not tcgen05.ld/st traffic and TMEM is never SMEM.
    for op_id in ("gemm_QK", "gemm_PV"):
        assert result.ops["main/kv/" + op_id].traffic_total.value("tmem_write_bytes") is None
        assert "tmem" not in dict(result.ops["main/kv/" + op_id].cost_groups[0].service_s)
    for op in (ld_s, st_p, ld_d):
        assert op.traffic_total.smem_read_bytes == 0.0 and op.traffic_total.smem_write_bytes == 0.0
        assert op.traffic_total.l2_request_bytes == 0.0 and op.traffic_total.ddr_read_bytes == 0.0
        service = dict(op.cost_groups[0].service_s)
        assert "tmem" in service and "smem" not in service
    assert result.regions["main/kv"].ii.witness is not None
    assert result.ops["main/kv/store_O"].count == units and result.ops["main/kv/load_Q"].count == units


def test_fa4_tmem_bandwidth_changes_tmem_service_and_ii():
    program = ex.fa4_program(seq_len=1024, heads=4)
    base = _analyze(program)
    faster = copy.deepcopy(B200_ARCH)
    faster.tmem_bandwidth *= 2.0
    quick = _analyze(program, faster)
    for op_id in TMEM_OPS:
        before = dict(base.ops["main/kv/" + op_id].cost_groups[0].service_s)
        after = dict(quick.ops["main/kv/" + op_id].cost_groups[0].service_s)
        assert after["tmem"] == pytest.approx(before["tmem"] / 2.0)
        assert after["register"] == pytest.approx(before["register"])
    assert base.input_digest != quick.input_digest
    slower = copy.deepcopy(B200_ARCH)
    slower.tmem_bandwidth /= 64.0
    slow = _analyze(program, slower)
    assert slow.regions["main/kv"].ii.selected_ii_s > base.regions["main/kv"].ii.selected_ii_s
    assert slow.program.total_s > base.program.total_s


def test_fa4_rejects_targets_without_tmem_or_tcgen05():
    program = ex.fa4_program(seq_len=512, heads=2)
    with pytest.raises(sight.UnsupportedError, match="requires tmem, tcgen05") as info:
        _analyze(program, H100_SXM())
    assert "tmem" in str(info.value) and "tcgen05" in str(info.value)
    no_tcgen05 = copy.deepcopy(B200_ARCH)
    no_tcgen05.support_utcmma = False
    with pytest.raises(sight.UnsupportedError, match="tcgen05"):
        _analyze(program, no_tcgen05)
    no_tmem = copy.deepcopy(B200_ARCH)
    no_tmem.tmem_bandwidth = 0.0
    with pytest.raises(sight.UnsupportedError, match="tmem"):
        _analyze(program, no_tmem)


def test_fa4_causal_boundary_and_gqa_follow_the_source():
    program = ex.fa4_program(seq_len=512, heads=1, block_m=128, block_n=64, causal=True)
    trips = [g.trip_count("kv") for _gid, g in program.launches[0].work_units.groups]
    assert trips == [2, 4, 6, 8]  # ceildiv((bx+1)*128, 64) capped at 8
    fractions = [g.effective_fraction("gemm_QK") for _gid, g in program.launches[0].work_units.groups]
    assert fractions[0] == pytest.approx(ex.causal_valid_fraction(0, 128, 64, 2))
    assert all(0.5 < f < 1.0 for f in fractions)
    with pytest.raises(ValueError, match="multiple of block_m"):
        ex.fa4_program(seq_len=500, heads=1)
    tail = ex.fa4_program(seq_len=1024 + 64, heads=1, block_m=64, block_n=128)
    group = tail.launches[0].work_units.groups[0][1]
    assert group.effective_fraction("load_K") == pytest.approx(1088 / 1152)
    program = ex.fa4_program(seq_len=1024, heads=4, kv_heads=2, causal=True)
    launch = program.launches[0]
    assert [axis.extent for axis in launch.work_axes] == [1, 2, 2, 8]
    result = _analyze(program, options=sight.Options(cache="fast", cache_sample_budget=64))
    assert not result.diagnostics.unsupported
    assert {g.group_id for g in result.launches["main"].groups} == {"q%d_kv%d" % (i, i + 1) for i in range(8)}
    load_k = result.ops["main/kv/load_K"]
    assert load_k.count == sum((i + 1) * 4 for i in range(8))
    assert load_k.traffic_total.ddr_read_bytes < 0.75 * load_k.traffic_total.payload_read_bytes
    assert result.ops["main/kv/gemm_QK"].useful_flops < result.ops["main/kv/gemm_QK"].executed_flops
