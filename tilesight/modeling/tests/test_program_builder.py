"""KernelBuilder: derived IR, lifetimes, carried chains, ambiguity errors."""

import pytest

from tilesight.arch.h100_sxm import H100_SXM
from tilesight.modeling import program as sight
from tilesight.modeling.program import examples as ex
from tilesight.modeling.program.builder import tiles

ARCH = H100_SXM()


def _gemm_builder(stages=3):
    kb = sight.KernelBuilder("main", grid={"n": 4, "m": 4}, threads=256)
    A = kb.tensor("A", (512, 256), "bf16")
    B = kb.tensor("B", (256, 512), "bf16")
    C = kb.tensor("C", (512, 512), "fp32")
    A_s = kb.shared("A_s", (128, 64), "bf16", stages=stages)
    B_s = kb.shared("B_s", (64, 128), "bf16", stages=stages)
    acc = kb.fragment("acc", (128, 128), "fp32", carried=True)
    return kb, A, B, C, A_s, B_s, acc


def test_builder_derives_ids_work_lifetimes_and_carry():
    kb, A, B, C, A_s, B_s, acc = _gemm_builder()
    with kb.loop("k", 4, stages=3) as loop:
        kb.copy(A["m", "k"], A_s)
        kb.copy(B["k", "n"], B_s)
        kb.gemm(A_s, B_s, acc)
        with loop.epilogue():
            kb.copy(acc, C["m", "n"])
    program = kb.program()
    launch = program.launches[0]
    assert launch.work_axis_names == ("m", "n")  # grid x (n) is fastest -> last axis
    assert program.op_paths() == ("main/k/load_A", "main/k/load_B", "main/k/gemm_A_s_B_s", "main/k/store_C")
    ops = {op.op_id: op for op in launch.ops()}
    assert ops["load_A"].access.tile_shape == (128, 64) and ops["load_A"].engine == "tma"
    assert ops["gemm_A_s_B_s"].spec.m == 128 and ops["gemm_A_s_B_s"].spec.k == 64 and ops["gemm_A_s_B_s"].spec.n == 128
    assert ops["store_C"].access.tile_shape == (128, 128) and ops["store_C"].source == "register"
    pipeline = launch.loop("k").body
    assert [a.name for a in pipeline.actors] == ["producer", "tensor"]
    assert dict((a.name, a.resource) for a in pipeline.actors) == {"producer": "tma", "tensor": "tensor"}
    slots = {s.buffer: s for s in pipeline.buffer_slots}
    assert slots["A_s"].capacity == 3 and slots["A_s"].acquire.op_id == "load_A" and slots["A_s"].release.op_id == "gemm_A_s_B_s"
    assert pipeline.carries[0].storage == "acc" and pipeline.carries[0].distance == 1
    assert all(op.timing is None for op in launch.ops())
    result = sight.analyze(program, ARCH, options=sight.Options(cache="fast"))
    assert result.regions["main/k"].ii is not None and not result.diagnostics.unsupported


def test_builder_rejects_ambiguous_descriptions():
    kb, A, B, C, A_s, B_s, acc = _gemm_builder()
    plain = kb.fragment("plain", (128, 128), "fp32")
    with kb.loop("k", 4, stages=2):
        kb.copy(A["m", "k"], A_s)
        kb.copy(B["k", "n"], B_s)
        kb.gemm(A_s, B_s, plain, name="g1")
        kb.gemm(A_s, B_s, plain, name="g2")
    with pytest.raises(sight.ContractError, match="written by .* declare it carried=True"):
        kb.program()
    kb2 = sight.KernelBuilder("main", grid={"m": 2})
    X = kb2.tensor("X", (256, 256), "bf16")
    X_s = kb2.shared("X_s", (128, 128), "bf16")
    with kb2.loop("outer", 2, stages=2):
        with kb2.loop("inner", 2):
            kb2.copy(X["m", "inner"], X_s)
    with pytest.raises(sight.ContractError, match="pipelined loop cannot contain loop"):
        kb2.program()
    kb3 = sight.KernelBuilder("main", grid={"m": 2})
    a = kb3.fragment("a", (8,), "fp32")
    b = kb3.fragment("b", (8,), "bf16")
    with pytest.raises(sight.ContractError, match="use cast"):
        kb3.copy(a, b)
    with pytest.raises(sight.ContractError, match="reduction extents"):
        kb, A, B, C, A_s, B_s, acc = _gemm_builder()
        kb.gemm(A_s, A_s, acc)


def test_builder_hints_and_in_place_chain():
    kb = sight.KernelBuilder("main", grid={"i": 2}, threads=128)
    X = kb.tensor("X", (2, 256, 64), "fp32")
    X_s = kb.shared("X_s", (64, 64), "fp32", stages=2)
    x_r = kb.fragment("x_r", (64, 64), "fp32")
    acc = kb.fragment("acc", (64, 64), "fp32", carried=True)
    with kb.loop("k", 4, stages=2) as loop:
        kb.copy(X["i", tiles("k"), :], X_s, engine="cp_async")
        kb.copy(X_s, x_r, name="to_reg", actor="compute")
        kb.elementwise("scale", [acc], acc, (("mul", 1.0, "cuda"),))
        kb.elementwise("accumulate", [x_r], acc, (("add", 1.0, "cuda"),))
        loop.after("accumulate", "to_reg", distance=1)
    pipeline = kb.launch().loop("k").body
    names = {d.name for d in pipeline.dependencies}
    assert "inplace:acc:scale->accumulate" in names and "hint:accumulate->to_reg" in names
    assert pipeline.carries[0].source.op_id == "accumulate" and pipeline.carries[0].target.op_id == "scale"
    result = sight.analyze(kb.program(), ARCH, options=sight.Options(cache="fast", ii_mode="periodic_best"))
    starts = dict(result.regions["main/k"].groups[0].ii.witness)
    latency = {c.op_id: c.completion_latency_s for c in result.regions["main/k"].groups[0].completion_trace}
    assert starts["accumulate"] + 1e-15 >= starts["scale"] + latency["scale"]
    assert starts["accumulate"] + 1e-15 >= starts["to_reg"] + latency["to_reg"]


@pytest.mark.parametrize("builder", [
    lambda: ex.gemm_program(m=512, n=512, k=256), lambda: ex.elementwise_add_program(m=1000, n=512),
    lambda: ex.reduce_sum_program(rows=256), lambda: ex.rms_norm_program(rows=256),
    lambda: ex.fa3_program(seq_len=512, heads=2), lambda: ex.fa3_program(seq_len=512, heads=2, causal=True),
    lambda: ex.fa4_program(seq_len=512, heads=2), lambda: ex.flashmla_decode_program(batch=2, kv_len=512, num_splits=2),
])
def test_examples_have_no_hand_written_timing(builder):
    program = builder()
    for launch in program.launches:
        for op in launch.ops():
            assert op.timing is None and op.latency_extra_s is None
        assert dict(launch.metadata)["tilelang_commit"] == ex.TILELANG_COMMIT


def test_parameter_change_recomputes_service_and_ii():
    import copy

    program = ex.gemm_program(m=512, n=512, k=1024)
    base = sight.analyze(program, ARCH, options=sight.Options(cache="fast"))
    slow = copy.deepcopy(ARCH)
    slow.fp16_tensor_flops /= 4.0
    changed = sight.analyze(program, slow, options=sight.Options(cache="fast"))
    assert dict(changed.ops["main/k/mma"].cost_groups[0].service_s)["tensor"] == pytest.approx(
        4 * dict(base.ops["main/k/mma"].cost_groups[0].service_s)["tensor"])
    assert changed.regions["main/k"].ii.resource_ii_s > base.regions["main/k"].ii.resource_ii_s
    table = sight.LatencyTable((sight.LatencyEntry("*", "load", engine="tma", latency=1.0, unit="us", source_file="fixture.csv"),))
    latency = sight.analyze(program, ARCH, options=sight.Options(cache="fast"), context=sight.Context(latency_table=table))
    assert latency.ops["main/k/load_A"].cost_groups[0].latency_extra_s == pytest.approx(1e-6)
    assert latency.regions["main/k"].ii.credit_ii_s > base.regions["main/k"].ii.credit_ii_s


def test_review6_buffer_lifetime_waits_for_every_reader():
    """Single-slot SMEM buffer with a slow and a fast reader in different actors."""

    kb = sight.KernelBuilder("main", grid={"i": 1}, threads=128)
    X = kb.tensor("X", (4096, 64), "fp32")
    X_s = kb.shared("X_s", (64, 64), "fp32", stages=1)
    slow_out = kb.fragment("slow_out", (64, 64), "fp32")
    fast_out = kb.fragment("fast_out", (64, 64), "fp32")
    with kb.loop("k", 64, stages=1):
        kb.copy(X[tiles("k"), :], X_s, engine="cp_async")
        kb.elementwise("slow", [X_s], slow_out, (("exp2", 8.0, "sfu"),), actor="slow_actor")
        kb.elementwise("fast", [X_s], fast_out, (("add", 1.0, "cuda"),), actor="fast_actor")  # declared last
    program = kb.program()
    pipeline = program.launches[0].loop("k").body
    slot = pipeline.buffer_slots[0]
    assert slot.release.op_id == "release_X_s"
    waits = {d.source.op_id for d in pipeline.dependencies if d.target.op_id == "release_X_s"}
    assert waits == {"slow", "fast"}
    for mode in ("periodic_best", "periodic_worst"):
        result = sight.analyze(program, ARCH, options=sight.Options(cache="fast", ii_mode=mode))
        group = result.regions["main/k"].groups[0]
        starts = dict(group.ii.witness)
        latency = {c.op_id: c.completion_latency_s for c in group.completion_trace}
        next_write = starts["load_X"] + group.ii.selected_ii_s
        for reader in ("slow", "fast"):
            finished = starts[reader] + latency[reader]
            assert next_write >= finished * (1 - 1e-9), (mode, reader, next_write, finished)
        assert latency["slow"] > latency["fast"]


def test_review6_copy_handles_dtype_conversion_and_access_range():
    kb = sight.KernelBuilder("main", grid={"i": 1})
    T = kb.tensor("T", (64,), "bf16")
    s_bf = kb.shared("s_bf", (64,), "bf16")
    r_fp = kb.fragment("r_fp", (64,), "fp32")
    t_out = kb.tmem("t_out", (64,), "bf16")
    kb.copy(T[tiles("i")], s_bf)
    kb.copy(s_bf, r_fp, name="to_reg")       # bf16 smem -> fp32 registers
    kb.copy(r_fp, t_out, name="to_tmem")     # fp32 registers -> bf16 tmem
    program = kb.program()
    assert [op.op_id for op in program.launches[0].ops()] == ["load_T", "to_reg", "to_reg_convert", "to_tmem_convert", "to_tmem"]
    ops = {op.op_id: op for op in program.launches[0].ops()}
    assert ops["to_reg"].bytes == 128 and ops["to_tmem"].bytes == 128  # the bf16 representation travels
    from tilesight.arch.b200 import B200
    result = sight.analyze(program, B200(), options=sight.Options(cache="fast"))
    move = result.ops["main/to_reg"].traffic_total
    assert move.smem_read_bytes == 128 and move.register_write_bytes == 128
    convert = result.ops["main/to_reg_convert"]
    assert convert.executed_flops == 64 and "cuda" in dict(convert.cost_groups[0].service_s)
    assert result.ops["main/to_tmem"].traffic_total.tmem_write_bytes == 128
    # tensor <-> buffer dtype mismatch, register-to-register, reshaping copies: rejected
    kb2 = sight.KernelBuilder("main", grid={"i": 1})
    wrong = kb2.fragment("wrong", (64,), "fp32")
    with pytest.raises(sight.ContractError, match="dtype bf16 != fp32"):
        kb2.copy(kb2.tensor("T", (64,), "bf16")[tiles("i")], wrong)
    with pytest.raises(sight.ContractError, match="element counts differ"):
        kb2.copy(kb2.shared("a", (64,), "fp32"), kb2.fragment("b", (32,), "fp32"))
    with pytest.raises(sight.ContractError, match="needs an explicit register stage"):
        kb2.copy(kb2.shared("c", (8,), "bf16"), kb2.tmem("d", (8,), "fp32"))
    # a tensor smaller than the tile gets a valid range; extents are checked against the axes
    kb3 = sight.KernelBuilder("main", grid={"i": 1})
    op = kb3.copy(kb3.tensor("small", (4,), "fp32")[tiles("i")], kb3.fragment("big", (1024,), "fp32"))
    assert op.access.valid_shape == (4,) and op.access.payload_bytes == 16 and op.access.executed_bytes == 4096
    kb4 = sight.KernelBuilder("main", grid={"i": 2})
    with pytest.raises(sight.ContractError, match="has 2 iterations but the tensor needs 1 tiles"):
        kb4.copy(kb4.tensor("small", (4,), "fp32")[tiles("i")], kb4.fragment("big", (1024,), "fp32"))
    kb5 = sight.KernelBuilder("main", grid={"i": 2})
    with pytest.raises(sight.ContractError, match="':' on K dim 1 has extent 96"):
        kb5.copy(kb5.tensor("K", (2, 96), "fp32")["i", :], kb5.fragment("k_r", (64,), "fp32"))
    with pytest.raises(sight.ContractError, match="pick by"):
        kb5.copy(kb5.tensor("P", (3, 64), "fp32")["i", :], kb5.fragment("p_r", (64,), "fp32"))


def test_review6_tail_tiles_are_derived_from_tensor_shapes():
    kb = sight.KernelBuilder("main", grid={"n": 2, "m": 2})
    A = kb.tensor("A", (100, 96), "fp16")
    out = kb.tensor("B", (100, 96), "fp16")
    a_r = kb.fragment("a_r", (64, 64), "fp16")
    b_r = kb.fragment("b_r", (64, 64), "fp16")
    kb.copy(A["m", "n"], a_r, name="load_A")
    kb.elementwise("scale", [a_r], b_r, (("mul", 1.0, "cuda"),))
    kb.copy(b_r, out["m", "n"], name="store_B")
    units = kb.launch().work_units
    groups = dict(units.groups)
    assert set(groups) == {"full", "tail_m", "tail_n", "tail_mn"}
    assert groups["tail_mn"].effective_fraction("load_A") == pytest.approx((36 / 64) * (32 / 64))
    assert groups["tail_m"].effective_fraction("scale") == pytest.approx(36 / 64)  # flows through the data
    assert [r.group_id for r in units.order] == ["full", "tail_n", "tail_m", "tail_mn"]
    result = sight.analyze(kb.program(), ARCH, options=sight.Options(cache="fast"))
    assert result.ops["main/load_A"].useful_bytes == pytest.approx(100 * 96 * 2)
    assert result.ops["main/load_A"].traffic_total.payload_read_bytes == pytest.approx(100 * 96 * 2)
    assert result.ops["main/scale"].useful_flops == pytest.approx(100 * 96)
    with pytest.raises(sight.ContractError, match="pass tails='declared'"):
        kb.launch(work_units=sight.WorkUnits((("g", sight.WorkGroup()),), (sight.WorkRun("g", 4),)))


def test_review7_small_tensor_range_is_applied_once():
    kb = sight.KernelBuilder("main", grid={"i": 1})
    small = kb.tensor("small", (4,), "fp32")
    big = kb.fragment("big", (1024,), "fp32")
    out = kb.fragment("out", (1024,), "fp32")
    op = kb.copy(small[tiles("i")], big, name="load_small")
    kb.elementwise("scale", [big], out, (("mul", 1.0, "cuda"),))
    assert op.access.valid_shape == (4,) and op.access.tensor_shape == (4,)
    assert "load_small" not in kb.tail_fractions()          # the range already lives in valid_shape
    assert kb.tail_fractions()["scale"] == {"i": 4 / 1024}   # the compute has no valid_shape: fraction applies
    result = sight.analyze(kb.program(), ARCH, options=sight.Options(cache="fast"))
    load = result.ops["main/load_small"]
    assert load.useful_bytes == 16 and load.traffic_total.payload_read_bytes == 16 and load.executed_bytes == 4096
    assert result.ops["main/scale"].useful_flops == 4 and result.ops["main/scale"].executed_flops == 1024


def test_review7_range_propagation_overwrite_and_reduced_axes():
    kb = sight.KernelBuilder("main", grid={"m": 2})
    partial = kb.tensor("T1", (228,), "fp32")
    full = kb.tensor("T2", (256,), "fp32")
    buffer = kb.fragment("buf", (128,), "fp32")
    out = kb.fragment("o", (128,), "fp32")
    kb.copy(partial[tiles("m")], buffer, name="load_partial")
    kb.copy(full[tiles("m")], buffer, name="load_full")       # an overwrite replaces the valid range
    kb.elementwise("use", [buffer], out, (("add", 1.0, "cuda"),))
    assert kb.tail_fractions() == {"load_partial": {"m": 100 / 128}}
    result = sight.analyze(ex.gemm_program(m=128, n=128, k=65), ARCH, options=sight.Options(cache="fast"))
    assert result.ops["main/k/mma"].useful_flops == pytest.approx(2 * 128 * 128 * 65)
    assert result.ops["main/k/cast_C"].useful_flops == 128 * 128   # the reduced K tail does not reach the output
    assert result.ops["main/k/store_C"].useful_bytes == 128 * 128 * 2
    # a reduction removes the reduced axis from the output range but not from its own work
    kb2 = sight.KernelBuilder("main", grid={"row": 1})
    x = kb2.tensor("X", (32, 1000), "fp32")
    x_s = kb2.shared("X_s", (32, 512), "fp32", stages=2)
    acc = kb2.fragment("acc", (32, 1), "fp32", carried=True)
    total = kb2.fragment("total", (32, 1), "fp32")
    with kb2.loop("k", 2, stages=2) as loop:
        kb2.copy(x[tiles("row"), tiles("k")], x_s, engine="cp_async")
        kb2.reduce("sum", x_s, acc, axis=1)
        with loop.epilogue():
            kb2.elementwise("finish", [acc], total, (("mul", 1.0, "cuda"),))
    fractions = kb2.tail_fractions()
    assert fractions["sum"] == {"k": 488 / 512} and "finish" not in fractions


def test_review7_repeated_build_is_idempotent():
    kb = sight.KernelBuilder("main", grid={"i": 1})
    X = kb.tensor("X", (640, 64), "fp32")
    shared = kb.shared("s", (64, 64), "fp32")
    a = kb.fragment("a", (64, 64), "fp32")
    b = kb.fragment("b", (64, 64), "fp32")
    with kb.loop("k", 10, stages=1):
        kb.copy(X[tiles("k"), :], shared, engine="cp_async")
        kb.elementwise("r1", [shared], a, (("add", 1.0, "cuda"),), actor="x")
        kb.elementwise("r2", [shared], b, (("add", 1.0, "cuda"),), actor="y")
    first, second = kb.program(), kb.program()
    assert first == second and sight.program_to_snapshot(first) == sight.program_to_snapshot(second)
    assert [op.op_id for op in first.launches[0].ops()].count("release_s") == 1
    kb.sync("release_other")  # user ids may still be added; the derived id is reserved deterministically
    clash = sight.KernelBuilder("main", grid={"i": 1})
    Y = clash.tensor("Y", (640, 64), "fp32")
    t = clash.shared("t", (64, 64), "fp32")
    with clash.loop("k", 10, stages=1):
        clash.sync("release_t")
        clash.copy(Y[tiles("k"), :], t, engine="cp_async")
        clash.elementwise("r1", [t], clash.fragment("c", (64, 64), "fp32"), (("add", 1.0, "cuda"),), actor="x")
        clash.elementwise("r2", [t], clash.fragment("d", (64, 64), "fp32"), (("add", 1.0, "cuda"),), actor="y")
    with pytest.raises(sight.ContractError, match="reserved for the derived lifetime barrier"):
        clash.program()


def test_block_mapper_interface_drives_cache_and_tail_groups():
    from tilesight.arch.h200_sxm import H200_SXM

    assert sight.PanelSwizzle(2, 1).coordinates((2, 4)) == [(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (0, 3), (1, 2), (1, 3)]
    reverse = sight.ExplicitOrder.from_function("reverse", lambda e: reversed(sight.LinearBlockId().coordinates(e)), (2, 2))
    assert reverse.coordinates((2, 2)) == [(1, 1), (1, 0), (0, 1), (0, 0)]
    with pytest.raises(sight.ContractError, match="not a permutation"):
        sight.ExplicitOrder(((0, 0), (0, 1))).coordinates((2, 2))
    with pytest.raises(sight.ContractError, match="not a permutation"):
        ex.gemm_program(m=512, n=512, k=256, mapper=reverse)   # a 2x2 order on a 4x4 grid is rejected
    reverse = sight.ExplicitOrder.from_function("reverse", lambda e: reversed(sight.LinearBlockId().coordinates(e)), (4, 4))
    program = ex.gemm_program(m=512, n=512, k=256, mapper=reverse)
    assert sight.program_from_snapshot(sight.program_to_snapshot(program)) == program
    assert program.launches[0].instance_coordinates()[0] == (3, 3)
    # tail groups follow the mapper's order
    tail = ex.gemm_program(m=500, n=512, k=256, mapper=sight.PanelSwizzle(row_panel=4, column_panel=1))
    order = [run.group_id for run in tail.launches[0].work_units.order]
    assert order == ["full", "tail_m"]
    # A panel swizzle changes B reuse (and therefore time), not the amount of work.
    arch = H200_SXM().set_to_microbench()
    shape = dict(m=16384, n=57344, k=14336, block_m=128, block_n=256, block_k=64, stages=4, dtype="bf16")
    options = sight.Options(cache="fast", cache_sample_budget=256)
    linear = sight.analyze(ex.gemm_program(**shape), arch, options=options)
    panel = sight.analyze(ex.gemm_program(mapper=sight.PanelSwizzle(8, 128, "along_m"), **shape), arch, options=options)
    assert panel.ops["main/k/mma"].executed_flops == linear.ops["main/k/mma"].executed_flops
    miss = lambda r: [a for a in r.cache[0].per_access if a.op_path.endswith("load_B")][0].ddr_miss_rate
    assert miss(linear) > 0.95 and miss(panel) < 0.2
    assert panel.launches["main"].kernel_body_s < 0.7 * linear.launches["main"].kernel_body_s
    # an arbitrary mapper is consumed by the fast path too: same order, waves of `slots` units
    small = sight.analyze(program, arch, options=options)
    assert dict(small.launches["main"].provenance)["cache_path"] == "reduction_axis"
    assert any("launch order from block mapper explicit_order, waves of 16" in a for a in small.cache[0].assumptions)
