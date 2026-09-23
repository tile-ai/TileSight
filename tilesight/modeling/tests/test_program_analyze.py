"""Generic analyze: costs, cost groups, work groups, launches, cache binding."""

import math

import pytest

from tilesight.arch.h100_sxm import H100_SXM
from tilesight.modeling import program as sight
from tilesight.modeling.ir import ResourceTiming, Timing
from tilesight.modeling.ops.schema import GemmOpSpec
from tilesight.modeling.program import examples as ex
from tilesight.modeling.program.results import scale_traffic, traffic_close


ARCH = H100_SXM()
FAST = sight.Options(ii_mode="resource_ii", cache="fast", cache_sample_budget=128)


def _check_conservation(result):
    for path, op in result.ops.items():
        assert op.count == sum(g.count for g in op.cost_groups), path
        total = sight.Traffic()
        for g in op.cost_groups:
            total = total.add(scale_traffic(g.traffic_per_execution, float(g.count)))
        assert traffic_close(total, op.traffic_total), path
    for path, region in result.regions.items():
        assert math.isclose(region.total_s, region.first_s + region.steady_s + region.drain_s, rel_tol=1e-9)
        for group in region.groups:
            assert math.isclose(group.total_s, group.first_s + group.steady_s + group.drain_s, rel_tol=1e-9)
    for name, launch in result.launches.items():
        assert math.isclose(launch.total_s, launch.kernel_body_s + launch.launch_overhead_s + launch.host_dispatch_s)
    assert math.isclose(
        result.program.total_s,
        sum(l.total_s for l in result.launches.values()) + result.program.inter_launch_gap_s,
    )


@pytest.mark.parametrize("mode", ["resource_ii", "periodic_best", "periodic_worst"])
def test_gemm_single_launch(mode):
    program = ex.gemm_program(m=2048, n=2048, k=1024)
    result = sight.analyze(program, ARCH, options=sight.Options(ii_mode=mode, cache="fast", cache_sample_budget=128))
    _check_conservation(result)
    assert not result.diagnostics.unsupported
    op = result.op("main/k/load_A")
    assert op.count == 16 * 16 * 16
    assert op.traffic_total.ddr_read_bytes > 0
    assert op.cost_groups[0].traffic_source.startswith("cache_fast")
    store = result.op("main/k/store_C")
    assert store.count == 256 and store.region_site == "k:epilogue"
    assert store.traffic_total.ddr_write_bytes == pytest.approx(2048 * 2048 * 2)
    region = result.region("main/k")
    assert region.ii is not None and region.ii.mode == mode
    assert region.ii.resource_ii_s <= region.ii.selected_ii_s + 1e-15
    if mode == "resource_ii":
        assert region.ii.witness is None and region.ii.scope == "resource_ii_lower_bound"
    else:
        assert region.ii.witness is not None and region.ii.search_complete is not None
    launch = result.launch("main")
    assert launch.work_units == 256 and launch.slots == 132 and launch.waves == 2
    assert launch.tail_work_units == 256 - 132
    assert launch.spatial_scope == "list_schedule_full_wave_costs"
    peak = 2.0 * 2048 * 2048 * 1024 / ARCH.fp16_tensor_flops
    assert launch.kernel_body_s >= peak


def test_cost_group_key_and_service_breakdown():
    result = sight.analyze(ex.gemm_program(m=1024, n=1024, k=512), ARCH, options=FAST)
    mma = result.op("main/k/mma")
    group = mma.cost_groups[0]
    assert group.key.work_group_id == "default" and group.key.region_site == "k:body"
    assert group.key.wave_kind == "full" and len(group.key.binding_context_id) == 16
    assert dict(group.service_s)["tensor"] == pytest.approx(
        2 * 128 * 128 * 64 / (ARCH.fp16_tensor_flops / ARCH.sm_count)
    )
    assert group.completion_latency_s == pytest.approx(dict(group.service_s)["tensor"])
    assert group.latency_extra_s == 0.0
    load = result.op("main/k/load_A").cost_groups[0]
    assert set(dict(load.service_s)) >= {"ddr", "l2", "tma"}
    assert dict(load.provenance)["engine_service"].startswith("proxy:smem_port_bandwidth")


def test_causal_work_groups_bind_trip_counts_and_counts():
    program = ex.fa3_program(seq_len=1024, heads=4, causal=True)
    result = sight.analyze(program, ARCH, options=FAST)
    _check_conservation(result)
    q_tiles = 8
    load_k = result.op("main/kv/load_K")
    assert len(load_k.cost_groups) == q_tiles
    assert load_k.count == sum((i + 1) * 4 for i in range(q_tiles))
    load_q = result.op("main/kv/load_Q")
    assert load_q.count == q_tiles * 4 and load_q.region_site == "kv:prologue"
    region = result.region("main/kv")
    assert {g.work_group_id: g.trip_count for g in region.groups} == {"q%d_kv%d" % (i, i + 1): i + 1 for i in range(q_tiles)}
    qk = result.op("main/kv/gemm_QK")
    assert qk.useful_flops < qk.executed_flops
    launch = result.launch("main")
    assert launch.critical_group_id == "q7_kv8"
    assert region.group("q0_kv1").total_s < region.group("q7_kv8").total_s


def test_review6_causal_groups_follow_grid_coordinates_and_masks():
    program = ex.fa3_program(seq_len=512, heads=2, causal=True)
    launch = program.launches[0]
    assert launch.work_axis_names == ("batch", "kv_head", "hg", "q")  # q is the fastest axis
    units = launch.work_units
    sequence = units.instance_group_ids()
    trips = [units.group(g).trip_count("kv") for g in sequence]
    assert trips == [1, 2, 3, 4, 1, 2, 3, 4]  # every head walks its q tiles in order
    # Equal trip counts with different masks must not be merged.
    ragged = ex.fa3_program(seq_len=256, heads=1, block_m=64, block_n=128, causal=True)
    groups = ragged.launches[0].work_units.groups
    assert [g.trip_count("kv") for _gid, g in groups] == [1, 1, 2, 2]
    assert len({g.effective_fraction("gemm_QK") for _gid, g in groups}) == 4
    result = sight.analyze(ragged, ARCH, options=FAST)
    assert result.ops["main/kv/gemm_QK"].useful_flops == pytest.approx(8421376)
    valid_scores = sum(min(r + 1, 256) for r in range(256))
    assert result.ops["main/kv/gemm_QK"].useful_flops == pytest.approx(2 * valid_scores * 128)


def test_zero_trip_loop_and_explicit_timing_override():
    op = sight.Compute("c", GemmOpSpec(m=8, n=8, k=8, compute_dtype="fp16"), actor="t",
                    timing=Timing(10e-9, (ResourceTiming("tensor", 8e-9),)))
    tail = sight.Compute("e", GemmOpSpec(m=8, n=8, k=8, compute_dtype="fp16"), actor="t", timing=Timing(5e-9))
    loop = sight.Loop("k", 0, body=op, epilogue=tail)
    program = sight.Program((sight.Launch("main", loop, (sight.WorkAxis("m", 3),)),))
    result = sight.analyze(program, ARCH, options=sight.Options(cache="none"))
    assert result.op("main/k/c").count == 0
    assert result.op("main/k/e").count == 3
    assert result.op("main/k/e").cost_groups[0].traffic_source == "explicit_timing_override"
    assert result.region("main/k").total_s == pytest.approx(5e-9)
    assert result.launch("main").kernel_body_s == pytest.approx(5e-9)


def test_missing_traffic_policy_and_explicit_assumption():
    load = sight.Load("l", "global", "smem", "tma", bytes=4096, actor="p")
    program = sight.Program((sight.Launch("main", load, (sight.WorkAxis("m", 4),)),))
    with pytest.raises(sight.UnsupportedError, match="no traffic"):
        sight.analyze(program, ARCH, options=sight.Options(cache="none"))
    result = sight.analyze(program, ARCH, options=sight.Options(cache="none", missing_traffic_policy="assume_miss"))
    op = result.op("main/l")
    assert op.traffic_total.ddr_read_bytes == pytest.approx(4 * 4096)
    assert op.cost_groups[0].traffic_source == "assumed_all_miss_no_cache_model"
    assumption = sight.TrafficAssumption(sight.Traffic(payload_read_bytes=4096, l2_read_request_bytes=4096, ddr_read_bytes=1024), source="ncu_fixture")
    result = sight.analyze(program, ARCH, options=sight.Options(cache="none"),
                        context=sight.Context(traffic_assumptions={"main/l": assumption}))
    op = result.op("main/l")
    assert op.traffic_total.ddr_read_bytes == pytest.approx(4 * 1024)
    assert op.cost_groups[0].traffic_source == "explicit_assumption:main/l:ncu_fixture"


def test_trace_mode_is_exact_and_supports_loops():
    elementwise = ex.elementwise_add_program(m=512, n=512)
    result = sight.analyze(elementwise, ARCH, options=sight.Options(cache="trace"))
    assert result.cache and result.cache[0].exact
    assert result.cache[0].reduction_fidelity == "exact_instance_trace"
    gemm = sight.analyze(ex.gemm_program(m=512, n=512, k=512), ARCH, options=sight.Options(cache="trace"))
    assert gemm.cache[0].exact and gemm.launches["main"].provenance
    assert dict(gemm.launches["main"].provenance)["cache_path"] == "explicit_trace"
    load_a = gemm.ops["main/k/load_A"]
    # A is reused across the 4 N tiles of one row: first touch only.
    assert load_a.traffic_total.ddr_read_bytes == pytest.approx(512 * 512 * 2, rel=0.2)


def test_elementwise_tail_group_reports_useful_versus_padded():
    result = sight.analyze(ex.elementwise_add_program(m=1000, n=512), ARCH, options=FAST)
    _check_conservation(result)
    add = result.op("main/add")
    assert len(add.cost_groups) == 2
    assert add.useful_flops < add.executed_flops
    assert add.count == math.ceil(1000 / 64) * 2
    launch = result.launch("main")
    assert {g.group_id for g in launch.groups} == {"full", "tail_m"}
    load = result.op("main/load_A")
    assert load.useful_bytes == pytest.approx(1000 * 512 * 2)
    assert load.executed_bytes == pytest.approx(1024 * 512 * 2)


def test_reduction_examples_share_representative_iteration():
    result = sight.analyze(ex.rms_norm_program(rows=512), ARCH, options=FAST)
    _check_conservation(result)
    assert result.op("main/k/load_X_pass1").count == result.op("main/k2/load_X_pass2").count
    assert any("share one representative iteration" in item for item in result.diagnostics.approximations)
    assert result.region("main/k").ii is not None and result.region("main/k2").ii is not None
    pure = sight.analyze(ex.reduce_sum_program(rows=512), ARCH, options=FAST)
    _check_conservation(pure)
    assert pure.op("main/k/store_sum").count == 16


def test_sequential_launches_sum_with_explicit_gap():
    program = ex.flashmla_decode_program(batch=2, kv_len=1024, num_splits=2)
    result = sight.analyze(program, ARCH, options=FAST)
    _check_conservation(result)
    assert result.program.launch_order == ("main", "combine")
    assert result.op("combine/split/load_partial").count == 2 * 2 * 2
    assert result.op("main/kv/load_KV").count == 2 * 2 * 2 * (1024 // 64 // 2)
    gapped = sight.Program(program.launches, edges=(sight.LaunchEdge("main", "combine", gap_s=3e-6),), name=program.name)
    result2 = sight.analyze(gapped, ARCH, options=FAST)
    assert result2.program.total_s == pytest.approx(result.program.total_s + 3e-6)
    assert result2.program.inter_launch_gap_s == pytest.approx(3e-6)
    assert result.launch("main").launch_overhead_s == pytest.approx(2e-6)


def test_persistent_scheduler_uses_declared_partitions():
    program = ex.gemm_program(m=512, n=512, k=256)
    launch = program.launches[0]
    # 16 work units on 2 persistent CTAs, 8 each.
    persistent = sight.Launch(
        launch.name, launch.body, launch.work_axes, threads=launch.threads,
        scheduler=sight.Persistent((((0, 8),), ((8, 16),))),
    )
    result = sight.analyze(sight.Program((persistent,)), ARCH, options=FAST)
    static = sight.analyze(program, ARCH, options=FAST)
    launch_result = result.launch("main")
    per_unit = launch_result.groups[0].per_unit_s
    assert launch_result.kernel_body_s == pytest.approx(8 * per_unit)
    assert launch_result.scheduler == "persistent"
    assert launch_result.active_sms == 2 and launch_result.slots == 2
    # Two persistent CTAs share the pooled DDR/L2 links two ways instead of 16 ways; the
    # per-SM L1.5 share does not grow, so a unit is never slower and its DDR service shrinks.
    assert per_unit <= static.launch("main").groups[0].per_unit_s
    ddr = lambda r: dict(r.op("main/k/load_A").cost_groups[0].service_s)["ddr"]
    l1_5 = lambda r: dict(r.op("main/k/load_A").cost_groups[0].service_s)["l1_5"]
    assert ddr(result) == pytest.approx(ddr(static) * 2 / 16)
    assert l1_5(result) == pytest.approx(l1_5(static))
    assert static.launch("main").kernel_body_s == pytest.approx(static.launch("main").groups[0].per_unit_s)


def test_review_ragged_groups_use_real_instance_sequence():
    """Two instances (1 and 8 trips) reading shared X[k] 1 KiB tiles: 8 KiB first touch."""

    x = sight.TileAccess("X", sight.Projection(("k",)), (512,), 2)
    load = sight.Load("load_X", "global", "smem", "tma", access=x, actor="p")
    loop = sight.Loop("k", sight.FromWorkGroup(), body=load)
    units = sight.WorkUnits(groups=(("short", sight.WorkGroup({"k": 1})), ("long", sight.WorkGroup({"k": 8}))),
                         order=(sight.WorkRun("short", 1), sight.WorkRun("long", 1)))
    program = sight.Program((sight.Launch("main", loop, (sight.WorkAxis("i", 2),), work_units=units),))
    result = sight.analyze(program, ARCH, options=sight.Options(cache="fast"))
    op = result.ops["main/k/load_X"]
    assert op.traffic_total.ddr_read_bytes == pytest.approx(8 * 1024)
    per_group = {g.key.work_group_id: g for g in op.cost_groups}
    assert per_group["short"].traffic_per_execution.ddr_read_bytes == pytest.approx(1024)
    assert per_group["long"].traffic_per_execution.ddr_read_bytes == pytest.approx(7 * 1024 / 8)
    assert any("fell back to the explicit instance trace" in item for item in result.diagnostics.approximations)


def test_review_loop_invariant_access_reuses_allocation():
    x = sight.TileAccess("X", sight.Projection(()), (512,), 2)
    load = sight.Load("load_X", "global", "smem", "tma", access=x, actor="p")
    program = sight.Program((sight.Launch("main", sight.Loop("k", 8, body=load), (sight.WorkAxis("i", 1),)),))
    for mode in ("fast", "trace"):
        result = sight.analyze(program, ARCH, options=sight.Options(cache=mode))
        op = result.ops["main/k/load_X"]
        assert op.count == 8
        assert op.traffic_total.ddr_read_bytes == pytest.approx(1024)
        # Every execution is a real request; hits are absorbed by L1.5 on H100.
        assert op.traffic_total.l1_5_request_bytes == pytest.approx(8 * 1024)
        assert op.traffic_total.payload_read_bytes == pytest.approx(8 * 1024)


def test_review_persistent_shares_bandwidth_by_active_ctas():
    big = sight.Load("load", "global", "smem", "tma", bytes=1 << 20, actor="p")
    launch = sight.Launch("main", big, (sight.WorkAxis("i", 16),), scheduler=sight.Persistent((((0, 16),),)))
    result = sight.analyze(sight.Program((launch,)), ARCH,
                        options=sight.Options(cache="none", missing_traffic_policy="assume_miss"))
    launch_result = result.launches["main"]
    assert launch_result.active_sms == 1 and launch_result.slots == 1
    ddr = dict(result.ops["main/load"].cost_groups[0].service_s)["ddr"]
    assert ddr == pytest.approx((1 << 20) / ARCH.ddr_bandwidth)
    assert launch_result.kernel_body_s == pytest.approx(16 * result.ops["main/load"].cost_groups[0].completion_latency_s)


def test_review_traffic_override_keeps_cache_history():
    x = sight.TileAccess("X", sight.Projection(("i",)), (512,), 2)
    first = sight.Load("l1", "global", "smem", "tma", access=x, actor="p")
    second = sight.Load("l2", "global", "smem", "tma", access=x, actor="p")
    program = sight.Program((sight.Launch("main", sight.Sequence((first, second)), (sight.WorkAxis("i", 1),)),))
    base = sight.analyze(program, ARCH, options=sight.Options(cache="fast"))
    assert base.ops["main/l2"].traffic_total.ddr_read_bytes == 0
    same = base.ops["main/l1"].cost_groups[0].traffic_per_execution
    context = sight.Context(traffic_assumptions={"main/l1": sight.TrafficAssumption(same, source="fixture")})
    calibrated = sight.analyze(program, ARCH, options=sight.Options(cache="fast"), context=context)
    assert calibrated.ops["main/l2"].traffic_total.ddr_read_bytes == 0
    assert calibrated.ops["main/l1"].cost_groups[0].traffic_source.startswith("explicit_assumption")
    assert any("stays in cache history" in item for item in calibrated.diagnostics.approximations)


def test_review_ddr_service_counts_rfo_and_flush_once():
    store = sight.Store("st", "smem", "global", "tma", bytes=1 << 20, actor="p")
    program = sight.Program((sight.Launch("main", store, (sight.WorkAxis("i", 1),)),))
    traffic = sight.Traffic(payload_write_bytes=1 << 20, l2_write_request_bytes=1 << 20, ddr_read_bytes=1 << 20,
                         rfo_bytes=1 << 20, ddr_write_bytes=1 << 20, dirty_created_bytes=1 << 20,
                         dirty_terminal_flush_bytes=1 << 20)
    result = sight.analyze(program, ARCH, options=sight.Options(cache="none"),
                        context=sight.Context(traffic_assumptions={"main/st": sight.TrafficAssumption(traffic, "fixture")}))
    ddr = dict(result.ops["main/st"].cost_groups[0].service_s)["ddr"]
    assert ddr == pytest.approx(2 * (1 << 20) / ARCH.ddr_bandwidth)


def test_review_completion_trace_records_every_execution():
    op = sight.Compute("c", GemmOpSpec(m=8, n=8, k=8, compute_dtype="fp16"), actor="t",
                    timing=Timing(10e-9, (ResourceTiming("tensor", 5e-9, offset=5e-9),)))
    sync = sight.Sync("bar", actor="t")
    program = sight.Program((sight.Launch("main", sight.Loop("k", 3, body=sight.Sequence((op, sync))), (sight.WorkAxis("i", 1),)),))
    result = sight.analyze(program, ARCH, options=sight.Options(cache="none"))
    completions = {x.op_id: x for x in result.regions["main"].groups[0].completion_trace}
    assert completions["c"].start_s == 0.0 and completions["c"].repeats == (("k", 3, 10e-9),)
    assert completions["bar"].start_s == pytest.approx(10e-9)
    services = result.regions["main"].groups[0].service_trace
    assert services[0].offset_s == pytest.approx(5e-9)


def test_review_zero_trip_loop_skips_cost_binding():
    load = sight.Load("l", "global", "smem", "tma", bytes=4096, actor="p")
    sync = sight.Sync("bar", actor="p")
    program = sight.Program((sight.Launch("main", sight.Loop("k", 0, body=load, epilogue=sync), (sight.WorkAxis("i", 2),)),))
    result = sight.analyze(program, ARCH, options=sight.Options(cache="none"))
    op = result.ops["main/k/l"]
    assert op.count == 0 and op.traffic_total.ddr_read_bytes == 0
    assert op.cost_groups[0].traffic_source == "not_executed"
    assert result.ops["main/k/bar"].count == 2


def test_review_completion_latency_rows_are_converted():
    spec = GemmOpSpec(m=128, n=128, k=64, compute_dtype="fp16")
    program = sight.Program((sight.Launch("main", sight.Compute("c", spec, actor="t"), (sight.WorkAxis("i", 1),)),))
    table = sight.LatencyTable((sight.LatencyEntry("*", "compute", latency=500, unit="ns",
                                             measured_quantity="completion_latency", source_file="bench.csv"),))
    result = sight.analyze(program, ARCH, options=sight.Options(cache="none"), context=sight.Context(latency_table=table))
    group = result.ops["main/c"].cost_groups[0]
    assert group.completion_latency_s == pytest.approx(500e-9)
    assert group.latency_extra_s == pytest.approx(500e-9 - group.service_total_s)
    assert "completion_latency-service" in dict(group.provenance)["latency_extra_source"]
    short = sight.LatencyTable((sight.LatencyEntry("*", "compute", latency=10, unit="ns", measured_quantity="completion_latency"),))
    result = sight.analyze(program, ARCH, options=sight.Options(cache="none"), context=sight.Context(latency_table=short))
    assert result.ops["main/c"].cost_groups[0].latency_extra_s == 0.0
    with pytest.raises(sight.ContractError, match="measured_quantity"):
        sight.LatencyEntry("*", "compute", latency=1, measured_quantity="round_trip")


def test_result_indexes_support_mapping_access():
    result = sight.analyze(ex.gemm_program(m=512, n=512, k=256), ARCH, options=FAST)
    assert result.ops["main/k/load_A"] is result.op("main/k/load_A")
    assert result.regions["main/k"].kind == "loop"
    assert "main" in result.launches and list(result.launches) == ["main"]
    assert dict(result.ops).keys() == set(result.ops)
    with pytest.raises(KeyError):
        result.ops["main/k/missing"]


def test_latency_table_changes_credit_bound_ii():
    program = ex.gemm_program(m=512, n=512, k=2048)
    base = sight.analyze(program, ARCH, options=FAST)
    table = sight.LatencyTable((sight.LatencyEntry("*", "load", engine="tma", latency=2.0, unit="us", source_file="microbench.csv"),))
    slow = sight.analyze(program, ARCH, options=FAST, context=sight.Context(latency_table=table, label="latency_fixture"))
    base_region = base.region("main/k")
    slow_region = slow.region("main/k")
    assert slow.op("main/k/load_A").cost_groups[0].latency_extra_s == pytest.approx(2e-6)
    assert slow_region.ii.credit_ii_s > base_region.ii.credit_ii_s
    assert slow_region.ii.selected_ii_s > base_region.ii.selected_ii_s
    assert slow_region.ii.resource_ii_s == pytest.approx(base_region.ii.resource_ii_s)
    assert slow.launch("main").kernel_body_s > base.launch("main").kernel_body_s
    assert dict(slow.op("main/k/load_A").cost_groups[0].provenance)["latency_extra_source"] == "latency_table:microbench.csv"


def test_clock_override_and_residency_override_are_recorded():
    program = ex.gemm_program(m=512, n=512, k=256)
    base = sight.analyze(program, ARCH, options=FAST)
    context = sight.Context(clock_overrides={"sm": ARCH.max_freq / 2}, residency_overrides={"main": 2}, label="calib")
    calibrated = sight.analyze(program, ARCH, options=FAST, context=context)
    base_mma = dict(base.op("main/k/mma").cost_groups[0].service_s)["tensor"]
    cal_mma = dict(calibrated.op("main/k/mma").cost_groups[0].service_s)["tensor"]
    assert cal_mma == pytest.approx(base_mma * 4)  # half clock x two resident CTAs
    launch = calibrated.launch("main")
    assert launch.residency == 2 and launch.residency_source == "context_override" and launch.slots == 264
    assert any("clock:sm" in item for item in calibrated.diagnostics.overridden_parameters)
    assert base.input_digest != calibrated.input_digest


def test_scenarios_policy_reports_shared_computation():
    result = sight.analyze(ex.gemm_program(m=512, n=512, k=256), ARCH,
                        options=sight.Options(cache="fast", cache_bound_policy="scenarios", cache_sample_budget=64))
    assert dict(result.scenarios).keys() == {"low", "high"}
    assert result.scenario("low").program.total_s == pytest.approx(result.program.total_s)
    assert any("share one computation" in note for note in result.diagnostics.notes)
    snapshot = result.to_snapshot()
    assert snapshot["schema"] == "tilesight.analysis_result/2"
    assert len(snapshot["scenarios"]) == 2


def test_resident_sharing_schedule_is_rejected_for_multi_residency():
    program = ex.gemm_program(m=512, n=512, k=256)
    with pytest.raises(sight.UnsupportedError, match="resident_sharing='schedule'"):
        sight.analyze(program, ARCH, options=sight.Options(cache="fast", resident_sharing="schedule"),
                   context=sight.Context(residency_overrides={"main": 2}))


def test_unsupported_compute_throughput_names_the_op():
    op = sight.Compute("c", GemmOpSpec(m=8, n=8, k=8, compute_dtype="fp64"), actor="t")
    program = sight.Program((sight.Launch("main", op, (sight.WorkAxis("m", 1),)),))
    with pytest.raises(sight.UnsupportedError, match="main/c"):
        sight.analyze(program, ARCH, options=sight.Options(cache="none"))


# ---------------------------------------------------------------------------
# Review round 2
# ---------------------------------------------------------------------------


def test_review2_fallback_sampling_is_stratified_and_seeded():
    x = sight.TileAccess("X", sight.Projection(("k",)), (512,), 2)
    load = sight.Load("load_X", "global", "smem", "tma", access=x, actor="p")
    loop = sight.Loop("k", sight.FromWorkGroup(), body=load)
    units = sight.WorkUnits(groups=(("a", sight.WorkGroup({"k": 256})), ("b", sight.WorkGroup({"k": 512}))),
                         order=(sight.WorkRun("a", 1), sight.WorkRun("b", 1)))
    program = sight.Program((sight.Launch("main", loop, (sight.WorkAxis("i", 2),), work_units=units),))
    exact = sight.analyze(program, ARCH, options=sight.Options(cache="trace")).ops["main/k/load_X"].traffic_total
    assert exact.ddr_read_bytes == pytest.approx(512 * 1024)
    for seed in (0, 7):
        sampled = sight.analyze(program, ARCH, options=sight.Options(cache="fast", cache_seed=seed))
        op = sampled.ops["main/k/load_X"]
        assert op.count == 768
        assert op.traffic_total.ddr_read_bytes == pytest.approx(exact.ddr_read_bytes, rel=0.15)
        report = sampled.cache[0]
        assert report.sampled_requests < report.represented_requests
        assert any("stratified systematic sampling per (op, work group), seed %d" % seed in a for a in report.assumptions)
        per_group = {g.key.work_group_id: g for g in op.cost_groups}
        assert per_group["a"].count == 256 and per_group["b"].count == 512


def test_review2_homogeneous_groups_get_per_group_attribution():
    x = sight.TileAccess("X", sight.Projection(()), (512,), 2)
    load = sight.Load("load_X", "global", "smem", "tma", access=x, actor="p")
    units = sight.WorkUnits(groups=(("a", sight.WorkGroup()), ("b", sight.WorkGroup())),
                         order=(sight.WorkRun("a", 1), sight.WorkRun("b", 1)))
    program = sight.Program((sight.Launch("main", load, (sight.WorkAxis("i", 2),), work_units=units),))
    result = sight.analyze(program, ARCH, options=sight.Options(cache="fast", cache_sample_budget=None))
    per_group = {g.key.work_group_id: g for g in result.ops["main/load_X"].cost_groups}
    assert per_group["a"].traffic_per_execution.ddr_read_bytes == pytest.approx(1024)
    assert per_group["b"].traffic_per_execution.ddr_read_bytes == 0.0
    assert per_group["a"].key.binding_context_id != per_group["b"].key.binding_context_id
    assert dict(result.launches["main"].provenance)["cache_path"] == "explicit_trace"


def test_review2_zero_trip_loop_keeps_epilogue_traffic():
    x = sight.TileAccess("X", sight.Projection(("i", "k")), (512,), 2)
    y = sight.TileAccess("Y", sight.Projection(("i",)), (512,), 2)
    load = sight.Load("load_X", "global", "smem", "tma", access=x, actor="p")
    store = sight.Store("store_Y", "smem", "global", "tma", access=y, actor="p")
    program = sight.Program((sight.Launch("main", sight.Loop("k", 0, body=load, epilogue=store), (sight.WorkAxis("i", 2),)),))
    result = sight.analyze(program, ARCH, options=sight.Options(cache="fast"))
    assert result.ops["main/k/load_X"].count == 0
    assert result.ops["main/k/load_X"].cost_groups[0].traffic_source == "not_executed"
    assert result.ops["main/k/store_Y"].traffic_total.ddr_write_bytes == pytest.approx(2 * 1024)
    assert any("never executed" in a for a in result.diagnostics.approximations)
    assert not result.diagnostics.unsupported


def test_review2_finite_witness_completion_uses_real_event_starts():
    load = sight.Load("ld", "global", "smem", "tma", bytes=1024, actor="p", writes=("buf",),
                   timing=Timing(4e-9, (ResourceTiming("tma", 4e-9),)))
    compute = sight.Compute("cp", GemmOpSpec(m=8, n=8, k=8, compute_dtype="fp16"), actor="t", reads=("buf",),
                         timing=Timing(6e-9, (ResourceTiming("tensor", 6e-9),)))
    pipeline = sight.Pipeline(actors=(sight.Actor("p", (load,)), sight.Actor("t", (compute,))),
                           buffers=(sight.Buffer("buf", "smem", (8,), "fp16", slots=2),),
                           buffer_slots=(sight.BufferSlots("buf", sight.start(load), sight.done(compute), 2),))
    program = sight.Program((sight.Launch("main", sight.Loop("k", 8, body=pipeline, boundary_anchor_op="cp"), (sight.WorkAxis("i", 1),)),))
    result = sight.analyze(program, ARCH, options=sight.Options(
        cache="none", ii_mode="periodic_best", boundary_policy="finite_witness", missing_traffic_policy="assume_miss"))
    group = result.regions["main/k"].groups[0]
    assert group.strategy == "periodic_finite_witness"
    events = group.completion_trace
    assert len(events) == 16 and all(not e.repeats for e in events)
    last = max(e.start_s + e.completion_latency_s for e in events)
    assert last == pytest.approx(group.total_s)
    assert min(e.start_s for e in events) == 0.0


def test_read_write_split_directions_and_shares():
    q = sight.TileAccess("Q", sight.Projection(("i",)), (512, 1024), 2)  # 1 MiB tile
    load = sight.Load("load_Q", "global", "smem", "tma", access=q, actor="p")
    store = sight.Store("store_Q", "smem", "global", "tma", access=q, actor="p")
    stage = sight.Load("to_reg", "smem", "register", "ldst", bytes=1 << 20, actor="p")
    body = sight.Sequence((load, stage, store))
    program = sight.Program((sight.Launch("main", body, (sight.WorkAxis("i", 4),)),))
    result = sight.analyze(program, ARCH, options=sight.Options(cache="trace"))
    mib = 4 << 20
    ld = result.ops["main/load_Q"].traffic_total
    assert ld.l2_read_request_bytes == pytest.approx(mib) and ld.l2_write_request_bytes == 0.0
    assert ld.smem_write_bytes == pytest.approx(mib) and ld.smem_read_bytes == 0.0
    assert ld.ddr_read_bytes == pytest.approx(mib) and ld.ddr_write_bytes == 0.0
    st = result.ops["main/store_Q"].traffic_total
    assert st.smem_read_bytes == pytest.approx(mib) and st.smem_write_bytes == 0.0
    assert st.l2_write_request_bytes == pytest.approx(mib) and st.l2_read_request_bytes == 0.0
    assert st.ddr_write_bytes == pytest.approx(mib) and st.ddr_read_bytes == 0.0
    assert st.l2_request_bytes == st.l2_read_request_bytes + st.l2_write_request_bytes
    reg = result.ops["main/to_reg"].traffic_total
    assert reg.smem_read_bytes == pytest.approx(mib) and reg.register_write_bytes == pytest.approx(mib)
    assert reg.ddr_read_bytes == 0.0 and reg.l2_request_bytes == 0.0
    assert dict(result.ops["main/to_reg"].cost_groups[0].service_s).keys() >= {"smem", "register"}
    # launch/program totals are field-wise sums of the ops
    total = ld.add(st).add(reg)
    assert result.launches["main"].traffic_total.close_to(total)
    assert result.program.traffic_total.close_to(total)
    assert result.regions["main"].traffic_total.close_to(total)
    # shares: same level, same direction, same scope; zero denominators are None with a reason
    share = result.ops["main/load_Q"].share("smem_write_bytes", "launch")
    assert share.share == pytest.approx(1.0) and share.denominator_id == "main"
    assert result.ops["main/load_Q"].share("smem_read_bytes", "launch").share == 0.0
    tmem = result.ops["main/load_Q"].share("tmem_read_bytes", "program")
    assert tmem.share is None and "denominator is zero" in tmem.reason
    rows = result.ops["main/store_Q"].breakdown("program")
    assert ("l2", "write", pytest.approx(mib), pytest.approx(1.0)) in [(l, d, b, s) for l, d, b, s in rows]
    snapshot = result.to_snapshot()
    assert snapshot["schema"] == "tilesight.analysis_result/2"
    assert snapshot["ops"][0][1]["traffic_shares"]


def test_read_write_split_hits_reduce_lower_reads_not_endpoint_writes():
    x = sight.TileAccess("X", sight.Projection(()), (512,), 2)
    load = sight.Load("load_X", "global", "smem", "tma", access=x, actor="p")
    program = sight.Program((sight.Launch("main", sight.Loop("k", 4, body=load), (sight.WorkAxis("i", 1),)),))
    result = sight.analyze(program, ARCH, options=sight.Options(cache="trace"))
    t = result.ops["main/k/load_X"].traffic_total
    assert t.smem_write_bytes == pytest.approx(4 * 1024)
    assert t.ddr_read_bytes == pytest.approx(1024)
    assert t.l1_5_read_request_bytes == pytest.approx(4 * 1024)
    with pytest.raises(Exception, match="rfo_bytes"):
        sight.Traffic(rfo_bytes=10.0)
    assert sight.Traffic.from_cache_access(
        __import__("tilesight.modeling.cache.result", fromlist=["CacheTraffic"]).CacheTraffic(l2_request_bytes=5.0), "write"
    ).l2_write_request_bytes == 5.0


# ---------------------------------------------------------------------------
# Review round 3
# ---------------------------------------------------------------------------


def test_review3_unknown_compute_traffic_propagates_to_shares_and_totals():
    result = sight.analyze(ex.gemm_program(m=512, n=512, k=256), ARCH, options=FAST)
    mma = result.ops["main/k/mma"]
    assert mma.traffic_total.value("register_read_bytes") is None
    assert mma.traffic_total.value("ddr_read_bytes") == 0.0
    assert mma.cost_groups[0].traffic_source == "compute_internal_unknown"
    share = mma.share("register_read_bytes", "launch")
    assert share.share is None and share.numerator_bytes is None and "numerator unknown" in share.reason
    # The launch total inherits the unknown, so even a known load cannot claim a share.
    load = result.ops["main/k/load_A"]
    assert load.traffic_total.value("smem_write_bytes") > 0
    load_share = load.share("smem_write_bytes", "launch")
    assert load_share.share is None and "denominator unknown" in load_share.reason
    assert result.launches["main"].traffic_total.value("smem_write_bytes") is None
    assert result.program.traffic_total.value("smem_write_bytes") is None
    assert load.share("ddr_read_bytes", "launch").share is not None
    snapshot = result.to_snapshot()
    op_dict = dict(snapshot["ops"])["main/k/mma"]
    assert op_dict["traffic_total"]["register_read_bytes"] is None
    assert op_dict["traffic_total"]["unknown"]
    with pytest.raises(Exception, match="marked unknown but carries a value"):
        sight.Traffic(smem_read_bytes=1.0, unknown=(("smem_read_bytes", "x"),))


def test_review3_cache_report_matches_priced_traffic_and_keeps_model_prediction():
    x = sight.TileAccess("X", sight.Projection(("i",)), (512,), 2)
    load = sight.Load("l", "global", "smem", "tma", access=x, actor="p")
    program = sight.Program((sight.Launch("main", load, (sight.WorkAxis("i", 1),)),))
    base = sight.analyze(program, ARCH, options=sight.Options(cache="fast"))
    row = base.cache[0].per_access[0]
    assert row.traffic_per_request.smem_write_bytes == pytest.approx(1024)
    assert row.interval_per_request.low.close_to(row.traffic_per_request)
    assert row.traffic_per_request.close_to(base.ops["main/l"].cost_groups[0].traffic_per_execution)
    assert not row.calibrated and row.model_traffic_per_request is None
    assumption = sight.TrafficAssumption(sight.Traffic(payload_read_bytes=1024, l2_read_request_bytes=1024), "ncu")
    calibrated = sight.analyze(program, ARCH, options=sight.Options(cache="fast"),
                            context=sight.Context(traffic_assumptions={"main/l": assumption}))
    row = calibrated.cache[0].per_access[0]
    assert row.calibrated and row.traffic_per_request.ddr_read_bytes == 0.0
    assert row.model_traffic_per_request.ddr_read_bytes == pytest.approx(1024)
    assert row.traffic_per_request.close_to(calibrated.ops["main/l"].cost_groups[0].traffic_per_execution)
    assert dict(calibrated.cache[0].per_tensor)["X"]["ddr_read_bytes"] == 0.0
    assert any("finally priced traffic" in a for a in calibrated.cache[0].assumptions)


def test_review3_on_chip_traffic_override_is_consumed():
    move = sight.Load("mv", "smem", "register", "ldst", bytes=1024, actor="p")
    program = sight.Program((sight.Launch("main", move, (sight.WorkAxis("i", 1),)),))
    base = sight.analyze(program, ARCH, options=sight.Options(cache="none"))
    assert base.ops["main/mv"].traffic_total.smem_read_bytes == pytest.approx(1024)
    assumption = sight.TrafficAssumption(sight.Traffic(smem_read_bytes=2048, register_write_bytes=2048), "fixture")
    calibrated = sight.analyze(program, ARCH, options=sight.Options(cache="none"),
                            context=sight.Context(traffic_assumptions={"main/mv": assumption}))
    op = calibrated.ops["main/mv"]
    assert op.traffic_total.smem_read_bytes == pytest.approx(2048)
    assert op.cost_groups[0].traffic_source.startswith("explicit_assumption")
    assert dict(op.cost_groups[0].service_s)["smem"] == pytest.approx(2 * dict(base.ops["main/mv"].cost_groups[0].service_s)["smem"])


def test_review3_legacy_traffic_totals_need_a_direction():
    with pytest.raises(Exception, match="no read/write direction"):
        sight.Traffic.from_dict({"l2_request_bytes": 4096})
    assert sight.Traffic.from_dict({"l2_request_bytes": 4096}, mode="write").l2_write_request_bytes == 4096
    with pytest.raises(Exception, match="disagrees"):
        sight.Traffic.from_dict({"l2_request_bytes": 4096, "l2_read_request_bytes": 1.0})
    consistent = sight.Traffic.from_dict({"l2_request_bytes": 4096, "l2_read_request_bytes": 4096})
    assert consistent.l2_request_bytes == 4096
    round_trip = sight.Traffic.from_dict(sight.Traffic.compute_internal().to_dict())
    assert round_trip.value("smem_read_bytes") is None
    assert sight.Traffic.from_dict({"l2_request_bytes": 0}).l2_request_bytes == 0


def test_review3_sampling_weights_and_ratio_rates():
    """Weights are the sampling period; rates are one ratio estimate per stratum."""

    from tilesight.modeling.program import explicit_trace as module

    x = sight.TileAccess("X", sight.Projection(()), (512,), 2)
    load = sight.Load("l", "global", "smem", "tma", access=x, actor="p")
    program = sight.Program((sight.Launch("main", sight.Loop("k", 5, body=load), (sight.WorkAxis("i", 1),)),))
    exact = sight.analyze(program, ARCH, options=sight.Options(cache="trace")).ops["main/k/l"].traffic_total.ddr_read_bytes
    full = sight.analyze(program, ARCH, options=sight.Options(cache="fast", cache_sample_budget=None)).ops["main/k/l"].traffic_total
    assert full.ddr_read_bytes == pytest.approx(exact)
    plan = module._sampling_plan({("l", "g"): 5}, 2, 0)
    assert plan[("l", "g")][0] == 3 and plan[("l", "g")][2] == 3.0
    original = module._sampling_plan
    try:
        for phase in range(3):
            module._sampling_plan = lambda counts, budget, seed, _phase=phase: {key: (3, _phase, 3.0) for key in counts}
            result = sight.analyze(program, ARCH, options=sight.Options(cache="fast", cache_sample_budget=2))
            row = result.cache[0].per_access[0]
            assert row.l1_5_hit_rate + row.l2_served_rate + row.ddr_miss_rate == pytest.approx(1.0)
            assert 0.0 <= row.ddr_miss_rate <= 1.0
            assert any("ratio estimates" in a for a in result.cache[0].assumptions)
    finally:
        module._sampling_plan = original


def test_review3_finite_witness_service_trace_uses_real_starts():
    load = sight.Load("ld", "global", "smem", "tma", bytes=1024, actor="p", writes=("buf",),
                   timing=Timing(4e-9, (ResourceTiming("tma", 4e-9),)))
    compute = sight.Compute("cp", GemmOpSpec(m=8, n=8, k=8, compute_dtype="fp16"), actor="t", reads=("buf",),
                         timing=Timing(6e-9, (ResourceTiming("tensor", 6e-9),)))
    pipeline = sight.Pipeline(actors=(sight.Actor("p", (load,)), sight.Actor("t", (compute,))),
                           buffers=(sight.Buffer("buf", "smem", (8,), "fp16", slots=2),),
                           buffer_slots=(sight.BufferSlots("buf", sight.start(load), sight.done(compute), 2),))
    program = sight.Program((sight.Launch("main", sight.Loop("k", 8, body=pipeline, boundary_anchor_op="cp"), (sight.WorkAxis("i", 1),)),))
    result = sight.analyze(program, ARCH, options=sight.Options(
        cache="none", ii_mode="periodic_best", boundary_policy="finite_witness", missing_traffic_policy="assume_miss"))
    group = result.regions["main/k"].groups[0]
    assert len(group.service_trace) == 16 and not any(item.repeats for item in group.service_trace)
    last_service = max(item.offset_s + item.service_time_s for item in group.service_trace)
    last_completion = max(item.start_s + item.completion_latency_s for item in group.completion_trace)
    assert last_service == pytest.approx(group.total_s) and last_completion == pytest.approx(group.total_s)
    trace = sight.perfetto_trace(result)
    bars = [e for e in trace["traceEvents"] if e["ph"] == "X" and e["cat"] in ("service", "completion")]
    assert max(e["ts"] + e["dur"] for e in bars) == pytest.approx(group.total_s * 1e6)


def test_review3_unused_work_group_is_not_cost_bound():
    x = sight.TileAccess("X", sight.Projection(("k",)), (512,), 2)
    loop = sight.Loop("k", sight.FromWorkGroup(), body=sight.Load("l", "global", "smem", "tma", access=x, actor="p"))
    units = sight.WorkUnits(groups=(("used", sight.WorkGroup({"k": 0})), ("unused", sight.WorkGroup({"k": 1}))),
                         order=(sight.WorkRun("used", 2),))
    program = sight.Program((sight.Launch("main", loop, (sight.WorkAxis("i", 2),), work_units=units),))
    result = sight.analyze(program, ARCH, options=sight.Options(cache="fast"))
    op = result.ops["main/k/l"]
    assert op.count == 0 and all(g.traffic_source == "not_executed" for g in op.cost_groups)
    assert [g.group_id for g in result.launches["main"].groups] == ["used"]
    assert any("has no instances" in a for a in result.diagnostics.approximations)


def test_review3_digests_include_architecture_parameters():
    import copy

    slower = copy.deepcopy(ARCH)
    slower.ddr_bandwidth /= 100.0
    program = ex.gemm_program(m=512, n=512, k=256)
    base = sight.analyze(program, ARCH, options=FAST)
    other = sight.analyze(program, slower, options=FAST)
    assert base.input_digest != other.input_digest
    assert base.ops["main/k/load_A"].cost_groups[0].key.binding_context_id != other.ops["main/k/load_A"].cost_groups[0].key.binding_context_id
    assert dict(base.diagnostics.provenance)["arch_digest"] != dict(other.diagnostics.provenance)["arch_digest"]
    again = sight.analyze(program, copy.deepcopy(ARCH), options=FAST)
    assert again.input_digest == base.input_digest


def test_review3_tail_tiles_report_useful_versus_padded_work():
    result = sight.analyze(ex.gemm_program(m=129, n=128, k=64), ARCH, options=FAST)
    mma = result.ops["main/k/mma"]
    assert mma.executed_flops == 2 * 2 * 128 * 128 * 64
    assert mma.useful_flops == pytest.approx(2 * 129 * 128 * 64)
    store = result.ops["main/k/store_C"]
    assert store.executed_bytes == 2 * 128 * 128 * 2
    assert store.useful_bytes == pytest.approx(129 * 128 * 2)
    load_b = result.ops["main/k/load_B"]
    assert load_b.useful_bytes == load_b.executed_bytes
    both = sight.analyze(ex.gemm_program(m=129, n=130, k=64), ARCH, options=FAST)
    assert {g.group_id for g in both.launches["main"].groups} == {"full", "tail_m", "tail_n", "tail_mn"}
    assert both.ops["main/k/mma"].useful_flops == pytest.approx(2 * 129 * 130 * 64)


# ---------------------------------------------------------------------------
# Review round 4
# ---------------------------------------------------------------------------


def test_review4_unknown_traffic_is_rejected_for_pricing_and_never_read_as_zero():
    x = sight.TileAccess("X", sight.Projection(("i",)), (512,), 2)
    load = sight.Load("l", "global", "smem", "tma", access=x, actor="p")
    program = sight.Program((sight.Launch("main", load, (sight.WorkAxis("i", 1),)),))
    unknown_ddr = sight.Traffic(payload_read_bytes=1024, l2_read_request_bytes=1024).with_unknown(["ddr_read_bytes"], "not measured")
    with pytest.raises(sight.UnsupportedError, match="unknown traffic"):
        sight.analyze(program, ARCH, options=sight.Options(cache="none"),
                   context=sight.Context(traffic_assumptions={"main/l": sight.TrafficAssumption(unknown_ddr, "x")}))
    partial = sight.Traffic(l2_write_request_bytes=7.0).with_unknown(["l2_read_request_bytes"], "r")
    assert partial.l2_request_bytes is None and partial.value("l2_request_bytes") is None
    assert partial.to_dict()["l2_request_bytes"] is None
    assert partial.level_bytes("l2") is None
    imported = sight.Traffic.from_dict({"ddr_read_bytes": None, "ddr_write_bytes": 3.0})
    assert imported.value("ddr_read_bytes") is None and imported.ddr_write_bytes == 3.0
    result = sight.analyze(ex.gemm_program(m=512, n=512, k=256), ARCH, options=FAST)
    rows = {(level, direction): value for level, direction, value, _share in result.ops["main/k/mma"].breakdown("launch")}
    assert rows[("register", "read")] is None and rows[("ddr", "read")] == 0.0
    move = sight.Load("mv", "smem", "register", "ldst", bytes=1024, actor="p")
    program = sight.Program((sight.Launch("main", move, (sight.WorkAxis("i", 1),)),))
    unknown_port = sight.Traffic().with_unknown(["smem_read_bytes"], "not modelled")
    with pytest.raises(sight.UnsupportedError, match="endpoint traffic is unknown"):
        sight.analyze(program, ARCH, options=sight.Options(cache="none"),
                   context=sight.Context(traffic_assumptions={"main/mv": sight.TrafficAssumption(unknown_port, "x")}))


def test_review4_tail_payload_reaches_traffic_and_k_tail_is_covered():
    for mode in ("fast", "trace"):
        result = sight.analyze(ex.gemm_program(m=129, n=128, k=64), ARCH, options=sight.Options(cache=mode))
        store = result.ops["main/k/store_C"]
        assert store.traffic_total.payload_write_bytes == pytest.approx(129 * 128 * 2)
        assert store.traffic_total.payload_write_bytes == pytest.approx(store.useful_bytes)
        assert store.traffic_total.ddr_write_bytes == pytest.approx(2 * 128 * 128 * 2)
        tail = [g for g in store.cost_groups if g.key.work_group_id == "tail_m"][0]
        assert tail.traffic_per_execution.payload_write_bytes == pytest.approx(128 * 2)
        assert tail.traffic_per_execution.transaction_amplification_bytes == pytest.approx(128 * 128 * 2 - 128 * 2)
    k_tail = sight.analyze(ex.gemm_program(m=128, n=128, k=65), ARCH, options=FAST)
    assert k_tail.ops["main/k/mma"].useful_flops == pytest.approx(2 * 128 * 128 * 65)
    assert k_tail.ops["main/k/mma"].executed_flops == pytest.approx(2 * 128 * 128 * 128)
    assert k_tail.ops["main/k/load_A"].traffic_total.payload_read_bytes == pytest.approx(128 * 65 * 2)
    assert k_tail.ops["main/k/load_A"].traffic_total.ddr_read_bytes == pytest.approx(128 * 128 * 2)
    assert k_tail.ops["main/k/store_C"].useful_bytes == k_tail.ops["main/k/store_C"].executed_bytes


def test_review4_sampled_rates_and_bytes_are_one_estimate():
    x = sight.TileAccess("X", sight.Projection(()), (512,), 2)
    program = sight.Program((sight.Launch("main", sight.Loop("k", 5, body=sight.Load("l", "global", "smem", "tma", access=x, actor="p")), (sight.WorkAxis("i", 1),)),))
    for seed in (0, 1, 2, 3):
        result = sight.analyze(program, ARCH, options=sight.Options(cache="fast", cache_sample_budget=2, cache_seed=seed))
        row = result.cache[0].per_access[0]
        op = result.ops["main/k/l"]
        assert row.request_count == 5 and op.count == 5
        assert op.traffic_total.payload_read_bytes == pytest.approx(5 * 1024)
        assert op.traffic_total.l1_5_read_request_bytes == pytest.approx(5 * 1024)
        assert op.traffic_total.ddr_read_bytes == pytest.approx(row.ddr_miss_rate * 5 * 1024)
        assert op.traffic_total.l2_read_request_bytes == pytest.approx((1.0 - row.l1_5_hit_rate) * 5 * 1024)
        assert 0.0 <= row.ddr_miss_rate <= 1.0 and 0.0 <= row.l1_5_hit_rate <= 1.0


def test_review4_override_hit_rates_are_consumed_or_unknown():
    x = sight.TileAccess("X", sight.Projection(("i",)), (512,), 2)
    program = sight.Program((sight.Launch("main", sight.Load("l", "global", "smem", "tma", access=x, actor="p"), (sight.WorkAxis("i", 1),)),))
    traffic = sight.Traffic(payload_read_bytes=1024, l2_read_request_bytes=1024)
    with_rate = sight.analyze(program, ARCH, options=sight.Options(cache="fast"),
                           context=sight.Context(traffic_assumptions={"main/l": sight.TrafficAssumption(traffic, "ncu", hit_rate_l2=1.0)}))
    row = with_rate.cache[0].per_access[0]
    assert row.calibrated and row.l2_served_rate == 1.0 and row.ddr_miss_rate == 0.0
    assert dict(row.model_rates)["ddr_miss_rate"] == pytest.approx(1.0)
    assert row.rates_source.startswith("explicit_assumption:hit_rate_l2")
    obs = sight.Observation("main", "l2_hit_rate", 0.9, "ratio", "kernel", "ncu", denominator="cache_model_read_requests")
    comparison = sight.compare(with_rate, with_rate, [obs])
    assert comparison.metrics[0].comparable and comparison.metrics[0].baseline == 1.0
    without = sight.analyze(program, ARCH, options=sight.Options(cache="fast"),
                         context=sight.Context(traffic_assumptions={"main/l": sight.TrafficAssumption(traffic, "ncu")}))
    row = without.cache[0].per_access[0]
    assert row.l2_served_rate is None and row.ddr_miss_rate is None and row.rates_source.startswith("unknown")
    assert dict(without.cache[0].per_tensor)["X"]["requests_with_unknown_rates"] == 1.0
    comparison = sight.compare(without, without, [obs])
    assert not comparison.metrics[0].comparable and "hit rate unknown" in comparison.metrics[0].reason


# ---------------------------------------------------------------------------
# Review round 6
# ---------------------------------------------------------------------------


def test_review6_cuda_instruction_units_and_semantic_flops_are_separate():
    kb = sight.KernelBuilder("main", grid={"i": 1})
    a = kb.fragment("a", (1024,), "fp32")
    outs = [kb.fragment(n, (1024,), "fp32") for n in ("b", "c", "d")]
    kb.elementwise("e_add", [a], outs[0], (("add", 1.0, "cuda"),))
    kb.elementwise("e_fma", [a], outs[1], (("fma", 1.0, "cuda"),))
    kb.elementwise("e_mul_add", [a], outs[2], (("mul", 1.0, "cuda"), ("add", 1.0, "cuda")))
    result = sight.analyze(kb.program(), ARCH, options=sight.Options(cache="none"))
    service = {p.split("/")[-1]: dict(o.cost_groups[0].service_s)["cuda"] for p, o in result.ops.items()}
    flops = {p.split("/")[-1]: o.executed_flops for p, o in result.ops.items()}
    unit = 1024 / (ARCH.fp32_cuda_core_flops / 2.0 / ARCH.sm_count)  # FMA-peak FLOP/s -> instruction/s
    assert service["e_add"] == pytest.approx(unit)
    assert service["e_fma"] == pytest.approx(unit)          # a fused FMA is one instruction unit
    assert service["e_mul_add"] == pytest.approx(2 * unit)  # separate mul + add are two
    assert (flops["e_add"], flops["e_fma"], flops["e_mul_add"]) == (1024, 2048, 2048)
    provenance = dict(result.ops["main/e_fma"].cost_groups[0].provenance)
    assert any("cuda_instruction/s" in item and "/ 2" in item for item in provenance["throughput_sources"])
    demand = provenance["issue_demand"][0]
    assert demand[0] == "fma" and demand[2] == "cuda_instruction" and demand[3] == 1024 and demand[4] == 2048
    # SFU keeps its own class and resource.
    kb2 = sight.KernelBuilder("main", grid={"i": 1})
    x = kb2.fragment("x", (256,), "fp32"); y = kb2.fragment("y", (256,), "fp32")
    kb2.elementwise("e_exp", [x], y, (("exp2", 1.0, "sfu"),))
    sfu = sight.analyze(kb2.program(), ARCH, options=sight.Options(cache="none")).ops["main/e_exp"]
    assert set(dict(sfu.cost_groups[0].service_s)) == {"sfu"}
    assert dict(sfu.cost_groups[0].service_s)["sfu"] == pytest.approx(256 / (ARCH.sfu_flops / ARCH.sm_count))
