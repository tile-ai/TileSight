"""§9.1: the ``fast`` GEMM cache problem is the same source as the general GEMM helper."""

import pytest

from tilesight.arch.h100_sxm import H100_SXM
from tilesight.modeling import program as sight
from tilesight.modeling.cache import SamplingConfig, gemm_cache_problem, model_cache
from tilesight.modeling.program import examples as ex
from tilesight.modeling.program.cache_binding import cache_problem_for

ARCH = H100_SXM()


def _strip_names(canonical):
    for access in canonical["accesses"]:
        access["name"] = "_"
    return canonical


def test_fast_gemm_cache_problem_matches_general_gemm_helper():
    block_m, block_n, block_k = 128, 128, 64
    m, n, k = 4 * block_m, ARCH.sm_count * block_n, 8 * block_k  # grid_n == sm_count -> row-major waves
    unit = float(block_m * block_k * 2)
    budget = None  # full observation: sampling phase is hashed from access names
    program = ex.gemm_program(m=m, n=n, k=k, block_m=block_m, block_n=block_n, block_k=block_k)
    options = sight.Options(cache="fast", cache_sample_budget=budget, cache_reuse_unit_bytes=unit)
    launch = program.launches[0]
    ours = cache_problem_for(launch, ARCH, options, slots=ARCH.sm_count, sm_count=ARCH.sm_count)
    reference = gemm_cache_problem(
        m, n, k, block_m, block_n, block_k,
        l2_capacity_bytes=ARCH.l2_capacity, sm_count=ARCH.sm_count,
        mem_levels={"in1": [1, 0, 0, 2], "in2": [1, 0, 0, 2], "out1": [1, 0, 0, 2]},
        row_panel=n // block_n, column_panel=1,
        l1_5_group_size=ARCH.l1_5_group_size, l1_5_capacity_per_group=ARCH.l1_5_capacity_per_group,
        reuse_unit_bytes=unit, reduction_mode="stable_shadow_cohort", representative_k=0,
        sampling=SamplingConfig(seed=0, sample_budget=budget),
    )
    ours_canonical = _strip_names(ours.canonical())
    reference_canonical = _strip_names(reference.canonical())
    # Traversals differ in spelling (row-major vs. degenerate panel) but visit the same order.
    ours_canonical.pop("traversal")
    reference_canonical.pop("traversal")
    assert ours_canonical == reference_canonical
    assert ours.reuse_digest != reference.reuse_digest  # names differ; nothing else does

    ours_result = model_cache(ours)
    reference_result = model_cache(reference)
    for mine, theirs in (("load_A", "A"), ("load_B", "B"), ("store_C", "C")):
        a = ours_result.by_name(mine)
        b = reference_result.by_name(theirs)
        assert a.request_count == b.request_count
        assert a.l1_5_hit_rate == b.l1_5_hit_rate
        assert a.l2_served_rate == b.l2_served_rate
        assert a.ddr_miss_rate == b.ddr_miss_rate
        assert a.traffic == b.traffic
    assert ours_result.histogram.exact is False  # stable shadow: representative K only

    # And analyze() prices exactly that CacheResult: per-execution = per-request average.
    result = sight.analyze(program, ARCH, options=options)
    report = result.cache[0]
    assert report.problem_digest == ours.digest
    for mine, theirs in (("load_A", "A"), ("load_B", "B")):
        access = [x for x in report.per_access if x.op_path == "main/k/" + mine][0]
        b = reference_result.by_name(theirs)
        assert access.traffic_per_request.ddr_read_bytes == pytest.approx(b.traffic.ddr_read_bytes / b.request_count)
        assert access.l2_served_rate == pytest.approx(b.l2_served_rate)
        op = result.ops["main/k/" + mine]
        # loads execute grid x K times; the representative-K model represents grid requests
        assert op.count == b.request_count * (k // block_k)
        assert op.traffic_total.ddr_read_bytes == pytest.approx(b.traffic.ddr_read_bytes * (k // block_k))
    store = result.ops["main/k/store_C"]
    c = reference_result.by_name("C")
    assert store.count == c.request_count
    assert store.traffic_total.ddr_write_bytes == pytest.approx(c.traffic.ddr_write_bytes)


def test_reference_gemm_binding_fixture_matches_periodic_reference():
    """§9.1 isolated binding fixture: the reference oracle's phase Timings drive analyze().

    The reference ``BoundGemmCostOracle`` prices L2/DDR service with its own
    architecture-specific conventions, so its Timings are injected as explicit
    overrides instead of re-deriving them through ``GenericOracle``.  With the
    same DAG, II components, the selected periodic II, the single-CTA natural
    K-loop boundary and per-iteration service must agree exactly.
    """

    from tilesight.modeling._pipeline.periodic_schedule import compute_lower_bounds
    from tilesight.modeling import GeneralGemmOptions, ThroughputFallbackPolicy, bind_general_gemm, make_gemm, model_general_gemm
    from tilesight.modeling.tests.fixtures.legacy_gemm import build_gemm
    from tilesight.modeling.ops.schema import GemmOpSpec

    arch = H100_SXM().set_to_microbench()
    op = make_gemm(m=4096, n=4096, k=4096, a_storage_dtype="bf16", b_storage_dtype="bf16",
                   result_storage_dtype="fp32", compute_dtype="bf16", accumulation_dtype="fp32")
    ir = build_gemm()
    options = GeneralGemmOptions(profile="native_periodic", cache_policy="stable_shadow", cache_sample_budget=256,
                                 tensor_throughput_fallback=ThroughputFallbackPolicy(dtype_aliases=(("bf16", "fp16"),)))
    oracle = bind_general_gemm(op, ir, arch, options)
    reference = model_general_gemm(op, ir, arch, options)
    timings = {phase.name: oracle.resolve(phase) for phase in ir.periodic_loop("ko").phases}
    assert set(timings) == {"load_A", "load_B", "mma"}

    block_m, block_n, block_k, stages = 128, 128, 64, 3
    a = sight.TileAccess("A", sight.Projection(("m", "ko")), (block_m, block_k), 2)
    b = sight.TileAccess("B", sight.Projection(("ko", "n")), (block_k, block_n), 2)
    load_a = sight.Load("load_A", "global", "smem", "tma", access=a, actor="producer", writes=("A_s",), timing=timings["load_A"])
    load_b = sight.Load("load_B", "global", "smem", "tma", access=b, actor="producer", writes=("B_s",), timing=timings["load_B"])
    mma = sight.Compute("mma", GemmOpSpec(m=block_m, n=block_n, k=block_k, compute_dtype="bf16", accumulation_dtype="fp32"),
                     engine="tensor", actor="tensor", reads=("A_s", "B_s"), timing=timings["mma"])
    pipeline = sight.Pipeline(
        actors=(sight.Actor("producer", (load_a, load_b), sequence=(("load_A", 0), ("load_B", 0)), resource="tma"),
                sight.Actor("tensor", (mma,))),
        buffers=(sight.Buffer("A_s", "smem", (block_m, block_k), "bf16", slots=stages),
                 sight.Buffer("B_s", "smem", (block_k, block_n), "bf16", slots=stages)),
        buffer_slots=(sight.BufferSlots("A_s", sight.start(load_a), sight.done(mma), stages),
                      sight.BufferSlots("B_s", sight.start(load_b), sight.done(mma), stages)),
        carries=(sight.Carry("accumulator", sight.done(mma), sight.start(mma), 1),),
        stages=stages,
    )
    program = sight.Program((sight.Launch(
        "main", sight.Loop("ko", 4096 // block_k, body=pipeline),
        (sight.WorkAxis("m", 32), sight.WorkAxis("n", 32)), threads=128,
    ),))
    result = sight.analyze(program, arch, options=sight.Options(ii_mode="periodic_best", cache="none"))

    bounds = compute_lower_bounds(reference.dag)
    region = result.regions["main/ko"]
    assert region.ii.resource_ii_s == pytest.approx(bounds.resource_ii)
    assert region.ii.recurrence_ii_s == pytest.approx(bounds.recurrence_ii)
    assert region.ii.credit_ii_s == pytest.approx(bounds.credit_ii)
    assert region.ii.resource_ii_s == pytest.approx(oracle.breakdown.service.resource_ii_s)
    assert region.ii.selected_ii_s == pytest.approx(reference.ii.ii)
    assert region.ii.scope == reference.ii.label
    assert region.total_s == pytest.approx(reference.natural_boundary.latency_s)
    assert region.trip_count == reference.natural_boundary.iterations

    service = oracle.breakdown.service
    groups = {path.split("/")[-1]: op_result.cost_groups[0] for path, op_result in result.ops.items()}
    assert dict(groups["mma"].service_s)["tensor"] == pytest.approx(service.tensor_s)
    assert dict(groups["load_A"].service_s)["tma"] + dict(groups["load_B"].service_s)["tma"] == pytest.approx(service.memory_s)
    assert groups["mma"].traffic_source == "explicit_timing_override"
    assert result.launches["main"].work_units == 1024
