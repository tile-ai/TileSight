"""Perfetto export and ComparisonResult semantics."""

import json

import pytest

from tilesight.arch.h100_sxm import H100_SXM
from tilesight.modeling import program as sight
from tilesight.modeling.program import examples as ex

ARCH = H100_SXM()


def _gemm(mode):
    return sight.analyze(ex.gemm_program(m=1024, n=1024, k=1024), ARCH,
                      options=sight.Options(ii_mode=mode, cache="fast", cache_sample_budget=64))


def test_perfetto_export_has_timed_bars_only_with_known_starts(tmp_path):
    witness = _gemm("periodic_best")
    bound = _gemm("resource_ii")
    path = sight.export_perfetto(witness, str(tmp_path / "w.json"))
    trace = json.load(open(path))
    events = trace["traceEvents"]
    assert trace["metadata"]["tilesight_model"] == "theoretical_model"
    complete = [e for e in events if e["ph"] == "X"]
    assert any(e["cat"] == "service" and e["name"] == "mma" for e in complete)
    assert any(e["cat"] == "completion" for e in complete)
    markers = [e for e in events if e["ph"] == "i" and e["name"].endswith("periods")]
    assert markers and markers[0]["args"]["omitted_periods"] > 0
    assert all(e["dur"] >= 0 for e in complete)
    # Without a witness the loop body service bars are not fabricated.
    bound_trace = sight.perfetto_trace(bound)
    bars = [e for e in bound_trace["traceEvents"] if e["ph"] == "X" and e["cat"] == "service"]
    assert not any(e["name"] == "mma" for e in bars)
    summaries = [e for e in bound_trace["traceEvents"] if e["name"] == "summary"]
    assert summaries and "cumulative_service_s" in summaries[0]["args"]


def test_compare_alignment_and_residual_rules():
    baseline = _gemm("resource_ii")
    calibrated = sight.analyze(ex.gemm_program(m=1024, n=1024, k=1024), ARCH,
                            options=sight.Options(cache="fast", cache_sample_budget=64),
                            context=sight.Context(clock_overrides={"sm": 1.2e9}, label="observed"))
    observations = [
        sight.Observation("main", "time_s", 30e-6, "s", "kernel_elapsed", "cupti"),
        sight.Observation("gemm", "time_s", 0.0, "s", "host_wall", "torch"),
        sight.Observation("main", "time_s", 31e-6, "s", "host_wall", "torch"),
        sight.Observation("main", "ddr_read_bytes", 1e6, "bytes", "kernel", "ncu"),
        sight.Observation("main", "l2_hit_rate", 0.5, "ratio", "kernel", "ncu", denominator="lts_requests"),
        sight.Observation("main", "l2_hit_rate", 0.5, "ratio", "kernel", "ncu",
                       denominator="cache_model_read_requests", instance_id="2"),
        sight.Observation("main/k/mma", "time_s", 1e-6, "s", "kernel_elapsed", "guess"),
        sight.Observation("nope", "time_s", 1e-6, "s", "kernel_elapsed", "guess"),
    ]
    comparison = sight.compare(baseline, calibrated, observations)
    rows = {(m.target, m.metric, m.scope, m.denominator): m for m in comparison.metrics}
    deltas = {(d.target, d.metric, d.scope): d for d in comparison.deltas}
    kernel = rows[("main", "time_s", "kernel_elapsed", None)]
    assert kernel.comparable and kernel.baseline == pytest.approx(baseline.launch("main").kernel_body_s)
    assert kernel.calibrated > kernel.baseline
    assert deltas[("main", "time_s", "kernel_elapsed")].model_delta > 0
    # observed == 0 -> relative error None, residual still signed
    zero = deltas[("gemm", "time_s", "host_wall")]
    assert zero.baseline_residual == pytest.approx(baseline.program.total_s) and zero.baseline_rel_error is None
    # scope mismatch -> three columns, no residual
    mismatch = rows[("main", "time_s", "host_wall", None)]
    assert not mismatch.comparable and mismatch.observed == 31e-6 and mismatch.baseline is None
    # denominator rules
    assert not rows[("main", "l2_hit_rate", "kernel", "lts_requests")].comparable
    assert rows[("main", "l2_hit_rate", "kernel", "cache_model_read_requests")].comparable
    # op-level time and unknown targets are mismatched, never apportioned
    statuses = {(a.target, a.metric, a.observation_scope): a.status for a in comparison.alignment}
    assert statuses[("main/k/mma", "time_s", "kernel_elapsed")] == "mismatched"
    assert statuses[("nope", "time_s", "kernel_elapsed")] == "mismatched"
    ops = {o.path: o for o in comparison.operations}
    assert ops["main/k/mma"].observed_residual is None
    assert ops["main/k/mma"].completion_calibrated_s > ops["main/k/mma"].completion_baseline_s
    assert any("different Context" in d for d in comparison.diagnostics)
    json.dumps(comparison.to_dict())


def test_compare_rejects_workload_mismatch_as_calibration():
    small = sight.analyze(ex.gemm_program(m=512, n=512, k=256), ARCH, options=sight.Options(cache="fast", cache_sample_budget=64))
    large = sight.analyze(ex.gemm_program(m=1024, n=1024, k=256), ARCH, options=sight.Options(cache="fast", cache_sample_budget=64))
    observations = [sight.Observation("main", "time_s", 1e-5, "s", "kernel_elapsed", "cupti"),
                    sight.Observation("main/k/mma", "ddr_read_bytes", 0.0, "bytes", "kernel", "ncu")]
    comparison = sight.compare(small, large, observations)
    for row in comparison.metrics:
        if row.observed is not None:
            assert not row.comparable and "workload identity" in row.reason
    assert all(d.baseline_residual is None for d in comparison.deltas)
    assert any("workload identity of main differs" in d for d in comparison.diagnostics)
    # Same workload, different Context: comparable; observation pinned to another workload: not.
    calibrated = sight.analyze(ex.gemm_program(m=512, n=512, k=256), ARCH,
                            options=sight.Options(cache="fast", cache_sample_budget=64),
                            context=sight.Context(clock_overrides={"sm": 1.0e9}, label="calib"))
    ok = sight.compare(small, calibrated, [sight.Observation("main", "time_s", 1e-5, "s", "kernel_elapsed", "cupti",
                                                       workload_digest=small.workload_digest("main"))])
    assert ok.metrics[0].comparable
    pinned = sight.compare(small, calibrated, [sight.Observation("main", "time_s", 1e-5, "s", "kernel_elapsed", "cupti",
                                                           workload_digest="0" * 64)])
    assert not pinned.metrics[0].comparable and "observation workload digest" in pinned.metrics[0].reason


def test_observation_validation():
    with pytest.raises(Exception, match="scope"):
        sight.Observation("main", "time_s", 1.0, "s", "wall", "x")
    with pytest.raises(Exception, match="unit"):
        sight.Observation("main", "time_s", 1.0, "ms", "kernel_elapsed", "x")


def test_perfetto_uses_per_edge_gaps():
    from tilesight.modeling.ir import Timing
    from tilesight.modeling.ops.schema import GemmOpSpec

    def op():
        return sight.Compute("c", GemmOpSpec(m=8, n=8, k=8, compute_dtype="fp16"), actor="t", timing=Timing(1e-6))

    launches = tuple(sight.Launch(name, op(), (sight.WorkAxis("i", 1),)) for name in ("A", "B", "C"))
    program = sight.Program(launches, edges=(sight.LaunchEdge("A", "B", gap_s=10e-6), sight.LaunchEdge("B", "C")))
    result = sight.analyze(program, ARCH, options=sight.Options(cache="none", kernel_launch_s=0.0))
    assert result.program.gap_before("B") == pytest.approx(10e-6) and result.program.gap_before("C") == 0.0
    trace = sight.perfetto_trace(result)
    starts = [e["ts"] for e in trace["traceEvents"] if e["name"] == "launch summary"]
    assert starts == pytest.approx([0.0, 11.0, 12.0])
    assert result.program.total_s == pytest.approx(13e-6)


def test_compare_traffic_needs_cumulative_scope():
    x = sight.TileAccess("X", sight.Projection(("i",)), (512,), 2)
    program = sight.Program((sight.Launch("main", sight.Load("l", "global", "smem", "tma", access=x, actor="p"), (sight.WorkAxis("i", 4),)),))
    result = sight.analyze(program, ARCH, options=sight.Options(cache="trace"))
    comparison = sight.compare(result, result, [
        sight.Observation("main/l", "ddr_read_bytes", 1024, "bytes", "per_execution", "ncu"),
        sight.Observation("main/l", "ddr_read_bytes", 4096, "bytes", "cumulative", "ncu"),
        sight.Observation("main", "l2_write_request_bytes", 0.0, "bytes", "kernel", "ncu"),
    ])
    rows = {(m.target, m.scope): m for m in comparison.metrics}
    assert not rows[("main/l", "per_execution")].comparable and "cumulative" in rows[("main/l", "per_execution")].reason
    assert rows[("main/l", "cumulative")].comparable and rows[("main/l", "cumulative")].baseline == pytest.approx(4096)
    assert rows[("main", "kernel")].comparable and rows[("main", "kernel")].baseline == 0.0
    deltas = {(d.target, d.scope): d for d in comparison.deltas}
    assert deltas[("main/l", "per_execution")].baseline_residual is None


def _identity(**overrides):
    base = {"kernel": "k", "implementation": "examples.fa3_program", "source_version": "v1", "variant": "128x128",
            "arch": "h100", "dtype": "bf16", "causal": True, "gqa": "none", "tile": "128x128",
            "scheduler": "static", "cache_state": "cold", "timing_scope": "host_wall"}
    base.update(overrides)
    return base


def test_validation_requires_matching_identity(tmp_path):
    from tilesight.modeling.program import examples as ex
    from tilesight.modeling.program import validation as v

    assert sight.parse_shape_id("b1_h32_kv8_s2048_d128") == {"b": 1, "h": 32, "kv": 8, "s": 2048, "d": 128}
    assert sight.parse_shape_id("M8192_N3584_K8192") == {"m": 8192, "n": 3584, "k": 8192}
    csv = tmp_path / "m.csv"
    csv.write_text(
        "model,kernel,shape_id,precision,arch,parallel_scheme,simulator,run_id,measured_us,launcher_us,provenance\n"
        ",k,b1_h2_s512_d64,bf16,h100,,measured-bench,r0,50.0,50.0,measured-bench\n"
        ",k,b1_h2_s500_d64,bf16,h100,,measured-bench,r1,60.0,60.0,measured-bench\n"
        ",k,b1_h2_s1024_d64,bf16,h100,,measured-bench,r2,,,failed\n"
        ",k,weird-shape,bf16,h100,,measured-bench,r3,10.0,10.0,measured-bench\n"
        ",other,b1_h2_s512_d64,bf16,h100,,measured-bench,r4,1.0,1.0,measured-bench\n"
    )
    declared = {k: val for k, val in _identity().items() if k not in ("kernel", "arch", "dtype")}
    measurements = sight.load_measurements(str(csv), kernel="k", identity=declared)
    assert [m.status for m in measurements] == ["ok", "ok", "failed_measurement", "unparsable_shape"]
    assert measurements[2].measured_us is None

    def factory(p, _row):
        return ex.fa3_program(batch=p["b"], heads=p["h"], seq_len=p["s"], head_dim=p["d"], causal=True)

    def shape_check(p, program):
        facts = v.program_facts(program)
        q = facts["tensors"]["Q"]["tensor_shape"]
        return [] if (q[0], q[-1]) == (p["s"], p["d"]) else ["Q extents %s != (%d, %d)" % (q, p["s"], p["d"])]

    options = sight.Options(cache="fast", cache_sample_budget=64)
    report = sight.validate(measurements, factory, ARCH, name="fixture", model_identity=_identity(), options=options,
                         baselines={"b1_h2_s512_d64": 40.0}, shape_check=shape_check)
    assert report.status_counts() == {"compared": 1, "unsupported": 1, "failed_measurement": 1, "unparsable_shape": 1}
    compared = report.compared[0]
    assert compared.relative_error == pytest.approx((compared.predicted_us - 50.0) / 50.0) and not compared.exploratory
    stats = report.comparable_stats
    assert stats["n"] == 1 and stats["mape_pct"] == pytest.approx(100 * abs(compared.relative_error))
    assert stats["wmape_pct"] == pytest.approx(100 * abs(compared.signed_error_us) / 50.0)
    assert report.baseline_stats["mape_pct"] == pytest.approx(20.0)
    report.write_json(str(tmp_path / "r.json"))

    # An unrelated kernel / architecture / dtype never enters the accuracy statistics.
    other = tmp_path / "o.csv"
    other.write_text("kernel,shape_id,precision,arch,measured_us\nunrelated_kernel,b1_h2_s512_d64,fp64,a100,50.0\n")
    foreign = sight.load_measurements(str(other), identity=declared)
    mismatch = sight.validate(foreign, factory, ARCH, name="fixture", model_identity=_identity(), options=options,
                           shape_check=shape_check)
    case = mismatch.cases[0]
    assert case.status == "unverified" and case.exploratory
    assert any("kernel differs" in r for r in case.reasons) and any("arch differs" in r for r in case.reasons)
    assert any("dtype differs" in r for r in case.reasons)
    assert mismatch.comparable_stats["n"] == 0 and mismatch.mape is None
    assert mismatch.exploratory_stats["n"] == 1
    # Missing identity information is also not comparable.
    bare = sight.load_measurements(str(csv), kernel="k")
    missing = sight.validate(bare[:1], factory, ARCH, name="fixture", model_identity=_identity(), options=options,
                          shape_check=shape_check)
    assert missing.cases[0].status == "unverified"
    assert any("missing on the measurement" in r for r in missing.cases[0].reasons)
    assert v.identity_differences(_identity(), _identity(causal=False)) == ("causal differs: model=True measurement=False",)
    with pytest.raises(sight.ContractError, match="timing_scope"):
        sight.validate(bare[:1], factory, ARCH, name="x", model_identity=_identity(timing_scope="wall"))


def test_review7_identity_is_bound_to_rows_arch_and_program(tmp_path):
    from tilesight.modeling.program import examples as ex
    from tilesight.modeling.program import validation as v

    options = sight.Options(cache="fast", cache_sample_budget=64)
    # (a) the CSV says a100/fp64; a file-level declaration of h100/fp16 must not win.
    lying = tmp_path / "lying.csv"
    lying.write_text("kernel,shape_id,precision,arch,measured_us\nk,b1_h2_s512_d64,fp64,a100,50.0\n")
    declared = dict(_identity(), dtype="bf16", arch="h100")
    rows = sight.load_measurements(str(lying), identity=declared)
    assert dict(rows[0].identity)["arch"] == "a100" and dict(rows[0].identity)["dtype"] == "fp64"
    assert any("declared arch='h100' contradicts the row value 'a100'" in c for c in rows[0].identity_conflicts)

    def fa3(p, _row):
        return ex.fa3_program(batch=p["b"], heads=p["h"], seq_len=p["s"], head_dim=p["d"], causal=True)

    report = sight.validate(rows, fa3, ARCH, name="x", model_identity=_identity(), options=options, shape_check=lambda p, pr: [])
    assert report.cases[0].status == "unverified"
    assert any("contradicts the row value" in r for r in report.cases[0].reasons)
    # (b) identical identity dictionaries but the factory builds another workload and dtype.
    gemm = tmp_path / "gemm.csv"
    gemm.write_text("kernel,shape_id,precision,arch,measured_us\ngemm,M128_N128_K64,fp16,h100,10.0\n")
    identity = dict(_identity(kernel="gemm", dtype="fp16", causal="n/a", gqa="n/a"))
    declared = {k: val for k, val in identity.items() if k not in ("kernel", "arch", "dtype")}
    measured = sight.load_measurements(str(gemm), identity=declared)

    def wrong_factory(_p, _row):
        return ex.gemm_program(m=256, n=256, k=128, dtype="bf16")

    def gemm_shape(p, program):
        tensors = v.program_facts(program)["tensors"]
        expected = (p["m"], p["k"])
        return [] if tensors["A"]["tensor_shape"] == expected else ["A extents %s != %s" % (tensors["A"]["tensor_shape"], expected)]

    wrong = sight.validate(measured, wrong_factory, ARCH, name="x", model_identity=identity, options=options, shape_check=gemm_shape)
    reasons = wrong.cases[0].reasons
    assert wrong.cases[0].status == "unverified" and wrong.comparable_stats["n"] == 0
    assert any("shape: A extents (256, 128) != (128, 64)" in r for r in reasons)
    assert any("dtype='fp16' is not the program's input dtype ['bf16']" in r for r in reasons)
    right = sight.validate(measured, lambda p, _r: ex.gemm_program(m=p["m"], n=p["n"], k=p["k"], dtype="fp16"), ARCH,
                        name="x", model_identity=identity, options=options, shape_check=gemm_shape)
    assert right.cases[0].status == "compared"
    # without a shape check nothing is comparable
    unchecked = sight.validate(measured, lambda p, _r: ex.gemm_program(m=p["m"], n=p["n"], k=p["k"], dtype="fp16"), ARCH,
                            name="x", model_identity=identity, options=options)
    assert unchecked.cases[0].status == "unverified" and any("no shape_check" in r for r in unchecked.cases[0].reasons)
    # a declared model arch that is not the analysed architecture is rejected as well
    from tilesight.arch.b200 import B200
    assert v.bound_identity_differences({"arch": "h100", "dtype": "fp16"}, B200(), ex.gemm_program(m=128, n=128, k=64))
    assert v.arch_names(ARCH) == ("h100",)


def test_review7_baseline_statistics_use_the_same_intersection(tmp_path):
    from tilesight.modeling.program import examples as ex

    csv = tmp_path / "m.csv"
    csv.write_text("kernel,shape_id,precision,arch,measured_us\nk,b1_h2_s512_d64,bf16,h100,50.0\nk,b1_h2_s500_d64,bf16,h100,100.0\n")
    rows = sight.load_measurements(str(csv))
    # a causal partial q tile is not modelled -> that row has no prediction
    report = sight.validate(rows, lambda p, _r: ex.fa3_program(batch=p["b"], heads=p["h"], seq_len=p["s"], head_dim=p["d"], causal=True),
                         ARCH, name="x", model_identity=_identity(), options=sight.Options(cache="fast", cache_sample_budget=64),
                         baselines={"b1_h2_s512_d64": 40.0, "b1_h2_s500_d64": 50.0})
    assert report.status_counts() == {"unverified": 1, "unsupported": 1}
    assert report.baseline_stats["n"] == 1 and report.baseline_stats["mape_pct"] == pytest.approx(20.0)
    assert report.baseline_stats_all["n"] == 2 and report.baseline_stats_all["mape_pct"] == pytest.approx(35.0)


def test_review7_compare_resolves_equal_program_and_launch_names():
    kb = sight.KernelBuilder("main", grid={"i": 2})
    x = kb.tensor("X", (2, 64), "fp32")
    kb.copy(x["i", :], kb.fragment("x_r", (64,), "fp32"))
    default = kb.program()
    assert default.name == "main_program" and default.launches[0].name == "main"
    clash = sight.Program(default.launches, name="main")
    result = sight.analyze(clash, ARCH, options=sight.Options(cache="fast"))
    body = result.launches["main"].kernel_body_s
    comparison = sight.compare(result, result, [
        sight.Observation("main", "time_s", body, "s", "kernel_elapsed", "x"),
        sight.Observation("main", "time_s", result.program.total_s, "s", "host_wall", "x"),
        sight.Observation("main", "time_s", result.program.total_s, "s", "host_wall", "x", target_kind="launch", instance_id="2"),
    ])
    kernel, wall, forced = comparison.metrics[:3]
    assert kernel.comparable and kernel.baseline == pytest.approx(body)
    assert wall.comparable and wall.baseline == pytest.approx(result.program.total_s)
    assert not forced.comparable  # a launch has no host_wall time
    assert result.workload_digest("main") != result.program_workload_digest
