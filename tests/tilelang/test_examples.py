"""The examples end to end: compiled facts and estimates (H200_SXM, set_to_microbench)."""

import pytest

from conftest import example_profile, needs_tilelang

pytestmark = needs_tilelang

# Estimates are regression values of this model, not accuracy claims.
ESTIMATES_US = {"gemm": 1399.526, "gemm_pipelined": 1399.526, "gqa": 447.054, "gelu": 67.970, "sum": 36.049}


@pytest.mark.parametrize("name", sorted(ESTIMATES_US))
def test_estimate(name):
    profile = example_profile(name)
    assert profile.status == "complete", profile.refusal
    assert profile.estimate.kernel_body_s * 1e6 == pytest.approx(ESTIMATES_US[name], rel=1e-4)


def test_hand_written_warp_specialized_gemm():
    payload = example_profile("gemm").payload()
    launch, residency = payload["launch"], payload["residency"]
    assert launch["grid"] == {"bx": 64, "by": 64} and launch["threads"] == 256
    assert launch["roles"] == ["producer", "consumer"] and set(launch["mechanisms"]) == {"tma", "wgmma"}
    assert launch["dynamic_shared_bytes"] == 3 * (128 * 64 + 128 * 64) * 2
    buffers = payload["program"]["buffers"]
    assert buffers["a_smem"]["slots"] == 3 and buffers["b_smem"]["slots"] == 3
    (loop,) = payload["loops"]
    assert loop["trips"] == 128
    # TileSight's declared storage: 96 KiB of fragments and 96 KiB of shared memory per CTA fit twice.
    assert residency["ctas_per_sm"] == 2 and residency["binding"] == ["register", "smem"]
    assert residency["storage_per_cta_bytes"] == {"register": 98304.0, "smem": 98304.0}


def test_compiler_scheduled_gemm_gets_the_compilers_schedule():
    """The source says 256 threads; on sm_90 the compiler adds a producer warpgroup and uses TMA."""

    payload = example_profile("gemm_pipelined").payload()
    launch = payload["launch"]
    assert launch["threads"] == 384 and launch["roles"] == ["producer", "consumer"]
    loads = [op for item in payload["program"]["items"] if "loop" in item for op in item["body"] if op["kind"] == "load"]
    assert [op["engine"] for op in loads] == ["tma", "tma"] and {op["role"] for op in loads} == {"producer"}
    assert payload["program"]["buffers"]["As"]["slots"] == 3


def test_attention_roles_and_barriers():
    payload = example_profile("gqa").payload()
    launch = payload["launch"]
    assert launch["threads"] == 384 and launch["roles"] == ["producer", "consumer", "consumer1"]
    assert launch["bindings"] == {"seq_len_kv": 8192, "seq_len_q": 8192}
    (loop,) = payload["loops"]
    assert loop["trips"] == 64
    items = payload["program"]["items"]
    barriers = [item["barriers"] for item in items if "loop" in item]
    assert barriers and barriers[0], "named barriers between the two consumers become schedule hints"


@pytest.mark.parametrize("name,ctas,threads,resident,binding,unchecked", [
    ("sum", 4096, 128, 7, ["smem"], ["acc", "out_local", "x_f32"]),
    ("gelu", 32768, 128, 24, ["max_blocks_per_sm"], ["x_reg", "y_reg"])])
def test_memory_bound_kernels(name, ctas, threads, resident, binding, unchecked):
    payload = example_profile(name).payload()
    assert payload["estimate"]["work_units"] == ctas and payload["launch"]["threads"] == threads
    residency = payload["residency"]
    assert residency["ctas_per_sm"] == resident and residency["binding"] == binding
    assert residency["unchecked_register_buffers"] == unchecked
    assert any("outside TileSight's liveness analysis" in note for note in payload["notes"])
    kinds = {d["kind"]: d for d in payload["diagnoses"]}
    assert kinds["pipeline_bottleneck"]["values"]["kernel_utilization"]["ddr"] > 0.8


def test_a_launch_smaller_than_one_wave_is_spread_over_the_sms():
    """64 CTAs on 132 SMs: TileSight would fit seven per SM, but each CTA gets an SM of its own."""

    from conftest import example_request
    from tilesight.tilelang import profile_request

    case = example_request("sum")
    profile = profile_request(example_request("sum", factory_args={**case.factory_args, "M": 128}))
    estimate = profile.payload()["estimate"]
    assert estimate["work_units"] == 64 and estimate["resident_ctas_per_sm"] == 7
    assert estimate["effective_resident_ctas_per_sm"] == 1 and estimate["composition"] == "single_cta"
    assert estimate["kernel_body_s"] == estimate["single_cta_kernel_body_s"]


def test_every_diagnosis_is_present_and_typed():
    payload = example_profile("gemm").payload()
    kinds = [d["kind"] for d in payload["diagnoses"]]
    assert kinds == ["pipeline_bottleneck", "compute_io_overlap", "transfer_anomaly", "cache_utilization",
                     "load_imbalance", "load_imbalance", "latency_exposure", "model_deviation"]
    assert all(d["status"] in ("detected", "not_detected", "inconclusive", "not_evaluated") for d in payload["diagnoses"])
    cache = next(d for d in payload["diagnoses"] if d["kind"] == "cache_utilization")
    assert cache["status"] == "detected" and cache["values"]["dram_refetch_ratio"]["b"] > 30


def test_profiling_runs_no_compiler(monkeypatch):
    """Residency comes from TileSight; profiling lowers with TileLang and starts no compiler process."""

    import os
    import subprocess

    from conftest import example_request
    from tilesight.tilelang import profile_request

    started = []
    real = subprocess.Popen

    class Recording(real):
        def __init__(self, args, *rest, **options):
            started.append(os.path.basename(str(args[0] if isinstance(args, (list, tuple)) else args).split()[0]))
            super().__init__(args, *rest, **options)

    monkeypatch.setattr(subprocess, "Popen", Recording)
    profile = profile_request(example_request("sum"))
    assert profile.status == "complete"
    assert not started, started
