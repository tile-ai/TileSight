"""GPU-independent tests for selection, measurement parsing and execution failures."""
import json
import subprocess
import shutil

import pytest

from micro_benchmark import run


def device(name="NVIDIA RTX PRO 6000 Blackwell Server Edition", cc="12.0"):
    return {"name": name, "compute_capability": cc,
            "uuid": "GPU-12345678-1234-5678-90ab-1234567890ab",
            "sm_count": 144, "l2_bytes": 96 << 20, "free_memory_bytes": 90 << 30}


@pytest.mark.parametrize("sm,special", [
    (75, set()), (80, set()), (86, set()), (89, set()),
    (90, {"wgmma"}), (100, {"tcgen05", "tmem_write", "tmem_read", "tmem_rw"}),
    (103, set()), (120, set()), (121, set()),
])
def test_special_instructions_are_gated_by_exact_sm(sm, special):
    selected = run.select_benchmarks(sm, None)
    assert {b.name for b in selected if b.exact_sm} == special
    assert set(run.CORE_MODES) <= {b.name for b in selected}
    assert ("dram" in {b.name for b in selected}) == (sm >= 80)
    assert all(b.arch(sm) == f"sm_{sm}{'a' if b.name in special else ''}" for b in selected)


def test_b6000_cannot_run_sm100_tmem():
    with pytest.raises(ValueError, match="Not supported"):
        run.select_benchmarks(120, ["tmem_read"])


def test_device_validation_accepts_auto_and_rejects_wrong_profile():
    assert run.validate_device(device(), "b6000") == 120
    assert run.validate_device(device("NVIDIA A100-SXM4-80GB", "8.0"), "auto") == 80
    with pytest.raises(ValueError, match="Requested b200"):
        run.validate_device(device(), "b200")
    with pytest.raises(ValueError, match="Requested b6000"):
        run.validate_device(device("NVIDIA GeForce RTX 5090"), "b6000")


def test_dram_allocation_and_iterations_cover_an_out_of_cache_working_set():
    benchmark = run.select_benchmarks(120, ["dram"])[0]
    info = device()
    args = dict(zip(*(iter(run.run_arguments(benchmark, True, info)),) * 2))
    size = int(args["--bytes"])
    assert size >= 8 * info["l2_bytes"] and size & (size - 1) == 0
    assert info["sm_count"] * 256 * 8 * 16 * int(args["--iters"]) >= size
    assert args["--blocks"] == "144,288,576"
    with pytest.raises(ValueError, match="free memory"):
        run.run_arguments(benchmark, True, {**info, "free_memory_bytes": 1 << 30})


def test_metric_units_and_invalid_measurements():
    result = run.parse_metrics("details\nTILESIGHT_METRIC l2_read_gbps 123.5 GB/s\n")
    assert result == [{"name": "l2_read_gbps", "value": 123.5, "unit": "GB/s"}]
    for value in ["nan", "inf", "0", "-1"]:
        with pytest.raises(ValueError):
            run.parse_metrics(f"TILESIGHT_METRIC rate {value} GB/s")
    with pytest.raises(ValueError, match="no recognized"):
        run.parse_metrics("kernel did not report a result")


def test_dram_csv_keeps_method_and_launch_configuration():
    output = "method,device,blocks,repeat,time_ms,achieved_ddr_bw_GB_s\nread_cg,0,144,2,1.0,2048\n"
    assert run.parse_metrics(output) == [{"name": "dram_read_cg", "value": 2048.0,
                                         "unit": "GB/s", "blocks": 144, "sample": 2, "time_ms": 1.0}]


def test_custom_dram_concurrency_preserves_requested_counts():
    blocks = run.block_counts("1,2,8,8,512")
    assert blocks == [1, 2, 8, 512]
    benchmark = run.select_benchmarks(120, ["dram"])[0]
    args = run.run_arguments(benchmark, False, device(), blocks)
    assert args[args.index("--blocks") + 1] == "1,2,8,512"
    for text in ["0", "-1", "1,abc", "", "2147483648"]:
        with pytest.raises(run.argparse.ArgumentTypeError):
            run.block_counts(text)


def test_dry_run_needs_no_compiler_gpu_or_output_directory(monkeypatch, tmp_path, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("Dry run attempted execution")
    monkeypatch.setattr(run, "execute", forbidden)
    output = tmp_path / "unused"
    assert run.main(["--gpu", "b6000", "--bench", "fp32", "--dry-run", "--output", str(output)]) == 0
    assert not output.exists()
    assert "-gencode=arch=compute_120,code=sm_120" in capsys.readouterr().out


@pytest.mark.parametrize("sm,name", [(90, "wgmma"), (100, "tcgen05"), (100, "tmem_read")])
def test_accelerated_instructions_compile_only_for_the_accelerated_target(monkeypatch, tmp_path, sm, name):
    commands = []
    def execute(command, *args):
        commands.append(command)
        return "0 bytes spill stores, 0 bytes spill loads"
    monkeypatch.setattr(run, "execute", execute)
    benchmark = run.select_benchmarks(sm, [name])[0]
    run.build(benchmark, sm, "/nvcc", "test", tmp_path, 1, {})
    assert f"-gencode=arch=compute_{sm}a,code=sm_{sm}a" in commands[0]
    assert not any(flag.startswith("-arch=") for flag in commands[0])


def test_nonzero_exit_is_reported_and_log_is_preserved(monkeypatch, tmp_path):
    monkeypatch.setattr(run.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a[0], 7, "CUDA failed\n"))
    log = tmp_path / "failure.log"
    with pytest.raises(RuntimeError, match="Exit 7"):
        run.execute(["probe"], log, 1)
    assert log.read_text() == "CUDA failed\n"


def test_timeout_preserves_partial_output(monkeypatch, tmp_path):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 1, output=b"partial output\n")
    monkeypatch.setattr(run.subprocess, "run", timeout)
    log = tmp_path / "timeout.log"
    with pytest.raises(RuntimeError, match="Timeout"):
        run.execute(["probe"], log, 1)
    assert log.read_text() == "partial output\n"


@pytest.mark.parametrize("fail", [False, True])
def test_runner_pins_selected_visible_device_and_persists_status(monkeypatch, tmp_path, fail):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5,3")
    monkeypatch.setattr(run.shutil, "which", lambda _: "/nvcc")
    calls = []
    def execute(command, log, timeout, env=None):
        calls.append((command, env))
        if command[-1] == "--version": return "nvcc test"
        if command[-2:] == ["--device", "1"]: return json.dumps(device())
        if command[0] == "/benchmark":
            assert env["CUDA_VISIBLE_DEVICES"] == device()["uuid"]
            if fail: raise RuntimeError("simulated runtime failure")
            return "TILESIGHT_METRIC fp32_cuda_tflops 10 TFLOP/s\n"
        return ""
    monkeypatch.setattr(run, "execute", execute)
    monkeypatch.setattr(run, "build", lambda *a: (run.Path("/benchmark"), ["/nvcc", "-arch=sm_120"], a[4] / "build/log"))
    output = tmp_path / "result"
    assert run.main(["--gpu", "auto", "--device", "1", "--bench", "fp32", "--output", str(output)]) == int(fail)
    report = json.loads((output / "results.json").read_text())
    assert report["original_cuda_visible_devices"] == "5,3"
    assert report["results"][0]["status"] == ("failed" if fail else "passed")
    assert report["status"] == ("failed" if fail else "complete")
    assert any(command[-2:] == ["--device", "1"] and env is None for command, env in calls)


def test_existing_outputs_are_not_overwritten(tmp_path):
    marker = tmp_path / "keep.txt"
    marker.write_text("previous measurements")
    with pytest.raises(SystemExit):
        run.main(["--gpu", "b200", "--build-only", "--output", str(tmp_path)])
    assert marker.read_text() == "previous measurements"


@pytest.mark.parametrize("name", ["wgmma", "tcgen05"])
def test_tensor_issue_commands_explicitly_use_nonzero_operands(name):
    benchmark = next(b for b in run.BENCHMARKS if b.name == name)
    for quick in (False, True):
        args = run.run_arguments(benchmark, quick)
        assert args[args.index("--data") + 1] == "nonzero"
        assert run.data_configuration(benchmark, 0)["input_fp16"] > 0


@pytest.mark.parametrize("seed", [0, 1729, 0xffffffff])
def test_payload_seed_reaches_every_seeded_probe(seed):
    selected = {"dram", "l2", "smem", "tmem_write", "tmem_read", "tmem_rw"}
    for benchmark in run.BENCHMARKS:
        if benchmark.name not in selected:
            continue
        args = run.run_arguments(benchmark, True, device(), seed=seed)
        assert int(args[args.index("--seed") + 1]) == seed
        assert run.data_configuration(benchmark, seed)["seed"] == seed


@pytest.mark.parametrize("seed", ["-1", "4294967296", "nan"])
def test_bad_data_seed_is_rejected_before_execution(monkeypatch, seed, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid seed reached process execution")
    monkeypatch.setattr(run, "execute", forbidden)
    output = tmp_path / "unused"
    with pytest.raises(SystemExit):
        run.main(["--gpu", "b200", "--data-seed", seed, "--output", str(output)])
    assert not output.exists()


def test_dram_validation_failure_cannot_be_reported_as_bandwidth():
    output = "method,device,blocks,repeat,time_ms,achieved_ddr_bw_GB_s,validation\nread_cg,0,1,0,1,100,failed\n"
    with pytest.raises(ValueError, match="payload validation"):
        run.parse_metrics(output)


def test_report_records_seed_and_data_policy_used_by_process(monkeypatch, tmp_path):
    monkeypatch.setattr(run.shutil, "which", lambda _: "/nvcc")
    def execute(command, log, timeout, env=None):
        if command[-1] == "--version": return "nvcc test"
        if command[-2:] == ["--device", "0"]: return json.dumps(device())
        if command[0] == "/benchmark":
            assert command[command.index("--seed") + 1] == "42"
            return "TILESIGHT_METRIC l2_read_gbps 100 GB/s\n"
        return ""
    monkeypatch.setattr(run, "execute", execute)
    monkeypatch.setattr(run, "build", lambda *a: (run.Path("/benchmark"), ["/nvcc"], a[4] / "build/log"))
    output = tmp_path / "run"
    assert run.main(["--bench", "l2", "--data-seed", "42", "--output", str(output)]) == 0
    report = json.loads((output / "results.json").read_text())
    assert report["data_policy_version"] == 2
    assert report["results"][0]["data_configuration"] == {"pattern": "index_hash_v1", "seed": 42}


def test_cpp_payloads_are_nonzero_nonuniform_and_reproducible(tmp_path):
    """Check the actual host/device generator without a CUDA installation."""
    compiler = shutil.which("c++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable for shared payload check")
    source = tmp_path / "check_pattern.cpp"
    source.write_text(r'''
#include "PATTERN_HEADER"
#include <cassert>
#include <set>
int main() {
    std::set<uint32_t> words;
    unsigned changed = 0;
    unsigned bits[32] = {};
    for (uint64_t i = 0; i < 65536; ++i) {
        uint32_t first = tilesight_bench::data_word(i, 1729);
        assert(first != 0);
        assert(first == tilesight_bench::data_word(i, 1729));
        changed += first != tilesight_bench::data_word(i, 1730);
        words.insert(first);
        for (unsigned bit = 0; bit < 32; ++bit) bits[bit] += (first >> bit) & 1u;
    }
    assert(words.size() >= 65535 && changed > 65000);
    for (unsigned count : bits) assert(count > 20000 && count < 45000);
    uint32_t seed = 0;
    assert(tilesight_bench::parse_data_seed("0", &seed) && seed == 0);
    assert(tilesight_bench::parse_data_seed("4294967295", &seed) && seed == 0xffffffffu);
    assert(!tilesight_bench::parse_data_seed("4294967296", &seed));
    assert(!tilesight_bench::parse_data_seed("-1", &seed));
}
'''.replace("PATTERN_HEADER", str(run.ROOT / "src/common/data_pattern.hpp")))
    binary = tmp_path / "check_pattern"
    subprocess.run([compiler, "-std=c++17", str(source), "-o", str(binary)], check=True, capture_output=True, text=True)
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)
