"""GPU-independent AMD planning, measurement validation and failure reporting."""
import json
import subprocess

import pytest

from micro_benchmark import run_amd as run


def device(arch="gfx942:sramecc+:xnack-"):
    return {"name": "AMD Instinct MI300X", "gcn_arch_name": arch, "compute_units": 304,
            "l2_bytes": 256 << 20, "free_memory_bytes": 180 << 30, "max_threads_per_block": 1024}


@pytest.mark.parametrize("arch", ["gfx90a", "gfx942", "gfx950", "gfx1100", "gfx90a:xnack-"])
def test_generic_gfx_targets_are_forwarded_without_inventing_model_capabilities(arch):
    assert run.validate_device(device(arch)) == arch
    assert f"--offload-arch={arch}" in run.build_command("hipcc", "fp32", arch, run.Path("probe"))


@pytest.mark.parametrize("arch", ["90", "sm_90", "gfx942;touch x", "gfx942 --help", "gfxxyz"])
def test_invalid_arch_is_rejected(arch):
    with pytest.raises(ValueError, match="valid AMD"):
        run.validate_device(device(arch))


def test_library_benchmark_is_explicit_opt_in():
    assert "rocblas_gemm" not in run.select_benchmarks(None)
    assert run.select_benchmarks(["dram", "rocblas_gemm", "dram"]) == ["dram", "rocblas_gemm"]
    with pytest.raises(ValueError, match="Unknown"):
        run.select_benchmarks(["wgmma"])


def test_dram_working_set_and_concurrency_use_actual_device():
    args = run.run_arguments("dram", True, 2, device())
    assert args[args.index("--device") + 1] == "2"
    assert int(args[args.index("--bytes") + 1]) >= 8 * device()["l2_bytes"]
    assert args[args.index("--blocks") + 1] == "304,608,1216"
    args = run.run_arguments("dram", True, 0, device(), [1, 16, 512])
    assert args[args.index("--blocks") + 1] == "1,16,512"
    with pytest.raises(ValueError, match="free memory"):
        run.run_arguments("dram", True, 0, {**device(), "free_memory_bytes": 1 << 20})


def test_cache_does_not_guess_a_missing_hardware_capacity():
    with pytest.raises(run.Unavailable, match="no cache size is guessed"):
        run.run_arguments("cache", True, 0, {**device(), "l2_bytes": 0})


def test_block_count_validation():
    assert run.block_counts("1,8,8,32") == [1, 8, 32]
    for value in ("0", "-1", "2147483648", "", "1,nan"):
        with pytest.raises(run.argparse.ArgumentTypeError):
            run.block_counts(value)


def test_metric_parser_preserves_concurrency_samples_and_rejects_invalid_values():
    metric = {"name": "dram_copy_payload", "value": 1200.5, "unit": "GB/s", "blocks": 304,
              "sample": 0, "time_ms": 2.5, "bytes": 1 << 30, "iterations": 2}
    assert run.parse_metrics("details\nTILESIGHT_METRIC_JSON " + json.dumps(metric)) == [metric]
    for value in (0, -1, float("nan"), float("inf"), "1200", True):
        with pytest.raises(ValueError):
            run.parse_metrics("TILESIGHT_METRIC_JSON " + json.dumps({**metric, "value": value}))
    with pytest.raises(ValueError, match="recognized"):
        run.parse_metrics("no result")


def test_dry_run_has_no_gpu_compiler_or_filesystem_side_effects(monkeypatch, tmp_path, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("dry run attempted execution")
    monkeypatch.setattr(run, "execute", forbidden)
    output = tmp_path / "unused"
    assert run.main(["--arch", "gfx950", "--device", "2", "--dry-run", "--output", str(output)]) == 0
    assert not output.exists()
    text = capsys.readouterr().out
    assert "--offload-arch=gfx950" in text and "--device 2" in text
    assert "placeholders" in text


def test_existing_output_is_never_overwritten(tmp_path):
    marker = tmp_path / "keep"
    marker.write_text("saved")
    with pytest.raises(SystemExit):
        run.main(["--arch", "gfx942", "--build-only", "--output", str(tmp_path)])
    assert marker.read_text() == "saved"


def test_missing_sdk_is_recorded_without_claiming_success(monkeypatch, tmp_path):
    monkeypatch.setattr(run.shutil, "which", lambda _: None)
    output = tmp_path / "missing"
    assert run.main(["--arch", "gfx942", "--build-only", "--output", str(output)]) == 1
    report = json.loads((output / "results.json").read_text())
    assert report["status"] == "failed"
    assert report["failure_kind"] == "unavailable"
    assert "hipcc not found" in report["error"]


def test_offline_build_reuses_core_and_does_not_query_a_device(monkeypatch, tmp_path):
    monkeypatch.setattr(run.shutil, "which", lambda _: "/hipcc")
    commands = []
    def execute(command, *args):
        commands.append(command)
        assert not any("device_info" in part for part in command)
        return "hipcc test" if command[-1] == "--version" else ""
    monkeypatch.setattr(run, "execute", execute)
    output = tmp_path / "offline"
    assert run.main(["--arch", "gfx950", "--build-only", "--bench", "fp32", "dram", "--output", str(output)]) == 0
    assert len(commands) == 2  # Compiler version + one shared core executable.
    report = json.loads((output / "results.json").read_text())
    assert all(row["status"] == "built" and "metrics" not in row for row in report["results"])


def test_missing_opt_in_rocblas_is_reported_as_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(run.shutil, "which", lambda _: "/hipcc")
    def execute(command, log, timeout, env):
        if command[-1] == "--version":
            return "hipcc test"
        log.write_text("fatal error: 'rocblas/rocblas.h' file not found\n")
        raise RuntimeError("compile failed")
    monkeypatch.setattr(run, "execute", execute)
    output = tmp_path / "missing_library"
    assert run.main(["--arch", "gfx942", "--build-only", "--bench", "rocblas_gemm", "--output", str(output)]) == 1
    report = json.loads((output / "results.json").read_text())
    assert report["results"][0]["status"] == "unavailable"
    assert "rocBLAS" in report["results"][0]["error"]


@pytest.mark.parametrize("fail", [False, True])
def test_visible_ordinal_and_masks_are_preserved_for_each_executable(monkeypatch, tmp_path, fail):
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "1,0")
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "3,5")
    monkeypatch.setattr(run.shutil, "which", lambda _: "/hipcc")
    calls = []
    def execute(command, log, timeout, env):
        calls.append(command)
        assert env["HIP_VISIBLE_DEVICES"] == "1,0"
        assert env["ROCR_VISIBLE_DEVICES"] == "3,5"
        assert env["HIP_PLATFORM"] == "amd"
        if command[-1] == "--version":
            return "hipcc test"
        if command[0].endswith("device_info"):
            assert command[-2:] == ["--device", "1"]
            return json.dumps(device())
        if command[0].endswith("/core"):
            assert command[command.index("--device") + 1] == "1"
            if fail:
                raise RuntimeError("runtime failure")
            return 'TILESIGHT_METRIC_JSON {"name":"fp32_fma","value":10,"unit":"TFLOP/s"}'
        return ""
    monkeypatch.setattr(run, "execute", execute)
    output = tmp_path / "results"
    assert run.main(["--device", "1", "--bench", "fp32", "--output", str(output)]) == int(fail)
    report = json.loads((output / "results.json").read_text())
    assert report["visibility"]["ROCR_VISIBLE_DEVICES"] == "3,5"
    assert report["arch"] == device()["gcn_arch_name"]
    assert report["results"][0]["status"] == ("failed" if fail else "passed")
    assert report["status"] == ("failed" if fail else "complete")
    assert any("--offload-arch=gfx942:sramecc+:xnack-" in command for command in calls)


def test_timeout_preserves_partial_log(monkeypatch, tmp_path):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 1, output=b"partial output\n")
    monkeypatch.setattr(run.subprocess, "run", timeout)
    log = tmp_path / "failure.log"
    with pytest.raises(RuntimeError, match="Timeout"):
        run.execute(["probe"], log, 1, {})
    assert log.read_text() == "partial output\n"


def test_nonzero_exit_is_not_treated_as_a_valid_measurement(monkeypatch, tmp_path):
    monkeypatch.setattr(run.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a[0], 2, "HIP failed\n"))
    log = tmp_path / "failure.log"
    with pytest.raises(RuntimeError, match="Exit 2"):
        run.execute(["probe"], log, 1, {})
    assert log.read_text() == "HIP failed\n"
