#!/usr/bin/env python3
"""Build and run portable AMD HIP probes; no TileSight or PyTorch dependency."""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SOURCES = ROOT / "src" / "amd"
CORE_MODES = ("dram", "cache", "fp32", "fp64", "sfu", "launch")
BENCHMARKS = (*CORE_MODES, "rocblas_gemm")
METRIC_FIELDS = ("name", "value", "unit", "blocks", "sample", "time_ms", "bytes", "iterations")
SCOPES = {
    "dram": "Kernel copy payload (read + write bytes), not memory-controller counters; sweep launched blocks.",
    "cache": "Warm working-set reads sized from queried L2; includes L1 and does not isolate L2 hardware.",
    "fp32": "Scalar FP32 FMA throughput (2 FLOPs/FMA); not matrix-core throughput or instruction latency.",
    "fp64": "Scalar FP64 FMA throughput (2 FLOPs/FMA); not matrix-core throughput or instruction latency.",
    "sfu": "sqrtf expression throughput including loop/dependency overhead; not isolated ISA issue rate.",
    "launch": "Host batch enqueue + device completion time divided by launches; not isolated device latency.",
    "rocblas_gemm": "Optional rocBLAS FP16 GEMM with FP32 accumulation (2*M*N*K FLOPs); library throughput, not MFMA issue rate.",
}


class Unavailable(RuntimeError):
    """A required compiler, library, or queried capability is unavailable."""


def positive(value: str) -> int:
    number = int(value)
    if number <= 0 or number > 2147483647:
        raise argparse.ArgumentTypeError("must be a positive 32-bit integer")
    return number


def block_counts(value: str) -> list[int]:
    try:
        return list(dict.fromkeys(positive(item) for item in value.split(",")))
    except (ValueError, argparse.ArgumentTypeError) as exc:
        raise argparse.ArgumentTypeError("use comma-separated positive 32-bit block counts") from exc


def target_arch(value: str) -> str:
    # Preserve target-ID features reported by HIP (for example gfx90a:xnack-).
    if not re.fullmatch(r"gfx[0-9a-f]{3,5}(?::(?:xnack|sramecc)[+-])*", value):
        raise argparse.ArgumentTypeError("expected an AMD gfx target, e.g. gfx90a, gfx942, gfx950")
    return value


def validate_device(info: dict) -> str:
    try:
        arch = target_arch(info["gcn_arch_name"])
    except argparse.ArgumentTypeError as exc:
        raise ValueError("HIP did not report a valid AMD gfx target") from exc
    if info["compute_units"] <= 0 or info["free_memory_bytes"] <= 0:
        raise ValueError("HIP returned invalid device capacity")
    if info["max_threads_per_block"] < 256:
        raise ValueError("These probes require at least 256 threads per block")
    return arch


def select_benchmarks(names: list[str] | None) -> list[str]:
    if names is None:
        return list(CORE_MODES)
    unknown = set(names) - set(BENCHMARKS)
    if unknown:
        raise ValueError("Unknown benchmarks: " + ", ".join(sorted(unknown)))
    return list(dict.fromkeys(names))


def run_arguments(name: str, quick: bool, device: int, info: dict | None = None,
                  dram_blocks: list[int] | None = None) -> list[str]:
    arguments = [name, "--device", str(device), "--repeat", "3" if quick else "7"]
    if name == "dram":
        size = max(256 << 20 if quick else 1 << 30, 8 * (info or {}).get("l2_bytes", 0))
        size = 1 << (size - 1).bit_length()
        if info and 2 * size > info["free_memory_bytes"] * 0.8:
            raise ValueError("Not enough free memory for the out-of-cache copy working set")
        cus = (info or {}).get("compute_units", 1)
        arguments += ["--bytes", str(size), "--iterations", "2" if quick else "8",
                      "--blocks", ",".join(map(str, dram_blocks or [cus, 2 * cus, 4 * cus]))]
    elif name == "cache":
        size = (info or {}).get("l2_bytes", 1 << 20) // 2
        if size < 4096:
            raise Unavailable("HIP did not report a usable L2 size; no cache size is guessed")
        arguments += ["--bytes", str(size // 4 * 4), "--iterations", "32" if quick else "256"]
    elif name == "rocblas_gemm":
        arguments += ["--size", "1024" if quick else "4096"]
    else:
        arguments += ["--iterations", "256" if quick else "4096"]
    return arguments


def parse_metrics(output: str) -> list[dict]:
    metrics = []
    for line in output.splitlines():
        if not line.startswith("TILESIGHT_METRIC_JSON "):
            continue
        metric = json.loads(line.removeprefix("TILESIGHT_METRIC_JSON "))
        if set(metric) - set(METRIC_FIELDS) or not {"name", "value", "unit"} <= metric.keys():
            raise ValueError("Unexpected metric fields")
        if not isinstance(metric["name"], str) or not isinstance(metric["unit"], str):
            raise ValueError("Invalid metric name or unit")
        for field in set(metric) - {"name", "unit"}:
            value = metric[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Invalid metric {field}")
            if value < 0 or (field != "sample" and value == 0):
                raise ValueError(f"Invalid metric {field}")
        metrics.append(metric)
    if not metrics:
        raise ValueError("Benchmark produced no recognized measurements")
    return metrics


def execute(command: list[str], log: Path, timeout: int, env: dict) -> str:
    try:
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                cwd=ROOT, env=env, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or b""
        log.write_text(output.decode(errors="replace") if isinstance(output, bytes) else output)
        raise RuntimeError(f"Timeout after {timeout}s; see {log}") from exc
    log.write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f"Exit {result.returncode}; see {log}")
    return result.stdout


def build_command(hipcc: str, name: str, arch: str, binary: Path) -> list[str]:
    source = "rocblas_gemm.cpp" if name == "rocblas_gemm" else "core.cpp"
    command = [hipcc, "-O3", "-std=c++17", "-x", "hip", f"--offload-arch={arch}", str(SOURCES / source)]
    if name == "rocblas_gemm":
        command.append("-lrocblas")
    return [*command, "-o", str(binary)]


def write_report(report: dict, path: Path) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0, help="HIP ordinal within inherited HIP/ROCR visibility")
    parser.add_argument("--arch", type=target_arch, help="Offline gfx target; runtime always queries the actual GPU")
    parser.add_argument("--bench", nargs="+", help="Default: core probes; rocblas_gemm is opt-in")
    parser.add_argument("--dram-blocks", type=block_counts, help="Launched-block concurrency sweep, e.g. 1,8,32,128,256")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Short smoke run, not a calibration result")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without compiler or GPU")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--hipcc", default=os.environ.get("HIPCC", "hipcc"))
    parser.add_argument("--timeout", type=positive, default=180)
    parser.add_argument("--output", type=Path, help="New output directory; never overwrites a previous run")
    args = parser.parse_args(argv)
    if args.device < 0:
        parser.error("--device must be nonnegative")
    try:
        selected = select_benchmarks(args.bench)
    except ValueError as exc:
        parser.error(str(exc))
    if args.list:
        for name in BENCHMARKS:
            print(f"{name:14} {SCOPES[name]}")
        return 0
    if (args.build_only or args.dry_run) and args.arch is None:
        parser.error("Offline operation requires --arch (for example gfx90a/gfx942/gfx950)")
    if args.arch is not None and not (args.build_only or args.dry_run):
        parser.error("--arch is offline-only; runtime target is queried from the selected HIP device")
    if args.dry_run:
        for name in selected:
            binary = Path("<build>") / name
            print(shlex.join(build_command(args.hipcc, name, args.arch, binary)))
            print("  run:", shlex.join([str(binary), *run_arguments(name, args.quick, args.device,
                                                                  dram_blocks=args.dram_blocks)]))
        print("Offline sizes/counts are placeholders: runtime uses queried L2, free memory and compute units.")
        print("HIP_PLATFORM=amd; visibility masks are inherited unchanged. Compiler support determines valid gfx targets.")
        return 0

    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    out = (args.output or ROOT / "results" / (stamp + "-amd")).resolve()
    try:
        out.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error(f"Output directory already exists: {out}; choose a new path")
    (out / "build").mkdir()
    (out / "logs").mkdir()
    env = dict(os.environ, HIP_PLATFORM="amd")
    report = {"schema": "tilesight.microbenchmark.amd/1", "created_utc": stamp,
              "mode": "build_only" if args.build_only else "smoke" if args.quick else "measurement",
              "requested_device": args.device, "requested_arch": args.arch, "results": [],
              "visibility": {key: os.environ.get(key) for key in
                             ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")},
              "device_scope": "Selected HIP-visible logical device; may be one GCD or a configured partition.",
              "source_hashes": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted(SOURCES.glob("*")) if p.is_file()}}
    report_path = out / "results.json"
    failures = 0
    try:
        hipcc = shutil.which(args.hipcc)
        if hipcc is None:
            raise Unavailable("hipcc not found; provide a compatible ROCm SDK via HIPCC or --hipcc")
        report["hipcc"] = hipcc
        report["hipcc_version"] = execute([hipcc, "--version"], out / "hipcc.log", args.timeout, env)
        info = None
        arch = args.arch
        if not args.build_only:
            probe = out / "build" / "device_info"
            # No device kernels: compile as host C++ so metadata discovery needs no guessed gfx target.
            command = [hipcc, "-O2", "-std=c++17", "-x", "c++", "-D__HIP_PLATFORM_AMD__=1",
                       str(SOURCES / "device_info.cpp"), "-o", str(probe)]
            execute(command, out / "build/device_info.log", args.timeout, env)
            info = json.loads(execute([str(probe), "--device", str(args.device)], out / "device.json", args.timeout, env))
            arch = validate_device(info)
            report["device"] = info
            print(f"GPU: {info['name']} ({arch}), visible device {args.device}", flush=True)
        report["arch"] = arch
        binaries = {}
        for name in selected:
            row = {"benchmark": name, "arch": arch, "status": "pending", "scope": SCOPES[name]}
            report["results"].append(row)
            try:
                arguments = [] if args.build_only else run_arguments(name, args.quick, args.device, info, args.dram_blocks)
                source_key = "rocblas_gemm" if name == "rocblas_gemm" else "core"
                binary = out / "build" / source_key
                command = build_command(hipcc, name, arch, binary)
                row["build_command"] = command
                if source_key not in binaries:
                    log = out / "build" / (source_key + ".log")
                    try:
                        execute(command, log, args.timeout, env)
                    except RuntimeError as exc:
                        diagnostic = log.read_text() if log.exists() else ""
                        if name == "rocblas_gemm" and re.search(
                            r"rocblas[/\\]rocblas.h.*(?:not found|No such file)|(?:cannot find|unable to find library).*rocblas", diagnostic):
                            raise Unavailable("rocBLAS development headers/library not found; see " + str(log)) from exc
                        raise
                    binaries[source_key] = binary
                if args.build_only:
                    row["status"] = "built"
                else:
                    command = [str(binary), *arguments]
                    row["run_command"] = command
                    row["run_log"] = f"logs/{name}.log"
                    output = execute(command, out / row["run_log"], args.timeout, env)
                    row.update(status="passed", metrics=parse_metrics(output))
            except Unavailable as exc:
                row.update(status="unavailable", error=str(exc))
                # Missing optional cache query does not invalidate other measurements.
                failures += int(name != "cache" or args.bench is not None)
            except (OSError, RuntimeError, ValueError) as exc:
                row.update(status="failed", error=str(exc))
                failures += 1
            print(f"[{name}] {row['status']}" + (": " + row["error"] if "error" in row else ""), flush=True)
            write_report(report, report_path)
    except (OSError, RuntimeError, ValueError) as exc:
        report.update(error=str(exc), failure_kind="unavailable" if isinstance(exc, Unavailable) else "failed")
        failures += 1
        print(str(exc), file=sys.stderr)
    except KeyboardInterrupt:
        report["error"] = "Interrupted"
        failures += 1
        for row in report["results"]:
            if row["status"] == "pending":
                row["status"] = "interrupted"
    finally:
        report["status"] = "failed" if failures else "complete"
        write_report(report, report_path)
        with (out / "metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["benchmark", *METRIC_FIELDS])
            writer.writeheader()
            for row in report["results"]:
                for metric in row.get("metrics", []):
                    writer.writerow({"benchmark": row["benchmark"], **metric})
        print(f"Results: {report_path}", flush=True)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
