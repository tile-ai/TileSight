#!/usr/bin/env python3
"""Build and run CUDA calibration probes without importing TileSight or PyTorch."""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass

ROOT = Path(__file__).resolve().parent
PROFILES = {"h200": 90, "b200": 100, "b6000": 120}
CORE_MODES = ("l2", "l2_latency", "smem", "fp32", "fp16", "fp64", "sfu", "clock", "launch")
DEFAULT_DATA_SEED = 1729
DATA_POLICY_VERSION = 2


@dataclass(frozen=True)
class Benchmark:
    name: str
    source: str
    minimum_sm: int = 75
    exact_sm: int | None = None
    accelerated: bool = False
    flags: tuple[str, ...] = ()

    def supports(self, sm: int) -> bool:
        return sm >= self.minimum_sm and (self.exact_sm is None or sm == self.exact_sm)

    def arch(self, sm: int) -> str:
        return f"sm_{sm}{'a' if self.accelerated else ''}"

    def gencode(self, sm: int) -> str:
        suffix = f"{sm}{'a' if self.accelerated else ''}"
        return f"-gencode=arch=compute_{suffix},code=sm_{suffix}"


BENCHMARKS = tuple(Benchmark(mode, "common/core.cu") for mode in CORE_MODES) + (
    Benchmark("dram", "common/dram_sweep.cu", minimum_sm=80),
    Benchmark("tensor_gemm", "common/tensor_gemm.cu", flags=("-lcublasLt", "-lcublas")),
    Benchmark("wgmma", "h200/wgmma.cu", exact_sm=90, accelerated=True),
    Benchmark("tcgen05", "b200/tcgen05.cu", exact_sm=100, accelerated=True),
    Benchmark("tmem_write", "b200/tmem.cu", exact_sm=100, accelerated=True,
              flags=("-DREP=128", "-DTEST_MODE=6", "-DN_ITERS=128", "-DTHD_NUM=256")),
    Benchmark("tmem_read", "b200/tmem.cu", exact_sm=100, accelerated=True,
              flags=("-DREP=32", "-DTEST_MODE=7", "-DN_ITERS=512", "-DTHD_NUM=256")),
    Benchmark("tmem_rw", "b200/tmem.cu", exact_sm=100, accelerated=True,
              flags=("-DREP=32", "-DTEST_MODE=8", "-DN_ITERS=512", "-DTHD_NUM=256")),
)


def positive(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def block_counts(value: str) -> list[int]:
    try:
        counts = list(dict.fromkeys(int(item) for item in value.split(",")))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use comma-separated positive block counts") from exc
    if not counts or any(n <= 0 or n > 2147483647 for n in counts):
        raise argparse.ArgumentTypeError("block counts must be positive CUDA grid sizes")
    return counts


def validate_device(info: dict, profile: str) -> int:
    major, minor = map(int, info["compute_capability"].split("."))
    sm = major * 10 + minor
    if sm < 75:
        raise ValueError("This suite supports NVIDIA SM75 and newer; use a compatible nvcc.")
    if profile != "auto":
        name = info["name"].lower()
        matches = ("rtx pro 6000" in name and "blackwell" in name) if profile == "b6000" else profile in name
        if sm != PROFILES[profile] or not matches:
            raise ValueError(f"Requested {profile}, found {info['name']} (SM{sm}); choose the right device or --gpu auto.")
    if "mig" in info["name"].lower():
        raise ValueError("Calibrate a full GPU, not a MIG partition")
    if not re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", info["uuid"]):
        raise ValueError("CUDA did not return a usable GPU UUID")
    return sm


def select_benchmarks(sm: int, names: list[str] | None) -> list[Benchmark]:
    known = {b.name: b for b in BENCHMARKS}
    if names:
        unknown = set(names) - known.keys()
        if unknown:
            raise ValueError(f"Unknown benchmarks: {', '.join(sorted(unknown))}")
        result = [known[name] for name in dict.fromkeys(names)]
        unsupported = [b.name for b in result if not b.supports(sm)]
        if unsupported:
            raise ValueError(f"Not supported on SM{sm}: {', '.join(unsupported)}")
        return result
    return [b for b in BENCHMARKS if b.supports(sm)]


def data_seed(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an unsigned 32-bit integer") from exc
    if not 0 <= number <= 0xffffffff:
        raise argparse.ArgumentTypeError("must be an unsigned 32-bit integer")
    return number


def data_configuration(benchmark: Benchmark, seed: int) -> dict:
    """Operand policy used by the corresponding recorded runtime command."""
    name = benchmark.name
    if name == "dram":
        return {"read_pattern": "index_hash_v1", "write_pattern": "thread_register_hash_v1",
                "seed": seed, "validation": "three_uint4_samples_per_timed_launch"}
    if name in {"l2", "smem", "tmem_write", "tmem_read", "tmem_rw"}:
        return {"pattern": "index_hash_v1", "seed": seed}
    if name in {"wgmma", "tcgen05"}:
        return {"pattern": "nonzero", "input_fp16": 1.0}
    if name == "tensor_gemm":
        return {"pattern": "constant_positive", "input_a": 0.5, "input_b": 0.25}
    if name == "l2_latency":
        return {"pattern": "128_byte_stride_pointer_cycle"}
    return {"pattern": "not_applicable" if name == "launch" else "nonzero_recurrence"}


def run_arguments(benchmark: Benchmark, quick: bool, info: dict | None = None,
                  dram_blocks: list[int] | None = None,
                  seed: int = DEFAULT_DATA_SEED) -> list[str]:
    name = benchmark.name
    if name in CORE_MODES:
        args = [name, "--iters", "256" if quick else "4096", "--repeat", "3" if quick else "7"]
        return args + (["--seed", str(seed)] if name in {"l2", "smem"} else [])
    if name == "dram":
        # Stream over >= 8x L2 to keep the probe out of a warmed cache.
        size = max(256 << 20 if quick else 1 << 30, 8 * (info or {}).get("l2_bytes", 0))
        size = 1 << (size - 1).bit_length()
        if info and 2 * size > info["free_memory_bytes"] * 0.8:
            raise ValueError("Not enough free memory for an out-of-cache DRAM probe")
        sms = (info or {}).get("sm_count", 1)
        # Ensure even the smallest launch traverses the whole allocation.
        iters = max(128 if quick else 1024, math.ceil(size / (sms * 256 * 8 * 16)))
        return ["--bytes", str(size), "--iters", str(iters), "--guard-kb", "0",
                "--repeats", "2" if quick else "5", "--blocks", ",".join(map(str, dram_blocks or [sms, 2*sms, 4*sms])),
                "--methods", "read_cg,write_cs,copy_cg_cs,cp_async", "--seed", str(seed)]
    if name == "tensor_gemm":
        n = "2048" if quick else "8192"
        return ["--m", n, "--n", n, "--k", n, "--warmup", "2", "--repeat", "3" if quick else "10"]
    if name in {"wgmma", "tcgen05"}:
        return ["--iters", "128" if quick else "10000", "--warmup", "2", "--repeat", "3" if quick else "10",
                "--data", "nonzero"]
    if name.startswith("tmem_"):
        return ["--seed", str(seed)]
    return []


def parse_metrics(output: str) -> list[dict]:
    metrics = []
    for line in output.splitlines():
        if not line.startswith("TILESIGHT_METRIC "):
            continue
        _, name, raw, unit = line.split()
        value = float(raw)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid metric {name}: {raw}")
        metrics.append({"name": name, "value": value, "unit": unit})
    if metrics:
        return metrics
    if output.startswith("method,device,"):
        rows = list(csv.DictReader(io.StringIO(output)))
        if not rows:
            raise ValueError("Empty DRAM result table")
        for row in rows:
            if "validation" in row and row["validation"] != "sampled_passed":
                raise ValueError("DRAM payload validation did not pass")
            value = float(row["achieved_ddr_bw_GB_s"])
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Invalid DRAM bandwidth")
            metrics.append({"name": "dram_" + row["method"], "value": value, "unit": "GB/s",
                            "blocks": int(row["blocks"]), "sample": int(row["repeat"]),
                            "time_ms": float(row["time_ms"])})
        return metrics
    raise ValueError("Benchmark produced no recognized measurements")


def execute(command: list[str], log: Path, timeout: int, env: dict | None = None) -> str:
    try:
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                timeout=timeout, env=env, cwd=ROOT)
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or b""
        log.write_text(output.decode(errors="replace") if isinstance(output, bytes) else output)
        raise RuntimeError(f"Timeout after {timeout}s; see {log}") from exc
    log.write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f"Exit {result.returncode}; see {log}")
    return result.stdout


def build(benchmark: Benchmark, sm: int, nvcc: str, nvcc_version: str, out: Path,
          timeout: int, cache: dict) -> tuple[Path, list[str], Path]:
    source = ROOT / "src" / benchmark.source
    flags = ["-O3", "-std=c++17", "-lineinfo", "-Xptxas=-v", benchmark.gencode(sm)]
    digest = hashlib.sha256((nvcc + nvcc_version + repr(flags) + repr(benchmark.flags)).encode())
    digest.update(source.read_bytes())
    headers = set(source.parent.glob("*.hpp")) | set((ROOT / "src/common").glob("*.hpp"))
    for header in sorted(headers):
        digest.update(header.read_bytes())
    key = digest.hexdigest()
    if key in cache:
        return cache[key]
    binary = out / "build" / (source.stem + "-" + key[:12])
    log = out / "build" / (binary.name + ".log")
    command = [nvcc, *flags, str(source), *benchmark.flags, "-o", str(binary)]
    output = execute(command, log, timeout)
    if benchmark.name.startswith("tmem_") and re.search(r"\b[1-9]\d* bytes spill (?:stores|loads)", output):
        raise RuntimeError(f"TMEM benchmark spilled registers; measurement rejected. See {log}")
    cache[key] = binary, command, log
    return cache[key]


def write_report(report: dict, path: Path) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", choices=("auto", *PROFILES), default="auto")
    parser.add_argument("--device", type=int, default=0, help="CUDA ordinal within the current CUDA_VISIBLE_DEVICES mask")
    parser.add_argument("--arch", type=positive, help="SM number for offline auto builds, e.g. 89, 90, 100, 120")
    parser.add_argument("--bench", nargs="+", help="Run only these benchmark names; default: all supported")
    parser.add_argument("--dram-blocks", type=block_counts, help="Custom DRAM concurrency sweep, e.g. 1,2,4,8,16,32,64,128,256,512")
    parser.add_argument("--data-seed", type=data_seed, default=DEFAULT_DATA_SEED,
                        help="Unsigned 32-bit seed for nonuniform DRAM/L2/shared/TMEM payloads")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Short smoke run, not a calibration result")
    parser.add_argument("--dry-run", action="store_true", help="Print an offline plan without compiling or running")
    parser.add_argument("--build-only", action="store_true", help="Cross-compile without requiring a GPU")
    parser.add_argument("--nvcc", default=os.environ.get("NVCC", "nvcc"))
    parser.add_argument("--timeout", type=positive, default=180, help="Timeout in seconds for each process")
    parser.add_argument("--output", type=Path, help="New output directory; existing paths are never overwritten")
    args = parser.parse_args(argv)
    if args.device < 0:
        parser.error("--device must be nonnegative")
    offline_sm = PROFILES.get(args.gpu, args.arch)
    if args.arch is not None and args.gpu != "auto" and args.arch != PROFILES[args.gpu]:
        parser.error("--arch disagrees with --gpu")
    if args.arch is not None and args.arch < 75:
        parser.error("--arch must be SM75 or newer")
    if args.list:
        for b in BENCHMARKS:
            if offline_sm is None or b.supports(offline_sm):
                scope = f"SM{b.exact_sm} only" if b.exact_sm else f"SM{b.minimum_sm}+"
                print(f"{b.name:14} {scope:12} {b.source}")
        return 0
    if (args.dry_run or args.build_only) and offline_sm is None:
        parser.error("Offline operation requires --gpu b200/h200/b6000 or --arch NUMBER")
    if args.arch is not None and not (args.dry_run or args.build_only):
        parser.error("--arch is for offline operation; runtime SM is detected from the GPU")
    if args.dry_run:
        try:
            selected = select_benchmarks(offline_sm, args.bench)
        except ValueError as exc:
            parser.error(str(exc))
        for b in selected:
            command = [args.nvcc, "-O3", "-std=c++17", b.gencode(offline_sm),
                       str(ROOT / "src" / b.source), *b.flags, "-o", f"<build>/{b.name}"]
            print(shlex.join(command))
            print("  run:", shlex.join([f"<build>/{b.name}", *run_arguments(b, args.quick, dram_blocks=args.dram_blocks, seed=args.data_seed)]))
        print("Runtime first validates the visible GPU and pins all probes to its UUID; DRAM sizes and blocks use detected L2/SM counts.")
        return 0

    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    out = (args.output or ROOT / "results" / (stamp + "-" + args.gpu)).resolve()
    try:
        out.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error(f"Output directory already exists: {out}; choose a new path")
    (out / "build").mkdir()
    (out / "logs").mkdir()
    report = {"schema": "tilesight.microbenchmark/1", "created_utc": stamp,
              "mode": "build_only" if args.build_only else "smoke" if args.quick else "measurement",
              "requested_gpu": args.gpu, "requested_device": args.device,
              "data_policy_version": DATA_POLICY_VERSION,
              "original_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "results": [],
              "source_hashes": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted((ROOT / "src").rglob("*")) if p.is_file()}}
    report_path = out / "results.json"
    failures = 0
    try:
        nvcc = shutil.which(args.nvcc)
        if nvcc is None:
            raise ValueError("nvcc not found; install CUDA Toolkit and set NVCC or --nvcc")
        version = execute([nvcc, "--version"], out / "nvcc.log", args.timeout)
        report.update(nvcc=nvcc, nvcc_version=version)
        info = None
        env = dict(os.environ)
        cache = {}
        if not args.build_only:
            probe = out / "build" / "device_info"
            probe_command = [nvcc, "-O2", "-std=c++17", str(ROOT / "src/common/device_info.cu"), "-o", str(probe)]
            execute(probe_command, out / "build/device_info.log", args.timeout)
            info = json.loads(execute([str(probe), "--device", str(args.device)], out / "device.json", args.timeout))
            sm = validate_device(info, args.gpu)
            report["device"] = info
            # The UUID was obtained within the caller's original visibility mask.
            env["CUDA_VISIBLE_DEVICES"] = info["uuid"]
            print(f"GPU: {info['name']} (SM{sm}), visible device {args.device}", flush=True)
        else:
            sm = offline_sm
        report["sm"] = sm
        selected = select_benchmarks(sm, args.bench)
        report["unavailable_benchmarks"] = [b.name for b in BENCHMARKS if not b.supports(sm)]
        write_report(report, report_path)
        for benchmark in selected:
            row = {"benchmark": benchmark.name, "arch": benchmark.arch(sm), "status": "pending",
                   "data_configuration": data_configuration(benchmark, args.data_seed)}
            report["results"].append(row)
            print(f"[{benchmark.name}] build", flush=True)
            try:
                binary, command, log = build(benchmark, sm, nvcc, version, out, args.timeout, cache)
                row.update(build_command=command, build_log=str(log.relative_to(out)))
                if args.build_only:
                    row["status"] = "built"
                else:
                    command = [str(binary), *run_arguments(benchmark, args.quick, info, args.dram_blocks, args.data_seed)]
                    log = out / "logs" / (benchmark.name + ".log")
                    row.update(run_command=command, run_log=str(log.relative_to(out)), device_uuid=info["uuid"])
                    output = execute(command, log, args.timeout, env)
                    row.update(status="passed", metrics=parse_metrics(output))
                    if benchmark.name == "dram":
                        (out / "dram_samples.csv").write_text(output)
                        row["samples_csv"] = "dram_samples.csv"
                print(f"[{benchmark.name}] {row['status']}", flush=True)
            except (OSError, RuntimeError, ValueError) as exc:
                failures += 1
                row.update(status="failed", error=str(exc))
                print(f"[{benchmark.name}] failed: {exc}", file=sys.stderr, flush=True)
            write_report(report, report_path)
    except (OSError, RuntimeError, ValueError) as exc:
        report["error"] = str(exc)
        failures += 1
        print(str(exc), file=sys.stderr)
    except KeyboardInterrupt:
        report["error"] = "Interrupted"
        for row in report["results"]:
            if row["status"] == "pending":
                row["status"] = "interrupted"
        failures += 1
    finally:
        report["status"] = "failed" if failures else "complete"
        write_report(report, report_path)
        with (out / "metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["benchmark", "name", "value", "unit", "blocks", "sample", "time_ms"])
            writer.writeheader()
            for row in report["results"]:
                for metric in row.get("metrics", []):
                    writer.writerow({"benchmark": row["benchmark"], **metric})
        print(f"Results: {report_path}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
