#!/usr/bin/env python3
"""Measure selected local GPUs with CUDA P2P and optional installed communication tests."""
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
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
BACKENDS = ("p2p", "nccl", "nvshmem")
UUID_RE = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}")
P2P_FIELDS = ("src", "dst", "bytes", "can_access_peer", "peer_native_atomics", "status", "time_us", "bandwidth_gb_s")


class Unavailable(RuntimeError):
    """An optional dependency or supported device arrangement is unavailable."""


def devices(value: str) -> list[int]:
    try:
        values = [int(x) for x in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use comma-separated visible CUDA ordinals, e.g. 0,1") from exc
    if len(values) < 2 or min(values) < 0 or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("select at least two distinct nonnegative visible CUDA ordinals")
    return values


def positive(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def finite(value: str, *, zero: bool = False) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0 or (number == 0 and not zero):
        raise ValueError(f"Invalid measurement: {value}")
    return number


def executable(explicit: str | None, env_name: str, default: str) -> str:
    candidate = explicit or os.environ.get(env_name) or default
    resolved = shutil.which(candidate)
    if not resolved:
        raise Unavailable(f"Executable unavailable: {candidate}; use an explicit path or {env_name}")
    return str(Path(resolved).resolve())


def execute(command: list[str], log: Path, timeout: int, env: dict | None = None) -> str:
    """Keep raw output and terminate a launcher's whole process group on timeout."""
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, env=env, cwd=ROOT, start_new_session=True)
    try:
        output, _ = proc.communicate(timeout=timeout)
    except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        output, _ = proc.communicate()
        log.write_text(output)
        if isinstance(exc, KeyboardInterrupt):
            raise
        raise RuntimeError(f"Timeout after {timeout}s; see {log}") from exc
    log.write_text(output)
    if proc.returncode:
        if "error while loading shared libraries" in output:
            raise Unavailable(f"Required shared library unavailable; see {log}")
        raise RuntimeError(f"Exit {proc.returncode}; see {log}")
    return output


def identity(path: str) -> dict:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return {"path": str(source), "sha256": digest.hexdigest(), "bytes": source.stat().st_size}


def pin_devices(infos: list[dict], environment: dict) -> dict:
    uuids = [info["uuid"] for info in infos]
    if len(set(uuids)) != len(uuids) or any(not UUID_RE.fullmatch(item) for item in uuids):
        raise ValueError("Selected devices must have distinct valid CUDA GPU UUIDs")
    if any("mig" in info["name"].lower() for info in infos):
        raise ValueError("Use full GPUs; MIG instances are not supported")
    return {**environment, "CUDA_VISIBLE_DEVICES": ",".join(uuids)}


def resolve_devices(args, out: Path, report: dict) -> tuple[list[dict], dict, str]:
    nvcc = executable(args.nvcc, "NVCC", "nvcc")
    report["nvcc"] = identity(nvcc)
    report["nvcc_version"] = execute([nvcc, "--version"], out / "nvcc.txt", args.timeout)
    source = ROOT / "src/common/device_info.cu"
    report["device_info_source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    probe = out / "build/device_info"
    command = [nvcc, "-O2", "-std=c++17", str(source), "-o", str(probe)]
    report["device_probe_build_command"] = command
    execute(command, out / "build/device_info.txt", args.timeout)
    infos = [json.loads(execute([str(probe), "--device", str(ordinal)],
                               out / f"device_{ordinal}.json", args.timeout)) for ordinal in args.devices]
    env = pin_devices(infos, dict(os.environ))
    report["devices"] = [{**info, "selected_index": i} for i, info in enumerate(infos)]
    report["pinned_cuda_visible_devices"] = env["CUDA_VISIBLE_DEVICES"]
    return infos, env, nvcc


def p2p_csv(output: str) -> str:
    header = ",".join(P2P_FIELDS)
    lines = output.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.strip() == header:
            return "".join(lines[index:])
    raise ValueError("Unrecognized P2P CSV header")


def parse_p2p(output: str, count: int) -> list[dict]:
    reader = csv.DictReader(io.StringIO(p2p_csv(output)))
    if tuple(reader.fieldnames or ()) != P2P_FIELDS:
        raise ValueError("Unrecognized P2P CSV header")
    rows = []
    for raw in reader:
        if None in raw or any(raw.get(field) is None for field in P2P_FIELDS):
            raise ValueError("Malformed P2P CSV row")
        src, dst, size = int(raw["src"]), int(raw["dst"]), int(raw["bytes"])
        access, atomics = int(raw["can_access_peer"]), int(raw["peer_native_atomics"])
        if src == dst or not (0 <= src < count and 0 <= dst < count) or size < 0 or access not in (0, 1) or atomics not in (0, 1):
            raise ValueError("Invalid P2P identity/capability row")
        row = {"src": src, "dst": dst, "bytes": size, "can_access_peer": access,
               "peer_native_atomics": atomics, "status": raw["status"]}
        if raw["status"] == "measured":
            if not access or size == 0:
                raise ValueError("Measured P2P row lacks peer access or payload")
            row.update(time_us=finite(raw["time_us"]), bandwidth_gb_s=finite(raw["bandwidth_gb_s"]))
        elif raw["status"] == "unsupported":
            if access or raw["time_us"] or raw["bandwidth_gb_s"]:
                raise ValueError("Unsupported P2P row must not contain measurements")
        else:
            raise ValueError(f"P2P did not measure successfully: {raw['status']}")
        rows.append(row)
    if not rows:
        raise ValueError("No P2P result rows")
    expected = {(src, dst) for src in range(count) for dst in range(count) if src != dst}
    if {(row["src"], row["dst"]) for row in rows} != expected:
        raise ValueError("P2P output is missing requested directed pairs")
    return rows


def parse_nccl(output: str) -> list[dict]:
    # all_reduce_perf's standard table has two independent four-column results.
    if not re.search(r"\(us\)", output) or "algbw" not in output or "busbw" not in output:
        raise ValueError("NCCL output lacks the expected microsecond/bandwidth header")
    summary = re.search(r"Out of bounds values\s*:\s*(\d+)\s+OK", output)
    if not summary or int(summary[1]) != 0:
        raise ValueError("NCCL correctness summary is missing or reports errors")
    rows = []
    for line in output.splitlines():
        columns = line.split()
        if not columns or not columns[0].isdigit():
            continue
        if len(columns) != 13:
            raise ValueError("Unrecognized NCCL result columns; raw output retained")
        size, count = int(columns[0]), int(columns[1])
        if size <= 0 or count * 4 != size or columns[2:5] != ["float", "sum", "-1"]:
            raise ValueError("NCCL result identity differs from requested float sum AllReduce")
        for placement, start in (("out_of_place", 5), ("in_place", 9)):
            wrong = int(columns[start + 3])
            if wrong != 0:
                raise ValueError(f"NCCL correctness failure: #wrong={wrong}")
            rows.append({"bytes": size, "count": count, "dtype": "float", "op": "sum",
                         "placement": placement, "time_us": finite(columns[start]),
                         "algbw_gb_s": finite(columns[start + 1], zero=True),
                         "busbw_gb_s": finite(columns[start + 2], zero=True), "wrong": wrong})
    if not rows:
        raise ValueError("NCCL produced no measurements")
    return rows


def parse_nvshmem(output: str, kind: str) -> list[dict]:
    # NVIDIA's machine-readable PERF / PERF_STATS formats preserve native scopes.
    rows = []
    for line in output.splitlines():
        match = re.fullmatch(r"&&&& (PERF|PERF_STATS) (\S+)___(\S+)___size__(\d+)___(\S+) (.+)", line.strip())
        if not match:
            continue
        form, benchmark, scope, size, metric, tail = match.groups()
        expected_metric, expected_unit = ("latency", "us") if kind == "latency" else ("BW", "GB/sec")
        valid_name = benchmark == "shmem_put_latency" if kind == "latency" else benchmark in {"shmem_put_bw", "shmem_put_bw_uni"}
        if not valid_name or metric != expected_metric:
            raise ValueError("NVSHMEM benchmark/scope differs from the requested device put test")
        fields = tail.split()
        if form == "PERF":
            if len(fields) != 2 or fields[1][1:] != expected_unit:
                raise ValueError("Unexpected NVSHMEM measurement unit")
            value = finite(fields[0])
            stats = {}
        else:
            if fields[0][1:] != expected_unit:
                raise ValueError("Unexpected NVSHMEM measurement unit")
            stats = dict(item.split("=", 1) for item in fields[1:])
            value = finite(stats["mean"])
            finite(stats["min"])
            finite(stats["max"])
            if stats.get("stddev") != "NA":
                finite(stats["stddev"], zero=True)
            if int(stats["repetitions"]) <= 0:
                raise ValueError("NVSHMEM has no timed repetitions")
        if int(size) <= 0:
            raise ValueError("NVSHMEM has an invalid size")
        rows.append({"benchmark": benchmark, "scope": scope, "bytes": int(size),
                     "metric": metric, "value": value, "unit": expected_unit, "statistics": stats})
    if not rows:
        raise ValueError("No recognized NVSHMEM device-put measurements; raw output retained")
    return rows


def bounds(args) -> tuple[int, int, int, int]:
    return args.min_bytes, args.max_bytes or ((1 << 20) if args.quick else (64 << 20)), args.iters or (10 if args.quick else 100), (2 if args.quick else 10)


def p2p_command(binary: str, args) -> list[str]:
    minimum, maximum, iters, warmup = bounds(args)
    return [binary, "--devices", ",".join(map(str, range(len(args.devices)))),
            "--min-bytes", str(minimum), "--max-bytes", str(maximum), "--factor", "2",
            "--iters", str(iters), "--warmup", str(warmup)]


def nccl_command(binary: str, args) -> list[str]:
    minimum, maximum, iters, warmup = bounds(args)
    return [binary, "-b", str(minimum), "-e", str(maximum), "-f", "2", "-g", str(len(args.devices)),
            "-t", "1", "-n", str(iters), "-w", str(warmup), "-c", "1", "-d", "float", "-o", "sum"]


def nvshmem_command(launcher: str, binary: str, args, kind: str, help_text: str) -> list[str]:
    minimum, maximum, iters, warmup = bounds(args)
    required = ("-b", "-e", "-f", "-n", "-w") + (("-c", "-t") if kind == "bandwidth" else ())
    if any(not re.search(r"(?<!\w)" + re.escape(flag) + r"(?:\W|$)", help_text) for flag in required):
        raise Unavailable("Installed NVSHMEM benchmark does not advertise the supported CLI; see help log")
    options = []
    if kind == "bandwidth":
        # One operation per CTA; each operation contains at least one double.
        minimum = max(minimum, 8 * args.nvshmem_ctas)
        if minimum > maximum:
            raise ValueError("Maximum bytes are too small for the NVSHMEM CTA count")
        options += ["-c", str(args.nvshmem_ctas), "-t", "256"]
        if "--scope" in help_text:
            options += ["-s", "block"]
    if "--use_smem" in help_text:
        options += ["--use_smem", "0"]
    return [launcher, "-n", "2", "-ppn", "2", "-hosts", "localhost", binary,
            "-b", str(minimum), "-e", str(maximum), "-f", "2", "-n", str(iters), "-w", str(warmup), *options]


def backend_tools(name: str, args) -> dict[str, str]:
    if name == "p2p":
        return {}
    if name == "nccl":
        if os.environ.get("NCCL_TESTS_SPLIT") or os.environ.get("NCCL_TESTS_SPLIT_MASK"):
            raise Unavailable("Unset NCCL_TESTS_SPLIT/SPLIT_MASK to measure all selected GPUs as one group")
        return {"all_reduce": executable(args.nccl_all_reduce, "NCCL_ALL_REDUCE_PERF", "all_reduce_perf")}
    if len(args.devices) != 2:
        raise Unavailable("NVSHMEM put latency adapter requires exactly two selected GPUs")
    return {"launcher": executable(args.nvshmem_launcher, "NVSHMEM_LAUNCHER", "nvshmrun"),
            "bandwidth": executable(args.nvshmem_put_bw, "NVSHMEM_PUT_BW", "shmem_put_bw"),
            "latency": executable(args.nvshmem_put_latency, "NVSHMEM_PUT_LATENCY", "shmem_put_latency")}


def run_backend(name: str, paths: dict, args, out: Path, env: dict, nvcc: str, row: dict) -> None:
    row["tools"] = {key: identity(path) for key, path in paths.items()}
    row["commands"] = []
    if name == "p2p":
        source = ROOT / "src/common/p2p.cu"
        binary = str(out / "build/p2p")
        command = [nvcc, "-O3", "-std=c++17", str(source), "-o", binary]
        row["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        row["build_command"] = command
        execute(command, out / "build/p2p.txt", args.timeout)
        command = p2p_command(binary, args)
        row["commands"].append(command)
        output = execute(command, out / "p2p.txt", args.timeout, env)
        (out / "p2p.csv").write_text(p2p_csv(output))
        row["measurements"] = parse_p2p(output, len(args.devices))
        minimum, maximum, _, _ = bounds(args)
        sizes = []
        size = minimum
        while size <= maximum:
            sizes.append(size)
            size *= 2
        expected = {(src, dst, size) for src in range(len(args.devices))
                    for dst in range(len(args.devices)) if src != dst for size in sizes}
        actual = [(item["src"], item["dst"], item["bytes"]) for item in row["measurements"]]
        if len(actual) != len(expected) or set(actual) != expected:
            raise ValueError("P2P output is incomplete or contains duplicate size/pair rows")
        row["timing_scope"] = "CUDA source-stream event elapsed per cudaMemcpyPeerAsync, including submission/queue overhead"
        row["correctness"] = "every transferred byte checked by the probe"
        measured = [item for item in row["measurements"] if item["status"] == "measured"]
        row["status"] = "measured" if len(measured) == len(row["measurements"]) else "partial" if measured else "unsupported"
    elif name == "nccl":
        command = nccl_command(paths["all_reduce"], args)
        row["commands"].append(command)
        row["measurements"] = parse_nccl(execute(command, out / "nccl.txt", args.timeout, env))
        row.update(status="measured", correctness="-c 1; all row #wrong counts and final out-of-bounds summary are zero",
                   timing_scope="nccl-tests AllReduce device operation time; out-of-place and in-place separately")
    else:
        row["measurements"] = []
        child_env = {**env, "NVSHMEM_MACHINE_READABLE_OUTPUT": "1"}
        # The documented Hydra launcher uses PMI. Respect an existing explicit bootstrap choice.
        child_env.setdefault("NVSHMEM_BOOTSTRAP_PMI", "PMI")
        row["environment_overrides"] = {"NVSHMEM_MACHINE_READABLE_OUTPUT": "1", "NVSHMEM_BOOTSTRAP_PMI": child_env["NVSHMEM_BOOTSTRAP_PMI"]}
        row["timing_scope"] = "Installed NVIDIA device/pt-to-pt put benchmarks; native thread/warp/block API timing, not pure fabric latency"
        row["correctness"] = "not independently verified; installed perftest exit status and recognized native measurements only"
        for kind in ("latency", "bandwidth"):
            help_text = execute([paths[kind], "--help"], out / f"nvshmem_{kind}_help.txt", args.timeout, env)
            command = nvshmem_command(paths["launcher"], paths[kind], args, kind, help_text)
            row["commands"].append(command)
            output = execute(command, out / f"nvshmem_{kind}.txt", args.timeout, child_env)
            row["measurements"].extend({"kind": kind, **item} for item in parse_nvshmem(output, kind))
        row["status"] = "measured"


def overall_status(rows: list[dict]) -> tuple[str, int]:
    measured = any(row["status"] in {"measured", "partial"} for row in rows)
    if any(row["status"] == "failed" for row in rows):
        return "failed", 1
    if not measured:
        return "unavailable", 1
    if any(row["status"] != "measured" for row in rows):
        return "partial", 0
    return "complete", 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", type=devices, required=True, help="Ordinals inside the inherited CUDA_VISIBLE_DEVICES mask")
    parser.add_argument("--backend", choices=(*BACKENDS, "all"), default="p2p")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--min-bytes", type=positive, default=8)
    parser.add_argument("--max-bytes", type=positive)
    parser.add_argument("--iters", type=positive)
    parser.add_argument("--nvshmem-ctas", type=positive, default=32)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timeout", type=positive, default=300)
    parser.add_argument("--nvcc")
    parser.add_argument("--nccl-all-reduce")
    parser.add_argument("--nvshmem-launcher", help="Installed Hydra nvshmrun launcher")
    parser.add_argument("--nvshmem-put-bw", help="Installed device/pt-to-pt/shmem_put_bw")
    parser.add_argument("--nvshmem-put-latency", help="Installed device/pt-to-pt/shmem_put_latency")
    args = parser.parse_args(argv)
    minimum, maximum, _, _ = bounds(args)
    if minimum > maximum or minimum < 8 or any(x & (x - 1) for x in (minimum, maximum, args.nvshmem_ctas)):
        parser.error("byte bounds and NVSHMEM CTA count must be powers of two; use 8 <= min <= max")
    requested = BACKENDS if args.backend == "all" else (args.backend,)
    if args.dry_run:
        print("Resolve visible CUDA ordinals in order:", args.devices)
        print("Then pin child CUDA_VISIBLE_DEVICES to their resolved UUIDs (no GPU queries in dry run).")
        for name in requested:
            if name == "p2p":
                print(shlex.join(p2p_command("<build>/p2p", args)))
            elif name == "nccl":
                print(shlex.join(nccl_command(args.nccl_all_reduce or os.environ.get("NCCL_ALL_REDUCE_PERF", "all_reduce_perf"), args)))
            else:
                help_flags = "-b -e -f -n -w -c -t --scope --use_smem"
                for kind in ("latency", "bandwidth"):
                    binary = (args.nvshmem_put_latency if kind == "latency" else args.nvshmem_put_bw) or f"shmem_put_{'latency' if kind == 'latency' else 'bw'}"
                    print(shlex.join(nvshmem_command(args.nvshmem_launcher or "nvshmrun", binary, args, kind, help_flags)))
                print("NVSHMEM requires exactly two GPUs; actual flags are checked against installed --help.")
        return 0
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    out = (args.output or ROOT / "results" / (stamp + "-multi_gpu")).resolve()
    try:
        out.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error(f"Output path already exists: {out}")
    (out / "build").mkdir()
    rows = [{"backend": name, "status": "pending"} for name in requested]
    report = {"schema": "tilesight.multi_gpu/1", "created_utc": stamp, "mode": "smoke" if args.quick else "measurement",
              "requested_devices": args.devices, "original_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "communication_environment": {key: value for key, value in os.environ.items() if key.startswith(("NCCL_", "NVSHMEM_", "NVSHMEMTEST_"))},
              "results": rows}
    paths = {}
    code = 1
    try:
        for row in rows:
            try:
                paths[row["backend"]] = backend_tools(row["backend"], args)
            except Unavailable as exc:
                row.update(status="unavailable", reason=str(exc))
        if any(row["status"] == "pending" for row in rows):
            _infos, env, nvcc = resolve_devices(args, out, report)
            topology = shutil.which("nvidia-smi")
            if topology:
                try:
                    execute([topology, "topo", "-m"], out / "topology.txt", min(args.timeout, 30), env)
                    execute([topology, "--query-gpu=index,uuid,pci.bus_id,name", "--format=csv"], out / "topology_devices.csv", min(args.timeout, 30), env)
                    report["topology"] = "topology.txt (physical indices; map UUIDs using topology_devices.csv)"
                except (OSError, RuntimeError) as exc:
                    report["topology_error"] = str(exc)
            else:
                report["topology_error"] = "nvidia-smi unavailable"
            for row in rows:
                if row["status"] != "pending":
                    continue
                try:
                    run_backend(row["backend"], paths[row["backend"]], args, out, env, nvcc, row)
                except Unavailable as exc:
                    row.update(status="unavailable", reason=str(exc))
                except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
                    row.update(status="failed", reason=str(exc))
        report["status"], code = overall_status(rows)
        if args.backend != "all" and rows[0]["status"] in {"unavailable", "unsupported"}:
            code = 1
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        for row in rows:
            if row["status"] == "pending":
                row.update(status="unavailable" if isinstance(exc, Unavailable) else "failed", reason=str(exc))
        report["status"], code = overall_status(rows)
    except KeyboardInterrupt:
        for row in rows:
            if row["status"] == "pending":
                row.update(status="failed", reason="Interrupted")
        report["status"] = "failed"
    finally:
        (out / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        for row in rows:
            print(f"[{row['backend']}] {row['status']}: {row.get('reason', '')}")
        print(f"Results: {out / 'results.json'}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
