"""Nsight Compute observations: request contract, metric registry, CSV import and capture.

A request names one exact kernel launch and the command that runs it; capture
runs NCU once (no shell, no clock or permission changes) and keeps the raw
report, its CSV export and the observation JSON. Missing or unsupported
metrics stay ``None``; they are never zero.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

SCHEMA = "tilesight-ncu-observation/v2"
REQUEST_SCHEMA = "tilesight-ncu-request/v2"


def digest(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _strict_pairs(items: Iterable[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result


def read_json(path: Any) -> Any:
    def reject(constant: str) -> None:
        raise ValueError("non-finite JSON constant: " + constant)
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=_strict_pairs, parse_constant=reject)


def write_json(path: Any, value: Any) -> None:
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def artifact(path: Any) -> Dict[str, Any]:
    value = Path(path)
    return {"path": str(value.resolve()), "sha256": hashlib.sha256(value.read_bytes()).hexdigest(),
            "bytes": value.stat().st_size}


# --------------------------------------------------------------------------- metric registry

@dataclass(frozen=True)
class MetricSpec:
    name: str
    key: str
    unit: str
    category: str = "C"        # A launch facts, B memory hierarchy, C everything else
    layer: str = "kernel"
    direction: str = "none"
    path: str = "all"
    denominator: Optional[str] = None


def _registry() -> Tuple[MetricSpec, ...]:
    values: List[MetricSpec] = []

    def add(name: str, key: str, unit: str, category: str = "C", layer: str = "kernel", direction: str = "none",
            path: str = "all", denominator: Optional[str] = None) -> None:
        values.append(MetricSpec(name, key, unit, category, layer, direction, path, denominator))

    for shape in ("grid", "block", "cluster"):
        for axis in "xyz":
            add("launch__%s_dim_%s" % (shape, axis), "%s_%s" % (shape, axis), "count", "A")
    for name in ("registers_per_thread", "registers_per_thread_allocated", "sm_count"):
        add("launch__" + name, name, "count", "A")
    add("launch__cluster_size", "cluster_size", "count", "A")
    for name in ("static", "dynamic", "driver", "allocated"):
        add("launch__shared_mem_per_block_" + name, "smem_" + name, "byte", "A")
    for name in ("blocks", "registers", "shared_mem", "warps", "barriers"):
        add("launch__occupancy_limit_" + name, "cta_limit_" + name, "count", "A")
    add("device__attribute_compute_capability_major", "cc_major", "count", "A")
    add("device__attribute_compute_capability_minor", "cc_minor", "count", "A")
    add("gpu__time_duration.sum", "duration_s", "second")
    for direction, op in (("read", "ld"), ("write", "st")):
        add("dram__bytes_%s.sum" % direction, "dram_%s_bytes" % direction, "byte", "B", "ddr", direction)
        add("lts__t_sectors_op_%s.sum" % direction, "l2_%s_sectors" % direction, "sector", "B", "l2", direction)
        for kind, unit in (("requests", "request"), ("sectors", "sector")):
            add("l1tex__t_%s_pipe_lsu_mem_global_op_%s.sum" % (kind, op), "global_%s_%s" % (direction, kind), unit, "B",
                "l1tex", direction, "lsu")
        add("l1tex__t_sectors_pipe_lsu_mem_global_op_%s_lookup_hit.sum" % op, "l1tex_%s_hit_sectors" % direction, "sector",
            "B", "l1tex", direction, "lsu", "global_%s_sectors" % direction)
        add("lts__t_sectors_op_%s_lookup_hit.sum" % direction, "l2_%s_hit_sectors" % direction, "sector", "B", "l2",
            direction, "all", "l2_%s_sectors" % direction)
    add("lts__t_sector_hit_rate.pct", "l2_hit_pct", "percent", "B", "l2", denominator="l2_all_sectors")
    add("lts__t_sectors.sum", "l2_all_sectors", "sector", "B", "l2")
    add("sm__warps_active.avg.pct_of_peak_sustained_active", "achieved_occupancy_pct", "percent")
    add("sm__warps_active.avg.per_cycle_active", "active_warps", "count")
    add("sm__maximum_warps_per_active_cycle_pct", "theoretical_occupancy_pct", "percent")
    add("sm__inst_executed.avg.per_cycle_active", "ipc", "count")
    for pipe in ("tensor", "fma", "sfu"):
        add("sm__pipe_%s_cycles_active.avg.pct_of_peak_sustained_elapsed" % pipe, "%s_utilization_pct" % pipe, "percent")
    for layer in ("dram", "l1tex", "lts"):
        add("%s__throughput.avg.pct_of_peak_sustained_elapsed" % layer, "%s_bandwidth_pct" % layer, "percent", layer=layer)
    for stall in ("barrier", "long_scoreboard", "short_scoreboard", "wait", "not_selected"):
        add("smsp__warp_issue_stalled_%s_per_warp_active.pct" % stall, "stall_%s_pct" % stall, "percent")
    for direction, op in (("read", "ld"), ("write", "st")):
        add("l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_%s.sum" % op, "bank_conflicts_%s" % direction, "count",
            layer="shared")
        add("l1tex__t_sectors_pipe_lsu_mem_local_op_%s.sum" % op, "local_%s_sectors" % direction, "sector",
            layer="local", direction=direction)
    add("l1tex__data_pipe_lsu_wavefronts_mem_shared.sum", "shared_wavefronts", "count", layer="shared")
    add("lts__t_sectors_op_atom.sum", "atomic_sectors", "sector", layer="l2")
    add("smsp__inst_executed.sum", "dynamic_instructions", "count")
    add("smsp__thread_inst_executed_pred_on.sum", "predicate_on_instructions", "count")
    add("smsp__sass_branch_targets_threads_divergent.sum", "divergent_branch_threads", "count")
    add("sm__cycles_elapsed.avg.per_second", "sm_frequency_hz", "hertz")
    add("lts__cycles_elapsed.avg.per_second", "l2_frequency_hz", "hertz")
    add("dram__cycles_elapsed.avg.per_second", "dram_frequency_hz", "hertz")
    for stat in ("min", "max", "avg"):
        add("sm__cycles_active.%s" % stat, "sm_active_cycles_%s" % stat, "cycle")
    add("sm__inst_executed_pipe_tma.sum", "tma_instructions", "count", "B", "tma", path="tma")
    return tuple(values)


METRICS = _registry()
BY_KEY = {spec.key: spec for spec in METRICS}


def metric(observation: Dict[str, Any], key: str) -> Optional[float]:
    record = observation.get("metrics", {}).get(key, {})
    return record.get("value") if record.get("status") == "supported" else None


# --------------------------------------------------------------------------- contracts

def validate_request(request: Dict[str, Any]) -> Dict[str, Any]:
    if request.get("schema_version") != REQUEST_SCHEMA:
        raise ValueError("expected " + REQUEST_SCHEMA)
    target = request.get("target", {})
    if not isinstance(target.get("symbol"), str) or not target["symbol"] or target["symbol"].startswith("regex:"):
        raise ValueError("target symbol must be a nonempty exact kernel name")
    if target.get("name_base") not in {"function", "demangled", "mangled"}:
        raise ValueError("explicit target name_base required")
    if type(target.get("launch_skip")) is not int or target["launch_skip"] < 0:
        raise ValueError("launch_skip must be a nonnegative integer")
    if not request.get("device", {}).get("uuid", "").startswith("GPU-"):
        raise ValueError("device GPU UUID required")
    config = request.get("capture", {})
    if config.get("cache_control") not in {"all", "none"} or config.get("clock_control") not in {"none", "base"}:
        raise ValueError("explicit cache and clock controls required")
    if config.get("replay_mode") not in {"kernel", "application"}:
        raise ValueError("explicit replay mode required")
    if type(config.get("warmup")) is not int or config["warmup"] < 0:
        raise ValueError("warmup must be declared by the application driver")
    if config.get("profile_from_start") not in {"on", "off"}:
        raise ValueError("profile_from_start must be explicit")
    command = request.get("command")
    if not isinstance(command, list) or not command or any(not isinstance(v, str) for v in command):
        raise ValueError("command must be an argv list")
    keys = request.get("metric_keys")
    if keys is not None and (not isinstance(keys, list) or not keys or len(keys) != len(set(keys))
                             or any(k not in BY_KEY for k in keys)):
        raise ValueError("metric_keys must be a nonempty unique list of registered keys")
    return request


def validate_observation(observation: Dict[str, Any]) -> Dict[str, Any]:
    if observation.get("schema_version") != SCHEMA:
        raise ValueError("expected " + SCHEMA)
    validate_request(observation["request"])
    if observation.get("request_digest") != digest(observation["request"]):
        raise ValueError("request digest mismatch")
    if observation.get("status") not in {"complete", "partial", "failed"}:
        raise ValueError("invalid observation status")
    for key, item in observation.get("metrics", {}).items():
        if item.get("status") not in {"supported", "missing", "invalid", "unsupported"}:
            raise ValueError("invalid metric status: " + key)
        value = item.get("value")
        if item["status"] == "supported" and (type(value) not in {int, float} or not math.isfinite(value) or value < 0):
            raise ValueError("invalid metric value: " + key)
        if item["status"] != "supported" and value is not None:
            raise ValueError("unavailable metric must be null: " + key)
    return observation


# --------------------------------------------------------------------------- CSV import

_UNITS = {
    "byte": ("byte", 1), "Kbyte": ("byte", 1e3), "Mbyte": ("byte", 1e6), "Gbyte": ("byte", 1e9),
    "sector": ("sector", 1), "Ksector": ("sector", 1e3), "Msector": ("sector", 1e6), "Gsector": ("sector", 1e9),
    "request": ("request", 1), "Krequest": ("request", 1e3), "Mrequest": ("request", 1e6),
    "second": ("second", 1), "msecond": ("second", 1e-3), "usecond": ("second", 1e-6), "nsecond": ("second", 1e-9),
    "s": ("second", 1), "ms": ("second", 1e-3), "us": ("second", 1e-6), "ns": ("second", 1e-9),
    "%": ("percent", 1), "cycle": ("cycle", 1), "Kcycle": ("cycle", 1e3), "Mcycle": ("cycle", 1e6), "Gcycle": ("cycle", 1e9),
    "cycle/second": ("hertz", 1), "cycle/nsecond": ("hertz", 1e9), "cycle/usecond": ("hertz", 1e6),
    "Hz": ("hertz", 1), "hz": ("hertz", 1), "Khertz": ("hertz", 1e3), "Mhertz": ("hertz", 1e6), "Ghertz": ("hertz", 1e9),
}
_COUNT_UNITS = {"", "register/thread", "register", "block", "warp", "warp/cycle", "inst", "inst/cycle", "thread",
                "wavefront", "bank conflict", "SM", "branches"}


def normalize(value: str, unit: str, expected: str) -> Tuple[Optional[float], str, Optional[str]]:
    if value.strip() in {"", "n/a", "N/A", "nan", "-"}:
        return None, "missing", "NCU returned no numeric value"
    try:
        number = float(value.replace(",", ""))
    except ValueError:
        return None, "invalid", "non-numeric counter"
    if not math.isfinite(number) or number < 0:
        return None, "invalid", "non-finite or negative counter"
    if expected == "count" and unit in _COUNT_UNITS:
        return number, "supported", None
    if expected == "request" and unit == "":
        return number, "supported", None   # NCU raw pages print LSU request counts without a unit
    converted = _UNITS.get(unit[:-len("/block")] if expected == "byte" and unit.endswith("/block") else unit)
    if converted is None or converted[0] != expected:
        return None, "invalid", "unit mismatch: %r -> %s" % (unit, expected)
    return number * converted[1], "supported", None


def parse_csv(text: str) -> Tuple[Dict[str, str], Dict[str, Tuple[str, str]]]:
    """One launch of an ``ncu --page raw --csv`` export (long or wide layout)."""

    try:
        rows = list(csv.reader(io.StringIO(text.lstrip("﻿")), strict=True))
    except csv.Error as error:
        raise ValueError("NCU CSV structural error: %s" % error) from error
    headers = [i for i, row in enumerate(rows) if "ID" in row and "Kernel Name" in row]
    if len(headers) != 1:
        raise ValueError("NCU CSV requires exactly one raw table header")
    header = rows[headers[0]]
    if len(header) != len(set(header)):
        raise ValueError("duplicate CSV columns")
    data, footer = [], False
    for number, row in enumerate(rows[headers[0] + 1:], start=headers[0] + 2):
        if not row or (len(row) == 1 and not row[0].strip()):
            continue
        if len(row) == 1 and row[0].startswith("==PROF=="):
            footer = True
            continue
        if footer:
            raise ValueError("NCU CSV structural error at record %d: data after footer" % number)
        if len(row) != len(header):
            raise ValueError("NCU CSV structural error at record %d: expected %d columns, got %d"
                             % (number, len(header), len(row)))
        data.append(row)
    identity_keys = ("ID", "Process ID", "Kernel Name", "Context", "Stream", "Device")
    long_layout = "Metric Name" in header
    if long_layout:
        if not {"Metric Name", "Metric Value", "Metric Unit"} <= set(header):
            raise ValueError("NCU long CSV metric columns missing")
        records = [dict(zip(header, row)) for row in data]
    else:
        if not data or data[0][header.index("ID")] != "":
            raise ValueError("NCU wide CSV unit row missing")
        units = dict(zip(header, data[0]))
        if any(units.get(k, "") for k in identity_keys):
            raise ValueError("NCU wide CSV unit row has nonempty launch identity")
        records = [dict(zip(header, row)) for row in data[1:]]
    if any(not r.get("ID", "").isdigit() or not r.get("Kernel Name") or (long_layout and not r.get("Metric Name"))
           for r in records):
        raise ValueError("NCU CSV structural error: missing or invalid launch/metric identity")
    launches = {tuple(r.get(k, "") for k in identity_keys) for r in records}
    if len(launches) != 1:
        raise ValueError("expected one kernel launch, found %d" % len(launches))
    identity = dict(zip(identity_keys, next(iter(launches))))
    raw: Dict[str, Tuple[str, str]] = {}
    if long_layout:
        for row in records:
            value = (row.get("Metric Value", ""), row.get("Metric Unit", ""))
            if row["Metric Name"] in raw and raw[row["Metric Name"]] != value:
                raise ValueError("conflicting repeated metric: " + row["Metric Name"])
            raw[row["Metric Name"]] = value
    else:
        if len(records) != 1:
            raise ValueError("duplicate wide launch rows")
        raw = {name: (value, units[name]) for name, value in records[0].items() if "__" in name}
    return identity, raw


def import_csv(path: Any, request: Dict[str, Any], *, provenance: Optional[Dict[str, Any]] = None,
               available: Optional[set] = None) -> Dict[str, Any]:
    validate_request(request)
    identity, raw = parse_csv(Path(path).read_text(encoding="utf-8-sig"))
    if identity["Kernel Name"] != request["target"]["symbol"]:
        raise ValueError("captured kernel %r differs from the requested %r" % (identity["Kernel Name"],
                                                                            request["target"]["symbol"]))
    metrics = {}
    for spec in METRICS:
        value, unit = raw.get(spec.name, ("", ""))
        number, status, reason = normalize(value, unit, spec.unit)
        if spec.name not in raw and available is not None and spec.name not in available:
            status, reason = "unsupported", "not advertised by the queried NCU/device"
        elif spec.name not in raw and request.get("metric_keys") is not None and spec.key not in request["metric_keys"]:
            status, reason = "missing", "not requested by metric_keys"
        metrics[spec.key] = {**asdict(spec), "value": number, "status": status, "reason": reason,
                             "raw_value": value or None, "raw_unit": unit or None, "scope": "kernel_launch",
                             "origin": str(Path(path).resolve())}
    complete = all(v["status"] == "supported" for v in metrics.values())
    return {"schema_version": SCHEMA, "status": "complete" if complete else "partial", "request": request,
            "request_digest": digest(request), "launch": identity, "metrics": metrics, "artifacts": [artifact(path)],
            "provenance": provenance or {}, "diagnostics": [],
            "unregistered_metrics": sorted(set(raw) - {s.name for s in METRICS})}


# --------------------------------------------------------------------------- capture

def _run(argv: List[str], *, env: Optional[Dict[str, str]] = None, timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run(argv, env=env, capture_output=True, text=True, timeout=timeout, check=False)


def _available_metrics(ncu: str, directory: Path, env: Dict[str, str]) -> set:
    result = _run([ncu, "--query-metrics", "--query-metrics-mode", "all", "--csv"], env=env)
    (directory / "metric-query.csv").write_text(result.stdout, encoding="utf-8")
    if result.returncode:
        raise RuntimeError("NCU metric query failed (exit %d): %s" % (result.returncode, result.stderr.strip()[-300:]))
    rows = list(csv.reader(io.StringIO(result.stdout)))
    header = next((i for i, row in enumerate(rows) if "Metric name" in row), None)
    if header is None:
        raise ValueError("NCU metric query CSV has no 'Metric name' column")
    column = rows[header].index("Metric name")
    return {row[column] for row in rows[header + 1:] if len(row) > column}


def capture(request: Dict[str, Any], output: Any, *, ncu: str = "ncu", timeout: int = 600) -> Dict[str, Any]:
    """Run NCU on ``request["command"]`` and write ``observation.json`` into a new directory."""

    validate_request(request)
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "request.json", request)
    config = request["capture"]
    env = {**os.environ, **request.get("environment", {}), "CUDA_VISIBLE_DEVICES": request["device"]["uuid"]}
    provenance: Dict[str, Any] = {"collector": "Nsight Compute", "exit_codes": {}}
    observation: Dict[str, Any] = {"schema_version": SCHEMA, "status": "failed", "request": request,
                                   "request_digest": digest(request), "metrics": {}, "artifacts": [],
                                   "provenance": provenance, "diagnostics": []}
    try:
        version = _run([ncu, "--version"], env=env)
        provenance["ncu_version"] = version.stdout.strip()
        provenance["exit_codes"]["version"] = version.returncode
        if version.returncode:
            raise RuntimeError("NCU version query failed")
        available = _available_metrics(ncu, directory, env)
        requested = request.get("metric_keys") or [s.key for s in METRICS]
        selected = [s.name for s in METRICS if s.name in available and s.key in requested]
        provenance["metric_selection"] = {"requested_keys": requested, "selected_names": selected,
                                          "software_section": "LaunchStats"}
        if not selected:
            raise RuntimeError("NCU advertises none of the registered metrics")
        target = request["target"]
        argv = [ncu, "--config-file", "0", "--target-processes", "application-only",
                "--kernel-name-base", target["name_base"], "--kernel-name", target["symbol"],
                "--launch-skip", str(target["launch_skip"]), "--launch-count", "1",
                "--profile-from-start", config["profile_from_start"], "--replay-mode", config["replay_mode"],
                "--cache-control", config["cache_control"], "--clock-control", config["clock_control"],
                "--section", "LaunchStats", "--metrics", ",".join(selected), "--export", str(directory / "capture"),
                *request["command"]]
        provenance["command"] = argv
        result = _run(argv, env=env, timeout=timeout)
        (directory / "capture.stdout").write_text(result.stdout, encoding="utf-8")
        (directory / "capture.stderr").write_text(result.stderr, encoding="utf-8")
        provenance["exit_codes"]["capture"] = result.returncode
        if result.returncode:
            raise RuntimeError("NCU capture failed (exit %d); see capture.stderr" % result.returncode)
        receipt = request.get("driver_receipt_path")
        if receipt:
            provenance["driver_receipt"] = read_json(receipt)
            provenance["driver_receipt_artifact"] = artifact(receipt)
        report = directory / "capture.ncu-rep"
        if not report.is_file():
            raise RuntimeError("NCU returned success without a report")
        exported = _run([ncu, "--import", str(report), "--page", "raw", "--csv", "--print-units", "base",
                         "--kernel-name-base", target["name_base"]], env=env)
        provenance["exit_codes"]["import"] = exported.returncode
        csv_path = directory / "capture.csv"
        csv_path.write_text(exported.stdout, encoding="utf-8")
        if exported.returncode:
            raise RuntimeError("NCU report export failed")
        observation = import_csv(csv_path, request, provenance=provenance, available=available)
        observation["artifacts"].extend([artifact(report), artifact(directory / "metric-query.csv")])
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        observation["diagnostics"].append({"code": "collection_failed", "message": str(error)})
    write_json(directory / "observation.json", observation)
    return observation
