"""Performance diagnoses read off one TileSight analysis.

Every value comes from the analysis (``AnalysisResult``) or from compile facts;
the thresholds are policy, listed in ``POLICY`` and echoed in each payload.
Loop-level diagnoses use the one-CTA-per-SM analysis, i.e. per-CTA service at
full SM rates. A diagnosis that lacks its inputs says ``inconclusive`` or
``not_evaluated``; it never guesses.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

POLICY = {
    "bound_tie_relative_tolerance": 0.05,
    "l2_request_amplification_ratio_threshold": 1.25,
    "ddr_traffic_amplification_ratio_threshold": 1.0,
    "transaction_amplification_bytes_threshold": 0.0,
    "dram_refetch_ratio_threshold": 2.0,
    "overlap_ratio_threshold": 0.10,
    "idle_slot_fraction_threshold": 0.10,
    "work_group_spread_threshold": 1.25,
    "latency_exposed_resident_warps": 16,
    "model_deviation_threshold": 0.15,
}
COMPUTE = frozenset({"tensor", "cuda", "sfu", "issue"})
_BYTES = {"fp64": 8, "int64": 8, "fp32": 4, "int32": 4, "fp16": 2, "bf16": 2, "int8": 1, "uint8": 1,
          "fp8_e4m3": 1, "fp8_e5m2": 1}


def _diagnosis(kind: str, status: str, scope: str, summary: str, values: Optional[Dict[str, Any]] = None,
               assumptions: Iterable[str] = ()) -> Dict[str, Any]:
    severity = "warning" if status == "detected" and kind not in ("pipeline_bottleneck",) else "info"
    return {"kind": kind, "status": status, "severity": severity, "scope": scope, "summary": summary,
            "values": dict(values or {}), "assumptions": list(assumptions)}


def _ns(seconds: Optional[float]) -> str:
    return "n/a" if seconds is None else "%.1f ns" % (seconds * 1e9)


# --------------------------------------------------------------------------- loops

def main_loop(single: Any) -> Optional[str]:
    """The loop region with the largest total time that has a periodic II."""

    loops = [p for p in single.regions if single.regions[p].kind == "loop" and single.regions[p].ii is not None
             and single.regions[p].ii.selected_ii_s is not None]
    return max(loops, key=lambda p: single.regions[p].total_s) if loops else None


def _loop_label(path: str) -> str:
    return path.rsplit("/", 1)[-1]


def loop_intervals(single: Any, path: str) -> List[Tuple[str, str, float, float]]:
    """(op, resource, offset, per-iteration service) of every resource use inside one loop body."""

    region = single.regions[path]
    label = _loop_label(path)
    out = []
    for interval in region.groups[0].service_trace:
        labels = [r[0] for r in interval.repeats]
        if label not in labels:
            continue
        inner = 1
        for _name, count, _stride in interval.repeats[labels.index(label) + 1:]:
            inner *= count
        out.append((interval.op_id, interval.resource, interval.offset_s, interval.service_time_s * inner))
    return out


def loop_service(single: Any, path: str) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    for _op, resource, _offset, seconds in loop_intervals(single, path):
        totals[resource] = totals.get(resource, 0.0) + seconds
    return totals


def kernel_utilization(single: Any, kernel_s: float) -> Dict[str, float]:
    """Fraction of the kernel time each resource is busy, averaged over SMs (speed-of-light style)."""

    launch = single.launches["main"]
    busy: Dict[str, float] = {}
    for path in single.ops:
        for group in single.ops[path].cost_groups:
            for resource, seconds in group.service_s:
                busy[resource] = busy.get(resource, 0.0) + seconds * group.count
    scale = launch.sm_count * kernel_s
    return {r: b / scale for r, b in sorted(busy.items(), key=lambda x: -x[1])} if scale else {}


def _top(utilization: Dict[str, float], n: int = 3) -> str:
    return ", ".join("%s %.0f%%" % (r, 100 * u) for r, u in list(utilization.items())[:n]) or "none"


def _pipeline(single: Any, path: Optional[str], kernel_s: float) -> Dict[str, Any]:
    utilization = kernel_utilization(single, kernel_s)
    if path is None:
        loops = [p for p in single.regions if single.regions[p].kind == "loop"]
        shape = ("loop %s runs serially (no periodic schedule)" % loops[0] if loops else
                 "the kernel has no loop; each CTA runs its operations once")
        return _diagnosis("pipeline_bottleneck", "detected", "kernel",
                          "%s%s. Busiest resources over the kernel: %s." % (shape[0].upper(), shape[1:], _top(utilization)),
                          {"kernel_utilization": utilization})
    ii = single.regions[path].ii
    bounds = {"resource": ii.resource_ii_s, "recurrence": ii.recurrence_ii_s, "credit": ii.credit_ii_s}
    known = {k: v for k, v in bounds.items() if v is not None}
    top = max(known.values()) if known else 0.0
    tolerance = POLICY["bound_tie_relative_tolerance"]
    limiting = [k for k, v in known.items() if top - v <= tolerance * max(top, 1e-30)]
    selected = ii.selected_ii_s
    if selected > top * (1.0 + tolerance):
        limiting = ["packing"]
    service = loop_service(single, path)
    per_iteration = {r: s / selected for r, s in sorted(service.items(), key=lambda x: -x[1])}
    reasons = {
        "resource": "the busiest resource",
        "recurrence": "a loop-carried dependency chain",
        "credit": "buffer slots (pipeline depth)",
        "packing": "resource reservations that do not pack at the lower bounds",
    }
    summary = ("Loop %s: II %s, set by %s (resource bound %s, recurrence bound %s, credit bound %s). "
               "Busiest per iteration: %s. Over the kernel: %s." % (
                   path, _ns(selected), " and ".join(reasons[k] for k in limiting), _ns(ii.resource_ii_s),
                   _ns(ii.recurrence_ii_s), _ns(ii.credit_ii_s), _top(per_iteration), _top(utilization)))
    values = {"selected_ii_s": selected, "resource_ii_s": ii.resource_ii_s, "recurrence_ii_s": ii.recurrence_ii_s,
              "credit_ii_s": ii.credit_ii_s, "limiting": limiting, "search_complete": ii.search_complete,
              "loop_utilization": per_iteration, "kernel_utilization": utilization}
    return _diagnosis("pipeline_bottleneck", "detected", "loop " + path, summary, values,
                      ("Lower bounds and the selected II are TileSight's periodic schedule of one CTA per SM, "
                       "not a measured pipeline.",))


def _fold(start: float, duration: float, period: float) -> List[Tuple[float, float]]:
    if duration <= 0.0:
        return []
    if duration >= period:
        return [(0.0, period)]
    start %= period
    end = start + duration
    return [(start, end)] if end <= period else [(start, period), (0.0, end - period)]


def _union(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    merged: List[List[float]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


def _overlap(single: Any, path: Optional[str]) -> Dict[str, Any]:
    if path is None:
        return _diagnosis("compute_io_overlap", "not_evaluated", "kernel",
                          "No loop has a periodic schedule; operations of one CTA run in sequence and overlap only "
                          "with other resident CTAs.")
    ii = single.regions[path].ii.selected_ii_s
    compute: List[Tuple[float, float]] = []
    memory: List[Tuple[float, float]] = []
    for _op, resource, offset, seconds in loop_intervals(single, path):
        if offset is None:
            return _diagnosis("compute_io_overlap", "inconclusive", "loop " + path,
                              "The schedule has no start time for some operations, so overlap is unknown.")
        (compute if resource in COMPUTE else memory).extend(_fold(offset, seconds, ii))
    compute, memory = _union(compute), _union(memory)
    compute_s = sum(b - a for a, b in compute)
    memory_s = sum(b - a for a, b in memory)
    both = sum(max(0.0, min(b, d) - max(a, c)) for a, b in compute for c, d in memory)
    compute_ratio = both / compute_s if compute_s else 0.0
    memory_ratio = both / memory_s if memory_s else 0.0
    if not compute_s or not memory_s:
        return _diagnosis("compute_io_overlap", "not_detected", "loop " + path,
                          "Loop %s uses only %s resources; there is nothing to overlap."
                          % (path, "memory" if memory_s else "compute"),
                          {"compute_busy_s": compute_s, "memory_busy_s": memory_s})
    poor = min(compute_ratio, memory_ratio) < POLICY["overlap_ratio_threshold"]
    summary = ("Loop %s: compute and memory work barely overlap in the schedule (%.0f%% of compute time "
               "overlaps memory); more pipeline stages or asynchronous copies would help." % (path, 100 * compute_ratio)
               if poor else
               "Loop %s: %.0f%% of compute time and %.0f%% of memory time overlap in one period of %s."
               % (path, 100 * compute_ratio, 100 * memory_ratio, _ns(ii)))
    return _diagnosis("compute_io_overlap", "detected" if poor else "not_detected", "loop " + path, summary,
                      {"period_s": ii, "compute_busy_s": compute_s, "memory_busy_s": memory_s, "overlap_s": both,
                       "compute_overlap_ratio": compute_ratio, "memory_overlap_ratio": memory_ratio},
                      ("Overlap is read from TileSight's periodic schedule, not from a measured timeline.",))


# --------------------------------------------------------------------------- memory

def _transfer(launch: Any) -> Dict[str, Any]:
    t = launch.traffic_total
    payload = t.payload_read_bytes + t.payload_write_bytes
    if not payload:
        return _diagnosis("transfer_anomaly", "inconclusive", "kernel", "The kernel moves no global payload.")
    l2 = t.l2_read_request_bytes + t.l2_write_request_bytes
    ddr = t.ddr_read_bytes + t.ddr_write_bytes
    values = {"payload_bytes": payload, "l2_request_bytes": l2, "ddr_bytes": ddr,
              "transaction_amplification_bytes": t.transaction_amplification_bytes,
              "l2_request_amplification": l2 / payload, "ddr_traffic_amplification": ddr / payload}
    findings = []
    if t.transaction_amplification_bytes > POLICY["transaction_amplification_bytes_threshold"]:
        findings.append("%.3g bytes of partial-sector transactions (uncoalesced or misaligned access)"
                        % t.transaction_amplification_bytes)
    if l2 / payload > POLICY["l2_request_amplification_ratio_threshold"]:
        findings.append("L2 requests are %.2fx the payload" % (l2 / payload))
    if ddr / payload > POLICY["ddr_traffic_amplification_ratio_threshold"]:
        findings.append("DRAM traffic is %.2fx the payload" % (ddr / payload))
    summary = ("Global traffic exceeds the payload: " + "; ".join(findings) + "." if findings else
               "Global traffic does not exceed the payload (L2 requests %.2fx, DRAM %.2fx of %.3g payload bytes)."
               % (l2 / payload, ddr / payload, payload))
    return _diagnosis("transfer_anomaly", "detected" if findings else "not_detected", "kernel", summary, values)


def _footprints(program: Any) -> Dict[str, float]:
    out = {}
    for name, tensor in program.tensors.items():
        size = _BYTES.get(tensor.dtype)
        if size is not None:
            out[name] = float(math.prod(tensor.shape)) * size
    return out


def _cache(single: Any, program: Any) -> Dict[str, Any]:
    if not single.cache:
        return _diagnosis("cache_utilization", "not_evaluated", "kernel", "The analysis ran without the cache model.")
    report = single.cache[0]
    tensor_of = {row.op_path: row.tensor for row in report.per_access}
    dram: Dict[str, float] = {}
    for path in single.ops:
        tensor = tensor_of.get(path)
        if tensor is not None:
            dram[tensor] = dram.get(tensor, 0.0) + single.ops[path].traffic_total.ddr_read_bytes
    footprint = _footprints(program)
    hit = {}
    for tensor, bucket in report.per_tensor:
        reaching = bucket["l2_served"] + bucket["ddr_misses"]
        if bucket["read_requests"] and reaching:
            hit[tensor] = bucket["l2_served"] / reaching
    refetch = {t: dram[t] / footprint[t] for t in dram if footprint.get(t) and dram[t] > 0}
    worst = sorted(((r, t) for t, r in refetch.items() if r > POLICY["dram_refetch_ratio_threshold"]), reverse=True)
    values = {"dram_read_bytes": dram, "footprint_bytes": {t: footprint[t] for t in dram if t in footprint},
              "dram_refetch_ratio": refetch, "l2_hit_rate": hit, "cache_scenario": report.scenario,
              "cache_mode": report.mode}
    if worst:
        summary = "DRAM re-reads: " + ", ".join("%s is read %.1fx its size (L2 hit rate %s)"
                                                % (t, r, "%.0f%%" % (100 * hit[t]) if t in hit else "n/a")
                                                for r, t in worst) + \
            ". The L2 does not hold the reuse across concurrent CTAs; a block swizzle or a different launch order may help."
    else:
        summary = "Every global tensor is read from DRAM at most %.1fx its size." % POLICY["dram_refetch_ratio_threshold"]
    return _diagnosis("cache_utilization", "detected" if worst else "not_detected", "kernel", summary, values,
                      ("TileSight cache model, %s scenario, %s mode; launch order is the CTA linear order."
                       % (report.scenario, report.mode),))


# --------------------------------------------------------------------------- balance and occupancy

def _cta_balance(launch: Any) -> Dict[str, Any]:
    groups = [g for g in launch.groups if g.count]
    if len(groups) < 2:
        return _diagnosis("load_imbalance", "not_detected", "cta",
                          "Every CTA runs the same work (one work group).", {"work_groups": len(groups)})
    total = sum(g.count for g in groups)
    mean = sum(g.per_unit_s * g.count for g in groups) / total
    spread = max(g.per_unit_s for g in groups) / mean
    detected = spread > POLICY["work_group_spread_threshold"]
    summary = "CTAs fall into %d work groups; the slowest takes %.2fx the mean CTA time." % (len(groups), spread)
    return _diagnosis("load_imbalance", "detected" if detected else "not_detected", "cta", summary,
                      {"work_groups": {g.group_id: {"count": g.count, "per_unit_s": g.per_unit_s} for g in groups},
                       "max_over_mean": spread})


def _sm_balance(launch: Any, residency: int) -> Dict[str, Any]:
    slots = launch.sm_count * residency
    waves = math.ceil(launch.work_units / slots)
    tail = launch.work_units - (waves - 1) * slots
    idle = 1.0 - launch.work_units / float(waves * slots)
    detected = idle > POLICY["idle_slot_fraction_threshold"]
    summary = ("%d CTAs on %d slots (%d SMs x %d resident): %d waves, the last one %.0f%% full; "
               "%.0f%% of slot time is idle." % (launch.work_units, slots, launch.sm_count, residency, waves,
                                                100.0 * tail / slots, 100 * idle))
    return _diagnosis("load_imbalance", "detected" if detected else "not_detected", "sm_wave", summary,
                      {"work_units": launch.work_units, "slots": slots, "waves": waves, "tail_work_units": tail,
                       "idle_slot_fraction": idle},
                      ("Equal CTA durations; slot time is waves x slots.",))


def _latency(profile: Any, single: Any, path: Optional[str]) -> Dict[str, Any]:
    residency = profile.estimate.effective_residency
    warps_per_cta = math.ceil(profile.capture.device.thread_count / 32)
    warps = residency * warps_per_cta
    launch = single.launches["main"]
    loop_share = single.regions[path].total_s / launch.groups[0].per_unit_s if path and launch.groups else 0.0
    detected = warps <= POLICY["latency_exposed_resident_warps"] and loop_share >= 0.5
    values = {"resident_warps_per_sm": warps, "resident_ctas_per_sm": residency, "warps_per_cta": warps_per_cta,
              "main_loop_share": loop_share}
    if detected:
        summary = ("%d warps per SM are resident (%d CTAs of %d warps). Latency that TileSight does not price "
                   "(instruction issue, barrier waits) is hidden only by the kernel's own pipelining, so the "
                   "estimate may be optimistic." % (warps, residency, warps_per_cta))
    else:
        summary = "%d warps per SM are resident (%d CTAs of %d warps)." % (warps, residency, warps_per_cta)
    return _diagnosis("latency_exposure", "detected" if detected else "not_detected", "sm", summary, values,
                      ("Screening rule, not a correction: the estimate is not adjusted.",
                       "Resident CTAs are TileSight's; it does not count compiler-allocated registers or thread limits."))


def _deviation(observation: Optional[Dict[str, Any]], estimate_s: float) -> Dict[str, Any]:
    if observation is None:
        return _diagnosis("model_deviation", "not_evaluated", "kernel",
                          "No runtime measurement was supplied, so the estimate is not compared with one.")
    measured = observation.get("kernel_time_s")
    if observation["association"]["status"] != "matched" or not measured:
        return _diagnosis("model_deviation", "inconclusive", "kernel",
                          "The measurement cannot be compared: %s." % "; ".join(
                              observation["association"]["problems"] or ["the kernel time is missing"]))
    error = estimate_s / measured - 1.0
    steps = " -> ".join("%s %.2f us" % (s["step"], s["kernel_body_s"] * 1e6) for s in observation["steps"])
    detected = abs(error) > POLICY["model_deviation_threshold"]
    summary = "Measured %.2f us; estimate %.2f us (%+.1f%%). Steps: %s." % (measured * 1e6, estimate_s * 1e6,
                                                                          100 * error, steps)
    return _diagnosis("model_deviation", "detected" if detected else "not_detected", "kernel", summary,
                      {"estimate_s": estimate_s, "measured_s": measured, "relative_error": error,
                       "steps": observation["steps"]},
                      ("Measured facts adopted: %s." % (", ".join(sorted(observation["adopted"])) or "none"),))


def diagnose(profile: Any, payload: Dict[str, Any], observation: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    single = profile.estimate.single
    launch = single.launches["main"]
    path = main_loop(single)
    return [
        _pipeline(single, path, profile.estimate.kernel_body_s),
        _overlap(single, path),
        _transfer(launch),
        _cache(single, profile.program),
        _cta_balance(launch),
        _sm_balance(launch, profile.estimate.effective_residency),
        _latency(profile, single, path),
        _deviation(observation, profile.estimate.kernel_body_s),
    ]
