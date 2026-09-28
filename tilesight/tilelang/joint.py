"""Joint analysis: the same TileSight model, re-run with facts that NCU measured.

Nothing is adopted unless the observation is proven to be this kernel (symbol,
grid, block, dynamic shared memory, SM count and, when the capture driver
recorded it, the hash of the compiled CUDA source). Adopted facts, one step at
a time so each step's effect is visible:

1. the SM clock (``sm__cycles_elapsed.avg.per_second``) as TileSight's ``sm``
   clock override, which scales SM-clocked rates;
2. L2 and DRAM traffic: every global access keeps its share of the model's
   traffic, scaled so that the kernel totals equal the measured totals;
3. resident CTAs per SM from NCU's occupancy limits, when they differ from
   TileSight's and TileSight's declared storage fits them.

The measured kernel time is compared with the result, never fitted.
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Any, Dict, List, Optional, Tuple

from ..modeling import program as sight
from . import backend
from .api import Profile, builder
from .ncu import metric, read_json, validate_observation


def load_observation(path: Any) -> Dict[str, Any]:
    observation = validate_observation(read_json(path))
    if observation["status"] == "failed":
        reasons = "; ".join(d.get("message", "") for d in observation.get("diagnostics", ()))
        raise ValueError("the observation records a failed capture: %s" % (reasons or "no reason recorded"))
    return observation


def measured(observation: Dict[str, Any]) -> Dict[str, Optional[float]]:
    def m(key: str) -> Optional[float]:
        return metric(observation, key)

    def sectors(key: str) -> Optional[float]:
        value = m(key)
        return None if value is None else 32.0 * value

    limits = [m("cta_limit_" + name) for name in ("blocks", "registers", "shared_mem", "warps")]
    if m("cta_limit_barriers"):
        limits.append(m("cta_limit_barriers"))
    return {
        "kernel_time_s": m("duration_s"), "sm_clock_hz": m("sm_frequency_hz"), "l2_clock_hz": m("l2_frequency_hz"),
        "dram_clock_hz": m("dram_frequency_hz"),
        "dram_read_bytes": m("dram_read_bytes"), "dram_write_bytes": m("dram_write_bytes"),
        "l2_read_bytes": sectors("l2_read_sectors"), "l2_write_bytes": sectors("l2_write_sectors"),
        "resident_ctas_per_sm": None if any(v is None for v in limits) else int(min(limits)),
        "registers_per_thread": m("registers_per_thread"), "achieved_occupancy_pct": m("achieved_occupancy_pct"),
        "smem_per_cta_bytes": None if m("smem_static") is None or m("smem_dynamic") is None
        else m("smem_static") + m("smem_dynamic"),
        "tensor_utilization_pct": m("tensor_utilization_pct"), "dram_bandwidth_pct": m("dram_bandwidth_pct"),
        "l2_bandwidth_pct": m("lts_bandwidth_pct"), "local_sectors": (
            None if m("local_read_sectors") is None or m("local_write_sectors") is None
            else m("local_read_sectors") + m("local_write_sectors")),
    }


def associate(profile: Profile, observation: Dict[str, Any]) -> List[str]:
    """Reasons the observation is not this kernel; empty when every recorded fact matches."""

    device = profile.capture.device
    launch = profile.result.launches["main"]
    problems = []
    symbol = observation.get("launch", {}).get("Kernel Name")
    if symbol != device.symbol:
        problems.append("kernel %r, model %r" % (symbol, device.symbol))
    dims = {kind: [metric(observation, "%s_%s" % (kind, a)) for a in "xyz"] for kind in ("grid", "block")}
    if any(v is None for v in dims["grid"] + dims["block"]):
        problems.append("launch dimensions are missing from the observation")
    else:
        ctas = int(dims["grid"][0] * dims["grid"][1] * dims["grid"][2])
        threads = int(dims["block"][0] * dims["block"][1] * dims["block"][2])
        if ctas != launch.physical_ctas:
            problems.append("grid of %d CTAs, model %d" % (ctas, launch.physical_ctas))
        if threads != device.thread_count:
            problems.append("%d threads per CTA, model %d" % (threads, device.thread_count))
    dynamic = metric(observation, "smem_dynamic")
    if dynamic is not None and device.dynamic_shared_bytes is not None and int(dynamic) != device.dynamic_shared_bytes:
        problems.append("%d bytes of dynamic shared memory, model %d" % (dynamic, device.dynamic_shared_bytes))
    cluster = metric(observation, "cluster_size")
    if cluster not in (None, 0, 1):
        problems.append("cluster launch of size %d is not modeled" % cluster)
    sms = metric(observation, "sm_count")
    if sms is not None and int(sms) != launch.sm_count:
        problems.append("%d SMs, model architecture %d" % (sms, launch.sm_count))
    receipt = observation.get("provenance", {}).get("driver_receipt", {})
    source = receipt.get("kernel_source_sha256")
    if source is not None and source != hashlib.sha256(profile.capture.kernel_source.encode()).hexdigest():
        problems.append("the captured kernel was compiled from different CUDA source")
    return problems


def _scaled_traffic(profile: Profile, facts: Dict[str, Optional[float]]) -> Tuple[Tuple[Any, ...], Dict[str, float]]:
    """Per-execution traffic of every global access, scaled to the measured kernel totals."""

    single = profile.result
    total = single.launches["main"].traffic_total
    pairs = (("l2_read_request_bytes", "l2_read_bytes"), ("l2_write_request_bytes", "l2_write_bytes"),
             ("ddr_read_bytes", "dram_read_bytes"), ("ddr_write_bytes", "dram_write_bytes"))
    scales = {}
    for field, key in pairs:
        model, value = getattr(total, field), facts.get(key)
        if value is not None and model > 0:
            scales[field] = value / model
    if not scales:
        return (), {}
    assumptions = []
    for path in single.ops:
        op = single.ops[path]
        traffic = op.traffic_total
        if not op.count or not any(getattr(traffic, field) for field, _ in pairs):
            continue
        per_execution = traffic.scale(1.0 / op.count)
        per_execution = dataclasses.replace(per_execution, **{
            field: getattr(per_execution, field) * scale for field, scale in scales.items()})
        assumptions.append((path, sight.TrafficAssumption(per_execution, "ncu kernel totals, model shares")))
    return tuple(assumptions), scales


def joint(profile: Profile, observation: Dict[str, Any]) -> Profile:
    facts = measured(observation)
    problems = associate(profile, observation)
    record: Dict[str, Any] = {
        "kernel_time_s": facts["kernel_time_s"], "measured": facts,
        "association": {"status": "mismatched" if problems else "matched", "problems": problems},
        "adopted": {}, "steps": [{"step": "compile facts", "kernel_body_s": profile.estimate.kernel_body_s}],
        "request_digest": observation.get("request_digest"),
        "device": observation.get("provenance", {}).get("driver_receipt", {}).get("device_name"),
    }
    if problems:
        notes = profile.notes + ["the observation does not match this kernel (%s); nothing was adopted" % "; ".join(problems)]
        return dataclasses.replace(profile, observation=record, notes=notes)
    arch_obj = backend.arch(profile.arch_name, profile.arch_profile)
    build = builder(profile)
    clocks: Dict[str, float] = {}
    traffic: Tuple[Any, ...] = ()
    residency = profile.estimate.residency
    estimate = profile.estimate

    def run(label: str) -> None:
        nonlocal estimate
        context = sight.Context(clock_overrides=tuple(sorted(clocks.items())), traffic_assumptions=traffic,
                                label="ncu")
        estimate = backend.estimate(build, arch_obj, residency, context=context)
        record["steps"].append({"step": label, "kernel_body_s": estimate.kernel_body_s})

    if facts["sm_clock_hz"]:
        clocks["sm"] = facts["sm_clock_hz"]
        record["adopted"]["sm_clock_hz"] = facts["sm_clock_hz"]
        run("measured SM clock %.4f GHz" % (facts["sm_clock_hz"] / 1e9))
    if facts["local_sectors"]:
        record["adopted"]["traffic"] = "not adopted: %d local-memory sectors (spills) are outside the model's accesses" % facts["local_sectors"]
    else:
        traffic, scales = _scaled_traffic(profile, facts)
        if traffic:
            record["adopted"]["traffic_scale"] = scales
            run("measured L2/DRAM traffic (" + ", ".join("%s x%.2f" % (k.replace("_request", ""), v)
                                                         for k, v in sorted(scales.items())) + ")")
    observed = facts["resident_ctas_per_sm"]
    if observed is not None and observed != residency:
        modeled, residency = residency, observed
        try:
            run("NCU residency %d CTAs per SM" % observed)
            record["adopted"]["resident_ctas_per_sm"] = observed
        except sight.UnsupportedError as error:  # TileSight's declared storage does not fit the measured residency
            residency = modeled
            record["adopted"]["resident_ctas_per_sm"] = "not adopted: %s" % error
    context = {"clock_overrides": dict(clocks), "traffic_assumptions": len(traffic),
               "residency": residency, "source": "ncu observation %s" % (observation.get("request_digest") or "")[:12]}
    return dataclasses.replace(profile, estimate=estimate, context=context, observation=record,
                               baseline_s=profile.estimate.kernel_body_s)
