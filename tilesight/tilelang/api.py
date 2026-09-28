"""Profile one TileLang kernel with TileSight.

    from tilesight.tilelang import profile_source_file
    result = profile_source_file("kernels/gemm.py", "build_kernel", factory_args={...})
    print(result.status, result.payload()["estimate"])

A refused analysis is a result, not an exception: ``status == "refused"`` and
``refusal`` names the missing fact.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from ..modeling import program as sight
from . import backend
from .capture import Capture, capture
from .diagnose import POLICY, diagnose, loop_service
from .emit import build
from .entry import KernelRequest, resolve
from .errors import Refusal
from .residency import Residency, solve as solve_residency
from .translate import LoopN, Op, Program, translate

SCHEMA = "tilesight-tilelang-profile/v1"


@dataclass
class Profile:
    status: str
    request: Dict[str, Any]
    refusal: Optional[Dict[str, str]] = None
    capture: Optional[Capture] = None
    program: Optional[Program] = None
    residency: Optional[Residency] = None
    estimate: Optional[backend.Estimate] = None
    arch_name: str = ""
    arch_profile: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    observation: Optional[Dict[str, Any]] = None   # joint analysis: measured facts, adopted inputs, steps
    baseline_s: Optional[float] = None               # estimate from compile facts alone (joint analysis)
    notes: List[str] = field(default_factory=list)

    @property
    def result(self) -> Any:
        """TileSight's analysis of one CTA per SM (per-op costs, loops, cache, traffic)."""

        return None if self.estimate is None else self.estimate.single

    def payload(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"schema": SCHEMA, "status": self.status, "request": self.request,
                               "refusal": self.refusal, "notes": list(self.notes)}
        if self.capture is not None:
            device = self.capture.device
            out["identity"] = {
                "planned_ir_sha256": self.capture.planned_sha256,
                "kernel_source_sha256": hashlib.sha256(self.capture.kernel_source.encode()).hexdigest(),
                "compiler": dict(self.capture.compiler),
                "pass_configs": {k: v for k, v in self.capture.pass_configs},
                "target_arch": self.capture.target_arch,
                "tilesight_version": _tilesight_version(),
                "arch": self.arch_name, "arch_profile": self.arch_profile, "context": self.context,
            }
            out["launch"] = {"symbol": device.symbol, "threads": device.thread_count,
                             "dynamic_shared_bytes": device.dynamic_shared_bytes,
                             "mechanisms": list(device.mechanisms)}
        if self.program is not None:
            out["launch"].update({"grid": dict(self.program.grid), "warp_specialized": self.program.warp_specialized,
                                  "roles": list(self.program.actors), "bindings": dict(self.program.bindings)})
            out["program"] = _program_payload(self.program)
        if self.residency is not None:
            r = self.residency
            out["residency"] = {"ctas_per_sm": r.resident_ctas_per_sm, "source": "TileSight declared storage",
                                "max_blocks_per_sm": r.max_blocks_per_sm, "binding": list(r.binding),
                                "storage_per_cta_bytes": dict(r.storage_per_cta),
                                "capacity_per_sm_bytes": dict(r.capacity_per_sm),
                                "unchecked_register_buffers": list(r.unchecked_registers)}
        if self.estimate is not None:
            out.update(_analysis_payload(self))
        if self.observation is not None:
            out["observation"] = dict(self.observation)
            if self.estimate is not None:
                out["observation"]["comparison"] = _comparison(self, out)
        return out


def _comparison(profile: "Profile", payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    facts = profile.observation["measured"]
    traffic = payload["traffic"]
    declared = dict(profile.residency.storage_per_cta) if profile.residency else {}
    threads = profile.capture.device.thread_count
    utilization = next((d["values"].get("kernel_utilization", {}) for d in payload["diagnoses"]
                        if d["kind"] == "pipeline_bottleneck"), {})
    rows = [
        ("kernel time (s)", profile.estimate.kernel_body_s, facts["kernel_time_s"]),
        ("DRAM read (bytes)", traffic["ddr_read_bytes"], facts["dram_read_bytes"]),
        ("DRAM write (bytes)", traffic["ddr_write_bytes"], facts["dram_write_bytes"]),
        ("L2 read (bytes)", traffic["l2_read_request_bytes"], facts["l2_read_bytes"]),
        ("L2 write (bytes)", traffic["l2_write_request_bytes"], facts["l2_write_bytes"]),
        ("resident CTAs per SM", profile.residency.resident_ctas_per_sm if profile.residency else profile.estimate.residency,
         facts["resident_ctas_per_sm"]),
        ("register file per CTA (bytes)", declared.get("register") or None,
         None if facts["registers_per_thread"] is None else facts["registers_per_thread"] * threads * 4),
        ("shared memory per CTA (bytes)", declared.get("smem") or None, facts["smem_per_cta_bytes"]),
        ("tensor pipe busy (%)", 100 * utilization["tensor"] if "tensor" in utilization else None,
         facts["tensor_utilization_pct"]),
    ]
    return [{"quantity": q, "model": m, "measured": v,
             "ratio": None if m is None or v is None or not v else m / v} for q, m, v in rows]


def _program_payload(program: Program) -> Dict[str, Any]:
    def item(x: Any) -> Dict[str, Any]:
        if isinstance(x, LoopN):
            return {"loop": x.name, "trips": x.trips if isinstance(x.trips, int) else str(x.trips), "stages": x.stages,
                    "prologue": [item(y) for y in x.prologue], "body": [item(y) for y in x.body],
                    "epilogue": [item(y) for y in x.epilogue],
                    "issue_order": {k: list(v) for k, v in x.issue_order.items()},
                    "barriers": [list(h) for h in x.hints]}
        op: Op = x
        def side(value: Any) -> Any:
            return value if isinstance(value, (str, type(None))) else {"tensor": value.tensor, "items": [list(i) for i in value.items]}
        return {"op": op.name, "kind": op.kind, "role": op.actor, "window": op.window, "src": side(op.src),
                "dst": side(op.dst), "engine": op.engine, "inputs": list(op.inputs), "output": op.output,
                "work_per_element": [list(w) for w in op.ops]}
    return {
        "tensors": {n: {"shape": list(t.shape), "dtype": t.dtype, "source": t.source,
                        "view": None if t.view is None else list(t.view)} for n, t in program.tensors.items()},
        "buffers": {n: {"storage": b.storage, "shape": list(b.shape), "dtype": b.dtype, "slots": b.slots,
                        "carried": b.carried} for n, b in program.buffers.items()},
        "items": [item(x) for x in program.items],
    }


def _traffic(t: Any) -> Dict[str, float]:
    return {k: float(getattr(t, k)) for k in ("payload_read_bytes", "payload_write_bytes", "ddr_read_bytes",
                                                "ddr_write_bytes", "l2_read_request_bytes", "l2_write_request_bytes",
                                                "l1_5_read_request_bytes", "l1_5_write_request_bytes")}


def _tilesight_version() -> Optional[str]:
    try:
        return importlib.metadata.version("tilesight")
    except importlib.metadata.PackageNotFoundError:
        return None


def _analysis_payload(profile: Profile) -> Dict[str, Any]:
    est = profile.estimate
    single = est.single
    launch = single.launches["main"]
    loops = []
    for path in single.regions:
        region = single.regions[path]
        if region.kind != "loop":
            continue
        ii = region.ii
        loops.append({
            "path": path, "trips": region.trip_count, "total_s": region.total_s, "steady_s": region.steady_s,
            "ii": None if ii is None else {k: getattr(ii, k) for k in ("resource_ii_s", "recurrence_ii_s", "credit_ii_s",
                                                                        "selected_ii_s", "scope", "search_complete")},
            "service_per_iteration_s": loop_service(single, path) if ii is not None else {},
        })
    ops = []
    for path in single.ops:
        op = single.ops[path]
        group = op.cost_groups[0] if op.cost_groups else None
        ops.append({"path": path, "kind": op.kind, "count": op.count, "useful_flops": op.useful_flops,
                    "executed_bytes": op.executed_bytes,
                    "latency_s": None if group is None else group.completion_latency_s,
                    "service_s": {} if group is None else {r: s for r, s in group.service_s},
                    "traffic": _traffic(op.traffic_total)})
    payload = {
        "estimate": {"kernel_body_s": est.kernel_body_s, "upper_s": est.upper_s, "composition": est.composition,
                     "compile_facts_only_s": profile.baseline_s,
                     "resident_ctas_per_sm": est.residency, "effective_resident_ctas_per_sm": est.effective_residency,
                     "single_cta_kernel_body_s": launch.kernel_body_s,
                     "waves_single_cta": launch.waves, "work_units": launch.work_units, "sm_count": launch.sm_count},
        "loops": loops, "ops": ops, "traffic": _traffic(launch.traffic_total),
        "tilesight_notes": list(single.diagnostics.notes) + list(getattr(single.diagnostics, "approximations", ())),
    }
    payload["diagnoses"] = diagnose(profile, payload, profile.observation)
    payload["diagnosis_policy"] = dict(POLICY)
    return payload


def builder(profile: Profile) -> Any:
    """``residency -> tilesight Program`` for this profile's translated kernel."""

    program, mechanisms = profile.program, profile.capture.device.mechanisms
    return lambda residency: build(program, sight, residency=residency, mechanisms=mechanisms)


def _request(kernel: KernelRequest) -> Dict[str, Any]:
    return {"file": kernel.file, "function": kernel.function, "factory_args": dict(kernel.factory_args),
            "specialization": dict(kernel.specialization), "target_arch": kernel.target_arch,
            "pass_configs": None if kernel.pass_configs is None else dict(kernel.pass_configs),
            "bindings": dict(kernel.bindings)}


def profile_request(kernel: KernelRequest, *, arch: str = "H200_SXM", arch_profile: str = "set_to_microbench",
                    clock_overrides: Optional[Mapping[str, float]] = None,
                    residency_override: Optional[int] = None,
                    prim_func: Any = None) -> Profile:
    profile = Profile("refused", _request(kernel), arch_name=arch, arch_profile=arch_profile)
    try:
        if prim_func is None:
            resolved = resolve(kernel)
            prim_func, configs = resolved.prim_func, dict(resolved.pass_configs)
            profile.notes.append("pass configs from %s" % resolved.pass_configs_source)
        else:
            configs = dict(kernel.pass_configs or {})
        profile.capture = capture(prim_func, target_arch=kernel.target_arch, pass_configs=configs)
        bindings = dict(kernel.bindings)
        profile.program = translate(profile.capture, bindings)
        profile.notes.extend(profile.program.notes)
        arch_obj = backend.arch(arch, arch_profile)
        profile.residency = solve_residency(builder(profile)(1), arch_obj, backend.options())
        profile.notes.extend(profile.residency.notes)
        residency = residency_override or profile.residency.resident_ctas_per_sm
        context = None
        if clock_overrides:
            context = sight.Context(clock_overrides=tuple(sorted(clock_overrides.items())), label="calibrated")
            profile.context = {"clock_overrides": dict(clock_overrides)}
        profile.estimate = backend.estimate(builder(profile), arch_obj, residency, context=context)
        profile.status = "complete"
    except Refusal as refusal:
        profile.refusal = refusal.payload()
    except Exception as error:  # a TileSight contract violation is a refusal of this program, with the reason
        name = type(error).__name__
        if name in ("ContractError", "UnsupportedError", "ModelingValidationError", "LoweringError"):
            profile.refusal = {"code": "TILESIGHT_" + name.upper(), "message": str(error)}
        else:
            raise
    return profile


def profile_source_file(file: str, function: str = "build_kernel", *, factory_args: Optional[Mapping[str, Any]] = None,
                        specialization: Optional[Mapping[str, Any]] = None, target_arch: str = "sm_90a",
                        pass_configs: Optional[Mapping[str, Any]] = None, bindings: Optional[Mapping[str, int]] = None,
                        prim_func_name: Optional[str] = None, **options: Any) -> Profile:
    kernel = KernelRequest(str(file), function, dict(factory_args or {}), dict(specialization or {}), target_arch,
                           None if pass_configs is None else dict(pass_configs), dict(bindings or {}), prim_func_name)
    return profile_request(kernel, **options)


def profile_primfunc(prim_func: Any, *, target_arch: str = "sm_90a", pass_configs: Optional[Mapping[str, Any]] = None,
                     bindings: Optional[Mapping[str, int]] = None, name: str = "primfunc", **options: Any) -> Profile:
    kernel = KernelRequest(name, "<primfunc>", {}, {}, target_arch, dict(pass_configs or {}), dict(bindings or {}))
    return profile_request(kernel, prim_func=prim_func, **options)


def dumps(profile: Profile) -> str:
    return json.dumps(profile.payload(), indent=1, sort_keys=True, default=str)
