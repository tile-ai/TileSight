"""``compare(baseline, calibrated, observations) -> ComparisonResult``.

Aligns theoretical and calibrated predictions with observations by stable
ids (program name, launch name, op path).  Residuals are only computed when
the measurement scope, unit, and denominator are provably comparable;
otherwise the three columns are still shown and the residual is ``None``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..errors import ModelingValidationError
from .results import AnalysisResult

TIME_SCOPES = ("kernel_elapsed", "sum_kernel_elapsed", "device_span", "host_wall")
METRICS = {
    "time_s": "s",
    "ddr_read_bytes": "bytes",
    "ddr_write_bytes": "bytes",
    "l2_read_request_bytes": "bytes",
    "l2_write_request_bytes": "bytes",
    "l2_request_bytes": "bytes",
    "l1_5_read_request_bytes": "bytes",
    "l1_5_write_request_bytes": "bytes",
    "l2_hit_rate": "ratio",
}
TRAFFIC_METRICS = tuple(name for name, unit in METRICS.items() if unit == "bytes")
TRAFFIC_SCOPES = ("cumulative", "kernel")  # cumulative bytes over every execution of the target
L2_HIT_DENOMINATOR = "cache_model_read_requests"


@dataclass(frozen=True)
class Observation:
    """One measured value tied to a model object by stable id.

    ``workload_digest`` optionally pins the observation to the workload
    identity (shapes/dtypes/work structure) reported by ``AnalysisResult``;
    a mismatch makes the row non-comparable.
    """

    target: str
    metric: str
    value: float
    unit: str
    scope: str
    source: str
    denominator: Optional[str] = None
    instance_id: Optional[str] = None
    workload_digest: Optional[str] = None
    target_kind: Optional[str] = None  # "program" | "launch" | "op"; resolves equal names explicitly
    metadata: Tuple[Tuple[str, Any], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.metric not in METRICS:
            raise ModelingValidationError("Observation.metric=%r: must be one of %s" % (self.metric, sorted(METRICS)))
        if self.unit != METRICS[self.metric]:
            raise ModelingValidationError(
                "Observation(%s).unit=%r: metric %s uses %r" % (self.target, self.unit, self.metric, METRICS[self.metric])
            )
        if self.metric == "time_s" and self.scope not in TIME_SCOPES:
            raise ModelingValidationError(
                "Observation(%s).scope=%r: time needs one of %s" % (self.target, self.scope, ", ".join(TIME_SCOPES))
            )
        if not self.source:
            raise ModelingValidationError("Observation.source must name the collector")
        object.__setattr__(self, "metadata", tuple(self.metadata))


@dataclass(frozen=True)
class AlignmentEntry:
    target: str
    metric: str
    kind: str  # program | launch | op | unknown
    status: str  # matched | mismatched | ambiguous | missing_observation
    reason: str
    observation_scope: Optional[str] = None
    model_scope: Optional[str] = None


@dataclass(frozen=True)
class MetricRow:
    target: str
    metric: str
    unit: str
    scope: str
    baseline: Optional[float]
    calibrated: Optional[float]
    observed: Optional[float]
    comparable: bool
    reason: str
    denominator: Optional[str] = None
    observation_source: Optional[str] = None


@dataclass(frozen=True)
class DeltaRow:
    target: str
    metric: str
    scope: str
    model_delta: Optional[float]
    baseline_residual: Optional[float]
    calibrated_residual: Optional[float]
    baseline_abs_error: Optional[float]
    calibrated_abs_error: Optional[float]
    baseline_rel_error: Optional[float]
    calibrated_rel_error: Optional[float]


@dataclass(frozen=True)
class OpDelta:
    path: str
    count_baseline: int
    count_calibrated: int
    completion_baseline_s: Optional[float]
    completion_calibrated_s: Optional[float]
    ddr_bytes_baseline: float
    ddr_bytes_calibrated: float
    observed_residual: Optional[float]
    observed_note: str


@dataclass(frozen=True)
class ComparisonResult:
    alignment: Tuple[AlignmentEntry, ...]
    metrics: Tuple[MetricRow, ...]
    deltas: Tuple[DeltaRow, ...]
    operations: Tuple[OpDelta, ...]
    diagnostics: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alignment": [vars(x) for x in self.alignment],
            "metrics": [vars(x) for x in self.metrics],
            "deltas": [vars(x) for x in self.deltas],
            "operations": [vars(x) for x in self.operations],
            "diagnostics": list(self.diagnostics),
        }


PROGRAM_TIME_SCOPES = ("sum_kernel_elapsed", "device_span", "host_wall")


def _candidates(result: AnalysisResult, target: str) -> Tuple[str, ...]:
    kinds = []
    if target == result.program.name:
        kinds.append("program")
    if target in result.launches:
        kinds.append("launch")
    if target in result.ops:
        kinds.append("op")
    return tuple(kinds)


def _kind(result: AnalysisResult, target: str, hint: Optional[str] = None, metric: str = "", scope: str = "") -> str:
    """Resolve a target name; equal program/launch names are disambiguated, never guessed silently."""

    kinds = _candidates(result, target)
    if not kinds:
        return "unknown"
    if hint is not None:
        return hint if hint in kinds else "unknown"
    if len(kinds) == 1:
        return kinds[0]
    if metric == "time_s":
        if scope == "kernel_elapsed" and "launch" in kinds:
            return "launch"
        if scope in PROGRAM_TIME_SCOPES and "program" in kinds:
            return "program"
    if "launch" in kinds and "program" in kinds and len(result.launches) == 1:
        return "launch"  # a single-launch program: both names denote the same work
    return "ambiguous"


def _model_value(result: AnalysisResult, target: str, metric: str, scope: str,
                 hint: Optional[str] = None) -> Tuple[Optional[float], Optional[str], str]:
    """Return (value, model_scope, reason) for one metric on one target."""

    kind = _kind(result, target, hint, metric, scope)
    if kind == "unknown":
        return None, None, "target %r is not a program, launch, or op path%s" % (
            target, "" if hint is None else " of kind %r" % hint)
    if kind == "ambiguous":
        return None, None, ("target %r names both the program and a launch; set Observation.target_kind" % target)
    if metric == "time_s":
        if kind == "launch":
            launch = result.launch(target)
            if scope == "kernel_elapsed":
                return launch.kernel_body_s, "kernel_body_s", "ok"
            return None, "kernel_body_s", "launch time is kernel_elapsed; observation scope %r differs" % scope
        if kind == "program":
            bodies = sum(item.kernel_body_s for item in result.launches.values())
            dispatch = sum(item.host_dispatch_s for item in result.launches.values())
            if scope == "sum_kernel_elapsed":
                return bodies, "sum(kernel_body_s)", "ok"
            if scope == "device_span":
                return result.program.total_s - dispatch, "program.total_s - host_dispatch", "ok"
            if scope == "host_wall":
                return result.program.total_s, "program.total_s", "ok"
            return None, "program.total_s", "program time needs sum_kernel_elapsed/device_span/host_wall"
        return None, None, "op-level time has no kernel-internal observation range"
    if metric in TRAFFIC_METRICS:
        if scope not in TRAFFIC_SCOPES:
            return None, "cumulative_bytes", (
                "model traffic is cumulative over all executions; observation scope %r is not one of %s"
                % (scope, ", ".join(TRAFFIC_SCOPES)))
        if kind == "op":
            value = result.op(target).traffic_total.value(metric)
            if value is None:
                return None, "op.traffic_total", "model value unknown: %s" % result.op(target).traffic_total.unknown_reason(metric)
            return value, "op.traffic_total", "ok"
        if kind == "launch":
            total = result.launch(target).traffic_total.value(metric)
            if total is None:
                return None, "launch.traffic_total", "model value unknown"
            return total, "launch.traffic_total", "ok"
        total = result.program.traffic_total.value(metric)
        if total is None:
            return None, "program.traffic_total", "model value unknown"
        return total, "program.traffic_total", "ok"
    if metric == "l2_hit_rate":
        reports = [r for r in result.cache if kind == "program" or r.launch == target]
        if kind == "op":
            for report in result.cache:
                for access in report.per_access:
                    if access.op_path == target and access.mode == "read":
                        if access.l2_served_rate is None:
                            return None, None, "model hit rate unknown: %s" % access.rates_source
                        return access.l2_served_rate, "cache.per_access.l2_served_rate", "ok"
            return None, None, "op has no read access in the cache report"
        requests = served = 0.0
        for report in reports:
            for access in report.per_access:
                if access.mode != "read":
                    continue
                if access.l2_served_rate is None:
                    return None, None, "model hit rate unknown for %s: %s" % (access.op_path, access.rates_source)
                requests += access.request_count
                served += access.request_count * access.l2_served_rate
        if requests <= 0.0:
            return None, None, "no cache-modelled read requests"
        return served / requests, "cache.reads.l2_served/requests", "ok"
    return None, None, "unsupported metric"


def compare(
    baseline: AnalysisResult,
    calibrated: AnalysisResult,
    observations: Sequence[Observation] = (),
) -> ComparisonResult:
    if not isinstance(baseline, AnalysisResult) or not isinstance(calibrated, AnalysisResult):
        raise ModelingValidationError("compare expects two AnalysisResult objects")
    diagnostics = []  # type: List[str]
    if baseline.program.launch_order != calibrated.program.launch_order:
        diagnostics.append("launch order differs between baseline and calibrated results")
    base_prov = dict(baseline.diagnostics.provenance)
    cal_prov = dict(calibrated.diagnostics.provenance)
    if base_prov.get("context") != cal_prov.get("context"):
        diagnostics.append("calibrated result uses a different Context (kept as context change, not workload mismatch)")
    if baseline.input_digest != calibrated.input_digest:
        diagnostics.append("input digests differ: %s vs %s" % (baseline.input_digest[:12], calibrated.input_digest[:12]))
    for target, digest in baseline.workload_digests:
        other = calibrated.workload_digest(target)
        if other != digest:
            diagnostics.append("workload identity of %s differs (%s vs %s): rows are not comparable"
                               % (target, digest[:12], (other or "missing")[:12]))
    for name, report in (("baseline", baseline), ("calibrated", calibrated)):
        for item in report.diagnostics.approximations:
            diagnostics.append("%s approximation retained: %s" % (name, item))

    alignment = []  # type: List[AlignmentEntry]
    metrics = []  # type: List[MetricRow]
    deltas = []  # type: List[DeltaRow]
    seen = set()
    def workload_target(target: str, hint: Optional[str], metric: str, scope: str) -> str:
        kind = _kind(baseline, target, hint, metric, scope)
        if kind == "op":
            return baseline.op(target).launch
        if kind == "program":
            return "program:" + target
        return target

    for obs in observations:
        key = (obs.target, obs.metric, obs.scope, obs.instance_id)
        if key in seen:
            alignment.append(AlignmentEntry(obs.target, obs.metric, _kind(baseline, obs.target, obs.target_kind, obs.metric, obs.scope), "ambiguous",
                                            "duplicate observation for the same target/metric/scope", obs.scope))
            continue
        seen.add(key)
        base_value, model_scope, reason = _model_value(baseline, obs.target, obs.metric, obs.scope, obs.target_kind)
        cal_value, _cal_scope, cal_reason = _model_value(calibrated, obs.target, obs.metric, obs.scope, obs.target_kind)
        kind = _kind(baseline, obs.target, obs.target_kind, obs.metric, obs.scope)
        comparable = base_value is not None and cal_value is not None
        status = "matched" if comparable else "mismatched"
        if comparable:
            identity_target = workload_target(obs.target, obs.target_kind, obs.metric, obs.scope)
            base_identity = baseline.workload_digest(identity_target)
            cal_identity = calibrated.workload_digest(identity_target)
            if base_identity is None or base_identity != cal_identity:
                comparable = False
                status = "mismatched"
                reason = ("workload identity differs between baseline and calibrated results "
                          "(%s vs %s); shape/dtype/work changes are not a calibration"
                          % ((base_identity or "?")[:12], (cal_identity or "?")[:12]))
            elif obs.workload_digest is not None and obs.workload_digest != base_identity:
                comparable = False
                status = "mismatched"
                reason = "observation workload digest %s does not match model workload %s" % (
                    obs.workload_digest[:12], base_identity[:12])
        if obs.metric == "l2_hit_rate" and comparable and obs.denominator != L2_HIT_DENOMINATOR:
            comparable = False
            status = "mismatched"
            reason = "hit-rate denominator %r != model denominator %r" % (obs.denominator, L2_HIT_DENOMINATOR)
        alignment.append(AlignmentEntry(obs.target, obs.metric, kind, status, reason if base_value is None or not comparable else "ok",
                                        obs.scope, model_scope))
        metrics.append(MetricRow(
            obs.target, obs.metric, obs.unit, obs.scope, base_value, cal_value, obs.value, comparable,
            reason if not comparable else "ok", obs.denominator, obs.source,
        ))
        model_delta = None if base_value is None or cal_value is None else cal_value - base_value
        if comparable:
            b_res = base_value - obs.value
            c_res = cal_value - obs.value
            rel_b = None if obs.value == 0 else abs(b_res) / abs(obs.value)
            rel_c = None if obs.value == 0 else abs(c_res) / abs(obs.value)
            deltas.append(DeltaRow(obs.target, obs.metric, obs.scope, model_delta, b_res, c_res, abs(b_res), abs(c_res), rel_b, rel_c))
        else:
            deltas.append(DeltaRow(obs.target, obs.metric, obs.scope, model_delta, None, None, None, None, None, None))
    # Model-only rows for launches/program without observations.
    observed_keys = {(o.target, o.metric) for o in observations}
    for name, launch in baseline.launches.items():
        if (name, "time_s") in observed_keys:
            continue
        other = calibrated.launches.get(name)
        cal_value = None if other is None else other.kernel_body_s
        metrics.append(MetricRow(name, "time_s", "s", "kernel_elapsed", launch.kernel_body_s, cal_value, None, False,
                                 "missing_observation", None, None))
        alignment.append(AlignmentEntry(name, "time_s", "launch", "missing_observation", "no observation supplied", None, "kernel_body_s"))
        deltas.append(DeltaRow(name, "time_s", "kernel_elapsed",
                               None if cal_value is None else cal_value - launch.kernel_body_s,
                               None, None, None, None, None, None))
    operations = []  # type: List[OpDelta]
    op_observations = {o.target: o for o in observations if _kind(baseline, o.target, o.target_kind) == "op" and o.metric == "time_s"}
    for path, op in baseline.ops.items():
        other = calibrated.ops.get(path)
        if other is None:
            diagnostics.append("op %s is missing from the calibrated result" % path)
            continue

        def completion(item):
            return item.cost_groups[0].completion_latency_s if item.cost_groups else None

        def ddr(item):
            return item.traffic_total.ddr_read_bytes + item.traffic_total.ddr_write_bytes

        note = "kernel-level observations are not apportioned to ops"
        residual = None
        obs = op_observations.get(path)
        if obs is not None:
            note = "op-level time observation has no matching model range; residual withheld"
        operations.append(OpDelta(path, op.count, other.count, completion(op), completion(other), ddr(op), ddr(other), residual, note))
    return ComparisonResult(tuple(alignment), tuple(metrics), tuple(deltas), tuple(operations), tuple(dict.fromkeys(diagnostics)))


__all__ = ["AlignmentEntry", "ComparisonResult", "DeltaRow", "L2_HIT_DENOMINATOR", "MetricRow", "Observation", "OpDelta", "compare"]
