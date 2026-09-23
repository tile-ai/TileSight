"""Independent comparison entry between ``analyze`` predictions and measurements.

A row only enters the accuracy statistics when the *identity* of the model and
the measurement agree on every field in ``IDENTITY_FIELDS`` (kernel,
implementation and source version, variant, architecture, dtype, causal/GQA
convention, tile, scheduler, cache state and timing scope) and the shape is
the same.  A missing or different field makes the row ``unverified`` with the
reasons listed.  For such rows the prediction is still reported as an
*exploratory* cross-implementation comparison whose statistics are kept
separate from the comparable ones.  Failed or empty measurements are kept as
``failed_measurement`` rows and never read as zero time.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .analysis import UnsupportedError, analyze
from .contract import ContractError, CutlassSm90, Program
from .options import Context, Options

IDENTITY_FIELDS = (
    "kernel", "implementation", "source_version", "variant", "arch", "dtype", "causal", "gqa",
    "tile", "scheduler", "cache_state", "timing_scope",
)
TIMING_SCOPES = ("kernel_elapsed", "device_span", "host_wall")
_TOKEN = re.compile(r"^([A-Za-z]+)(\d+)$")


def parse_shape_id(shape_id: str) -> Dict[str, int]:
    """``b1_h32_kv8_s2048_d128`` or ``M8192_N3584_K8192`` -> lower-case keyed integers."""

    result = {}  # type: Dict[str, int]
    for token in shape_id.split("_"):
        match = _TOKEN.match(token)
        if match is None:
            raise ValueError("unrecognised shape token %r in %r" % (token, shape_id))
        key = match.group(1).lower()
        if key in result:
            raise ValueError("shape token %r repeated in %r" % (key, shape_id))
        result[key] = int(match.group(2))
    return result


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class Measurement:
    shape_id: str
    status: str  # ok | failed_measurement | unparsable_shape
    measured_us: Optional[float]
    identity: Tuple[Tuple[str, Any], ...]
    source_file: str
    row: int
    run_id: str = ""
    reason: str = ""
    raw: Tuple[Tuple[str, str], ...] = ()
    identity_conflicts: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {"shape_id": self.shape_id, "status": self.status, "measured_us": self.measured_us,
                "identity": dict(self.identity), "source_file": self.source_file, "row": self.row,
                "run_id": self.run_id, "reason": self.reason, "identity_conflicts": list(self.identity_conflicts)}


def load_measurements(
    path: str,
    *,
    kernel: Optional[str] = None,
    identity: Mapping[str, Any] = (),
    value_column: str = "measured_us",
    status_column: Optional[str] = None,
    column_identity: Mapping[str, str] = (("kernel", "kernel"), ("arch", "arch"), ("dtype", "precision")),
) -> Tuple[Measurement, ...]:
    """Read a measurement CSV.

    ``identity`` holds what the file does not say (implementation, source
    version, variant, causal, tile, scheduler, cache state, timing scope) and
    should come from the measurement log or launcher.  Columns named in
    ``column_identity`` fill identity fields per row.  Rows with an empty,
    non-numeric or non-positive value are kept as ``failed_measurement``.
    """

    rows = []
    with open(path, newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle), start=2):
            if kernel is not None and row.get("kernel") not in (None, kernel) and "kernel" in row:
                continue
            # The row is authoritative; the file-level declaration only fills gaps,
            # and a declaration that contradicts the row is reported, never applied.
            ident = {}
            conflicts = []
            for name, column in dict(column_identity).items():
                value = row.get(column)
                if value not in (None, ""):
                    ident[name] = value.lower() if isinstance(value, str) else value
            for name, value in dict(identity).items():
                if name in ident and str(ident[name]).lower() != str(value).lower():
                    conflicts.append("declared %s=%r contradicts the row value %r" % (name, value, ident[name]))
                else:
                    ident.setdefault(name, value)
            shape_id = row.get("shape_id", "")
            status, reason, measured = "ok", "", None
            raw_value = (row.get(value_column) or "").strip()
            try:
                parse_shape_id(shape_id)
            except ValueError as error:
                status, reason = "unparsable_shape", str(error)
            if status == "ok":
                try:
                    measured = float(raw_value)
                    if not (measured > 0.0):
                        status, reason, measured = "failed_measurement", "non-positive %s=%r" % (value_column, raw_value), None
                except ValueError:
                    status, reason = "failed_measurement", "empty or non-numeric %s=%r" % (value_column, raw_value)
            if status == "ok" and status_column and (row.get(status_column) or "ok").lower() not in ("ok", "measured", ""):
                status, reason, measured = "failed_measurement", "%s=%s" % (status_column, row.get(status_column)), None
            rows.append(Measurement(shape_id, status, measured, tuple(sorted(ident.items())), path, index,
                                    row.get("run_id", "") or "", reason, tuple(sorted((k, v or "") for k, v in row.items())),
                                    tuple(conflicts)))
    return tuple(rows)


@dataclass(frozen=True)
class ValidationCase:
    shape_id: str
    params: Tuple[Tuple[str, Any], ...]
    status: str  # compared | unverified | unsupported | failed_measurement | unparsable_shape
    predicted_us: Optional[float]
    measured_us: Optional[float]
    relative_error: Optional[float]
    signed_error_us: Optional[float]
    reasons: Tuple[str, ...]
    exploratory: bool = False
    baseline_us: Optional[float] = None
    metadata: Tuple[Tuple[str, Any], ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        data = {name: getattr(self, name) for name in self.__dataclass_fields__}
        data["params"] = dict(self.params)
        data["metadata"] = dict(self.metadata)
        return data


def _stats(cases: Sequence[ValidationCase]) -> Dict[str, Any]:
    usable = [c for c in cases if c.relative_error is not None and c.measured_us]
    if not usable:
        return {"n": 0, "mape_pct": None, "median_ape_pct": None, "wmape_pct": None}
    errors = sorted(abs(c.relative_error) for c in usable)
    middle = len(errors) // 2
    median = errors[middle] if len(errors) % 2 else 0.5 * (errors[middle - 1] + errors[middle])
    wmape = sum(abs(c.signed_error_us) for c in usable) / sum(c.measured_us for c in usable)
    return {"n": len(usable), "mape_pct": 100.0 * sum(errors) / len(errors), "median_ape_pct": 100.0 * median,
            "wmape_pct": 100.0 * wmape}


@dataclass(frozen=True)
class ValidationReport:
    name: str
    model_identity: Tuple[Tuple[str, Any], ...]
    measurement_identity: Tuple[Tuple[str, Any], ...]
    cases: Tuple[ValidationCase, ...]
    configuration: Tuple[Tuple[str, Any], ...]
    inputs: Tuple[Tuple[str, Any], ...] = ()

    def by_status(self, status: str) -> Tuple[ValidationCase, ...]:
        return tuple(c for c in self.cases if c.status == status)

    @property
    def compared(self) -> Tuple[ValidationCase, ...]:
        return self.by_status("compared")

    @property
    def exploratory(self) -> Tuple[ValidationCase, ...]:
        return tuple(c for c in self.cases if c.exploratory)

    @property
    def comparable_stats(self) -> Dict[str, Any]:
        return _stats(self.compared)

    @property
    def exploratory_stats(self) -> Dict[str, Any]:
        return _stats(self.exploratory)

    @property
    def baseline_stats(self) -> Dict[str, Any]:
        """Frozen baseline on the *intersection*: rows where this entry also has a prediction."""

        return self._baseline([c for c in self.cases if c.predicted_us is not None])

    @property
    def baseline_stats_all(self) -> Dict[str, Any]:
        """Frozen baseline over every measured row (includes rows this entry could not predict)."""

        return self._baseline(list(self.cases))

    @staticmethod
    def _baseline(cases: Sequence[ValidationCase]) -> Dict[str, Any]:
        usable = [c for c in cases if c.baseline_us is not None and c.measured_us]
        if not usable:
            return {"n": 0, "mape_pct": None, "median_ape_pct": None, "wmape_pct": None}
        errors = sorted(abs(c.baseline_us - c.measured_us) / c.measured_us for c in usable)
        middle = len(errors) // 2
        median = errors[middle] if len(errors) % 2 else 0.5 * (errors[middle - 1] + errors[middle])
        wmape = sum(abs(c.baseline_us - c.measured_us) for c in usable) / sum(c.measured_us for c in usable)
        return {"n": len(usable), "mape_pct": 100.0 * sum(errors) / len(errors), "median_ape_pct": 100.0 * median,
                "wmape_pct": 100.0 * wmape}

    @property
    def baseline_agreement(self) -> Dict[str, Any]:
        """Distance between this entry's predictions and the frozen baseline (model vs model)."""

        ratios = sorted(c.predicted_us / c.baseline_us for c in self.cases
                        if c.predicted_us is not None and c.baseline_us)
        if not ratios:
            return {"n": 0, "mean_abs_diff_pct": None, "median_ratio": None, "p10_ratio": None, "p90_ratio": None,
                    "max_abs_diff_pct": None}
        diffs = [abs(r - 1.0) for r in ratios]
        return {"n": len(ratios), "mean_abs_diff_pct": 100.0 * sum(diffs) / len(diffs),
                "median_ratio": ratios[len(ratios) // 2], "p10_ratio": ratios[len(ratios) // 10],
                "p90_ratio": ratios[9 * len(ratios) // 10], "max_abs_diff_pct": 100.0 * max(diffs)}

    @property
    def mape(self) -> Optional[float]:
        value = self.comparable_stats["mape_pct"]
        return None if value is None else value / 100.0

    def status_counts(self) -> Dict[str, int]:
        counts = {}  # type: Dict[str, int]
        for case in self.cases:
            counts[case.status] = counts.get(case.status, 0) + 1
        return counts

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name, "model_identity": dict(self.model_identity),
            "measurement_identity": dict(self.measurement_identity), "configuration": dict(self.configuration),
            "inputs": dict(self.inputs), "status_counts": self.status_counts(),
            "comparable_stats": self.comparable_stats, "exploratory_stats": self.exploratory_stats,
            "baseline_stats": self.baseline_stats, "baseline_stats_all": self.baseline_stats_all,
            "baseline_agreement": self.baseline_agreement,
            "cases": [c.to_dict() for c in self.cases],
        }

    def write_json(self, path: str) -> str:
        with open(path, "w") as handle:
            json.dump(self.to_dict(), handle, indent=1, sort_keys=True, default=str)
        return path


def arch_names(arch: Any) -> Tuple[str, ...]:
    """Names an architecture object answers to (class prefix and declared core), lower case."""

    names = {type(arch).__name__.split("_")[0].lower()}
    core = getattr(arch, "core", None)
    if isinstance(core, str) and core:
        names.add(core.lower())
    return tuple(sorted(names))


def program_facts(program: Program) -> Dict[str, Any]:
    """Identity facts read from the Program itself (not from a declaration).

    ``dtype_roles`` keeps the four roles apart: ``input``/``output`` are the dtypes
    of the on-chip buffers that global loads fill / global stores drain (i.e. the
    dtype of the tensor as it sits in device memory), ``compute`` and
    ``accumulation`` come from the op specs.  ``None`` inside a role means an op of
    that role carries no dtype, so the role cannot be verified.
    """

    from .contract import Load, Store

    roles = {"input": set(), "output": set(), "compute": set(), "accumulation": set()}  # type: Dict[str, set]
    tensors = {}  # type: Dict[str, Dict[str, Any]]
    axes = {}
    for launch in program.launches:
        buffers = {}
        for loop in launch.loops():
            for item in getattr(loop.body, "buffers", ()):
                buffers[item.name] = item.dtype or None
        for axis in launch.work_axes:
            axes["%s/%s" % (launch.name, axis.name)] = axis.extent
        # Buffers outside a pipeline body (e.g. an epilogue's register fragment) take the
        # result dtype of the compute that produces them.
        for op in launch.ops():
            spec = getattr(op, "spec", None)
            result = getattr(getattr(spec, "result", None), "storage_dtype", None) or getattr(spec, "result_storage_dtype", None)
            if result is not None:
                for name in op.writes:
                    buffers.setdefault(name, result.name)
        for op in launch.ops():
            spec = getattr(op, "spec", None)
            if spec is not None:
                for role in ("compute", "accumulation"):
                    value = getattr(spec, role + "_dtype", None)
                    if value is not None:
                        roles[role].add(value.name)
            declared = dict(op.metadata).get("tensor_dtype")
            if isinstance(op, Load) and op.source == "global":
                roles["input"].update([declared] if declared else [buffers.get(name) for name in op.writes] or [None])
            if isinstance(op, Store) and op.destination == "global":
                roles["output"].update([declared] if declared else [buffers.get(name) for name in op.reads] or [None])
            access = getattr(op, "access", None)
            if access is not None:
                record = tensors.setdefault(access.tensor, {"tile_shape": access.tile_shape, "element_bytes": access.element_bytes})
                if access.tensor_shape is not None:
                    record["tensor_shape"] = access.tensor_shape
    dtype_roles = {role: tuple(sorted(values, key=lambda v: (v is None, v or ""))) for role, values in roles.items()}
    return {"dtype_roles": dtype_roles, "tensors": tensors, "work_axes": axes}


DTYPE_ROLE_FIELDS = (("compute_dtype", "compute"), ("accumulation_dtype", "accumulation"))


def bound_identity_differences(model_identity: Mapping[str, Any], arch: Any, program: Program) -> Tuple[str, ...]:
    """Reasons why the *declared* model identity is not backed by the actual arch/program.

    ``dtype`` is the tensor dtype: every global input and output of the program must
    have exactly that dtype (an FP32 accumulator does not make a BF16 program FP32).
    ``compute_dtype``/``accumulation_dtype`` are checked against the op specs when declared.
    """

    reasons = []
    declared_arch = str(model_identity.get("arch", "")).lower()
    if declared_arch and declared_arch not in arch_names(arch):
        reasons.append("model identity arch=%r is not the analysed architecture %s" % (declared_arch, list(arch_names(arch))))
    roles = program_facts(program)["dtype_roles"]
    declared_dtype = str(model_identity.get("dtype", "")).lower()
    if declared_dtype:
        for role in ("input", "output"):
            found = roles[role]
            if not found:
                reasons.append("model identity dtype=%r: the program has no global %s to verify it against" % (declared_dtype, role))
            elif None in found:
                reasons.append("model identity dtype=%r: a global %s buffer carries no dtype" % (declared_dtype, role))
            elif set(found) != {declared_dtype}:
                reasons.append("model identity dtype=%r is not the program's %s dtype %s" % (declared_dtype, role, list(found)))
    for field_name, role in DTYPE_ROLE_FIELDS:
        declared = str(model_identity.get(field_name, "")).lower()
        if declared and set(roles[role]) != {declared}:
            reasons.append("model identity %s=%r is not the program's %s dtype %s" % (field_name, declared, role, list(roles[role])))
    return tuple(reasons)


def identity_differences(model: Mapping[str, Any], measured: Mapping[str, Any]) -> Tuple[str, ...]:
    """Reasons why the two identities are not provably the same."""

    reasons = []
    for name in IDENTITY_FIELDS:
        a, b = model.get(name), measured.get(name)
        if a is None and b is None:
            reasons.append("identity field %r missing on both sides" % name)
        elif a is None:
            reasons.append("identity field %r missing on the model" % name)
        elif b is None:
            reasons.append("identity field %r missing on the measurement" % name)
        elif str(a).lower() != str(b).lower():
            reasons.append("%s differs: model=%r measurement=%r" % (name, a, b))
    for name, _role in DTYPE_ROLE_FIELDS:
        a, b = model.get(name), measured.get(name)
        if a is None and b is None:
            continue
        if a is None or b is None:
            reasons.append("%s is declared on only one side (model=%r measurement=%r)" % (name, a, b))
        elif str(a).lower() != str(b).lower():
            reasons.append("%s differs: model=%r measurement=%r" % (name, a, b))
    return tuple(reasons)


def validate(
    measurements: Sequence[Measurement],
    program_factory: Callable[[Dict[str, int], Dict[str, str]], Program],
    arch: Any,
    *,
    name: str,
    model_identity: Mapping[str, Any],
    options: Optional[Options] = None,
    context: Optional[Context] = None,
    configuration: Mapping[str, Any] = (),
    baselines: Mapping[str, float] = (),
    inputs: Mapping[str, Any] = (),
    shape_check: Optional[Callable[[Dict[str, int], Program], Sequence[str]]] = None,
    kernel_body_transform: Optional[Callable[[Any, Program], float]] = None,
) -> ValidationReport:
    """Predict every measured shape; compare only when identities match.

    The predicted quantity follows ``model_identity['timing_scope']``:
    ``kernel_elapsed`` (kernel bodies), ``device_span`` (bodies + launch
    overhead + explicit gaps) or ``host_wall`` (program total incl. host
    dispatch).  ``baselines`` maps shape ids to frozen predictions of an older
    model for a side-by-side with the same metric.
    """

    options = options or Options()
    context = context or Context()
    model_identity = dict(model_identity)
    scope = model_identity.get("timing_scope")
    if scope not in TIMING_SCOPES:
        raise ContractError("model_identity['timing_scope']=%r must be one of %s" % (scope, ", ".join(TIMING_SCOPES)))
    baselines = dict(baselines)
    cases = []
    measurement_identity = {}  # type: Dict[str, Any]
    for item in measurements:
        measurement_identity = measurement_identity or dict(item.identity)
        meta = {"source_file": item.source_file, "row": item.row, "run_id": item.run_id}
        if item.status != "ok":
            cases.append(ValidationCase(item.shape_id, (), item.status, None, None, None, None, (item.reason,),
                                        metadata=tuple(meta.items())))
            continue
        params = parse_shape_id(item.shape_id)
        reasons = identity_differences(model_identity, dict(item.identity))
        try:
            program = program_factory(params, dict(item.raw))
            result = analyze(program, arch, options=options, context=context)
        except (UnsupportedError, ContractError, ValueError, NotImplementedError) as error:
            cases.append(ValidationCase(item.shape_id, tuple(sorted(params.items())), "unsupported", None, item.measured_us,
                                        None, None, (str(error)[:300],), baseline_us=baselines.get(item.shape_id),
                                        metadata=tuple(meta.items())))
            continue
        bodies = sum(l.kernel_body_s for l in result.launches.values())
        if kernel_body_transform is not None:
            # A labelled what-if on top of the analysis (e.g. another II definition); the
            # caller must describe it in ``configuration`` -- it is never a comparable sample.
            bodies = float(kernel_body_transform(result, program))
            reasons = tuple(reasons) + ("prediction is a diagnostic transform of the analysis result, not the entry's own total",)
        overhead = sum(l.launch_overhead_s for l in result.launches.values())
        if scope == "kernel_elapsed":
            predicted = bodies
        elif scope == "device_span":
            predicted = bodies + overhead + result.program.inter_launch_gap_s
        else:
            predicted = result.program.total_s
        predicted_us = predicted * 1e6
        signed = predicted_us - item.measured_us
        relative = signed / item.measured_us
        meta.update({
            "predicted_scope": scope, "input_digest": result.input_digest,
            "workload_digest": result.program_workload_digest,
            "model_unsupported": list(result.diagnostics.unsupported),
            "approximation_count": len(result.diagnostics.approximations),
            "kernel_body_us": bodies * 1e6, "launch_overhead_us": overhead * 1e6,
            "launches": {n: {"waves": l.waves, "slots": l.slots, "residency": l.residency, "work_units": l.work_units}
                         for n, l in result.launches.items()},
            "selected_ii_ns": {path: region.ii.selected_ii_s * 1e9 for path, region in result.regions.items()
                               if region.ii is not None and region.ii.selected_ii_s is not None},
            "launch_metadata": dict(program.launches[0].metadata),
        })
        reasons = tuple(reasons) + tuple(item.identity_conflicts) + bound_identity_differences(model_identity, arch, program)
        if shape_check is None:
            reasons += ("the measured shape was not checked against the generated Program (no shape_check)",)
        else:
            reasons += tuple("shape: %s" % r for r in shape_check(params, program))
        meta["program_facts"] = program_facts(program)
        comparable = not reasons
        cases.append(ValidationCase(
            item.shape_id, tuple(sorted(params.items())), "compared" if comparable else "unverified",
            predicted_us, item.measured_us, relative, signed, reasons, exploratory=not comparable,
            baseline_us=baselines.get(item.shape_id), metadata=tuple(meta.items()),
        ))
    return ValidationReport(name, tuple(sorted(model_identity.items())), tuple(sorted(measurement_identity.items())),
                            tuple(cases), tuple(dict(configuration).items()), tuple(dict(inputs).items()))


def cutlass_sm90_mapper_from_records(
    candidates: Sequence[Mapping[str, str]], grid_m: int, grid_n: int,
) -> CutlassSm90:
    """The CUTLASS mapper of one measured row, or ``ValueError`` when the join cannot decide it.

    Several profiler records may share a row's join key.  They are accepted only
    when every one of them resolves to the same launch order for this tile grid
    (e.g. different max swizzle sizes that CUTLASS clamps to the same value).
    """

    if not candidates:
        raise ValueError("no profiler record for this row")
    mappers = {}
    for item in candidates:
        if item.get("raster_order") not in ("along_m", "along_n", "heuristic") or not item.get("swizzle_size"):
            raise ValueError("profiler record has no raster_order/swizzle_size (kernel %s)" % item.get("kernel_name"))
        mapper = CutlassSm90(cluster_m=int(item["cluster_m"]), cluster_n=int(item["cluster_n"]),
                             max_swizzle_size=int(item["swizzle_size"]), raster_order=item["raster_order"])
        mappers[json.dumps(mapper.canonical(), sort_keys=True)] = mapper
    ordered = sorted(mappers.items())
    if len(ordered) > 1:
        first = ordered[0][1].coordinates((grid_m, grid_n))
        if any(other.coordinates((grid_m, grid_n)) != first for _key, other in ordered[1:]):
            raise ValueError("ambiguous scheduler configuration: %d profiler records match this row and give "
                             "different launch orders (%s)" % (len(candidates), "; ".join(key for key, _m in ordered)))
    return ordered[0][1]



def load_baseline_predictions(path: str, *, shape_column: str = "shape_id", value_column: str = "model_us") -> Dict[str, float]:
    """Frozen per-shape predictions of an earlier model (for a same-metric side-by-side)."""

    result = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            value = (row.get(value_column) or "").strip()
            if value:
                try:
                    result[row[shape_column]] = float(value)
                except ValueError:
                    continue
    return result


__all__ = [
    "IDENTITY_FIELDS", "Measurement", "TIMING_SCOPES", "ValidationCase", "ValidationReport", "arch_names",
    "bound_identity_differences", "cutlass_sm90_mapper_from_records", "file_sha256", "identity_differences", "program_facts", "load_baseline_predictions", "load_measurements", "parse_shape_id", "validate",
]
