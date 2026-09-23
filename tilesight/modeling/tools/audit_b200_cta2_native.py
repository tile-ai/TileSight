#!/usr/bin/env python3
"""Audit the source-partitioned B200 CTA2 protocol model against CUTLASS.

This is deliberately a *legacy-calibrated native protocol audit*.  The
explicit two-participant DAG comes from ``adapters/b200_cta2_gemm.py``, while
its memory/Tensor/store costs are bound to the existing calibrated GEMM
model.  It is not an independent CTA2 hardware calibration.

The tool samples the legacy cache once per cache key and reuses that exact
result inside ``model_b200_cta2_native``.  It therefore avoids evaluating the
cache model separately for the legacy and protocol predictions.  Input may be
the raw unified B200 CSV or a generated prediction CSV; if ``legacy_body_us``
is present, exact parity with that maintained column is checked fail-closed.

The current legacy cost function changes a requested M cluster to N whenever
``gridM < 2``.  The audit never lets that heuristic alter the source-declared
native 2x1 spatial grid: it restores M and marks those rows
``legacy_axis_mismatch=true``.  Their native values are sensitivity results
bound to the old N-axis per-iteration costs, not M-axis accuracy evidence.
Rows whose measured kernel name says ``_1sm`` are also retained only for the
requested-vs-measured parsing sensitivity and excluded from the primary clean
scope, even when the unified CSV labels them ``utcmma2, cluster_m=2``.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import tilesight.modeling._pipeline.matmul_pipeline_wave as legacy_module
import tilesight.modeling.adapters.b200_cta2_gemm as cta2_module
from tilesight.arch.b200 import B200
from tilesight.modeling import (
    B200Cta2NativeOptions,
    Cta2ProtocolCosts,
    Kernel,
    ResourceTiming,
    Timing,
    Work,
    model_b200_cta2_native,
)
from tilesight.modeling.tools.compare_gemm_cutlass import (
    REQUIRED,
    select_covered_rows,
)
from tilesight.modeling.tools.generate_gemm_cutlass_predictions import (
    GenerationConfig,
    MEM_LEVELS,
    _cache_key,
    _legacy_cache,
    _raster,
)


AUDIT_VERSION = "b200_cta2_source_partitioned_selectable_boundary_v3"
OUTPUT_FIELDS = (
    "legacy_body_us",
    "legacy_us",
    "native_protocol_body_us",
    "native_protocol_us",
    "native_over_legacy_body_ratio",
    "native_body_delta_us",
    "native_ii_ns",
    "native_mainloop_warmup_ns",
    "native_mainloop_steady_ns",
    "native_mainloop_drain_ns",
    "native_boundary_strategy",
    "modeled_total_waves",
    "modeled_waves_float",
    "modeled_full_waves",
    "modeled_tail_work_units",
    "modeled_work_units_per_wave",
    "modeled_logical_work_units",
    "modeled_physical_ctas",
    "modeled_cluster_axis",
    "legacy_cost_cluster_axis",
    "legacy_axis_mismatch",
    "kernel_name_cluster_evidence",
    "modeled_cross_sm_payload_bytes_per_k_iter",
    "native_protocol_audit_version",
    "native_protocol_calibration_scope",
    "native_protocol_interpretation",
    "prediction_launch_us",
    "prediction_boundary_policy",
    "prediction_tma_fixed_latency_ns",
    "prediction_seed",
)


@dataclass(frozen=True)
class AuditTiming:
    selection_s: float
    cache_s: float
    modeling_s: float
    total_s: float
    unique_cache_keys: int


def _integer(row: Mapping[str, str], field: str) -> int:
    value = int(row[field])
    if value <= 0:
        raise ValueError("%s must be positive" % field)
    return value


def _shape(row: Mapping[str, str]) -> Tuple[int, int, int]:
    return tuple(_integer(row, field) for field in ("M", "N", "K"))


def _tile(row: Mapping[str, str]) -> Tuple[int, int, int]:
    return tuple(_integer(row, field) for field in ("tile_m", "tile_n", "tile_k"))


def _compatibility_kernel(row: Mapping[str, str], arch: Any) -> Any:
    """Build the canonical validated GEMM IR for one measured 2x1 cluster.

    The CUTLASS rows in this audit explicitly say ``cluster=(2,1,1)``.  The M
    axis remains authoritative even when the second rank is a predicated tail
    CTA (for example M < tile_M); silently changing such a launch to 1x2 would
    compare a different kernel.
    """

    m, n, k = _shape(row)
    tm, tn, tk = _tile(row)
    stages = _integer(row, "stages")
    grid_m = int(math.ceil(float(m) / tm))
    grid_n = int(math.ceil(float(n) / tn))
    # The old calibrated function falls back to N when gridM<2.  This
    # compatibility IR must first pass that old function's validator so its
    # costs can be recovered.  ``audit_rows`` then overrides only the spatial
    # wave facts to the measured source M cluster before the native DAG is
    # built.  The mismatch is retained explicitly in every output row.
    legacy_axis = "M" if grid_m >= 2 else "N"
    if legacy_axis == "M":
        legacy_grid = (int(math.ceil(float(grid_m) / 2)), grid_n, 1)
        legacy_cluster = (2, 1, 1)
    else:
        legacy_grid = (grid_m, int(math.ceil(float(grid_n) / 2)), 1)
        legacy_cluster = (1, 2, 1)
    physical_grid = (
        legacy_grid[0] * legacy_cluster[0],
        legacy_grid[1] * legacy_cluster[1],
        1,
    )
    raster = _raster(row, arch)

    # The maintained CSV audit has no warp tile.  This is the same explicit
    # CTA-sized shadow assumption used by generate_gemm_cutlass_predictions.
    warp_tile = (tm, tn, tk)
    threads = 32
    dtype = "fp16"
    tiny = 1.0e-12
    kernel = Kernel(
        "b200_cta2_cutlass_audit",
        params={
            "model_adapter": "gemm_pipeline_wave",
            "shape": (m, n, k),
            "tile": (tm, tn, tk),
            "warp_tile": warp_tile,
            "mma_type": "utcmma_cta2",
            "mem_levels": MEM_LEVELS,
            "row_panel": raster.row_panel,
            "column_panel": raster.column_panel,
            "raster_axis": raster.axis,
            "batch": 1,
            "dtype": dtype,
            "accumulator_dtype": dtype,
            # This source fact must dominate any shape-based preference.
            "cluster_axis": "M",
        },
    )
    with kernel.launch(
        work_grid=legacy_grid,
        physical_grid=physical_grid,
        threads=threads,
        cluster=legacy_cluster,
        residency="auto",
        scheduler="static",
    ) as launch:
        with launch.periodic(
            "ko", iterations=int(math.ceil(float(k) / tk)), stages=stages
        ) as loop:
            a = loop.buffer(
                "A_shared",
                scope="smem",
                shape=(tm, tk),
                dtype=dtype,
                slots=stages,
                execution_scope="cta",
            )
            b = loop.buffer(
                "B_shared",
                scope="smem",
                shape=(tk, tn),
                dtype=dtype,
                slots=stages,
                execution_scope="cta",
            )
            accumulator = loop.buffer(
                "C_accumulator",
                scope="fragment",
                shape=(tm, tn),
                dtype=dtype,
                slots=1,
                execution_scope="warpgroup",
            )
            with loop.actor("producer") as producer:
                load_a = producer.phase(
                    "load_A",
                    work=Work.copy(
                        tm * tk * MEM_LEVELS["in1"][-1],
                        source="gmem",
                        target="smem",
                        dtype=dtype,
                    ),
                    timing=Timing(tiny, (ResourceTiming("tma", tiny),)),
                    writes=(a,),
                )
                load_b = producer.phase(
                    "load_B",
                    work=Work.copy(
                        tk * tn * MEM_LEVELS["in2"][-1],
                        source="gmem",
                        target="smem",
                        dtype=dtype,
                    ),
                    timing=Timing(tiny, (ResourceTiming("tma", tiny),)),
                    writes=(b,),
                )
                producer.sequence(load_a.at(0), load_b.at(0), resource="tma")
            with loop.actor("tensor") as tensor:
                mma = tensor.phase(
                    "mma",
                    work=Work.mma(
                        2 * tm * tn * tk * 2,
                        shape=(2 * tm, tn, tk),
                        input_dtype=dtype,
                        accumulator_dtype=dtype,
                    ),
                    timing=Timing(tiny, (ResourceTiming("tensor", tiny),)),
                    reads=(a, b),
                    writes=(accumulator,),
                )
            loop.pipeline_buffer(a, acquire=load_a.start, release=mma.done, capacity=stages)
            loop.pipeline_buffer(b, acquire=load_b.start, release=mma.done, capacity=stages)
            state = loop.state("accumulator", storage=accumulator)
            loop.carry(state, source=mma.done, target=mma.start, distance=1)
    return kernel.build()


def _source_m_evaluation(evaluation: Any, arch: Any) -> Any:
    """Replace heuristic spatial facts with the measured 2x1 source launch.

    Resource costs and ``legacy_result.total_latency`` stay untouched.  For
    skinny-M cases they therefore remain the old N-fallback calibration and
    are marked as an axis mismatch; only the native protocol launch uses the
    source-declared M cluster and its predicated tail rank.
    """

    spec = evaluation.spec
    grid_m = int(math.ceil(float(spec.op_shape[0]) / spec.tb_shape[0]))
    grid_n = int(math.ceil(float(spec.op_shape[1]) / spec.tb_shape[1]))
    work_grid = (int(math.ceil(float(grid_m) / 2)), grid_n)
    total_work_units = work_grid[0] * work_grid[1]
    effective_sm_count = int(arch.sm_count) // 2
    occupancy = int(evaluation.legacy_result.tiles_per_sm)
    work_units_per_wave = max(effective_sm_count * occupancy, 1)
    full_waves = total_work_units // work_units_per_wave
    tail = total_work_units % work_units_per_wave
    wave = replace(
        evaluation.wave,
        work_grid=work_grid,
        total_work_units=total_work_units,
        effective_sm_count=effective_sm_count,
        effective_work_units_per_logical_slot=occupancy,
        work_units_per_wave=work_units_per_wave,
        full_waves=full_waves,
        tail_work_units=tail,
        total_waves=full_waves + (1 if tail else 0),
        waves_float=float(total_work_units) / work_units_per_wave,
        cluster_axis="M",
        cluster_size=2,
        effective_mma_type="utcmma_cta2",
    )
    return replace(evaluation, wave=wave)


def _metrics(rows: Sequence[Mapping[str, str]], prediction: str) -> Dict[str, float]:
    measured = [float(row["measured_us"]) for row in rows]
    predicted = [float(row[prediction]) for row in rows]
    relative = [
        (estimate - actual) / actual
        for estimate, actual in zip(predicted, measured)
    ]
    return {
        "n": float(len(rows)),
        "mape": 100.0 * sum(abs(value) for value in relative) / len(relative),
        "wmape": 100.0
        * sum(abs(estimate - actual) for estimate, actual in zip(predicted, measured))
        / sum(measured),
        "bias": 100.0 * sum(relative) / len(relative),
        "weighted_bias": 100.0 * (sum(predicted) - sum(measured)) / sum(measured),
    }


def _wave_bucket(row: Mapping[str, str]) -> str:
    waves = int(row["modeled_total_waves"])
    if waves == 1:
        return "1"
    if waves <= 4:
        return "2-4"
    if waves <= 16:
        return "5-16"
    if waves <= 64:
        return "17-64"
    return ">=65"


def _groups() -> Tuple[Tuple[str, Any], ...]:
    return (
        (
            "tile",
            lambda row: "%sx%sx%s"
            % (row["tile_m"], row["tile_n"], row["tile_k"]),
        ),
        ("stage", lambda row: row["stages"]),
        ("waves", _wave_bucket),
        ("legacy_axis_mismatch", lambda row: row["legacy_axis_mismatch"]),
        (
            "kernel_cluster_evidence",
            lambda row: row["kernel_name_cluster_evidence"],
        ),
    )


def _print_metrics(label: str, metric: Mapping[str, float]) -> None:
    print(
        "%s n=%d MAPE=%.4f%% wMAPE=%.4f%% bias=%+.4f%% wBias=%+.4f%%"
        % (
            label,
            int(metric["n"]),
            metric["mape"],
            metric["wmape"],
            metric["bias"],
            metric["weighted_bias"],
        )
    )


def _print_ratio(label: str, rows: Sequence[Mapping[str, str]]) -> None:
    ratios = sorted(float(row["native_over_legacy_body_ratio"]) for row in rows)
    weighted = sum(float(row["native_protocol_body_us"]) for row in rows) / sum(
        float(row["legacy_body_us"]) for row in rows
    )
    p95 = ratios[min(int(math.ceil(0.95 * len(ratios))) - 1, len(ratios) - 1)]
    print(
        "%s ratio_native_body/legacy_body n=%d min=%.6f p50=%.6f "
        "mean=%.6f p95=%.6f max=%.6f weighted=%.6f"
        % (
            label,
            len(ratios),
            ratios[0],
            statistics.median(ratios),
            statistics.mean(ratios),
            p95,
            ratios[-1],
            weighted,
        )
    )


def audit_rows(
    rows: Sequence[Dict[str, str]],
    *,
    seed: int = 17,
    launch_us: float = 2.0,
    boundary_policy: str = "periodic_witness",
    tma_fixed_latency_ns: float = 0.0,
    limit: Optional[int] = None,
) -> Tuple[List[Dict[str, str]], AuditTiming, Mapping[str, int]]:
    started = time.perf_counter()
    selection_started = time.perf_counter()
    selected, reasons, before_dedupe = select_covered_rows(
        rows,
        "b200",
        "cta2",
        prediction_columns=("legacy_body_us",) if rows and "legacy_body_us" in rows[0] else (),
    )
    if limit is not None:
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            raise ValueError("limit must be a positive integer")
        selected = selected[:limit]
    selection_s = time.perf_counter() - selection_started
    if not selected:
        raise ValueError("CTA2 selector produced no rows")
    if boundary_policy not in ("periodic_witness", "finite_witness"):
        raise ValueError("unsupported CTA2 audit boundary_policy")
    if not math.isfinite(tma_fixed_latency_ns) or tma_fixed_latency_ns < 0.0:
        raise ValueError("tma_fixed_latency_ns must be finite and non-negative")

    config = GenerationConfig("b200", "cta2", seed=seed, launch_us=launch_us)
    arch = B200().set_to_microbench()
    cache_memo = {}  # type: Dict[Tuple[str, ...], Mapping[str, float]]
    cache_s = 0.0
    modeling_s = 0.0
    output = []  # type: List[Dict[str, str]]
    max_existing_delta_us = 0.0
    axis_mismatch_rows = 0

    for index, row in enumerate(selected):
        key = _cache_key(row)
        if key not in cache_memo:
            cache_started = time.perf_counter()
            cache_memo[key] = _legacy_cache(row, arch, config)
            cache_s += time.perf_counter() - cache_started

        model_started = time.perf_counter()
        ir = _compatibility_kernel(row, arch)
        original_cache = legacy_module.multi_level_hit_rate
        original_evaluate = cta2_module.evaluate_gemm_legacy
        legacy_axis = {}  # type: Dict[str, Optional[str]]

        def evaluate_with_source_m(kernel: Any, source_arch: Any) -> Any:
            raw = original_evaluate(kernel, source_arch)
            legacy_axis["value"] = raw.wave.cluster_axis
            return _source_m_evaluation(raw, source_arch)

        try:
            legacy_module.multi_level_hit_rate = (
                lambda *_args, _cached=cache_memo[key], **_kwargs: dict(_cached)
            )
            cta2_module.evaluate_gemm_legacy = evaluate_with_source_m
            evaluation = model_b200_cta2_native(
                ir,
                arch,
                B200Cta2NativeOptions(
                    plan_kind="source_partitioned",
                    plan_source="cutlass_sm100_dense_2sm_partition_contract",
                    protocol_costs=Cta2ProtocolCosts(
                        tma_fixed_latency_s=tma_fixed_latency_ns * 1.0e-9,
                    ),
                    boundary_policy=boundary_policy,
                    kernel_launch_s=launch_us * 1.0e-6,
                ),
            )
        finally:
            cta2_module.evaluate_gemm_legacy = original_evaluate
            legacy_module.multi_level_hit_rate = original_cache
        modeling_s += time.perf_counter() - model_started

        if evaluation.plan.cluster_axis != "M":
            raise ValueError(
                "row %d changed measured cluster 2x1 into axis %s"
                % (index, evaluation.plan.cluster_axis)
            )
        if evaluation.plan.payload_transfer_bytes_per_iteration != 0.0:
            raise ValueError("source-partitioned plan unexpectedly created DSM payload")
        provenance = dict(evaluation.provenance)
        if provenance.get("independent_hardware_calibration") is not False:
            raise ValueError("CTA2 protocol audit lost legacy-bound provenance")
        legacy_cost_axis = legacy_axis.get("value")
        axis_mismatch = legacy_cost_axis != "M"
        axis_mismatch_rows += int(axis_mismatch)

        legacy_body_us = evaluation.legacy.legacy_result.total_latency * 1.0e6
        native_body_us = evaluation.native.kernel_body_s * 1.0e6
        mainloop = evaluation.native.region.children[0]
        if mainloop.selected_ii_s is None:
            raise ValueError("CTA2 native mainloop did not return a periodic witness")
        if row.get("legacy_body_us", ""):
            delta = abs(float(row["legacy_body_us"]) - legacy_body_us)
            max_existing_delta_us = max(max_existing_delta_us, delta)
            tolerance = 1.0e-9 + 1.0e-9 * max(abs(legacy_body_us), 1.0)
            if delta > tolerance:
                raise ValueError(
                    "row %d disagrees with existing legacy_body_us by %.12g us"
                    % (index, delta)
                )

        wave = evaluation.legacy.wave
        item = dict(row)
        kernel_cluster_evidence = (
            "contradictory_1sm_suffix_with_utcmma2_cluster2"
            if "_1sm" in row["kernel_name"].lower()
            else "consistent_2sm_suffix"
        )
        item.update(
            {
                "legacy_body_us": repr(legacy_body_us),
                "legacy_us": repr(legacy_body_us + launch_us),
                "native_protocol_body_us": repr(native_body_us),
                "native_protocol_us": repr(native_body_us + launch_us),
                "native_over_legacy_body_ratio": repr(
                    native_body_us / legacy_body_us
                ),
                "native_body_delta_us": repr(native_body_us - legacy_body_us),
                "native_ii_ns": repr(mainloop.selected_ii_s * 1.0e9),
                "native_mainloop_warmup_ns": repr(mainloop.first_s * 1.0e9),
                "native_mainloop_steady_ns": repr(mainloop.steady_s * 1.0e9),
                "native_mainloop_drain_ns": repr(mainloop.drain_s * 1.0e9),
                "native_boundary_strategy": mainloop.strategy,
                "modeled_total_waves": str(wave.total_waves),
                "modeled_waves_float": repr(wave.waves_float),
                "modeled_full_waves": str(wave.full_waves),
                "modeled_tail_work_units": str(wave.tail_work_units),
                "modeled_work_units_per_wave": str(wave.work_units_per_wave),
                "modeled_logical_work_units": str(wave.total_work_units),
                "modeled_physical_ctas": str(2 * wave.total_work_units),
                "modeled_cluster_axis": str(wave.cluster_axis),
                "legacy_cost_cluster_axis": str(legacy_cost_axis),
                "legacy_axis_mismatch": str(axis_mismatch).lower(),
                "kernel_name_cluster_evidence": kernel_cluster_evidence,
                "modeled_cross_sm_payload_bytes_per_k_iter": repr(
                    evaluation.plan.payload_transfer_bytes_per_iteration
                ),
                "native_protocol_audit_version": AUDIT_VERSION,
                "native_protocol_calibration_scope": (
                    "legacy_N_axis_per_iteration_costs_source_M_spatial_DAG"
                    if axis_mismatch
                    else "legacy_M_axis_per_iteration_costs_source_M_spatial_DAG"
                ),
                "native_protocol_interpretation": (
                    "invalid_contradictory_1sm_kernel_evidence"
                    if kernel_cluster_evidence
                    == "contradictory_1sm_suffix_with_utcmma2_cluster2"
                    else (
                        "axis_mismatch_sensitivity_only"
                        if axis_mismatch
                        else "axis_consistent_legacy_calibrated_protocol_audit"
                    )
                ),
                "prediction_launch_us": repr(launch_us),
                "prediction_boundary_policy": boundary_policy,
                "prediction_tma_fixed_latency_ns": repr(tma_fixed_latency_ns),
                "prediction_seed": str(seed),
            }
        )
        output.append(item)

    total_s = time.perf_counter() - started
    metadata = {
        "input_rows": len(rows),
        "selected_before_dedupe": before_dedupe,
        "selected_after_dedupe": len(selected),
        "existing_legacy_max_delta_femtoseconds": int(
            round(max_existing_delta_us * 1.0e9)
        ),
        "legacy_axis_mismatch_rows": axis_mismatch_rows,
        "contradictory_1sm_suffix_rows": sum(
            item["kernel_name_cluster_evidence"]
            == "contradictory_1sm_suffix_with_utcmma2_cluster2"
            for item in output
        ),
    }
    metadata.update({"filtered_%s" % key: value for key, value in reasons.items()})
    return (
        output,
        AuditTiming(selection_s, cache_s, modeling_s, total_s, len(cache_memo)),
        metadata,
    )


def _read(path: Path) -> Tuple[List[Dict[str, str]], Tuple[str, ...]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        missing = sorted(set(REQUIRED) - set(fields))
        if missing:
            raise ValueError("CSV is missing required fields %s" % missing)
        return [dict(row) for row in reader], fields


def _write(
    path: Path, rows: Sequence[Mapping[str, str]], input_fields: Iterable[str]
) -> None:
    fields = list(input_fields)
    for field in OUTPUT_FIELDS:
        if field not in fields:
            fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def report(rows: Sequence[Mapping[str, str]], timing: AuditTiming) -> None:
    print(
        "audit_scope=legacy-calibrated native protocol audit; "
        "independent_hardware_calibration=false"
    )
    print(
        "aggregate_warning=native all-row metrics mix source-M spatial DAGs with "
        "legacy N-axis costs wherever legacy_axis_mismatch=true"
    )
    for prediction in ("legacy_us", "native_protocol_us"):
        _print_metrics("overall/%s" % prediction, _metrics(rows, prediction))
    _print_ratio("overall", rows)
    axis_consistent = [row for row in rows if row["legacy_axis_mismatch"] == "false"]
    if len(axis_consistent) != len(rows):
        print(
            "primary_protocol_scope=axis_consistent kept=%d sensitivity_only=%d"
            % (len(axis_consistent), len(rows) - len(axis_consistent))
        )
        for prediction in ("legacy_us", "native_protocol_us"):
            _print_metrics(
                "axis_consistent/%s" % prediction,
                _metrics(axis_consistent, prediction),
            )
        _print_ratio("axis_consistent", axis_consistent)
    primary_clean = [
        row
        for row in axis_consistent
        if row["kernel_name_cluster_evidence"] == "consistent_2sm_suffix"
    ]
    if len(primary_clean) != len(rows):
        print(
            "primary_clean_protocol_scope=axis_consistent_and_2sm_suffix "
            "kept=%d excluded=%d"
            % (len(primary_clean), len(rows) - len(primary_clean))
        )
        for prediction in ("legacy_us", "native_protocol_us"):
            _print_metrics(
                "primary_clean/%s" % prediction,
                _metrics(primary_clean, prediction),
            )
        _print_ratio("primary_clean", primary_clean)
    consistent = [
        row
        for row in rows
        if row["kernel_name_cluster_evidence"] == "consistent_2sm_suffix"
    ]
    if len(consistent) != len(rows):
        print(
            "sensitivity=exclude_contradictory_1sm_suffix kept=%d excluded=%d"
            % (len(consistent), len(rows) - len(consistent))
        )
        for prediction in ("legacy_us", "native_protocol_us"):
            _print_metrics(
                "consistent_2sm/%s" % prediction,
                _metrics(consistent, prediction),
            )
        _print_ratio("consistent_2sm", consistent)
    for group_name, key in _groups():
        groups = defaultdict(list)
        for row in rows:
            groups[key(row)].append(row)
        for value, group_rows in sorted(groups.items(), key=lambda item: str(item[0])):
            for prediction in ("legacy_us", "native_protocol_us"):
                _print_metrics(
                    "%s=%s/%s" % (group_name, value, prediction),
                    _metrics(group_rows, prediction),
                )
            _print_ratio("%s=%s" % (group_name, value), group_rows)
    print(
        "runtime selection_s=%.6f cache_s=%.6f modeling_s=%.6f total_s=%.6f "
        "unique_cache_keys=%d rows=%d rows_per_s=%.3f"
        % (
            timing.selection_s,
            timing.cache_s,
            timing.modeling_s,
            timing.total_s,
            timing.unique_cache_keys,
            len(rows),
            len(rows) / timing.total_s,
        )
    )


def main(argv: Sequence[str] = ()) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--launch-us", type=float, default=2.0)
    parser.add_argument(
        "--boundary-policy",
        choices=("periodic_witness", "finite_witness"),
        default="periodic_witness",
    )
    parser.add_argument("--tma-fixed-latency-ns", type=float, default=0.0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(tuple(argv) if argv else None)
    if not math.isfinite(args.launch_us) or args.launch_us < 0.0:
        parser.error("--launch-us must be finite and non-negative")
    if (
        not math.isfinite(args.tma_fixed_latency_ns)
        or args.tma_fixed_latency_ns < 0.0
    ):
        parser.error("--tma-fixed-latency-ns must be finite and non-negative")
    try:
        rows, fields = _read(args.csv)
        audited, timing, metadata = audit_rows(
            rows,
            seed=args.seed,
            launch_us=args.launch_us,
            boundary_policy=args.boundary_policy,
            tma_fixed_latency_ns=args.tma_fixed_latency_ns,
            limit=args.limit,
        )
        _write(args.out, audited, fields)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print("selection=%s" % dict(sorted(metadata.items())))
    print("audit_csv=%s" % args.out)
    report(audited, timing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
