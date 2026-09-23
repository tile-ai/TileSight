#!/usr/bin/env python3
"""Audit CUTLASS GEMM CSV coverage and summarize prediction error.

This tool intentionally does not guess missing warp tiles or reinterpret a
cluster launch.  It accepts raw profiler CSVs or result CSVs augmented with
prediction columns (in microseconds), applies the exact coverage filters used
by the new-API audit, and reports MAPE/wMAPE/bias.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


REQUIRED = (
    "M",
    "N",
    "K",
    "precision",
    "measured_us",
    "runtime_ms",
    "kernel_name",
    "tile_m",
    "tile_n",
    "tile_k",
    "stages",
    "tc_path",
    "cluster_m",
    "cluster_n",
    "cluster_k",
    "raster_order",
    "swizzle_size",
    "scheduler_policy",
    "source_tag",
)


def _positive(row: Mapping[str, str], field: str) -> bool:
    try:
        return float(row[field]) > 0.0
    except (KeyError, TypeError, ValueError):
        return False


def _base_reason(row: Mapping[str, str]) -> str:
    for field in (
        "M",
        "N",
        "K",
        "measured_us",
        "tile_m",
        "tile_n",
        "tile_k",
        "stages",
        "swizzle_size",
    ):
        if not _positive(row, field):
            return "missing_or_nonpositive_%s" % field
    if row["raster_order"] not in ("along_m", "along_n"):
        return "unsupported_raster"
    if "stream_k" in row["kernel_name"].lower():
        return "stream_k"
    measured = float(row["measured_us"])
    if row.get("runtime_ms"):
        runtime_us = float(row["runtime_ms"]) * 1000.0
        tolerance = 1.0e-9 + 1.0e-6 * max(measured, runtime_us)
        if abs(measured - runtime_us) > tolerance:
            return "timer_unit_mismatch"
    return "selected"


def _coverage_reason(
    row: Mapping[str, str], arch: str, mapping: str
) -> str:
    reason = _base_reason(row)
    if reason != "selected":
        return reason
    cluster = (row["cluster_m"], row["cluster_n"], row["cluster_k"])
    if arch == "a100":
        if mapping != "cluster1":
            return "invalid_mapping_for_a100"
        if row["precision"] != "fp16" or row["tc_path"] != "h16816":
            return "unsupported_precision_or_tc_path"
        if cluster != ("1", "1", "1"):
            return "cluster_not_1x1x1"
        if row["source_tag"] != "tile_sweep_keep_all":
            return "outside_clean_tile_sweep"
        return "selected"
    if arch == "b200" and mapping == "cta1":
        if row["precision"] != "fp16":
            return "unsupported_precision"
        if row["tc_path"] not in ("wgmma", "utcmma1"):
            return "not_wgmma_or_utcmma1"
        if cluster != ("1", "1", "1"):
            return "cluster_not_1x1x1"
        return "selected"
    if arch == "b200" and mapping == "cta2":
        if row["precision"] != "fp16" or row["tc_path"] != "utcmma2":
            return "not_fp16_utcmma2"
        if cluster != ("2", "1", "1"):
            return "not_measured_2x1_cluster"
        # The audited legacy selector chooses M clustering when tile_m<=tile_n.
        # Reject the other axis rather than silently changing CUTLASS semantics.
        if int(row["tile_m"]) > int(row["tile_n"]):
            return "legacy_cluster_axis_mismatch"
        return "selected"
    if arch == "h200":
        return "h200_ordinary_cluster_not_covered"
    return "unsupported_arch_mapping"


DEDUPE_FIELDS = (
    "M",
    "N",
    "K",
    "precision",
    "kernel_name",
    "tile_m",
    "tile_n",
    "tile_k",
    "stages",
    "tc_path",
    "cluster_m",
    "cluster_n",
    "cluster_k",
    "raster_order",
    "swizzle_size",
    "scheduler_policy",
)


def _dedupe_median(
    rows: Sequence[Dict[str, str]], prediction_columns: Sequence[str]
) -> List[Dict[str, str]]:
    grouped = defaultdict(list)  # type: Dict[Tuple[str, ...], List[Dict[str, str]]]
    for row in rows:
        grouped[tuple(row[field] for field in DEDUPE_FIELDS)].append(row)
    result = []
    for values in grouped.values():
        item = dict(values[0])
        # Keep the normalized timer pair consistent so a generated/deduped CSV
        # can be passed through this selector again without a false unit error.
        for field in ("measured_us", "runtime_ms") + tuple(prediction_columns):
            if field in item and all(value.get(field, "") for value in values):
                item[field] = str(
                    statistics.median(float(value[field]) for value in values)
                )
        result.append(item)
    return result


def select_covered_rows(
    rows: Sequence[Dict[str, str]],
    arch: str,
    mapping: str,
    prediction_columns: Sequence[str] = (),
) -> Tuple[List[Dict[str, str]], Counter, int]:
    """Apply the maintained fail-closed selector and CTA2 median dedupe.

    The prediction generator imports this function so selection semantics
    cannot drift between prediction production and metric reporting.  The
    returned integer is the selected count before the CTA2 dedupe.
    """

    reasons = Counter(_coverage_reason(row, arch, mapping) for row in rows)
    selected = [
        dict(row)
        for row in rows
        if _coverage_reason(row, arch, mapping) == "selected"
    ]
    selected_before_dedupe = len(selected)
    if arch == "b200" and mapping == "cta2":
        selected = _dedupe_median(selected, prediction_columns)
    return selected, reasons, selected_before_dedupe


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


def _latency_bucket(row: Mapping[str, str]) -> str:
    value = float(row["measured_us"])
    if value < 50.0:
        return "<50us"
    if value < 200.0:
        return "50-200us"
    if value < 1000.0:
        return "200-1000us"
    return ">=1000us"


def _groups(rows: Sequence[Mapping[str, str]]) -> Tuple[Tuple[str, Any], ...]:
    return (
        ("tile", lambda row: "%sx%sx%s" % (
            row["tile_m"], row["tile_n"], row["tile_k"]
        )),
        ("stage", lambda row: row["stages"]),
        ("latency", _latency_bucket),
    )


def _print_metric(label: str, metric: Mapping[str, float]) -> None:
    print(
        "%s n=%d MAPE=%.2f%% wMAPE=%.2f%% bias=%+.2f%% wBias=%+.2f%%"
        % (
            label,
            int(metric["n"]),
            metric["mape"],
            metric["wmape"],
            metric["bias"],
            metric["weighted_bias"],
        )
    )


def main(argv: Sequence[str] = ()) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--arch", required=True, choices=("a100", "b200", "h200"))
    parser.add_argument(
        "--mapping",
        required=True,
        choices=("cluster1", "cta1", "cta2", "ordinary_cluster"),
    )
    parser.add_argument(
        "--prediction",
        action="append",
        default=[],
        help="prediction column in microseconds including launch cost; repeatable",
    )
    parser.add_argument(
        "--body-prediction",
        action="append",
        default=[],
        help="body-only prediction column; --launch-us is added before metrics",
    )
    parser.add_argument("--launch-us", type=float, default=2.0)
    parser.add_argument("--selected-out", type=Path)
    args = parser.parse_args(tuple(argv) if argv else None)

    with args.csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        missing = sorted(set(REQUIRED) - set(fields))
        if missing:
            parser.error("CSV is missing required fields %s" % missing)
        rows = [dict(row) for row in reader]
    prediction_columns = tuple(args.prediction) + tuple(args.body_prediction)
    absent_predictions = sorted(set(prediction_columns) - set(fields))
    if absent_predictions:
        parser.error("CSV is missing prediction fields %s" % absent_predictions)
    if not math.isfinite(args.launch_us) or args.launch_us < 0.0:
        parser.error("--launch-us must be finite and non-negative")

    selected, reasons, selected_before_dedupe = select_covered_rows(
        rows,
        args.arch,
        args.mapping,
        prediction_columns,
    )
    print(
        "input_rows=%d selected_before_dedupe=%d"
        % (len(rows), selected_before_dedupe)
    )
    print("filter_reasons=%s" % dict(sorted(reasons.items())))
    if not selected:
        print(
            "no covered rows: this selector fails closed instead of remapping "
            "cluster/tensor-core semantics",
            file=sys.stderr,
        )
        return 2

    if args.arch == "b200" and args.mapping == "cta2":
        print("selected_after_median_dedupe=%d" % len(selected))

    for field in args.body_prediction:
        adjusted = "%s__plus_launch" % field
        for row in selected:
            row[adjusted] = str(float(row[field]) + args.launch_us)
        args.prediction.append(adjusted)

    if args.selected_out is not None:
        output_fields = list(selected[0])
        args.selected_out.parent.mkdir(parents=True, exist_ok=True)
        with args.selected_out.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=output_fields)
            writer.writeheader()
            writer.writerows(selected)
        print("selected_csv=%s" % args.selected_out)

    if not args.prediction:
        print(
            "coverage audit only; add --prediction COL for error metrics. "
            "No warp tile or cluster interpretation was inferred."
        )
        return 0
    for prediction in args.prediction:
        _print_metric("overall/%s" % prediction, _metrics(selected, prediction))
        for group_name, key in _groups(selected):
            grouped = defaultdict(list)
            for row in selected:
                grouped[key(row)].append(row)
            for value, group_rows in sorted(grouped.items(), key=lambda item: str(item[0])):
                _print_metric(
                    "%s=%s/%s" % (group_name, value, prediction),
                    _metrics(group_rows, prediction),
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
