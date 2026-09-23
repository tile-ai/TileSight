"""Small reproducibility fixtures for the CUTLASS GEMM GT tools."""

from __future__ import annotations

import csv

import pytest

from tilesight.modeling.tools.compare_gemm_cutlass import (
    REQUIRED,
    main as compare_main,
)
from tilesight.modeling.tools.generate_gemm_cutlass_predictions import (
    GENERAL_MODEL_VERSION,
    GENERAL_UNSUPPORTED_CTA2,
    GenerationConfig,
    write_predictions,
)


def _row(**updates):
    result = {
        "M": "1024",
        "N": "4096",
        "K": "4096",
        "precision": "fp16",
        "measured_us": "267.571",
        "runtime_ms": "0.267571",
        "kernel_name": "cutlass_tensorop_h16816gemm_256x128_32x3_nt_align8",
        "tile_m": "256",
        "tile_n": "128",
        "tile_k": "32",
        "stages": "3",
        "tc_path": "h16816",
        "cluster_m": "1",
        "cluster_n": "1",
        "cluster_k": "1",
        "raster_order": "along_m",
        "swizzle_size": "1",
        "scheduler_policy": "",
        "source_tag": "tile_sweep_keep_all",
    }
    result.update({key: str(value) for key, value in updates.items()})
    return result


def _write(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED)
        writer.writeheader()
        writer.writerows(rows)


def _read(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_a100_fixture_is_deterministic_and_launch_is_added_once(tmp_path):
    source = tmp_path / "a100.csv"
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    _write(source, [_row()])
    config = GenerationConfig(
        "a100",
        "cluster1",
        seed=17,
        sample_budget=4,
        reuse_unit_bytes=8192,
        launch_us=2.0,
    )

    summary = write_predictions(source, first, config)
    write_predictions(source, second, config)
    assert summary.selected_before_dedupe == 1
    assert summary.output_rows == 1
    assert first.read_bytes() == second.read_bytes()

    row = _read(first)[0]
    assert float(row["legacy_us"]) - float(row["legacy_body_us"]) == pytest.approx(2.0)
    assert (
        float(row["distinct_legacy_outer_us"])
        - float(row["distinct_legacy_outer_body_us"])
    ) == pytest.approx(2.0)
    assert (
        float(row["general_shadow_v1_us"])
        - float(row["general_shadow_v1_body_us"])
    ) == pytest.approx(2.0)
    assert row["general_prediction_status"] == GENERAL_MODEL_VERSION
    assert row["prediction_seed"] == "17"
    assert row["prediction_sample_budget"] == "4"
    assert row["prediction_reuse_unit_bytes"] == "8192"

    assert compare_main(
        (
            "--csv",
            str(first),
            "--arch",
            "a100",
            "--mapping",
            "cluster1",
            "--prediction",
            "legacy_us",
            "--prediction",
            "distinct_legacy_outer_us",
            "--prediction",
            "general_shadow_v1_us",
        )
    ) == 0


def test_cta2_fixture_dedupes_timer_pair_and_marks_general_unsupported(tmp_path):
    source = tmp_path / "b200_cta2.csv"
    output = tmp_path / "predictions.csv"
    common = {
        "N": "1024",
        "K": "1024",
        "kernel_name": "cutlass_utcmma2_fixture",
        "tile_m": "128",
        "tile_n": "256",
        "tile_k": "64",
        "tc_path": "utcmma2",
        "cluster_m": "2",
        "cluster_n": "1",
        "source_tag": "fixture",
    }
    _write(
        source,
        [
            _row(measured_us="10", runtime_ms="0.010", **common),
            _row(measured_us="14", runtime_ms="0.014", **common),
        ],
    )
    summary = write_predictions(
        source,
        output,
        GenerationConfig("b200", "cta2", sample_budget=4),
    )

    assert summary.selected_before_dedupe == 2
    assert summary.output_rows == 1
    row = _read(output)[0]
    assert float(row["measured_us"]) == pytest.approx(12.0)
    assert float(row["runtime_ms"]) == pytest.approx(0.012)
    assert row["general_shadow_v1_us"] == ""
    assert row["general_prediction_status"] == GENERAL_UNSUPPORTED_CTA2
    assert float(row["legacy_us"]) - float(row["legacy_body_us"]) == pytest.approx(2.0)

    # A generated CTA2 CSV remains valid input to the maintained selector;
    # deduplication must not leave measured_us/runtime_ms inconsistent.
    assert compare_main(
        (
            "--csv",
            str(output),
            "--arch",
            "b200",
            "--mapping",
            "cta2",
            "--prediction",
            "legacy_us",
            "--prediction",
            "distinct_legacy_outer_us",
        )
    ) == 0
