"""Reproducibility checks for the B200 CTA2 protocol audit tool."""

from __future__ import annotations

import pytest

from tilesight.modeling.tools.audit_b200_cta2_native import (
    AUDIT_VERSION,
    audit_rows,
)


def _skinny_m_row():
    return {
        "M": "16",
        "N": "1024",
        "K": "1024",
        "precision": "fp16",
        "measured_us": "20",
        "runtime_ms": "0.020",
        "kernel_name": "cutlass_fixture_128x256x64_1sm",
        "tile_m": "128",
        "tile_n": "256",
        "tile_k": "64",
        "stages": "8",
        "tc_path": "utcmma2",
        "cluster_m": "2",
        "cluster_n": "1",
        "cluster_k": "1",
        "raster_order": "along_m",
        "swizzle_size": "1",
        "scheduler_policy": "",
        "source_tag": "fixture",
    }


def test_skinny_m_keeps_source_cluster_and_marks_legacy_axis_mismatch():
    audited, timing, metadata = audit_rows([_skinny_m_row()])
    assert len(audited) == 1
    row = audited[0]
    assert row["modeled_cluster_axis"] == "M"
    assert row["legacy_cost_cluster_axis"] == "N"
    assert row["legacy_axis_mismatch"] == "true"
    assert row["modeled_logical_work_units"] == "4"
    assert row["modeled_physical_ctas"] == "8"
    assert float(row["modeled_cross_sm_payload_bytes_per_k_iter"]) == 0.0
    assert float(row["native_mainloop_warmup_ns"]) > 0.0
    assert float(row["native_mainloop_drain_ns"]) > 0.0
    assert row["native_boundary_strategy"] == "periodic_macro"
    assert row["kernel_name_cluster_evidence"] == (
        "contradictory_1sm_suffix_with_utcmma2_cluster2"
    )
    assert row["native_protocol_audit_version"] == AUDIT_VERSION
    assert float(row["legacy_us"]) - float(row["legacy_body_us"]) == pytest.approx(2.0)
    assert float(row["native_protocol_us"]) - float(
        row["native_protocol_body_us"]
    ) == pytest.approx(2.0)
    assert float(row["prediction_tma_fixed_latency_ns"]) == 0.0
    assert row["prediction_boundary_policy"] == "periodic_witness"
    assert metadata["legacy_axis_mismatch_rows"] == 1
    assert metadata["contradictory_1sm_suffix_rows"] == 1
    assert timing.unique_cache_keys == 1


def test_audit_exposes_tma_first_ready_latency_as_explicit_sensitivity():
    baseline, _, _ = audit_rows([_skinny_m_row()])
    delayed, _, _ = audit_rows(
        [_skinny_m_row()],
        tma_fixed_latency_ns=100.0,
    )
    assert float(delayed[0]["prediction_tma_fixed_latency_ns"]) == 100.0
    assert float(delayed[0]["native_protocol_body_us"]) > float(
        baseline[0]["native_protocol_body_us"]
    )


def test_audit_can_opt_into_exact_finite_boundary():
    audited, _, _ = audit_rows(
        [_skinny_m_row()],
        boundary_policy="finite_witness",
    )
    assert audited[0]["prediction_boundary_policy"] == "finite_witness"
    assert audited[0]["native_boundary_strategy"] == "periodic_finite_witness"
