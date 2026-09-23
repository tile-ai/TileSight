"""Tests for the typed full-kernel result and report-only analysis passes."""

from __future__ import annotations

from types import SimpleNamespace

from dataclasses import replace

import pytest

from tilesight.modeling import (
    FeasibilityStatus,
    FullModelError,
    FullModelOptions,
    TimingBreakdown,
    Kernel,
    Timing,
    Work,
    analyze_fusion,
    analyze_liveness,
    model,
)
from tilesight.modeling.errors import ModelingValidationError
from tilesight.modeling.tests.fixtures.legacy_fa4 import build_fa4


def test_fa4_static_liveness_honors_explicit_sp_alias_groups():
    report = analyze_liveness(build_fa4())

    assert report.status is FeasibilityStatus.UNKNOWN
    tmem = next(
        peak
        for peak in report.peaks
        if peak.storage_scope == "tmem" and peak.execution_scope == "cta"
    )
    assert tmem.lower_bytes <= tmem.selected_bytes < tmem.upper_bytes
    assert any("report-only" in item for item in report.diagnostics)


def test_absent_fusion_plan_is_typed_not_requested():
    report = analyze_fusion(build_fa4())

    assert report.status is FeasibilityStatus.NOT_REQUESTED
    assert report.eliminated_global_bytes == 0.0
    assert any("not requested" in item for item in report.diagnostics)


def test_liveness_reports_a_declared_smem_capacity_violation_without_guarding():
    kernel = Kernel("oversized")
    with kernel.launch(work_grid=(1,), threads=32) as launch:
        with launch.periodic("loop", iterations=1) as loop:
            storage = loop.buffer(
                "storage",
                scope="smem",
                shape=(4096,),
                dtype="fp32",
            )
            with loop.actor("actor") as actor:
                actor.phase(
                    "write",
                    work=Work.copy(4096 * 4),
                    timing=Timing(1.0),
                    writes=(storage,),
                )
    arch = SimpleNamespace(configurable_smem_capacity=1024, tmem_capacity_per_sm=0)
    report = analyze_liveness(kernel.build(), arch=arch, resident_ctas_per_sm=1)

    assert report.status is FeasibilityStatus.PROVEN_INFEASIBLE
    assert any("exceeds" in item for item in report.diagnostics)


def test_liveness_takes_max_instead_of_sum_across_sequential_launches():
    kernel = Kernel("two_launches")
    for index in range(2):
        with kernel.launch(
            name="launch%d" % index,
            work_grid=(1,),
            threads=32,
        ) as launch:
            with launch.periodic("loop%d" % index, iterations=1) as loop:
                storage = loop.buffer(
                    "storage",
                    scope="smem",
                    shape=(1024,),
                    dtype="fp32",
                )
                with loop.actor("actor") as actor:
                    actor.phase(
                        "write",
                        work=Work.copy(4096),
                        timing=Timing(1.0),
                        writes=(storage,),
                    )

    report = analyze_liveness(kernel.build())
    peak = next(item for item in report.peaks if item.storage_scope == "smem")
    assert peak.selected_bytes == 4096
    assert peak.upper_bytes == 4096


def test_full_model_timing_requires_one_launch_and_dispatch_accounting():
    with pytest.raises(ModelingValidationError, match="total_s must equal"):
        TimingBreakdown(
            kernel_body_s=10.0,
            launch_s=2.0,
            host_dispatch_s=3.0,
            total_s=14.0,
        )


def test_frontend_without_complete_adapter_fails_actionably():
    from tilesight.arch.h200_sxm import H200_SXM

    arch = H200_SXM().set_to_microbench()
    ir = build_fa4()
    stripped = replace(ir, params=tuple(item for item in ir.params if item[0] != "model_adapter"))
    with pytest.raises(FullModelError, match="model_adapter"):
        model(stripped, arch, FullModelOptions())
