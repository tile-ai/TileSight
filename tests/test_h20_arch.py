"""CPU-only regression coverage for the H20 architecture profile."""

from dataclasses import replace

import pytest

from tilesight.arch.h20 import H20
from tilesight.modeling import program as sight
from tilesight.modeling.program.examples import gemm_program


@pytest.fixture(params=[None, "set_to_spec", "set_to_microbench", "set_to_ncu"])
def arch(request):
    profile = H20()
    if request.param is not None:
        getattr(profile, request.param)()
    return profile


def test_h20_capacity_and_capabilities(arch):
    assert arch.ddr_capacity == 96 * (1024**3)
    assert arch.support_wgmma is True
    assert arch.support_utcmma is False
    assert arch.tmem_capacity_per_sm == 0
    assert arch.tmem_bandwidth == 0


def test_h20_accepts_wgmma_requirement(arch):
    program = gemm_program(m=256, n=256, k=128, dtype="fp16")
    required = replace(
        program,
        launches=(replace(program.launches[0], requires_arch=("wgmma",)),),
    )
    options = sight.Options(cache="fast", ii_mode="periodic_best")
    baseline = sight.analyze(program, arch, options=options)
    result = sight.analyze(required, arch, options=options)

    assert not result.diagnostics.unsupported
    assert result.launch("main").kernel_body_s > 0
    assert result.launch("main").kernel_body_s == pytest.approx(
        baseline.launch("main").kernel_body_s
    )


@pytest.mark.parametrize("capability", ["tcgen05", "tmem"])
def test_h20_rejects_blackwell_requirements(arch, capability):
    program = gemm_program(m=256, n=256, k=128, dtype="fp16")
    required = replace(
        program,
        launches=(replace(program.launches[0], requires_arch=(capability,)),),
    )

    with pytest.raises(sight.UnsupportedError, match=f"requires {capability}"):
        sight.analyze(required, arch, options=sight.Options(cache="fast"))
