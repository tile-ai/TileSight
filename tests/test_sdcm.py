"""Stochastic-distance cache model: CDF polynomial and binomial agreement."""
import math

import numpy as np
import pytest
from scipy.stats import binom, norm

from tilesight.util.sdcm import norm_cdf_approx_3, norm_cdf_approx_5, sdcm


@pytest.mark.parametrize("cdf", [norm_cdf_approx_3, norm_cdf_approx_5])
def test_cdf_polynomial_matches_scipy(cdf):
    xs = np.linspace(-6.0, 6.0, 4801)
    err = np.array([cdf(x) - norm.cdf(x) for x in xs])
    assert np.max(np.abs(err)) < 2e-5
    assert abs(cdf(0.0) - 0.5) < 1e-6
    # continuity and monotonicity across the |x| branch split
    vals = np.array([cdf(x) for x in xs])
    assert np.all(np.diff(vals) >= -1e-12)
    assert abs(cdf(1e-9) - cdf(-1e-9)) < 1e-6
    # symmetry Phi(-x) = 1 - Phi(x)
    assert max(abs(cdf(x) + cdf(-x) - 1.0) for x in xs[::50]) < 1e-6
    assert cdf(-8.0) < 1e-6 and cdf(8.0) > 1 - 1e-6


@pytest.mark.parametrize("A,B", [(8, 2048), (8, 16384), (16, 1024), (16, 65536)])
def test_sdcm_tracks_exact_binomial(A, B):
    # Hit iff fewer than A of the D intervening distinct blocks map to the
    # same set; each does so with probability A/B, so X ~ Bin(D, A/B) and
    # P(hit) = P(X <= A-1).  The Gaussian branch uses a continuity correction.
    Ds = np.unique(np.linspace(9, 3 * B, 300).astype(int))
    exact = binom.cdf(A - 1, Ds, A / B)
    approx = np.array([sdcm(D, A, B) for D in Ds])
    assert np.max(np.abs(approx - exact)) < 0.03
    assert np.mean(np.abs(approx - exact)) < 0.01
    # monotone non-increasing in reuse distance
    assert np.all(np.diff(approx) <= 1e-9)


def test_sdcm_boundaries():
    assert sdcm(0, 8, 1024) == 1
    assert sdcm(7, 8, 1024) == 1
    assert sdcm(1e9, 8, 1024) == 0
    assert sdcm(10, 8, 0) == 0
    # small-D branch is the exact binomial
    assert sdcm(8, 8, 64) == pytest.approx(binom.cdf(7, 8, 8 / 64), abs=1e-12)
