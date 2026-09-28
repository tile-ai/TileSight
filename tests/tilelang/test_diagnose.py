"""Diagnosis arithmetic that does not need an analysis."""

from types import SimpleNamespace

from tilesight.tilelang import diagnose


def test_fold_and_union():
    assert diagnose._fold(9.0, 3.0, 10.0) == [(9.0, 10.0), (0.0, 2.0)]
    assert diagnose._fold(0.0, 12.0, 10.0) == [(0.0, 10.0)]
    assert diagnose._union([(0.0, 2.0), (1.0, 3.0), (5.0, 6.0)]) == [(0.0, 3.0), (5.0, 6.0)]


def test_tail_wave_idle_fraction():
    launch = SimpleNamespace(sm_count=132, work_units=4096)
    small_tail = diagnose._sm_balance(launch, 1)          # 32 waves, last one 4 CTAs: 3% idle
    assert small_tail["status"] == "not_detected"
    assert small_tail["values"]["waves"] == 32 and small_tail["values"]["tail_work_units"] == 4
    big_tail = diagnose._sm_balance(launch, 6)            # 6 waves of 792 slots: 14% idle
    assert big_tail["status"] == "detected"


def test_deviation_without_measurement_is_not_evaluated():
    assert diagnose._deviation(None, 1e-3)["status"] == "not_evaluated"
