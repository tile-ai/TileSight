import pytest

from tilesight.modeling._pipeline.overlap_analysis import (
    HardwareUsage,
    LoopNode,
    OpGroup,
    model_overlap,
    overlap_analysis,
    overlap_analysis_full,
    simulate_schedule,
)


def test_stage_two_default_is_resource_ii_not_single_iteration_makespan():
    load = OpGroup("load", HardwareUsage(ddr_time=10.0))
    mma = OpGroup(
        "mma", HardwareUsage(tensor_time=10.0), depends_on=[load]
    )

    ii, _ = simulate_schedule([load, mma], stage=2, order=[0, 1])

    assert ii == pytest.approx(10.0)


def test_simulate_schedule_rejects_non_topological_order():
    load = OpGroup("load", HardwareUsage(ddr_time=1.0))
    mma = OpGroup(
        "mma", HardwareUsage(tensor_time=1.0), depends_on=[load]
    )

    with pytest.raises(ValueError, match="illegal topological order"):
        simulate_schedule([load, mma], stage=2, order=[1, 0])


def test_cycle_fails_loudly_instead_of_conflicting_with_order_validator():
    first = OpGroup("first", HardwareUsage(cuda_time=1.0))
    second = OpGroup(
        "second", HardwareUsage(cuda_time=1.0), depends_on=[first]
    )
    first.depends_on = [second]

    with pytest.raises(ValueError, match="no legal topological order"):
        model_overlap([first, second], stage=2, try_all_orders=True)


def test_legacy_recursive_entry_point_emits_deprecation_warning():
    loop = LoopNode(
        "legacy",
        [OpGroup("phase", HardwareUsage(cuda_time=1.0))],
        num_iters=1,
    )

    with pytest.warns(DeprecationWarning, match="deprecated LoopNode-based"):
        latency, _ = overlap_analysis(loop)

    assert latency == pytest.approx(1.0)


def test_legacy_full_entry_point_emits_only_one_deprecation_warning():
    loop = LoopNode(
        "legacy",
        [OpGroup("phase", HardwareUsage(cuda_time=1.0))],
        num_iters=1,
    )
    arch = type("Arch", (), {"sm_count": 1})()

    with pytest.warns(DeprecationWarning, match="overlap_analysis_full") as records:
        result = overlap_analysis_full(loop, grids=(1,), arch=arch)

    assert len(records) == 1
    assert result[0] == pytest.approx(1.0)
