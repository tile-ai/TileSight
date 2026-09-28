"""NCU contract: CSV import, units, request and observation validation (no GPU, no TileLang)."""

import copy

import pytest

from tilesight.tilelang import ncu

REQUEST = {"schema_version": ncu.REQUEST_SCHEMA, "target": {"symbol": "main_kernel", "name_base": "function",
                                                            "launch_skip": 0},
           "device": {"uuid": "GPU-00000000-0000-0000-0000-000000000000"},
           "capture": {"cache_control": "none", "clock_control": "none", "replay_mode": "application", "warmup": 5,
                       "profile_from_start": "off"},
           "command": ["python", "driver.py"]}


def _long_csv(rows):
    lines = ['"ID","Process ID","Kernel Name","Context","Stream","Device","Metric Name","Metric Unit","Metric Value"']
    lines += ['"0","1","main_kernel","1","7","0","%s","%s","%s"' % row for row in rows]
    return "\n".join(lines) + "\n"


def test_long_csv_import_normalizes_units(tmp_path):
    path = tmp_path / "capture.csv"
    path.write_text(_long_csv([("gpu__time_duration.sum", "usecond", "12.5"),
                               ("dram__bytes_read.sum", "Mbyte", "100.25"),
                               ("lts__t_sectors_op_read.sum", "sector", "1,000,000"),
                               ("sm__cycles_elapsed.avg.per_second", "cycle/nsecond", "1.5"),
                               ("launch__grid_dim_x", "", "1024")]))
    observation = ncu.import_csv(path, REQUEST)
    assert observation["status"] == "partial"           # most registered metrics are absent here
    assert ncu.metric(observation, "duration_s") == pytest.approx(12.5e-6)
    assert ncu.metric(observation, "dram_read_bytes") == pytest.approx(100.25e6)
    assert ncu.metric(observation, "l2_read_sectors") == 1000000.0
    assert ncu.metric(observation, "sm_frequency_hz") == pytest.approx(1.5e9)
    assert ncu.metric(observation, "grid_x") == 1024.0
    assert ncu.metric(observation, "dram_write_bytes") is None   # missing is never zero
    ncu.validate_observation(observation)


def test_wide_csv_and_unit_mismatch(tmp_path):
    path = tmp_path / "wide.csv"
    path.write_text('"ID","Process ID","Kernel Name","Context","Stream","Device","gpu__time_duration.sum","dram__bytes_read.sum"\n'
                    '"","","","","","","msecond","cycle"\n'
                    '"0","1","main_kernel","1","7","0","0.5","10"\n')
    observation = ncu.import_csv(path, REQUEST)
    assert ncu.metric(observation, "duration_s") == pytest.approx(0.5e-3)
    assert observation["metrics"]["dram_read_bytes"]["status"] == "invalid"


def test_csv_rejects_other_kernel_and_two_launches(tmp_path):
    path = tmp_path / "capture.csv"
    path.write_text(_long_csv([("gpu__time_duration.sum", "second", "1")]).replace("main_kernel", "other_kernel"))
    with pytest.raises(ValueError, match="differs from the requested"):
        ncu.import_csv(path, REQUEST)
    text = _long_csv([("gpu__time_duration.sum", "second", "1")])
    path.write_text(text + text.splitlines()[1].replace('"0","1"', '"1","1"', 1) + "\n")
    with pytest.raises(ValueError, match="one kernel launch"):
        ncu.import_csv(path, REQUEST)


@pytest.mark.parametrize("change", [
    lambda r: r["target"].update(symbol="regex:main.*"),
    lambda r: r["device"].update(uuid="0"),
    lambda r: r["capture"].update(replay_mode="range"),
    lambda r: r.update(command="python driver.py"),
    lambda r: r.update(metric_keys=["no_such_metric"]),
])
def test_request_validation(change):
    request = copy.deepcopy(REQUEST)
    change(request)
    with pytest.raises(ValueError):
        ncu.validate_request(request)


def test_observation_is_tamper_evident(tmp_path):
    path = tmp_path / "capture.csv"
    path.write_text(_long_csv([("gpu__time_duration.sum", "usecond", "12.5")]))
    observation = ncu.import_csv(path, REQUEST)
    ncu.validate_observation(observation)
    observation["request"]["target"]["launch_skip"] = 1
    with pytest.raises(ValueError, match="digest"):
        ncu.validate_observation(observation)
