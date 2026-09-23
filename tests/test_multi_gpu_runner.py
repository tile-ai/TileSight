"""GPU-independent behavior checks; external benchmark output below is a parser fixture."""
import json
import subprocess
from types import SimpleNamespace

import pytest

from micro_benchmark import multi_gpu as runner

UUIDS = ["GPU-12345678-1234-5678-90ab-1234567890ab", "GPU-abcdef12-1234-5678-90ab-1234567890ab"]


def options(**overrides):
    return SimpleNamespace(**{**dict(devices=[3, 1], min_bytes=8, max_bytes=None, iters=None,
                                     quick=True, nvshmem_ctas=32), **overrides})


@pytest.mark.parametrize("raw", ["", "1", "0,0", "-1,0", "1,x", "0,1,"])
def test_device_selection_rejects_invalid_or_ambiguous_input(raw):
    with pytest.raises(runner.argparse.ArgumentTypeError):
        runner.devices(raw)


def test_device_order_and_inherited_visibility_are_preserved():
    assert runner.devices("3,1") == [3, 1]
    original = {"CUDA_VISIBLE_DEVICES": "7,5,3,1", "USER_OPTION": "unchanged"}
    infos = [{"uuid": uuid, "name": "NVIDIA H100"} for uuid in UUIDS]
    pinned = runner.pin_devices(infos, original)
    assert pinned == {**original, "CUDA_VISIBLE_DEVICES": ",".join(UUIDS)}
    assert original["CUDA_VISIBLE_DEVICES"] == "7,5,3,1"
    with pytest.raises(ValueError, match="distinct"):
        runner.pin_devices([infos[0], infos[0]], original)


def test_resolution_uses_original_ordinals_then_pins_selected_order(monkeypatch, tmp_path):
    commands = []
    monkeypatch.setattr(runner, "executable", lambda *a: "/nvcc")
    monkeypatch.setattr(runner, "identity", lambda p: {"path": p})
    def execute(command, *args, **kwargs):
        commands.append(command)
        if "--device" in command:
            ordinal = int(command[-1])
            return json.dumps({"uuid": UUIDS[[3,1].index(ordinal)], "name": "NVIDIA H100", "visible_device": ordinal})
        return "CUDA version fixture"
    monkeypatch.setattr(runner, "execute", execute)
    infos, env, _ = runner.resolve_devices(SimpleNamespace(nvcc=None, timeout=1, devices=[3,1]), tmp_path, {})
    assert [c[-1] for c in commands if "--device" in c] == ["3", "1"]
    assert [x["visible_device"] for x in infos] == [3,1]
    assert env["CUDA_VISIBLE_DEVICES"] == ",".join(UUIDS)


def test_missing_explicit_tool_never_falls_back(monkeypatch):
    seen = []
    monkeypatch.setattr(runner.shutil, "which", lambda p: seen.append(p))
    with pytest.raises(runner.Unavailable):
        runner.executable("/missing/tool", "TOOL", "fallback")
    assert seen == ["/missing/tool"]


def test_p2p_stdout_with_stderr_preface_preserves_directions_and_unsupported():
    output = "Visible device 0: fixture\n" + ",".join(runner.P2P_FIELDS) + "\n" + (
        "0,1,8,1,0,measured,2,0.004\n1,0,8,0,0,unsupported,,\n")
    rows = runner.parse_p2p(output, 2)
    assert rows[0]["time_us"] == 2
    assert rows[1]["status"] == "unsupported" and "time_us" not in rows[1]
    assert runner.p2p_csv(output).startswith("src,dst,")
    with pytest.raises(ValueError, match="missing requested"):
        runner.parse_p2p(output.rsplit("1,0",1)[0],2)
    with pytest.raises(ValueError, match="must not contain"):
        runner.parse_p2p(output.replace("unsupported,,", "unsupported,0,0"),2)


NCCL = """# size count type redop root time algbw busbw #wrong time algbw busbw #wrong
# (B) (elements) (us) (GB/s) (GB/s) (us) (GB/s) (GB/s)
8 2 float sum -1 10.25 0.00 0.00 0 9.25 0.00 0.00 0
# Out of bounds values : 0 OK
"""


def test_nccl_preserves_placement_and_independent_bandwidths():
    rows = runner.parse_nccl(NCCL)
    assert [r["placement"] for r in rows] == ["out_of_place", "in_place"]
    assert [r["time_us"] for r in rows] == [10.25,9.25]
    assert all(r["wrong"] == 0 for r in rows)
    # Native output rounds tiny payload bandwidth to 0.00; this is not an estimate.
    assert rows[0]["algbw_gb_s"] == 0
    assert rows[0]["busbw_gb_s"] == 0


@pytest.mark.parametrize("output", [NCCL.replace("0 OK", "1 FAILED"), NCCL.replace("0\n# Out", "1\n# Out"),
                                    NCCL.replace("(us)","(ms)"), NCCL.replace("10.25","nan"),
                                    NCCL.split("# Out")[0]])
def test_nccl_rejects_corruption_missing_correctness_or_unknown_units(output):
    with pytest.raises(ValueError):
        runner.parse_nccl(output)


def test_commands_remap_devices_and_enable_nccl_correctness():
    args = options()
    p2p = runner.p2p_command("/p2p",args)
    assert p2p[p2p.index("--devices")+1] == "0,1"
    nccl = runner.nccl_command("/all_reduce_perf",args)
    assert nccl[nccl.index("-g")+1] == "2"
    assert nccl[nccl.index("-c")+1] == "1"
    assert nccl[nccl.index("-d")+1] == "float"


def test_nvshmem_native_units_scopes_and_stats():
    legacy = "&&&& PERF shmem_put_latency___Warp___size__8___latency 0.42 -us\n"
    assert runner.parse_nvshmem(legacy,"latency")[0]["scope"] == "Warp"
    stats = "&&&& PERF_STATS shmem_put_bw_uni___None___size__256___BW +GB/sec mean=1.5 stddev=NA min=1.5 max=1.5 repetitions=1\n"
    row = runner.parse_nvshmem(stats,"bandwidth")[0]
    assert row["unit"] == "GB/sec" and row["value"] == 1.5
    assert row["statistics"]["repetitions"] == "1"
    with pytest.raises(ValueError,match="unit"):
        runner.parse_nvshmem(legacy.replace("-us","-ms"),"latency")
    with pytest.raises(ValueError,match="No recognized"):
        runner.parse_nvshmem("No GPU measurements", "latency")


def test_nvshmem_validates_cli_and_uses_two_local_pes():
    help_text = "-b -e -f -n -w -c -t --scope --use_smem"
    cmd = runner.nvshmem_command("/nvshmrun","/shmem_put_bw",options(),"bandwidth",help_text)
    assert cmd[:8] == ["/nvshmrun","-n","2","-ppn","2","-hosts","localhost","/shmem_put_bw"]
    assert cmd[cmd.index("-b")+1] == "256"
    assert cmd[-2:] == ["--use_smem","0"]
    with pytest.raises(runner.Unavailable,match="CLI"):
        runner.nvshmem_command("/nvshmrun","/tool",options(),"bandwidth","unknown version")


@pytest.mark.parametrize("statuses,expected", [
    (["unavailable","unsupported"], ("unavailable",1)), (["measured","unavailable"],("partial",0)),
    (["measured","failed"],("failed",1)), (["measured"],("complete",0)), (["partial"],("partial",0)),
])
def test_overall_status_requires_real_measurement(statuses, expected):
    assert runner.overall_status([{"status":s} for s in statuses]) == expected


def test_missing_explicit_backend_writes_unavailable_without_querying_gpu(monkeypatch,tmp_path):
    monkeypatch.setattr(runner, "backend_tools", lambda *a: (_ for _ in ()).throw(runner.Unavailable("not installed")))
    monkeypatch.setattr(runner, "resolve_devices", lambda *a: pytest.fail("GPU query for unavailable backend"))
    out=tmp_path/"missing"
    assert runner.main(["--devices","0,1","--backend","nccl","--output",str(out)]) == 1
    report=json.loads((out/"results.json").read_text())
    assert report["status"] == "unavailable"
    assert report["results"][0]["status"] == "unavailable"


def test_dry_run_queries_nothing_and_creates_no_output(monkeypatch,tmp_path,capsys):
    monkeypatch.setattr(runner, "execute", lambda *a: pytest.fail("dry-run executed a process"))
    out=tmp_path/"unused"
    assert runner.main(["--devices","3,1","--backend","all","--dry-run","--output",str(out)]) == 0
    assert not out.exists()
    assert "--devices 0,1" in capsys.readouterr().out


def test_failed_process_keeps_output_and_cannot_pass(tmp_path):
    log=tmp_path/"failed.txt"
    with pytest.raises(RuntimeError,match="Exit 7"):
        runner.execute([runner.sys.executable,"-c","print('fixture failure');raise SystemExit(7)"],log,5)
    assert "fixture failure" in log.read_text()


def test_all_unavailable_does_not_report_success_or_query_gpus(monkeypatch,tmp_path):
    monkeypatch.setattr(runner,"backend_tools",lambda *a: (_ for _ in ()).throw(runner.Unavailable("not installed")))
    monkeypatch.setattr(runner,"resolve_devices",lambda *a: pytest.fail("GPU query for unavailable backends"))
    out=tmp_path/"missing_all"
    assert runner.main(["--devices","0,1","--backend","all","--output",str(out)]) == 1
    report=json.loads((out/"results.json").read_text())
    assert report["status"] == "unavailable"
    assert all(row["status"] == "unavailable" for row in report["results"])


def test_execution_failure_recorded_in_final_report(monkeypatch,tmp_path):
    monkeypatch.setattr(runner,"backend_tools",lambda *a: {})
    monkeypatch.setattr(runner,"resolve_devices",lambda *a: ([],{},"/nvcc"))
    monkeypatch.setattr(runner.shutil,"which",lambda *a: None)
    monkeypatch.setattr(runner,"run_backend",lambda *a: (_ for _ in ()).throw(RuntimeError("fixture process failed")))
    out=tmp_path/"failed"
    assert runner.main(["--devices","0,1","--output",str(out)]) == 1
    report=json.loads((out/"results.json").read_text())
    assert report["status"] == "failed"
    assert report["results"][0]["reason"] == "fixture process failed"
