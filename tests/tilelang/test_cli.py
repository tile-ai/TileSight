"""``python -m tilesight.tilelang`` exit codes and outputs."""

import json

from conftest import EXAMPLES, needs_tilelang
from tilesight.tilelang.cli import main


@needs_tilelang
def test_refusal_exits_3_and_writes_the_reason(tmp_path):
    output = tmp_path / "report.json"
    code = main([str(EXAMPLES / "kernels" / "gemm.py") + ":build_kernel", "--target", "sm_90a", "--output", str(output)])
    assert code == 3
    payload = json.loads(output.read_text())
    assert payload["status"] == "refused" and payload["refusal"]["code"] == "ENTRY_FACTORY_ARGUMENT_MISMATCH"


@needs_tilelang
def test_complete_run_writes_the_bundle(tmp_path):
    output, bundle = tmp_path / "sum.json", tmp_path / "sum"
    code = main([str(EXAMPLES / "kernels" / "sum.py") + ":build_kernel", "--target", "sm_90a",
                 "--factory-args", '{"M": 8192, "N": 8192, "block_m": 2, "threads": 128, "dtype": "float16"}',
                 "--output", str(output), "--report-dir", str(bundle)])
    assert code == 0
    assert json.loads(output.read_text())["status"] == "complete"
    for name in ("report.json", "report.html", "trace.perfetto.json", "tables/ops.csv", "tables/diagnoses.csv"):
        assert (bundle / name).is_file(), name
    assert "<title>TileSight profile</title>" in (bundle / "report.html").read_text()
