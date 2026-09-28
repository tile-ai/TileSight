"""Every failure to model is a refusal with a code, never a number."""

import pytest

from conftest import EXAMPLES, example_request, needs_tilelang
from tilesight.tilelang import Refusal, profile_request
from tilesight.tilelang.capture import capture
from tilesight.tilelang.entry import KernelRequest, resolve
from tilesight.tilelang.translate import translate

pytestmark = needs_tilelang


def test_factory_argument_mismatch_is_refused_before_lowering():
    profile = profile_request(example_request("gemm", factory_args={}))
    assert profile.status == "refused" and profile.refusal["code"] == "ENTRY_FACTORY_ARGUMENT_MISMATCH"


def test_missing_kernel_file():
    with pytest.raises(Refusal) as error:
        resolve(KernelRequest(str(EXAMPLES / "kernels" / "missing.py"), "build_kernel"))
    assert error.value.code == "ENTRY_FILE_NOT_FOUND"


def test_unbound_shape_symbols_are_named():
    resolved = resolve(example_request("gqa"))
    planned = capture(resolved.prim_func, target_arch="sm_90a", pass_configs=dict(resolved.pass_configs))
    with pytest.raises(Refusal) as error:
        translate(planned, {})
    assert error.value.code == "UNBOUND_SYMBOL" and "seq_len_q" in error.value.message


def test_a_kernel_that_does_not_lower_is_refused(tmp_path):
    source = tmp_path / "broken.py"
    source.write_text(
        "import tilelang.language as T\n"
        "def build_kernel():\n"
        "    @T.prim_func\n"
        "    def main(A: T.Tensor((128, 128), 'float16'), B: T.Tensor((128, 128), 'float16')):\n"
        "        with T.Kernel(1, threads=128) as bx:\n"
        "            S = T.alloc_shared((64, 64), 'float16')\n"
        "            T.copy(A[0:128, 0:128], S)\n"
        "            T.copy(S, B[0:128, 0:128])\n"
        "    return main\n")
    profile = profile_request(KernelRequest(str(source), "build_kernel"))
    assert profile.status == "refused"
    assert profile.refusal["code"] in ("LOWERING_FAILED", "INDEX_UNSUPPORTED", "BUFFER_VIEW_CONFLICT")
