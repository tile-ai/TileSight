"""GEMM ``C = A @ B`` (TileLang ``examples/gemm/example_gemm.py``).

    python -m tilesight.modeling.program.examples.gemm
"""

from __future__ import annotations

from typing import Any

from ..builder import KernelBuilder
from ..contract import Program
from ._common import TILELANG_COMMIT, _ceil, report

# ---------------------------------------------------------------------------
# GEMM  (reference: tilelang/examples/gemm/example_gemm.py)
# ---------------------------------------------------------------------------


def gemm_program(
    m: int = 4096, n: int = 4096, k: int = 4096,
    block_m: int = 128, block_n: int = 128, block_k: int = 64,
    stages: int = 3, dtype: str = "fp16", accumulation_dtype: str = "fp32",
    mapper: Any = "linear_block_id",
) -> Program:
    """``C = A @ B``: TMA loads into a ``stages``-deep SMEM ring, tensor-core MMA, register epilogue.

    ``mapper`` is the block mapper that decides the launch order of the (m, n)
    tiles (``linear_block_id`` by default; e.g. ``PanelSwizzle`` for a CUTLASS
    style swizzle).  It changes cache reuse, not the amount of work.

    Tail tiles along M/N become work groups and the K tail an averaged
    per-execution fraction; both are derived by the builder from the tensor
    and tile shapes.
    """

    tiles_m, tiles_n, tiles_k = _ceil(m, block_m), _ceil(n, block_n), _ceil(k, block_k)
    kb = KernelBuilder("main", grid={"n": tiles_n, "m": tiles_m}, threads=256, dispatch_order=mapper,
                       metadata={"kernel": "gemm", "reference": "tilelang/examples/gemm/example_gemm.py",
                                 "tilelang_commit": TILELANG_COMMIT})
    A = kb.tensor("A", (m, k), dtype)
    B = kb.tensor("B", (k, n), dtype)
    C = kb.tensor("C", (m, n), dtype)
    A_s = kb.shared("A_s", (block_m, block_k), dtype, stages=stages)
    B_s = kb.shared("B_s", (block_k, block_n), dtype, stages=stages)
    acc = kb.fragment("acc", (block_m, block_n), accumulation_dtype, carried=True)
    C_r = kb.fragment("C_r", (block_m, block_n), dtype)
    with kb.loop("k", tiles_k, stages=stages) as loop:
        kb.copy(A["m", "k"], A_s, name="load_A")
        kb.copy(B["k", "n"], B_s, name="load_B")
        kb.gemm(A_s, B_s, acc, name="mma", compute_dtype=dtype, accumulation_dtype=accumulation_dtype)
        with loop.epilogue():
            kb.cast(acc, C_r, name="cast_C", actor="epilogue")
            kb.copy(C_r, C["m", "n"], name="store_C", engine="ldst")
    return kb.program(name="gemm")


def main() -> None:
    import tilesight as sight
    from tilesight.arch.h200_sxm import H200_SXM

    program = gemm_program(m=4096, n=4096, k=4096, block_m=128, block_n=128, block_k=64, stages=3, dtype="bf16")
    result = sight.analyze(program, H200_SXM().set_to_microbench(), options=sight.Options(cache="fast", ii_mode="periodic_best"))
    report(result, loop="main/k", op="main/k/load_B")


if __name__ == "__main__":
    main()
