"""Reductions: reduce-sum and RMSNorm (TileLang ``examples/norm/rms_norm.py``).

    python -m tilesight.modeling.program.examples.reductions
"""

from __future__ import annotations


from ..builder import KernelBuilder, tiles
from ..contract import Program
from ._common import TILELANG_COMMIT, _ceil, report

# ---------------------------------------------------------------------------
# Reductions  (reference: tilelang/examples/norm/rms_norm.py)
# ---------------------------------------------------------------------------


def reduce_sum_program(
    rows: int = 16384, hidden: int = 4096, block_rows: int = 32, block_k: int = 512, dtype: str = "fp16",
) -> Program:
    """Pure row-sum reduction: a pipelined loop over the hidden axis and one store."""

    k_tiles = _ceil(hidden, block_k)
    kb = KernelBuilder("main", grid={"row": _ceil(rows, block_rows)}, threads=128,
                       metadata={"kernel": "reduce_sum", "tilelang_commit": TILELANG_COMMIT})
    X = kb.tensor("X", (rows, hidden), dtype)
    S = kb.tensor("sum", (rows, 1), "fp32")
    X_s = kb.shared("X_s", (block_rows, block_k), dtype, stages=2)
    acc = kb.fragment("acc", (block_rows, 1), "fp32", carried=True)
    with kb.loop("k", k_tiles, stages=2) as loop:
        kb.copy(X["row", "k"], X_s, name="load_X", engine="cp_async")
        kb.reduce_sum(X_s, acc, axis=1, name="reduce")
        with loop.epilogue():
            kb.copy(acc, S[tiles("row"), :], name="store_sum")
    return kb.program(name="reduce_sum")


def rms_norm_program(
    rows: int = 16384, hidden: int = 4096, block_rows: int = 32, block_k: int = 512, dtype: str = "fp16",
) -> Program:
    """RMSNorm: loop 1 accumulates sum of squares, loop 2 re-reads X and writes Y."""

    k_tiles = _ceil(hidden, block_k)
    kb = KernelBuilder("main", grid={"row": _ceil(rows, block_rows)}, threads=128,
                       metadata={"kernel": "rms_norm", "reference": "tilelang/examples/norm/rms_norm.py",
                                 "tilelang_commit": TILELANG_COMMIT})
    X = kb.tensor("X", (rows, hidden), dtype)
    Y = kb.tensor("Y", (rows, hidden), dtype)
    X_s = kb.shared("X_s", (block_rows, block_k), dtype, stages=2)
    X2_s = kb.shared("X2_s", (block_rows, block_k), dtype, stages=2)
    Y_s = kb.shared("Y_s", (block_rows, block_k), dtype, stages=2)
    sum_sq = kb.fragment("sum_sq", (block_rows, 1), "fp32", carried=True)
    scale = kb.fragment("scale", (block_rows, 1), "fp32")
    with kb.loop("k", k_tiles, stages=2) as loop:
        kb.copy(X["row", "k"], X_s, name="load_X_pass1", engine="cp_async")
        kb.reduce("sum_squares", X_s, sum_sq, axis=1, ops=(("mul", 1.0, "cuda"), ("add", 1.0, "cuda")))
        with loop.epilogue():
            kb.unary("rsqrt", sum_sq, scale, name="rsqrt")
    with kb.loop("k2", k_tiles, stages=2):
        kb.copy(X["row", "k2"], X2_s, name="load_X_pass2", engine="cp_async")
        kb.mul(X2_s, scale, Y_s, name="normalize")
        kb.copy(Y_s, Y["row", "k2"], name="store_Y", actor="storer")
    return kb.program(name="rms_norm")


def main() -> None:
    import tilesight as sight
    from tilesight.arch.h200_sxm import H200_SXM

    arch = H200_SXM().set_to_microbench()
    for program in (reduce_sum_program(), rms_norm_program()):
        print(program.name)
        report(sight.analyze(program, arch, options=sight.Options(cache="fast")))


if __name__ == "__main__":
    main()
