"""Elementwise ``C = A + B`` with tail tiles (TileLang ``examples/elementwise/example_elementwise_add.py``).

    python -m tilesight.modeling.program.examples.elementwise_add
"""

from __future__ import annotations


from ..builder import KernelBuilder
from ..contract import Program
from ._common import TILELANG_COMMIT, _ceil, report

# ---------------------------------------------------------------------------
# Elementwise  (reference: tilelang/examples/elementwise/example_elementwise_add.py)
# ---------------------------------------------------------------------------


def elementwise_add_program(
    m: int = 8192, n: int = 8192, block_m: int = 64, block_n: int = 256,
    in_dtype: str = "fp16", compute_dtype: str = "fp32", out_dtype: str = "fp16",
) -> Program:
    """``C = A + B`` with dtype conversion; tail tiles carry their useful fraction."""

    tiles_m, tiles_n = _ceil(m, block_m), _ceil(n, block_n)
    kb = KernelBuilder("main", grid={"n": tiles_n, "m": tiles_m}, threads=256,
                       metadata={"kernel": "elementwise_add", "tilelang_commit": TILELANG_COMMIT,
                                 "reference": "tilelang/examples/elementwise/example_elementwise_add.py"})
    A = kb.tensor("A", (m, n), in_dtype)
    B = kb.tensor("B", (m, n), in_dtype)
    C = kb.tensor("C", (m, n), out_dtype)
    a = kb.fragment("A_r", (block_m, block_n), in_dtype)
    b = kb.fragment("B_r", (block_m, block_n), in_dtype)
    c = kb.fragment("C_r", (block_m, block_n), out_dtype)
    kb.copy(A["m", "n"], a, name="load_A", actor="threads")
    kb.copy(B["m", "n"], b, name="load_B", actor="threads")
    kb.elementwise("add", [a, b], c, (("convert", 3.0, "cuda"), ("add", 1.0, "cuda")),
                   compute_dtype=compute_dtype, actor="threads")
    kb.copy(c, C["m", "n"], name="store_C", actor="threads")
    return kb.program(name="elementwise_add")


def main() -> None:
    import tilesight as sight
    from tilesight.arch.h200_sxm import H200_SXM

    program = elementwise_add_program(m=8100, n=8200)          # neither is a multiple of the 64x256 tile: tail work groups
    result = sight.analyze(program, H200_SXM().set_to_microbench(), options=sight.Options(cache="fast"))
    report(result, op="main/load_A")
    print("  work groups:", [(g.group_id, g.count) for g in result.launches["main"].groups])


if __name__ == "__main__":
    main()
