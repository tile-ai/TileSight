"""FA4 forward on Blackwell SM100, TileLang ``flash_attention_sm100/mha_fwd_bshd.py::flashattn(variant="ts")``.

    python -m tilesight.modeling.program.examples.fa4_ts
"""

from __future__ import annotations

from typing import Any, Optional

from ..builder import KernelBuilder, tiles
from ..contract import FromWorkGroup, Program, dispatch_coordinates
from ._common import TILELANG_COMMIT, _attention_shapes, _attention_work_units, report

# ---------------------------------------------------------------------------
# FA4-style forward on Blackwell SM100 / B200
# (reference: tilelang/examples/flash_attention_sm100/mha_fwd_bshd.py::flashattn(variant="ts")
#  and gqa_fwd_bshd.py::flashattn(variant="ts") for the ``by // groups`` mapping)
# ---------------------------------------------------------------------------


def fa4_program(
    batch: int = 1, heads: int = 32, kv_heads: Optional[int] = None, seq_len: int = 4096, head_dim: int = 128,
    block_m: int = 128, block_n: int = 128, causal: bool = False, dtype: str = "bf16", stages: int = 1,
    mapper: Any = "linear_block_id",
) -> Program:
    """SM100 ``flashattn(variant="ts")``: S and D in TMEM, P via TMEM, O accumulated in registers.

    Per KV iteration, exactly as in the source: ``load_K`` (TMA) → ``gemm_QK``
    (tcgen05.mma, S_tmem) → ``tmem_ld_S`` (tcgen05.ld) → mask + online softmax
    in registers → ``rescale_O`` (O_reg *= scale) → ``cast_P`` → ``tmem_st_P``
    (tcgen05.st) → ``load_V`` → ``gemm_PV`` (tcgen05.mma, P_tmem x V_shared →
    D_tmem, ``clear_accum``) → ``tmem_ld_D`` → ``accumulate_O`` (O_reg +=
    D_reg).  Epilogue: normalise, copy through SMEM, store.  ``T.Pipelined(...,
    num_stages=1)`` → single-stage buffers by default (``stages`` sets the K/V
    ring depth of a pipelined variant, e.g. the production SM100 kernel's KV
    ring); ``mbarrier`` waits are represented
    by the S/D TMEM data dependencies.  The MMA accumulator writes into TMEM
    are tensor-core internal and are not tcgen05.ld/st traffic.

    Scope: dense or causal (per-q-tile trip counts and exact mask fractions as
    in the source), GQA via ``kv_heads``.  ``seq_len`` must be a multiple of
    ``block_m``; a KV tail is applied as an averaged mask fraction.  ``mapper``
    is the block mapper (or ``extents -> mapper``, e.g. ``lambda e:
    sectioned_lpt_order(e, 128)`` for the production kernel's longest-first
    grid order); causal work groups follow its launch order.  Not
    expressed: the ``ss`` variant (P via SMEM, 128 threads) and the ``wasp``
    warp-specialised variant with O in TMEM.  The launch requires ``tmem`` and
    ``tcgen05``; other targets are rejected.  Accuracy against B200
    measurements is whatever ``validation.py`` has recorded.
    """

    kv_heads = heads if kv_heads is None else kv_heads
    if heads % kv_heads:
        raise ValueError("kv_heads must divide heads")
    group = heads // kv_heads
    q_tiles, kv_tiles, kv_valid = _attention_shapes(seq_len, block_m, block_n)
    extents = (batch, kv_heads, group, q_tiles)
    if callable(mapper):
        mapper = mapper(extents)
    kb = KernelBuilder("main", grid={"q": q_tiles, "hg": group, "kv_head": kv_heads, "batch": batch}, threads=256,
                       requires_arch=("tmem", "tcgen05"), dispatch_order=mapper,
                       metadata={"kernel": "fa4_forward_sm100_ts", "causal": causal, "kv_heads": kv_heads,
                                 "target": "B200 (SM100)", "variant": "ts",
                                 "reference": "tilelang/examples/flash_attention_sm100/mha_fwd_bshd.py::flashattn(variant='ts')",
                                 "tilelang_commit": TILELANG_COMMIT})
    Q = kb.tensor("Q", (batch, seq_len, heads, head_dim), dtype)
    K = kb.tensor("K", (batch, seq_len, kv_heads, head_dim), dtype)
    V = kb.tensor("V", (batch, seq_len, kv_heads, head_dim), dtype)
    O = kb.tensor("Output", (batch, seq_len, heads, head_dim), dtype)
    Q_s = kb.shared("Q_shared", (block_m, head_dim), dtype)
    K_s = kb.shared("K_shared", (block_n, head_dim), dtype, stages=stages)
    V_s = kb.shared("V_shared", (block_n, head_dim), dtype, stages=stages)
    O_s = kb.shared("O_shared", (block_m, head_dim), dtype)
    S_t = kb.tmem("S_tmem", (block_m, block_n), "fp32")
    D_t = kb.tmem("D_tmem", (block_m, head_dim), "fp32")
    P_t = kb.tmem("P_tmem", (block_m, block_n), dtype)
    S_r = kb.fragment("S_reg", (block_m, block_n), "fp32")
    P_c = kb.fragment("P_cast", (block_m, block_n), dtype)
    O_r = kb.fragment("O_reg", (block_m, head_dim), "fp32", carried=True)
    D_r = kb.fragment("D_reg", (block_m, head_dim), "fp32")
    stats = kb.fragment("softmax_state", (block_m, 4), "fp32", carried=True)
    O_out = kb.fragment("O_out", (block_m, head_dim), dtype)
    with kb.loop("kv", FromWorkGroup() if causal else kv_tiles, stages=stages) as loop:
        with loop.prologue():
            kb.copy(Q["batch", tiles("q"), ("kv_head", "hg"), :], Q_s, name="load_Q")
        kb.copy(K["batch", tiles("kv"), "kv_head", :], K_s, name="load_K")
        kb.gemm(Q_s, K_s, S_t, transpose_b=True, name="gemm_QK", op_id="gemm.qk")
        kb.copy(S_t, S_r, name="tmem_ld_S", actor="softmax")
        kb.elementwise("online_softmax", [S_r, kb.row_state(stats)], P_c,
                       (("max", 1.0, "cuda"), ("sub", 1.0, "cuda"), ("exp2", 1.0, "sfu"), ("add", 1.0, "cuda"),
                        ("convert", 1.0, "cuda")), updates=[stats], actor="softmax")
        kb.mul(O_r, kb.row_state(stats), O_r, name="rescale_O", actor="softmax")
        kb.copy(P_c, P_t, name="tmem_st_P", actor="softmax")
        kb.copy(V["batch", tiles("kv"), "kv_head", :], V_s, name="load_V")
        kb.gemm(P_t, V_s, D_t, name="gemm_PV", op_id="gemm.pv")
        kb.copy(D_t, D_r, name="tmem_ld_D", actor="softmax")
        kb.add(O_r, D_r, O_r, name="accumulate_O", actor="softmax")
        with loop.epilogue():
            kb.mul(O_r, kb.row_state(stats), O_out, name="normalize_O", actor="softmax")
            kb.copy(O_out, O_s, name="stage_O", actor="softmax")
            kb.copy(O_s, O["batch", tiles("q"), ("kv_head", "hg"), :], name="store_O", actor="softmax")
    q_order = [coordinate[-1] for coordinate in dispatch_coordinates(extents, mapper)]
    units = _attention_work_units(q_tiles, group * kv_heads * batch, block_m, block_n, kv_tiles, causal, kv_valid,
                                  masked_ops=("gemm_QK", "gemm_PV", "online_softmax"), kv_payload_ops=("load_K", "load_V"),
                                  q_order=q_order)
    return kb.program(work_units=units, tails="declared", name="fa4_causal" if causal else "fa4")


def main() -> None:
    import tilesight as sight
    from tilesight.arch.b200 import B200

    program = fa4_program(batch=1, heads=32, seq_len=2048, head_dim=128, causal=True)
    result = sight.analyze(program, B200().set_to_microbench(), options=sight.Options(cache="fast", ii_mode="periodic_best"))
    report(result, loop="main/kv", op="main/kv/tmem_ld_S")


if __name__ == "__main__":
    main()
