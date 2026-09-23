"""FA3-style attention forward on Hopper (TileLang ``flash_attention/example_mha_fwd_bhsd.py::flashattn``).

    python -m tilesight.modeling.program.examples.fa3
"""

from __future__ import annotations

from typing import Optional

from ..builder import KernelBuilder, tiles
from ..contract import FromWorkGroup, Program
from ._common import TILELANG_COMMIT, _attention_shapes, _attention_work_units, report

# ---------------------------------------------------------------------------
# FA3-style forward on Hopper
# (reference: tilelang/examples/flash_attention/example_mha_fwd_bhsd.py::flashattn
#  with block_M=block_N=128, num_stages=2, threads=256; GQA head mapping as in
#  example_gqa_fwd_bshd.py where K/V use head ``by // groups``)
# ---------------------------------------------------------------------------


def fa3_program(
    batch: int = 1, heads: int = 32, kv_heads: Optional[int] = None, seq_len: int = 4096, head_dim: int = 128,
    block_m: int = 128, block_n: int = 128, stages: int = 2, causal: bool = False, dtype: str = "bf16",
    pv_via_smem: bool = False, store_p_window_offset: int = 0,
) -> Program:
    """Hopper flash attention: Q load once, K/V pipeline, register S/P/O, O store at the end.

    Structure follows ``example_mha_fwd_bhsd.py::flashattn`` (K-load, QK gemm,
    online softmax, rescale, V-load, PV gemm per pipelined KV iteration;
    ``num_stages`` deep shared buffers).  The producer issues ``load_K`` one
    window ahead so the QK of block ``k+1`` overlaps the PV of block ``k``.
    Causal loop trip counts and mask fractions are per q tile as in the source;
    ``kv_heads`` applies the GQA ``by // groups`` mapping (grid order q, head,
    batch as in ``T.Kernel``).  A causal ``seq_len`` must be a multiple of ``block_m``.

    ``pv_via_smem=True`` is the production kernel's SS-PV layout (flash-attention
    hopper, non-causal d=64 tile): P is staged in SMEM (``store_P``, issued by its
    own ``p_stage`` actor) and the PV gemm reads it from there; the default is the
    TileLang example's register P.  ``store_p_window_offset`` labels which
    iteration's store shares a steady-state window with ``load_K[k+1]`` /
    ``load_V[k]`` on the SMEM port (0: ``store_P[k]``, 1: ``store_P[k+1]``); the
    periodic search explores resource orders for the declared labels only.
    """

    kv_heads = heads if kv_heads is None else kv_heads
    if heads % kv_heads:
        raise ValueError("kv_heads must divide heads")
    group = heads // kv_heads
    q_tiles, kv_tiles, kv_valid = _attention_shapes(seq_len, block_m, block_n, partial_q=not causal)
    kb = KernelBuilder("main", grid={"q": q_tiles, "hg": group, "kv_head": kv_heads, "batch": batch}, threads=256,
                       metadata={"kernel": "fa3_forward", "causal": causal, "kv_heads": kv_heads,
                                 "reference": "tilelang/examples/flash_attention/example_mha_fwd_bhsd.py::flashattn",
                                 "tilelang_commit": TILELANG_COMMIT})
    Q = kb.tensor("Q", (batch, heads, seq_len, head_dim), dtype)
    K = kb.tensor("K", (batch, kv_heads, seq_len, head_dim), dtype)
    V = kb.tensor("V", (batch, kv_heads, seq_len, head_dim), dtype)
    O = kb.tensor("O", (batch, heads, seq_len, head_dim), dtype)
    Q_s = kb.shared("Q_s", (block_m, head_dim), dtype)
    K_s = kb.shared("K_s", (block_n, head_dim), dtype, stages=stages)
    V_s = kb.shared("V_s", (block_n, head_dim), dtype, stages=stages)
    S = kb.fragment("S", (block_m, block_n), "fp32")
    P = kb.fragment("P", (block_m, block_n), dtype)
    O_acc = kb.fragment("O_acc", (block_m, head_dim), "fp32", carried=True)
    stats = kb.fragment("softmax_state", (block_m, 2), "fp32", carried=True)
    O_out = kb.fragment("O_out", (block_m, head_dim), dtype)
    with kb.loop("kv", FromWorkGroup() if causal else kv_tiles, stages=stages) as loop:
        with loop.prologue():
            kb.copy(Q["batch", ("kv_head", "hg"), tiles("q"), :], Q_s, name="load_Q")
        kb.copy(K["batch", "kv_head", tiles("kv"), :], K_s, name="load_K", window_offset=1)
        kb.gemm(Q_s, K_s, S, transpose_b=True, name="gemm_QK", op_id="gemm.qk", window_offset=1)
        kb.elementwise("online_softmax", [S, kb.row_state(stats)], P,
                       (("max", 1.0, "cuda"), ("sub", 1.0, "cuda"), ("exp2", 1.0, "sfu"), ("add", 1.0, "cuda"),
                        ("convert", 1.0, "cuda")), updates=[stats], actor="softmax", window_offset=1)
        kb.mul(O_acc, kb.row_state(stats), O_acc, name="rescale_O", actor="softmax", window_offset=1)   # O_acc *= rescale[row]
        kb.copy(V["batch", "kv_head", tiles("kv"), :], V_s, name="load_V")
        if pv_via_smem:
            P_s = kb.shared("P_shared", (block_m, block_n), dtype, stages=1)
            kb.copy(P, P_s, name="store_P", actor="p_stage", window_offset=store_p_window_offset)
            kb.gemm(P_s, V_s, O_acc, name="gemm_PV", op_id="gemm.pv")
        else:
            kb.gemm(P, V_s, O_acc, name="gemm_PV", op_id="gemm.pv")
        with loop.epilogue():
            kb.mul(O_acc, kb.row_state(stats), O_out, name="normalize_O", actor="softmax")
            kb.copy(O_out, O["batch", ("kv_head", "hg"), tiles("q"), :], name="store_O", actor="softmax")
    units = _attention_work_units(q_tiles, group * kv_heads * batch, block_m, block_n, kv_tiles, causal, kv_valid,
                                  masked_ops=("gemm_QK", "gemm_PV", "online_softmax"), kv_payload_ops=("load_K", "load_V"))
    if not causal and seq_len % block_m:
        # a partial last q tile: the builder derives the q/kv tail groups from the tensor and tile shapes
        return kb.program(name="fa3")
    return kb.program(work_units=units, tails="declared", name="fa3_causal" if causal else "fa3")


def main() -> None:
    import tilesight as sight
    from tilesight.arch.h200_sxm import H200_SXM

    program = fa3_program(batch=1, heads=32, seq_len=2048, head_dim=128, causal=True)
    result = sight.analyze(program, H200_SXM().set_to_microbench(), options=sight.Options(cache="fast", ii_mode="periodic_best"))
    report(result, loop="main/kv", op="main/kv/load_K")


if __name__ == "__main__":
    main()
