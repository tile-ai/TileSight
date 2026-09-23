"""Dense FlashMLA decode, one launch or main + combine (TileLang ``deepseek_mla/example_mla_decode.py``).

    python -m tilesight.modeling.program.examples.flashmla_decode
"""

from __future__ import annotations


from ..builder import KernelBuilder, sequential_program, tiles
from ..contract import Program
from ._common import TILELANG_COMMIT, _ceil, report

# ---------------------------------------------------------------------------
# FlashMLA dense decode
# (reference: tilelang/examples/deepseek_mla/example_mla_decode.py; split variant
#  prices main and combine launches separately)
# ---------------------------------------------------------------------------


def flashmla_decode_program(
    batch: int = 64, heads: int = 128, kv_len: int = 8192, head_group: int = 64,
    d_qk: int = 576, d_v: int = 512, block_n: int = 64, num_splits: int = 1, dtype: str = "bf16",
) -> Program:
    """FlashMLA dense decode: no-split (one launch) or main + combine (two launches)."""

    if num_splits < 1:
        raise ValueError("num_splits must be >= 1")
    head_groups = _ceil(heads, head_group)
    kv_tiles = _ceil(kv_len, block_n)
    kv_per_split = _ceil(kv_tiles, num_splits)
    d_rope = d_qk - d_v
    grid = {"hg": head_groups, "batch": batch}
    if num_splits > 1:
        grid = {"split": num_splits, "hg": head_groups, "batch": batch}
    kb = KernelBuilder("main", grid=grid, threads=256,
                       metadata={"kernel": "flashmla_dense_decode", "num_splits": num_splits,
                                 "reference": "tilelang/examples/deepseek_mla/example_mla_decode.py",
                                 "tilelang_commit": TILELANG_COMMIT})
    split = num_splits > 1
    padded_kv = kv_per_split * block_n  # KV cache viewed per split, padded to whole blocks
    padded_heads = head_groups * head_group
    Q = kb.tensor("Q", (batch, padded_heads, d_v), dtype)
    Qpe = kb.tensor("Q_pe", (batch, padded_heads, d_rope), dtype)
    KV = kb.tensor("KV", (batch, num_splits, padded_kv, d_v) if split else (batch, padded_kv, d_v), dtype)
    Kpe = kb.tensor("K_pe", (batch, num_splits, padded_kv, d_rope) if split else (batch, padded_kv, d_rope), dtype)
    out_dtype = "fp32" if split else dtype
    Out = kb.tensor("O_partial" if split else "O",
                    (batch, num_splits, padded_heads, d_v) if split else (batch, padded_heads, d_v), out_dtype)
    LSE = kb.tensor("LSE", (batch, num_splits, padded_heads, 1) if split else (batch, padded_heads, 1), "fp32")

    def kv_slice(tensor):
        return tensor["batch", "split", tiles("kv"), :] if split else tensor["batch", tiles("kv"), :]

    def out_slice(tensor):
        return tensor["batch", "split", tiles("hg"), :] if split else tensor["batch", tiles("hg"), :]

    Q_s = kb.shared("Q_s", (head_group, d_v), dtype)
    Qpe_s = kb.shared("Qpe_s", (head_group, d_rope), dtype)
    KV_s = kb.shared("KV_s", (block_n, d_v), dtype, stages=2)
    Kpe_s = kb.shared("Kpe_s", (block_n, d_rope), dtype, stages=2)
    S = kb.fragment("S", (head_group, block_n), "fp32")
    S_pe = kb.fragment("S_pe", (head_group, block_n), "fp32")
    P = kb.fragment("P", (head_group, block_n), dtype)
    O_acc = kb.fragment("O_acc", (head_group, d_v), "fp32", carried=True)
    stats = kb.fragment("softmax_state", (head_group, 2), "fp32", carried=True)
    O_out = kb.fragment("O_out", (head_group, d_v), out_dtype)
    lse_out = kb.fragment("lse_out", (head_group, 1), "fp32")
    with kb.loop("kv", kv_per_split, stages=2) as loop:
        with loop.prologue():
            kb.copy(Q["batch", tiles("hg"), :], Q_s, name="load_Q")
            kb.copy(Qpe["batch", tiles("hg"), :], Qpe_s, name="load_Q_pe")
        kb.copy(kv_slice(KV), KV_s, name="load_KV", window_offset=1)
        kb.copy(kv_slice(Kpe), Kpe_s, name="load_K_pe", window_offset=1)
        kb.gemm(Q_s, KV_s, S, transpose_b=True, name="gemm_QK", op_id="gemm.qk", window_offset=1)
        kb.gemm(Qpe_s, Kpe_s, S_pe, transpose_b=True, name="gemm_QK_pe", op_id="gemm.qk_pe", window_offset=1)
        kb.elementwise("online_softmax", [S, S_pe, kb.row_state(stats)], P,
                       (("add", 1.0, "cuda"), ("max", 1.0, "cuda"), ("exp2", 1.0, "sfu"), ("add", 1.0, "cuda")),
                       updates=[stats], actor="softmax", window_offset=1)
        kb.mul(O_acc, kb.row_state(stats), O_acc, name="rescale_O", actor="softmax", window_offset=1)
        kb.gemm(P, KV_s, O_acc, name="gemm_PV", op_id="gemm.pv")
        with loop.epilogue():
            kb.mul(O_acc, kb.row_state(stats), O_out, name="finalize", actor="softmax")
            kb.unary("log2", kb.row_state(stats), lse_out, name="finalize_lse", actor="softmax")
            kb.copy(O_out, out_slice(Out), name="store_O", actor="softmax")
            kb.copy(lse_out, out_slice(LSE), name="store_LSE", actor="softmax")
    main = kb.launch()
    if num_splits == 1:
        return Program((main,), name="flashmla_decode")
    cb = KernelBuilder("combine", grid={"hg": head_groups, "batch": batch}, threads=128,
                       metadata={"kernel": "flashmla_combine", "tilelang_commit": TILELANG_COMMIT})
    partial = cb.tensor("O_partial", (batch, num_splits, padded_heads, d_v), "fp32")
    lse_in = cb.tensor("LSE", (batch, num_splits, padded_heads, 1), "fp32")
    final = cb.tensor("O", (batch, padded_heads, d_v), dtype)
    part_r = cb.fragment("partial_r", (head_group, d_v), "fp32")
    lse_r = cb.fragment("lse_r", (head_group, 1), "fp32")
    o_acc = cb.fragment("o_acc", (head_group, d_v), "fp32", carried=True)
    o_out = cb.fragment("o_out", (head_group, d_v), dtype)
    with cb.loop("split", num_splits) as loop:
        cb.copy(partial["batch", "split", tiles("hg"), :], part_r, name="load_partial", actor="combine")
        cb.copy(lse_in["batch", "split", tiles("hg"), :], lse_r, name="load_lse", actor="combine")
        cb.elementwise("combine", [part_r, lse_r], o_acc, (("exp2", 1.0 / d_v, "sfu"), ("fma", 1.0, "cuda")), actor="combine")
        with loop.epilogue():
            cb.elementwise("normalize", [o_acc], o_out, (("mul", 1.0, "cuda"),), actor="combine")
            cb.copy(o_out, final["batch", tiles("hg"), :], name="store_O", actor="combine")
    return sequential_program((main, cb.launch()), name="flashmla_decode_split")


__all__ = [
    "TILELANG_COMMIT", "causal_kv_trips", "causal_valid_fraction", "elementwise_add_program", "fa3_program",
    "fa4_program", "flashmla_decode_program", "gemm_program", "reduce_sum_program", "rms_norm_program",
]


def main() -> None:
    import tilesight as sight
    from tilesight.arch.h200_sxm import H200_SXM

    arch = H200_SXM().set_to_microbench()
    for splits in (1, 4):
        print("num_splits =", splits)
        program = flashmla_decode_program(batch=4, heads=128, kv_len=1024, num_splits=splits)
        report(sight.analyze(program, arch, options=sight.Options(cache="fast", ii_mode="periodic_best")), op="main/kv/load_KV")


if __name__ == "__main__":
    main()
