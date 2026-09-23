"""FA4 forward on Blackwell SM100, warp-specialised: TileLang ``flashattn_wasp`` (``q_stages=1``) and the
production two-Q-stage layout (``q_stages=2``: shared K/V ring, P aliased over S).

    python -m tilesight.modeling.program.examples.fa4_wasp
"""

from __future__ import annotations

from typing import Any, Optional

from ..builder import KernelBuilder, tiles
from ..contract import FromWorkGroup, Program, dispatch_coordinates
from ._common import TILELANG_COMMIT, _attention_shapes, _attention_work_units, report, sectioned_lpt_order

def fa4_wasp_program(
    batch: int = 1, heads: int = 32, kv_heads: Optional[int] = None, seq_len: int = 4096, head_dim: int = 128,
    block_m: int = 128, block_n: int = 128, causal: bool = False, dtype: str = "bf16",
    kv_stages: int = 2, q_stages: int = 1, mapper: Any = "linear_block_id",
) -> Program:
    """SM100 ``flashattn_wasp``: warp-specialised pipeline with S, P **and O** in TMEM.

    ``q_stages=1`` follows ``flash_attention_sm100/mha_fwd_bshd.py::flashattn_wasp``
    statement by statement.  DMA warp: ``load_K`` then ``load_V`` into
    ``K_shared[kv_stages]`` / ``V_shared[kv_stages]`` (two separate allocations,
    as in the source).  BMM warp: ``tcgen05_gemm(Q, K) -> S_tmem`` then, after the
    softmax barrier, ``tcgen05_gemm(P_tmem, V) -> O_tmem`` (accumulating).  The
    softmax warp is **one** thread group that first waits for the QK barrier
    (``mbar_bmm1_full``) and the previous PV, then runs, in this order,
    ``copy(O_tmem, O_reg)``, ``copy(S_tmem, S_reg)``, mask + online softmax,
    ``O_reg *= rescale``, ``copy(P_cast, P_tmem)``, ``copy(O_reg, O_tmem)``; the PV
    gemm waits for its ``mbar_softmax_full``.  Carried state: softmax statistics
    and ``O_tmem``.

    ``q_stages=2`` is the production ``FlashAttentionForwardSm100`` layout that the
    paper's periodic model describes (flash-attention ``flash_fwd_sm100.py``, *not*
    a TileLang example): one work unit owns two ``block_m``-row Q stages, each with
    its own softmax state, softmax warp and TMEM; inside a stage **P is written
    over S** (one aliased S/P slot, released by the PV gemm before the next QK may
    overwrite it); K and V share **one** SMEM ring of ``kv_stages`` slots issued
    K, V, K, V ...; one correction warp serves stage 0 then stage 1; the tensor
    stream issues ``PV0[k], QK0[k+1], PV1[k], QK1[k+1]``.  The two stages read the
    even/odd ``block_m`` tiles of a ``2 x block_m`` row q block, written as the
    tensor views ``Q_stage{s}`` / ``Output_stage{s}`` (``seq_len`` must be a
    multiple of ``q_stages x block_m``); causal mask fractions are those of each
    stage's own rows.  Not expressed: the split-P (75/25) barrier, on-chip
    transfers fused into (and overlapping) a compute op, PackGQA.
    """

    kv_heads = heads if kv_heads is None else kv_heads
    if heads % kv_heads:
        raise ValueError("kv_heads must divide heads")
    if q_stages not in (1, 2):
        raise ValueError("q_stages must be 1 or 2")
    group = heads // kv_heads
    unit_rows = q_stages * block_m
    q_tiles, kv_tiles, kv_valid = _attention_shapes(seq_len, unit_rows, block_n)
    extents = (batch, kv_heads, group, q_tiles)
    if callable(mapper):
        mapper = mapper(extents)
    production = q_stages == 2
    kb = KernelBuilder("main", grid={"q": q_tiles, "hg": group, "kv_head": kv_heads, "batch": batch}, threads=256,
                       requires_arch=("tmem", "tcgen05"), dispatch_order=mapper,
                       metadata={"kernel": "fa4_forward_sm100_wasp", "causal": causal, "kv_heads": kv_heads,
                                 "target": "B200 (SM100)", "variant": "wasp", "q_stages": q_stages, "kv_stages": kv_stages,
                                 "reference": "tilelang/examples/flash_attention_sm100/mha_fwd_bshd.py::flashattn_wasp",
                                 "tilelang_commit": TILELANG_COMMIT})
    K = kb.tensor("K", (batch, seq_len, kv_heads, head_dim), dtype)
    V = kb.tensor("V", (batch, seq_len, kv_heads, head_dim), dtype)
    if production:
        K_s = kb.shared("K_shared", (block_n, head_dim), dtype)
        V_s = kb.shared("V_shared", (block_n, head_dim), dtype)
        kb.ring("KV_ring", [K_s, V_s], slots=kv_stages)
    else:
        K_s = kb.shared("K_shared", (block_n, head_dim), dtype, stages=kv_stages)
        V_s = kb.shared("V_shared", (block_n, head_dim), dtype, stages=kv_stages)
    suffixes = ["_s%d" % index for index in range(q_stages)] if production else [""]
    stage = {}
    for suffix in suffixes:
        view = "_stage" + suffix[2:] if production else ""
        b = dict(
            Q=kb.tensor("Q" + view, (batch, seq_len // q_stages, heads, head_dim), dtype),
            O=kb.tensor("Output" + view, (batch, seq_len // q_stages, heads, head_dim), dtype),
            Q_s=kb.shared("Q_shared" + suffix, (block_m, head_dim), dtype),
            O_s=kb.shared("O_shared" + suffix, (block_m, head_dim), dtype),
            S_t=kb.tmem("S_tmem" + suffix, (block_m, block_n), "fp32"),
            P_t=kb.tmem("P_tmem" + suffix, (block_m, block_n), dtype),
            O_t=kb.tmem("O_tmem" + suffix, (block_m, head_dim), "fp32", carried=True),
            S_r=kb.fragment("S_reg" + suffix, (block_m, block_n), "fp32"),
            P_c=kb.fragment("P_cast" + suffix, (block_m, block_n), dtype),
            O_r=kb.fragment("O_reg" + suffix, (block_m, head_dim), "fp32"),
            O_c=kb.fragment("O_corrected" + suffix, (block_m, head_dim), "fp32"),
            stats=kb.fragment("softmax_state" + suffix, (block_m, 4), "fp32", carried=True),
            O_f=kb.fragment("O_final" + suffix, (block_m, head_dim), "fp32"),
            O_out=kb.fragment("O_out" + suffix, (block_m, head_dim), dtype),
        )
        if production:
            kb.alias("SP_tmem" + suffix, [b["S_t"], b["P_t"]])
        stage[suffix] = b
    softmax_ops = (("max", 1.0, "cuda"), ("sub", 1.0, "cuda"), ("exp2", 1.0, "sfu"), ("add", 1.0, "cuda"), ("convert", 1.0, "cuda"))
    with kb.loop("kv", FromWorkGroup() if causal else kv_tiles, stages=kv_stages) as loop:
        with loop.prologue():
            for suffix in suffixes:
                kb.copy(stage[suffix]["Q"]["batch", tiles("q"), ("kv_head", "hg"), :], stage[suffix]["Q_s"], name="load_Q" + suffix)
        kb.copy(K["batch", tiles("kv"), "kv_head", :], K_s, name="load_K")
        kb.copy(V["batch", tiles("kv"), "kv_head", :], V_s, name="load_V")
        if not production:
            b = stage[""]
            kb.gemm(b["Q_s"], K_s, b["S_t"], transpose_b=True, name="gemm_QK", op_id="gemm.qk", actor="bmm")
            # one softmax thread group, in source order; its first statement waits for the QK barrier
            kb.copy(b["O_t"], b["O_r"], name="tmem_ld_O", actor="softmax")
            kb.copy(b["S_t"], b["S_r"], name="tmem_ld_S", actor="softmax")
            kb.elementwise("online_softmax", [b["S_r"], kb.row_state(b["stats"])], b["P_c"], softmax_ops, updates=[b["stats"]],
                           actor="softmax")
            kb.mul(b["O_r"], kb.row_state(b["stats"]), b["O_c"], name="rescale_O", actor="softmax")
            kb.copy(b["P_c"], b["P_t"], name="tmem_st_P", actor="softmax")
            kb.copy(b["O_c"], b["O_t"], name="tmem_st_O", actor="softmax")
            kb.gemm(b["P_t"], V_s, b["O_t"], name="gemm_PV", op_id="gemm.pv", actor="bmm")
            loop.schedule.after("gemm_QK", "tmem_ld_O")          # mbarrier_wait_parity(mbar_bmm1_full) precedes copy(O_tmem, O_reg)
        else:
            for suffix in suffixes:
                b = stage[suffix]
                kb.gemm(b["Q_s"], K_s, b["S_t"], transpose_b=True, name="gemm_QK" + suffix, op_id="gemm.qk", actor="bmm")
                kb.copy(b["S_t"], b["S_r"], name="tmem_ld_S" + suffix, actor="softmax" + suffix)
                kb.elementwise("online_softmax" + suffix, [b["S_r"], kb.row_state(b["stats"])], b["P_c"], softmax_ops,
                               updates=[b["stats"]], actor="softmax" + suffix)
                kb.copy(b["P_c"], b["P_t"], name="tmem_st_P" + suffix, actor="softmax" + suffix)
            for suffix in suffixes:   # one correction warp: stage 0 then stage 1
                b = stage[suffix]
                kb.copy(b["O_t"], b["O_r"], name="tmem_ld_O" + suffix, actor="correction")
                kb.mul(b["O_r"], kb.row_state(b["stats"]), b["O_c"], name="rescale_O" + suffix, actor="correction")
                kb.copy(b["O_c"], b["O_t"], name="tmem_st_O" + suffix, actor="correction")
            for suffix in suffixes:
                b = stage[suffix]
                kb.gemm(b["P_t"], V_s, b["O_t"], name="gemm_PV" + suffix, op_id="gemm.pv", actor="bmm")
            loop.schedule.actor_order("bmm", [("gemm_PV_s0", 0), ("gemm_QK_s0", 1), ("gemm_PV_s1", 0), ("gemm_QK_s1", 1)])
        with loop.epilogue():
            for suffix in suffixes:
                b = stage[suffix]
                actor = "correction" if production else "softmax"
                kb.copy(b["O_t"], b["O_f"], name="tmem_ld_O_final" + suffix, actor=actor)
                kb.mul(b["O_f"], kb.row_state(b["stats"]), b["O_out"], name="normalize_O" + suffix, actor=actor)
                kb.copy(b["O_out"], b["O_s"], name="stage_O" + suffix, actor=actor)
                kb.copy(b["O_s"], b["O"]["batch", tiles("q"), ("kv_head", "hg"), :], name="store_O" + suffix, actor=actor)
    masked = tuple(name + suffix for suffix in suffixes for name in ("gemm_QK", "gemm_PV", "online_softmax"))
    op_rows = {name + suffix: (index * block_m, block_m) for index, suffix in enumerate(suffixes)
               for name in ("gemm_QK", "gemm_PV", "online_softmax")}
    q_order = [coordinate[-1] for coordinate in dispatch_coordinates(extents, mapper)]
    units = _attention_work_units(q_tiles, group * kv_heads * batch, unit_rows, block_n, kv_tiles, causal, kv_valid,
                                  masked_ops=masked, kv_payload_ops=("load_K", "load_V"), q_order=q_order, op_rows=op_rows)
    return kb.program(work_units=units, tails="declared", name="fa4_wasp_causal" if causal else "fa4_wasp")


def main() -> None:
    import tilesight as sight
    from tilesight.arch.b200 import B200

    arch = B200().set_to_microbench()
    options = sight.Options(cache="fast", ii_mode="periodic_best", periodic_topology_trials=500)
    shape = dict(batch=1, heads=32, seq_len=2048, head_dim=128, causal=True)
    for label, program in (
        ("TileLang flashattn_wasp (one Q stage)", fa4_wasp_program(q_stages=1, kv_stages=2, **shape)),
        ("production layout (two Q stages, 3-slot K/V ring)", fa4_wasp_program(
            q_stages=2, kv_stages=3, mapper=lambda extents: sectioned_lpt_order(extents, 128), **shape)),
    ):
        print(label)
        pipeline = program.launches[0].loop("kv").body
        print("  on-chip storage per work unit (KiB):", {k: v / 1024 for k, v in pipeline.storage_bytes().items()},
              " groups:", [(g.kind, g.name, g.slots) for g in pipeline.storage_groups])
        report(sight.analyze(program, arch, options=options), loop="main/kv")


if __name__ == "__main__":
    main()
