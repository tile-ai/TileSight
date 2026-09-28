"""Self-contained TileOps specimen; see ../README.md and ../LICENSE.TileOps."""

def build_kernel(m: int, n: int, k: int, trans_a: bool = False, trans_b: bool = True, dtype: str = "float16"):
    """One producer WG, one consumer WG; returns an uncompiled JIT builder."""
    import tilelang
    import tilelang.language as T

    accum_dtype = "float"
    a_shape = (k, m) if trans_a else (m, k)
    b_shape = (n, k) if trans_b else (k, n)

    @tilelang.jit(
        out_idx=[2],
        pass_configs={"tl.disable_warp_specialized": True},
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _gemm_func(block_m: int = 128,
                   block_n: int = 128,
                   block_k: int = 64,
                   num_stages: int = 3):
        # Manual 2-warpgroup WS: 1 producer WG (128 threads) issues TMA, 1
        # consumer WG (128 threads) runs WGMMA. Barrier arrive_counts (128) are
        # bound to this layout, so threads is fixed at 256.
        threads = 256
        k_iters = T.ceildiv(k, block_k)
        # SMEM tile shapes follow the storage layout; the WGMMA transpose flags
        # reconcile them with the logical (M,K) x (K,N) contraction.
        a_tile = (block_k, block_m) if trans_a else (block_m, block_k)
        b_tile = (block_n, block_k) if trans_b else (block_k, block_n)

        @T.prim_func
        def _gemm_main(
                a: T.Tensor(a_shape, dtype),  # type: ignore
                b: T.Tensor(b_shape, dtype),  # type: ignore
                c: T.Tensor((m, n), dtype),  # type: ignore
        ) -> None:
            with T.Kernel(
                    T.ceildiv(n, block_n), T.ceildiv(m, block_m), threads=threads) as (bx, by):
                # Multi-stage ring of A/B SMEM buffers. Indexed by stage = gi %
                # num_stages; the phase bit flips every num_stages iterations.
                a_smem = T.alloc_shared((num_stages,) + a_tile, dtype)
                b_smem = T.alloc_shared((num_stages,) + b_tile, dtype)
                c_local = T.alloc_fragment((block_m, block_n), accum_dtype)

                T.annotate_layout({
                    a_smem: tilelang.layout.make_swizzled_layout(a_smem),
                    b_smem: tilelang.layout.make_swizzled_layout(b_smem),
                })

                # Producer→consumer (buffer full) and consumer→producer (buffer
                # empty) barriers, one per ring slot. Each is arrived by exactly
                # one warpgroup (128 threads). Allocated as length-num_stages
                # barrier arrays and indexed by the static slot id.
                ab_full = T.alloc_barrier([128] * num_stages)
                ab_empty = T.alloc_barrier([128] * num_stages)

                # Monotonic per-warpgroup iteration counters; stage = gi %
                # num_stages, phase = (gi // num_stages) % 2.
                gi_prod = T.alloc_var("int32", init=0)
                gi_cons = T.alloc_var("int32", init=0)

                m_start = by * block_m
                n_start = bx * block_n

                tx = T.get_thread_binding()

                if tx < 128:
                    # ── Producer warpgroup: issue TMA loads of A and B tiles. ──
                    T.dec_max_nreg(24)
                    for ki in T.serial(k_iters):
                        stage = gi_prod % num_stages
                        phase = (gi_prod // num_stages) % 2
                        k_start = ki * block_k
                        # Unroll the ring-slot dispatch when building TIR: each
                        # slot gets a static SMEM/barrier index under a
                        # dynamic `stage == s` guard.
                        for s in range(num_stages):
                            if stage == s:
                                # Wait for this slot to be drained before
                                # reuse. The consumer leaves the slot in
                                # empty-phase (phase ^ 1) for the round the
                                # producer is about to refill; rounds
                                # 0..num_stages-1 see the init-0 state (phase
                                # ^ 1 == 1) which is already satisfied by the
                                # barrier's initial parity.
                                T.barrier_wait(ab_empty[s], phase ^ 1)
                                if trans_a:
                                    T.tma_copy(
                                        a[k_start:k_start + block_k,
                                          m_start:m_start + block_m],
                                        a_smem[s, :, :], barrier=ab_full[s])
                                else:
                                    T.tma_copy(
                                        a[m_start:m_start + block_m,
                                          k_start:k_start + block_k],
                                        a_smem[s, :, :], barrier=ab_full[s])
                                if trans_b:
                                    T.tma_copy(
                                        b[n_start:n_start + block_n,
                                          k_start:k_start + block_k],
                                        b_smem[s, :, :], barrier=ab_full[s])
                                else:
                                    T.tma_copy(
                                        b[k_start:k_start + block_k,
                                          n_start:n_start + block_n],
                                        b_smem[s, :, :], barrier=ab_full[s])
                                T.barrier_arrive(ab_full[s])
                        gi_prod = gi_prod + 1
                else:
                    # ── Consumer warpgroup: run WGMMA, accumulate over K. ──
                    T.inc_max_nreg(240)
                    T.clear(c_local)
                    for ki in T.serial(k_iters):
                        stage = gi_cons % num_stages
                        phase = (gi_cons // num_stages) % 2
                        for s in range(num_stages):
                            if stage == s:
                                T.barrier_wait(ab_full[s], phase)
                                T.wgmma_gemm(
                                    a_smem[s, :, :],
                                    b_smem[s, :, :],
                                    c_local,
                                    transpose_A=trans_a,
                                    transpose_B=trans_b,
                                    policy=T.GemmWarpPolicy.FullRow,
                                    clear_accum=(ki == 0),
                                )
                                T.wait_wgmma(0)
                                T.warpgroup_fence_operand(c_local, num_regs=64)
                                T.barrier_arrive(ab_empty[s])
                        gi_cons = gi_cons + 1

                    # Epilogue: guard the M/N tail so partial tiles don't
                    # write out of bounds (m / n need not be multiples of the
                    # block sizes; K tails are zero-filled by TMA).
                    for i, j in T.Parallel(block_m, block_n):
                        if m_start + i < m and n_start + j < n:
                            c[m_start + i, n_start + j] = c_local[i, j]


        return _gemm_main

    return _gemm_func
