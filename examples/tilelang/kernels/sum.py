"""TileOps row-wise Sum, shared-memory path; see ../LICENSE.TileOps."""


def build_kernel(M: int = 8192, N: int = 8192, block_m: int = 2, threads: int = 128, dtype: str = "float16"):
    """Sum each row, with float32 accumulation and output in the input dtype."""
    import tilelang
    import tilelang.language as T

    if M <= 0 or N <= 0 or block_m <= 0:
        raise ValueError("M, N and block_m must be positive")
    N_padded = ((N + 255) // 256) * 256
    _needs_pad = N_padded != N
    _pad_val = 0.0
    out_dtype = dtype

    @tilelang.jit(out_idx=[1])
    def _func(block_m, threads):
        @T.prim_func
        def main(
            x: T.Tensor[(M, N), dtype],
            out: T.Tensor[(M,), out_dtype],
        ):
            with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                shared_buf = T.alloc_shared((block_m, N_padded), dtype)
                x_f32 = T.alloc_fragment((block_m, N_padded), "float32")
                acc = T.alloc_fragment((block_m,), "float32")
                out_local = T.alloc_fragment((block_m,), out_dtype)

                if _needs_pad:
                    # Kernel-side boundary handling: element-wise load
                    # with T.if_then_else masking for padding columns
                    # and row-tail safety (M % block_m != 0).
                    for i in T.serial(block_m):
                        for j in T.Parallel(N_padded):
                            x_f32[i, j] = T.if_then_else(
                                T.And(pid_m * block_m + i < M, j < N),
                                T.cast(x[pid_m * block_m + i, j], "float32"),
                                T.cast(_pad_val, "float32"),
                            )
                else:
                    # Load via shared memory (fast vectorized path)
                    T.copy(x[pid_m * block_m, 0], shared_buf)

                    # Cast to fp32
                    for i in T.serial(block_m):
                        for j in T.Parallel(N_padded):
                            x_f32[i, j] = T.cast(shared_buf[i, j], "float32")

                T.reduce_sum(x_f32, acc, dim=1)

                # Cast back to output dtype
                for i in T.Parallel(block_m):
                    out_local[i] = T.cast(acc[i], out_dtype)

                T.copy(out_local, out[pid_m * block_m])

        return main

    return _func.get_tir(block_m, threads)
