"""GEMM written the ordinary way: the compiler chooses warp specialization, TMA and buffering."""


def build_kernel(m: int = 8192, n: int = 8192, k: int = 8192, block_m: int = 128, block_n: int = 128,
                 block_k: int = 64, num_stages: int = 3, threads: int = 256):
    import tilelang.language as T

    @T.prim_func
    def gemm(A: T.Tensor((m, k), "float16"), B: T.Tensor((k, n), "float16"), C: T.Tensor((m, n), "float16")):
        with T.Kernel(T.ceildiv(n, block_n), T.ceildiv(m, block_m), threads=threads) as (bx, by):
            As = T.alloc_shared((block_m, block_k), "float16")
            Bs = T.alloc_shared((block_k, block_n), "float16")
            Cl = T.alloc_fragment((block_m, block_n), "float")
            T.clear(Cl)
            for ko in T.Pipelined(T.ceildiv(k, block_k), num_stages=num_stages):
                T.copy(A[by * block_m, ko * block_k], As)
                T.copy(B[ko * block_k, bx * block_n], Bs)
                T.gemm(As, Bs, Cl)
            T.copy(Cl, C[by * block_m, bx * block_n])

    return gemm
