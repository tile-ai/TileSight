"""TileOps GELU register-copy path with erf polynomial; see ../LICENSE.TileOps."""


def build_kernel(N_total: int = 67108864, threads: int = 128, num_per_thread: int = 16, dtype: str = "bfloat16"):
    """BF16 GELU over a flat array (8192 x 8192 elements in the supplied case)."""
    import tilelang
    import tilelang.language as T

    if N_total <= 0 or threads <= 0 or num_per_thread <= 0:
        raise ValueError("positive element and thread counts required")
    N = N_total
    INV_SQRT2 = 0.7071067811865476

    # |x| taken as saturated. erf(3.6) = 1 - 7.4e-7, far below half a float16 ulp.
    _CLAMP = 3.6

    # erf(x) = clip(t * P(w), -1, 1), t = clamp(x, -_CLAMP, _CLAMP), w = 1 - (t / _CLAMP)**2,
    # highest degree first. Worst case over the real line is 1.7e-5, an order below half
    # a float16 ulp at 1.0.
    _POLY_COEFFS = (
        8.971932411193848,
        -33.1397590637207,
        53.60123062133789,
        -48.02798080444336,
        26.02545738220215,
        -8.488715171813965,
        1.7499406337738037,
        -0.09359221160411835,
        0.11330155283212662,
        0.1387363225221634,
        0.2777777910232544,
    )


    def erf(x, out_dtype):
        """Element-wise erf(x), evaluated and returned in float32.

        ``erff`` is correct to 2 ulp of float32 and costs roughly twice the
        polynomial's instruction count, which only a float32 result can hold.

        NaN is not preserved: the clamp lowers to ``fminf``/``fmaxf``, which return
        their non-NaN operand, so a NaN argument reads back as -1. Restoring it
        costs a ``T.if_then_else``, which scalarises the element loop.

        Args:
            x: The argument. Any float dtype; promoted to float32 before evaluation.
            out_dtype: The dtype the caller stores the result in. It selects the
                evaluation, not the return dtype.

        Returns:
            erf(x) in float32.
        """
        wide = T.cast(x, "float32")
        if out_dtype == "float32":
            return T.erf(wide)
        one = T.cast(1.0, "float32")
        clamped = T.min(T.max(wide, T.cast(-_CLAMP, "float32")), T.cast(_CLAMP, "float32"))
        # Scaling the clamp out before squaring, not after, is what makes w exactly zero
        # at the clamp: float32(1 / _CLAMP) * _CLAMP is 1.0, where 1 - x**2 / _CLAMP**2
        # leaves 1e-8 once the backend contracts it into an FMA.
        scaled = clamped * T.cast(1.0 / _CLAMP, "float32")
        w = one - scaled * scaled
        acc = T.cast(_POLY_COEFFS[0], "float32")
        for coeff in _POLY_COEFFS[1:]:
            acc = acc * w + T.cast(coeff, "float32")
        # The clip keeps the backend from contracting the product into a caller's add,
        # which would evaluate it to full width and land the tail 7e-9 short of +-1.
        # GELU scales the 1 - erf(x) residual by x, so that error is unbounded in |x|.
        return T.min(T.max(clamped * acc, -one), one)

    def op_func(x):
        inv_sqrt_2 = T.cast(INV_SQRT2, "float32")
        half = T.cast(0.5, "float32")
        one = T.cast(1.0, "float32")
        wide = T.cast(x, "float32")
        return half * wide * (one + erf(wide * inv_sqrt_2, x.dtype))

    @tilelang.jit(out_idx=[1])
    def kernel(threads_arg, npt_arg):
        block_size = threads_arg * npt_arg

        @T.prim_func
        def main(x: T.Tensor((N,), dtype), y: T.Tensor((N,), dtype)):
            with T.Kernel(T.ceildiv(N, block_size), threads=threads_arg) as bx:
                x_reg = T.alloc_fragment((block_size,), dtype)
                y_reg = T.alloc_fragment((block_size,), dtype)
                T.copy(x[bx * block_size : (bx + 1) * block_size], x_reg)
                for i, j in T.Parallel(threads_arg, npt_arg):
                    y_reg[i * npt_arg + j] = op_func(x_reg[i * npt_arg + j])
                T.copy(y_reg, y[bx * block_size : (bx + 1) * block_size])

        return main

    return kernel.get_tir(threads, num_per_thread)
